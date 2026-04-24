"""Combined strategy on validation: all pieces from the session folded
together into a single pipeline.

    1. Walk-forward *scarcity classifier* over the full history, producing
       one out-of-sample probability per local calendar date.
    2. Walk-forward *LightGBM q50 point forecaster* with:
           - prices-only lag + calendar features
           - load-based features (2011+)
           - scarcity_prob_today as an additional feature
    3. Forecast-gated dispatch (skip days the forecaster's own simulation
       says will lose money).
    4. Compare to floor, ceiling, and the previous best (q50 + gate).

This combines the best ideas from the session:
    - Load features improve interval selection  (FINDINGS 2026-04-24)
    - Forecast gate cuts tail losses and lifts Sharpe
    - Scarcity signal as a FEATURE, not a dispatch switch
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.baselines import daily_oracle_schedule, daily_spread_gated_schedule  # noqa: E402
from src.battery import BatterySpec  # noqa: E402
from src.data.ercot import INTERVAL_MINUTES, get_rtm_spp_series  # noqa: E402
from src.data.ercot_load import get_load_series  # noqa: E402
from src.dispatch import run_dispatch  # noqa: E402
from src.evaluation import walk_forward_predict  # noqa: E402
from src.features import build_features  # noqa: E402
from src.features_daily import build_daily_features  # noqa: E402
from src.metrics import compare, pct_of_ceiling  # noqa: E402
from src.ml.lgbm import make_quantile_fit_fn  # noqa: E402
from src.ml.scarcity_classifier import scarcity_fit_fn  # noqa: E402
from src.optimization import perfect_foresight_schedule  # noqa: E402


DEFAULT_SPEC = BatterySpec(
    power_mw=100.0, capacity_mwh=200.0, roundtrip_eff=0.85,
    soc_min_frac=0.05, soc_max_frac=0.95, initial_soc_frac=0.5,
    degradation_cost_per_mwh=2.0,
)


def walk_forward_scarcity_probs(
    daily_features: pd.DataFrame,
    start_date,
    end_date,
    retrain_every_days: int,
    min_train_rows: int = 300,
) -> pd.Series:
    """Produce one out-of-sample scarcity probability per local date in
    [start_date, end_date]. Uses only data strictly before each retrain
    boundary — legitimate to pass into a downstream forecaster's features.
    """
    feat_cols = [c for c in daily_features.columns if c not in {
        "target_scarcity", "target_max_price"
    }]
    df = daily_features.sort_index().dropna(subset=feat_cols + ["target_scarcity"])
    probs = pd.Series(np.nan, index=df.index, name="scarcity_prob")

    eval_dates = df.index[(df.index >= start_date) & (df.index <= end_date)]
    if len(eval_dates) == 0:
        return probs

    boundaries = [eval_dates[0]]
    while boundaries[-1] + pd.Timedelta(days=retrain_every_days) <= eval_dates[-1]:
        boundaries.append(boundaries[-1] + pd.Timedelta(days=retrain_every_days))

    for i, b in enumerate(boundaries):
        next_b = (boundaries[i + 1] if i + 1 < len(boundaries)
                  else eval_dates[-1] + pd.Timedelta(days=1))
        train_mask = df.index < b
        if train_mask.sum() < min_train_rows:
            continue
        clf = scarcity_fit_fn(df.loc[train_mask, feat_cols],
                              df.loc[train_mask, "target_scarcity"])
        pred_mask = (df.index >= b) & (df.index < next_b)
        probs.loc[df.index[pred_mask]] = clf.predict(df.loc[pred_mask, feat_cols])
    return probs


def schedule_from_fcst(fcst: pd.Series, spec, dt, tz):
    filled = fcst.fillna(fcst.mean(skipna=True))
    s = daily_oracle_schedule(filled, spec, dt, cycles_per_day=1.0, tz=tz)
    local_idx = fcst.index.tz_convert(tz) if tz else fcst.index
    day_key = pd.Series(local_idx.date, index=fcst.index)
    for _, idx in fcst.groupby(day_key).groups.items():
        if fcst.loc[idx].isna().all():
            s.loc[idx] = 0.0
    return s


def gated_schedule_from_fcst(fcst: pd.Series, spec, dt, tz):
    filled = fcst.fillna(fcst.mean(skipna=True))
    s = daily_spread_gated_schedule(
        decision_prices=filled, execution_prices=filled,
        spec=spec, interval_hours=dt, cycles_per_day=1.0, tz=tz,
    )
    local_idx = fcst.index.tz_convert(tz) if tz else fcst.index
    day_key = pd.Series(local_idx.date, index=fcst.index)
    for _, idx in fcst.groupby(day_key).groups.items():
        if fcst.loc[idx].isna().all():
            s.loc[idx] = 0.0
    return s


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--splits-config", default="configs/splits.yaml")
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--solver", default="HIGHS")
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--scarcity-threshold-dollars", type=float, default=500.0)
    args = ap.parse_args()

    logging.basicConfig(level=logging.WARNING,
                        format="%(asctime)s %(levelname)s: %(message)s")

    with open(args.splits_config) as f:
        cfg = yaml.safe_load(f)
    location = cfg["location"]
    tz = cfg["tz"]
    retrain_every_days = cfg["walk_forward"]["retrain_every_days"]
    window = cfg[args.split]
    test_start = pd.Timestamp(window["start"], tz=tz).tz_convert("UTC")
    test_end = pd.Timestamp(window["end"], tz=tz).tz_convert("UTC") + pd.Timedelta(days=1) - pd.Timedelta("15min")
    start_year = pd.Timestamp(cfg["train"]["start"]).year
    end_year = pd.Timestamp(window["end"]).year

    print(f"Loading {location} RTM SPP + ERCOT load {start_year}–{end_year} …")
    prices = get_rtm_spp_series(location, start_year, end_year)
    load_series = get_load_series(start_year, end_year)["ercot_mw"]

    # ---------- Stage 1: walk-forward scarcity probs across full history.
    print("\n[1] Building daily features + walk-forward scarcity classifier over "
          "full history …")
    t0 = time.time()
    daily = build_daily_features(prices, tz=tz, load=load_series,
                                 scarcity_threshold=args.scarcity_threshold_dollars)
    scarcity_probs = walk_forward_scarcity_probs(
        daily, start_date=daily.index[60], end_date=daily.index[-1],
        retrain_every_days=retrain_every_days,
    )
    print(f"    probs generated: {scarcity_probs.notna().sum():,}/{len(scarcity_probs):,} "
          f"({time.time() - t0:.1f}s)")
    # Index must be datetime-like for reindex in feature broadcast.
    scarcity_probs.index = pd.to_datetime(scarcity_probs.index)

    # ---------- Stage 2: intraday features + walk-forward q50 LGBM.
    print("\n[2] Building intraday features (prices + load + scarcity_prob) …")
    feats = build_features(prices, tz=tz, load=load_series,
                           scarcity_prob_daily=scarcity_probs)
    feat_cols = [c for c in feats.columns if c != "target"]
    print(f"    feature cols ({len(feat_cols)}): {feat_cols}")

    print(f"\n[3] Walk-forward LGBM q50 on {args.split} window "
          f"[{test_start} → {test_end}], refit every {retrain_every_days}d, "
          f"{args.iters} iters …")
    t0 = time.time()
    q50_preds = walk_forward_predict(
        feats, "target", make_quantile_fit_fn(alpha=0.5, num_iterations=args.iters),
        test_start=test_start, test_end=test_end,
        retrain_every_days=retrain_every_days, min_train_rows=96 * 30,
    )
    print(f"    done ({time.time() - t0:.1f}s); "
          f"predictions: {q50_preds.notna().sum():,}")

    # ---------- Forecast quality on val.
    mask = ((q50_preds.index >= test_start) & (q50_preds.index <= test_end)
            & q50_preds.notna())
    y_true = prices.loc[mask].values
    y_pred = q50_preds.loc[mask].values
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    print(f"\n[4] q50 forecast quality on {args.split}: "
          f"MAE=${mae:.2f}, RMSE=${rmse:.2f}, n={mask.sum():,}")

    # ---------- Dispatch comparison.
    print("\n[5] Dispatching …")
    dt = INTERVAL_MINUTES / 60.0
    spec = DEFAULT_SPEC
    window_prices = prices.loc[test_start:test_end]
    window_preds = q50_preds.loc[test_start:test_end]

    runs: dict[str, pd.DataFrame] = {}
    sched = daily_oracle_schedule(window_prices, spec, dt, cycles_per_day=1.0, tz=tz)
    runs["1_floor"] = run_dispatch(sched, window_prices, spec, dt)

    t0 = time.time()
    sched = perfect_foresight_schedule(window_prices, spec, dt,
                                       cycles_per_day_cap=1.0, tz=tz,
                                       solver=args.solver)
    runs["4_ceiling"] = run_dispatch(sched, window_prices, spec, dt)

    sched = schedule_from_fcst(window_preds, spec, dt, tz)
    runs["2_combined_q50"] = run_dispatch(sched, window_prices, spec, dt)

    sched = gated_schedule_from_fcst(window_preds, spec, dt, tz)
    runs["3_combined_q50_gated"] = run_dispatch(sched, window_prices, spec, dt)

    print("\n=== Summary — val window ===")
    summary = compare(runs, tz=tz)
    print(summary.to_string(float_format=lambda x: f"{x:,.2f}"))

    floor_rev = runs["1_floor"]["net_revenue"].sum()
    ceiling_rev = runs["4_ceiling"]["net_revenue"].sum()
    print("\n=== % of ceiling captured ===")
    for name, res in runs.items():
        rev = res["net_revenue"].sum()
        print(f"  {name:28s}  {pct_of_ceiling(rev, ceiling_rev):5.1f}%   "
              f"${rev:>15,.2f}   lift vs floor: ${rev - floor_rev:>+14,.2f}")


if __name__ == "__main__":
    main()
