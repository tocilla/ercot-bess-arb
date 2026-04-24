"""Scarcity classifier walk-forward on validation.

1. Build daily features + target from intraday prices and (optional) load.
2. Walk-forward fit a LightGBM binary classifier monthly.
3. Report PR-AUC, precision-at-recall = {0.3, 0.5}, confusion matrix.
4. Integrate predictions with a fallback schedule (LGBM forecast dispatch,
   re-fit here for consistency) to produce a scarcity-aware dispatch.
5. Compare: floor / LGBM-only / scarcity-aware / ceiling.
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
from sklearn.metrics import (
    average_precision_score, precision_recall_curve, confusion_matrix,
)

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.baselines import daily_oracle_schedule  # noqa: E402
from src.battery import BatterySpec  # noqa: E402
from src.data.ercot import INTERVAL_MINUTES, get_rtm_spp_series  # noqa: E402
from src.data.ercot_load import get_load_series  # noqa: E402
from src.dispatch import run_dispatch  # noqa: E402
from src.evaluation import walk_forward_predict  # noqa: E402
from src.features import build_features  # noqa: E402
from src.features_daily import build_daily_features  # noqa: E402
from src.metrics import compare, pct_of_ceiling  # noqa: E402
from src.ml.lgbm import lgbm_fit_fn  # noqa: E402
from src.ml.scarcity_classifier import scarcity_fit_fn  # noqa: E402
from src.optimization import perfect_foresight_schedule  # noqa: E402
from src.scarcity_dispatch import combined_schedule  # noqa: E402


DEFAULT_SPEC = BatterySpec(
    power_mw=100.0, capacity_mwh=200.0, roundtrip_eff=0.85,
    soc_min_frac=0.05, soc_max_frac=0.95, initial_soc_frac=0.5,
    degradation_cost_per_mwh=2.0,
)


def _schedule_from_point_forecast(forecast, spec, dt, tz):
    filled = forecast.fillna(forecast.mean(skipna=True))
    sched = daily_oracle_schedule(filled, spec, dt, cycles_per_day=1.0, tz=tz)
    local_idx = forecast.index.tz_convert(tz) if tz else forecast.index
    day_key = pd.Series(local_idx.date, index=forecast.index)
    for _, idx in forecast.groupby(day_key).groups.items():
        if forecast.loc[idx].isna().all():
            sched.loc[idx] = 0.0
    return sched


def _walk_forward_daily_classifier(
    daily_features: pd.DataFrame,
    test_start_date,
    test_end_date,
    retrain_every_days: int = 30,
    min_train_rows: int = 300,
) -> pd.Series:
    """Walk-forward LGBM binary for daily scarcity. Returns probs indexed by date."""
    feat_cols = [c for c in daily_features.columns if c not in {
        "target_scarcity", "target_max_price"
    }]

    df = daily_features.sort_index()
    df = df.dropna(subset=feat_cols + ["target_scarcity"])

    probs = pd.Series(np.nan, index=df.index, name="p_scarcity")

    test_dates = df.index[(df.index >= test_start_date) & (df.index <= test_end_date)]
    if len(test_dates) == 0:
        return probs

    boundaries = [test_dates[0]]
    while boundaries[-1] + pd.Timedelta(days=retrain_every_days) <= test_dates[-1]:
        boundaries.append(boundaries[-1] + pd.Timedelta(days=retrain_every_days))

    for i, b in enumerate(boundaries):
        next_b = (boundaries[i + 1] if i + 1 < len(boundaries)
                  else test_dates[-1] + pd.Timedelta(days=1))
        train_mask = df.index < b
        if train_mask.sum() < min_train_rows:
            raise RuntimeError(f"At {b}: only {train_mask.sum()} train rows")
        clf = scarcity_fit_fn(df.loc[train_mask, feat_cols],
                              df.loc[train_mask, "target_scarcity"])
        pred_mask = (df.index >= b) & (df.index < next_b)
        probs.loc[df.index[pred_mask]] = clf.predict(df.loc[pred_mask, feat_cols])
    return probs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--splits-config", default="configs/splits.yaml")
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--solver", default="HIGHS")
    ap.add_argument("--scarcity-threshold-dollars", type=float, default=500.0)
    ap.add_argument("--prob-threshold", type=float, default=0.5,
                    help="Decision threshold for scarcity-aware dispatch.")
    args = ap.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s: %(message)s")

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

    print(f"Loading {location} RTM SPP and ERCOT load {start_year}–{end_year} …")
    prices = get_rtm_spp_series(location, start_year, end_year)
    load_df = get_load_series(start_year, end_year)
    load_series = load_df["ercot_mw"]

    print("\n--- Scarcity classifier (daily LGBM) ---")
    daily = build_daily_features(prices, tz=tz, load=load_series,
                                 scarcity_threshold=args.scarcity_threshold_dollars)
    print(f"  daily rows: {len(daily)}, scarcity rate overall: "
          f"{daily['target_scarcity'].mean() * 100:.1f}%")

    test_start_date = pd.Timestamp(window["start"]).date()
    test_end_date = pd.Timestamp(window["end"]).date()

    t0 = time.time()
    scarcity_probs = _walk_forward_daily_classifier(
        daily, test_start_date, test_end_date,
        retrain_every_days=retrain_every_days,
    )
    print(f"  walk-forward classify done in {time.time() - t0:.1f}s")

    # Classification metrics on val window — align on the classifier's
    # index (which may drop warmup NaN rows).
    valid_idx = scarcity_probs.dropna().index
    val_mask = (valid_idx >= test_start_date) & (valid_idx <= test_end_date)
    val_idx = valid_idx[val_mask]
    y_true = daily.loc[val_idx, "target_scarcity"].astype(int).values
    y_prob = scarcity_probs.loc[val_idx].values
    pos_rate = float(np.mean(y_true))
    pr_auc = float(average_precision_score(y_true, y_prob))
    prec, rec, thresh = precision_recall_curve(y_true, y_prob)

    def prec_at_recall(target_recall):
        mask = rec >= target_recall
        if mask.sum() == 0:
            return float("nan")
        return float(prec[mask].max())

    y_pred = (y_prob > args.prob_threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    print(f"  val scarcity-rate (positive): {pos_rate * 100:.1f}% "
          f"({int(y_true.sum())} / {len(y_true)} days)")
    print(f"  PR-AUC: {pr_auc:.3f}  (baseline = {pos_rate:.3f})")
    print(f"  precision@recall=0.3: {prec_at_recall(0.3):.3f}")
    print(f"  precision@recall=0.5: {prec_at_recall(0.5):.3f}")
    print(f"  at threshold={args.prob_threshold}: TP={tp} FP={fp} TN={tn} FN={fn}")

    # --- Dispatch comparison ---
    print("\n--- Dispatch comparison on validation window ---")
    window_prices = prices.loc[test_start:test_end]
    dt = INTERVAL_MINUTES / 60.0
    spec = DEFAULT_SPEC

    runs: dict[str, pd.DataFrame] = {}

    # Floor.
    sched = daily_oracle_schedule(window_prices, spec, dt, cycles_per_day=1.0, tz=tz)
    runs["1_floor"] = run_dispatch(sched, window_prices, spec, dt)

    # Ceiling.
    sched = perfect_foresight_schedule(window_prices, spec, dt,
                                       cycles_per_day_cap=1.0, tz=tz,
                                       solver=args.solver)
    runs["5_ceiling"] = run_dispatch(sched, window_prices, spec, dt)

    # LGBM + load forecast dispatch (same as previous experiment).
    print("  fitting LGBM point-forecast walk-forward (reused for fallback)…")
    t0 = time.time()
    feats = build_features(prices, tz=tz, load=load_series)
    preds = walk_forward_predict(feats, "target", lgbm_fit_fn,
                                 test_start=test_start, test_end=test_end,
                                 retrain_every_days=retrain_every_days,
                                 min_train_rows=96 * 30)
    print(f"    done ({time.time() - t0:.1f}s)")
    window_preds = preds.loc[test_start:test_end]
    lgbm_sched = _schedule_from_point_forecast(window_preds, spec, dt, tz)
    runs["2_lgbm_load"] = run_dispatch(lgbm_sched, window_prices, spec, dt)

    # Scarcity-aware: override LGBM schedule with rule on predicted-scarcity days.
    combined = combined_schedule(
        fallback=lgbm_sched, scarcity_prob=scarcity_probs,
        threshold=args.prob_threshold,
        prices_index=window_prices.index,
        spec=spec, interval_hours=dt, tz=tz,
    )
    runs["3_scarcity_aware"] = run_dispatch(combined, window_prices, spec, dt)

    # Oracle scarcity (cheating upper bound — shows value of a PERFECT classifier)
    oracle_combined = combined_schedule(
        fallback=lgbm_sched,
        scarcity_prob=daily["target_scarcity"].astype(float),  # 1.0 on true scarcity
        threshold=0.5,
        prices_index=window_prices.index,
        spec=spec, interval_hours=dt, tz=tz,
    )
    runs["4_scarcity_oracle"] = run_dispatch(oracle_combined, window_prices, spec, dt)

    print("\n=== Summary — val window ===")
    summary = compare(runs, tz=tz)
    print(summary.to_string(float_format=lambda x: f"{x:,.2f}"))

    floor_rev = runs["1_floor"]["net_revenue"].sum()
    ceiling_rev = runs["5_ceiling"]["net_revenue"].sum()
    print("\n=== % of ceiling captured ===")
    for name, res in runs.items():
        rev = res["net_revenue"].sum()
        print(f"  {name:25s}  {pct_of_ceiling(rev, ceiling_rev):5.1f}%   "
              f"${rev:>15,.2f}   lift vs floor: ${rev - floor_rev:>+14,.2f}")


if __name__ == "__main__":
    main()
