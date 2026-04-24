"""Quantile LightGBM walk-forward (q10 / q50 / q90) on validation.

Reports:
    - Quantile (pinball) loss at each tau.
    - 80% prediction interval coverage.
    - Dispatch using q50 as the point forecast.
    - Dispatch with a quantile-spread gate: skip days where the forecast's
      expected spread (median of q90 − median of q10 across intervals) is
      too narrow to clear round-trip + degradation costs.
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
from src.metrics import compare, pct_of_ceiling  # noqa: E402
from src.ml.lgbm import make_quantile_fit_fn  # noqa: E402
from src.optimization import perfect_foresight_schedule  # noqa: E402


DEFAULT_SPEC = BatterySpec(
    power_mw=100.0, capacity_mwh=200.0, roundtrip_eff=0.85,
    soc_min_frac=0.05, soc_max_frac=0.95, initial_soc_frac=0.5,
    degradation_cost_per_mwh=2.0,
)


def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, alpha: float) -> float:
    """Quantile (pinball) loss at level alpha."""
    d = y_true - y_pred
    return float(np.mean(np.maximum(alpha * d, (alpha - 1) * d)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--splits-config", default="configs/splits.yaml")
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--solver", default="HIGHS")
    ap.add_argument("--iters", type=int, default=300,
                    help="LGBM iterations per quantile fit (tune for speed).")
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
    feats = build_features(prices, tz=tz, load=load_df["ercot_mw"])
    print(f"  feature cols: {len([c for c in feats.columns if c != 'target'])}")

    preds: dict[str, pd.Series] = {}
    for alpha in [0.1, 0.5, 0.9]:
        print(f"\nWalk-forward fit LGBM quantile alpha={alpha} ({args.iters} iters)…")
        t0 = time.time()
        fit_fn = make_quantile_fit_fn(alpha=alpha, num_iterations=args.iters)
        p = walk_forward_predict(
            feats, "target", fit_fn,
            test_start=test_start, test_end=test_end,
            retrain_every_days=retrain_every_days,
            min_train_rows=96 * 30,
        )
        preds[f"q{int(alpha * 100):02d}"] = p
        print(f"  done ({time.time() - t0:.1f}s); n_preds={p.notna().sum():,}")

    # Forecast metrics.
    window_mask = (preds["q50"].index >= test_start) & (preds["q50"].index <= test_end) & preds["q50"].notna()
    y_true = prices.loc[window_mask].values
    print("\n=== Quantile forecast metrics on val window ===")
    for k in ["q10", "q50", "q90"]:
        yp = preds[k].loc[window_mask].values
        alpha = int(k[1:]) / 100
        pl = pinball_loss(y_true, yp, alpha)
        print(f"  {k}: pinball loss={pl:.3f}")
    lo = preds["q10"].loc[window_mask].values
    hi = preds["q90"].loc[window_mask].values
    coverage_80 = float(((y_true >= lo) & (y_true <= hi)).mean())
    print(f"  80% prediction-interval coverage = {coverage_80 * 100:.1f}% "
          f"(target 80.0%)")
    mean_width = float(np.mean(hi - lo))
    print(f"  mean q90-q10 width = ${mean_width:.2f}")

    # Dispatch.
    window_prices = prices.loc[test_start:test_end]
    dt = INTERVAL_MINUTES / 60.0
    spec = DEFAULT_SPEC
    runs: dict[str, pd.DataFrame] = {}

    # Floor, ceiling.
    sched = daily_oracle_schedule(window_prices, spec, dt, cycles_per_day=1.0, tz=tz)
    runs["1_floor"] = run_dispatch(sched, window_prices, spec, dt)
    t0 = time.time()
    sched = perfect_foresight_schedule(window_prices, spec, dt,
                                       cycles_per_day_cap=1.0, tz=tz,
                                       solver=args.solver)
    runs["5_ceiling"] = run_dispatch(sched, window_prices, spec, dt)
    print(f"  ceiling done ({time.time() - t0:.1f}s)")

    def schedule_from_fcst(fcst: pd.Series) -> pd.Series:
        filled = fcst.fillna(fcst.mean(skipna=True))
        s = daily_oracle_schedule(filled, spec, dt, cycles_per_day=1.0, tz=tz)
        local_idx = fcst.index.tz_convert(tz) if tz else fcst.index
        day_key = pd.Series(local_idx.date, index=fcst.index)
        for _, idx in fcst.groupby(day_key).groups.items():
            if fcst.loc[idx].isna().all():
                s.loc[idx] = 0.0
        return s

    # Q50 dispatch.
    q50 = preds["q50"].loc[test_start:test_end]
    sched_q50 = schedule_from_fcst(q50)
    runs["2_q50_dispatch"] = run_dispatch(sched_q50, window_prices, spec, dt)

    # Quantile-spread-gated dispatch: feed q50 as decision prices and
    # use the ORIGINAL gate (daily_spread_gated_schedule) but with forecast
    # used both as decision and execution (so gate reflects expected net).
    sched_q50_gated = daily_spread_gated_schedule(
        decision_prices=q50.fillna(q50.mean(skipna=True)),
        execution_prices=q50.fillna(q50.mean(skipna=True)),
        spec=spec, interval_hours=dt, cycles_per_day=1.0, tz=tz,
    )
    # Zero out warmup days where q50 was fully NaN.
    local_idx = q50.index.tz_convert(tz)
    day_key = pd.Series(local_idx.date, index=q50.index)
    for _, idx in q50.groupby(day_key).groups.items():
        if q50.loc[idx].isna().all():
            sched_q50_gated.loc[idx] = 0.0
    runs["3_q50_gated_fcst"] = run_dispatch(sched_q50_gated, window_prices, spec, dt)

    # Quantile-spread gate: explicit — require (q90_day_max − q10_day_min) to
    # exceed a breakeven threshold that roughly covers round-trip loss +
    # 2×degradation per MWh moved. If it doesn't, set that day idle.
    q10 = preds["q10"].loc[test_start:test_end]
    q90 = preds["q90"].loc[test_start:test_end]

    breakeven_per_mwh = (
        # per-MWh-grid-side round trip loss ≈ (1/eta − eta) × price
        # plus 2 × degradation (charge + discharge)
        (1 / spec.eta_half - spec.eta_half) * q50.median()
        + 2 * spec.degradation_cost_per_mwh
    )
    print(f"\n  q90-q10 gate breakeven spread = ${breakeven_per_mwh:.2f}/MWh")

    # Start from the q50 schedule, then zero out low-confidence days.
    sched_q_gate = sched_q50.copy()
    skipped = 0
    kept = 0
    for day, idx in q50.groupby(day_key).groups.items():
        day_q10 = q10.loc[idx]
        day_q90 = q90.loc[idx]
        if day_q10.isna().all() or day_q90.isna().all():
            sched_q_gate.loc[idx] = 0.0
            continue
        predicted_spread = float(day_q90.max() - day_q10.min())
        if predicted_spread < breakeven_per_mwh:
            sched_q_gate.loc[idx] = 0.0
            skipped += 1
        else:
            kept += 1
    print(f"  quantile-spread gate: kept {kept} days, skipped {skipped} days")
    runs["4_q_spread_gate"] = run_dispatch(sched_q_gate, window_prices, spec, dt)

    print("\n=== Dispatch summary — val window ===")
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
