"""Walk-forward LightGBM forecasting on HB_NORTH, validation period only.

Per METHODOLOGY.md §1: train on earliest 70% of history, tune on the
validation window. The test window is untouched here.

Pipeline:
    1. Load cached RTM SPPs from data/raw/ercot/.
    2. Build lag + calendar features (src.features.build_features).
    3. Walk-forward predict over the validation window, re-fitting every
       `retrain_every_days`.
    4. Build a dispatch schedule from the predictions (same threshold rule
       used for persistence / seasonal-naive).
    5. Execute on realized prices and compare to the natural-spread floor
       and the perfect-foresight ceiling *restricted to the same window*.

Forecasting metrics (MAE, RMSE) are reported alongside dispatch metrics so
we can see whether point-forecast accuracy translates into revenue lift.
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

from src.baselines import daily_oracle_schedule  # noqa: E402
from src.battery import BatterySpec  # noqa: E402
from src.data.ercot import INTERVAL_MINUTES, get_rtm_spp_series  # noqa: E402
from src.data.ercot_load import get_load_series  # noqa: E402
from src.dispatch import run_dispatch  # noqa: E402
from src.evaluation import walk_forward_predict  # noqa: E402
from src.features import build_features  # noqa: E402
from src.metrics import compare, pct_of_ceiling  # noqa: E402
from src.ml.lgbm import lgbm_fit_fn  # noqa: E402
from src.optimization import perfect_foresight_schedule  # noqa: E402


DEFAULT_SPEC = BatterySpec(
    power_mw=100.0,
    capacity_mwh=200.0,
    roundtrip_eff=0.85,
    soc_min_frac=0.05,
    soc_max_frac=0.95,
    initial_soc_frac=0.5,
    degradation_cost_per_mwh=2.0,
)


def _schedule_from_forecast(forecast: pd.Series, spec, dt: float, tz: str) -> pd.Series:
    """Build a day-by-day charge/discharge schedule from a (possibly-NaN)
    forecast. NaN days → idle."""
    filled = forecast.fillna(forecast.mean(skipna=True))
    sched = daily_oracle_schedule(filled, spec, dt, cycles_per_day=1.0, tz=tz)
    # Zero out days where forecast was entirely NaN.
    local_idx = forecast.index.tz_convert(tz) if tz else forecast.index
    day_key = pd.Series(local_idx.date, index=forecast.index)
    for _, idx in forecast.groupby(day_key).groups.items():
        if forecast.loc[idx].isna().all():
            sched.loc[idx] = 0.0
    return sched


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--splits-config", default="configs/splits.yaml")
    ap.add_argument("--split", default="val", choices=["val", "test"],
                    help="Which window to evaluate on (val by default; "
                         "test requires intentional reveal).")
    ap.add_argument("--solver", default="HIGHS")
    ap.add_argument("--with-load", action="store_true",
                    help="Include ERCOT load features (exogenous).")
    args = ap.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s: %(message)s")

    with open(args.splits_config) as f:
        cfg = yaml.safe_load(f)

    location = cfg["location"]
    tz = cfg["tz"]
    retrain_every_days = cfg["walk_forward"]["retrain_every_days"]
    window = cfg[args.split]
    train_start = pd.Timestamp(cfg["train"]["start"], tz=tz).tz_convert("UTC")
    test_start = pd.Timestamp(window["start"], tz=tz).tz_convert("UTC")
    test_end = pd.Timestamp(window["end"], tz=tz).tz_convert("UTC") + pd.Timedelta(days=1) - pd.Timedelta("15min")

    start_year = pd.Timestamp(cfg["train"]["start"]).year
    end_year = pd.Timestamp(window["end"]).year

    print(f"Loading {location} RTM SPP {start_year}–{end_year} …")
    prices = get_rtm_spp_series(location, start_year, end_year)
    print(f"  rows: {len(prices):,}, "
          f"span: {prices.index.min()} → {prices.index.max()}")

    load_series = None
    if args.with_load:
        print(f"Loading ERCOT load {start_year}–{end_year} …")
        load_df = get_load_series(start_year, end_year)
        load_series = load_df["ercot_mw"].rename("ercot_mw")
        print(f"  load rows: {len(load_series):,}")

    print("Building features …")
    t0 = time.time()
    feats = build_features(prices, tz=tz, load=load_series)
    print(f"  {len(feats.columns)} columns, built in {time.time() - t0:.1f}s")
    print(f"  feature cols: {[c for c in feats.columns if c != 'target']}")

    print(f"Walk-forward predict on {args.split} window "
          f"[{test_start} → {test_end}], refit every {retrain_every_days}d …")
    t0 = time.time()
    preds = walk_forward_predict(
        feats, target_col="target", fit_fn=lgbm_fit_fn,
        test_start=test_start, test_end=test_end,
        retrain_every_days=retrain_every_days,
        min_train_rows=96 * 30,  # at least ~30 days of features
    )
    print(f"  done in {time.time() - t0:.1f}s; "
          f"predictions: {preds.notna().sum():,}/{len(preds):,}")

    # Forecasting metrics on the window.
    window_mask = (preds.index >= test_start) & (preds.index <= test_end) & preds.notna()
    y_true = prices.loc[window_mask].values
    y_pred = preds.loc[window_mask].values
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    print(f"\nForecast quality on {args.split}: MAE=${mae:.2f}, RMSE=${rmse:.2f}, "
          f"n={window_mask.sum():,}")

    # Restrict to the test window for dispatch comparison.
    window_prices = prices.loc[test_start:test_end]
    window_preds = preds.loc[test_start:test_end]

    dt = INTERVAL_MINUTES / 60.0
    spec = DEFAULT_SPEC
    runs: dict[str, pd.DataFrame] = {}

    print("\nComputing in-window baselines and ceiling …")
    t0 = time.time()
    sched = daily_oracle_schedule(window_prices, spec, dt, cycles_per_day=1.0, tz=tz)
    runs["natural_spread_floor"] = run_dispatch(sched, window_prices, spec, dt)
    print(f"  floor done ({time.time() - t0:.1f}s)")

    t0 = time.time()
    sched = _schedule_from_forecast(window_preds, spec, dt, tz)
    runs["lgbm_walkforward"] = run_dispatch(sched, window_prices, spec, dt)
    print(f"  LGBM walk-forward dispatch done ({time.time() - t0:.1f}s)")

    t0 = time.time()
    sched = perfect_foresight_schedule(window_prices, spec, dt,
                                       cycles_per_day_cap=1.0, tz=tz,
                                       solver=args.solver)
    runs["perfect_foresight_ceiling"] = run_dispatch(sched, window_prices, spec, dt)
    print(f"  ceiling done ({time.time() - t0:.1f}s)")

    summary = compare(runs, tz=tz)
    print(f"\n=== Summary — {args.split} window ===")
    print(summary.to_string(float_format=lambda x: f"{x:,.2f}"))

    floor_rev = runs["natural_spread_floor"]["net_revenue"].sum()
    ceiling_rev = runs["perfect_foresight_ceiling"]["net_revenue"].sum()
    print("\n=== % of ceiling captured ===")
    for name, res in runs.items():
        rev = res["net_revenue"].sum()
        lift_over_floor = rev - floor_rev
        print(f"  {name:30s}  {pct_of_ceiling(rev, ceiling_rev):5.1f}%   "
              f"${rev:>15,.2f}   lift vs floor: ${lift_over_floor:>+14,.2f}")


if __name__ == "__main__":
    main()
