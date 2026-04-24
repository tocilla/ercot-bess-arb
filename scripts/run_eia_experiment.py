"""Head-to-head experiment: does training on 2011+ (NaN exogenous) beat
training on 2019+ only, given a shared validation window?

Both variants:
    - Same features (prices + load + EIA-930 demand forecast + wind/solar lags)
    - Same walk-forward retrain cadence (monthly)
    - Same q50 LGBM with NaN handling enabled
    - Same forecast-gate dispatch
    - Same validation window (2020-11-01 → 2022-12-31)

Difference: training data availability.

    Variant A (full history):   train_start = 2011-01-01, NaN EIA for
                                pre-2019 rows. Uses full 10 years.
    Variant B (truncated):      train_start = 2019-01-01. Uses only 2019+
                                (data with full feature coverage).

Result: which dispatch revenue is higher on the val window.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.baselines import daily_oracle_schedule, daily_spread_gated_schedule  # noqa: E402
from src.battery import BatterySpec  # noqa: E402
from src.data.ercot import INTERVAL_MINUTES, get_rtm_spp_series  # noqa: E402
from src.data.ercot_load import get_load_series  # noqa: E402
from src.data.eia930 import load_eia_series  # noqa: E402
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


def schedule_from_fcst(fcst: pd.Series, spec, dt, tz):
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
    logging.basicConfig(level=logging.WARNING,
                        format="%(asctime)s %(levelname)s: %(message)s")

    with open("configs/splits.yaml") as f:
        cfg = yaml.safe_load(f)
    location = cfg["location"]
    tz = cfg["tz"]
    retrain_every_days = cfg["walk_forward"]["retrain_every_days"]
    window = cfg["val"]
    test_start = pd.Timestamp(window["start"], tz=tz).tz_convert("UTC")
    test_end = (pd.Timestamp(window["end"], tz=tz).tz_convert("UTC")
                + pd.Timedelta(days=1) - pd.Timedelta("15min"))

    print("Loading prices, load, and EIA-930 …")
    # Prices: 2011–2022 covers train+val.
    prices = get_rtm_spp_series(location, 2011, 2022)
    load_df = get_load_series(2011, 2022)
    load_series = load_df["ercot_mw"]
    eia = load_eia_series(2019, 2022, respondent="ERCO")  # EIA starts 2019
    print(f"  prices: {len(prices):,} rows")
    print(f"  load: {len(load_series):,} rows")
    print(f"  EIA: {len(eia):,} hourly rows, cols={list(eia.columns)}")

    print("\nBuilding features …")
    feats = build_features(prices, tz=tz, load=load_series, eia=eia)
    feat_cols = [c for c in feats.columns if c != "target"]
    print(f"  feature cols ({len(feat_cols)}): {feat_cols}")
    print(f"  NaN rates:")
    for c in feat_cols:
        nan_frac = feats[c].isna().mean()
        if nan_frac > 0.01:
            print(f"    {c}: {nan_frac * 100:.1f}% NaN")

    fit_fn = make_quantile_fit_fn(alpha=0.5, num_iterations=200)
    dt = INTERVAL_MINUTES / 60.0
    spec = DEFAULT_SPEC

    variants = {
        "A_full_history": {"train_start": None,
                           "label": "train 2011-01-01+ (full history, NaN for pre-2019 EIA)"},
        "B_truncated":    {"train_start": pd.Timestamp("2019-01-01", tz="UTC"),
                           "label": "train 2019-01-01+ (truncated)"},
    }

    all_results: dict[str, dict] = {}
    for name, cfg_v in variants.items():
        print(f"\n--- {name}: {cfg_v['label']} ---")
        t0 = time.time()
        preds = walk_forward_predict(
            feats, "target", fit_fn,
            test_start=test_start, test_end=test_end,
            retrain_every_days=retrain_every_days,
            min_train_rows=96 * 14,  # 2 weeks minimum
            allow_nan_features=True,
            train_start=cfg_v["train_start"],
        )
        print(f"  walk-forward done ({time.time() - t0:.1f}s); "
              f"n_preds={preds.notna().sum():,}")

        # Forecast quality.
        mask = ((preds.index >= test_start) & (preds.index <= test_end)
                & preds.notna())
        import numpy as np
        y_true = prices.loc[mask].values
        y_pred = preds.loc[mask].values
        mae = float(np.mean(np.abs(y_true - y_pred)))
        rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

        # Dispatch.
        window_preds = preds.loc[test_start:test_end]
        window_prices = prices.loc[test_start:test_end]
        sched = schedule_from_fcst(window_preds, spec, dt, tz)
        res = run_dispatch(sched, window_prices, spec, dt)

        revenue = float(res["net_revenue"].sum())
        all_results[name] = {
            "revenue": revenue,
            "mae": mae,
            "rmse": rmse,
            "result": res,
        }
        print(f"  MAE=${mae:.2f}, RMSE=${rmse:.2f}, revenue=${revenue:,.2f}")

    # Floor + ceiling for reference.
    window_prices = prices.loc[test_start:test_end]
    sched_floor = daily_oracle_schedule(window_prices, spec, dt,
                                        cycles_per_day=1.0, tz=tz)
    floor_res = run_dispatch(sched_floor, window_prices, spec, dt)
    floor_rev = float(floor_res["net_revenue"].sum())
    sched_ceil = perfect_foresight_schedule(window_prices, spec, dt,
                                            cycles_per_day_cap=1.0, tz=tz)
    ceil_res = run_dispatch(sched_ceil, window_prices, spec, dt)
    ceil_rev = float(ceil_res["net_revenue"].sum())

    print("\n=== Head-to-head summary (val window) ===")
    print(f"{'variant':30s} {'revenue':>16s} {'% ceiling':>10s} "
          f"{'lift vs floor':>16s} {'MAE':>10s}")
    print(f"{'floor':30s} ${floor_rev:>15,.2f} {87.0:>9.1f}% "
          f"{'baseline':>16s}")
    print(f"{'ceiling':30s} ${ceil_rev:>15,.2f} {100.0:>9.1f}% "
          f"${ceil_rev - floor_rev:>+15,.2f}")
    for name, r in all_results.items():
        rev = r["revenue"]
        pc = pct_of_ceiling(rev, ceil_rev)
        lift = rev - floor_rev
        print(f"{name:30s} ${rev:>15,.2f} {pc:>9.1f}% "
              f"${lift:>+15,.2f} {r['mae']:>9.2f}")


if __name__ == "__main__":
    main()
