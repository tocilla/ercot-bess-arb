"""Quantify seed-noise in the session-best ML setup.

Runs the q50 LGBM + forecast-gate baseline (prices + load + EIA-930,
2019+ truncated training, val window) with N different random seeds.
Reports revenue mean ± std + forecast MAE mean ± std.

Every finding that claims "strategy X beat strategy Y by K pp" requires
K to exceed this noise floor. If seed std is large, small headline
differences are not real.
"""

from __future__ import annotations

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
from src.data.eia930 import load_eia_series  # noqa: E402
from src.dispatch import run_dispatch  # noqa: E402
from src.evaluation import walk_forward_predict  # noqa: E402
from src.features import build_features  # noqa: E402
from src.metrics import pct_of_ceiling  # noqa: E402
from src.ml.lgbm import make_quantile_fit_fn  # noqa: E402
from src.optimization import perfect_foresight_schedule  # noqa: E402


DEFAULT_SPEC = BatterySpec(
    power_mw=100.0, capacity_mwh=200.0, roundtrip_eff=0.85,
    soc_min_frac=0.05, soc_max_frac=0.95, initial_soc_frac=0.5,
    degradation_cost_per_mwh=2.0,
)

SEEDS = [7, 13, 23, 42, 101]


def schedule_from_fcst(fcst, spec, dt, tz):
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
    logging.basicConfig(level=logging.WARNING)
    with open("configs/splits.yaml") as f:
        cfg = yaml.safe_load(f)
    tz = cfg["tz"]
    retrain_every_days = cfg["walk_forward"]["retrain_every_days"]
    test_start = pd.Timestamp(cfg["val"]["start"], tz=tz).tz_convert("UTC")
    test_end = (pd.Timestamp(cfg["val"]["end"], tz=tz).tz_convert("UTC")
                + pd.Timedelta(days=1) - pd.Timedelta("15min"))

    print("Loading …")
    prices = get_rtm_spp_series(cfg["location"], 2011, 2022)
    load_series = get_load_series(2011, 2022)["ercot_mw"]
    eia = load_eia_series(2019, 2022, respondent="ERCO")

    feats = build_features(prices, tz=tz, load=load_series, eia=eia)
    print(f"  {len([c for c in feats.columns if c != 'target'])} features")

    dt = INTERVAL_MINUTES / 60.0
    spec = DEFAULT_SPEC
    train_start = pd.Timestamp("2019-01-01", tz="UTC")
    window_prices = prices.loc[test_start:test_end]

    # Fixed anchors: floor + ceiling don't depend on seed.
    floor_rev = float(run_dispatch(
        daily_oracle_schedule(window_prices, spec, dt, cycles_per_day=1.0, tz=tz),
        window_prices, spec, dt,
    )["net_revenue"].sum())
    ceil_rev = float(run_dispatch(
        perfect_foresight_schedule(window_prices, spec, dt,
                                   cycles_per_day_cap=1.0, tz=tz),
        window_prices, spec, dt,
    )["net_revenue"].sum())

    rows = []
    for seed in SEEDS:
        print(f"\n--- seed {seed} ---")
        t0 = time.time()
        fit_fn = make_quantile_fit_fn(alpha=0.5, num_iterations=200, seed=seed)
        preds = walk_forward_predict(
            feats, "target", fit_fn,
            test_start=test_start, test_end=test_end,
            retrain_every_days=retrain_every_days,
            min_train_rows=96 * 14,
            allow_nan_features=True,
            train_start=train_start,
        )
        mask = ((preds.index >= test_start) & (preds.index <= test_end)
                & preds.notna())
        y_true = prices.loc[mask].values
        y_pred = preds.loc[mask].values
        mae = float(np.mean(np.abs(y_true - y_pred)))

        window_preds = preds.loc[test_start:test_end]
        sched = schedule_from_fcst(window_preds, spec, dt, tz)
        revenue = float(run_dispatch(sched, window_prices, spec, dt)["net_revenue"].sum())
        pc = pct_of_ceiling(revenue, ceil_rev)

        rows.append({"seed": seed, "revenue": revenue, "pct_ceiling": pc, "mae": mae})
        print(f"  revenue=${revenue:,.2f}  pct={pc:.2f}%  MAE=${mae:.2f}  "
              f"({time.time() - t0:.0f}s)")

    df = pd.DataFrame(rows)
    print("\n=== Seed-stability summary ===")
    print(df.to_string(index=False, float_format=lambda x: f"{x:,.2f}"))
    print()
    print(f"floor:   ${floor_rev:,.2f}  87.0% of ceiling")
    print(f"ceiling: ${ceil_rev:,.2f}")
    print()
    print(f"revenue:     mean=${df['revenue'].mean():,.2f}  "
          f"std=${df['revenue'].std():,.2f}  "
          f"(range=${df['revenue'].max() - df['revenue'].min():,.2f})")
    print(f"pct_ceiling: mean={df['pct_ceiling'].mean():.2f}%  "
          f"std={df['pct_ceiling'].std():.2f} pp  "
          f"(range={df['pct_ceiling'].max() - df['pct_ceiling'].min():.2f} pp)")
    print(f"MAE:         mean=${df['mae'].mean():.2f}  "
          f"std=${df['mae'].std():.2f}")


if __name__ == "__main__":
    main()
