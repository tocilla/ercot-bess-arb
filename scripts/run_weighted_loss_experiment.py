"""Decision-aware-loss experiment: same model, same features, different
sample weights. Measures whether weighted loss closes the MAE-vs-revenue
gap that has shown up in 5 prior experiments this session.

Variants (all q50 LGBM, 200 iters, 2019+ truncated training, forecast-
gate dispatch, multi-seed):

    A. baseline (uniform weights)
    B. devmean    — weight ∝ |y - daily_mean|
    C. topbotk    — weight = 5× on top-k and bottom-k of each day
    D. pricemag   — weight ∝ 1 + |y|/100   (heavier on extremes incl. neg)

Same val window (2020-11 → 2022-12). Reports mean ± std and
effect/noise vs A across 5 seeds.
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
from src.ml.loss_weights import (  # noqa: E402
    deviation_from_daily_mean, price_magnitude, top_bottom_k_per_day,
)
from src.optimization import perfect_foresight_schedule  # noqa: E402


DEFAULT_SPEC = BatterySpec(
    power_mw=100.0, capacity_mwh=200.0, roundtrip_eff=0.85,
    soc_min_frac=0.05, soc_max_frac=0.95, initial_soc_frac=0.5,
    degradation_cost_per_mwh=2.0,
)
DEFAULT_SEEDS = (7, 13, 23, 42, 101)


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

    print("Loading inputs …")
    prices = get_rtm_spp_series(cfg["location"], 2011, 2022)
    load_series = get_load_series(2011, 2022)["ercot_mw"]
    eia = load_eia_series(2019, 2022, respondent="ERCO")
    feats = build_features(prices, tz=tz, load=load_series, eia=eia)
    print(f"  feature cols: {len([c for c in feats.columns if c != 'target'])}")

    dt = INTERVAL_MINUTES / 60.0
    spec = DEFAULT_SPEC
    train_start = pd.Timestamp("2019-01-01", tz="UTC")
    window_prices = prices.loc[test_start:test_end]
    floor_rev = float(run_dispatch(
        daily_oracle_schedule(window_prices, spec, dt, cycles_per_day=1.0, tz=tz),
        window_prices, spec, dt,
    )["net_revenue"].sum())
    ceil_rev = float(run_dispatch(
        perfect_foresight_schedule(window_prices, spec, dt,
                                   cycles_per_day_cap=1.0, tz=tz),
        window_prices, spec, dt,
    )["net_revenue"].sum())
    print(f"\nfloor:   ${floor_rev:,.2f}")
    print(f"ceiling: ${ceil_rev:,.2f}\n")

    variants: dict[str, "callable | None"] = {
        "A_uniform": None,
        "B_devmean": deviation_from_daily_mean,
        "C_topbotk": top_bottom_k_per_day,
        "D_pricemag": price_magnitude,
    }

    rows: list[dict] = []
    for name, weight_fn in variants.items():
        print(f"=== {name} ===")
        for seed in DEFAULT_SEEDS:
            t0 = time.time()
            fit_fn = make_quantile_fit_fn(
                alpha=0.5, num_iterations=200, seed=seed, weight_fn=weight_fn,
            )
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
            rows.append({"variant": name, "seed": seed, "revenue": revenue,
                         "pct_ceiling": pc, "mae": mae})
            print(f"  seed {seed:3d}: revenue=${revenue:>14,.2f}  "
                  f"pct={pc:>5.2f}%  MAE=${mae:.2f}  ({time.time() - t0:.0f}s)")

    df = pd.DataFrame(rows)
    print("\n=== Per-seed table ===")
    print(df.to_string(index=False, float_format=lambda x: f"{x:,.2f}"))

    print("\n=== Mean ± std by variant ===")
    summary = df.groupby("variant").agg(
        revenue_mean=("revenue", "mean"),
        revenue_std=("revenue", "std"),
        pct_mean=("pct_ceiling", "mean"),
        pct_std=("pct_ceiling", "std"),
        mae_mean=("mae", "mean"),
    )
    print(summary.to_string(float_format=lambda x: f"{x:,.2f}"))

    base_pct = float(summary.loc["A_uniform", "pct_mean"])
    base_std = float(summary.loc["A_uniform", "pct_std"])
    print("\n=== Δ vs uniform baseline ===")
    for name in summary.index:
        if name == "A_uniform":
            continue
        d = float(summary.loc[name, "pct_mean"]) - base_pct
        v_std = float(summary.loc[name, "pct_std"])
        pooled = (base_std + v_std) / 2
        sigma = d / pooled if pooled > 0 else float("inf")
        print(f"  {name:12s}  Δ={d:+.2f} pp  pooled_std={pooled:.2f}  "
              f"effect/noise={sigma:+.2f}σ")


if __name__ == "__main__":
    main()
