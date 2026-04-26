"""POST-HOC curiosity experiment — does ERCOT outage capacity (NP3-233-CD)
help dispatch revenue on the validation window?

VAL ONLY. The test set is BURNED per METHODOLOGY §1; this experiment
does not touch test and does not update the published 77.13% headline.

Setup: same locked spec as the published model (q50 LightGBM ensemble
+ EIA-930 + truncated training + forecast-gate). One arm with outage
features added, one without. Multi-seed (5).
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
from src.data.ercot_forecasts import load_forecasts  # noqa: E402
from src.data.eia930 import load_eia_series  # noqa: E402
from src.dispatch import run_dispatch  # noqa: E402
from src.evaluation import walk_forward_predict  # noqa: E402
from src.features import build_features  # noqa: E402
from src.metrics import pct_of_ceiling  # noqa: E402
from src.ml.lgbm import make_quantile_fit_fn  # noqa: E402
from src.optimization import perfect_foresight_schedule  # noqa: E402


SPEC = BatterySpec(
    power_mw=100.0, capacity_mwh=200.0, roundtrip_eff=0.85,
    soc_min_frac=0.05, soc_max_frac=0.95, initial_soc_frac=0.5,
    degradation_cost_per_mwh=2.0,
)
SEEDS = (7, 13, 23, 42, 101)


def schedule_from_fcst(fcst, dt, tz):
    filled = fcst.fillna(fcst.mean(skipna=True))
    s = daily_spread_gated_schedule(
        decision_prices=filled, execution_prices=filled,
        spec=SPEC, interval_hours=dt, cycles_per_day=1.0, tz=tz,
    )
    local_idx = fcst.index.tz_convert(tz) if tz else fcst.index
    day_key = pd.Series(local_idx.date, index=fcst.index)
    for _, idx in fcst.groupby(day_key).groups.items():
        if fcst.loc[idx].isna().all():
            s.loc[idx] = 0.0
    return s


def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    print("⚠️  POST-HOC EXPERIMENT — VAL WINDOW ONLY")
    print("This does not touch test; published 77.13% number is not affected.")
    print()

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
    outage = load_forecasts("outage_capacity")
    print(f"  prices: {len(prices):,}  load: {len(load_series):,}  EIA: {len(eia):,}")
    print(f"  outage docs: {len(outage):,} rows from "
          f"{outage['doc_id'].nunique() if len(outage) else 0} docs")

    dt = INTERVAL_MINUTES / 60.0
    train_start = pd.Timestamp("2019-01-01", tz="UTC")
    window_prices = prices.loc[test_start:test_end]
    floor_rev = float(run_dispatch(
        daily_oracle_schedule(window_prices, SPEC, dt, cycles_per_day=1.0, tz=tz),
        window_prices, SPEC, dt,
    )["net_revenue"].sum())
    ceil_rev = float(run_dispatch(
        perfect_foresight_schedule(window_prices, SPEC, dt,
                                   cycles_per_day_cap=1.0, tz=tz),
        window_prices, SPEC, dt,
    )["net_revenue"].sum())
    print(f"\nfloor on val:   ${floor_rev:,.2f}  (87.0% of ceiling)")
    print(f"ceiling on val: ${ceil_rev:,.2f}\n")

    variants = {
        "baseline (EIA only)": {"ercot_outage": None},
        "+outage capacity":    {"ercot_outage": outage if len(outage) else None},
    }

    rows: list[dict] = []
    for name, kw in variants.items():
        if "outage" in name and kw["ercot_outage"] is None:
            print(f"--- {name}: SKIPPED (no outage cache)")
            continue
        print(f"=== {name} ===")
        feats = build_features(prices, tz=tz, load=load_series, eia=eia, **kw)
        feat_cols = [c for c in feats.columns if c != "target"]
        print(f"  feature cols: {len(feat_cols)}")
        if "ercot_outage_total_mw" in feats.columns:
            print(f"  outage_total NaN rate: "
                  f"{feats['ercot_outage_total_mw'].isna().mean() * 100:.1f}%")

        for seed in SEEDS:
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
            sched = schedule_from_fcst(window_preds, dt, tz)
            revenue = float(run_dispatch(sched, window_prices, SPEC, dt)["net_revenue"].sum())
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

    if df["variant"].nunique() == 2:
        v_baseline = "baseline (EIA only)"
        v_outage = "+outage capacity"
        d = float(summary.loc[v_outage, "pct_mean"] - summary.loc[v_baseline, "pct_mean"])
        pooled = (float(summary.loc[v_baseline, "pct_std"])
                  + float(summary.loc[v_outage, "pct_std"])) / 2
        sigma = (d / pooled) if pooled > 0 else float("inf")
        print(f"\nΔ ('+outage' − 'baseline'): {d:+.2f} pp")
        print(f"Pooled std: {pooled:.2f} pp")
        print(f"Effect/noise: {sigma:+.2f}σ")
        print()
        if abs(sigma) >= 2:
            print("Above 2σ. Real signal — would warrant a fresh holdout test.")
        else:
            print("Within seed noise. Inconclusive.")


if __name__ == "__main__":
    main()
