"""Compare LGBM walk-forward with and without ERCOT STWPF + STPPF features.

Uses truncated training (2019-01-01+) and forecast-gated dispatch — the
session-best setup. Shared val window 2020-11-01 → 2022-12-31.

Multi-seed: each variant runs over multiple LGBM seeds (default 5).
Reports mean ± std for both arms so the head-to-head delta can be
compared against seed noise (~±2.84 pp baseline std, see FINDINGS).

    Baseline:  prices + load + EIA-930
    +ERCOT:    baseline + STWPF + STPPF
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


DEFAULT_SPEC = BatterySpec(
    power_mw=100.0, capacity_mwh=200.0, roundtrip_eff=0.85,
    soc_min_frac=0.05, soc_max_frac=0.95, initial_soc_frac=0.5,
    degradation_cost_per_mwh=2.0,
)

DEFAULT_SEEDS = (7, 13, 23, 42, 101)


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
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-solar", action="store_true",
                    help="Ignore the solar cache (use even if partial).")
    ap.add_argument("--no-wind", action="store_true",
                    help="Ignore the wind cache.")
    args = ap.parse_args()

    logging.basicConfig(level=logging.WARNING,
                        format="%(asctime)s %(levelname)s: %(message)s")
    with open("configs/splits.yaml") as f:
        cfg = yaml.safe_load(f)
    tz = cfg["tz"]
    retrain_every_days = cfg["walk_forward"]["retrain_every_days"]
    window = cfg["val"]
    test_start = pd.Timestamp(window["start"], tz=tz).tz_convert("UTC")
    test_end = (pd.Timestamp(window["end"], tz=tz).tz_convert("UTC")
                + pd.Timedelta(days=1) - pd.Timedelta("15min"))

    print("Loading inputs …")
    prices = get_rtm_spp_series(cfg["location"], 2011, 2022)
    load_series = get_load_series(2011, 2022)["ercot_mw"]
    eia = load_eia_series(2019, 2022, respondent="ERCO")
    wind_fcst = load_forecasts("wind") if not args.no_wind else pd.DataFrame()
    solar_fcst = load_forecasts("solar") if not args.no_solar else pd.DataFrame()
    print(f"  prices: {len(prices):,}")
    print(f"  load:   {len(load_series):,}")
    print(f"  EIA:    {len(eia):,}")
    print(f"  wind forecasts:  {len(wind_fcst):,} rows from "
          f"{len(set(wind_fcst['doc_id'])) if len(wind_fcst) else 0} docs")
    print(f"  solar forecasts: {len(solar_fcst):,} rows from "
          f"{len(set(solar_fcst['doc_id'])) if len(solar_fcst) else 0} docs")

    dt = INTERVAL_MINUTES / 60.0
    spec = DEFAULT_SPEC
    train_start = pd.Timestamp("2019-01-01", tz="UTC")
    window_prices = prices.loc[test_start:test_end]
    floor_rev = float(run_dispatch(
        daily_oracle_schedule(window_prices, spec, dt,
                              cycles_per_day=1.0, tz=tz),
        window_prices, spec, dt,
    )["net_revenue"].sum())
    ceil_rev = float(run_dispatch(
        perfect_foresight_schedule(window_prices, spec, dt,
                                   cycles_per_day_cap=1.0, tz=tz),
        window_prices, spec, dt,
    )["net_revenue"].sum())
    print(f"\nfloor:   ${floor_rev:,.2f}")
    print(f"ceiling: ${ceil_rev:,.2f}\n")

    variants: dict[str, dict] = {
        "baseline (EIA only)": {
            "ercot_wind_forecasts": None,
            "ercot_solar_forecasts": None,
        },
        "+ERCOT STWPF+STPPF": {
            "ercot_wind_forecasts": wind_fcst if len(wind_fcst) else None,
            "ercot_solar_forecasts": solar_fcst if len(solar_fcst) else None,
        },
    }

    seeds = DEFAULT_SEEDS
    rows: list[dict] = []
    for name, kw in variants.items():
        if ("ERCOT" in name and kw["ercot_wind_forecasts"] is None
                and kw["ercot_solar_forecasts"] is None):
            print(f"--- {name}: SKIPPED (no ERCOT forecasts cached)")
            continue
        print(f"=== {name} ===")
        feats = build_features(prices, tz=tz, load=load_series, eia=eia, **kw)
        feat_cols = [c for c in feats.columns if c != "target"]
        print(f"  feature cols: {len(feat_cols)}")
        if "ercot_stwpf_system_wide" in feats.columns:
            print(f"  STWPF NaN rate: "
                  f"{feats['ercot_stwpf_system_wide'].isna().mean() * 100:.1f}%")
        if "ercot_stppf_system_wide" in feats.columns:
            print(f"  STPPF NaN rate: "
                  f"{feats['ercot_stppf_system_wide'].isna().mean() * 100:.1f}%")

        for seed in seeds:
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
            rows.append({"variant": name, "seed": seed, "revenue": revenue,
                         "pct_ceiling": pc, "mae": mae})
            print(f"  seed {seed:3d}: revenue=${revenue:>14,.2f}  "
                  f"pct={pc:>5.2f}%  MAE=${mae:.2f}  ({time.time() - t0:.0f}s)")

    if not rows:
        print("\nNo runs completed.")
        return

    df = pd.DataFrame(rows)
    print("\n=== Per-seed table ===")
    print(df.to_string(index=False, float_format=lambda x: f"{x:,.2f}"))

    print("\n=== Mean ± std by variant ===")
    print(f"floor:   ${floor_rev:,.2f}  (87.0% of ceiling)")
    print(f"ceiling: ${ceil_rev:,.2f}\n")
    for variant in df["variant"].unique():
        sub = df[df["variant"] == variant]
        mean_rev = sub["revenue"].mean()
        std_rev = sub["revenue"].std()
        mean_pc = sub["pct_ceiling"].mean()
        std_pc = sub["pct_ceiling"].std()
        mean_mae = sub["mae"].mean()
        std_mae = sub["mae"].std()
        print(f"{variant:30s} revenue=${mean_rev:,.0f} ± ${std_rev:,.0f}  "
              f"pct={mean_pc:.2f} ± {std_pc:.2f} pp  "
              f"MAE=${mean_mae:.2f} ± ${std_mae:.2f}")

    if df["variant"].nunique() == 2:
        v1, v2 = df["variant"].unique()
        s1 = df[df["variant"] == v1]
        s2 = df[df["variant"] == v2]
        delta_pp = s2["pct_ceiling"].mean() - s1["pct_ceiling"].mean()
        pooled_std = ((s1["pct_ceiling"].std() + s2["pct_ceiling"].std()) / 2)
        print(f"\nDelta ('{v2}' − '{v1}'): {delta_pp:+.2f} pp of ceiling")
        print(f"Pooled std: {pooled_std:.2f} pp  →  effect/noise ≈ "
              f"{(delta_pp / pooled_std) if pooled_std > 0 else float('inf'):.2f}σ")
        print("Treat as real if |Δ| ≥ 2σ AND signs of individual seed deltas agree.")


if __name__ == "__main__":
    main()
