"""Compare LGBM walk-forward with and without HRRR features, using the
truncated training window (2019+) from the prior experiment.

All three runs share: same val window, same q50 fit, forecast-gate
dispatch, 2019+ training start.

    Baseline: prices + load + EIA-930 features
    HRRR:     baseline + HRRR Texas temperature + wind summary features

Reports revenue, % of ceiling, MAE on val.
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
from src.data.hrrr import CACHE_DIR as HRRR_CACHE  # noqa: E402
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


def load_hrrr_cache() -> pd.DataFrame:
    """Concatenate every cached HRRR summary parquet into one DataFrame."""
    files = sorted(HRRR_CACHE.glob("summary_*.parquet"))
    if not files:
        return pd.DataFrame()
    frames = [pd.read_parquet(f) for f in files]
    df = pd.concat(frames, ignore_index=True)
    return df


def main() -> None:
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
    hrrr = load_hrrr_cache()
    print(f"  prices: {len(prices):,}")
    print(f"  load:   {len(load_series):,}")
    print(f"  EIA:    {len(eia):,}")
    print(f"  HRRR:   {len(hrrr):,} rows ({hrrr['valid_utc'].min()} → "
          f"{hrrr['valid_utc'].max()}" if len(hrrr) else "HRRR:   empty")

    fit_fn = make_quantile_fit_fn(alpha=0.5, num_iterations=200)
    dt = INTERVAL_MINUTES / 60.0
    spec = DEFAULT_SPEC
    train_start = pd.Timestamp("2019-01-01", tz="UTC")

    variants = {
        "baseline (EIA only)": {"eia": eia, "hrrr": None},
        "EIA + HRRR":          {"eia": eia, "hrrr": hrrr if len(hrrr) else None},
    }

    results: dict[str, dict] = {}
    for name, kw in variants.items():
        if "HRRR" in name and kw["hrrr"] is None:
            print(f"\n--- {name}: skipped (HRRR cache empty)")
            continue
        print(f"\n--- {name} ---")
        feats = build_features(prices, tz=tz, load=load_series,
                               eia=kw["eia"], hrrr=kw["hrrr"])
        feat_cols = [c for c in feats.columns if c != "target"]
        print(f"  feature cols ({len(feat_cols)})")
        t0 = time.time()
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
        rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

        window_preds = preds.loc[test_start:test_end]
        window_prices = prices.loc[test_start:test_end]
        sched = schedule_from_fcst(window_preds, spec, dt, tz)
        res = run_dispatch(sched, window_prices, spec, dt)
        revenue = float(res["net_revenue"].sum())
        results[name] = {"revenue": revenue, "mae": mae, "rmse": rmse}
        print(f"  MAE=${mae:.2f}  RMSE=${rmse:.2f}  revenue=${revenue:,.2f}  "
              f"({time.time() - t0:.0f}s)")

    window_prices = prices.loc[test_start:test_end]
    floor = run_dispatch(
        daily_oracle_schedule(window_prices, spec, dt,
                              cycles_per_day=1.0, tz=tz),
        window_prices, spec, dt,
    )
    floor_rev = float(floor["net_revenue"].sum())
    ceil = run_dispatch(
        perfect_foresight_schedule(window_prices, spec, dt,
                                   cycles_per_day_cap=1.0, tz=tz),
        window_prices, spec, dt,
    )
    ceil_rev = float(ceil["net_revenue"].sum())

    print("\n=== Summary ===")
    print(f"{'variant':30s} {'revenue':>15s} {'% ceiling':>10s} "
          f"{'lift vs floor':>16s} {'MAE':>10s}")
    print(f"{'floor':30s} ${floor_rev:>14,.2f} {87.0:>9.1f}%")
    print(f"{'ceiling':30s} ${ceil_rev:>14,.2f} {100.0:>9.1f}% "
          f"${ceil_rev - floor_rev:>+15,.2f}")
    for name, r in results.items():
        pc = pct_of_ceiling(r["revenue"], ceil_rev)
        lift = r["revenue"] - floor_rev
        print(f"{name:30s} ${r['revenue']:>14,.2f} {pc:>9.1f}% "
              f"${lift:>+15,.2f} {r['mae']:>9.2f}")


if __name__ == "__main__":
    main()
