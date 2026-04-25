"""Two cheap pre-reveal analyses on the session-best ML setup:

  (1) Seed ensemble — average predictions from 5 trained models, run
      dispatch on the ensemble. Possible ~1-2 pp lift from variance
      reduction.

  (2) Regime-stratified revenue — show where the session-best ML
      beats/loses to the floor (normal vs scarcity vs negative-only
      days). Diagnostic, frames the test-set reveal honestly.

Same setup as session-best:
  q50 LGBM + EIA-930 + load + 2019+ truncated training + gate dispatch
  on the val window 2020-11 → 2022-12.
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
from src.metrics import pct_of_ceiling, regime_breakdown  # noqa: E402
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

    dt = INTERVAL_MINUTES / 60.0
    train_start = pd.Timestamp("2019-01-01", tz="UTC")
    window_prices = prices.loc[test_start:test_end]
    floor_res = run_dispatch(
        daily_oracle_schedule(window_prices, SPEC, dt, cycles_per_day=1.0, tz=tz),
        window_prices, SPEC, dt,
    )
    floor_rev = float(floor_res["net_revenue"].sum())
    ceil_res = run_dispatch(
        perfect_foresight_schedule(window_prices, SPEC, dt,
                                   cycles_per_day_cap=1.0, tz=tz),
        window_prices, SPEC, dt,
    )
    ceil_rev = float(ceil_res["net_revenue"].sum())
    print(f"\nfloor:   ${floor_rev:,.2f}  (87.0% of ceiling)")
    print(f"ceiling: ${ceil_rev:,.2f}\n")

    # ----- (1) Seed ensemble -----
    print("=== (1) Seed ensemble ===")
    seed_preds: dict[int, pd.Series] = {}
    seed_rows: list[dict] = []
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
        seed_preds[seed] = preds.loc[test_start:test_end]
        sched = schedule_from_fcst(seed_preds[seed], dt, tz)
        res = run_dispatch(sched, window_prices, SPEC, dt)
        rev = float(res["net_revenue"].sum())
        pc = pct_of_ceiling(rev, ceil_rev)
        seed_rows.append({"seed": seed, "revenue": rev, "pct_ceiling": pc, "result": res})
        print(f"  seed {seed:3d}: revenue=${rev:>14,.2f}  pct={pc:>5.2f}%  "
              f"({time.time() - t0:.0f}s)")

    ensemble_preds = pd.concat([s.rename(seed) for seed, s in seed_preds.items()],
                               axis=1).mean(axis=1)
    sched_e = schedule_from_fcst(ensemble_preds, dt, tz)
    ens_res = run_dispatch(sched_e, window_prices, SPEC, dt)
    ens_rev = float(ens_res["net_revenue"].sum())
    ens_pc = pct_of_ceiling(ens_rev, ceil_rev)

    df_seeds = pd.DataFrame([{k: v for k, v in r.items() if k != "result"}
                             for r in seed_rows])
    seed_mean_rev = float(df_seeds["revenue"].mean())
    seed_mean_pc = float(df_seeds["pct_ceiling"].mean())
    seed_std_pc = float(df_seeds["pct_ceiling"].std())

    print(f"\n  individual mean: revenue=${seed_mean_rev:,.2f}  "
          f"pct={seed_mean_pc:.2f} ± {seed_std_pc:.2f} pp")
    print(f"  ENSEMBLE:        revenue=${ens_rev:,.2f}  pct={ens_pc:.2f}%")
    print(f"  Δ ensemble vs individual mean: "
          f"{ens_pc - seed_mean_pc:+.2f} pp")
    print(f"  Δ ensemble vs floor:           "
          f"{(ens_pc - 87.0):+.2f} pp")

    # ----- (2) Regime breakdown -----
    print("\n=== (2) Regime breakdown (ensemble vs floor) ===")
    floor_rb = regime_breakdown(floor_res, window_prices, tz=tz)
    ml_rb = regime_breakdown(ens_res, window_prices, tz=tz)
    ceil_rb = regime_breakdown(ceil_res, window_prices, tz=tz)

    print("\nFloor:")
    print(floor_rb.to_string(float_format=lambda x: f"{x:,.2f}"))
    print("\nEnsemble (session-best ML):")
    print(ml_rb.to_string(float_format=lambda x: f"{x:,.2f}"))
    print("\nCeiling (perfect foresight):")
    print(ceil_rb.to_string(float_format=lambda x: f"{x:,.2f}"))

    # Side-by-side per regime
    print("\n=== Side-by-side per regime ===")
    regimes = sorted(set(floor_rb.index) | set(ml_rb.index))
    print(f"{'regime':18s} {'n_days':>7s} "
          f"{'floor_rev':>13s} {'ml_rev':>13s} {'ml-floor':>13s} "
          f"{'ml/floor':>9s}")
    for r in regimes:
        n = int(floor_rb.loc[r, "n_days"]) if r in floor_rb.index else 0
        fr = float(floor_rb.loc[r, "total_revenue"]) if r in floor_rb.index else 0
        mr = float(ml_rb.loc[r, "total_revenue"]) if r in ml_rb.index else 0
        delta = mr - fr
        ratio = (mr / fr) if fr != 0 else float("nan")
        print(f"{r:18s} {n:>7d} "
              f"${fr:>12,.0f} ${mr:>12,.0f} ${delta:>+12,.0f} "
              f"{ratio:>8.1%}" if not np.isnan(ratio) else
              f"{r:18s} {n:>7d} "
              f"${fr:>12,.0f} ${mr:>12,.0f} ${delta:>+12,.0f}      n/a")


if __name__ == "__main__":
    main()
