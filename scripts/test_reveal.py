"""Final test-set reveal — one-shot.

Runs the locked-in spec from FINDINGS.md (q50 LGBM ensemble + EIA-930
features + 2019+ truncated training + forecast-gate) on the held-out
test window 2023-01-01 → 2024-12-31, alongside every deployable
baseline + theoretical bound.

After this script runs the test set is BURNED per METHODOLOGY §1; no
further model selection or tuning permitted on test results.
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
from src.forecasters import (  # noqa: E402
    persistence_forecast_same_interval_yesterday, seasonal_naive_forecast,
)
from src.metrics import pct_of_ceiling, regime_breakdown  # noqa: E402
from src.ml.lgbm import make_quantile_fit_fn  # noqa: E402
from src.optimization import perfect_foresight_schedule  # noqa: E402


SPEC = BatterySpec(
    power_mw=100.0, capacity_mwh=200.0, roundtrip_eff=0.85,
    soc_min_frac=0.05, soc_max_frac=0.95, initial_soc_frac=0.5,
    degradation_cost_per_mwh=2.0,
)
SEEDS = (7, 13, 23, 42, 101)
INTERVALS_PER_DAY = 96


def schedule_from_decision_prices(decision_prices, dt, tz, gated=True):
    filled = decision_prices.fillna(decision_prices.mean(skipna=True))
    if gated:
        s = daily_spread_gated_schedule(
            decision_prices=filled, execution_prices=filled,
            spec=SPEC, interval_hours=dt, cycles_per_day=1.0, tz=tz,
        )
    else:
        s = daily_oracle_schedule(filled, SPEC, dt, cycles_per_day=1.0, tz=tz)
    local_idx = decision_prices.index.tz_convert(tz) if tz else decision_prices.index
    day_key = pd.Series(local_idx.date, index=decision_prices.index)
    for _, idx in decision_prices.groupby(day_key).groups.items():
        if decision_prices.loc[idx].isna().all():
            s.loc[idx] = 0.0
    return s


def fixed_time_of_day_schedule(prices_index, dt, tz, charge_hour=3, discharge_hour=17):
    """Charge at a fixed local hour, discharge at another. No forecast,
    no model. Sized to roughly match cycles_per_day=1 throughput."""
    local = prices_index.tz_convert(tz)
    hour = local.hour
    sched = pd.Series(0.0, index=prices_index, name="grid_power_mw")
    sched[hour == charge_hour] = SPEC.power_mw
    sched[hour == discharge_hour] = -SPEC.power_mw
    return sched


def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    with open("configs/splits.yaml") as f:
        cfg = yaml.safe_load(f)
    tz = cfg["tz"]
    retrain_every_days = cfg["walk_forward"]["retrain_every_days"]
    test_window = cfg["test"]
    test_start = pd.Timestamp(test_window["start"], tz=tz).tz_convert("UTC")
    test_end = (pd.Timestamp(test_window["end"], tz=tz).tz_convert("UTC")
                + pd.Timedelta(days=1) - pd.Timedelta("15min"))

    print("=" * 70)
    print(f"TEST-SET REVEAL — {test_window['start']} → {test_window['end']}")
    print("=" * 70)
    print()

    print("Loading inputs (full available history) …")
    prices = get_rtm_spp_series(cfg["location"], 2011, 2024)
    load_series = get_load_series(2011, 2024)["ercot_mw"]
    eia = load_eia_series(2019, 2024, respondent="ERCO")
    feats = build_features(prices, tz=tz, load=load_series, eia=eia)
    print(f"  prices: {len(prices):,}  load: {len(load_series):,}  EIA: {len(eia):,}")
    print(f"  features: {len([c for c in feats.columns if c != 'target'])}")
    print()

    dt = INTERVAL_MINUTES / 60.0
    train_start = pd.Timestamp("2019-01-01", tz="UTC")
    window_prices = prices.loc[test_start:test_end]
    n_days = len(set(window_prices.index.tz_convert(tz).date))
    print(f"Test window: {len(window_prices):,} intervals across {n_days} days")
    print(f"Price stats: mean=${window_prices.mean():.2f}  median=${window_prices.median():.2f}  "
          f"std=${window_prices.std():.2f}  min=${window_prices.min():.2f}  max=${window_prices.max():.2f}")
    print()

    # ----- Theoretical bounds -----
    print("Computing theoretical bounds …")
    floor_res = run_dispatch(
        daily_oracle_schedule(window_prices, SPEC, dt, cycles_per_day=1.0, tz=tz),
        window_prices, SPEC, dt,
    )
    floor_rev = float(floor_res["net_revenue"].sum())
    t0 = time.time()
    ceil_res = run_dispatch(
        perfect_foresight_schedule(window_prices, SPEC, dt,
                                   cycles_per_day_cap=1.0, tz=tz),
        window_prices, SPEC, dt,
    )
    ceil_rev = float(ceil_res["net_revenue"].sum())
    print(f"  floor   (natural-spread oracle): ${floor_rev:>15,.2f}")
    print(f"  ceiling (perfect-foresight LP):   ${ceil_rev:>15,.2f}  ({time.time() - t0:.0f}s)")
    print()

    # ----- Deployable baselines -----
    print("Deployable baselines …")
    results: dict[str, dict] = {}

    # 1. Do nothing.
    idle_sched = pd.Series(0.0, index=window_prices.index, name="grid_power_mw")
    idle_res = run_dispatch(idle_sched, window_prices, SPEC, dt)
    results["do_nothing"] = {"result": idle_res, "mae": None}

    # 2. Fixed time of day (3am charge, 5pm discharge — picked from val mode).
    ftod_sched = fixed_time_of_day_schedule(window_prices.index, dt, tz,
                                            charge_hour=3, discharge_hour=17)
    ftod_res = run_dispatch(ftod_sched, window_prices, SPEC, dt)
    results["fixed_time_3a_5p"] = {"result": ftod_res, "mae": None}

    # 3. Persistence forecast (same interval yesterday) + threshold dispatch.
    persistence_fcst = persistence_forecast_same_interval_yesterday(
        prices, intervals_per_day=INTERVALS_PER_DAY,
    ).loc[test_start:test_end]
    pers_sched = schedule_from_decision_prices(persistence_fcst, dt, tz, gated=False)
    pers_res = run_dispatch(pers_sched, window_prices, SPEC, dt)
    pers_mae = float(np.mean(np.abs(window_prices.values - persistence_fcst.fillna(0).values)))
    results["persistence"] = {"result": pers_res, "mae": pers_mae}

    # 4. Seasonal-naive (4-week median of same-DOW same-interval).
    seasonal_fcst = seasonal_naive_forecast(
        prices, lookback_weeks=4, intervals_per_day=INTERVALS_PER_DAY,
    ).loc[test_start:test_end]
    seas_sched = schedule_from_decision_prices(seasonal_fcst, dt, tz, gated=False)
    seas_res = run_dispatch(seas_sched, window_prices, SPEC, dt)
    seas_mae = float(np.mean(np.abs(window_prices.values - seasonal_fcst.fillna(0).values)))
    results["seasonal_naive_4w"] = {"result": seas_res, "mae": seas_mae}

    for name, r in results.items():
        rev = float(r["result"]["net_revenue"].sum())
        pc = pct_of_ceiling(rev, ceil_rev) if ceil_rev > 0 else 0
        print(f"  {name:25s} revenue=${rev:>14,.2f}  pct={pc:>6.2f}%")
    print()

    # ----- ML ensemble (locked spec) -----
    print(f"ML ensemble — {len(SEEDS)} seeds, q50 LGBM, 200 iters, "
          f"train_start=2019-01-01 …")
    seed_preds: dict[int, pd.Series] = {}
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
        sched = schedule_from_decision_prices(seed_preds[seed], dt, tz, gated=True)
        res = run_dispatch(sched, window_prices, SPEC, dt)
        rev = float(res["net_revenue"].sum())
        pc = pct_of_ceiling(rev, ceil_rev)
        mae_mask = seed_preds[seed].notna()
        mae = float(np.mean(np.abs(
            window_prices.loc[mae_mask].values - seed_preds[seed].loc[mae_mask].values
        )))
        print(f"  seed {seed:3d}: revenue=${rev:>14,.2f}  pct={pc:>5.2f}%  "
              f"MAE=${mae:.2f}  ({time.time() - t0:.0f}s)")
        results[f"ml_seed_{seed}"] = {"result": res, "mae": mae}

    # Ensemble = mean of seed predictions
    ensemble_preds = pd.concat(
        [s.rename(seed) for seed, s in seed_preds.items()], axis=1
    ).mean(axis=1)
    ens_sched = schedule_from_decision_prices(ensemble_preds, dt, tz, gated=True)
    ens_res = run_dispatch(ens_sched, window_prices, SPEC, dt)
    ens_rev = float(ens_res["net_revenue"].sum())
    ens_pc = pct_of_ceiling(ens_rev, ceil_rev)
    ens_mask = ensemble_preds.notna()
    ens_mae = float(np.mean(np.abs(
        window_prices.loc[ens_mask].values - ensemble_preds.loc[ens_mask].values
    )))
    results["ml_ensemble"] = {"result": ens_res, "mae": ens_mae}
    print(f"  ENSEMBLE: revenue=${ens_rev:>14,.2f}  pct={ens_pc:>5.2f}%  "
          f"MAE=${ens_mae:.2f}")
    print()

    # ----- Final table -----
    print("=" * 70)
    print("FINAL RESULTS — TEST WINDOW")
    print("=" * 70)
    print(f"{'Strategy':30s} {'Revenue':>16s} {'% ceiling':>10s} {'$/kW-yr':>10s} {'MAE':>9s}")
    print("-" * 70)

    test_years = (test_end - test_start) / pd.Timedelta(days=365.25)

    rows_out = []
    for name, r in [
        ("do_nothing", results["do_nothing"]),
        ("fixed_time_3a_5p", results["fixed_time_3a_5p"]),
        ("persistence", results["persistence"]),
        ("seasonal_naive_4w", results["seasonal_naive_4w"]),
        ("ml_ensemble", results["ml_ensemble"]),
        ("floor (oracle)", {"result": floor_res, "mae": None}),
        ("ceiling (perfect)", {"result": ceil_res, "mae": None}),
    ]:
        rev = float(r["result"]["net_revenue"].sum())
        pc = pct_of_ceiling(rev, ceil_rev) if ceil_rev > 0 else 0
        per_kw_yr = rev / SPEC.power_mw / 1000 / test_years
        mae_str = f"${r['mae']:.2f}" if r['mae'] is not None else "    —"
        print(f"{name:30s} ${rev:>14,.0f} {pc:>9.2f}% {per_kw_yr:>9.1f} {mae_str:>9s}")
        rows_out.append({
            "strategy": name, "revenue": rev, "pct_ceiling": pc,
            "per_kw_yr_dollars": per_kw_yr, "mae": r["mae"],
        })
    print()

    # Per-seed std
    seed_pcs = []
    for seed in SEEDS:
        rev = float(results[f"ml_seed_{seed}"]["result"]["net_revenue"].sum())
        seed_pcs.append(pct_of_ceiling(rev, ceil_rev))
    seed_pcs = np.array(seed_pcs)
    print(f"ML individual-seed: mean={seed_pcs.mean():.2f}% ± {seed_pcs.std():.2f} pp  "
          f"(range: {seed_pcs.min():.2f}–{seed_pcs.max():.2f}%)")
    print(f"ML ensemble:        {ens_pc:.2f}%  (Δ vs seed mean: "
          f"{ens_pc - seed_pcs.mean():+.2f} pp)")
    print()

    # ----- Regime breakdown for ensemble vs floor -----
    print("=== Regime breakdown — ensemble vs floor on test window ===")
    floor_rb = regime_breakdown(floor_res, window_prices, tz=tz)
    ml_rb = regime_breakdown(ens_res, window_prices, tz=tz)
    regimes = sorted(set(floor_rb.index) | set(ml_rb.index))
    print(f"{'regime':18s} {'n_days':>7s} {'floor_rev':>13s} {'ml_rev':>13s} "
          f"{'ml-floor':>13s} {'ml/floor':>9s}")
    for r in regimes:
        n = int(floor_rb.loc[r, "n_days"]) if r in floor_rb.index else 0
        fr = float(floor_rb.loc[r, "total_revenue"]) if r in floor_rb.index else 0
        mr = float(ml_rb.loc[r, "total_revenue"]) if r in ml_rb.index else 0
        delta = mr - fr
        ratio = (mr / fr) if fr != 0 else float("nan")
        print(f"{r:18s} {n:>7d} ${fr:>12,.0f} ${mr:>12,.0f} ${delta:>+12,.0f} "
              f"{ratio:>8.1%}")
    print()

    # Write results to disk for RESULTS.md to consume.
    out = pd.DataFrame(rows_out)
    out.to_csv(ROOT / "results" / "test_reveal.csv", index=False)
    print(f"Saved CSV: results/test_reveal.csv")


if __name__ == "__main__":
    main()
