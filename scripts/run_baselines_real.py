"""Compare all phase-1 dispatch strategies on real ERCOT RTM SPPs.

Strategies (METHODOLOGY §5.3 revenue-attribution terms in parentheses):

    1. natural_spread_floor       — oracle timing, blind cycling (floor)
    2. natural_spread_gated       — oracle timing + skip unprofitable days
    3. persistence                — forecast = yesterday; threshold dispatch
    4. seasonal_naive             — forecast = 4-week same-DOW median; threshold
    5. perfect_foresight_ceiling  — LP over realized prices (ceiling)

Prerequisite: cache populated via `scripts/fetch_ercot_rtm.py`.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.baselines import (  # noqa: E402
    daily_oracle_schedule,
    daily_spread_gated_schedule,
)
from src.battery import BatterySpec  # noqa: E402
from src.data.ercot import INTERVAL_MINUTES, get_rtm_spp_series  # noqa: E402
from src.dispatch import run_dispatch  # noqa: E402
from src.forecasters import (  # noqa: E402
    persistence_forecast_same_interval_yesterday,
    seasonal_naive_forecast,
)
from src.metrics import compare, pct_of_ceiling, regime_breakdown  # noqa: E402
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

INTERVALS_PER_DAY = 24 * 60 // INTERVAL_MINUTES  # 96 for 15-min


def _safe_schedule(forecast: pd.Series, prices: pd.Series, spec, interval_hours, tz):
    """Build a schedule from a forecast; fall back to idle where forecast is NaN.

    NaNs fall to the end of a stable sort in pandas — but to be safe we fill
    NaN forecasts with the price mean so the ranker doesn't pick them as
    cheapest/most-expensive. Days entirely made of NaN produce idle.
    """
    filled = forecast.fillna(forecast.mean(skipna=True))
    sched = daily_oracle_schedule(filled, spec, interval_hours, cycles_per_day=1.0, tz=tz)
    # Any day where forecast was entirely NaN → zero out.
    local_idx = prices.index.tz_convert(tz) if tz else prices.index
    day_key = pd.Series(local_idx.date, index=prices.index)
    for _, idx in forecast.groupby(day_key).groups.items():
        if forecast.loc[idx].isna().all():
            sched.loc[idx] = 0.0
    return sched


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", type=int, required=True)
    ap.add_argument("--end", type=int, required=True)
    ap.add_argument("--location", default="HB_NORTH")
    ap.add_argument("--tz", default="US/Central")
    ap.add_argument("--scarcity-threshold", type=float, default=500.0)
    ap.add_argument("--solver", default="HIGHS")
    args = ap.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s: %(message)s")

    print(f"Loading {args.location} RTM SPP {args.start}–{args.end} …")
    prices = get_rtm_spp_series(args.location, args.start, args.end)
    print(f"  rows: {len(prices):,}, "
          f"span: {prices.index.min()} → {prices.index.max()}")
    print(f"  price: mean=${prices.mean():.2f}, median=${prices.median():.2f}, "
          f"std=${prices.std():.2f}, min=${prices.min():.2f}, max=${prices.max():.2f}")

    dt = INTERVAL_MINUTES / 60.0
    spec = DEFAULT_SPEC

    runs: dict[str, pd.DataFrame] = {}

    t0 = time.time()
    sched = daily_oracle_schedule(prices, spec, dt, cycles_per_day=1.0, tz=args.tz)
    runs["1_natural_spread_floor"] = run_dispatch(sched, prices, spec, dt)
    print(f"  [1] floor done ({time.time() - t0:.1f}s)")

    t0 = time.time()
    sched = daily_spread_gated_schedule(prices, prices, spec, dt,
                                        cycles_per_day=1.0, tz=args.tz)
    runs["2_natural_spread_gated"] = run_dispatch(sched, prices, spec, dt)
    print(f"  [2] gated floor done ({time.time() - t0:.1f}s)")

    t0 = time.time()
    forecast = persistence_forecast_same_interval_yesterday(prices, INTERVALS_PER_DAY)
    sched = _safe_schedule(forecast, prices, spec, dt, args.tz)
    runs["3_persistence"] = run_dispatch(sched, prices, spec, dt)
    print(f"  [3] persistence done ({time.time() - t0:.1f}s)")

    t0 = time.time()
    forecast = seasonal_naive_forecast(prices, lookback_weeks=4,
                                       intervals_per_day=INTERVALS_PER_DAY)
    sched = _safe_schedule(forecast, prices, spec, dt, args.tz)
    runs["4_seasonal_naive_4w"] = run_dispatch(sched, prices, spec, dt)
    print(f"  [4] seasonal-naive done ({time.time() - t0:.1f}s)")

    t0 = time.time()
    sched = perfect_foresight_schedule(prices, spec, dt,
                                       cycles_per_day_cap=1.0, tz=args.tz,
                                       solver=args.solver)
    runs["5_perfect_foresight_ceiling"] = run_dispatch(sched, prices, spec, dt)
    print(f"  [5] ceiling done ({time.time() - t0:.1f}s)")

    summary = compare(runs, tz=args.tz)
    print("\n=== Summary ===")
    print(summary.to_string(float_format=lambda x: f"{x:,.2f}"))

    ceiling_rev = runs["5_perfect_foresight_ceiling"]["net_revenue"].sum()
    print("\n=== % of ceiling captured ===")
    for name, result in runs.items():
        rev = result["net_revenue"].sum()
        print(f"  {name:40s}  {pct_of_ceiling(rev, ceiling_rev):5.1f}%   "
              f"${rev:>15,.2f}   (lift over floor: ${rev - runs['1_natural_spread_floor']['net_revenue'].sum():>+14,.2f})")

    for name, result in runs.items():
        rb = regime_breakdown(result, prices, tz=args.tz,
                              scarcity_threshold=args.scarcity_threshold)
        print(f"\n=== {name} — regime breakdown ===")
        print(rb.to_string(float_format=lambda x: f"{x:,.2f}"))


if __name__ == "__main__":
    main()
