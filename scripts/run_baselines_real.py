"""Run natural-spread baseline + perfect-foresight LP ceiling on real
ERCOT RTM SPPs and print a comparison, with regime-stratified revenue.

Prerequisite: cache populated via `scripts/fetch_ercot_rtm.py`.

Usage:
    python scripts/run_baselines_real.py --start 2023 --end 2023
    python scripts/run_baselines_real.py --start 2011 --end 2024 --location HB_NORTH
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

from src.baselines import daily_oracle_schedule  # noqa: E402
from src.battery import BatterySpec  # noqa: E402
from src.data.ercot import INTERVAL_MINUTES, get_rtm_spp_series  # noqa: E402
from src.dispatch import run_dispatch  # noqa: E402
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

    interval_hours = INTERVAL_MINUTES / 60.0

    # Natural-spread baseline (floor)
    t0 = time.time()
    oracle_sched = daily_oracle_schedule(
        prices, DEFAULT_SPEC, interval_hours=interval_hours,
        cycles_per_day=1.0, tz=args.tz,
    )
    oracle_result = run_dispatch(oracle_sched, prices, DEFAULT_SPEC, interval_hours)
    print(f"  natural-spread baseline done ({time.time() - t0:.1f}s)")

    # Perfect-foresight ceiling
    t0 = time.time()
    ceiling_sched = perfect_foresight_schedule(
        prices, DEFAULT_SPEC, interval_hours=interval_hours,
        cycles_per_day_cap=1.0, tz=args.tz, solver=args.solver,
    )
    ceiling_result = run_dispatch(ceiling_sched, prices, DEFAULT_SPEC, interval_hours)
    print(f"  LP ceiling done ({time.time() - t0:.1f}s)")

    # Summary side-by-side
    runs = {
        "natural_spread_floor": oracle_result,
        "perfect_foresight_ceiling": ceiling_result,
    }
    summary = compare(runs, tz=args.tz)
    print("\n=== Summary ===")
    print(summary.to_string(float_format=lambda x: f"{x:,.2f}"))

    # Ceiling capture
    floor_rev = oracle_result["net_revenue"].sum()
    ceiling_rev = ceiling_result["net_revenue"].sum()
    print(f"\nFloor captures {pct_of_ceiling(floor_rev, ceiling_rev):.1f}% of ceiling.")
    print(f"Headroom (ceiling − floor) = ${ceiling_rev - floor_rev:,.2f}")

    # Per-year
    for name, result in runs.items():
        per_year = (
            result.groupby(result.index.tz_convert(args.tz).year)["net_revenue"]
            .sum()
        )
        print(f"\n=== {name} — revenue by year ===")
        print(per_year.to_string(float_format=lambda x: f"{x:,.2f}"))

    # Regime breakdown — where is the money?
    for name, result in runs.items():
        rb = regime_breakdown(result, prices, tz=args.tz,
                              scarcity_threshold=args.scarcity_threshold)
        print(f"\n=== {name} — regime breakdown ===")
        print(rb.to_string(float_format=lambda x: f"{x:,.2f}"))


if __name__ == "__main__":
    main()
