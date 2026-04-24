"""Run the natural-spread oracle baseline on real ERCOT HB_NORTH RTM SPPs.

Prerequisite: run `scripts/fetch_ercot_rtm.py --start <Y1> --end <Y2>` first
to populate the cache.

Usage:
    python scripts/run_oracle_real.py --start 2015 --end 2024
    python scripts/run_oracle_real.py --start 2023 --end 2023 --location HB_HOUSTON
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.baselines import daily_oracle_schedule  # noqa: E402
from src.battery import BatterySpec  # noqa: E402
from src.data.ercot import INTERVAL_MINUTES, get_rtm_spp_series  # noqa: E402
from src.dispatch import run_dispatch  # noqa: E402
from src.metrics import compare  # noqa: E402


DEFAULT_SPEC = BatterySpec(
    power_mw=100.0,
    capacity_mwh=200.0,        # 2-hour battery
    roundtrip_eff=0.85,
    soc_min_frac=0.05,
    soc_max_frac=0.95,
    initial_soc_frac=0.5,
    degradation_cost_per_mwh=2.0,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", type=int, required=True, help="start year")
    ap.add_argument("--end", type=int, required=True, help="end year (inclusive)")
    ap.add_argument("--location", default="HB_NORTH")
    ap.add_argument("--tz", default="US/Central",
                    help="local timezone for daily grouping")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    print(f"Loading {args.location} RTM SPP {args.start}–{args.end} …")
    prices = get_rtm_spp_series(args.location, args.start, args.end)
    print(f"  rows: {len(prices):,}, "
          f"span: {prices.index.min()} → {prices.index.max()}")
    print(f"  price: mean=${prices.mean():.2f}, median=${prices.median():.2f}, "
          f"std=${prices.std():.2f}, min=${prices.min():.2f}, max=${prices.max():.2f}")
    print(f"  negative-price intervals: {(prices < 0).mean() * 100:.2f}%, "
          f"scarcity (>$500) intervals: {(prices > 500).mean() * 100:.2f}%")

    interval_hours = INTERVAL_MINUTES / 60.0

    # Idle (do-nothing) reference
    idle_sched = pd.Series(0.0, index=prices.index, name="grid_power_mw")
    idle_result = run_dispatch(idle_sched, prices, DEFAULT_SPEC, interval_hours)

    # Natural-spread oracle (uses realized prices — this is the floor)
    oracle_sched = daily_oracle_schedule(
        prices, DEFAULT_SPEC, interval_hours=interval_hours,
        cycles_per_day=1.0, tz=args.tz,
    )
    oracle_result = run_dispatch(oracle_sched, prices, DEFAULT_SPEC, interval_hours)

    summary = compare({"idle": idle_result, "natural_spread_oracle": oracle_result}, tz=args.tz)
    print("\n=== Summary ===")
    print(summary.to_string(float_format=lambda x: f"{x:,.2f}"))

    # Per-year breakdown for regime awareness
    per_year = (
        oracle_result.groupby(oracle_result.index.tz_convert(args.tz).year)["net_revenue"]
        .agg(["sum", "mean", "std", "min", "max"])
    )
    per_year.columns = ["total_revenue", "mean_per_interval",
                        "std_per_interval", "min_interval", "max_interval"]
    print("\n=== Oracle revenue by year ===")
    print(per_year.to_string(float_format=lambda x: f"{x:,.2f}"))


if __name__ == "__main__":
    main()
