"""Fetch ERCOT RTM Settlement Point Prices for a year range and cache to disk.

Usage:
    python scripts/fetch_ercot_rtm.py --start 2015 --end 2024
    python scripts/fetch_ercot_rtm.py --start 2023 --end 2023 --refresh
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.ercot import ERCOT_DATA_START_YEAR, get_rtm_spp_year  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", type=int, required=True,
                    help=f"start year (>= {ERCOT_DATA_START_YEAR})")
    ap.add_argument("--end", type=int, required=True, help="end year (inclusive)")
    ap.add_argument("--refresh", action="store_true", help="ignore cache and re-fetch")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    for year in range(args.start, args.end + 1):
        df = get_rtm_spp_year(year, refresh=args.refresh)
        print(f"  {year}: {len(df):,} rows, "
              f"{df['location'].nunique()} locations, "
              f"span {df['timestamp_utc'].min()} → {df['timestamp_utc'].max()}")


if __name__ == "__main__":
    main()
