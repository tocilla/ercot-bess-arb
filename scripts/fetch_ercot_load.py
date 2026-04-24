"""Fetch ERCOT historical hourly load (by weather zone + system-wide).

Usage:
    python scripts/fetch_ercot_load.py --start 2011 --end 2024
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.ercot_load import get_load_year  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", type=int, required=True)
    ap.add_argument("--end", type=int, required=True)
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    for year in range(args.start, args.end + 1):
        df = get_load_year(year, refresh=args.refresh)
        print(f"  {year}: {len(df):,} rows, "
              f"span {df['timestamp_utc'].min()} → {df['timestamp_utc'].max()}")


if __name__ == "__main__":
    main()
