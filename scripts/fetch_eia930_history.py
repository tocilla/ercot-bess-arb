"""Backfill EIA-930 ERCOT region + fuel-mix data for the available years.

EIA-930 for ERCOT (respondent=ERCO) has data starting 2019-01. Region
data (D/DF/NG/TI) is available all years. Fuel-mix data started later
(~2018-07 globally, but ERCOT availability is also 2019+).

Usage:
    python scripts/fetch_eia930_history.py --start 2019 --end 2024
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.eia930 import get_region_data_year, get_fuel_type_year  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", type=int, required=True)
    ap.add_argument("--end", type=int, required=True)
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--skip-fuel", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    for year in range(args.start, args.end + 1):
        print(f"--- {year} region-data ---")
        df = get_region_data_year(year, respondent="ERCO", refresh=args.refresh)
        if df.empty:
            print(f"  no data for {year}")
        else:
            print(f"  {year}: {len(df):,} rows, types {sorted(df['type'].unique().tolist())}")

        if not args.skip_fuel:
            print(f"--- {year} fuel-type ---")
            df = get_fuel_type_year(year, respondent="ERCO", refresh=args.refresh)
            if df.empty:
                print(f"  no fuel data for {year}")
            else:
                print(f"  {year}: {len(df):,} rows, fuels {sorted(df['fueltype'].unique().tolist())}")


if __name__ == "__main__":
    main()
