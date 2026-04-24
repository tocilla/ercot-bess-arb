"""End-to-end smoke test for all data sources.

Each source is exercised just enough to prove the path works. If any
source is misconfigured (missing key, network blocked, etc.), the test
prints a clear diagnostic and continues with the others.

Usage:
    python scripts/smoke_data_sources.py
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    print("=" * 60)
    print("Data-source smoke tests")
    print("=" * 60)

    # --- ERCOT historical prices + load (anonymous; should always work)
    print("\n[1] ERCOT RTM SPP + historical load (anonymous)")
    try:
        from src.data.ercot import get_rtm_spp_series
        from src.data.ercot_load import get_load_series
        s = get_rtm_spp_series("HB_NORTH", 2024, 2024)
        print(f"  prices OK — {len(s):,} rows, mean ${s.mean():.2f}")
        ld = get_load_series(2024, 2024)
        print(f"  load OK — {len(ld):,} rows, mean {ld['ercot_mw'].mean():.0f} MW")
    except Exception as e:
        print(f"  FAIL — {e}")

    # --- FRED (no key)
    print("\n[2] FRED — Henry Hub natural gas (no key)")
    try:
        from src.data.fred import get_fred_series
        s = get_fred_series("DHHNGSP")
        print(f"  OK — {len(s):,} daily obs, most recent ${s.iloc[-1]:.2f}")
    except Exception as e:
        print(f"  FAIL — {e}")

    # --- EIA-930 (needs EIA_API_KEY)
    print("\n[3] EIA-930 region-data (needs EIA_API_KEY)")
    try:
        from src.data.eia930 import get_region_data_year
        df = get_region_data_year(2024, respondent="ERCO")
        types = sorted(df["type"].unique())
        print(f"  OK — {len(df):,} rows, types {types}")
    except Exception as e:
        print(f"  FAIL — {e}")

    # --- HRRR via herbie (no key, S3 anonymous)
    print("\n[4] HRRR Texas summary (anonymous S3)")
    try:
        from src.data.hrrr import fetch_hrrr_day
        df = fetch_hrrr_day(datetime(2024, 6, 15), cycles=(12,), fxx_range=(3,))
        if not df.empty:
            row = df.iloc[0]
            tk = row["tx_mean_t2m_k"]
            print(f"  OK — cycle {row['cycle_utc']} F{row['forecast_hour']:02d}: "
                  f"Texas mean T = {tk:.1f} K ({(tk - 273.15) * 9 / 5 + 32:.1f} F)")
        else:
            print(f"  FAIL — empty dataframe")
    except Exception as e:
        print(f"  FAIL — {e}")

    # --- ERCOT Public API (needs username + password)
    print("\n[5] ERCOT Public API (needs username+password in .env)")
    try:
        from src.data.ercot_api import smoke_test_auth
        smoke_test_auth()
    except Exception as e:
        print(f"  expected-fail (or misconfigured): {str(e)[:150]}")


if __name__ == "__main__":
    main()
