"""ERCOT Real-Time Market Settlement Point Price fetcher.

Wraps `gridstatus.Ercot.get_rtm_spp` with:
    - disk caching as parquet in data/raw/ercot/
    - timezone normalization to UTC on read
    - a convenience function to get a single location's price series
      across a year range

ERCOT RTM SPPs are 15-minute settlement-interval prices at hubs and load
zones. Data begins 2011-01-01. Source documentation:
https://www.ercot.com/mp/data-products/data-product-details?id=NP6-785-ER

Note on semantics: RTM SPP is the *settled* price that determines what a
battery operator is paid per MWh of grid-side energy in/out during a 15-min
interval. This is distinct from 5-min SCED LMPs used for dispatch signals.
For revenue modeling, RTM SPP is the correct target.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.paths import DATA_RAW

logger = logging.getLogger(__name__)

CACHE_DIR = DATA_RAW / "ercot"
INTERVAL_MINUTES = 15
ERCOT_DATA_START_YEAR = 2011

HUBS = ["HB_NORTH", "HB_HOUSTON", "HB_SOUTH", "HB_WEST", "HB_PAN", "HB_BUSAVG", "HB_HUBAVG"]
LOAD_ZONES = ["LZ_NORTH", "LZ_HOUSTON", "LZ_SOUTH", "LZ_WEST",
              "LZ_AEN", "LZ_CPS", "LZ_LCRA", "LZ_RAYBN"]


def _cache_path(year: int) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"rtm_spp_{year}.parquet"


def get_rtm_spp_year(year: int, refresh: bool = False) -> pd.DataFrame:
    """Fetch one year of ERCOT RTM SPPs for all hubs + load zones.

    Cached as parquet. Returns a DataFrame with a tz-aware UTC `timestamp_utc`
    index, a `location` column, and the `spp` price in $/MWh.
    """
    if year < ERCOT_DATA_START_YEAR:
        raise ValueError(f"ERCOT RTM SPP archive starts in {ERCOT_DATA_START_YEAR}")

    path = _cache_path(year)
    if path.exists() and not refresh:
        logger.info("Loading %s from cache", path)
        return pd.read_parquet(path)

    import gridstatus  # heavy dep — import lazily

    logger.info("Fetching ERCOT RTM SPP for %d from public archive", year)
    iso = gridstatus.Ercot()
    raw = iso.get_rtm_spp(year=year, verbose=False)

    # Normalize schema: UTC index, tidy columns.
    out = raw[["Interval Start", "Location", "SPP"]].copy()
    out.columns = ["interval_start_local", "location", "spp"]
    out["timestamp_utc"] = pd.to_datetime(out["interval_start_local"], utc=True)
    out = out[["timestamp_utc", "location", "spp"]].sort_values(
        ["location", "timestamp_utc"]
    ).reset_index(drop=True)
    out["location"] = out["location"].astype("string")
    out["spp"] = out["spp"].astype("float64")

    out.to_parquet(path, index=False)
    logger.info("Cached %d rows to %s", len(out), path)
    return out


def get_rtm_spp_series(
    location: str,
    start_year: int,
    end_year: int,
    refresh: bool = False,
) -> pd.Series:
    """Return a single-location RTM SPP series indexed by tz-aware UTC timestamps.

    Example:
        >>> s = get_rtm_spp_series("HB_NORTH", 2015, 2024)
        >>> s.head()
    """
    if end_year < start_year:
        raise ValueError("end_year must be >= start_year")

    frames = []
    for year in range(start_year, end_year + 1):
        df = get_rtm_spp_year(year, refresh=refresh)
        sub = df.loc[df["location"] == location, ["timestamp_utc", "spp"]]
        if sub.empty:
            raise ValueError(f"No rows for location={location!r} in {year}")
        frames.append(sub)

    all_df = pd.concat(frames, ignore_index=True).sort_values("timestamp_utc")
    s = all_df.set_index("timestamp_utc")["spp"].rename(f"rtm_spp_{location}")
    # Deduplicate any DST-duplicated interval starts (ERCOT local tz quirks).
    s = s[~s.index.duplicated(keep="first")]
    return s
