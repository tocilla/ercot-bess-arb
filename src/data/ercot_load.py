"""ERCOT historical load (actuals) fetcher with parquet caching.

Uses `gridstatus.Ercot.get_hourly_load_post_settlements`, which downloads
from https://www.ercot.com/gridinfo/load/load_hist archive. Hourly data,
settled actuals, 2011+ by ERCOT weather zone plus system-wide.

Note: these are ACTUALS. For a fair walk-forward, use them only with a
lag (e.g. yesterday's load at same hour as a proxy for today's forecast).
Historical as-of forecasts are NOT available via gridstatus — documenting
this limitation in DATA.md.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.paths import DATA_RAW

logger = logging.getLogger(__name__)

CACHE_DIR = DATA_RAW / "ercot_load"
WEATHER_ZONES = ["Coast", "East", "Far West", "North", "North Central",
                 "South", "South Central", "West"]


def _cache_path(year: int) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"load_{year}.parquet"


def get_load_year(year: int, refresh: bool = False) -> pd.DataFrame:
    """Return hourly ERCOT load for `year` with UTC index and zone columns."""
    path = _cache_path(year)
    if path.exists() and not refresh:
        logger.info("Loading %s from cache", path)
        return pd.read_parquet(path)

    import gridstatus  # lazy import

    logger.info("Fetching ERCOT load for %d", year)
    iso = gridstatus.Ercot()
    raw = iso.get_hourly_load_post_settlements(
        date=f"{year}-01-01", end=f"{year}-12-31", verbose=False,
    )

    out = raw.copy()
    out["timestamp_utc"] = pd.to_datetime(out["Interval Start"], utc=True)
    out = out.drop(columns=["Interval Start", "Interval End"])
    # Rename zone columns to snake_case, keep system-wide as `ercot_mw`.
    rename = {z: f"load_{z.lower().replace(' ', '_')}_mw" for z in WEATHER_ZONES}
    rename["ERCOT"] = "ercot_mw"
    out = out.rename(columns=rename)
    cols = ["timestamp_utc", "ercot_mw"] + [v for v in rename.values() if v != "ercot_mw"]
    out = out[cols].sort_values("timestamp_utc").reset_index(drop=True)

    out.to_parquet(path, index=False)
    logger.info("Cached %d rows to %s", len(out), path)
    return out


def get_load_series(start_year: int, end_year: int, refresh: bool = False) -> pd.DataFrame:
    """Concat multiple years of hourly load. Returns a DataFrame indexed by
    tz-aware UTC timestamp."""
    frames = [get_load_year(y, refresh=refresh) for y in range(start_year, end_year + 1)]
    df = pd.concat(frames, ignore_index=True).sort_values("timestamp_utc")
    df = df.drop_duplicates("timestamp_utc", keep="first")
    return df.set_index("timestamp_utc")
