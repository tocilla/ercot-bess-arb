"""HRRR (NOAA High-Resolution Rapid Refresh) weather-forecast fetcher.

Pulls from the public AWS S3 bucket `noaa-hrrr-bdp-pds` anonymously — no
API key, no AWS account. Archive since 2014. Uses `herbie` for S3 access
and GRIB2 parsing (which pulls in `cfgrib` + `eccodes`).

HRRR publishes hourly cycles (00..23 UTC). Each cycle produces a 18-hour
forecast at 3 km CONUS grid, in GRIB2, with a native Lambert-conformal
projection. For our use case we treat it as gridded 2D lat/lon.

What this module produces (per HRRR cycle):
    - Texas-averaged and max 2m temperature
    - Texas-averaged 10m wind speed
    (extensible — see `_summarize_cycle`)

Data volume considerations:
    - A single HRRR GRIB2 file is ~300–500 MB. We cannot store the
      full grid for 10+ years of hourly cycles locally.
    - Herbie can subset by variable at download time using its idx file,
      but it still fetches one file per variable per cycle.
    - Our strategy: pull the variable we need, immediately aggregate to
      a Texas summary, and persist only the summary numbers. This
      reduces data from ~500 MB/cycle to a few bytes/cycle.

Cache layout:
    data/raw/hrrr/summary_YYYYMMDD.parquet  (all cycles + fxx for that date)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.paths import DATA_RAW

logger = logging.getLogger(__name__)

CACHE_DIR = DATA_RAW / "hrrr"

# Texas bounding box in conventional coords; converted to 0-360 inside.
TX_LAT_MIN, TX_LAT_MAX = 25.0, 37.0
TX_LON_MIN, TX_LON_MAX = -107.0, -93.0


@dataclass
class HrrrSummary:
    cycle_utc: pd.Timestamp
    valid_utc: pd.Timestamp
    forecast_hour: int
    tx_mean_t2m_k: float
    tx_max_t2m_k: float
    tx_mean_wind10m_mps: float


def _cache_path(date_utc: datetime) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"summary_{date_utc:%Y%m%d}.parquet"


def _texas_mask(ds):
    """Boolean mask of HRRR cells inside the Texas bbox."""
    lat = ds["latitude"].values
    lon_w = (ds["longitude"].values + 180) % 360 - 180  # to -180..180
    return (lat >= TX_LAT_MIN) & (lat <= TX_LAT_MAX) & \
           (lon_w >= TX_LON_MIN) & (lon_w <= TX_LON_MAX)


def _summarize_cycle(cycle_utc: datetime, fxx: int) -> HrrrSummary | None:
    """Download one HRRR cycle + forecast hour, extract Texas summary.

    Returns None if the cycle isn't available (e.g. retrospective missing)."""
    from herbie import Herbie

    try:
        h_t = Herbie(cycle_utc.strftime("%Y-%m-%d %H:%M"), model="hrrr",
                     product="sfc", fxx=fxx, verbose=False)
        ds_t = h_t.xarray("TMP:2 m")
    except Exception as e:
        logger.warning("HRRR TMP fetch failed for %s F%02d: %s",
                       cycle_utc.isoformat(), fxx, e)
        return None

    try:
        h_w = Herbie(cycle_utc.strftime("%Y-%m-%d %H:%M"), model="hrrr",
                     product="sfc", fxx=fxx, verbose=False)
        ds_u = h_w.xarray("UGRD:10 m")
        ds_v = h_w.xarray("VGRD:10 m")
    except Exception as e:
        logger.warning("HRRR wind fetch failed for %s F%02d: %s",
                       cycle_utc.isoformat(), fxx, e)
        return None

    mask = _texas_mask(ds_t)
    t2m = ds_t["t2m"].where(mask)
    u10 = ds_u["u10"].where(mask)
    v10 = ds_v["v10"].where(mask)
    wind10 = np.sqrt(u10.values ** 2 + v10.values ** 2)

    valid = pd.Timestamp(cycle_utc).tz_localize("UTC") + pd.Timedelta(hours=fxx)
    return HrrrSummary(
        cycle_utc=pd.Timestamp(cycle_utc).tz_localize("UTC"),
        valid_utc=valid,
        forecast_hour=fxx,
        tx_mean_t2m_k=float(np.nanmean(t2m.values)),
        tx_max_t2m_k=float(np.nanmax(t2m.values)),
        tx_mean_wind10m_mps=float(np.nanmean(wind10)),
    )


def fetch_hrrr_day(
    date_utc: datetime,
    cycles: tuple[int, ...] = (0, 6, 12, 18),
    fxx_range: tuple[int, ...] = (3, 6, 9, 12, 15, 18),
    refresh: bool = False,
) -> pd.DataFrame:
    """Fetch Texas-averaged HRRR summaries for one UTC date.

    Args:
        date_utc: date (UTC-midnight) to fetch cycles for.
        cycles: which UTC hours to pull (defaults to 4 cycles/day — 00, 06,
            12, 18 — which gives full CONUS coverage at minimal cost).
        fxx_range: which forecast hours to pull from each cycle.
        refresh: redownload even if cached.

    Returns:
        DataFrame with columns:
            cycle_utc, valid_utc, forecast_hour,
            tx_mean_t2m_k, tx_max_t2m_k, tx_mean_wind10m_mps
    """
    path = _cache_path(date_utc)
    if path.exists() and not refresh:
        logger.info("Loading %s from cache", path)
        return pd.read_parquet(path)

    rows: list[dict] = []
    for cyc_hr in cycles:
        cyc = date_utc.replace(hour=cyc_hr, minute=0, second=0, microsecond=0,
                               tzinfo=None)
        for fxx in fxx_range:
            s = _summarize_cycle(cyc, fxx)
            if s is None:
                continue
            rows.append(
                {
                    "cycle_utc": s.cycle_utc,
                    "valid_utc": s.valid_utc,
                    "forecast_hour": s.forecast_hour,
                    "tx_mean_t2m_k": s.tx_mean_t2m_k,
                    "tx_max_t2m_k": s.tx_max_t2m_k,
                    "tx_mean_wind10m_mps": s.tx_mean_wind10m_mps,
                }
            )

    if not rows:
        logger.warning("No HRRR data fetched for %s", date_utc.date())
        return pd.DataFrame()

    df = pd.DataFrame(rows).sort_values(["cycle_utc", "forecast_hour"]).reset_index(drop=True)
    df.to_parquet(path, index=False)
    logger.info("Cached %d rows to %s", len(df), path)
    return df
