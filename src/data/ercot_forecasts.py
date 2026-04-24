"""ERCOT Public API wind / solar / load forecast backfill.

For each day in range, fetch one document published at roughly 06:00 UTC
(late-evening CT the day before) — giving us a vintaged day-ahead
forecast. This is the cheap-but-useful scope: one forecast per day per
endpoint, which is enough to test whether these features carry dispatch
value before committing to a finer-grained (hourly) backfill.

Cache layout:
    data/raw/ercot_forecasts/<endpoint_key>/YYYYMMDD.parquet
"""

from __future__ import annotations

import io
import logging
import time
import zipfile
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from src.data.ercot_api import (
    ENDPOINTS, download_archive, list_archives,
)
from src.paths import DATA_RAW

logger = logging.getLogger(__name__)

CACHE_ROOT = DATA_RAW / "ercot_forecasts"


def _cache_dir(endpoint_key: str) -> Path:
    d = CACHE_ROOT / endpoint_key
    d.mkdir(parents=True, exist_ok=True)
    return d


def fetch_daily_forecast(
    endpoint_key: str,
    day: datetime,
    target_publish_hour_utc: int = 6,
    refresh: bool = False,
    pause_seconds: float = 0.5,
) -> pd.DataFrame:
    """Fetch one forecast document published around `target_publish_hour_utc`
    on the given UTC day and return its contents as a DataFrame.

    If no doc was published exactly at that hour, picks the closest later
    publish on the same day.
    """
    spec = ENDPOINTS[endpoint_key]
    cache = _cache_dir(endpoint_key) / f"{day:%Y%m%d}.parquet"
    if cache.exists() and not refresh:
        return pd.read_parquet(cache)

    window_start = datetime(day.year, day.month, day.day,
                            target_publish_hour_utc, 0, 0)
    window_end = window_start + timedelta(hours=3)
    archives = list_archives(
        spec.report_id,
        post_datetime_from=pd.Timestamp(window_start),
        post_datetime_to=pd.Timestamp(window_end),
    )
    if archives.empty:
        # Try a broader window on the same day.
        archives = list_archives(
            spec.report_id,
            post_datetime_from=pd.Timestamp(day.replace(hour=0, minute=0)),
            post_datetime_to=pd.Timestamp(day.replace(hour=23, minute=59)),
        )
    if archives.empty:
        logger.warning("No %s doc found on %s", endpoint_key, day.date())
        return pd.DataFrame()

    # Choose the doc whose post time is closest to target hour but >= target.
    target_ts = pd.Timestamp(window_start)
    archives = archives.sort_values("post_datetime")
    after = archives[archives["post_datetime"] >= target_ts]
    chosen = after.iloc[0] if not after.empty else archives.iloc[0]
    doc_id = int(chosen["doc_id"])
    post_time = chosen["post_datetime"]

    if pause_seconds > 0:
        time.sleep(pause_seconds)

    data = download_archive(spec.report_id, doc_id)

    try:
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            names = [n for n in z.namelist() if n.lower().endswith(".csv")]
            if not names:
                logger.warning("No CSV in doc %d", doc_id)
                return pd.DataFrame()
            with z.open(names[0]) as f:
                df = pd.read_csv(f)
    except zipfile.BadZipFile:
        df = pd.read_csv(io.BytesIO(data))

    df["post_datetime_utc"] = pd.to_datetime(post_time, utc=True)
    df["doc_id"] = doc_id
    df.to_parquet(cache, index=False)
    return df


def backfill_daily_forecasts(
    endpoint_key: str,
    start_date: datetime,
    end_date: datetime,
    target_publish_hour_utc: int = 6,
    pause_seconds: float = 0.5,
    on_progress=None,
) -> None:
    """Backfill one doc per day for a date range. Sequential — sacrifices
    speed for rate-limit safety. Prints periodic progress."""
    total_days = (end_date - start_date).days + 1
    done = 0
    d = start_date
    while d <= end_date:
        cache = _cache_dir(endpoint_key) / f"{d:%Y%m%d}.parquet"
        if not cache.exists():
            try:
                fetch_daily_forecast(
                    endpoint_key, d,
                    target_publish_hour_utc=target_publish_hour_utc,
                    pause_seconds=pause_seconds,
                )
            except Exception as e:  # noqa: BLE001
                logger.warning("Skip %s %s: %s", endpoint_key, d.date(), e)
        done += 1
        if on_progress:
            on_progress(done, total_days)
        d = d + timedelta(days=1)


def load_forecasts(endpoint_key: str) -> pd.DataFrame:
    """Load all cached daily forecasts for an endpoint into one DataFrame."""
    files = sorted(_cache_dir(endpoint_key).glob("*.parquet"))
    if not files:
        return pd.DataFrame()
    frames = [pd.read_parquet(f) for f in files]
    return pd.concat(frames, ignore_index=True)
