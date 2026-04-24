"""FRED time-series fetcher.

No API key required — uses FRED's public CSV download endpoint:
    https://fred.stlouisfed.org/graph/fredgraph.csv?id=<series>

Cached to `data/raw/fred/<series>.parquet`. Values re-fetched on `refresh=True`.

Typical series we care about:
    DHHNGSP — Henry Hub natural gas spot price ($/MMBtu, daily)
    DCOILWTICO — WTI crude spot ($/bbl, daily)
    MCOILBRENTEU — Brent crude ($/bbl, monthly)
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import requests

from src.paths import DATA_RAW

logger = logging.getLogger(__name__)

CACHE_DIR = DATA_RAW / "fred"
_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"


def _cache_path(series_id: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{series_id}.parquet"


def get_fred_series(series_id: str, refresh: bool = False) -> pd.Series:
    """Fetch a FRED series as a daily-indexed pd.Series of floats.

    Missing observations (FRED uses '.') are dropped. Index is naive-UTC
    dates — no time-of-day component.
    """
    path = _cache_path(series_id)
    if path.exists() and not refresh:
        logger.info("Loading %s from cache", path)
        df = pd.read_parquet(path)
        return df["value"].rename(series_id)

    logger.info("Fetching FRED series %s", series_id)
    r = requests.get(_CSV_URL, params={"id": series_id}, timeout=30)
    r.raise_for_status()

    from io import StringIO
    df = pd.read_csv(StringIO(r.text))
    # FRED CSV layout: columns are "observation_date" (or "DATE") and <series_id>.
    date_col = "observation_date" if "observation_date" in df.columns else "DATE"
    df = df[[date_col, series_id]].copy()
    df.columns = ["date", "value"]
    # Drop missing observations ('.' or NaN).
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value"])
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    df.to_parquet(path, index=False)
    logger.info("Cached %d rows to %s", len(df), path)

    return df.set_index("date")["value"].rename(series_id)
