"""EIA-930 balancing-authority fetcher.

Uses EIA Open Data API v2. Endpoint:
    https://api.eia.gov/v2/electricity/rto/region-data/data/

Scope for this project:
    - respondent = ERCO (ERCOT)
    - types = D (hourly demand actual), DF (day-ahead demand forecast),
              NG (net generation), TI (total interchange)
    - coverage 2015-07-01 onward

Historical fuel-mix (coal/gas/nuclear/wind/solar/hydro/other) is at a
separate endpoint:
    https://api.eia.gov/v2/electricity/rto/fuel-type-data/data/

Cached as parquet per (endpoint, respondent, year) in data/raw/eia930/.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import pandas as pd
import requests

from src.config_env import get_env_var
from src.paths import DATA_RAW

logger = logging.getLogger(__name__)

CACHE_DIR = DATA_RAW / "eia930"
_BASE = "https://api.eia.gov/v2"
_ROW_LIMIT = 5000  # EIA API server-side cap


def _cache_path(endpoint: str, respondent: str, year: int) -> Path:
    safe = endpoint.replace("/", "_")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{safe}_{respondent}_{year}.parquet"


def _fetch_chunk(
    route: str,
    api_key: str,
    respondent: str,
    start: str,
    end: str,
    types: list[str] | None = None,
    offset: int = 0,
    facet_type_key: str = "type",
    retries: int = 4,
) -> list[dict]:
    params = [
        ("api_key", api_key),
        ("frequency", "hourly"),
        ("data[]", "value"),
        ("facets[respondent][]", respondent),
        ("start", start),
        ("end", end),
        ("sort[0][column]", "period"),
        ("sort[0][direction]", "asc"),
        ("offset", str(offset)),
        ("length", str(_ROW_LIMIT)),
    ]
    if types:
        for t in types:
            params.append((f"facets[{facet_type_key}][]", t))
    url = f"{_BASE}/{route.lstrip('/')}/data/"
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=60)
            # 502/503/504 are transient gateway / upstream errors — retry.
            if r.status_code in (502, 503, 504):
                raise requests.HTTPError(f"{r.status_code} transient", response=r)
            r.raise_for_status()
            payload = r.json()
            return payload.get("response", {}).get("data", [])
        except (requests.HTTPError, requests.ConnectionError, requests.Timeout) as e:
            last_err = e
            wait = 2 ** attempt
            logger.warning("EIA fetch attempt %d failed (%s), retrying in %ds",
                           attempt + 1, e, wait)
            time.sleep(wait)
    assert last_err is not None
    raise last_err


def _fetch_range(
    route: str,
    respondent: str,
    start: str,
    end: str,
    types: list[str] | None = None,
    facet_type_key: str = "type",
) -> pd.DataFrame:
    """Fetch an arbitrary range in chunks of `_ROW_LIMIT`."""
    api_key = get_env_var("EIA_API_KEY")
    all_rows: list[dict] = []
    offset = 0
    while True:
        rows = _fetch_chunk(route, api_key, respondent, start, end, types,
                            offset, facet_type_key)
        if not rows:
            break
        all_rows.extend(rows)
        if len(rows) < _ROW_LIMIT:
            break
        offset += _ROW_LIMIT
    if not all_rows:
        return pd.DataFrame()
    return pd.DataFrame(all_rows)


def _fetch_year_by_month(
    route: str,
    respondent: str,
    year: int,
    types: list[str] | None = None,
    facet_type_key: str = "type",
) -> pd.DataFrame:
    """Fetch a full year, one month at a time, concatenating results.
    Using month chunks keeps each request well under the 5000-row cap per
    page and avoids large offset queries that sometimes 502 the API."""
    frames: list[pd.DataFrame] = []
    for m in range(1, 13):
        start = f"{year}-{m:02d}-01T00"
        if m == 12:
            end = f"{year + 1}-01-01T00"
        else:
            end = f"{year}-{m + 1:02d}-01T00"
        df = _fetch_range(route, respondent, start, end, types, facet_type_key)
        if not df.empty:
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def get_region_data_year(
    year: int, respondent: str = "ERCO", refresh: bool = False,
    types: tuple[str, ...] = ("D", "DF", "NG", "TI"),
) -> pd.DataFrame:
    """Fetch hourly EIA-930 region-data for a given year and BA.

    Columns of interest after tidy-up: timestamp_utc, type (D/DF/NG/TI), value.
    """
    endpoint = "electricity/rto/region-data"
    path = _cache_path(endpoint, respondent, year)
    if path.exists() and not refresh:
        logger.info("Loading %s from cache", path)
        return pd.read_parquet(path)

    logger.info("Fetching EIA-930 region-data %s year=%d types=%s",
                respondent, year, types)
    df = _fetch_year_by_month(endpoint, respondent, year, list(types),
                              facet_type_key="type")
    if df.empty:
        logger.warning("No data returned for %s %d", respondent, year)
        return df

    df["timestamp_utc"] = pd.to_datetime(df["period"] + ":00Z", utc=True)
    out = df[["timestamp_utc", "type", "type-name", "respondent", "value"]].copy()
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    out = out.sort_values(["type", "timestamp_utc"]).reset_index(drop=True)
    out.to_parquet(path, index=False)
    logger.info("Cached %d rows to %s", len(out), path)
    return out


def get_fuel_type_year(
    year: int, respondent: str = "ERCO", refresh: bool = False,
) -> pd.DataFrame:
    """Fetch hourly net-generation-by-fuel-type for a year and BA.

    Fuel types include COL (coal), NG (natural gas), NUC (nuclear),
    WND (wind), SUN (solar), WAT (hydro), OIL, OTH, etc.
    """
    endpoint = "electricity/rto/fuel-type-data"
    path = _cache_path(endpoint, respondent, year)
    if path.exists() and not refresh:
        logger.info("Loading %s from cache", path)
        return pd.read_parquet(path)

    logger.info("Fetching EIA-930 fuel-type-data %s year=%d", respondent, year)
    df = _fetch_year_by_month(endpoint, respondent, year, types=None,
                              facet_type_key="fueltype")
    if df.empty:
        logger.warning("No data returned for %s %d", respondent, year)
        return df

    df["timestamp_utc"] = pd.to_datetime(df["period"] + ":00Z", utc=True)
    out = df[["timestamp_utc", "fueltype", "type-name", "respondent", "value"]].copy()
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    out = out.sort_values(["fueltype", "timestamp_utc"]).reset_index(drop=True)
    out.to_parquet(path, index=False)
    logger.info("Cached %d rows to %s", len(out), path)
    return out
