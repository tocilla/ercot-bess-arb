"""ERCOT Public API client — vintaged load/wind/solar/outage forecasts.

Uses the modern ERCOT Public API (apiexplorer.ercot.com) which archives
≥7 years of history, distinct from the old MIS archive that rolls off
in ~8 days.

Authentication: ROPC OAuth2 against ERCOT's B2C tenant.
    1. POST username + password + client_id → get access_token
    2. Use `Authorization: Bearer <token>` + `Ocp-Apim-Subscription-Key: <key>` headers
Token expires; refresh as needed.

Endpoints of immediate interest (IDs from DATA_GAP.md):
    NP4-732-CD — Wind Power Production (actual + STWPF + WGRPP)
    NP4-737-CD — Solar Power Production (actual + STPPF)
    NP3-560-CD — 7-Day Load Forecast by Model & Weather Zone
    NP3-233-CD — Hourly Resource Outage Capacity

Requires `.env` variables:
    ERCOT_API_USERNAME, ERCOT_API_PASSWORD, ERCOT_API_SUBSCRIPTION_KEY.

Status: scaffolded + authentication flow implemented + tested structurally.
End-to-end verification against live endpoints is blocked until the user
populates username/password in .env.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import requests

from src.config_env import get_env_var
from src.paths import DATA_RAW

logger = logging.getLogger(__name__)

CACHE_DIR = DATA_RAW / "ercot_api"

# ERCOT B2C OAuth constants (public, documented in ERCOT developer portal).
_AUTH_URL = (
    "https://ercotb2c.b2clogin.com/ercotb2c.onmicrosoft.com/"
    "B2C_1_PUBAPI-ROPC-FLOW/oauth2/v2.0/token"
)
_CLIENT_ID = "fec253ea-0d06-4272-a5e6-b478baeecd70"
_SCOPE = f"openid {_CLIENT_ID} offline_access"
_API_BASE = "https://api.ercot.com/api/public-reports"

# Token cache — simple in-memory, refresh on expiry.
_token_cache: dict[str, float | str] = {}


def _get_access_token() -> str:
    """ROPC auth: exchange username+password for a bearer access token."""
    now = time.time()
    if _token_cache and _token_cache.get("expires_at", 0) > now + 30:
        return str(_token_cache["access_token"])

    username = get_env_var("ERCOT_API_USERNAME")
    password = get_env_var("ERCOT_API_PASSWORD")
    data = {
        "username": username,
        "password": password,
        "grant_type": "password",
        "scope": _SCOPE,
        "client_id": _CLIENT_ID,
        "response_type": "id_token",
    }
    r = requests.post(_AUTH_URL, data=data, timeout=30)
    if r.status_code != 200:
        raise RuntimeError(
            f"ERCOT auth failed: {r.status_code} {r.text[:200]}. "
            "Verify ERCOT_API_USERNAME / ERCOT_API_PASSWORD."
        )
    payload = r.json()
    token = payload["access_token"]
    expires_in = int(payload.get("expires_in", 3600))
    _token_cache["access_token"] = token
    _token_cache["expires_at"] = now + expires_in
    return token


def _headers() -> dict[str, str]:
    return {
        "Authorization": f"Bearer {_get_access_token()}",
        "Ocp-Apim-Subscription-Key": get_env_var("ERCOT_API_SUBSCRIPTION_KEY"),
    }


def _get(endpoint: str, params: dict | None = None, _retries: int = 5) -> dict:
    """GET a Public API endpoint with auth + subscription key.

    Retries on 401 (token refresh) and 429 (rate limit) with backoff.
    """
    url = f"{_API_BASE}/{endpoint.lstrip('/')}"
    last_err: Exception | None = None
    backoff = 2.0
    for attempt in range(_retries):
        r = requests.get(url, headers=_headers(), params=params, timeout=60)
        if r.status_code == 401:
            _token_cache.clear()
            continue
        if r.status_code == 429:
            wait = float(r.headers.get("Retry-After", backoff))
            logger.warning("429 rate-limited, sleeping %.1fs (attempt %d)",
                           wait, attempt + 1)
            time.sleep(wait)
            backoff = min(backoff * 2, 60.0)
            continue
        try:
            r.raise_for_status()
        except requests.HTTPError as e:
            last_err = e
            break
        return r.json()
    if last_err:
        raise last_err
    raise RuntimeError(f"Exhausted retries on {url}")


# ---------- High-level fetchers ----------

@dataclass
class EndpointSpec:
    report_id: str
    description: str


ENDPOINTS: dict[str, EndpointSpec] = {
    "wind": EndpointSpec(
        report_id="NP4-732-CD",
        description="Wind Power Production: Hourly Averaged Actual + STWPF + WGRPP",
    ),
    "solar": EndpointSpec(
        report_id="NP4-737-CD",
        description="Solar Power Production: Hourly Averaged Actual + STPPF",
    ),
    "load_forecast_7d": EndpointSpec(
        report_id="NP3-560-CD",
        description="Seven-Day Load Forecast by Forecast Zone",
    ),
    "outage_capacity": EndpointSpec(
        report_id="NP3-233-CD",
        description="Hourly Resource Outage Capacity (next 168h)",
    ),
}


def list_archives(
    report_id: str,
    post_datetime_from: pd.Timestamp | None = None,
    post_datetime_to: pd.Timestamp | None = None,
    page_size: int = 1000,
) -> pd.DataFrame:
    """List archive documents (docId + post datetime) for a report.

    The archive endpoint returns metadata — one row per time a report
    was published. To get the actual data, call `download_archive(docId)`.

    `postDatetimeFrom`/`postDatetimeTo` filter by the *publish* time,
    which is exactly the "as-of" semantic we want (METHODOLOGY §3).
    """
    params: dict[str, str | int] = {"size": page_size}
    if post_datetime_from is not None:
        params["postDatetimeFrom"] = post_datetime_from.strftime("%Y-%m-%dT%H:%M:%S")
    if post_datetime_to is not None:
        params["postDatetimeTo"] = post_datetime_to.strftime("%Y-%m-%dT%H:%M:%S")

    all_rows: list[dict] = []
    page = 1
    while True:
        params["page"] = page
        payload = _get(f"archive/{report_id}", params=params)
        docs = payload.get("archives", [])
        if not docs:
            break
        for d in docs:
            all_rows.append({
                "doc_id": d.get("docId"),
                "friendly_name": d.get("friendlyName"),
                "post_datetime": d.get("postDatetime"),
            })
        meta = payload.get("_meta", {})
        total_pages = meta.get("totalPages", 1)
        if page >= total_pages:
            break
        page += 1

    df = pd.DataFrame(all_rows)
    if not df.empty:
        df["post_datetime"] = pd.to_datetime(df["post_datetime"])
    return df


def download_archive(report_id: str, doc_id: int, _retries: int = 5) -> bytes:
    """Download a single archive document's raw bytes (typically a ZIP
    containing CSV). Retries on 401 (token refresh) and 429 (rate limit)."""
    url = f"{_API_BASE}/archive/{report_id}"
    backoff = 2.0
    last_err: Exception | None = None
    for attempt in range(_retries):
        r = requests.get(url, headers=_headers(),
                         params={"download": doc_id}, timeout=120)
        if r.status_code == 401:
            _token_cache.clear()
            continue
        if r.status_code == 429:
            wait = float(r.headers.get("Retry-After", backoff))
            logger.warning("429 on download, sleeping %.1fs", wait)
            time.sleep(wait)
            backoff = min(backoff * 2, 60.0)
            continue
        try:
            r.raise_for_status()
        except requests.HTTPError as e:
            last_err = e
            break
        return r.content
    if last_err:
        raise last_err
    raise RuntimeError(f"Exhausted retries downloading doc {doc_id}")


def download_archive_as_df(report_id: str, doc_id: int) -> pd.DataFrame:
    """Download an archive document and return the first CSV inside as a
    DataFrame. Most ERCOT reports ship as a ZIP with one CSV."""
    import io
    import zipfile
    data = download_archive(report_id, doc_id)
    # Most responses are zipped CSV; some are raw CSV.
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            names = [n for n in z.namelist() if n.lower().endswith(".csv")]
            if not names:
                raise ValueError(f"No CSV inside ZIP for docId={doc_id}")
            with z.open(names[0]) as f:
                return pd.read_csv(f)
    except zipfile.BadZipFile:
        return pd.read_csv(io.BytesIO(data))


def fetch_endpoint(
    key: str,
    post_datetime_from: pd.Timestamp | None = None,
    post_datetime_to: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Convenience: list archives for a named endpoint in a publish-time
    window. Use `download_archive_as_df` to fetch individual docs.
    """
    if key not in ENDPOINTS:
        raise KeyError(f"Unknown endpoint key: {key}. Available: {list(ENDPOINTS)}")
    return list_archives(ENDPOINTS[key].report_id,
                         post_datetime_from=post_datetime_from,
                         post_datetime_to=post_datetime_to)


def smoke_test_auth() -> None:
    """Raise a clean error if username/password are missing; otherwise
    attempt a token exchange and print the expiry."""
    username = get_env_var("ERCOT_API_USERNAME", required=False)
    if not username:
        raise RuntimeError(
            "ERCOT_API_USERNAME is not set. Fill it in .env — it's the "
            "email you used to register at https://apiexplorer.ercot.com/."
        )
    token = _get_access_token()
    logger.info("Got token of length %d", len(token))
    print(f"OK — token acquired (length {len(token)}), cache expiry at "
          f"{_token_cache.get('expires_at')}")
