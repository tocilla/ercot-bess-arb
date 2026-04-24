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


def _get(endpoint: str, params: dict | None = None) -> dict:
    """GET a Public API endpoint with auth + subscription key."""
    url = f"{_API_BASE}/{endpoint.lstrip('/')}"
    r = requests.get(url, headers=_headers(), params=params, timeout=60)
    if r.status_code == 401:
        # Token may have expired between cache check and use. Force refresh.
        _token_cache.clear()
        r = requests.get(url, headers=_headers(), params=params, timeout=60)
    r.raise_for_status()
    return r.json()


# ---------- High-level fetchers ----------

@dataclass
class EndpointSpec:
    report_id: str
    path: str      # path suffix under /api/public-reports
    description: str


ENDPOINTS: dict[str, EndpointSpec] = {
    "wind": EndpointSpec(
        report_id="NP4-732-CD",
        path="np4-732-cd/wpp_hrly_avrg_actl_fcast",
        description="Wind Power Production: Hourly Averaged Actual + STWPF + WGRPP",
    ),
    "solar": EndpointSpec(
        report_id="NP4-737-CD",
        path="np4-737-cd/spp_hrly_avrg_actl_fcast",
        description="Solar Power Production: Hourly Averaged Actual + STPPF",
    ),
    "load_forecast_7d": EndpointSpec(
        report_id="NP3-560-CD",
        path="np3-560-cd/lf_by_model_weather_zone",
        description="Seven-Day Load Forecast by Model and Weather Zone",
    ),
    "outage_capacity": EndpointSpec(
        report_id="NP3-233-CD",
        path="np3-233-cd/hourly_res_outage_cap",
        description="Hourly Resource Outage Capacity (next 168h)",
    ),
}


def fetch_endpoint(
    key: str,
    post_datetime_from: pd.Timestamp | None = None,
    post_datetime_to: pd.Timestamp | None = None,
    page_size: int = 1000,
) -> pd.DataFrame:
    """Fetch a Public API endpoint for a publish-time window.

    The Public API key parameter for historical filtering is
    `postDatetimeFrom` / `postDatetimeTo` — the timestamp at which the
    report was *published*. This is exactly the "as-of" semantic we want
    for vintaged features (METHODOLOGY §3).
    """
    if key not in ENDPOINTS:
        raise KeyError(f"Unknown endpoint key: {key}. Available: {list(ENDPOINTS)}")
    spec = ENDPOINTS[key]

    params: dict[str, str | int] = {"size": page_size}
    if post_datetime_from is not None:
        params["postDatetimeFrom"] = post_datetime_from.strftime("%Y-%m-%dT%H:%M:%S")
    if post_datetime_to is not None:
        params["postDatetimeTo"] = post_datetime_to.strftime("%Y-%m-%dT%H:%M:%S")

    all_rows: list[dict] = []
    page = 1
    while True:
        params["page"] = page
        payload = _get(spec.path, params=params)
        fields = payload.get("fields", [])
        rows = payload.get("data", [])
        if not rows:
            break
        col_names = [f["name"] for f in fields]
        all_rows.extend([dict(zip(col_names, r)) for r in rows])
        meta = payload.get("_meta", {})
        total_pages = meta.get("totalPages", 1)
        if page >= total_pages:
            break
        page += 1
    return pd.DataFrame(all_rows)


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
