"""Centralized environment / secrets access.

All modules that need API keys go through `get_env_var()` so we can give a
useful error message when something's missing. Never import raw os.environ
inside data modules.

Load order:
    1. process environment (highest precedence)
    2. `.env` file at repo root
    3. `.env.example` — only used to detect "this var is known but blank"
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from dotenv import dotenv_values

from src.paths import ROOT


@lru_cache(maxsize=1)
def _load_dotenv_values() -> dict[str, str]:
    env_path = ROOT / ".env"
    if not env_path.exists():
        return {}
    values = dotenv_values(env_path)
    return {k: v for k, v in values.items() if v is not None}


def get_env_var(name: str, required: bool = True, default: str | None = None) -> str | None:
    """Resolve an env var from process env first, then .env file.

    Args:
        name: env var name (e.g. "EIA_API_KEY").
        required: raise if missing and no default.
        default: returned when var is unset and not required.

    Raises:
        RuntimeError with a link to the right registration page when an
        expected key is missing.
    """
    val = os.environ.get(name)
    if val is None or val == "":
        val = _load_dotenv_values().get(name)

    if val is None or val == "":
        if default is not None:
            return default
        if not required:
            return None
        hint = _registration_hint(name)
        raise RuntimeError(
            f"Missing env var {name}. Put it in .env at the repo root.\n"
            f"{hint}"
        )
    return val


def _registration_hint(name: str) -> str:
    hints = {
        "EIA_API_KEY":
            "Register free at https://www.eia.gov/opendata/register.php "
            "(key arrives by email).",
        "ERCOT_API_USERNAME":
            "Your ERCOT account email. Register at https://apiexplorer.ercot.com/.",
        "ERCOT_API_PASSWORD":
            "Your ERCOT account password. Register at https://apiexplorer.ercot.com/.",
        "ERCOT_API_SUBSCRIPTION_KEY":
            "Primary key of the 'Public API' subscription at "
            "https://apiexplorer.ercot.com/.",
        "CDSAPI_KEY":
            "Personal access token from https://cds.climate.copernicus.eu/ "
            "(only needed for ERA5 calibration — optional).",
    }
    return hints.get(name, "")
