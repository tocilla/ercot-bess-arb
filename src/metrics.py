"""Dispatch metrics.

Primary metrics (METHODOLOGY §5.2):
    - revenue per day ($/MWh-installed per day, or total $/day)
    - % of ceiling (perfect-foresight) captured
    - Sharpe across days
    - worst-day drawdown
    - missed opportunity cost (energy left on the table)

Attribution (METHODOLOGY §5.3) belongs in a higher-level analysis that
combines multiple dispatch runs (natural spread, threshold-rule, LP, etc.).
This module provides the primitives.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def daily_revenue(result: pd.DataFrame, tz: str | None = None) -> pd.Series:
    """Sum net_revenue by local calendar day."""
    idx = result.index
    if idx.tz is None:
        raise ValueError("result index must be tz-aware")
    local = idx.tz_convert(tz) if tz else idx
    return result.groupby(local.date)["net_revenue"].sum()


def summarize(result: pd.DataFrame, tz: str | None = None) -> dict[str, float]:
    """One-number summary of a dispatch run."""
    daily = daily_revenue(result, tz=tz)
    return {
        "total_revenue": float(result["net_revenue"].sum()),
        "mean_revenue_per_day": float(daily.mean()),
        "std_revenue_per_day": float(daily.std(ddof=1)) if len(daily) > 1 else 0.0,
        "sharpe_daily": float(daily.mean() / daily.std(ddof=1))
        if len(daily) > 1 and daily.std(ddof=1) > 0
        else float("nan"),
        "worst_day": float(daily.min()),
        "best_day": float(daily.max()),
        "n_days": int(len(daily)),
        "total_throughput_mwh": float(result["throughput_mwh"].sum()),
        "pct_intervals_clipped": 100.0 * float(result["clipped"].mean()),
    }


def pct_of_ceiling(actual: float, ceiling: float) -> float:
    """actual / ceiling as a percentage. Returns nan if ceiling is non-positive."""
    if ceiling <= 0:
        return float("nan")
    return 100.0 * actual / ceiling


def compare(runs: dict[str, pd.DataFrame], tz: str | None = None) -> pd.DataFrame:
    """Side-by-side summary of multiple dispatch runs."""
    rows = {name: summarize(df, tz=tz) for name, df in runs.items()}
    return pd.DataFrame(rows).T
