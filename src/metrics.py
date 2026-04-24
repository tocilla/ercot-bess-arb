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


def classify_days(
    prices: pd.Series,
    tz: str | None = None,
    scarcity_threshold: float = 500.0,
    negative_threshold: float = 0.0,
) -> pd.DataFrame:
    """Per-local-day regime flags: scarcity (max > threshold), negative
    (min < threshold). A day can be both.

    Returns a DataFrame indexed by local calendar date with boolean columns
    `has_scarcity` and `has_negative`, plus `max_price`, `min_price`.
    """
    if prices.index.tz is None:
        raise ValueError("prices index must be tz-aware")
    local = prices.index.tz_convert(tz) if tz else prices.index
    by_day = prices.groupby(local.date)
    out = pd.DataFrame(
        {
            "max_price": by_day.max(),
            "min_price": by_day.min(),
        }
    )
    out["has_scarcity"] = out["max_price"] > scarcity_threshold
    out["has_negative"] = out["min_price"] < negative_threshold
    return out


def regime_breakdown(
    result: pd.DataFrame,
    prices: pd.Series,
    tz: str | None = None,
    scarcity_threshold: float = 500.0,
    negative_threshold: float = 0.0,
) -> pd.DataFrame:
    """Revenue by regime: normal / negative-only / scarcity-only / both.

    Rows are mutually exclusive regimes. Columns: `n_days`, `total_revenue`,
    `mean_per_day`, `pct_of_total_revenue`. Useful for answering "where is
    the money actually earned?".
    """
    daily = daily_revenue(result, tz=tz)
    flags = classify_days(prices, tz=tz,
                          scarcity_threshold=scarcity_threshold,
                          negative_threshold=negative_threshold)
    # Align; drop days missing from either side.
    df = daily.to_frame("revenue").join(flags, how="inner")

    def _regime(row: pd.Series) -> str:
        s = bool(row["has_scarcity"])
        n = bool(row["has_negative"])
        if s and n:
            return "both"
        if s:
            return "scarcity_only"
        if n:
            return "negative_only"
        return "normal"

    df["regime"] = df.apply(_regime, axis=1)
    grp = df.groupby("regime")["revenue"].agg(["count", "sum", "mean"])
    grp.columns = ["n_days", "total_revenue", "mean_per_day"]
    total = grp["total_revenue"].sum()
    grp["pct_of_total_revenue"] = 100.0 * grp["total_revenue"] / total if total != 0 else 0.0
    return grp.sort_values("total_revenue", ascending=False)
