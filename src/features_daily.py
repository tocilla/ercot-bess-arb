"""Daily-aggregated features for a scarcity-day classifier.

Target: daily binary — `max_price_on_day > scarcity_threshold`, in local tz.

All features at day D are computed strictly from data before day D —
specifically from the preceding 7-day window of intraday prices + load.
Each returned row has index = local calendar date (one row per day).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_daily_features(
    prices: pd.Series,
    tz: str,
    load: pd.Series | None = None,
    scarcity_threshold: float = 500.0,
) -> pd.DataFrame:
    """Build daily features + target for scarcity classification.

    Args:
        prices: tz-aware UTC 15-min SPPs.
        tz: local timezone for daily grouping.
        load: optional tz-aware hourly load series.
        scarcity_threshold: day is "scarcity" if max price > this value.

    Returns:
        DataFrame indexed by local date with columns:
            target_scarcity       1 if max price that day > threshold
            target_max_price      max price that day
            prev1d_*              statistics of day D-1 intraday prices
            prev7d_*              statistics of past 7 days of prices
            prev1d_load_peak_mw   peak load on day D-1 (if load given)
            prev7d_load_max_mw    peak load over past 7 days
            dow                   day-of-week (0=Mon)
            month                 calendar month
            days_since_scarcity   days since last scarcity day (clipped to 30)
    """
    if prices.index.tz is None:
        raise ValueError("prices must be tz-aware")
    local = prices.index.tz_convert(tz)
    day_key = pd.Series(local.date, index=prices.index, name="local_date")

    per_day = (
        prices.groupby(day_key)
        .agg(["max", "min", "mean", "std"])
        .rename(columns={"max": "day_max", "min": "day_min",
                         "mean": "day_mean", "std": "day_std"})
    )
    per_day["day_range"] = per_day["day_max"] - per_day["day_min"]
    per_day["target_max_price"] = per_day["day_max"]
    per_day["target_scarcity"] = (per_day["day_max"] > scarcity_threshold).astype(int)

    # Lagged features (strictly from previous days).
    for col in ["day_max", "day_min", "day_mean", "day_std", "day_range"]:
        per_day[f"prev1d_{col}"] = per_day[col].shift(1)
        per_day[f"prev7d_{col}_max"] = per_day[col].shift(1).rolling(
            7, min_periods=1, closed="right"
        ).max()
        per_day[f"prev7d_{col}_mean"] = per_day[col].shift(1).rolling(
            7, min_periods=1, closed="right"
        ).mean()
    # Drop the same-day raw columns — those would leak.
    per_day = per_day.drop(columns=["day_max", "day_min", "day_mean",
                                    "day_std", "day_range"])

    # Load features (daily peak, prev day).
    if load is not None:
        if load.index.tz is None:
            raise ValueError("load must be tz-aware")
        load_local = load.copy()
        load_local.index = load_local.index.tz_convert(tz)
        daily_load = load_local.groupby(load_local.index.date).agg(["max", "mean"])
        daily_load.columns = ["load_peak_mw", "load_mean_mw"]
        daily_load.index = pd.Index(daily_load.index, name="local_date")
        per_day = per_day.join(daily_load, how="left")
        per_day["prev1d_load_peak_mw"] = per_day["load_peak_mw"].shift(1)
        per_day["prev7d_load_max_mw"] = per_day["load_peak_mw"].shift(1).rolling(
            7, min_periods=1, closed="right"
        ).max()
        per_day["prev7d_load_mean_mw"] = per_day["load_mean_mw"].shift(1).rolling(
            7, min_periods=1, closed="right"
        ).mean()
        per_day = per_day.drop(columns=["load_peak_mw", "load_mean_mw"])

    # Calendar features.
    dates = pd.to_datetime(per_day.index)
    per_day["dow"] = dates.dayofweek.astype("int16")
    per_day["month"] = dates.month.astype("int16")

    # Days since last scarcity — encodes recency risk clustering.
    scarcity = per_day["target_scarcity"].shift(1)
    last_hit_idx = np.where(scarcity.fillna(0).values > 0,
                            np.arange(len(scarcity)), -1)
    # Forward-fill the last observed hit index.
    last_hit_ff = pd.Series(last_hit_idx, index=per_day.index).replace(-1, np.nan).ffill()
    days_since = np.arange(len(per_day)) - last_hit_ff.values
    days_since = np.where(np.isnan(days_since), 30, days_since)  # no hit yet → cap
    per_day["days_since_scarcity"] = np.minimum(days_since, 30).astype(int)

    return per_day
