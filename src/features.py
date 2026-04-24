"""Feature engineering for price forecasting.

Minimal, auditable feature set for phase 1:
    - Lag features (past prices at fixed offsets).
    - Rolling statistics with `closed='left'` so they can't peek at `t`.
    - Calendar features (hour, day-of-week, month, interval-of-day).

Every feature at time *t* is computed only from data strictly before *t*
(respecting the feature-availability rule, METHODOLOGY §3). We do not yet
model publication delays beyond "strictly before t" — that will matter
once we add real-time features like load forecasts.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

INTERVALS_PER_DAY = 96  # 15-min
LAG_INTERVALS = {
    "lag_15min": 1,
    "lag_1h": 4,
    "lag_4h": 16,
    "lag_1d": INTERVALS_PER_DAY,
    "lag_2d": 2 * INTERVALS_PER_DAY,
    "lag_1w": 7 * INTERVALS_PER_DAY,
    "lag_2w": 14 * INTERVALS_PER_DAY,
}


def _rolling_left(x: pd.Series, window: int, fn: str) -> pd.Series:
    """Rolling aggregate that does NOT include the current value — uses only
    prior observations. `closed='left'` means the window is [t-window, t).
    """
    r = x.rolling(window=window, min_periods=1, closed="left")
    return getattr(r, fn)()


def build_features(prices: pd.Series, tz: str | None = None) -> pd.DataFrame:
    """Build a feature DataFrame with lag + rolling + calendar columns.

    Args:
        prices: tz-aware UTC-indexed price series ($/MWh, 15-min).
        tz: local tz for calendar features (e.g. "US/Central").

    Returns:
        DataFrame indexed by the same UTC index with columns:
            target                  price at t (for supervised training)
            lag_15min, lag_1h, …    see LAG_INTERVALS
            roll_mean_1d_left       mean of last 96 intervals, strictly before t
            roll_std_1d_left
            roll_mean_1w_left
            hour, interval_of_day, dow, month, is_weekend
    """
    if prices.index.tz is None:
        raise ValueError("prices must be tz-aware")
    df = pd.DataFrame(index=prices.index)
    df["target"] = prices.values

    for name, k in LAG_INTERVALS.items():
        df[name] = prices.shift(k)

    df["roll_mean_1d_left"] = _rolling_left(prices, INTERVALS_PER_DAY, "mean")
    df["roll_std_1d_left"] = _rolling_left(prices, INTERVALS_PER_DAY, "std")
    df["roll_mean_1w_left"] = _rolling_left(prices, 7 * INTERVALS_PER_DAY, "mean")

    local_idx = prices.index.tz_convert(tz) if tz else prices.index
    df["hour"] = local_idx.hour.astype("int16")
    df["interval_of_day"] = (
        local_idx.hour * (60 // 15) + local_idx.minute // 15
    ).astype("int16")
    df["dow"] = local_idx.dayofweek.astype("int16")
    df["month"] = local_idx.month.astype("int16")
    df["is_weekend"] = (local_idx.dayofweek >= 5).astype("int8")

    return df


def feature_columns(df: pd.DataFrame) -> list[str]:
    """All columns except the target."""
    return [c for c in df.columns if c != "target"]
