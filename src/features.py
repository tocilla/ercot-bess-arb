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


def build_features(
    prices: pd.Series,
    tz: str | None = None,
    load: pd.Series | None = None,
    scarcity_prob_daily: pd.Series | None = None,
) -> pd.DataFrame:
    """Build a feature DataFrame with lag + rolling + calendar columns.

    Args:
        prices: tz-aware UTC-indexed price series ($/MWh, 15-min).
        tz: local tz for calendar features (e.g. "US/Central").
        load: optional tz-aware UTC-indexed hourly load series (MW). If
            provided, adds load-derived features (lag 1d/1w, rolling
            stats, relative-to-baseline). Load is forward-filled from
            hourly to 15-min and treated as an *actual* that must be
            used only with a lag (no same-time features).

    Returns:
        DataFrame indexed by the same UTC index with columns:
            target                  price at t (for supervised training)
            lag_15min, lag_1h, …    see LAG_INTERVALS
            roll_mean_1d_left       mean of last 96 intervals, strictly before t
            roll_std_1d_left
            roll_mean_1w_left
            hour, interval_of_day, dow, month, is_weekend
            (if load provided:)
            load_lag_1h, load_lag_1d, load_lag_1w
            load_roll_mean_1d_left
            load_rel_to_7d_mean     load_lag_1h / trailing-7d mean of load
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

    if load is not None:
        df = _add_load_features(df, prices.index, load)

    if scarcity_prob_daily is not None:
        df = _add_scarcity_feature(df, prices.index, scarcity_prob_daily, tz)

    return df


def _add_scarcity_feature(
    df: pd.DataFrame,
    target_index: pd.DatetimeIndex,
    scarcity_prob_daily: pd.Series,
    tz: str | None,
) -> pd.DataFrame:
    """Broadcast a per-local-date scarcity probability to the 15-min grid.

    The probability at local date D is used as a feature on every interval
    within date D. This is a legitimate feature for predicting intraday
    prices on that day — but the probability itself must have been
    generated out-of-sample (walk-forward) from data strictly before D,
    not from D itself. Enforcing that is the caller's job.
    """
    local = target_index.tz_convert(tz) if tz else target_index
    local_dates = pd.to_datetime(local.date)
    prob_by_date = scarcity_prob_daily.reindex(local_dates, fill_value=np.nan)
    df["scarcity_prob_today"] = prob_by_date.to_numpy()
    return df


def _add_load_features(df: pd.DataFrame, target_index: pd.DatetimeIndex,
                       load: pd.Series) -> pd.DataFrame:
    """Add load-derived features. Load is hourly; reindex to the 15-min
    target grid via forward-fill (each hour's load applies to its 4 intervals)."""
    if load.index.tz is None:
        raise ValueError("load must be tz-aware")

    # Forward-fill to target (15-min) grid. Limit the fill to 3 intervals so a
    # gap > 1 hour stays NaN.
    load_15 = load.reindex(target_index, method="ffill", limit=3)

    # Lag 1 hour (= 4 intervals), 1 day, 1 week.
    df["load_lag_1h"] = load_15.shift(4)
    df["load_lag_1d"] = load_15.shift(INTERVALS_PER_DAY)
    df["load_lag_1w"] = load_15.shift(7 * INTERVALS_PER_DAY)

    # Rolling mean (24h, closed='left'): trailing-day demand baseline.
    df["load_roll_mean_1d_left"] = _rolling_left(load_15, INTERVALS_PER_DAY, "mean")

    # Relative: load_lag_1h vs trailing-7d mean (excluding current value).
    trailing_7d = _rolling_left(load_15, 7 * INTERVALS_PER_DAY, "mean")
    df["load_rel_to_7d_mean"] = df["load_lag_1h"] / trailing_7d
    return df


def feature_columns(df: pd.DataFrame) -> list[str]:
    """All columns except the target."""
    return [c for c in df.columns if c != "target"]
