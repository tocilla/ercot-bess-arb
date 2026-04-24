"""Simple non-ML forecasters that produce a next-24h price series given
only past prices. These serve as forecasting baselines per PLAN §6.2–6.3
and as sanity anchors for any later ML model.

All forecasters return a pd.Series with the SAME index as `prices` where
each value at timestamp t is the forecast *of the price at t*, produced
using only data available at a time strictly before t (respecting the
feature-availability rule in METHODOLOGY §3).

Typical usage:
    forecast = persistence_forecast_same_interval_yesterday(prices)
    schedule = daily_oracle_schedule(forecast, spec, ...)
    result = run_dispatch(schedule, prices, spec, ...)  # execute on real
"""

from __future__ import annotations

import pandas as pd


def persistence_forecast_same_interval_yesterday(
    prices: pd.Series, intervals_per_day: int = 96,
) -> pd.Series:
    """Forecast(t) = Price(t − 1 day). For 15-min data, that's a 96-step shift.

    Drops the first day (no history to forecast from). Caller may need to
    `.dropna()` or handle the leading NaNs explicitly.
    """
    return prices.shift(intervals_per_day)


def persistence_forecast_same_interval_same_dow(
    prices: pd.Series, intervals_per_day: int = 96,
) -> pd.Series:
    """Forecast(t) = Price(t − 7 days). Same weekday, same interval."""
    return prices.shift(intervals_per_day * 7)


def seasonal_naive_forecast(
    prices: pd.Series,
    lookback_weeks: int = 4,
    intervals_per_day: int = 96,
) -> pd.Series:
    """Forecast(t) = median of Price(t − 7d), Price(t − 14d), …, out to
    `lookback_weeks` weeks back. Picks same DOW, same interval-of-day.

    Median (not mean) for robustness to scarcity spikes in the training
    window — a single $9,000 print shouldn't move tomorrow's forecast.
    """
    step = intervals_per_day * 7
    cols = [prices.shift(step * k) for k in range(1, lookback_weeks + 1)]
    stacked = pd.concat(cols, axis=1)
    return stacked.median(axis=1).rename(f"seasonal_naive_{lookback_weeks}w")
