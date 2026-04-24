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
    eia: pd.DataFrame | None = None,
    hrrr: pd.DataFrame | None = None,
    ercot_wind_forecasts: pd.DataFrame | None = None,
    ercot_solar_forecasts: pd.DataFrame | None = None,
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

    if eia is not None:
        df = _add_eia_features(df, prices.index, eia)

    if hrrr is not None:
        df = _add_hrrr_features(df, prices.index, hrrr)

    if ercot_wind_forecasts is not None or ercot_solar_forecasts is not None:
        df = _add_ercot_forecast_features(
            df, prices.index, tz or "UTC",
            ercot_wind_forecasts, ercot_solar_forecasts,
        )

    return df


def _parse_delivery_ts(df: pd.DataFrame, tz: str) -> pd.DatetimeIndex:
    """ERCOT reports use DELIVERY_DATE (local date, MM/DD/YYYY) + HOUR_ENDING
    (1..24 or 25). Interval start = hour - 1 in local time. Handle DST via
    DSTFlag when present.
    """
    date = pd.to_datetime(df["DELIVERY_DATE"], format="%m/%d/%Y")
    hour_start = df["HOUR_ENDING"].astype(int) - 1
    local = date + pd.to_timedelta(hour_start, unit="h")
    # DST: ERCOT uses DSTFlag ('Y'/'N') on the repeated fall-back hour.
    if "DSTFlag" in df.columns:
        # ambiguous=True marks the FIRST occurrence (DST) of the repeated
        # local hour; ambiguous=False is the second (standard).
        ambiguous = (df["DSTFlag"].astype(str).str.upper() == "Y")
        return local.dt.tz_localize(tz, ambiguous=ambiguous.to_numpy(),
                                    nonexistent="shift_forward").dt.tz_convert("UTC")
    return local.dt.tz_localize(tz, ambiguous="infer",
                                nonexistent="shift_forward").dt.tz_convert("UTC")


def _add_ercot_forecast_features(
    df: pd.DataFrame,
    target_index: pd.DatetimeIndex,
    tz: str,
    wind: pd.DataFrame | None,
    solar: pd.DataFrame | None,
) -> pd.DataFrame:
    """Add STWPF (wind) and STPPF (solar) features keyed by publish-time-
    aware lookup: for each target t, use the forecast from the latest
    document whose `post_datetime_utc < t`.
    """
    if wind is not None and not wind.empty:
        df["ercot_stwpf_system_wide"] = _lookup_latest_forecast_by_valid(
            wind, "STWPF_SYSTEM_WIDE", tz, target_index
        )
    if solar is not None and not solar.empty:
        df["ercot_stppf_system_wide"] = _lookup_latest_forecast_by_valid(
            solar, "STPPF_SYSTEM_WIDE", tz, target_index
        )
    return df


def _lookup_latest_forecast_by_valid(
    forecasts: pd.DataFrame,
    value_col: str,
    tz: str,
    target_index: pd.DatetimeIndex,
) -> np.ndarray:
    """Build a per-`valid_utc` view with the latest publish whose
    `post_datetime_utc < valid_utc`, then align to `target_index`.
    """
    f = forecasts.copy()
    f["valid_utc"] = _parse_delivery_ts(f, tz)
    f["post_datetime_utc"] = pd.to_datetime(f["post_datetime_utc"], utc=True)
    # Only keep rows where the forecast was PUBLISHED before its valid time —
    # past-hour rows in the doc are historical / not forecasts at publish.
    f = f[f["post_datetime_utc"] < f["valid_utc"]]
    # For each valid_utc, keep the newest publish.
    f = f.sort_values(["valid_utc", "post_datetime_utc"])
    f = f.drop_duplicates("valid_utc", keep="last")
    series = f.set_index("valid_utc")[value_col]
    # Reindex to the target grid, forward-fill up to 24h.
    union = target_index.union(series.index)
    aligned = series.reindex(union).sort_index().ffill(
        limit=INTERVALS_PER_DAY
    ).reindex(target_index)
    return aligned.to_numpy()


def _add_hrrr_features(
    df: pd.DataFrame,
    target_index: pd.DatetimeIndex,
    hrrr: pd.DataFrame,
) -> pd.DataFrame:
    """Add HRRR weather-forecast features. The HRRR summary table has one
    row per (cycle, forecast_hour) with `valid_utc` and the Texas aggregates.

    For each target interval *t*, we pick the HRRR forecast whose
    `valid_utc` is closest to *t* AND whose `cycle_utc` is strictly
    before *t* (so we only use forecasts available at decision time).
    """
    if "valid_utc" not in hrrr.columns or "cycle_utc" not in hrrr.columns:
        raise ValueError("hrrr must have cycle_utc and valid_utc columns")
    # Hourly-index the HRRR values by valid_utc. If a target t falls in
    # [valid_utc, valid_utc+1h), we use that forecast — provided its
    # cycle was published before t.
    h = hrrr.copy()
    h["cycle_utc"] = pd.to_datetime(h["cycle_utc"], utc=True)
    h["valid_utc"] = pd.to_datetime(h["valid_utc"], utc=True)
    # Keep the most-recent forecast (newest cycle) per valid time.
    h = h.sort_values(["valid_utc", "cycle_utc"])
    h = h.drop_duplicates(subset=["valid_utc"], keep="last")
    h = h.set_index("valid_utc")

    # Reindex to the 15-min target grid, forward-filling from the most
    # recent valid forecast. Limit fill to 24h so one HRRR obs/day covers
    # the full day, but a multi-day outage still stays NaN.
    cols = ["tx_mean_t2m_k", "tx_max_t2m_k", "tx_mean_wind10m_mps",
            "cycle_utc"]
    h_15 = h[cols].reindex(
        target_index.union(h.index), method=None
    ).sort_index().ffill(limit=INTERVALS_PER_DAY).reindex(target_index)

    # Leak guard: forecast is admissible only if its cycle was BEFORE t.
    # If cycle_utc >= target timestamp, null that row's forecast values.
    mask_ok = pd.to_datetime(h_15["cycle_utc"], utc=True, errors="coerce") < target_index
    for c in ("tx_mean_t2m_k", "tx_max_t2m_k", "tx_mean_wind10m_mps"):
        vals = h_15[c].where(mask_ok)
        df[f"hrrr_{c}"] = vals.to_numpy()
    return df


def _add_eia_features(
    df: pd.DataFrame,
    target_index: pd.DatetimeIndex,
    eia: pd.DataFrame,
) -> pd.DataFrame:
    """Add EIA-930 derived features. Forecast-vs-actual delta is the
    operator's own forecast error, which is a proxy for scarcity risk.

    The DF (day-ahead forecast) is published the day before. So using
    the forecast for hour t is legitimate at time t (forecast was known
    ≥ some hours before t). The ACTUAL demand values at t are only known
    post-hoc and must be LAGGED.
    """
    if eia.index.tz is None:
        raise ValueError("eia must be tz-aware")
    # Forward-fill hourly to the 15-min target grid, limit to 3 intervals (1h).
    eia_15 = eia.reindex(target_index, method="ffill", limit=3)

    if "demand_forecast_mw" in eia_15.columns:
        # Today's forecast — safe (was known ≥ 1 day prior).
        df["eia_demand_forecast_mw"] = eia_15["demand_forecast_mw"]
    if "demand_actual_mw" in eia_15.columns:
        # Actuals must be lagged.
        df["eia_demand_actual_lag_1d"] = eia_15["demand_actual_mw"].shift(INTERVALS_PER_DAY)
    if "demand_forecast_mw" in eia_15.columns and "demand_actual_mw" in eia_15.columns:
        # Yesterday's forecast minus yesterday's actual: forecaster error signal.
        prev_err = (
            eia_15["demand_forecast_mw"].shift(INTERVALS_PER_DAY)
            - eia_15["demand_actual_mw"].shift(INTERVALS_PER_DAY)
        )
        df["eia_forecast_error_lag_1d"] = prev_err
    if "wind_mw" in eia_15.columns:
        df["eia_wind_lag_1d"] = eia_15["wind_mw"].shift(INTERVALS_PER_DAY)
        df["eia_wind_lag_1w"] = eia_15["wind_mw"].shift(7 * INTERVALS_PER_DAY)
    if "solar_mw" in eia_15.columns:
        df["eia_solar_lag_1d"] = eia_15["solar_mw"].shift(INTERVALS_PER_DAY)
        df["eia_solar_lag_1w"] = eia_15["solar_mw"].shift(7 * INTERVALS_PER_DAY)
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
