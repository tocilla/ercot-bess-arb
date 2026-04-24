"""Tests for the simple forecasters and the gated natural-spread baseline."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.baselines import daily_oracle_schedule, daily_spread_gated_schedule
from src.battery import BatterySpec
from src.dispatch import run_dispatch
from src.forecasters import (
    persistence_forecast_same_interval_same_dow,
    persistence_forecast_same_interval_yesterday,
    seasonal_naive_forecast,
)


@pytest.fixture
def spec() -> BatterySpec:
    return BatterySpec(
        power_mw=10.0,
        capacity_mwh=20.0,
        roundtrip_eff=0.81,
        soc_min_frac=0.0,
        soc_max_frac=1.0,
        initial_soc_frac=0.5,
        degradation_cost_per_mwh=2.0,
    )


def _hourly_prices(days: int, shape_fn, seed: int = 0, tz: str = "UTC") -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=days * 24, freq="h", tz=tz)
    hours = idx.hour + idx.minute / 60.0
    vals = np.array([shape_fn(h, rng) for h in hours])
    return pd.Series(vals, index=idx, name="lmp")


def test_persistence_shifts_one_day_of_intervals():
    idx = pd.date_range("2024-01-01", periods=24 * 3, freq="h", tz="UTC")
    prices = pd.Series(np.arange(len(idx), dtype=float), index=idx)
    f = persistence_forecast_same_interval_yesterday(prices, intervals_per_day=24)
    assert f.index.equals(prices.index)
    # First 24 intervals: no history → NaN.
    assert f.iloc[:24].isna().all()
    # After: forecast equals price one day earlier.
    assert (f.iloc[24:] == prices.iloc[:-24].values).all()


def test_persistence_same_dow_shifts_one_week():
    idx = pd.date_range("2024-01-01", periods=24 * 14, freq="h", tz="UTC")
    prices = pd.Series(np.arange(len(idx), dtype=float), index=idx)
    f = persistence_forecast_same_interval_same_dow(prices, intervals_per_day=24)
    assert f.iloc[: 24 * 7].isna().all()
    assert (f.iloc[24 * 7:] == prices.iloc[: -24 * 7].values).all()


def test_seasonal_naive_produces_values_after_warmup():
    """With skipna=True median, forecast becomes valid as soon as the first
    (1-week) lookback is available. Using all 4 weeks requires 28 days."""
    idx = pd.date_range("2024-01-01", periods=24 * 35, freq="h", tz="UTC")
    prices = pd.Series(np.arange(len(idx), dtype=float), index=idx)
    f = seasonal_naive_forecast(prices, lookback_weeks=4, intervals_per_day=24)
    # First 7*24 intervals: no lookback → NaN.
    assert f.iloc[: 7 * 24].isna().all()
    # After 7 days: at least one lookback is valid → forecast is valid.
    assert f.iloc[7 * 24:].notna().all()
    # At t = 28 days, all 4 lookbacks are valid and median equals the
    # median of the 4 corresponding historical values.
    t = 4 * 7 * 24
    expected = float(np.median([prices.iloc[t - 7 * 24 * k] for k in range(1, 5)]))
    assert f.iloc[t] == pytest.approx(expected)


def test_gate_skips_flat_day_keeps_spread_day(spec):
    """Day 0: flat at $2 → even SOC-drain revenue can't cover degradation
    costs → gate skips. Day 1: big intraday spread → keep."""
    flat = [2.0] * 24
    spread = [10.0 if h < 12 else 100.0 for h in range(24)]
    idx = pd.date_range("2024-01-01", periods=48, freq="h", tz="UTC")
    prices = pd.Series(flat + spread, index=idx, name="lmp")

    base = daily_oracle_schedule(prices, spec, interval_hours=1.0)
    gated = daily_spread_gated_schedule(prices, prices, spec, interval_hours=1.0)

    day0_idx = idx[:24]
    day1_idx = idx[24:]

    assert (base.loc[day0_idx] != 0).any()          # base cycles blindly
    assert (gated.loc[day0_idx] == 0).all()         # gate drops flat/cheap day
    assert (gated.loc[day1_idx] != 0).any()         # spread day survives


def test_gate_never_reduces_total_revenue(spec):
    """The gate can only help: it replaces loss-making days with zero."""
    rng = np.random.default_rng(42)
    idx = pd.date_range("2024-01-01", periods=24 * 10, freq="h", tz="UTC")
    prices = pd.Series(20 + rng.normal(0, 3, size=len(idx)), index=idx, name="lmp")

    base = daily_oracle_schedule(prices, spec, interval_hours=1.0)
    gated = daily_spread_gated_schedule(prices, prices, spec, interval_hours=1.0)

    base_rev = run_dispatch(base, prices, spec, 1.0)["net_revenue"].sum()
    gated_rev = run_dispatch(gated, prices, spec, 1.0)["net_revenue"].sum()
    assert gated_rev >= base_rev - 1e-6
