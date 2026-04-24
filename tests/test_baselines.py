"""Baseline scheduler tests.

End-to-end check: on synthetic daily-sinusoid prices, the oracle baseline
should produce positive revenue, not over-cycle, and respect battery limits.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.baselines import daily_oracle_schedule
from src.battery import BatterySpec
from src.dispatch import run_dispatch
from src.synthetic import synthetic_lmp


@pytest.fixture
def spec() -> BatterySpec:
    return BatterySpec(
        power_mw=10.0,
        capacity_mwh=20.0,
        roundtrip_eff=0.81,
        soc_min_frac=0.0,
        soc_max_frac=1.0,
        initial_soc_frac=0.5,
        degradation_cost_per_mwh=0.0,
    )


@pytest.fixture
def prices() -> pd.Series:
    return synthetic_lmp(days=7, interval_minutes=60, noise_std=0.0, seed=0)


def test_oracle_schedule_same_index_as_prices(prices, spec):
    sched = daily_oracle_schedule(prices, spec, interval_hours=1.0)
    assert sched.index.equals(prices.index)


def test_oracle_produces_positive_revenue(prices, spec):
    sched = daily_oracle_schedule(prices, spec, interval_hours=1.0)
    result = run_dispatch(sched, prices, spec, interval_hours=1.0)
    assert result["net_revenue"].sum() > 0


def test_oracle_respects_power_limits(prices, spec):
    sched = daily_oracle_schedule(prices, spec, interval_hours=1.0)
    assert (sched.abs() <= spec.power_mw + 1e-9).all()


def test_oracle_respects_soc_limits(prices, spec):
    sched = daily_oracle_schedule(prices, spec, interval_hours=1.0)
    result = run_dispatch(sched, prices, spec, interval_hours=1.0)
    assert (result["soc_mwh"] >= spec.soc_min_mwh - 1e-6).all()
    assert (result["soc_mwh"] <= spec.soc_max_mwh + 1e-6).all()


def test_oracle_approximate_daily_cycle_budget(prices, spec):
    """With cycles_per_day=1, daily throughput should be ~ 2 * usable_mwh
    (once in on charge, once out on discharge), up to efficiency and clipping."""
    sched = daily_oracle_schedule(prices, spec, interval_hours=1.0, cycles_per_day=1.0)
    result = run_dispatch(sched, prices, spec, interval_hours=1.0)
    usable = (spec.soc_max_frac - spec.soc_min_frac) * spec.capacity_mwh
    # Battery-side throughput per cycle = 2 * usable (in + out).
    daily_throughput = result["throughput_mwh"].groupby(result.index.date).sum()
    # Allow headroom for days where initial SOC prevents a full cycle.
    assert (daily_throughput <= 2 * usable + 1e-6).all()
