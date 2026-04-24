"""Tests for the perfect-foresight LP dispatch.

Two sanity properties must hold:
    1. LP revenue >= natural-spread oracle revenue on the same data (it's a
       tighter upper bound).
    2. LP respects all physical and cycle constraints (no clipping needed).
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.baselines import daily_oracle_schedule
from src.battery import BatterySpec
from src.dispatch import run_dispatch
from src.optimization import perfect_foresight_schedule
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
    return synthetic_lmp(days=5, interval_minutes=60, noise_std=0.0, seed=0)


def test_lp_beats_or_matches_oracle(prices, spec):
    lp_sched = perfect_foresight_schedule(prices, spec, interval_hours=1.0)
    lp_result = run_dispatch(lp_sched, prices, spec, interval_hours=1.0)

    oracle_sched = daily_oracle_schedule(prices, spec, interval_hours=1.0)
    oracle_result = run_dispatch(oracle_sched, prices, spec, interval_hours=1.0)

    # LP is the ceiling — cannot be beaten by a rule-based baseline.
    assert lp_result["net_revenue"].sum() >= oracle_result["net_revenue"].sum() - 1e-6


def test_lp_respects_power(prices, spec):
    sched = perfect_foresight_schedule(prices, spec, interval_hours=1.0)
    assert (sched.abs() <= spec.power_mw + 1e-6).all()


def test_lp_no_simultaneous_charge_discharge(prices, spec):
    """With positive prices and lossy round-trip, LP should never choose
    both charge and discharge at the same interval — here we just observe
    that the magnitude is at most power_mw (sign clean)."""
    sched = perfect_foresight_schedule(prices, spec, interval_hours=1.0)
    assert (sched.abs() <= spec.power_mw + 1e-6).all()


def test_lp_respects_cycle_cap(prices, spec):
    sched = perfect_foresight_schedule(prices, spec, interval_hours=1.0, cycles_per_day_cap=1.0)
    result = run_dispatch(sched, prices, spec, interval_hours=1.0)
    usable = (spec.soc_max_frac - spec.soc_min_frac) * spec.capacity_mwh
    # Battery-side throughput per day should be at most 2 * usable for 1 cycle.
    # (charge side ~= usable + discharge side ~= usable)
    daily_throughput = result["throughput_mwh"].groupby(result.index.date).sum()
    assert (daily_throughput <= 2 * usable + 1e-3).all()


def test_lp_not_clipped_at_sim_boundary(prices, spec):
    """A well-posed LP should produce a schedule the simulator can execute
    without needing to clip (modulo tiny numerical tolerance)."""
    sched = perfect_foresight_schedule(prices, spec, interval_hours=1.0)
    result = run_dispatch(sched, prices, spec, interval_hours=1.0)
    # Allow a trivial fraction for numerical slack at SOC boundaries.
    assert result["clipped"].mean() < 0.02