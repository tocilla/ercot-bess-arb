"""Battery simulator unit tests. Sign conventions, efficiency, clipping."""

from __future__ import annotations

import math

import pytest

from src.battery import BatterySimulator, BatterySpec


def make_spec(**overrides) -> BatterySpec:
    base = dict(
        power_mw=10.0,
        capacity_mwh=20.0,
        roundtrip_eff=0.81,  # sqrt = 0.9
        soc_min_frac=0.0,
        soc_max_frac=1.0,
        initial_soc_frac=0.5,
        degradation_cost_per_mwh=0.0,
    )
    base.update(overrides)
    return BatterySpec(**base)


def test_idle_action_no_change():
    b = BatterySimulator(make_spec())
    soc_before = b.soc_mwh
    r = b.step(0.0, 1.0, 50.0)
    assert r.actual_grid_power_mw == 0.0
    assert r.energy_to_grid_mwh == 0.0
    assert r.gross_revenue == 0.0
    assert r.net_revenue == 0.0
    assert b.soc_mwh == soc_before
    assert not r.clipped


def test_charge_sign_conventions():
    b = BatterySimulator(make_spec())
    r = b.step(5.0, 1.0, 40.0)  # charge 5 MW for 1h at $40/MWh
    assert r.actual_grid_power_mw == 5.0
    assert r.energy_to_grid_mwh == pytest.approx(-5.0)  # grid gave us 5 MWh
    assert r.gross_revenue == pytest.approx(-200.0)     # we paid $200
    # SOC increase: 5 MW * 1h * sqrt(0.81) = 4.5 MWh
    assert b.soc_mwh == pytest.approx(10.0 + 4.5)


def test_discharge_sign_conventions():
    b = BatterySimulator(make_spec())
    r = b.step(-5.0, 1.0, 40.0)
    assert r.actual_grid_power_mw == -5.0
    assert r.energy_to_grid_mwh == pytest.approx(5.0)
    assert r.gross_revenue == pytest.approx(200.0)
    # SOC decrease: 5 * 1 / sqrt(0.81) = 5.555...
    assert b.soc_mwh == pytest.approx(10.0 - 5.0 / 0.9)


def test_round_trip_is_lossy_in_soc():
    """Charge then discharge the same grid-side energy at equal prices:
    revenue cancels exactly, but SOC drops — that's where the round-trip
    loss lives."""
    b = BatterySimulator(make_spec())
    soc_start = b.soc_mwh
    r1 = b.step(5.0, 1.0, 40.0)
    r2 = b.step(-5.0, 1.0, 40.0)
    # Equal grid-side energy at equal price → gross revenue cancels.
    assert r1.gross_revenue + r2.gross_revenue == pytest.approx(0.0)
    # But the battery paid the round-trip loss out of its stored energy.
    assert b.soc_mwh < soc_start
    expected_soc_loss = 5.0 * 1.0 * (1.0 / 0.9 - 0.9)  # = 5 * 0.2111...
    assert (soc_start - b.soc_mwh) == pytest.approx(expected_soc_loss)


def test_round_trip_revenue_loss_when_returning_to_starting_soc():
    """A charge+discharge pair that restores starting SOC must lose money at
    equal prices, because we bought more grid-side energy than we sold."""
    b = BatterySimulator(make_spec())
    # Charge 5 MW for 1h → SOC +4.5. To drain 4.5 MWh of storage back to grid
    # we can sell only 4.5 * 0.9 = 4.05 MWh grid-side at equal price.
    r1 = b.step(5.0, 1.0, 40.0)         # cost $200, SOC +4.5
    r2 = b.step(-4.05, 1.0, 40.0)       # revenue $162, SOC -4.5
    total = r1.gross_revenue + r2.gross_revenue
    assert total == pytest.approx(-200.0 + 162.0)
    assert total < 0


def test_soc_upper_clip():
    b = BatterySimulator(make_spec(initial_soc_frac=0.95, soc_max_frac=1.0))
    # Headroom = 1.0 MWh; at 10 MW charging for 1h with eta=0.9, SOC would
    # increase by 9 MWh → must clip.
    r = b.step(10.0, 1.0, 50.0)
    assert r.clipped
    assert b.soc_mwh == pytest.approx(20.0)  # full
    # Actual grid power reduced so that 0.9 * P * 1h = 1.0 → P = 1.111...
    assert r.actual_grid_power_mw == pytest.approx(1.0 / 0.9)


def test_soc_lower_clip():
    b = BatterySimulator(make_spec(initial_soc_frac=0.05, soc_min_frac=0.0))
    # Available = 1.0 MWh; at 10 MW discharging for 1h with eta=0.9, SOC would
    # decrease by 10/0.9 ≈ 11.111 MWh → must clip to deplete only 1.0.
    r = b.step(-10.0, 1.0, 50.0)
    assert r.clipped
    assert b.soc_mwh == pytest.approx(0.0)
    # |P| * 1h / 0.9 = 1.0 → |P| = 0.9
    assert r.actual_grid_power_mw == pytest.approx(-0.9)


def test_power_clip():
    b = BatterySimulator(make_spec())
    r = b.step(100.0, 1.0, 50.0)
    assert r.clipped
    assert r.actual_grid_power_mw == 10.0  # rating


def test_degradation_cost_proportional_to_throughput():
    spec = make_spec(degradation_cost_per_mwh=2.0)
    b = BatterySimulator(spec)
    r = b.step(5.0, 1.0, 0.0)  # throughput = 5 * 1 * 0.9 = 4.5 MWh
    assert r.throughput_mwh == pytest.approx(4.5)
    assert r.degradation_cost == pytest.approx(9.0)
    assert r.net_revenue == pytest.approx(-9.0)  # zero price, only deg cost


def test_spec_validation():
    with pytest.raises(ValueError):
        BatterySpec(power_mw=10, capacity_mwh=20, roundtrip_eff=1.5)
    with pytest.raises(ValueError):
        BatterySpec(power_mw=10, capacity_mwh=20, soc_min_frac=0.5, soc_max_frac=0.3)
    with pytest.raises(ValueError):
        BatterySpec(power_mw=-1, capacity_mwh=20)
