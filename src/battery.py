"""Battery simulator with round-trip efficiency, SOC bounds, and degradation cost.

Sign convention (grid-side power):
    grid_power_mw > 0  → charging from grid  (cost to operator)
    grid_power_mw < 0  → discharging to grid (revenue to operator)
    grid_power_mw = 0  → idle

Round-trip efficiency η_rt is split symmetrically as √η_rt on charge and √η_rt
on discharge, which is the standard convention when only the round-trip figure
is known.

    Charging:    SOC += P · Δt · √η_rt              (grid meters P · Δt)
    Discharging: SOC -= |P| · Δt / √η_rt            (grid meters |P| · Δt)
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class BatterySpec:
    power_mw: float
    capacity_mwh: float
    roundtrip_eff: float = 0.85
    soc_min_frac: float = 0.05
    soc_max_frac: float = 0.95
    initial_soc_frac: float = 0.5
    degradation_cost_per_mwh: float = 0.0  # $ per MWh of battery-side throughput

    def __post_init__(self) -> None:
        if not (0 < self.roundtrip_eff <= 1):
            raise ValueError("roundtrip_eff must be in (0, 1]")
        if not (0 <= self.soc_min_frac < self.soc_max_frac <= 1):
            raise ValueError("soc_min_frac must be < soc_max_frac within [0, 1]")
        if not (self.soc_min_frac <= self.initial_soc_frac <= self.soc_max_frac):
            raise ValueError("initial_soc_frac must be within [soc_min_frac, soc_max_frac]")
        if self.power_mw <= 0 or self.capacity_mwh <= 0:
            raise ValueError("power_mw and capacity_mwh must be positive")

    @property
    def soc_min_mwh(self) -> float:
        return self.soc_min_frac * self.capacity_mwh

    @property
    def soc_max_mwh(self) -> float:
        return self.soc_max_frac * self.capacity_mwh

    @property
    def eta_half(self) -> float:
        return math.sqrt(self.roundtrip_eff)


@dataclass
class StepResult:
    requested_grid_power_mw: float
    actual_grid_power_mw: float
    energy_to_grid_mwh: float   # +ve when discharging
    gross_revenue: float        # energy_to_grid_mwh * price
    throughput_mwh: float       # battery-side, for degradation accounting
    degradation_cost: float
    net_revenue: float
    soc_mwh: float
    clipped: bool


class BatterySimulator:
    def __init__(self, spec: BatterySpec) -> None:
        self.spec = spec
        self._soc_mwh = spec.initial_soc_frac * spec.capacity_mwh
        self._cum_throughput_mwh = 0.0

    @property
    def soc_mwh(self) -> float:
        return self._soc_mwh

    @property
    def cumulative_throughput_mwh(self) -> float:
        return self._cum_throughput_mwh

    def reset(self) -> None:
        self._soc_mwh = self.spec.initial_soc_frac * self.spec.capacity_mwh
        self._cum_throughput_mwh = 0.0

    def step(self, grid_power_mw: float, duration_h: float, price_per_mwh: float) -> StepResult:
        if duration_h <= 0:
            raise ValueError("duration_h must be positive")

        spec = self.spec
        p = float(grid_power_mw)
        clipped = False

        # Clip to power rating.
        if abs(p) > spec.power_mw:
            p = math.copysign(spec.power_mw, p)
            clipped = True

        eta = spec.eta_half

        # Compute SOC delta and clip to SOC bounds.
        if p > 0:  # charging
            soc_delta = p * duration_h * eta
            headroom = spec.soc_max_mwh - self._soc_mwh
            if soc_delta > headroom:
                soc_delta = max(0.0, headroom)
                p = soc_delta / (duration_h * eta) if duration_h * eta > 0 else 0.0
                clipped = True
        elif p < 0:  # discharging
            soc_delta = p * duration_h / eta  # negative
            floor = spec.soc_min_mwh - self._soc_mwh  # negative or zero
            if soc_delta < floor:
                soc_delta = min(0.0, floor)
                p = soc_delta * eta / duration_h if duration_h > 0 else 0.0
                clipped = True
        else:
            soc_delta = 0.0

        self._soc_mwh += soc_delta

        energy_to_grid = -p * duration_h  # +ve when discharging
        gross_revenue = energy_to_grid * price_per_mwh

        throughput = abs(soc_delta)  # battery-side energy moved
        self._cum_throughput_mwh += throughput
        degradation_cost = throughput * spec.degradation_cost_per_mwh

        net_revenue = gross_revenue - degradation_cost

        return StepResult(
            requested_grid_power_mw=float(grid_power_mw),
            actual_grid_power_mw=p,
            energy_to_grid_mwh=energy_to_grid,
            gross_revenue=gross_revenue,
            throughput_mwh=throughput,
            degradation_cost=degradation_cost,
            net_revenue=net_revenue,
            soc_mwh=self._soc_mwh,
            clipped=clipped,
        )
