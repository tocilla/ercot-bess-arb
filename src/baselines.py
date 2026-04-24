"""Baseline dispatch schedulers.

These produce a grid_power_mw schedule aligned with a price series. Execution
is handled by `dispatch.run_dispatch`.

Included here:
    - daily_oracle_schedule: the "natural-spread" floor (PLAN §6.1). Uses
      REALIZED prices to charge at each day's cheapest intervals and discharge
      at each day's most expensive, up to a cycle budget. It is NOT a
      perfect-foresight optimum (no LP, no skip on unprofitable days); it is
      the mechanical reference floor that answers "how much revenue lives in
      the raw daily spread?" On flat-price days it can lose money by paying
      round-trip + degradation costs on a spread that doesn't cover them —
      that's intentional: the baseline is a floor, not a strategy.

    A true perfect-foresight dispatch (the ceiling) will be added later via
    LP, and unlike this baseline, it will skip cycles on unprofitable days.

Placeholders for later (§6.2, §6.3):
    - persistence, seasonal_naive: forecast-driven, not yet implemented.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.battery import BatterySpec, BatterySimulator


def daily_oracle_schedule(
    prices: pd.Series,
    spec: BatterySpec,
    interval_hours: float,
    cycles_per_day: float = 1.0,
    tz: str | None = None,
) -> pd.Series:
    """Produce a daily-oracle charge/discharge schedule.

    For each local calendar day, the schedule charges during the *k* cheapest
    intervals of that day and discharges during the *k* most expensive, sized
    so that total round-trip throughput does not exceed `cycles_per_day` full
    cycles. Uses REALIZED prices — this is a floor baseline, not a strategy.

    Args:
        prices: $/MWh indexed by UTC tz-aware timestamps.
        spec: battery spec (capacity, power, efficiency).
        interval_hours: length of each interval in hours.
        cycles_per_day: max full cycles per day (charge+discharge = 1 cycle).
        tz: local timezone for "day" grouping. Defaults to UTC if None.

    Returns:
        pd.Series of grid_power_mw, same index as prices. Sign convention:
        +ve = charge from grid, -ve = discharge to grid.
    """
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise TypeError("prices must have a DatetimeIndex")
    if prices.index.tz is None:
        raise ValueError("prices index must be tz-aware (UTC)")

    # Energy budget per day: cycles_per_day * usable_capacity, grid-side.
    usable_mwh = (spec.soc_max_frac - spec.soc_min_frac) * spec.capacity_mwh
    # Grid-side energy to fully charge: usable_mwh / eta_half. To fully discharge
    # an already-full battery: usable_mwh * eta_half. Per full cycle the grid
    # sees usable_mwh * (1/eta_half) on the way in and usable_mwh * eta_half
    # on the way out. Throttle by power rating at interval granularity.
    max_energy_per_interval = spec.power_mw * interval_hours  # grid-side, both directions

    # Group by local day.
    local_index = prices.index.tz_convert(tz) if tz else prices.index
    day_key = pd.Series(local_index.date, index=prices.index)

    schedule = pd.Series(0.0, index=prices.index, name="grid_power_mw")

    for _, idx in prices.groupby(day_key).groups.items():
        day_prices = prices.loc[idx].sort_index()

        # Charge budget (grid-side MWh) for this day — enough to move
        # cycles_per_day * usable_mwh of energy into the battery.
        charge_budget_grid_mwh = cycles_per_day * usable_mwh / spec.eta_half
        discharge_budget_grid_mwh = cycles_per_day * usable_mwh * spec.eta_half

        # Charge during cheapest intervals.
        ascending = day_prices.sort_values(kind="stable")
        remaining = charge_budget_grid_mwh
        for ts, _ in ascending.items():
            if remaining <= 0:
                break
            take = min(max_energy_per_interval, remaining)
            schedule.loc[ts] = take / interval_hours  # MW, +ve
            remaining -= take

        # Discharge during most-expensive intervals that were NOT used for charging.
        used = schedule.loc[day_prices.index] != 0
        descending = day_prices.loc[~used].sort_values(ascending=False, kind="stable")
        remaining = discharge_budget_grid_mwh
        for ts, _ in descending.items():
            if remaining <= 0:
                break
            take = min(max_energy_per_interval, remaining)
            schedule.loc[ts] = -take / interval_hours  # MW, -ve
            remaining -= take

    return schedule


def _simulate_day_net_revenue(
    schedule: pd.Series,
    prices: pd.Series,
    spec: BatterySpec,
    interval_hours: float,
    initial_soc_mwh: float,
) -> float:
    """Simulate a single-day schedule against realized prices and return net
    revenue. Used by gated variants to check ex-post profitability."""
    sim = BatterySimulator(spec)
    sim._soc_mwh = initial_soc_mwh  # type: ignore[attr-defined]
    total = 0.0
    for ts, p in schedule.items():
        step = sim.step(float(p), interval_hours, float(prices.loc[ts]))
        total += step.net_revenue
    return total


def daily_spread_gated_schedule(
    decision_prices: pd.Series,
    execution_prices: pd.Series,
    spec: BatterySpec,
    interval_hours: float,
    cycles_per_day: float = 1.0,
    tz: str | None = None,
) -> pd.Series:
    """Build a natural-spread schedule from `decision_prices`, then skip any
    day whose simulated net revenue (using `execution_prices`) is <= 0.

    With `decision_prices = execution_prices`, this is a *realized-prices*
    gate — removes the days where the baseline loses money to efficiency /
    degradation (METHODOLOGY §5.2, FINDINGS 2026-04-24: this is the simple
    improvement that closes much of the floor-to-ceiling gap).

    With `decision_prices = some_forecast`, this is a *forecast-driven*
    gate — the operator abstains when the forecast spread doesn't cover costs,
    even if realized prices would have been profitable.

    Note: this uses a fresh BatterySimulator per day starting at
    `spec.initial_soc_frac` — day-level checks only. Carry-over is handled
    by `run_dispatch` downstream.
    """
    base = daily_oracle_schedule(
        decision_prices, spec, interval_hours, cycles_per_day=cycles_per_day, tz=tz
    )

    local_index = execution_prices.index.tz_convert(tz) if tz else execution_prices.index
    day_key = pd.Series(local_index.date, index=execution_prices.index)

    initial_soc_mwh = spec.initial_soc_frac * spec.capacity_mwh
    gated = base.copy()

    for _, idx in execution_prices.groupby(day_key).groups.items():
        day_sched = base.loc[idx]
        if (day_sched == 0).all():
            continue
        net = _simulate_day_net_revenue(
            day_sched, execution_prices.loc[idx], spec, interval_hours, initial_soc_mwh
        )
        if net <= 0:
            gated.loc[idx] = 0.0

    return gated
