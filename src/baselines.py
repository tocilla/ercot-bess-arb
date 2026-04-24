"""Baseline dispatch schedulers.

These produce a grid_power_mw schedule aligned with a price series. Execution
is handled by `dispatch.run_dispatch`.

Included here:
    - daily_oracle_schedule: the "natural spread" floor (PLAN §6.1). Uses
      REALIZED prices to charge at each day's cheapest intervals and discharge
      at each day's most expensive. This is an oracle — not a strategy, but the
      reference floor for how much revenue lives in the raw daily spread.

Placeholders for later (§6.2, §6.3):
    - persistence, seasonal_naive: forecast-driven, not yet implemented.
"""

from __future__ import annotations

import pandas as pd

from src.battery import BatterySpec


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
