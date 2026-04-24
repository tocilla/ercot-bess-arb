"""Scarcity-aware dispatch: on days flagged as likely-scarcity, use a
'hold then dump' schedule that sits at full SOC through the morning and
discharges into the historical peak window. On other days, use a
fallback schedule (e.g. LGBM forecast-driven).

This is deliberately simple — just enough to test whether the classifier
output is actionable. A more sophisticated version would learn the
optimal hold window from history.
"""

from __future__ import annotations

import pandas as pd

from src.battery import BatterySpec


def scarcity_rule_schedule(
    prices_index: pd.DatetimeIndex,
    spec: BatterySpec,
    interval_hours: float,
    tz: str,
    charge_hours: tuple[int, int] = (1, 7),        # 01:00–07:00 local
    discharge_hours: tuple[int, int] = (15, 21),   # 15:00–21:00 local
) -> pd.Series:
    """A fixed-window rule schedule for days we expect to spike: charge
    overnight at full power, hold, discharge into the afternoon peak."""
    local = prices_index.tz_convert(tz)
    hour_of_day = local.hour + local.minute / 60.0
    is_charge = (hour_of_day >= charge_hours[0]) & (hour_of_day < charge_hours[1])
    is_discharge = (hour_of_day >= discharge_hours[0]) & (hour_of_day < discharge_hours[1])

    schedule = pd.Series(0.0, index=prices_index, name="grid_power_mw")
    schedule[is_charge] = spec.power_mw        # +ve = charge
    schedule[is_discharge] = -spec.power_mw    # -ve = discharge
    return schedule


def combined_schedule(
    fallback: pd.Series,
    scarcity_prob: pd.Series,      # one value per LOCAL DATE
    threshold: float,
    prices_index: pd.DatetimeIndex,
    spec: BatterySpec,
    interval_hours: float,
    tz: str,
) -> pd.Series:
    """Compose a per-interval schedule by picking the scarcity-rule schedule
    for days whose predicted scarcity probability exceeds `threshold`, and
    `fallback` for all other days.

    Args:
        fallback: grid_power_mw schedule aligned with prices_index.
        scarcity_prob: one probability per local calendar date.
        threshold: decision threshold (e.g. 0.5, tuned on validation).
    """
    local = prices_index.tz_convert(tz)
    day_labels = pd.Series(local.date, index=prices_index)

    rule_sched = scarcity_rule_schedule(prices_index, spec, interval_hours, tz)

    combined = fallback.copy()
    for day, idx in fallback.groupby(day_labels).groups.items():
        prob = scarcity_prob.get(day)
        if prob is not None and prob > threshold:
            combined.loc[idx] = rule_sched.loc[idx]
    return combined
