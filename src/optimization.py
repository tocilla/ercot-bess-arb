"""Perfect-foresight dispatch via linear programming.

This is the CEILING (METHODOLOGY §5.2 / PLAN §6), not a baseline we try to
beat. Solves one LP per local day, given realized prices, with battery
physics and an explicit cycle cap. The resulting schedule answers:

    "If we knew every price in advance, what is the most revenue a battery
     with our spec — including efficiency, SOC bounds, cycle cap, and
     degradation cost — could possibly earn on this day?"

Every strategy we build is scored as `% of ceiling captured`.

LP formulation (per day):
    variables     p_c[t] >= 0   charge power (grid-side, MW)
                  p_d[t] >= 0   discharge power (grid-side, MW)
                  soc[t]        state of charge (MWh)
    maximize      sum(price[t] * (p_d[t] - p_c[t]) * dt)
                  - degradation * sum(eta_c * p_c[t] * dt + p_d[t] * dt / eta_d)
    subject to    soc[0] = initial_soc  (carried from previous day's sim state)
                  soc[t+1] = soc[t] + eta_c * p_c[t] * dt - p_d[t] * dt / eta_d
                  soc_min <= soc[t] <= soc_max
                  p_c[t], p_d[t] <= power_mw
                  sum(eta_c * p_c[t] * dt + p_d[t] * dt / eta_d)
                      <= 2 * cycles_per_day_cap * usable_mwh
                  (per-day total battery-side throughput — 2x usable per cycle)

SOC is carried across days by propagating the simulator state forward after
each day's LP. No closure constraint; the LP can legitimately drain initial
SOC on day 1 and leave the battery partially discharged at day-end.

We do NOT enforce complementarity p_c · p_d = 0 explicitly. With positive
prices and any round-trip loss, simultaneous charge+discharge is strictly
suboptimal, so the LP will not exercise it at an optimum. Verify ex-post.
"""

from __future__ import annotations

import logging
from math import sqrt

import cvxpy as cp
import numpy as np
import pandas as pd

from src.battery import BatterySpec

logger = logging.getLogger(__name__)


def _solve_day_lp(
    day_prices: np.ndarray,
    spec: BatterySpec,
    interval_hours: float,
    cycles_per_day_cap: float,
    initial_soc_mwh: float,
    solver: str,
) -> np.ndarray:
    """Solve one day. Returns grid_power_mw per interval (+ve = charge)."""
    n = len(day_prices)
    eta = sqrt(spec.roundtrip_eff)
    dt = interval_hours
    usable = (spec.soc_max_frac - spec.soc_min_frac) * spec.capacity_mwh

    p_c = cp.Variable(n, nonneg=True)
    p_d = cp.Variable(n, nonneg=True)

    # soc[t+1] - soc[t] = eta * p_c[t] * dt - p_d[t] * dt / eta
    delta = eta * p_c * dt - p_d * dt / eta
    soc_expr = initial_soc_mwh + cp.cumsum(delta)

    total_throughput = cp.sum(eta * p_c * dt + p_d * dt / eta)

    constraints = [
        p_c <= spec.power_mw,
        p_d <= spec.power_mw,
        soc_expr >= spec.soc_min_mwh,
        soc_expr <= spec.soc_max_mwh,
        # Per-day total battery-side throughput cap. 1 full cycle = 2 * usable
        # (charge side + discharge side).
        total_throughput <= 2 * cycles_per_day_cap * usable,
    ]

    revenue = day_prices @ ((p_d - p_c) * dt)
    objective = cp.Maximize(revenue - spec.degradation_cost_per_mwh * total_throughput)

    problem = cp.Problem(objective, constraints)
    problem.solve(solver=solver)

    if problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
        raise RuntimeError(f"LP failed: status={problem.status}")
    if problem.status == cp.OPTIMAL_INACCURATE:
        logger.warning("LP solved but status=OPTIMAL_INACCURATE")

    pc_val = np.maximum(p_c.value, 0.0)
    pd_val = np.maximum(p_d.value, 0.0)
    return pc_val - pd_val  # grid-side net (+ve = charging)


def perfect_foresight_schedule(
    prices: pd.Series,
    spec: BatterySpec,
    interval_hours: float,
    cycles_per_day_cap: float = 1.0,
    tz: str | None = None,
    solver: str = "HIGHS",
) -> pd.Series:
    """Produce a perfect-foresight grid_power_mw schedule aligned with `prices`.

    Per local day, solve an LP with cycle cap `cycles_per_day_cap`. Initial
    SOC for each day is `spec.initial_soc_frac * spec.capacity_mwh` — days
    are treated independently to match the natural-spread baseline convention
    in `daily_oracle_schedule`.
    """
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise TypeError("prices must have a DatetimeIndex")
    if prices.index.tz is None:
        raise ValueError("prices index must be tz-aware (UTC)")

    local_index = prices.index.tz_convert(tz) if tz else prices.index
    day_key = pd.Series(local_index.date, index=prices.index, name="day")

    schedule = pd.Series(0.0, index=prices.index, name="grid_power_mw")

    # Track the simulator's SOC across days to keep per-day LPs consistent.
    from src.battery import BatterySimulator
    sim_for_soc = BatterySimulator(spec)

    for day, idx in prices.groupby(day_key).groups.items():
        day_prices = prices.loc[idx].sort_index()
        if len(day_prices) < 2:
            continue
        try:
            day_schedule = _solve_day_lp(
                day_prices.to_numpy(dtype=float),
                spec,
                interval_hours,
                cycles_per_day_cap,
                initial_soc_mwh=sim_for_soc.soc_mwh,
                solver=solver,
            )
        except Exception as e:  # noqa: BLE001
            logger.error("LP failed on %s: %s", day, e)
            continue
        schedule.loc[day_prices.index] = day_schedule

        # Advance the SOC tracker through this day's schedule so the next
        # day's LP starts from the correct state.
        for p in day_schedule:
            sim_for_soc.step(float(p), interval_hours, 0.0)

    return schedule
