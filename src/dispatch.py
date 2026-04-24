"""Run a dispatch schedule through the battery simulator and return per-interval
results as a DataFrame.

A "schedule" is a sequence of (timestamp, grid_power_mw) rows aligned with a
price series. The battery simulator enforces physical constraints; if the
scheduler requested an infeasible action it will be clipped and the result
will record `clipped=True`.
"""

from __future__ import annotations

import pandas as pd

from src.battery import BatterySimulator, BatterySpec


def run_dispatch(
    schedule: pd.Series,
    prices: pd.Series,
    spec: BatterySpec,
    interval_hours: float,
) -> pd.DataFrame:
    """Step the battery through `schedule` against `prices`.

    Args:
        schedule: grid_power_mw per interval, indexed by UTC timestamp.
        prices:   $/MWh per interval, indexed by UTC timestamp.
        spec:     battery specification.
        interval_hours: length of each interval in hours (e.g. 5/60 for 5-min).

    Returns:
        DataFrame indexed by timestamp with columns:
        requested_mw, actual_mw, energy_to_grid_mwh, price,
        gross_revenue, throughput_mwh, degradation_cost, net_revenue,
        soc_mwh, clipped.
    """
    if not schedule.index.equals(prices.index):
        raise ValueError("schedule and prices must share the same index")

    battery = BatterySimulator(spec)
    rows: list[dict] = []
    for ts, p_grid in schedule.items():
        price = float(prices.loc[ts])
        step = battery.step(float(p_grid), interval_hours, price)
        rows.append(
            {
                "timestamp": ts,
                "requested_mw": step.requested_grid_power_mw,
                "actual_mw": step.actual_grid_power_mw,
                "energy_to_grid_mwh": step.energy_to_grid_mwh,
                "price": price,
                "gross_revenue": step.gross_revenue,
                "throughput_mwh": step.throughput_mwh,
                "degradation_cost": step.degradation_cost,
                "net_revenue": step.net_revenue,
                "soc_mwh": step.soc_mwh,
                "clipped": step.clipped,
            }
        )
    return pd.DataFrame(rows).set_index("timestamp")
