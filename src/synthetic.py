"""Synthetic price generator for tests and smoke runs.

Deliberately simple — a daily sinusoidal shape plus noise, optionally with
occasional scarcity spikes. Enough to exercise the pipeline; not a substitute
for real ISO data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def synthetic_lmp(
    start: str = "2024-01-01",
    days: int = 30,
    interval_minutes: int = 60,
    mean_price: float = 40.0,
    daily_amplitude: float = 25.0,
    noise_std: float = 3.0,
    scarcity_prob: float = 0.0,
    scarcity_multiplier: float = 20.0,
    seed: int = 0,
    tz: str = "UTC",
) -> pd.Series:
    """Generate a synthetic LMP series with daily seasonality."""
    rng = np.random.default_rng(seed)
    n_per_day = int(24 * 60 / interval_minutes)
    n = n_per_day * days
    idx = pd.date_range(start=start, periods=n, freq=f"{interval_minutes}min", tz=tz)

    # Phase: peak around hour 18, trough around hour 4 (shift accordingly).
    hour_of_day = idx.hour + idx.minute / 60.0
    seasonal = daily_amplitude * np.sin(2 * np.pi * (hour_of_day - 10) / 24)
    noise = rng.normal(0.0, noise_std, size=n)
    price = mean_price + seasonal + noise

    if scarcity_prob > 0:
        spikes = rng.random(n) < scarcity_prob
        price[spikes] = mean_price * scarcity_multiplier + rng.normal(0, 50, size=spikes.sum())

    return pd.Series(price, index=idx, name="lmp")
