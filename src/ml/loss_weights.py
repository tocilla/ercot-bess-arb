"""Per-sample loss weights for decision-aware training.

When fitting a price forecaster whose downstream use is dispatch, the
sample-level MAE is a poor proxy for revenue. These weight functions
overweight intervals that drive dispatch decisions.

Weight functions take (X, y) and return a non-negative weight array
the same length as y. Higher weight = more loss penalty on that row.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


INTERVALS_PER_DAY = 96  # 15-min


def _local_date_key(timestamps: pd.DatetimeIndex, tz: str = "US/Central") -> np.ndarray:
    """Local calendar date as a string key, useful for daily groupby."""
    local = timestamps.tz_convert(tz) if timestamps.tz else timestamps
    return np.array([d.isoformat() for d in local.date])


def deviation_from_daily_mean(X: pd.DataFrame, y: pd.Series,
                              tz: str = "US/Central",
                              floor: float = 1.0) -> np.ndarray:
    """Weight ∝ |y − daily_mean(y)|. Smooth, peaks-and-troughs both
    overweighted, middle hours underweighted. Useful starting point.

    `floor` is added to every weight so no row collapses to zero.
    """
    key = _local_date_key(y.index, tz=tz)
    df = pd.DataFrame({"y": y.values, "k": key})
    daily_mean = df.groupby("k")["y"].transform("mean")
    w = np.abs(df["y"].to_numpy() - daily_mean.to_numpy()) + floor
    return w
deviation_from_daily_mean.__name__ = "devmean"


def top_bottom_k_per_day(X: pd.DataFrame, y: pd.Series,
                          tz: str = "US/Central",
                          k: int = 8,
                          weight_decision: float = 5.0,
                          weight_other: float = 1.0) -> np.ndarray:
    """Hard mask: per local day, mark the k cheapest and k most-expensive
    intervals as 'decision intervals' and weight them `weight_decision`;
    the rest get `weight_other`. With INTERVALS_PER_DAY=96, k=8 means
    16/96 ≈ 17% of rows carry the elevated weight.
    """
    key = _local_date_key(y.index, tz=tz)
    n = len(y)
    mask = np.zeros(n, dtype=bool)
    df = pd.DataFrame({"y": y.values, "k": key, "i": np.arange(n)})
    for _, group in df.groupby("k"):
        idx = group["i"].to_numpy()
        prices = group["y"].to_numpy()
        ord_ = np.argsort(prices)
        # Lowest k and highest k per day.
        mask[idx[ord_[:k]]] = True
        mask[idx[ord_[-k:]]] = True
    w = np.where(mask, weight_decision, weight_other).astype(float)
    return w
top_bottom_k_per_day.__name__ = "topbotk"


def price_magnitude(X: pd.DataFrame, y: pd.Series,
                     scale: float = 100.0) -> np.ndarray:
    """Weight ∝ 1 + |y| / scale. Naturally heavier on high-price hours
    (scarcity) and on negative-price hours. Doesn't need a daily groupby.
    """
    return 1.0 + np.abs(y.to_numpy()) / scale
price_magnitude.__name__ = "pricemag"
