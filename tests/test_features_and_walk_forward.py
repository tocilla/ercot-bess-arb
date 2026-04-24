"""Tests for feature engineering and the walk-forward harness.

Leakage tests (CRITICAL):
    - Every feature at time t must depend only on data strictly before t.
    - Walk-forward fit at boundary b must not see any row with timestamp >= b.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluation import _retrain_boundaries, walk_forward_predict
from src.features import build_features, feature_columns


def _make_prices(n_days: int = 90, seed: int = 0) -> pd.Series:
    idx = pd.date_range("2024-01-01", periods=n_days * 96, freq="15min", tz="UTC")
    rng = np.random.default_rng(seed)
    hours = idx.hour + idx.minute / 60.0
    shape = 25 * np.sin(2 * np.pi * (hours - 10) / 24)
    noise = rng.normal(0, 3, size=len(idx))
    return pd.Series(40 + shape + noise, index=idx, name="lmp")


def test_features_no_future_leak():
    """Modifying a single future price must NOT change any present-or-past
    feature value."""
    prices = _make_prices(30)
    feats = build_features(prices, tz="US/Central")
    # Pick a mid-series timestamp.
    t = prices.index[1000]

    prices2 = prices.copy()
    # Modify every value strictly after t.
    prices2.loc[prices2.index > t] = prices2.loc[prices2.index > t] + 10_000
    feats2 = build_features(prices2, tz="US/Central")

    cols = feature_columns(feats)
    for col in cols:
        diff = (feats.loc[:t, col].fillna(0) - feats2.loc[:t, col].fillna(0)).abs().max()
        assert diff == 0, f"feature '{col}' leaked future data at t={t}"


def test_retrain_boundaries():
    start = pd.Timestamp("2024-01-01", tz="UTC")
    end = pd.Timestamp("2024-03-15", tz="UTC")
    bs = _retrain_boundaries(start, end, retrain_every_days=30)
    assert bs[0] == start
    assert bs[-1] <= end
    assert (pd.Series(bs).diff().dropna() == pd.Timedelta(days=30)).all()


def test_walk_forward_respects_chronology():
    """The harness must never fit on a row dated >= its training boundary."""
    prices = _make_prices(90)
    feats = build_features(prices, tz="US/Central")

    seen_max_train_idx: list[pd.Timestamp] = []

    def recording_fit(X, y):
        seen_max_train_idx.append(X.index.max())
        class Const:
            def predict(self, X):
                return np.full(len(X), y.mean())
        return Const()

    test_start = prices.index[30 * 96]
    test_end = prices.index[-1]

    preds = walk_forward_predict(
        feats, "target", recording_fit,
        test_start=test_start, test_end=test_end,
        retrain_every_days=7, min_train_rows=100,
    )

    boundaries = _retrain_boundaries(test_start, test_end, 7)
    for b, max_t in zip(boundaries, seen_max_train_idx):
        assert max_t < b, f"walk-forward leaked: max train {max_t} >= boundary {b}"

    # Predictions exist over the test window, NaN before.
    assert preds.loc[preds.index >= test_start].notna().sum() > 0
    assert preds.loc[preds.index < test_start].isna().all()
