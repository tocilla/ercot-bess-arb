"""Walk-forward evaluation harness (METHODOLOGY §2).

Given a feature frame and a fit/predict interface, produce out-of-sample
predictions across a test window by:

    1. Re-fitting the model at fixed retraining boundaries (e.g. monthly),
       using all data strictly before the boundary.
    2. Using the most recently fit model to predict for each interval
       in [boundary, next_boundary).

This respects chronological ordering and the feature-availability rule:
no model predicts using data that post-dates its fit time.
"""

from __future__ import annotations

from typing import Callable, Protocol

import numpy as np
import pandas as pd


class FittedModel(Protocol):
    def predict(self, X: pd.DataFrame) -> np.ndarray: ...


FitFn = Callable[[pd.DataFrame, pd.Series], FittedModel]


def _retrain_boundaries(
    start: pd.Timestamp, end: pd.Timestamp, retrain_every_days: int
) -> list[pd.Timestamp]:
    """Inclusive list of boundary timestamps: start, start+k, start+2k, ..."""
    boundaries = []
    t = start
    step = pd.Timedelta(days=retrain_every_days)
    while t <= end:
        boundaries.append(t)
        t = t + step
    return boundaries


def walk_forward_predict(
    features: pd.DataFrame,
    target_col: str,
    fit_fn: FitFn,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
    retrain_every_days: int = 30,
    min_train_rows: int = 1000,
) -> pd.Series:
    """Produce a walk-forward prediction series over [test_start, test_end].

    Args:
        features: index = tz-aware timestamp, includes `target_col`.
        target_col: name of the target column.
        fit_fn: (X_train, y_train) -> fitted model implementing .predict(X).
        test_start, test_end: inclusive start, inclusive end of the test window.
        retrain_every_days: re-fit the model every N days walking forward.
        min_train_rows: raise if at any boundary training rows < this.

    Returns:
        pd.Series of predictions indexed by feature index, valued only in
        the test window (NaN elsewhere).
    """
    if features.index.tz is None:
        raise ValueError("features index must be tz-aware")
    if not features.index.is_monotonic_increasing:
        features = features.sort_index()

    feature_cols = [c for c in features.columns if c != target_col]
    predictions = pd.Series(np.nan, index=features.index, name="prediction")

    # Drop rows with any NaN in features or target — lags create warmup NaN.
    clean_mask = features.notna().all(axis=1)

    boundaries = _retrain_boundaries(test_start, test_end, retrain_every_days)

    for i, boundary in enumerate(boundaries):
        next_boundary = (
            boundaries[i + 1] if i + 1 < len(boundaries)
            else test_end + pd.Timedelta(seconds=1)
        )

        # Train on everything strictly before `boundary` AND with usable features.
        train_mask = (features.index < boundary) & clean_mask
        if train_mask.sum() < min_train_rows:
            raise RuntimeError(
                f"At boundary {boundary}: only {train_mask.sum()} training rows "
                f"(need >= {min_train_rows}). Check warmup period."
            )
        X_train = features.loc[train_mask, feature_cols]
        y_train = features.loc[train_mask, target_col]
        model = fit_fn(X_train, y_train)

        # Predict on [boundary, next_boundary) with clean feature rows.
        predict_mask = (
            (features.index >= boundary) & (features.index < next_boundary) & clean_mask
        )
        if predict_mask.any():
            X_pred = features.loc[predict_mask, feature_cols]
            predictions.loc[X_pred.index] = model.predict(X_pred)

    return predictions
