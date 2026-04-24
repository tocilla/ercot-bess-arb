"""LightGBM regression wrapper compatible with the walk-forward harness.

Intentionally minimal: point forecast, MAE objective (robust to ERCOT's
heavy tails), a small fixed set of hyperparameters. We do not tune on the
validation set in phase 1 — the goal is a clean, reproducible first
ML number, not a squeezed leaderboard score.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd


DEFAULT_PARAMS: dict[str, Any] = {
    "objective": "regression_l1",   # MAE — robust to scarcity tails
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_data_in_leaf": 200,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.9,
    "bagging_freq": 5,
    "num_iterations": 500,
    "verbose": -1,
    "force_col_wise": True,
}


@dataclass
class LGBMForecaster:
    params: dict[str, Any] = field(default_factory=lambda: dict(DEFAULT_PARAMS))
    model: lgb.Booster | None = None
    feature_cols: list[str] | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LGBMForecaster":
        self.feature_cols = list(X.columns)
        dtrain = lgb.Dataset(X.to_numpy(), label=y.to_numpy(), feature_name=self.feature_cols)
        self.model = lgb.train(self.params, dtrain)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        assert self.model is not None, "call fit() first"
        return self.model.predict(X[self.feature_cols].to_numpy())


def lgbm_fit_fn(X: pd.DataFrame, y: pd.Series) -> LGBMForecaster:
    """Fit factory usable with `walk_forward_predict`."""
    return LGBMForecaster().fit(X, y)
