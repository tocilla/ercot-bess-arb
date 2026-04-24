"""LightGBM binary classifier for daily scarcity events.

Target: `target_scarcity` (max price that day > threshold).

Imbalance: scarcity is ~7.5% of days in ERCOT. We use `is_unbalance=True`
and L1-regularized L2 loss, and report PR-AUC (not ROC-AUC) and
precision-at-recall.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd


DEFAULT_PARAMS: dict[str, Any] = {
    "objective": "binary",
    "metric": "average_precision",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_data_in_leaf": 50,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "num_iterations": 400,
    "is_unbalance": True,
    "verbose": -1,
    "force_col_wise": True,
}


@dataclass
class ScarcityClassifier:
    params: dict[str, Any] = field(default_factory=lambda: dict(DEFAULT_PARAMS))
    model: lgb.Booster | None = None
    feature_cols: list[str] | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "ScarcityClassifier":
        self.feature_cols = list(X.columns)
        dtrain = lgb.Dataset(X.to_numpy(), label=y.to_numpy(),
                             feature_name=self.feature_cols)
        self.model = lgb.train(self.params, dtrain)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Returns probabilities (not binary labels)."""
        assert self.model is not None
        return self.model.predict(X[self.feature_cols].to_numpy())


def scarcity_fit_fn(X: pd.DataFrame, y: pd.Series) -> ScarcityClassifier:
    return ScarcityClassifier().fit(X, y)
