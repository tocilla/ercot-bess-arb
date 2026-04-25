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

    def fit(self, X: pd.DataFrame, y: pd.Series,
            sample_weight: np.ndarray | None = None) -> "LGBMForecaster":
        self.feature_cols = list(X.columns)
        dtrain = lgb.Dataset(
            X.to_numpy(), label=y.to_numpy(),
            weight=sample_weight, feature_name=self.feature_cols,
        )
        self.model = lgb.train(self.params, dtrain)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        assert self.model is not None, "call fit() first"
        return self.model.predict(X[self.feature_cols].to_numpy())


def lgbm_fit_fn(X: pd.DataFrame, y: pd.Series) -> LGBMForecaster:
    """Fit factory usable with `walk_forward_predict`."""
    return LGBMForecaster().fit(X, y)


def make_quantile_fit_fn(
    alpha: float,
    num_iterations: int = 300,
    seed: int | None = None,
    weight_fn: "Callable[[pd.DataFrame, pd.Series], np.ndarray] | None" = None,
):
    """Return a fit_fn that trains an LGBM with quantile objective at
    quantile `alpha` (0 < alpha < 1). Use alpha=0.5 for median, 0.1 for
    lower tail, 0.9 for upper tail. Optional `seed` sets deterministic
    bagging / feature-sampling for seed-stability analysis. Optional
    `weight_fn(X, y) → np.ndarray` produces per-sample weights for
    decision-aware loss shaping (e.g. price-weighted MAE).
    """
    params = dict(DEFAULT_PARAMS)
    params["objective"] = "quantile"
    params["alpha"] = alpha
    params["num_iterations"] = num_iterations
    if seed is not None:
        params["seed"] = seed
        params["bagging_seed"] = seed
        params["feature_fraction_seed"] = seed
        params["data_random_seed"] = seed
        params["deterministic"] = True

    def fit_fn(X: pd.DataFrame, y: pd.Series) -> LGBMForecaster:
        weights = None
        if weight_fn is not None:
            weights = weight_fn(X, y)
            # LGBM requires non-negative weights summing to > 0.
            weights = np.maximum(weights, 0.0)
            if weights.sum() == 0:
                raise ValueError("weight_fn returned all-zero weights")
        return LGBMForecaster(params=dict(params)).fit(X, y, sample_weight=weights)

    suffix = f"_s{seed}" if seed is not None else ""
    weight_tag = f"_w{weight_fn.__name__}" if weight_fn is not None else ""
    fit_fn.__name__ = f"lgbm_q{int(alpha * 100):02d}{suffix}{weight_tag}_fit_fn"
    return fit_fn
