"""Gradient boosted tree model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

try:  # pragma: no cover - optional dependency
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:  # pragma: no cover
    HAS_XGBOOST = False

from .base import ForecastModel


@dataclass
class XGBParams:
    """Hyperparameters for XGBoost."""

    objective: str = "reg:squarederror"
    n_estimators: int = 500
    learning_rate: float = 0.05
    max_depth: int = 4
    subsample: float = 0.9
    colsample_bytree: float = 0.9
    gamma: float = 0.0
    min_child_weight: int = 1
    reg_lambda: float = 1.0
    reg_alpha: float = 0.0
    random_state: int = 42


class XGBoostModel(ForecastModel):
    """Wrapper around the xgboost regressor."""

    def __init__(self, name: str = "xgboost", params: Optional[XGBParams | Dict[str, Any]] = None):
        if not HAS_XGBOOST:
            raise ImportError("xgboost is required for XGBoostModel")
        super().__init__(name)
        if params is None:
            self.params = XGBParams()
        elif isinstance(params, dict):
            self.params = XGBParams(**params)
        else:
            self.params = params
        self.estimator = xgb.XGBRegressor(**self.params.__dict__)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "XGBoostModel":
        self.estimator.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.estimator.predict(X)

    def get_params(self) -> Dict[str, Any]:
        return self.estimator.get_xgb_params()
