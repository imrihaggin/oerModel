"""Econometric and linear baseline models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV

from .base import ForecastModel

try:  # pragma: no cover - optional dependency
    from statsmodels.tsa.vector_ar.var_model import VAR
    HAS_STATSMODELS = True
except ImportError:  # pragma: no cover
    HAS_STATSMODELS = False


class LassoModel(ForecastModel):
    """Regularized linear baseline."""

    def __init__(self, name: str = "lasso", alphas: Optional[np.ndarray] = None, cv: int = 5, max_iter: int = 5000):
        super().__init__(name)
        self.alphas = alphas
        self.cv = cv
        self.max_iter = max_iter
        self.estimator = LassoCV(alphas=alphas, cv=cv, max_iter=max_iter, n_jobs=None)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LassoModel":
        self.estimator.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.estimator.predict(X)

    def get_params(self) -> Dict[str, Any]:
        return {
            "alpha": float(self.estimator.alpha_)
        }


@dataclass
class VARParams:
    """Hyperparameters for the VAR model."""

    lags: int = 6


class VARModel(ForecastModel):
    """Vector autoregression leveraging endogenous feature dynamics."""

    def __init__(self, name: str = "var", params: Optional[VARParams | Dict[str, Any]] = None):
        if not HAS_STATSMODELS:
            raise ImportError("statsmodels is required for VARModel")
        super().__init__(name)
        if params is None:
            self.params = VARParams()
        elif isinstance(params, dict):
            self.params = VARParams(**params)
        else:
            self.params = params
        self._fit_result = None
        self._train_frame: Optional[pd.DataFrame] = None
        self._target: Optional[str] = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "VARModel":
        self._target = y.name or "target"
        frame = pd.concat([y, X], axis=1)
        frame = frame.dropna()
        self._train_frame = frame
        model = VAR(frame)
        self._fit_result = model.fit(self.params.lags)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self._fit_result is None or self._train_frame is None:
            raise RuntimeError("Model must be fitted before calling predict")
        steps = len(X)
        history = self._train_frame.values[-self.params.lags :]
        forecast = self._fit_result.forecast(y=history, steps=steps)
        target_idx = self._train_frame.columns.get_loc(self._target)
        return forecast[:, target_idx]

    def get_params(self) -> Dict[str, Any]:
        return {"lags": self.params.lags}
