"""Model registry and base classes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd


class ForecastModel(ABC):
    """Abstract base class for forecasting models."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "ForecastModel":
        """Train the model."""

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate predictions."""

    def save(self, path: Path) -> Path:
        """Persist the fitted model."""
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        return path

    @classmethod
    def load(cls, path: Path) -> "ForecastModel":
        """Load a persisted model."""
        return joblib.load(path)

    def get_params(self) -> Dict[str, Any]:
        """Return model parameters for logging."""
        return {}
