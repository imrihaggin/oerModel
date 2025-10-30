"""Custom transformers for feature processing."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class DateTimeFeatures(BaseEstimator, TransformerMixin):
    """Create calendar-based features from a DatetimeIndex."""

    def __init__(self, month: bool = True, quarter: bool = True):
        self.month = month
        self.quarter = quarter

    def fit(self, X: pd.DataFrame, y=None):  # noqa: D401 - scikit-learn signature
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X.index, pd.DatetimeIndex):
            raise ValueError("DateTimeFeatures requires a DatetimeIndex")
        feats = {}
        if self.month:
            feats["month_sin"] = np.sin(2 * np.pi * X.index.month / 12)
            feats["month_cos"] = np.cos(2 * np.pi * X.index.month / 12)
        if self.quarter:
            feats["quarter"] = X.index.quarter
        return pd.DataFrame(feats, index=X.index)
