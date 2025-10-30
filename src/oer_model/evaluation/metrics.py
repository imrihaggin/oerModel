"""Evaluation metrics for forecasting."""

from __future__ import annotations

import numpy as np
import pandas as pd


def rmse(y_true, y_pred) -> float:
    """Root mean squared error."""
    return float(np.sqrt(np.mean((np.array(y_true) - np.array(y_pred)) ** 2)))


def mae(y_true, y_pred) -> float:
    """Mean absolute error."""
    return float(np.mean(np.abs(np.array(y_true) - np.array(y_pred))))


def mape(y_true, y_pred) -> float:
    """Mean absolute percentage error."""
    y_true_arr = np.array(y_true)
    eps = np.finfo(float).eps
    return float(np.mean(np.abs((y_true_arr - np.array(y_pred)) / (y_true_arr + eps))) * 100)


def summarize_errors(y_true: pd.Series, y_pred: pd.Series) -> pd.Series:
    """Return standard error metrics as a Series."""
    return pd.Series({
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "mape": mape(y_true, y_pred),
    })
