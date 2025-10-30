"""Ensembling utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd


def equal_weight_average(forecasts: dict[str, pd.Series]) -> pd.Series:
    """Return an equal-weight ensemble of forecasts."""
    if not forecasts:
        raise ValueError("No forecasts provided")
    aligned = pd.concat(forecasts.values(), axis=1)
    aligned.columns = list(forecasts.keys())
    return aligned.mean(axis=1)
