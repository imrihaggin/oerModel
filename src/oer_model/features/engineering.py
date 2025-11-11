"""Feature engineering pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from ..config import AppConfig
from ..utils.io import ensure_directory, read_dataframe, write_dataframe
from ..utils.logging import get_logger

LOGGER = get_logger(__name__)


VALID_OPS = {"pct_change", "diff", "lag", "rolling_mean", "rolling_std"}


@dataclass
class FeatureOp:
    """Sequential operation description."""

    type: str
    periods: int = 1
    alias: Optional[str] = None
    window: Optional[int] = None
    center: bool = False

    @classmethod
    def from_dict(cls, payload: Dict[str, object]) -> "FeatureOp":
        op_type = payload.get("type")
        if op_type not in VALID_OPS:
            raise ValueError(f"Unsupported operation: {op_type}")
        return cls(
            type=str(op_type),
            periods=int(payload.get("periods", 1)),
            alias=payload.get("alias"),
            window=payload.get("window"),
            center=bool(payload.get("center", False)),
        )


@dataclass
class FeatureRecipe:
    """Recipe describing how to derive features from a source column."""

    source: str
    operations: List[FeatureOp]
    set_as_target: bool = False

    @classmethod
    def from_dict(cls, payload: Dict[str, object]) -> "FeatureRecipe":
        ops = [FeatureOp.from_dict(op) for op in payload.get("operations", [])]
        return cls(
            source=str(payload["source"]),
            operations=ops,
            set_as_target=bool(payload.get("set_as_target", False)),
        )


def _apply_operation(series: pd.Series, op: FeatureOp) -> pd.Series:
    if op.type == "pct_change":
        result = series.pct_change(op.periods) * 100.0
    elif op.type == "diff":
        result = series.diff(op.periods)
    elif op.type == "lag":
        result = series.shift(op.periods)
    elif op.type == "rolling_mean":
        if op.window is None:
            raise ValueError("rolling_mean requires 'window'")
        result = series.rolling(window=op.window, center=op.center).mean()
    elif op.type == "rolling_std":
        if op.window is None:
            raise ValueError("rolling_std requires 'window'")
        result = series.rolling(window=op.window, center=op.center).std()
    else:
        raise ValueError(f"Unknown operation {op.type}")
    if op.alias:
        result.name = op.alias
    return result


def build_feature_matrix(config: AppConfig, raw_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Generate the processed feature matrix."""
    if raw_df is None:
        raw_path = config.raw_dir / "raw_combined.csv"
        raw_df = read_dataframe(raw_path, parse_dates=[0])
        raw_df.index = pd.to_datetime(raw_df.index)
    raw_df = raw_df.sort_index().asfreq("M")
    
    # Forward-fill missing values to maximize data availability
    # This allows features to be computed even when some series start later
    raw_df = raw_df.fillna(method='ffill').fillna(method='bfill')

    recipes = [FeatureRecipe.from_dict(recipe) for recipe in config.features.get("recipes", [])]
    if not recipes:
        raise RuntimeError("No feature recipes configured.")

    engineered = pd.DataFrame(index=raw_df.index)
    target_col: Optional[str] = None

    for recipe in recipes:
        if recipe.source not in raw_df.columns:
            LOGGER.warning("Source column %s not found; skipping", recipe.source)
            continue
        current_series = raw_df[recipe.source]
        for op in recipe.operations:
            transformed = _apply_operation(current_series, op)
            alias = transformed.name or f"{recipe.source}_{op.type}_{op.periods}"
            engineered[alias] = transformed
            current_series = transformed
        if recipe.set_as_target:
            target_col = transformed.name

    drop_na = config.features.get("drop_na", True)
    if drop_na:
        engineered.dropna(inplace=True)
    
    # Replace any infinite values with NaN, then forward-fill
    engineered.replace([np.inf, -np.inf], np.nan, inplace=True)
    engineered.fillna(method='ffill', inplace=True)
    engineered.fillna(method='bfill', inplace=True)
    
    # If any NaN/inf remain, drop those rows
    if engineered.isna().any().any() or np.isinf(engineered.select_dtypes(include=[np.number])).any().any():
        LOGGER.warning("Dropping rows with remaining NaN or inf values")
        engineered = engineered.replace([np.inf, -np.inf], np.nan).dropna()
    
    ensure_directory(config.processed_dir)
    processed_path = config.processed_dir / "features_processed.csv"
    write_dataframe(engineered, processed_path)
    LOGGER.info("Saved processed features to %s", processed_path)
    if target_col and target_col not in engineered.columns:
        raise RuntimeError("Configured target column missing from processed features.")
    return engineered
