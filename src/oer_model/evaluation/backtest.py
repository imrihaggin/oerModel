"""Rolling-origin backtesting utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Tuple

import pandas as pd

from ..models.base import ForecastModel
from ..utils.logging import get_logger
from .metrics import summarize_errors

LOGGER = get_logger(__name__)


@dataclass
class BacktestConfig:
    """Parameters controlling the rolling evaluation."""

    train_window: int
    test_window: int
    step: int = 1
    max_windows: Optional[int] = None
    log_frequency: int = 1


def rolling_windows(index: Iterable[pd.Timestamp], cfg: BacktestConfig) -> List[Tuple[int, int, int, int]]:
    """Return index bounds for rolling windows.

    Each tuple is (train_start, train_end, test_start, test_end) inclusive indices.
    """
    n = len(index)
    windows: List[Tuple[int, int, int, int]] = []
    for train_end in range(cfg.train_window - 1, n - cfg.test_window, cfg.step):
        train_start = train_end - cfg.train_window + 1
        test_start = train_end + 1
        test_end = test_start + cfg.test_window - 1
        windows.append((train_start, train_end, test_start, test_end))
        if cfg.max_windows is not None and len(windows) >= cfg.max_windows:
            break
    return windows


def run_backtest(
    data: pd.DataFrame,
    target_column: str,
    model_factory: Callable[[], ForecastModel],
    cfg: BacktestConfig,
) -> pd.DataFrame:
    """Run a walk-forward validation returning predictions and metrics."""
    if target_column not in data.columns:
        raise KeyError(f"Target column {target_column} missing from dataset")

    index = data.index.to_list()
    splits = rolling_windows(index, cfg)
    if not splits:
        raise RuntimeError("Backtest configuration produced no splits")

    results = []
    for window_idx, (train_start, train_end, test_start, test_end) in enumerate(splits):
        train = data.iloc[train_start:train_end + 1]
        test = data.iloc[test_start:test_end + 1]
        model = model_factory()
        X_train = train.drop(columns=[target_column])
        y_train = train[target_column]
        X_test = test.drop(columns=[target_column])
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        y_true = test[target_column]
        metrics = summarize_errors(y_true, preds)
        for ts, true_val, pred_val in zip(y_true.index, y_true.values, preds):
            results.append({
                "timestamp": ts,
                "y_true": true_val,
                "y_pred": pred_val,
                "window_train_end": index[train_end],
                **metrics.to_dict(),
                "model": model.name,
            })
        should_log = (
            cfg.log_frequency <= 1
            or (window_idx + 1) % cfg.log_frequency == 0
            or window_idx == len(splits) - 1
        )
        if should_log:
            LOGGER.info("Backtest window ending %s RMSE %.4f", index[train_end], metrics["rmse"])
    return pd.DataFrame(results)
