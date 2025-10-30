"""Reporting helpers for evaluation outputs."""

from __future__ import annotations

import pandas as pd


def aggregate_backtest_results(results: pd.DataFrame) -> pd.DataFrame:
    """Aggregate rolling backtest metrics by model."""
    metrics = [col for col in ["rmse", "mae", "mape"] if col in results.columns]
    return results.groupby("model")[metrics].mean().sort_values("rmse")


def prepare_dashboard_frame(results: pd.DataFrame) -> pd.DataFrame:
    """Reshape backtest results for plotting."""
    pivot = results.pivot_table(index="timestamp", columns="model", values="y_pred")
    pivot["actual"] = results.drop_duplicates("timestamp").set_index("timestamp")["y_true"]
    return pivot.sort_index()
