"""Model training, evaluation, and forecasting orchestrator."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, Callable, Dict

import pandas as pd

from ..config import AppConfig
from ..evaluation.backtest import BacktestConfig, run_backtest
from ..evaluation.reporting import aggregate_backtest_results
from ..features.engineering import build_feature_matrix
from ..models.base import ForecastModel
from ..utils.io import ensure_directory, read_dataframe, write_dataframe
from ..utils.logging import get_logger

LOGGER = get_logger(__name__)


def _load_processed_features(config: AppConfig) -> pd.DataFrame:
    path = config.processed_dir / "features_processed.csv"
    if not path.exists():
        LOGGER.info("Processed features missing; building now")
        return build_feature_matrix(config)
    df = read_dataframe(path, parse_dates=[0])
    df.index = pd.to_datetime(df.index)
    return df


def _import_class(path: str) -> type[ForecastModel]:
    module_name, class_name = path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def _build_model_factory(model_spec: Dict[str, Any]) -> Callable[[], ForecastModel]:
    class_path = model_spec["class"]
    params = model_spec.get("params", {})
    model_cls = _import_class(class_path)

    def factory() -> ForecastModel:
        return model_cls(**params)

    return factory


def run_training_pipeline(config: AppConfig) -> Dict[str, pd.DataFrame]:
    """Train and evaluate all configured models."""
    features = _load_processed_features(config)
    target_column = config.features.get("target_column")
    if not target_column:
        raise RuntimeError("features.target_column must be defined in config")
    if target_column not in features.columns:
        raise KeyError(f"Target column {target_column} not present in processed features")

    backtest_cfg = config.backtest or {}
    bt = BacktestConfig(
        train_window=backtest_cfg.get("train_window", 120),
        test_window=backtest_cfg.get("test_window", 6),
        step=backtest_cfg.get("step", 1),
    )

    artifacts_dir = ensure_directory(config.artifacts_dir)
    model_outputs: Dict[str, pd.DataFrame] = {}

    for model_name, model_cfg in config.models.get("candidates", {}).items():
        if not model_cfg.get("enabled", False):
            LOGGER.info("Skipping disabled model %s", model_name)
            continue
        LOGGER.info("Running backtest for model %s", model_name)
        model_factory = _build_model_factory(model_cfg)
        results = run_backtest(features, target_column, model_factory, bt)
        model_outputs[model_name] = results
        out_path = artifacts_dir / f"backtest_{model_name}.csv"
        write_dataframe(results, out_path)
        summary = aggregate_backtest_results(results)
        summary_path = artifacts_dir / f"summary_{model_name}.csv"
        write_dataframe(summary, summary_path)
        LOGGER.info("Model %s performance:\n%s", model_name, summary)
    return model_outputs


def fit_final_models(config: AppConfig) -> Dict[str, ForecastModel]:
    """Fit all enabled models on the full dataset for deployment."""
    features = _load_processed_features(config)
    target_column = config.features.get("target_column")
    if not target_column:
        raise RuntimeError("features.target_column must be defined in config")
    if target_column not in features.columns:
        raise KeyError(f"Target column {target_column} missing from processed features")

    X = features.drop(columns=[target_column])
    y = features[target_column]
    models: Dict[str, ForecastModel] = {}

    for model_name, model_cfg in config.models.get("candidates", {}).items():
        if not model_cfg.get("deploy", model_cfg.get("enabled", False)):
            continue
        factory = _build_model_factory(model_cfg)
        model = factory()
        LOGGER.info("Fitting final model %s", model_name)
        model.fit(X, y)
        model_path = config.models_dir / f"{model_name}.pkl"
        model.save(model_path)
        models[model_name] = model
    return models

def generate_latest_forecasts(config: AppConfig, trained_models: Dict[str, ForecastModel], horizon: int = 12) -> pd.DataFrame:
    """Generate forecasts for the specified horizon using trained models."""
    features = _load_processed_features(config)
    target_column = config.features.get("target_column")
    X = features.drop(columns=[target_column])
    forecasts = {}
    last_observation = features.index[-1]
    forecast_index = pd.date_range(last_observation + pd.offsets.MonthEnd(), periods=horizon, freq="M")

    for name, model in trained_models.items():
        LOGGER.info("Generating %d-step forecast for %s", horizon, name)
        preds = model.predict(X.tail(horizon))
        forecasts[name] = pd.Series(preds, index=forecast_index, name=name)
    frame = pd.concat(forecasts.values(), axis=1)
    ensure_directory(config.artifacts_dir)
    write_dataframe(frame, config.artifacts_dir / "latest_forecasts.csv")
    return frame
