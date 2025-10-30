"""Command line interface for OER forecasting workflows."""

from __future__ import annotations

from pathlib import Path

import click
from .config import load_config
from .data.pipeline import collect_raw_data
from .features.engineering import build_feature_matrix
from .forecasting.pipeline import fit_final_models, generate_latest_forecasts, run_training_pipeline
from .models.base import ForecastModel
from .utils.logging import enable_file_logging, get_logger

LOGGER = get_logger(__name__)


@click.group()
@click.option("--config", "config_path", default=None, help="Path to configuration file.")
@click.option("--log-dir", default="logs", help="Directory for log files.")
@click.pass_context
def cli(ctx: click.Context, config_path: str | None, log_dir: str) -> None:
    """OER forecasting command suite."""
    cfg = load_config(config_path)
    enable_file_logging(Path(log_dir))
    ctx.obj = cfg


@cli.command("fetch-data")
@click.pass_obj
def fetch_data(config) -> None:
    """Download raw datasets from all configured sources."""
    collect_raw_data(config)


@cli.command("build-features")
@click.pass_obj
def build_features(config) -> None:
    """Construct the processed feature matrix."""
    build_feature_matrix(config)


@cli.command("backtest")
@click.pass_obj
def backtest(config) -> None:
    """Run rolling cross-validation for all models."""
    run_training_pipeline(config)


@cli.command("train")
@click.pass_obj
def train(config) -> None:
    """Train final models and persist artifacts."""
    models = fit_final_models(config)
    LOGGER.info("Trained %d models", len(models))


@cli.command("forecast")
@click.option("--horizon", default=12, show_default=True)
@click.pass_obj
def forecast(config, horizon: int) -> None:
    """Generate latest forecasts for all trained models."""
    ensure_models = {}
    for model_name in config.models.get("candidates", {}).keys():
        model_path = config.models_dir / f"{model_name}.pkl"
        if model_path.exists():
            ensure_models[model_name] = ForecastModel.load(model_path)
    if not ensure_models:
        LOGGER.warning("No trained models found. Run the train command first.")
        return
    frame = generate_latest_forecasts(config, ensure_models, horizon)
    LOGGER.info("Generated forecasts for %d models", frame.shape[1])


if __name__ == "__main__":
    cli()
