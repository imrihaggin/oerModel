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


@cli.command("dashboard")
@click.option("--port", default=8050, show_default=True, help="Port to run dashboard on")
@click.option("--host", default="127.0.0.1", show_default=True, help="Host to bind to")
@click.pass_obj
def dashboard(config, port: int, host: str) -> None:
    """Launch interactive dashboard to visualize results."""
    from .viz.dashboard import create_dashboard
    
    artifacts_dir = Path(config.artifacts_dir)
    forecasts_path = artifacts_dir / "latest_forecasts.csv"
    
    # Check if required files exist
    if not artifacts_dir.exists():
        LOGGER.error("Artifacts directory not found: %s", artifacts_dir)
        click.echo("❌ Error: No artifacts found. Run backtest and forecast first.")
        return
    
    if not forecasts_path.exists():
        LOGGER.warning("Forecasts file not found: %s", forecasts_path)
        click.echo("⚠️  Warning: No forecasts found. Run: oer-model forecast --horizon 12")
        click.echo("Dashboard will launch with backtest results only.\n")
    
    click.echo("\n" + "="*60)
    click.echo("🚀 OER Forecasting Dashboard")
    click.echo("="*60)
    click.echo(f"\n📊 Dashboard running at: http://{host}:{port}")
    click.echo(f"\n📁 Loading data from: {artifacts_dir}")
    click.echo("\n⚡ Press CTRL+C to stop\n")
    
    app = create_dashboard(artifacts_dir, forecasts_path)
    app.run(debug=False, host=host, port=port)


if __name__ == "__main__":
    cli()
