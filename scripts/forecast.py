"""Script to generate forecasts from trained models."""

from oer_model.config import load_config
from oer_model.forecasting.pipeline import generate_latest_forecasts
from oer_model.models.base import ForecastModel


def main(horizon: int = 12) -> None:
    config = load_config()
    trained = {}
    for model_name in config.models.get("candidates", {}).keys():
        model_path = config.models_dir / f"{model_name}.pkl"
        if model_path.exists():
            trained[model_name] = ForecastModel.load(model_path)
    if not trained:
        raise RuntimeError("No trained models found. Run train_models.py first.")
    generate_latest_forecasts(config, trained, horizon)


if __name__ == "__main__":
    main()
