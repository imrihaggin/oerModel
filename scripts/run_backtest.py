"""Script to run backtesting workflow."""

from oer_model.config import load_config
from oer_model.forecasting.pipeline import run_training_pipeline


def main() -> None:
    config = load_config()
    run_training_pipeline(config)


if __name__ == "__main__":
    main()
