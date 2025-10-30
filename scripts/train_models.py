"""Script to train final models."""

from oer_model.config import load_config
from oer_model.forecasting.pipeline import fit_final_models


def main() -> None:
    config = load_config()
    fit_final_models(config)


if __name__ == "__main__":
    main()
