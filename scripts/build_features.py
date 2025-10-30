"""Script to construct processed feature matrix."""

from oer_model.config import load_config
from oer_model.features.engineering import build_feature_matrix


def main() -> None:
    config = load_config()
    build_feature_matrix(config)


if __name__ == "__main__":
    main()
