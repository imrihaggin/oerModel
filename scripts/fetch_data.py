"""Convenience script for downloading raw datasets."""

from oer_model.config import load_config
from oer_model.data.pipeline import collect_raw_data


def main() -> None:
    config = load_config()
    collect_raw_data(config)


if __name__ == "__main__":
    main()
