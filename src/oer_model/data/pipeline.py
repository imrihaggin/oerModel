"""Data ingestion pipeline orchestrator."""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Dict

import pandas as pd

from ..config import AppConfig
from ..utils.io import ensure_directory, write_dataframe
from ..utils.logging import get_logger
from . import bloomberg, fred, zillow

LOGGER = get_logger(__name__)


def _parse_date(value: str | None) -> dt.datetime | None:
    if not value:
        return None
    return dt.datetime.fromisoformat(value)


def collect_raw_data(config: AppConfig) -> pd.DataFrame:
    """Fetch all configured raw datasets and persist them."""
    data_cfg: Dict[str, Dict] = config.data_sources
    start = _parse_date(data_cfg.get("start_date")) or dt.datetime(2000, 1, 1)
    end = _parse_date(data_cfg.get("end_date")) or dt.datetime.today()
    frames: list[pd.DataFrame] = []

    fred_cfg = data_cfg.get("fred", {})
    if fred_cfg.get("enabled", False):
        LOGGER.info("Fetching FRED datasets")
        frames.append(fred.fetch_bulk(fred_cfg.get("series", []), start, end))

    blp_cfg = data_cfg.get("bloomberg", {})
    if blp_cfg.get("enabled", False):
        LOGGER.info("Fetching Bloomberg datasets")
        frames.append(bloomberg.fetch_bulk(blp_cfg.get("series", []), start, end))

    manual_cfg = data_cfg.get("manual", {})
    for dataset in manual_cfg.get("datasets", []):
        csv_path = Path(dataset["path"])
        if not csv_path.exists():
            LOGGER.warning("Skipping manual dataset (not found): %s", csv_path)
            continue
        rename_map = dataset.get("rename", {})
        drop_cols = dataset.get("drop", [])
        frames.append(zillow.load_zillow_csv(csv_path, rename_map, drop_cols))

    if not frames:
        raise RuntimeError("No data sources enabled; nothing to fetch.")

    combined = pd.concat(frames, axis=1).sort_index().ffill()
    combined.index.name = "date"
    ensure_directory(config.raw_dir)
    raw_out = config.raw_dir / "raw_combined.csv"
    write_dataframe(combined, raw_out)
    LOGGER.info("Saved combined raw dataset to %s", raw_out)
    return combined
