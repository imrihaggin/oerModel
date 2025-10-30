"""Utilities for handling Zillow datasets (manual ingestion)."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable

import pandas as pd

from ..utils.logging import get_logger

LOGGER = get_logger(__name__)


def load_zillow_csv(path: Path, rename_map: Dict[str, str] | None = None, drop_cols: Iterable[str] | None = None) -> pd.DataFrame:
    """Load a Zillow dataset exported to CSV."""
    if not path.exists():
        raise FileNotFoundError(f"Zillow dataset not found at {path}")
    LOGGER.info("Loading Zillow data from %s", path)
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    if rename_map:
        df.rename(columns=rename_map, inplace=True)
    if drop_cols:
        df.drop(columns=list(drop_cols), inplace=True, errors="ignore")
    return df
