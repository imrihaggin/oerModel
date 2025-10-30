"""FRED data ingestion utilities."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Dict, Iterable, List

import pandas as pd
from pandas_datareader import data as web
from pandas_datareader._utils import RemoteDataError

from ..utils.logging import get_logger

LOGGER = get_logger(__name__)


@dataclass
class FredSeries:
    """Specification for a FRED series."""

    series_id: str
    alias: str | None = None
    transformation: str | None = None


def fetch_series(series: FredSeries, start: dt.datetime, end: dt.datetime) -> pd.DataFrame:
    """Download a single FRED series."""
    try:
        raw = web.DataReader(series.series_id, "fred", start, end)
        if series.alias:
            raw.rename(columns={series.series_id: series.alias}, inplace=True)
        return raw
    except RemoteDataError as exc:
        LOGGER.error("Failed to download %s: %s", series.series_id, exc)
        raise


def fetch_bulk(series_specs: Iterable[Dict[str, str]], start: dt.datetime, end: dt.datetime) -> pd.DataFrame:
    """Download multiple series and align them by date."""
    frames: List[pd.DataFrame] = []
    for spec in series_specs:
        series = FredSeries(**spec)
        LOGGER.info("Downloading FRED series %s", series.series_id)
        frames.append(fetch_series(series, start, end))
    combined = pd.concat(frames, axis=1)
    combined.index.name = "date"
    return combined
