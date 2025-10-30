"""Bloomberg data ingestion utilities."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Dict, Iterable, List

import pandas as pd

try:
    from xbbg import blp
    HAS_BLOOMBERG = True
except ImportError:  # pragma: no cover - optional dependency
    HAS_BLOOMBERG = False

from ..utils.logging import get_logger

LOGGER = get_logger(__name__)


@dataclass
class BloombergSeries:
    """Specification for a Bloomberg field request."""

    ticker: str
    field: str = "PX_LAST"
    periodicity: str = "monthly"
    alias: str | None = None


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
    return df


def fetch_series(series: BloombergSeries, start: dt.datetime, end: dt.datetime) -> pd.DataFrame:
    """Download a single Bloomberg series."""
    if not HAS_BLOOMBERG:
        raise RuntimeError("xbbg is not available; Bloomberg data cannot be fetched.")
    LOGGER.info("Downloading Bloomberg series %s", series.ticker)
    raw = blp.bdh(
        tickers=series.ticker,
        flds=series.field,
        start_date=start.strftime("%Y-%m-%d"),
        end_date=end.strftime("%Y-%m-%d"),
        Per=series.periodicity,
    )
    normalized = _normalize_columns(raw)
    if series.alias:
        normalized.rename(columns={series.ticker: series.alias}, inplace=True)
    return normalized


def fetch_bulk(series_specs: Iterable[Dict[str, str]], start: dt.datetime, end: dt.datetime) -> pd.DataFrame:
    """Download multiple Bloomberg series."""
    frames: List[pd.DataFrame] = []
    for spec in series_specs:
        series = BloombergSeries(**spec)
        frames.append(fetch_series(series, start, end))
    combined = pd.concat(frames, axis=1)
    combined.index.name = "date"
    return combined
