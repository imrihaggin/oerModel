"""Input/output helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


def ensure_directory(path: Path) -> Path:
    """Create directory if needed and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_dataframe(df: pd.DataFrame, path: Path, *, index: bool = True) -> Path:
    """Write dataframe to CSV with deterministic settings."""
    ensure_directory(path.parent)
    df.to_csv(path, index=index)
    return path


def read_dataframe(path: Path, parse_dates: Iterable[str] | None = None) -> pd.DataFrame:
    """Read a CSV into a DataFrame."""
    return pd.read_csv(path, parse_dates=parse_dates, index_col=0 if parse_dates else None)
