"""Application logging utilities."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional


DEFAULT_LOG_LEVEL = logging.INFO


def get_logger(name: Optional[str] = None, level: int = DEFAULT_LOG_LEVEL) -> logging.Logger:
    """Return a configured logger instance."""
    logger = logging.getLogger(name if name else "oer_model")
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        handler.setFormatter(logging.Formatter(fmt))
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def enable_file_logging(log_dir: Path, filename: str = "oer_model.log", level: int = DEFAULT_LOG_LEVEL) -> Path:
    """Configure file-based logging and return the log path."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / filename
    handler = logging.FileHandler(log_path)
    fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    root_logger = logging.getLogger("oer_model")
    root_logger.addHandler(handler)
    root_logger.setLevel(level)
    return log_path
