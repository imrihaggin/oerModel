"""Configuration loading utilities."""

from __future__ import annotations

import yaml
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from .utils.paths import get_project_root


DEFAULT_CONFIG_PATH = Path("config/config.yaml")


@dataclass
class AppConfig:
    """Typed wrapper for application configuration."""

    raw_dir: Path
    interim_dir: Path
    processed_dir: Path
    artifacts_dir: Path
    models_dir: Path
    dashboards_dir: Path
    data_sources: Dict[str, Any]
    features: Dict[str, Any]
    backtest: Dict[str, Any]
    models: Dict[str, Any]


class ConfigError(RuntimeError):
    """Raised when configuration cannot be loaded."""


def load_config(config_path: Optional[Path | str] = None) -> AppConfig:
    """Load the YAML config into an AppConfig instance."""
    root = get_project_root()
    cfg_path = root / Path(config_path or DEFAULT_CONFIG_PATH)
    if not cfg_path.exists():
        raise ConfigError(f"Config file not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as fh:
        payload: Dict[str, Any] = yaml.safe_load(fh)
    project = payload.get("project", {})
    data_cfg = payload.get("data", {})
    feature_cfg = payload.get("features", {})
    backtest_cfg = payload.get("backtest", {})
    models_cfg = payload.get("models", {})
    return AppConfig(
        raw_dir=(root / project.get("raw_dir", "data/raw")),
        interim_dir=(root / project.get("interim_dir", "data/interim")),
        processed_dir=(root / project.get("processed_dir", "data/processed")),
        artifacts_dir=(root / project.get("artifacts_dir", "artifacts")),
        models_dir=(root / project.get("models_dir", "models")),
        dashboards_dir=(root / project.get("dashboards_dir", "dashboards")),
        data_sources=data_cfg,
        features=feature_cfg,
        backtest=backtest_cfg,
        models=models_cfg,
    )
