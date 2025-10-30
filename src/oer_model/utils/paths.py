"""Path helpers for locating project resources."""

from __future__ import annotations

from pathlib import Path


def get_project_root(start: Path | None = None) -> Path:
    """Return the project root by walking up until setup.py or .git is found."""
    current = start or Path(__file__).resolve().parents[3]
    for parent in [current, *current.parents]:
        if (parent / ".git").exists() or (parent / "setup.py").exists():
            return parent
    return current


def resolve_path(relative_path: str | Path, start: Path | None = None) -> Path:
    """Resolve a relative path from the project root."""
    root = get_project_root(start)
    return root / Path(relative_path)
