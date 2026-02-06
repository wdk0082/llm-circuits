"""Path resolution utilities."""

from __future__ import annotations

from pathlib import Path

import yaml


def load_config(path: str | Path) -> dict:
    """Load a YAML config file and return its contents as a dict."""
    path = Path(path)
    with path.open() as f:
        return yaml.safe_load(f)


def configs_dir() -> Path:
    """Return the ``configs/`` directory at the repo root."""
    return Path(__file__).resolve().parents[3] / "configs"
