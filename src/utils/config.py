"""
LegalSaarthi configuration loader.

Loads YAML configuration with environment variable overrides.
"""
import os
from pathlib import Path
from typing import Any

import yaml


# Project root: two levels up from src/utils/
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load YAML configuration file.

    Args:
        config_path: Path to YAML config. Defaults to configs/default.yaml.

    Returns:
        Configuration dictionary.

    Raises:
        FileNotFoundError: If config file does not exist.
        yaml.YAMLError: If config file is malformed.
    """
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if config is None:
        config = {}

    return config


def get_project_root() -> Path:
    """Return the project root directory."""
    return PROJECT_ROOT


def resolve_path(config: dict[str, Any], key: str) -> Path:
    """Resolve a relative path from config against the project root.

    Args:
        config: Configuration dictionary.
        key: Dot-separated key path, e.g. 'paths.raw_pdfs'.

    Returns:
        Absolute Path.
    """
    parts = key.split(".")
    value = config
    for part in parts:
        value = value[part]

    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path

    return path
