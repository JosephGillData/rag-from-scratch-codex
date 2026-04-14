"""Configuration loading utilities for the application.

The project intentionally keeps configuration simple: one small YAML file,
one dataclass that represents the loaded settings, and a few readable checks.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class AppConfig:
    """Application settings loaded from ``config.yaml``."""

    docs_path: str
    chunk_size: int
    chunk_overlap: int
    use_saved_chunks: bool
    chunks_path: str
    embedding_model: str
    vector_store_path: str
    top_k: int
    show_sources: bool
    show_retrieved_chunks: bool


def load_config(path: Path = Path("config.yaml")) -> AppConfig:
    """Load application settings from a YAML file.

    Parameters
    ----------
    path:
        Path to the YAML configuration file.
    """
    raw_config = _read_yaml(path)
    config = AppConfig(
        docs_path=_require_string(raw_config, "docs_path"),
        chunk_size=_require_positive_int(raw_config, "chunk_size"),
        chunk_overlap=_require_non_negative_int(raw_config, "chunk_overlap"),
        use_saved_chunks=_require_bool(raw_config, "use_saved_chunks"),
        chunks_path=_require_string(raw_config, "chunks_path"),
        embedding_model=_require_string(raw_config, "embedding_model"),
        vector_store_path=_require_string(raw_config, "vector_store_path"),
        top_k=_require_positive_int(raw_config, "top_k"),
        show_sources=_require_bool(raw_config, "show_sources"),
        show_retrieved_chunks=_require_bool(raw_config, "show_retrieved_chunks"),
    )
    _validate_config(config)
    return config


def _read_yaml(path: Path) -> dict[str, Any]:
    """Read and parse a YAML file into a dictionary."""
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}

    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping at the root of {path}.")

    return data


def _require_string(data: dict[str, Any], key: str) -> str:
    """Return a required string value from the config data."""
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Config value '{key}' must be a non-empty string.")
    return value


def _require_positive_int(data: dict[str, Any], key: str) -> int:
    """Return a required integer that must be greater than zero."""
    value = data.get(key)
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"Config value '{key}' must be a positive integer.")
    return value


def _require_non_negative_int(data: dict[str, Any], key: str) -> int:
    """Return a required integer that must be zero or greater."""
    value = data.get(key)
    if not isinstance(value, int) or value < 0:
        raise ValueError(f"Config value '{key}' must be a non-negative integer.")
    return value


def _require_bool(data: dict[str, Any], key: str) -> bool:
    """Return a required boolean value from the config data."""
    value = data.get(key)
    if not isinstance(value, bool):
        raise ValueError(f"Config value '{key}' must be a boolean.")
    return value


def _validate_config(config: AppConfig) -> None:
    """Apply a few cross-field checks that keep the config sensible."""
    if config.chunk_overlap >= config.chunk_size:
        raise ValueError("Config value 'chunk_overlap' must be smaller than 'chunk_size'.")
