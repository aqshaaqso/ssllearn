from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when an experiment configuration is missing required values."""


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def read_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"Configuration must be a YAML mapping: {path}")
    return data


def load_config(config_path: str | Path, env_config_path: str | Path | None = None) -> dict[str, Any]:
    config = read_yaml(config_path)
    if env_config_path:
        config = deep_merge(config, read_yaml(env_config_path))
    validate_config(config)
    return config


def require_path(config: dict[str, Any], dotted_key: str) -> Any:
    current: Any = config
    for part in dotted_key.split("."):
        if not isinstance(current, dict) or part not in current:
            raise ConfigError(f"Missing required config key: {dotted_key}")
        current = current[part]
    if current is None or current == "":
        raise ConfigError(f"Config key must not be empty: {dotted_key}")
    return current


def validate_config(config: dict[str, Any]) -> None:
    required = [
        "project_name",
        "seed",
        "device",
        "output_dir",
        "model.variant",
        "model.embedding_dim",
        "ssl.image_dir",
        "ssl.image_size",
        "ssl.batch_size",
        "ssl.epochs",
        "ssl.temperature",
        "ssl.learning_rate",
        "detection.dataset_yaml",
        "detection.image_size",
        "detection.epochs",
        "detection.batch_size",
        "detection.labeled_fraction",
    ]
    for key in required:
        require_path(config, key)

    fraction = float(require_path(config, "detection.labeled_fraction"))
    if not 0 < fraction <= 1:
        raise ConfigError("detection.labeled_fraction must be in the range (0, 1].")

    if int(require_path(config, "ssl.batch_size")) < 2:
        raise ConfigError("ssl.batch_size must be at least 2 for contrastive learning.")


def resolve_path(path_value: str | Path, base_dir: str | Path | None = None) -> Path:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    if base_dir is None:
        return path
    return Path(base_dir).expanduser().resolve() / path

