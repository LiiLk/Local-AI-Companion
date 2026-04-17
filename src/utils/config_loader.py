"""
Helpers for loading tracked config plus local ignored overrides.
"""

from __future__ import annotations

from pathlib import Path
import tempfile
from typing import Any

import yaml


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        current = merged.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            merged[key] = _deep_merge(current, value)
        else:
            merged[key] = value
    return merged


def _read_yaml_file(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def get_local_override_path(config_path: Path) -> Path:
    return config_path.with_name("config.local.yaml")


def load_local_yaml_config(config_path: Path) -> dict[str, Any]:
    """Load ``config.local.yaml`` when present, otherwise return an empty dict."""
    local_override_path = get_local_override_path(config_path)
    if not local_override_path.exists():
        return {}
    return _read_yaml_file(local_override_path)


def write_local_yaml_config(config_path: Path, data: dict[str, Any]) -> None:
    """Write ``config.local.yaml`` atomically."""
    local_override_path = get_local_override_path(config_path)
    local_override_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=local_override_path.parent,
        prefix=f"{local_override_path.stem}.",
        suffix=".tmp",
        delete=False,
    ) as temp_file:
        yaml.safe_dump(
            data,
            temp_file,
            sort_keys=False,
            allow_unicode=False,
            default_flow_style=False,
        )
        temp_path = Path(temp_file.name)

    temp_path.replace(local_override_path)


def load_yaml_config(config_path: Path) -> dict[str, Any]:
    """Load ``config.yaml`` and merge ``config.local.yaml`` when present."""
    config = _read_yaml_file(config_path)

    local_override = load_local_yaml_config(config_path)
    if local_override:
        config = _deep_merge(config, local_override)

    return config
