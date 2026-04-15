"""
Helpers for loading tracked config plus local ignored overrides.
"""

from __future__ import annotations

from pathlib import Path
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


def load_yaml_config(config_path: Path) -> dict[str, Any]:
    """Load ``config.yaml`` and merge ``config.local.yaml`` when present."""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    local_override_path = config_path.with_name("config.local.yaml")
    if local_override_path.exists():
        with open(local_override_path, "r", encoding="utf-8") as f:
            local_override = yaml.safe_load(f) or {}
        config = _deep_merge(config, local_override)

    return config
