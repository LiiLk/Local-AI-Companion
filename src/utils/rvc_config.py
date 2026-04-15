"""Helpers to resolve optional RVC config from character and folder conventions."""

from __future__ import annotations

from src.tts.rvc_provider import resolve_rvc_paths


def build_rvc_runtime_config(config: dict) -> dict | None:
    """
    Build a normalized runtime RVC config.

    Returns None when RVC is disabled. When enabled, model/index paths are resolved
    from either explicit paths or a folder convention.
    """
    tts_config = config.get("tts", {})
    rvc_config = dict(tts_config.get("rvc", {}))
    if not rvc_config.get("enabled", False):
        return None

    character_config = config.get("character", {})
    model_path, index_path = resolve_rvc_paths(rvc_config, character_config)
    rvc_config["model_path"] = str(model_path)
    if index_path:
        rvc_config["index_path"] = str(index_path)
    else:
        rvc_config["index_path"] = None
    return rvc_config
