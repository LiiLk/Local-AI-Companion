"""Helpers to resolve optional RVC config from character and folder conventions."""

from __future__ import annotations

from src.tts.rvc_provider import resolve_rvc_paths


def _character_rvc_overrides(character_config: dict) -> dict:
    overrides: dict = {}

    direct_rvc = character_config.get("rvc")
    if isinstance(direct_rvc, dict):
        overrides.update(direct_rvc)

    voice_config = character_config.get("voice")
    if isinstance(voice_config, dict):
        voice_rvc = voice_config.get("rvc")
        if isinstance(voice_rvc, dict):
            overrides.update(voice_rvc)

    return overrides


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
    rvc_config.update(_character_rvc_overrides(character_config))
    model_path, index_path = resolve_rvc_paths(rvc_config, character_config)
    rvc_config["model_path"] = str(model_path)
    if index_path:
        rvc_config["index_path"] = str(index_path)
    else:
        rvc_config["index_path"] = None
    return rvc_config
