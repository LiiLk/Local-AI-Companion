"""Smoke tests for ASR profile resolution (LIL-37)."""

from src.assistant.pipeline_runtime import (
    DEFAULT_WHISPER_PROFILE,
    resolve_whisper_profile,
)


def test_default_profile_is_balanced_small():
    settings = resolve_whisper_profile({})
    assert settings["profile"] == "balanced"
    assert settings["model_size"] == "small"
    assert settings["beam_size"] == 3


def test_quality_local_profile_uses_large_v3_turbo():
    settings = resolve_whisper_profile({"profile": "quality-local"})
    assert settings["profile"] == "quality-local"
    assert settings["model_size"] == "large-v3-turbo"
    assert settings["beam_size"] == 5


def test_explicit_model_size_overrides_profile():
    settings = resolve_whisper_profile({"profile": "quality-local", "model_size": "medium"})
    assert settings["model_size"] == "medium"  # explicit wins
    assert settings["beam_size"] == 5  # profile still supplies the beam size


def test_unknown_profile_falls_back_to_default():
    settings = resolve_whisper_profile({"profile": "does-not-exist"})
    assert settings["profile"] == DEFAULT_WHISPER_PROFILE
    assert settings["model_size"] == "small"


def test_compute_type_and_device_passthrough():
    settings = resolve_whisper_profile(
        {"profile": "quality-local", "compute_type": "int8_float16", "device": "cuda"}
    )
    assert settings["compute_type"] == "int8_float16"
    assert settings["device"] == "cuda"
