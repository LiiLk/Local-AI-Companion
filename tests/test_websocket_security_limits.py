import base64
from types import SimpleNamespace

import pytest

from src.server.websocket import (
    MAX_WEBSOCKET_AUDIO_BYTES,
    MAX_WEBSOCKET_AUDIO_SAMPLES,
    _audio_sample_count,
    _config_for_websocket,
    _estimated_base64_decoded_size,
)


def test_estimated_base64_decoded_size_handles_plain_payloads():
    payload = base64.b64encode(b"hello").decode("ascii")

    assert _estimated_base64_decoded_size(payload) == 5


def test_estimated_base64_decoded_size_handles_data_urls():
    payload = "data:audio/webm;base64," + base64.b64encode(b"hello").decode("ascii")

    assert _estimated_base64_decoded_size(payload) == 5


def test_audio_limit_constant_is_bounded_for_realtime_payloads():
    assert MAX_WEBSOCKET_AUDIO_BYTES == 8 * 1024 * 1024


def test_estimated_base64_decoded_size_rejects_invalid_payloads():
    with pytest.raises(ValueError):
        _estimated_base64_decoded_size("not valid base64!")


def test_audio_sample_count_counts_nested_chunks():
    samples = [[0.0] * 3, [0.1] * 4, 0.2]

    assert _audio_sample_count(samples) == 8
    oversized_samples = [[0.0] * MAX_WEBSOCKET_AUDIO_SAMPLES, [0.0]]

    assert _audio_sample_count(oversized_samples) > MAX_WEBSOCKET_AUDIO_SAMPLES


def test_config_for_websocket_uses_app_state_config_without_reload(monkeypatch):
    config = {"server": {"websocket": {"allow_origins": ["http://example.test"]}}}
    websocket = SimpleNamespace(
        scope={"app": SimpleNamespace(state=SimpleNamespace(config=config))}
    )
    monkeypatch.setattr(
        "src.server.websocket.load_config",
        lambda: pytest.fail("load_config should not be called for app-scoped config"),
    )

    assert _config_for_websocket(websocket) is config


def test_config_for_websocket_falls_back_to_safe_defaults_without_reload(monkeypatch):
    websocket = SimpleNamespace(scope={"app": SimpleNamespace(state=SimpleNamespace())})
    monkeypatch.setattr(
        "src.server.websocket.load_config",
        lambda: pytest.fail("load_config should not be called during origin checks"),
    )

    assert _config_for_websocket(websocket) == {}
