import base64

from src.server.websocket import (
    MAX_WEBSOCKET_AUDIO_BYTES,
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
