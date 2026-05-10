import sys
from types import SimpleNamespace

import numpy as np

from src.assistant.audio_service import AudioService, MicState


class DummyVAD:
    def __init__(self):
        self.reset_calls = 0

    def reset(self):
        self.reset_calls += 1


def make_audio_service_state(
    *,
    muted_by_user: bool = False,
    processing_blocked: bool = False,
) -> AudioService:
    service = object.__new__(AudioService)
    service._muted_by_user = muted_by_user
    service._processing_blocked = processing_blocked
    service._state = service._effective_state()
    service._vad = DummyVAD()
    service.on_state_change = None
    return service


def test_audio_service_toggle_mute_works_while_processing():
    service = make_audio_service_state(processing_blocked=True)

    muted = service.toggle_mute()

    assert muted is True
    assert service.state == MicState.MUTED
    assert service._vad.reset_calls == 1


def test_audio_service_unmute_while_processing_returns_to_processing_state():
    service = make_audio_service_state(muted_by_user=True, processing_blocked=True)

    muted = service.toggle_mute()

    assert muted is False
    assert service.state == MicState.PROCESSING
    assert service._vad.reset_calls == 1


def test_audio_service_processing_release_preserves_user_mute():
    service = make_audio_service_state(muted_by_user=True, processing_blocked=True)

    service.set_processing(False)

    assert service.state == MicState.MUTED
    assert service._vad.reset_calls == 1


def test_audio_service_processing_release_restores_listening_when_not_muted():
    service = make_audio_service_state(processing_blocked=True)

    service.set_processing(False)

    assert service.state == MicState.LISTENING
    assert service._vad.reset_calls == 1


def test_audio_service_resample_uses_soxr_when_available(monkeypatch):
    calls = []

    def fake_resample(audio, src_rate, dst_rate):
        calls.append((audio.dtype, src_rate, dst_rate))
        return np.ones(5, dtype=np.float64)

    monkeypatch.setitem(sys.modules, "soxr", SimpleNamespace(resample=fake_resample))
    service = object.__new__(AudioService)

    result = service._resample_audio(np.zeros(10, dtype=np.float32), 48000, 16000)

    assert calls == [(np.dtype("float32"), 48000, 16000)]
    assert result.dtype == np.float32
    assert result.shape == (5,)


def test_audio_service_resample_falls_back_to_linear_interpolation(monkeypatch):
    monkeypatch.setitem(sys.modules, "soxr", None)
    service = object.__new__(AudioService)

    result = service._resample_audio(np.linspace(-1, 1, 9, dtype=np.float32), 48000, 16000)

    assert result.dtype == np.float32
    assert result.shape == (3,)
