import sys
import threading
from types import SimpleNamespace

import numpy as np
import pytest

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
    service._capture_ready = threading.Event()
    service._capture_error = None
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


def test_audio_service_start_preserves_pre_start_user_mute(monkeypatch):
    class FakeThread:
        def __init__(self, target, daemon, name):
            self.target = target
            self.daemon = daemon
            self.name = name
            self.started = False

        def start(self):
            self.started = True
            self.target()

    service = make_audio_service_state(muted_by_user=True)
    service.config = SimpleNamespace(start_muted=False)
    service._running = False
    service._loop = None
    service._stream = None
    service._capture_thread = None
    service._capture_loop = lambda: service._capture_ready.set()
    monkeypatch.setattr("src.assistant.audio_service.SOUNDDEVICE_AVAILABLE", True)
    monkeypatch.setattr("src.assistant.audio_service.threading.Thread", FakeThread)

    service.start()

    assert service.state == MicState.MUTED
    assert service._muted_by_user is True
    assert service._capture_thread.started is True


def test_audio_service_start_surfaces_capture_failure(monkeypatch):
    class FakeThread:
        def __init__(self, target, daemon, name):
            self.target = target

        def start(self):
            self.target()

        def join(self, timeout):
            return None

    service = make_audio_service_state()
    service.config = SimpleNamespace(start_muted=False)
    service._running = False
    service._loop = None
    service._stream = None
    service._capture_thread = None

    def fail_capture():
        service._capture_error = RuntimeError("no microphone")
        service._running = False
        service._capture_ready.set()

    service._capture_loop = fail_capture
    monkeypatch.setattr("src.assistant.audio_service.SOUNDDEVICE_AVAILABLE", True)
    monkeypatch.setattr("src.assistant.audio_service.threading.Thread", FakeThread)

    with pytest.raises(RuntimeError, match="no microphone"):
        service.start()

    assert service._running is False
    assert service._capture_thread is None


def test_audio_service_start_timeout_closes_late_stream(monkeypatch):
    class FakeStream:
        def __init__(self):
            self.stopped = False
            self.closed = False

        def stop(self):
            self.stopped = True

        def close(self):
            self.closed = True

    class FakeThread:
        def __init__(self, target, daemon, name):
            self.target = target
            self.started = False

        def start(self):
            self.started = True

        def join(self, timeout):
            service._stream = late_stream

    service = make_audio_service_state()
    service.config = SimpleNamespace(start_muted=False)
    service._running = False
    service._loop = None
    service._stream = None
    service._capture_thread = None
    service._capture_loop = lambda: None
    late_stream = FakeStream()
    monkeypatch.setattr("src.assistant.audio_service.SOUNDDEVICE_AVAILABLE", True)
    monkeypatch.setattr("src.assistant.audio_service.threading.Thread", FakeThread)

    with pytest.raises(RuntimeError, match="Timed out"):
        service.start(startup_timeout_sec=0.01)

    assert service._running is False
    assert service._capture_thread is None
    assert service._stream is None
    assert late_stream.stopped is True
    assert late_stream.closed is True


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
