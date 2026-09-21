"""Integration tests for adaptive end-of-turn commit delays.

Both turn paths (desktop ``app.py`` and WebSocket ``websocket.py``) must arm a
short delay when Smart Turn reports a complete turn, keep the long delay when
the turn looks incomplete, and fall back to the legacy fixed delay when the
detector is disabled or unavailable. No model is downloaded and no inference
runs: the detector is faked.
"""

import asyncio
import threading
from types import SimpleNamespace

import pytest

from src.assistant.app import Live2DAssistant
from src.server.websocket import WebSocketManager
from src.vad.smart_turn import SmartTurnConfig


class FakeHandle:
    def __init__(self):
        self.cancelled = False

    def cancel(self):
        self.cancelled = True


class FakeLoop:
    """Loop stub that runs call_soon_threadsafe inline and records call_later."""

    def __init__(self):
        self.scheduled = []

    def call_later(self, delay, callback):
        handle = FakeHandle()
        handle.callback = callback
        self.scheduled.append((delay, handle))
        return handle

    def call_soon_threadsafe(self, callback, *args):
        callback(*args)

    def is_closed(self):
        return False


class FakeDetector:
    def __init__(self, result, config, available=True):
        self.result = result
        self.config = config
        self.available = available
        self.calls = 0
        self.last_audio = None

    def predict(self, audio, sample_rate=16000):
        self.calls += 1
        self.last_audio = audio
        return self.result


def _detection_config(**overrides):
    base = dict(
        enabled=True,
        complete_delay_ms=250,
        incomplete_delay_ms=2500,
        fallback_delay_ms=700,
    )
    base.update(overrides)
    return SmartTurnConfig(**base)


def _make_desktop_assistant(config, detector):
    assistant = Live2DAssistant.__new__(Live2DAssistant)
    assistant._loop = FakeLoop()
    assistant._pending_speech_lock = threading.Lock()
    assistant._pending_speech_audio = bytearray()
    assistant._speech_active = False
    assistant._drop_current_speech = False
    assistant._pending_speech_commit_handle = None
    assistant._pending_speech_commit_delay_ms = 700
    assistant._pending_speech_end_monotonic = None
    assistant._pending_speech_generation = 0
    assistant._pending_speech_detection_thread = None
    assistant._speech_commit_delay_ms = 700
    assistant._turn_detection_config = config
    assistant._smart_turn = detector
    assistant._active_turn_id = None
    assistant._latest_audio_turn_id = None
    assistant._assistant_busy = lambda: False
    assistant._dispatch_frontend_event = lambda *args: None
    return assistant


def test_desktop_complete_verdict_shortens_delay_to_complete_window():
    config = _detection_config()
    detector = FakeDetector((True, 0.93), config)
    assistant = _make_desktop_assistant(config, detector)

    assistant._on_speech_start()
    assistant._on_speech_detected(b"A" * 3200)
    assistant._on_speech_end()

    # Long window armed immediately so the user is never cut off.
    assert assistant._loop.scheduled[0][0] == pytest.approx(2.5)

    assistant._pending_speech_detection_thread.join(timeout=5)

    # Then shortened once the verdict arrives (minus inference elapsed time).
    assert detector.calls == 1
    assert 0.0 <= assistant._loop.scheduled[-1][0] <= 0.25


def test_desktop_incomplete_verdict_keeps_long_delay():
    config = _detection_config()
    detector = FakeDetector((False, 0.05), config)
    assistant = _make_desktop_assistant(config, detector)

    assistant._on_speech_start()
    assistant._on_speech_detected(b"A" * 3200)
    assistant._on_speech_end()
    assistant._pending_speech_detection_thread.join(timeout=5)

    assert detector.calls == 1
    assert len(assistant._loop.scheduled) == 1
    assert assistant._loop.scheduled[-1][0] == pytest.approx(2.5)


def test_desktop_uncertain_verdict_uses_uncertain_delay():
    config = _detection_config()
    detector = FakeDetector((False, 0.30), config)
    assistant = _make_desktop_assistant(config, detector)

    assistant._on_speech_start()
    assistant._on_speech_detected(b"A" * 3200)
    assistant._on_speech_end()

    # Long window armed first, then shortened to the uncertain window.
    assert assistant._loop.scheduled[0][0] == pytest.approx(2.5)

    assistant._pending_speech_detection_thread.join(timeout=5)

    assert detector.calls == 1
    assert len(assistant._loop.scheduled) == 2
    assert 0.5 <= assistant._loop.scheduled[-1][0] <= 0.9


def test_desktop_vad_misses_shorten_when_detector_available():
    config = _detection_config()
    detector = FakeDetector((True, 0.99), config)
    assistant = _make_desktop_assistant(config, detector)
    calls = []
    assistant._default_vad_required_misses = 20
    assistant.audio_service = SimpleNamespace(
        set_vad_required_misses=lambda misses: calls.append(misses)
    )

    assistant._sync_vad_turn_detection_misses()

    assert calls == [8]


def test_desktop_vad_misses_restore_when_detector_unavailable():
    config = _detection_config()
    detector = FakeDetector((True, 0.99), config, available=False)
    assistant = _make_desktop_assistant(config, detector)
    calls = []
    assistant._default_vad_required_misses = 20
    assistant.audio_service = SimpleNamespace(
        set_vad_required_misses=lambda misses: calls.append(misses)
    )

    assistant._sync_vad_turn_detection_misses()

    assert calls == [20]


def test_desktop_unavailable_detector_rearms_on_fallback():
    config = _detection_config()
    detector = FakeDetector(None, config, available=True)
    assistant = _make_desktop_assistant(config, detector)

    assistant._on_speech_start()
    assistant._on_speech_detected(b"A" * 3200)
    assistant._on_speech_end()
    assistant._pending_speech_detection_thread.join(timeout=5)

    # Inference ran, returned None, and the long window was replaced by the
    # fallback window (700ms minus elapsed inference time).
    assert detector.calls == 1
    assert len(assistant._loop.scheduled) == 2
    assert 0.0 <= assistant._loop.scheduled[-1][0] <= 0.7


def test_desktop_known_unavailable_detector_arms_fallback_without_inference():
    config = _detection_config()
    detector = FakeDetector((True, 0.99), config, available=False)
    assistant = _make_desktop_assistant(config, detector)

    assistant._on_speech_start()
    assistant._on_speech_detected(b"A" * 3200)
    assistant._on_speech_end()

    assert detector.calls == 0
    assert assistant._pending_speech_detection_thread is None
    assert len(assistant._loop.scheduled) == 1
    assert assistant._loop.scheduled[-1][0] == pytest.approx(0.7)


def test_desktop_disabled_turn_detection_uses_fixed_delay():
    config = _detection_config(enabled=False)
    detector = FakeDetector((True, 0.99), config)
    assistant = _make_desktop_assistant(config, detector)

    assistant._on_speech_start()
    assistant._on_speech_detected(b"A" * 3200)
    assistant._on_speech_end()

    assert detector.calls == 0
    assert assistant._pending_speech_detection_thread is None
    assert len(assistant._loop.scheduled) == 1
    assert assistant._loop.scheduled[-1][0] == pytest.approx(0.7)


def test_desktop_stale_verdict_after_new_speech_is_ignored():
    config = _detection_config()
    detector = FakeDetector((True, 0.9), config)
    assistant = _make_desktop_assistant(config, detector)

    assistant._on_speech_start()
    assistant._on_speech_detected(b"A" * 3200)
    assistant._on_speech_end()
    assistant._pending_speech_detection_thread.join(timeout=5)

    stale_generation = assistant._pending_speech_generation

    # A new utterance invalidates the previous turn's verdict.
    assistant._on_speech_start()
    assistant._on_speech_detected(b"B" * 1600)

    scheduled_before = list(assistant._loop.scheduled)
    assistant._apply_turn_verdict(
        stale_generation, (True, 0.9), 5.0, assistant._pending_speech_end_monotonic
    )

    assert assistant._loop.scheduled == scheduled_before


# --------------------------------------------------------------------------
# WebSocket path
# --------------------------------------------------------------------------


def _make_ws_state(config, detector, vad):
    return SimpleNamespace(
        mode="pipeline",
        config={"audio": {}},
        is_recording=False,
        pending_speech_audio=bytearray(),
        pending_speech_commit_task=None,
        pending_speech_end_epoch_ms=None,
        pending_speech_delay_ms=None,
        pending_speech_commit_generation=0,
        pending_speech_detection_task=None,
        pending_speech_infer_started=None,
        get_vad=lambda: vad,
        get_smart_turn=lambda: detector,
    )


class FakeVAD:
    def __init__(self):
        self.calls = 0

    def process_audio(self, _samples):
        self.calls += 1
        script = [b"<|START|>", b"A" * 3200, b"<|END|>"]
        return iter(script)


async def _stream_once(manager, client_id):
    await manager.handle_audio_stream(client_id, [0.1])


@pytest.mark.asyncio
async def test_websocket_complete_verdict_shortens_delay():
    manager = WebSocketManager()
    client_id = "ws-complete"
    config = _detection_config()
    detector = FakeDetector((True, 0.93), config)
    state = _make_ws_state(config, detector, FakeVAD())
    manager.states[client_id] = state

    async def fake_transcribe(*args, **kwargs):
        return None

    async def fake_send_json(*args, **kwargs):
        return None

    manager._transcribe_and_respond_turn = fake_transcribe  # type: ignore[method-assign]
    manager.send_json = fake_send_json  # type: ignore[method-assign]

    await _stream_once(manager, client_id)

    assert state.pending_speech_delay_ms == 2500

    await state.pending_speech_detection_task

    assert detector.calls == 1
    assert state.pending_speech_delay_ms is not None
    assert 0 <= state.pending_speech_delay_ms <= 250

    manager._cancel_pending_speech_commit(state)
    manager._cancel_pending_speech_detection(state)


@pytest.mark.asyncio
async def test_websocket_incomplete_verdict_keeps_long_delay():
    manager = WebSocketManager()
    client_id = "ws-incomplete"
    config = _detection_config()
    detector = FakeDetector((False, 0.1), config)
    state = _make_ws_state(config, detector, FakeVAD())
    manager.states[client_id] = state

    async def fake_transcribe(*args, **kwargs):
        return None

    async def fake_send_json(*args, **kwargs):
        return None

    manager._transcribe_and_respond_turn = fake_transcribe  # type: ignore[method-assign]
    manager.send_json = fake_send_json  # type: ignore[method-assign]

    await _stream_once(manager, client_id)
    await state.pending_speech_detection_task

    assert detector.calls == 1
    assert state.pending_speech_delay_ms == 2500

    manager._cancel_pending_speech_commit(state)
    manager._cancel_pending_speech_detection(state)


@pytest.mark.asyncio
async def test_websocket_uncertain_verdict_uses_uncertain_delay():
    manager = WebSocketManager()
    client_id = "ws-uncertain"
    config = _detection_config()
    detector = FakeDetector((False, 0.30), config)
    state = _make_ws_state(config, detector, FakeVAD())
    manager.states[client_id] = state

    async def fake_transcribe(*args, **kwargs):
        return None

    async def fake_send_json(*args, **kwargs):
        return None

    manager._transcribe_and_respond_turn = fake_transcribe  # type: ignore[method-assign]
    manager.send_json = fake_send_json  # type: ignore[method-assign]

    await _stream_once(manager, client_id)
    assert state.pending_speech_delay_ms == 2500

    await state.pending_speech_detection_task

    assert detector.calls == 1
    assert state.pending_speech_delay_ms is not None
    assert 500 <= state.pending_speech_delay_ms <= 900

    manager._cancel_pending_speech_commit(state)
    manager._cancel_pending_speech_detection(state)


@pytest.mark.asyncio
async def test_websocket_inference_failure_rearms_on_fallback():
    manager = WebSocketManager()
    client_id = "ws-infer-failure"
    config = _detection_config()
    detector = FakeDetector(None, config)
    state = _make_ws_state(config, detector, FakeVAD())
    manager.states[client_id] = state

    async def fake_transcribe(*args, **kwargs):
        return None

    async def fake_send_json(*args, **kwargs):
        return None

    manager._transcribe_and_respond_turn = fake_transcribe  # type: ignore[method-assign]
    manager.send_json = fake_send_json  # type: ignore[method-assign]

    await _stream_once(manager, client_id)
    assert state.pending_speech_delay_ms == 2500

    await state.pending_speech_detection_task

    assert detector.calls == 1
    assert state.pending_speech_delay_ms is not None
    assert 0 <= state.pending_speech_delay_ms <= 700

    manager._cancel_pending_speech_commit(state)
    manager._cancel_pending_speech_detection(state)


@pytest.mark.asyncio
async def test_websocket_known_unavailable_detector_arms_fallback_without_inference():
    manager = WebSocketManager()
    client_id = "ws-unavailable"
    config = _detection_config()
    detector = FakeDetector((True, 0.99), config, available=False)
    state = _make_ws_state(config, detector, FakeVAD())
    manager.states[client_id] = state

    async def fake_transcribe(*args, **kwargs):
        return None

    async def fake_send_json(*args, **kwargs):
        return None

    manager._transcribe_and_respond_turn = fake_transcribe  # type: ignore[method-assign]
    manager.send_json = fake_send_json  # type: ignore[method-assign]

    await _stream_once(manager, client_id)

    assert detector.calls == 0
    assert state.pending_speech_detection_task is None
    assert state.pending_speech_delay_ms == 700

    manager._cancel_pending_speech_commit(state)


@pytest.mark.asyncio
async def test_websocket_disabled_turn_detection_uses_fixed_delay():
    manager = WebSocketManager()
    client_id = "ws-disabled"
    config = _detection_config(enabled=False)
    detector = FakeDetector((True, 0.99), config)
    state = _make_ws_state(config, detector, FakeVAD())
    manager.states[client_id] = state

    async def fake_transcribe(*args, **kwargs):
        return None

    async def fake_send_json(*args, **kwargs):
        return None

    manager._transcribe_and_respond_turn = fake_transcribe  # type: ignore[method-assign]
    manager.send_json = fake_send_json  # type: ignore[method-assign]

    await _stream_once(manager, client_id)

    assert detector.calls == 0
    assert state.pending_speech_detection_task is None
    assert state.pending_speech_delay_ms == 700

    manager._cancel_pending_speech_commit(state)