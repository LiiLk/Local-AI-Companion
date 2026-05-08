import asyncio
from concurrent.futures import Future
import threading
from types import SimpleNamespace
from unittest.mock import patch

from src.assistant.app import CURRENT_DESKTOP_TURN_ID, Live2DAssistant, resolve_turn_timeout_sec
from src.assistant.conversation_pipeline import AudioPayload


class FakeWindow:
    def __init__(self):
        self.calls = []

    def evaluate_js(self, code: str):
        self.calls.append(code)


class FakeFuture:
    def __init__(self, done: bool = False):
        self._done = done
        self.cancelled = False

    def done(self) -> bool:
        return self._done

    def cancel(self):
        self.cancelled = True
        self._done = True


class FakeThreadsafeFuture:
    def __init__(self):
        self.callbacks = []

    def done(self) -> bool:
        return False

    def add_done_callback(self, callback):
        self.callbacks.append(callback)


class FakeHandle:
    def __init__(self, callback=None):
        self.callback = callback
        self.cancelled = False

    def cancel(self):
        self.cancelled = True


class FakeLoop:
    def __init__(self):
        self.scheduled = []

    def call_later(self, delay, callback):
        handle = FakeHandle(callback)
        self.scheduled.append((delay, handle))
        return handle

    def call_soon_threadsafe(self, callback, *args):
        callback(*args)

    def is_closed(self):
        return False


class FakeBridgeServer:
    def __init__(self):
        self.events = []

    def emit_frontend_event_sync(self, event_name: str, *args):
        self.events.append((event_name, args))


class FakeAudioService:
    def __init__(self):
        self.state = SimpleNamespace(value="listening")
        self.processing_calls = []

    def set_processing(self, processing: bool):
        self.processing_calls.append(processing)
        self.state.value = "processing" if processing else "listening"


def _make_assistant() -> Live2DAssistant:
    assistant = Live2DAssistant.__new__(Live2DAssistant)
    assistant._window = FakeWindow()
    assistant._bridge_server = None
    assistant._active_response_future = None
    assistant._active_turn_id = None
    assistant._latest_audio_turn_id = None
    assistant._playback_deadline = 0.0
    assistant._playback_release_handle = None
    assistant._audio_processing_owned = False
    assistant._drop_current_speech = False
    assistant._speech_active = False
    assistant._pending_speech_audio = bytearray()
    assistant._pending_speech_commit_handle = None
    assistant._pending_speech_lock = threading.Lock()
    assistant._speech_commit_delay_ms = 700
    assistant._debug_visible = False
    assistant._turn_counter = 0
    assistant._backend_state = "ready"
    assistant._degraded_reason = None
    assistant._runtime_error = None
    assistant.audio_service = FakeAudioService()
    assistant.config = {"mode": "pipeline", "character": {"name": "March 7th"}, "audio": {}}
    assistant.pipeline = SimpleNamespace(
        llm=SimpleNamespace(model="qwen3.5:4b", degraded_reason=None),
        tts=SimpleNamespace(active_provider_name="qwen3", degraded_reason=None),
        process_speech=None,
        _current_language_code="en",
    )
    assistant._omni_pipeline = None
    assistant._gemma_pipeline = None
    assistant._loop = FakeLoop()
    assistant._shutdown_requested = threading.Event()
    assistant._preload_runtime_lock = threading.Lock()
    return assistant


def test_interrupt_current_turn_cancels_future_and_stops_playback():
    assistant = _make_assistant()
    future = FakeFuture(done=False)
    cancel_reasons = []
    assistant.pipeline = SimpleNamespace(cancel_active_run=lambda reason: cancel_reasons.append(reason))
    assistant._active_response_future = future
    assistant._active_turn_id = 3
    assistant._latest_audio_turn_id = 3
    assistant._playback_deadline = 999999999.0

    runtime = assistant._interrupt_current_turn("test")

    assert future.cancelled is True
    assert cancel_reasons == ["test"]
    assert assistant._latest_audio_turn_id is None
    assert assistant._playback_deadline == 0.0
    assert runtime["interrupted"] is True
    assert any("window.onPlaybackStop?.(3)" in call for call in assistant._window.calls)


def test_shutdown_cancels_active_turn_and_clears_pending_audio():
    assistant = _make_assistant()
    future = Future()
    cancel_reasons = []
    assistant.pipeline = SimpleNamespace(cancel_active_run=lambda reason: cancel_reasons.append(reason))
    assistant._active_response_future = future
    assistant._active_turn_id = 9
    assistant._latest_audio_turn_id = 9
    assistant._playback_deadline = 999999999.0
    assistant._pending_speech_audio.extend(b"pending")
    assistant._pending_speech_commit_handle = FakeHandle()

    asyncio.run(assistant._cancel_active_turn_for_shutdown("test-shutdown", timeout_sec=0.1))

    assert future.cancelled() is True
    assert cancel_reasons == ["test-shutdown"]
    assert assistant._active_response_future is None
    assert assistant._active_turn_id is None
    assert assistant._latest_audio_turn_id is None
    assert assistant._playback_deadline == 0.0
    assert assistant._pending_speech_audio == bytearray()
    assert any("window.onPlaybackStop?.(9)" in call for call in assistant._window.calls)


def test_on_speech_start_interrupts_when_busy():
    assistant = _make_assistant()
    events = []

    assistant.config["audio"]["allow_barge_in"] = True
    assistant._assistant_busy = lambda: True
    assistant._active_turn_id = 5
    assistant._interrupt_current_turn = lambda reason="interrupt": events.append(("interrupt", reason)) or {}
    assistant._dispatch_frontend_event = lambda event_name, *args: events.append((event_name, args))

    assistant._on_speech_start()

    assert ("interrupt", "barge-in") in events
    assert ("onSpeechStart", (5,)) in events


def test_on_speech_start_ignores_when_busy_and_barge_in_disabled():
    assistant = _make_assistant()
    events = []

    assistant._assistant_busy = lambda: True
    assistant._interrupt_current_turn = lambda reason="interrupt": events.append(("interrupt", reason)) or {}
    assistant._dispatch_frontend_event = lambda event_name, *args: events.append((event_name, args))

    assistant._on_speech_start()

    assert assistant._drop_current_speech is True
    assert events == []


def test_on_speech_detected_discards_buffer_marked_for_drop():
    assistant = _make_assistant()
    events = []
    assistant._drop_current_speech = True
    assistant._start_turn = lambda *_args, **_kwargs: events.append("start_turn")

    assistant._on_speech_detected(b"\x00\x00" * 1024)

    assert events == []


def test_on_speech_detected_buffers_until_commit_window_expires():
    assistant = _make_assistant()
    captured = {}

    async def process_speech(audio_bytes: bytes):
        captured["audio"] = audio_bytes
        return "ok"

    assistant.pipeline.process_speech = process_speech
    assistant._start_turn = lambda turn_id, runner, source: captured.update(
        turn_id=turn_id,
        source=source,
        runner=runner,
    )

    assistant._on_speech_start()
    assistant._on_speech_detected(b"A" * 3200)
    assistant._on_speech_end()

    assert "runner" not in captured
    assert len(assistant._loop.scheduled) == 1

    _delay, handle = assistant._loop.scheduled[-1]
    handle.callback()
    asyncio.run(captured["runner"]())

    assert captured["source"] == "speech"
    assert captured["audio"] == b"A" * 3200
    assert assistant._pending_speech_audio == bytearray()


def test_resumed_speech_cancels_pending_commit_and_merges_segments():
    assistant = _make_assistant()
    captured = {}

    async def process_speech(audio_bytes: bytes):
        captured["audio"] = audio_bytes
        return "ok"

    assistant.pipeline.process_speech = process_speech
    assistant._start_turn = lambda turn_id, runner, source: captured.update(
        turn_id=turn_id,
        source=source,
        runner=runner,
    )

    first_segment = b"A" * 3200
    second_segment = b"B" * 1600

    assistant._on_speech_start()
    assistant._on_speech_detected(first_segment)
    assistant._on_speech_end()

    _delay, first_handle = assistant._loop.scheduled[-1]
    assert first_handle.cancelled is False

    assistant._on_speech_start()
    assert first_handle.cancelled is True

    assistant._on_speech_detected(second_segment)
    assistant._on_speech_end()

    _delay, second_handle = assistant._loop.scheduled[-1]
    assert second_handle is not first_handle

    second_handle.callback()
    asyncio.run(captured["runner"]())

    assert captured["source"] == "speech"
    assert captured["audio"] == first_segment + second_segment


def test_on_audio_ready_includes_trace_and_tts_metrics():
    assistant = _make_assistant()
    payload = AudioPayload(
        audio_bytes=b"\x00\x00" * 240,
        audio_base64="ZmFrZQ==",
        wav_bytes=None,
        volumes=[0.1, 0.2],
        duration_ms=850,
        sample_rate=24000,
        text="Hello from desktop",
        expression="happy",
        tts_metrics={"synth_ms": 42.0},
        trace={"speech_end_epoch_ms": 111, "asr_done_epoch_ms": 222},
    )

    token = CURRENT_DESKTOP_TURN_ID.set(7)
    try:
        asyncio.run(assistant._on_audio_ready(payload))
    finally:
        CURRENT_DESKTOP_TURN_ID.reset(token)

    assert assistant._latest_audio_turn_id == 7
    assert assistant._playback_deadline > 0.0
    assert assistant.audio_service.processing_calls == [True]
    assert any('"turn_id": 7' in call for call in assistant._window.calls)
    assert any('"backend_audio_ready_epoch_ms"' in call for call in assistant._window.calls)
    assert any('"speech_end_epoch_ms": 111' in call for call in assistant._window.calls)
    assert any('"tts_metrics": {"synth_ms": 42.0}' in call for call in assistant._window.calls)
    assert any("window.onAudioReady?.(" in call for call in assistant._window.calls)


def test_on_audio_ready_extends_deadline_for_queued_audio():
    assistant = _make_assistant()
    assistant._latest_audio_turn_id = 7
    assistant._playback_deadline = 100.0

    payload = AudioPayload(
        audio_bytes=b"",
        audio_base64="ZmFrZQ==",
        wav_bytes=None,
        volumes=[],
        duration_ms=3000,
        sample_rate=24000,
        text="queued chunk",
    )

    token = CURRENT_DESKTOP_TURN_ID.set(7)
    try:
        with patch("src.assistant.app.time.monotonic", return_value=96.0):
            asyncio.run(assistant._on_audio_ready(payload))
    finally:
        CURRENT_DESKTOP_TURN_ID.reset(token)

    assert assistant._playback_deadline == 103.0


def test_sync_audio_capture_mode_blocks_and_releases_mic():
    assistant = _make_assistant()
    assistant._active_response_future = FakeFuture(done=False)

    assistant._sync_audio_capture_mode()
    assert assistant.audio_service.processing_calls == [True]
    assert assistant.audio_service.state.value == "processing"

    assistant._active_response_future = None
    assistant._playback_deadline = 0.0
    assistant._latest_audio_turn_id = None

    assistant._sync_audio_capture_mode()
    assert assistant.audio_service.processing_calls == [True, False]
    assert assistant.audio_service.state.value == "listening"


def test_dispatch_frontend_event_also_reaches_bridge_server():
    assistant = _make_assistant()
    bridge = FakeBridgeServer()
    assistant._bridge_server = bridge

    assistant._dispatch_frontend_event("onMicStateChange", "muted")

    assert ("onMicStateChange", ("muted",)) in bridge.events
    assert any('window.onMicStateChange?.("muted")' in call for call in assistant._window.calls)


def test_start_turn_cancels_previous_pipeline_without_waiting(monkeypatch):
    assistant = _make_assistant()
    assistant._loop = object()
    assistant._turn_timeout_sec = 5
    previous_future = FakeFuture(done=False)
    next_future = FakeThreadsafeFuture()
    cancel_reasons = []
    captured = {}

    assistant.pipeline = SimpleNamespace(cancel_active_run=lambda reason: cancel_reasons.append(reason))
    assistant._active_response_future = previous_future
    assistant._active_turn_id = 4

    async def runner():
        captured["runner_executed"] = True
        return "ok"

    def fake_run_coroutine_threadsafe(coro, loop):
        captured["coro"] = coro
        captured["loop"] = loop
        return next_future

    def fail_wrap_future(_future):
        raise AssertionError("wrap_future should not be called for superseded turns")

    monkeypatch.setattr(asyncio, "run_coroutine_threadsafe", fake_run_coroutine_threadsafe)
    monkeypatch.setattr(asyncio, "wrap_future", fail_wrap_future)

    assistant._start_turn(5, runner, source="speech")
    result = asyncio.run(captured["coro"])

    assert result == "ok"
    assert captured["runner_executed"] is True
    assert captured["loop"] is assistant._loop
    assert previous_future.cancelled is True
    assert cancel_reasons == ["superseded by speech turn 5"]


def test_get_runtime_state_exposes_backend_health_fields():
    assistant = _make_assistant()

    runtime = assistant.get_runtime_state()

    assert runtime["backend_state"] == "ready"
    assert runtime["active_language"] == "en"
    assert runtime["active_llm_model"] == "qwen3.5:4b"
    assert runtime["active_tts_provider"] == "qwen3"
    assert runtime["degraded_reason"] is None
    assert runtime["runtime_error"] is None


def test_submit_text_returns_warming_up_until_backend_ready():
    assistant = _make_assistant()
    assistant._loop = object()
    assistant._backend_state = "warming_up"
    assistant.pipeline.process_text = lambda _text: None

    runtime = assistant.submit_text("Hello")

    assert runtime["status"] == "warming_up"
    assert runtime["backend_state"] == "warming_up"


def test_resolve_audio_start_muted_defaults_to_bridge_muted():
    assistant = _make_assistant()
    assistant._bridge_only = True
    assistant.config["audio"] = {}

    assert assistant._resolve_audio_start_muted() is True


def test_resolve_audio_start_muted_bridge_override_can_unmute():
    assistant = _make_assistant()
    assistant._bridge_only = True
    assistant.config["audio"] = {"bridge_start_muted": False}

    assert assistant._resolve_audio_start_muted() is False


def test_resolve_turn_timeout_uses_openrouter_request_timeout():
    config = {
        "llm": {
            "provider": "openrouter",
            "openrouter": {
                "request_timeout_sec": 180,
            },
        }
    }

    assert resolve_turn_timeout_sec(config) == 210
