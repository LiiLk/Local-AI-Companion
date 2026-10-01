import asyncio
from concurrent.futures import Future
import json
import logging
import threading
import time
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import src.assistant.app as assistant_app
from src.assistant.app import (
    CURRENT_DESKTOP_TURN_ID,
    DesktopBridgeServer,
    Live2DAssistant,
    resolve_turn_timeout_sec,
)
from src.assistant.audio_service import MicState
from src.assistant.conversation_pipeline import AudioPayload, ConversationPipeline


class FakeWindow:
    def __init__(self):
        self.calls = []

    def evaluate_js(self, code: str):
        self.calls.append(code)


class FakeDispatchWindow(FakeWindow):
    def __init__(self):
        super().__init__()
        self.events = []

    def dispatch_frontend_event(self, event_name: str, *args):
        self.events.append((event_name, args))


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
        self._capture_unavailable = False

    def set_processing(self, processing: bool):
        self.processing_calls.append(processing)
        self.state.value = "processing" if processing else "listening"

    def toggle_mute(self):
        self.state.value = "listening"


def test_desktop_bridge_rejects_opaque_and_file_origins():
    assert DesktopBridgeServer._is_origin_allowed(None)
    assert DesktopBridgeServer._is_origin_allowed("http://127.0.0.1:8765")
    assert DesktopBridgeServer._is_origin_allowed("tauri://localhost")
    assert not DesktopBridgeServer._is_origin_allowed("https://evil.example")
    assert not DesktopBridgeServer._is_origin_allowed("null")
    assert not DesktopBridgeServer._is_origin_allowed("file:///tmp/companion.html")


def test_bridge_acknowledges_quit_before_requesting_shutdown():
    events = []

    class Assistant:
        def request_shutdown(self, source):
            events.append(("shutdown", source))

    class WebSocket:
        async def send(self, payload):
            events.append(("send", payload))

    server = DesktopBridgeServer(Assistant())
    message = {
        "type": "command",
        "name": "quit",
        "request_id": "quit-1",
    }

    asyncio.run(server._handle_message(WebSocket(), json.dumps(message)))

    assert events[0][0] == "send"
    payload = json.loads(events[0][1])
    assert payload["ok"] is True
    assert payload["result"]["status"] == "stopping"
    assert events[1] == ("shutdown", "bridge")


def test_bridge_requests_shutdown_when_quit_acknowledgement_fails():
    events = []

    class Assistant:
        def request_shutdown(self, source):
            events.append(("shutdown", source))

    class WebSocket:
        async def send(self, payload):
            raise ConnectionError("client disconnected")

    server = DesktopBridgeServer(Assistant())
    message = {
        "type": "command",
        "name": "quit",
        "request_id": "quit-1",
    }

    with pytest.raises(ConnectionError, match="client disconnected"):
        asyncio.run(server._handle_message(WebSocket(), json.dumps(message)))

    assert events == [("shutdown", "bridge")]


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
    assistant._microphone_degraded_reason = None
    assistant._runtime_error = None
    assistant.audio_service = FakeAudioService()
    assistant.config = {
        "mode": "pipeline",
        "character": {"name": "Starling"},
        "audio": {},
        "asr": {"min_audio_ms": 0},
    }
    assistant.pipeline = SimpleNamespace(
        llm=SimpleNamespace(model="test-llm", degraded_reason=None),
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


def test_desktop_preload_profiles_early_rvc_spawn(monkeypatch):
    assistant = _make_assistant()
    assistant._running = True
    assistant.audio_service = None
    events = []
    marks = []
    runtime = SimpleNamespace(
        llm=object(), asr=object(), tts=object(), rvc=object(),
        spawn_rvc_worker=lambda: events.append("rvc_spawn"),
        preload_llm=lambda: events.append("llm"),
        preload_asr=lambda: events.append("asr"),
        preload_tts=lambda: events.append("tts"),
        preload_rvc=lambda: events.append("rvc_ready_and_warmup"),
        collect_degraded_reason=lambda **_kwargs: None,
    )
    assistant._pipeline_runtime = runtime
    monkeypatch.setattr(assistant, "_mark_startup_step", marks.append)
    monkeypatch.setattr(assistant, "_finish_startup_profile", lambda state: marks.append(state))
    monkeypatch.setattr(assistant, "_turn_detection_active", lambda: False)
    monkeypatch.setattr(assistant, "_set_backend_health", lambda **_kwargs: None)
    monkeypatch.setattr(assistant, "get_runtime_state", lambda: {})

    assistant._preload_models_and_start_audio()

    assert events == ["rvc_spawn", "llm", "asr", "tts", "rvc_ready_and_warmup"]
    assert "preload_rvc_spawn_start" in marks
    assert marks.index("preload_rvc_spawn_done") < marks.index("preload_llm_start")
    assert marks[-1] == "ready"


@pytest.mark.parametrize("abort", ["error", "shutdown"])
def test_desktop_preload_abort_closes_runtime_after_early_spawn(monkeypatch, abort):
    assistant = _make_assistant()
    assistant._running = True
    assistant._loop = None
    assistant.audio_service = None
    events = []

    def preload_llm():
        events.append("llm")
        if abort == "error":
            raise RuntimeError("LLM unavailable")
        assistant._shutdown_requested.set()

    async def close():
        events.append("close")

    assistant._pipeline_runtime = SimpleNamespace(
        spawn_rvc_worker=lambda: events.append("spawn"),
        preload_llm=preload_llm, close=close,
    )
    monkeypatch.setattr(assistant, "_mark_startup_step", lambda _step: None)
    monkeypatch.setattr(assistant, "_finish_startup_profile", events.append)
    monkeypatch.setattr(assistant, "_set_backend_health", lambda **_kwargs: None)

    assistant._preload_models_and_start_audio()

    assert events == ["spawn", "llm", "close", abort]


def test_request_shutdown_is_idempotent_and_closes_window():
    assistant = _make_assistant()
    assistant._running = True
    close_calls = []
    assistant._window.close = lambda: close_calls.append("close")

    first = assistant.request_shutdown("test")
    second = assistant.request_shutdown("test-again")

    assert first == {"status": "stopping"}
    assert second == {"status": "stopping"}
    assert assistant._running is False
    assert assistant._shutdown_requested.is_set()
    assert close_calls == ["close"]


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


@pytest.mark.parametrize("after_commit", [False, True])
def test_stop_waits_for_asr_idle_after_cancel_before_runtime_close(monkeypatch, after_commit):
    assistant = _make_assistant()
    events = []
    budgets = []
    active_turn = Future()
    speculation = Future()
    speculation.set_result(None)
    assistant._speculative_asr = None if after_commit else (b"audio", speculation)
    assistant._speculative_asr_finished_at = 1.0
    assistant._speculative_asr_started_at = 0.5
    assistant._speculative_asr_skip_reason = None
    assistant._active_response_future = active_turn
    assistant.pipeline = ConversationPipeline.__new__(ConversationPipeline)
    assistant.pipeline.cancel_active_run = lambda reason: events.append("cancel_turn")

    async def wait_for_asr_idle():
        assert active_turn.cancelled()
        assert assistant._speculative_asr is None
        await asyncio.sleep(0)
        events.append("asr_idle")

    async def close():
        events.append("runtime_close")

    assistant.pipeline.wait_for_asr_idle = wait_for_asr_idle
    assistant._pipeline_runtime = SimpleNamespace(close=close)
    assistant._running = True
    assistant._window = None
    assistant.audio_service = None
    assistant._hotkey_listener = None
    assistant._hybrid_ui_server = None
    assistant._loop_thread = None
    assistant._loop.is_running = lambda: True
    assistant._loop.stop = lambda: None

    class CompletedFuture(Future):
        def result(self, timeout=None):
            budgets.append(timeout)
            return super().result(timeout)

    def run_inline(coro, loop):
        future = CompletedFuture()
        future.set_result(asyncio.run(coro))
        return future

    monkeypatch.setattr(asyncio, "run_coroutine_threadsafe", run_inline)
    assistant.stop()

    assert assistant._speculative_asr is None
    assert assistant._speculative_asr_started_at is None
    assert events == ["cancel_turn", "asr_idle", "runtime_close"]
    assert budgets[0] == (
        assistant_app._ACTIVE_TURN_SHUTDOWN_TIMEOUT_SEC
        + assistant_app._PIPELINE_RUNTIME_CLOSE_TIMEOUT_SEC
        + 1.0
    )


def test_shutdown_logs_asr_idle_timeout_and_returns(monkeypatch, caplog):
    assistant = _make_assistant()
    assistant.pipeline = ConversationPipeline.__new__(ConversationPipeline)
    wait_cancelled = []

    async def wait_for_asr_idle():
        try:
            await asyncio.Event().wait()
        finally:
            wait_cancelled.append(True)

    assistant.pipeline.wait_for_asr_idle = wait_for_asr_idle
    monkeypatch.setattr(assistant_app, "_PIPELINE_RUNTIME_CLOSE_TIMEOUT_SEC", 0.01)

    with caplog.at_level(logging.ERROR):
        asyncio.run(assistant._cancel_active_turn_for_shutdown("test-shutdown", timeout_sec=0.1))

    assert wait_cancelled == [True]
    assert assistant._speculative_asr is None
    assert any(
        record.levelno == logging.ERROR
        and "ASR still running after 0.01s, closing runtime anyway" in record.getMessage()
        for record in caplog.records
    )


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


def _arm_barge_in(assistant, turn_id):
    events = []
    assistant.config["audio"]["allow_barge_in"] = True
    assistant._assistant_busy = lambda: True
    assistant._active_turn_id = turn_id
    assistant._interrupt_current_turn = (
        lambda reason="interrupt": events.append(("interrupt", reason)) or {}
    )
    assistant._dispatch_frontend_event = (
        lambda event_name, *args: events.append((event_name, args))
    )
    return events


def test_barge_in_requeues_unspoken_turn_audio():
    assistant = _make_assistant()
    _arm_barge_in(assistant, 5)
    first_turn_audio = b"A" * 3200
    assistant._inflight_turn_audio = first_turn_audio
    assistant._inflight_turn_audio_turn_id = 5
    assistant._pending_speech_audio.extend(b"B" * 1600)

    assistant._on_speech_start()

    assert assistant._pending_speech_audio == bytearray(first_turn_audio + b"B" * 1600)
    assert assistant._inflight_turn_audio is None
    assert assistant._inflight_turn_audio_turn_id is None


def test_barge_in_does_not_requeue_once_turn_audio_started():
    assistant = _make_assistant()
    _arm_barge_in(assistant, 5)
    # The turn already produced response audio, so its input is not re-merged.
    assistant._inflight_turn_audio = None
    assistant._inflight_turn_audio_turn_id = None
    assistant._pending_speech_audio.extend(b"B" * 1600)

    assistant._on_speech_start()

    assert assistant._pending_speech_audio == bytearray(b"B" * 1600)


def test_barge_in_requeue_respects_thirty_second_cap():
    assistant = _make_assistant()
    _arm_barge_in(assistant, 5)
    assistant._inflight_turn_audio = b"A" * (30 * 16000 * 2 + 2)
    assistant._inflight_turn_audio_turn_id = 5
    assistant._pending_speech_audio.extend(b"B" * 1600)

    assistant._on_speech_start()

    assert assistant._pending_speech_audio == bytearray(b"B" * 1600)
    assert assistant._inflight_turn_audio is None


def test_barge_in_requeues_audio_when_real_future_cancel_runs_done_callback(monkeypatch):
    assistant = _make_assistant()
    assistant.config["audio"]["allow_barge_in"] = True
    assistant._assistant_busy = lambda: True
    assistant._active_turn_id = 5
    assistant._latest_audio_turn_id = 5
    assistant._turn_timeout_sec = 5
    assistant._dispatch_frontend_event = lambda *args: None

    first_turn_audio = b"A" * 3200
    assistant._inflight_turn_audio = first_turn_audio
    assistant._inflight_turn_audio_turn_id = 5
    assistant._pending_speech_audio.extend(b"B" * 1600)

    pending = Future()

    def fake_run_coroutine_threadsafe(coro, loop):
        coro.close()
        return pending

    monkeypatch.setattr(asyncio, "run_coroutine_threadsafe", fake_run_coroutine_threadsafe)

    assistant._start_turn(5, lambda: asyncio.sleep(0), source="speech")
    assert assistant._active_response_future is pending

    assistant._on_speech_start()

    # A real concurrent.futures.Future runs its done callback synchronously on
    # cancel; the re-merge must still see the turn's input buffered.
    assert pending.cancelled() is True
    assert assistant._pending_speech_audio == bytearray(first_turn_audio + b"B" * 1600)
    assert assistant._inflight_turn_audio is None
    assert assistant._inflight_turn_audio_turn_id is None


def test_commit_retains_audio_until_first_response_audio():
    assistant = _make_assistant()
    captured = {}

    assistant.pipeline.process_speech = lambda audio_bytes, **kwargs: None
    assistant._start_turn = lambda turn_id, runner, source: captured.update(turn_id=turn_id)

    assistant._on_speech_start()
    assistant._on_speech_detected(b"A" * 3200)
    assistant._on_speech_end()

    _delay, handle = assistant._loop.scheduled[-1]
    handle.callback()

    assert assistant._inflight_turn_audio == b"A" * 3200
    assert assistant._inflight_turn_audio_turn_id == captured["turn_id"]


def test_commit_pending_speech_records_latency_for_omni_pipeline(monkeypatch):
    """Omni/gemma turns must start and finish a latency turn like the pipeline."""
    from src.utils.turn_latency import TurnLatencyTracker

    assistant = _make_assistant()
    tracker = TurnLatencyTracker()
    monkeypatch.setattr(
        "src.assistant.app.get_turn_latency_tracker", lambda: tracker
    )

    async def process_speech(audio_bytes):
        tracker.mark("first_audio_out")
        return "ok"

    assistant.pipeline = SimpleNamespace(process_speech=process_speech)
    captured = {}
    assistant._start_turn = lambda turn_id, runner, source: captured.update(
        turn_id=turn_id, runner=runner
    )

    assistant._on_speech_start()
    assistant._on_speech_detected(b"A" * 3200)
    assistant._on_speech_end()

    _delay, handle = assistant._loop.scheduled[-1]
    handle.callback()

    assert asyncio.run(captured["runner"]()) == "ok"
    assert tracker.summary()["count"] == 1


def test_first_response_audio_clears_retained_turn_audio():
    assistant = _make_assistant()
    assistant._inflight_turn_audio = b"A" * 3200
    assistant._inflight_turn_audio_turn_id = 7

    payload = AudioPayload(
        audio_bytes=b"\x00\x00" * 240,
        audio_base64="ZmFrZQ==",
        wav_bytes=None,
        volumes=[0.1],
        duration_ms=200,
        sample_rate=24000,
        text="hi",
    )

    token = CURRENT_DESKTOP_TURN_ID.set(7)
    try:
        asyncio.run(assistant._on_audio_ready(payload))
    finally:
        CURRENT_DESKTOP_TURN_ID.reset(token)

    assert assistant._inflight_turn_audio is None
    assert assistant._inflight_turn_audio_turn_id is None


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


def test_dispatch_frontend_event_uses_structured_shell_without_duplicate_js():
    assistant = _make_assistant()
    assistant._window = FakeDispatchWindow()

    assistant._dispatch_frontend_event("onAudioReady", {"audio": "ZmFrZQ=="})

    assert assistant._window.events == [
        ("onAudioReady", ({"audio": "ZmFrZQ=="},)),
    ]
    assert assistant._window.calls == []


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
    assert runtime["active_llm_model"] == "test-llm"
    assert runtime["active_tts_provider"] == "qwen3"
    assert runtime["degraded_reason"] is None
    assert runtime["runtime_error"] is None


def test_get_runtime_state_exposes_rvc_startup_degradation():
    assistant = _make_assistant()
    assistant.config["tts"] = {"rvc": {"enabled": True}}
    owner = assistant_app.create_pipeline_runtime(assistant.config)
    assistant._pipeline_runtime = owner

    def fail_preload():
        raise TimeoutError("startup timed out after 90s")

    owner.rvc = SimpleNamespace(preload=fail_preload, close=lambda: None)
    owner.preload_rvc()
    runtime = assistant.get_runtime_state()

    assert runtime["backend_state"] == "degraded"
    assert runtime["degraded_reason"] == (
        "RVC unavailable: voice conversion disabled (startup timed out after 90s)"
    )
    assert owner.rvc is None


def test_listening_state_clears_microphone_degradation_after_capture_recovers():
    assistant = _make_assistant()
    assistant._backend_state = "degraded"
    assistant._microphone_degraded_reason = "Microphone unavailable: no input device"

    assistant._on_mic_state_change(MicState.LISTENING)
    runtime = assistant.get_runtime_state()

    assert runtime["backend_state"] == "ready"
    assert runtime["degraded_reason"] is None
    assert assistant._microphone_degraded_reason is None


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
