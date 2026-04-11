import asyncio
from types import SimpleNamespace

from src.assistant.app import CURRENT_DESKTOP_TURN_ID, Live2DAssistant
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


class FakeBridgeServer:
    def __init__(self):
        self.events = []

    def emit_frontend_event_sync(self, event_name: str, *args):
        self.events.append((event_name, args))


def _make_assistant() -> Live2DAssistant:
    assistant = Live2DAssistant.__new__(Live2DAssistant)
    assistant._window = FakeWindow()
    assistant._bridge_server = None
    assistant._active_response_future = None
    assistant._active_turn_id = None
    assistant._latest_audio_turn_id = None
    assistant._playback_deadline = 0.0
    assistant._debug_visible = False
    assistant._turn_counter = 0
    assistant.audio_service = SimpleNamespace(state=SimpleNamespace(value="listening"))
    assistant.config = {"mode": "pipeline", "character": {"name": "March 7th"}}
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


def test_on_speech_start_interrupts_when_busy():
    assistant = _make_assistant()
    events = []

    assistant._assistant_busy = lambda: True
    assistant._active_turn_id = 5
    assistant._interrupt_current_turn = lambda reason="interrupt": events.append(("interrupt", reason)) or {}
    assistant._dispatch_frontend_event = lambda event_name, *args: events.append((event_name, args))

    assistant._on_speech_start()

    assert ("interrupt", "barge-in") in events
    assert ("onSpeechStart", (5,)) in events


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
    )

    token = CURRENT_DESKTOP_TURN_ID.set(7)
    try:
        asyncio.run(assistant._on_audio_ready(payload))
    finally:
        CURRENT_DESKTOP_TURN_ID.reset(token)

    assert assistant._latest_audio_turn_id == 7
    assert assistant._playback_deadline > 0.0
    assert any('"turn_id": 7' in call for call in assistant._window.calls)
    assert any('"backend_audio_ready_epoch_ms"' in call for call in assistant._window.calls)
    assert any('"tts_metrics": {"synth_ms": 42.0}' in call for call in assistant._window.calls)
    assert any("window.onAudioReady?.(" in call for call in assistant._window.calls)


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
