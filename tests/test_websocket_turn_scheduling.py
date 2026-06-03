import asyncio
import threading
import time
from types import SimpleNamespace

import pytest

from src.server import websocket as websocket_module
from src.server.websocket import WebSocketManager


@pytest.mark.asyncio
async def test_new_turn_cancels_previous_turn_for_same_client():
    manager = WebSocketManager()
    client_id = "client-test"
    cancel_reasons: list[str] = []
    state = SimpleNamespace(
        response_task=None,
        response_lock=asyncio.Lock(),
        cancel_active_generation=lambda reason: cancel_reasons.append(reason),
    )
    manager.states[client_id] = state

    events: list[str] = []

    async def fake_stop_audio(cid: str) -> None:
        assert cid == client_id
        events.append("stop_audio")

    manager._stop_client_audio = fake_stop_audio  # type: ignore[method-assign]

    started = asyncio.Event()
    allow_cancel = asyncio.Event()

    async def long_turn() -> None:
        events.append("long_start")
        started.set()
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            events.append("long_cancelled")
            allow_cancel.set()
            raise

    async def short_turn() -> None:
        await allow_cancel.wait()
        events.append("short_start")

    first_task = await manager._schedule_turn(client_id, long_turn())
    assert first_task is not None
    await started.wait()

    second_task = await manager._schedule_turn(client_id, short_turn())
    assert second_task is not None
    await second_task

    assert events == [
        "long_start",
        "stop_audio",
        "long_cancelled",
        "short_start",
    ]
    assert cancel_reasons == ["superseded turn"]
    assert state.response_task is None


@pytest.mark.asyncio
async def test_interrupt_cancels_active_turn_and_resets_vad():
    manager = WebSocketManager()
    client_id = "client-interrupt"

    class FakeVAD:
        def __init__(self):
            self.reset_calls = 0

        def reset(self):
            self.reset_calls += 1

    vad = FakeVAD()
    cancel_reasons: list[str] = []
    state = SimpleNamespace(
        response_task=None,
        response_lock=asyncio.Lock(),
        vad=vad,
        cancel_active_generation=lambda reason: cancel_reasons.append(reason),
    )
    manager.states[client_id] = state

    events: list[str] = []
    started = asyncio.Event()

    async def fake_stop_audio(cid: str) -> None:
        assert cid == client_id
        events.append("stop_audio")

    manager._stop_client_audio = fake_stop_audio  # type: ignore[method-assign]

    async def long_turn() -> None:
        events.append("long_start")
        started.set()
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            events.append("long_cancelled")
            raise

    task = await manager._schedule_turn(client_id, long_turn())
    assert task is not None
    await started.wait()

    await manager.handle_interrupt(client_id)

    assert events == ["long_start", "long_cancelled", "stop_audio"]
    assert cancel_reasons == ["interrupt"]
    assert state.response_task is None
    assert vad.reset_calls == 1


@pytest.mark.asyncio
async def test_disconnect_stops_audio_and_cleans_up_active_generation():
    manager = WebSocketManager()
    client_id = "client-disconnect"
    events: list[str] = []

    class FakeConnection:
        async def send_json(self, data):
            events.append(data["type"])

    class FakeState:
        async def cleanup(self):
            events.append("cleanup")

    manager.active_connections[client_id] = FakeConnection()
    manager.states[client_id] = FakeState()

    await manager.disconnect(client_id)

    assert events == ["stop_audio", "cleanup"]
    assert client_id not in manager.active_connections
    assert client_id not in manager.states


@pytest.mark.asyncio
async def test_disconnect_cancels_background_preload_before_cleanup():
    manager = WebSocketManager()
    client_id = "client-preload-disconnect"
    events: list[str] = []
    preload_started = asyncio.Event()

    class FakeConnection:
        async def send_json(self, data):
            events.append(data["type"])

    class FakeState:
        async def cleanup(self):
            events.append("cleanup")

    async def slow_preload():
        preload_started.set()
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            events.append("preload_cancelled")
            raise

    preload_task = asyncio.create_task(slow_preload())
    await preload_started.wait()

    manager.active_connections[client_id] = FakeConnection()
    manager.states[client_id] = FakeState()
    manager._preload_tasks[client_id] = preload_task
    manager._preloading[client_id] = True

    await manager.disconnect(client_id)

    assert events == ["stop_audio", "preload_cancelled", "cleanup"]
    assert client_id not in manager.active_connections
    assert client_id not in manager.states
    assert client_id not in manager._preload_tasks
    assert client_id not in manager._preloading


@pytest.mark.asyncio
async def test_manual_pipeline_preload_runs_gpu_steps_sequentially():
    manager = WebSocketManager()
    client_id = "client-manual-preload"
    sent_messages: list[dict] = []

    class FakeConnection:
        async def send_json(self, data):
            sent_messages.append(data)

    class FakeState:
        mode = "pipeline"

        def __init__(self):
            self.config = {
                "llm": {"provider": "gemma"},
                "tts": {"rvc": {"enabled": True}},
            }
            self.vad = None
            self.llm = None
            self.tts = None
            self.asr = None
            self.rvc = None
            self.events: list[str] = []
            self.active_loads = 0
            self.max_active_loads = 0
            self.lock = threading.Lock()

        def pipeline_ready(self):
            return bool(self.llm and self.tts and self.asr and self.rvc)

        def _load(self, name: str):
            with self.lock:
                self.events.append(f"{name}:start")
                self.active_loads += 1
                self.max_active_loads = max(self.max_active_loads, self.active_loads)

            time.sleep(0.01)
            setattr(self, name, object())

            with self.lock:
                self.events.append(f"{name}:end")
                self.active_loads -= 1

        def get_vad(self):
            self._load("vad")
            return self.vad

        def preload_llm(self):
            self._load("llm")
            return self.llm

        def preload_tts(self):
            self._load("tts")
            return self.tts

        def preload_asr(self):
            self._load("asr")
            return self.asr

        def preload_rvc(self):
            self._load("rvc")
            return self.rvc

    state = FakeState()
    manager.active_connections[client_id] = FakeConnection()
    manager.states[client_id] = state

    await manager.preload_models(client_id)

    assert state.max_active_loads == 1
    assert state.events == [
        "vad:start",
        "vad:end",
        "llm:start",
        "llm:end",
        "tts:start",
        "tts:end",
        "asr:start",
        "asr:end",
        "rvc:start",
        "rvc:end",
    ]
    assert [message["type"] for message in sent_messages] == [
        "models_loading",
        "models_ready",
    ]
    assert client_id not in manager._preloading


@pytest.mark.asyncio
async def test_pipeline_audio_stream_merges_segments_before_commit_window():
    manager = WebSocketManager()
    client_id = "client-vad-merge"
    sent_messages: list[dict] = []
    scheduled_audio: list[bytes] = []

    class FakeVAD:
        def __init__(self):
            self.calls = 0

        def process_audio(self, _samples):
            self.calls += 1
            if self.calls == 1:
                return iter([b"<|START|>", b"A" * 3200, b"<|END|>"])
            return iter([b"<|START|>", b"B" * 1600, b"<|END|>"])

    async def fake_transcribe(_client_id, audio_bytes, speech_end_epoch_ms=None):
        scheduled_audio.append(audio_bytes)

    async def fake_schedule(_client_id, turn_coro):
        await turn_coro
        return None

    async def fake_send_json(_client_id, data):
        sent_messages.append(data)

    state = SimpleNamespace(
        mode="pipeline",
        config={"audio": {"speech_commit_delay_ms": 700}},
        is_recording=False,
        pending_speech_audio=bytearray(),
        pending_speech_commit_task=None,
        pending_speech_end_epoch_ms=None,
        get_vad=lambda: vad,
    )
    vad = FakeVAD()
    manager.states[client_id] = state
    manager.send_json = fake_send_json  # type: ignore[method-assign]
    manager._schedule_turn = fake_schedule  # type: ignore[method-assign]
    manager._transcribe_and_respond_turn = fake_transcribe  # type: ignore[method-assign]

    await manager.handle_audio_stream(client_id, [0.1])
    first_task = state.pending_speech_commit_task

    assert first_task is not None
    assert scheduled_audio == []
    assert state.pending_speech_audio == bytearray(b"A" * 3200)

    await manager.handle_audio_stream(client_id, [0.1])
    await asyncio.sleep(0)

    assert first_task.cancelled()
    assert scheduled_audio == []
    assert state.pending_speech_audio == bytearray(b"A" * 3200 + b"B" * 1600)

    manager._cancel_pending_speech_commit(state)
    await manager._commit_pending_speech(client_id, state)

    assert scheduled_audio == [b"A" * 3200 + b"B" * 1600]
    assert state.pending_speech_audio == bytearray()
    assert [message["type"] for message in sent_messages] == [
        "vad_start",
        "vad_end",
        "vad_start",
        "vad_end",
    ]


@pytest.mark.asyncio
async def test_pipeline_audio_stream_committed_pause_stays_separate_turns():
    manager = WebSocketManager()
    client_id = "client-vad-separate"
    scheduled_audio: list[bytes] = []

    class FakeVAD:
        def __init__(self):
            self.calls = 0

        def process_audio(self, _samples):
            self.calls += 1
            if self.calls == 1:
                return iter([b"<|START|>", b"A" * 3200, b"<|END|>"])
            return iter([b"<|START|>", b"B" * 1600, b"<|END|>"])

    async def fake_transcribe(_client_id, audio_bytes, speech_end_epoch_ms=None):
        scheduled_audio.append(audio_bytes)

    async def fake_schedule(_client_id, turn_coro):
        await turn_coro
        return None

    async def fake_send_json(*_args, **_kwargs):
        return None

    state = SimpleNamespace(
        mode="pipeline",
        config={"audio": {"speech_commit_delay_ms": 700}},
        is_recording=False,
        pending_speech_audio=bytearray(),
        pending_speech_commit_task=None,
        pending_speech_end_epoch_ms=None,
        get_vad=lambda: vad,
    )
    vad = FakeVAD()
    manager.states[client_id] = state
    manager.send_json = fake_send_json  # type: ignore[method-assign]
    manager._schedule_turn = fake_schedule  # type: ignore[method-assign]
    manager._transcribe_and_respond_turn = fake_transcribe  # type: ignore[method-assign]

    await manager.handle_audio_stream(client_id, [0.1])
    manager._cancel_pending_speech_commit(state)
    await manager._commit_pending_speech(client_id, state)

    await manager.handle_audio_stream(client_id, [0.1])
    manager._cancel_pending_speech_commit(state)
    await manager._commit_pending_speech(client_id, state)

    assert scheduled_audio == [b"A" * 3200, b"B" * 1600]


@pytest.mark.asyncio
async def test_pipeline_audio_stream_force_commits_when_pending_buffer_reaches_cap(monkeypatch):
    monkeypatch.setattr(websocket_module, "MAX_PENDING_SPEECH_AUDIO_BYTES", 4000)

    manager = WebSocketManager()
    client_id = "client-vad-cap"
    sent_messages: list[dict] = []
    scheduled_audio: list[bytes] = []

    class FakeVAD:
        def __init__(self):
            self.calls = 0

        def process_audio(self, _samples):
            self.calls += 1
            if self.calls == 1:
                return iter([b"<|START|>", b"A" * 3200, b"<|END|>"])
            return iter([b"<|START|>", b"B" * 1600, b"<|END|>"])

    async def fake_transcribe(_client_id, audio_bytes, speech_end_epoch_ms=None):
        scheduled_audio.append(audio_bytes)

    async def fake_schedule(_client_id, turn_coro):
        await turn_coro
        return None

    async def fake_send_json(_client_id, data):
        sent_messages.append(data)

    state = SimpleNamespace(
        mode="pipeline",
        config={"audio": {"speech_commit_delay_ms": 700}},
        is_recording=False,
        pending_speech_audio=bytearray(),
        pending_speech_commit_task=None,
        pending_speech_end_epoch_ms=None,
        get_vad=lambda: vad,
    )
    vad = FakeVAD()
    manager.states[client_id] = state
    manager.send_json = fake_send_json  # type: ignore[method-assign]
    manager._schedule_turn = fake_schedule  # type: ignore[method-assign]
    manager._transcribe_and_respond_turn = fake_transcribe  # type: ignore[method-assign]

    await manager.handle_audio_stream(client_id, [0.1])
    first_task = state.pending_speech_commit_task

    assert first_task is not None
    assert scheduled_audio == []
    assert state.pending_speech_audio == bytearray(b"A" * 3200)

    await manager.handle_audio_stream(client_id, [0.1])
    await asyncio.sleep(0)

    assert first_task.cancelled()
    assert scheduled_audio == [b"A" * 3200]
    assert state.pending_speech_audio == bytearray(b"B" * 1600)

    manager._cancel_pending_speech_commit(state)
    await manager._commit_pending_speech(client_id, state)

    assert scheduled_audio == [b"A" * 3200, b"B" * 1600]
    assert state.pending_speech_audio == bytearray()
    assert [message["type"] for message in sent_messages] == [
        "vad_start",
        "vad_end",
        "vad_start",
        "vad_end",
    ]


@pytest.mark.asyncio
async def test_pipeline_audio_stream_rejects_single_segment_over_pending_cap(monkeypatch):
    monkeypatch.setattr(websocket_module, "MAX_PENDING_SPEECH_AUDIO_BYTES", 1000)

    manager = WebSocketManager()
    client_id = "client-vad-segment-too-long"
    sent_messages: list[dict] = []
    scheduled_audio: list[bytes] = []

    class FakeVAD:
        def process_audio(self, _samples):
            return iter([b"<|START|>", b"A" * 1600, b"<|END|>"])

    async def fake_transcribe(_client_id, audio_bytes, speech_end_epoch_ms=None):
        scheduled_audio.append(audio_bytes)

    async def fake_schedule(_client_id, turn_coro):
        await turn_coro
        return None

    async def fake_send_json(_client_id, data):
        sent_messages.append(data)

    state = SimpleNamespace(
        mode="pipeline",
        config={"audio": {"speech_commit_delay_ms": 700}},
        is_recording=False,
        pending_speech_audio=bytearray(),
        pending_speech_commit_task=None,
        pending_speech_end_epoch_ms=None,
        get_vad=lambda: FakeVAD(),
    )
    manager.states[client_id] = state
    manager.send_json = fake_send_json  # type: ignore[method-assign]
    manager._schedule_turn = fake_schedule  # type: ignore[method-assign]
    manager._transcribe_and_respond_turn = fake_transcribe  # type: ignore[method-assign]

    await manager.handle_audio_stream(client_id, [0.1])

    assert scheduled_audio == []
    assert state.pending_speech_audio == bytearray()
    assert state.pending_speech_commit_task is None
    assert {"type": "error", "message": "Speech segment is too long"} in sent_messages


@pytest.mark.asyncio
async def test_pipeline_audio_stream_force_commit_merges_pending_audio():
    manager = WebSocketManager()
    client_id = "client-vad-force"
    scheduled_audio: list[bytes] = []

    async def fake_transcribe(_client_id, audio_bytes, speech_end_epoch_ms=None):
        scheduled_audio.append(audio_bytes)

    async def fake_schedule(_client_id, turn_coro):
        await turn_coro
        return None

    state = SimpleNamespace(
        mode="pipeline",
        config={"audio": {"speech_commit_delay_ms": 700}},
        is_recording=True,
        pending_speech_audio=bytearray(b"A" * 3200),
        pending_speech_commit_task=None,
        pending_speech_end_epoch_ms=None,
    )
    manager.states[client_id] = state
    manager._schedule_turn = fake_schedule  # type: ignore[method-assign]
    manager._transcribe_and_respond_turn = fake_transcribe  # type: ignore[method-assign]

    await manager._commit_pipeline_speech_now(client_id, state, b"B" * 1600)

    assert scheduled_audio == [b"A" * 3200 + b"B" * 1600]
    assert state.pending_speech_audio == bytearray()
    assert state.is_recording is False


@pytest.mark.asyncio
async def test_handle_clear_clears_pending_pipeline_speech():
    manager = WebSocketManager()
    client_id = "client-clear-pending-speech"
    sent_messages: list[dict] = []
    pending_task = asyncio.create_task(asyncio.sleep(10))

    async def fake_stop_client_audio(_client_id):
        return None

    async def fake_send_json(_client_id, data):
        sent_messages.append(data)

    state = SimpleNamespace(
        config={"character": {"system_prompt": "System"}},
        memory_store=None,
        messages=[{"role": "system", "content": "System"}, {"role": "user", "content": "Hi"}],
        response_task=None,
        pending_speech_audio=bytearray(b"A" * 3200),
        pending_speech_commit_task=pending_task,
        pending_speech_end_epoch_ms=123,
        omni_pipeline=None,
        gemma_pipeline=None,
    )
    manager.states[client_id] = state
    manager._stop_client_audio = fake_stop_client_audio  # type: ignore[method-assign]
    manager.send_json = fake_send_json  # type: ignore[method-assign]

    await manager.handle_clear(client_id)
    await asyncio.sleep(0)

    assert pending_task.cancelled()
    assert state.pending_speech_audio == bytearray()
    assert state.pending_speech_commit_task is None
    assert state.pending_speech_end_epoch_ms is None
    assert len(state.messages) == 1
    assert state.messages[0].role == "system"
    assert state.messages[0].content == "System"
    assert sent_messages == [
        {"type": "cleared", "message": "Conversation history cleared"},
    ]
