import asyncio
from types import SimpleNamespace

import pytest

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
