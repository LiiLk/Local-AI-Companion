import asyncio
from types import SimpleNamespace

import pytest

from src.server.websocket import WebSocketManager


@pytest.mark.asyncio
async def test_new_turn_cancels_previous_turn_for_same_client():
    manager = WebSocketManager()
    client_id = "client-test"
    state = SimpleNamespace(response_task=None, response_lock=asyncio.Lock())
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
    state = SimpleNamespace(response_task=None, response_lock=asyncio.Lock(), vad=vad)
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
    assert state.response_task is None
    assert vad.reset_calls == 1
