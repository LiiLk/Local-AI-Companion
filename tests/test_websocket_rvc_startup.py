"""RVC startup failures stay visible while base TTS remains usable."""

from types import SimpleNamespace

import pytest

from src.assistant.pipeline_runtime import PipelineRuntime
from src.server import websocket as websocket_module
from src.server.websocket import ConversationState, WebSocketManager


@pytest.mark.asyncio
@pytest.mark.parametrize("progressive", [True, False])
@pytest.mark.parametrize("failure_stage", ["spawn", "preload", None])
async def test_rvc_startup_status_reaches_client(monkeypatch, progressive, failure_stage):
    config = {"mode": "pipeline", "tts": {"rvc": {"enabled": True}}}
    monkeypatch.setattr(websocket_module, "load_config", lambda: config)
    state = ConversationState()
    state.get_vad = lambda: setattr(state, "vad", object())
    state.get_smart_turn = lambda: SimpleNamespace(config=SimpleNamespace(enabled=False))
    state.sync_vad_required_misses = lambda: None
    runtime = PipelineRuntime(config)
    state.pipeline_runtime = runtime
    runtime.llm = object()
    runtime.tts = SimpleNamespace(preload=lambda: None)
    runtime.asr = SimpleNamespace(preload=lambda: None)

    def spawn():
        if failure_stage == "spawn":
            raise RuntimeError("simulated RVC spawn failure")

    def preload():
        if failure_stage == "preload":
            raise TimeoutError("simulated RVC startup timeout")

    runtime.rvc = SimpleNamespace(spawn_worker=spawn, preload=preload, warmup=lambda: None)
    sent = []

    async def send_json(data):
        sent.append(data)

    manager = WebSocketManager()
    client_id = "rvc-startup"
    manager.states[client_id] = state
    manager.active_connections[client_id] = SimpleNamespace(send_json=send_json)
    if progressive:
        await manager._preload_models_progressive(client_id)
    else:
        await manager.preload_models(client_id)

    ready = [message for message in sent if message["type"] == "models_ready"]
    rvc_ready = [message for message in sent if message.get("message") == "RVC ready!"]
    errors = [message for message in sent if message["type"] == "error"]
    assert len(ready) == 1  # Base TTS is still usable.
    if failure_stage:
        reason = runtime.collect_degraded_reason()
        assert reason and "simulated RVC" in reason
        assert errors == [{"type": "error", "message": reason}]
        assert sent[-1] == errors[0]  # Readiness must not hide the warning.
        assert rvc_ready == []
        # A manual request for already loaded fallback models retains the warning.
        sent.clear()
        await manager.preload_models(client_id)
        assert [message["type"] for message in sent] == ["models_ready", "error"]
        assert sent[-1]["message"] == reason
    else:
        assert errors == []
        assert len(rvc_ready) == int(progressive)
        assert state.rvc is runtime.rvc
