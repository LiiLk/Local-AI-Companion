"""RVC startup failures stay visible while base TTS remains usable."""

from types import SimpleNamespace
from unittest.mock import Mock

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


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_stage", ["get_vad", "preload_llm", "preload_tts", "preload_asr"])
async def test_manual_preload_failure_closes_early_rvc_worker(monkeypatch, failure_stage):
    config = {
        "mode": "pipeline",
        "llm": {"provider": "gemma"},
        "tts": {"rvc": {"enabled": True}},
    }
    monkeypatch.setattr(websocket_module, "load_config", lambda: config)
    state = ConversationState()
    runtime = PipelineRuntime(config)
    state.pipeline_runtime = runtime
    rvc = Mock()
    runtime.rvc = rvc
    state.get_vad = lambda: setattr(state, "vad", object())
    for stage in ("preload_llm", "preload_tts", "preload_asr", "preload_rvc"):
        setattr(state, stage, Mock())
    setattr(state, failure_stage, Mock(side_effect=RuntimeError("simulated preload failure")))
    sent = []

    async def send_json(data):
        sent.append(data)

    manager = WebSocketManager()
    client_id = "manual-preload-failure"
    manager.states[client_id] = state
    manager.active_connections[client_id] = SimpleNamespace(send_json=send_json)

    await manager.preload_models(client_id)

    rvc.spawn_worker.assert_called_once_with()
    rvc.close.assert_called_once_with()
    state.preload_rvc.assert_not_called()
    assert sent[-1] == {
        "type": "error", "message": "Failed to load models: simulated preload failure",
    }
    assert client_id not in manager._preloading


@pytest.mark.asyncio
async def test_manual_preload_retry_recreates_discarded_rvc(monkeypatch):
    config = {"mode": "pipeline", "tts": {"rvc": {"enabled": True}}}
    monkeypatch.setattr(websocket_module, "load_config", lambda: config)
    state = ConversationState()
    runtime = PipelineRuntime(config)
    state.pipeline_runtime = runtime
    runtime.llm = object()
    first, second = Mock(), Mock()
    factory = Mock(side_effect=[(first, "first"), (second, "second")])
    monkeypatch.setattr("src.assistant.pipeline_runtime.create_pipeline_rvc", factory)
    # A user turn may already hold a reference before manual preload starts.
    assert state.get_rvc() is first
    state.get_vad = lambda: setattr(state, "vad", object())

    def preload_tts():
        runtime.tts = state.tts = SimpleNamespace(preload=lambda: None)

    def preload_asr():
        runtime.asr = state.asr = SimpleNamespace(preload=lambda: None)
        raise RuntimeError("simulated failure after ASR load")

    state.preload_tts = preload_tts
    state.preload_asr = preload_asr
    sent = []

    async def send_json(data):
        sent.append(data)

    manager = WebSocketManager()
    client_id = "manual-preload-retry"
    manager.states[client_id] = state
    manager.active_connections[client_id] = SimpleNamespace(send_json=send_json)

    await manager.preload_models(client_id)

    first.spawn_worker.assert_called_once_with()
    first.close.assert_called_once_with()
    assert runtime.rvc is None
    assert state.rvc is None
    assert state.pipeline_ready() is False
    assert runtime.collect_degraded_reason() is None
    assert sent[-1]["type"] == "error"

    await manager.preload_models(client_id)

    assert factory.call_count == 2
    assert runtime.rvc is state.rvc is second
    second.spawn_worker.assert_called_once_with()
    second.preload.assert_called_once_with()
    second.warmup.assert_called_once_with()
    assert state.pipeline_ready() is True
    assert sent[-1]["type"] == "models_ready"
