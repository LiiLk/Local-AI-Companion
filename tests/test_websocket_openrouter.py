from types import SimpleNamespace

import pytest

from src.server.websocket import ConversationState


@pytest.mark.asyncio
async def test_conversation_state_initialize_selects_openrouter(monkeypatch):
    state = ConversationState()
    state.config = {
        "mode": "pipeline",
        "character": {"system_prompt": "You are helpful."},
        "llm": {
            "provider": "openrouter",
            "openrouter": {
                "model": "x-ai/grok-4.1-fast",
                "api_key": "test-key",
            },
        },
    }
    state.mode = "pipeline"

    marker = SimpleNamespace(name="openrouter")

    def fake_create_pipeline_llm(config):
        openrouter_config = config["llm"]["openrouter"]
        assert openrouter_config["model"] == "x-ai/grok-4.1-fast"
        assert openrouter_config["api_key"] == "test-key"
        return marker, "OpenRouter (x-ai/grok-4.1-fast)"

    monkeypatch.setattr("src.server.websocket.create_pipeline_llm", fake_create_pipeline_llm)

    await state.initialize()

    assert state.llm is marker
