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

    def fake_openrouter_llm(**kwargs):
        assert kwargs["model"] == "x-ai/grok-4.1-fast"
        assert kwargs["api_key"] == "test-key"
        return marker

    monkeypatch.setattr("src.server.websocket.OpenRouterLLM", fake_openrouter_llm)
    monkeypatch.setattr("src.server.websocket.OllamaLLM", lambda **kwargs: SimpleNamespace(name="ollama"))
    monkeypatch.setattr("src.server.websocket.GemmaTextVisionLLM", lambda **kwargs: SimpleNamespace(name="gemma"))

    await state.initialize()

    assert state.llm is marker
