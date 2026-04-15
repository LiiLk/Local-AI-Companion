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


def test_conversation_state_preload_tts_falls_back_to_kokoro(monkeypatch):
    class Qwen3TTSProvider:
        pass

    class FakeKokoroProvider:
        def __init__(self, voice: str):
            self.voice = voice

    state = ConversationState()
    state.config = {
        "character": {"voice": {"kokoro_voice": "march-fast"}},
        "tts": {"warmup_on_start": True},
    }
    state.current_language = "fr"

    qwen_tts = Qwen3TTSProvider()

    def fake_create_pipeline_tts(config, initial_language=None):
        assert initial_language == "fr"
        return qwen_tts, "Qwen3"

    def fake_preload_pipeline_tts(tts, *, warmup=False, on_load_error=None):
        assert tts is qwen_tts
        assert warmup is True
        assert on_load_error is not None
        return on_load_error(tts, RuntimeError("warmup failed"))

    monkeypatch.setattr("src.server.websocket.create_pipeline_tts", fake_create_pipeline_tts)
    monkeypatch.setattr("src.server.websocket.preload_pipeline_tts", fake_preload_pipeline_tts)
    monkeypatch.setattr("src.server.websocket.KokoroProvider", FakeKokoroProvider)

    tts = state.preload_tts()

    assert isinstance(tts, FakeKokoroProvider)
    assert tts.voice == "march-fast"
    assert state.tts is tts
