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
    runtime = SimpleNamespace(ensure_llm=lambda: marker)

    def fake_create_pipeline_runtime(config, *, initial_tts_language=None):
        openrouter_config = config["llm"]["openrouter"]
        assert openrouter_config["model"] == "x-ai/grok-4.1-fast"
        assert openrouter_config["api_key"] == "test-key"
        assert initial_tts_language == "en"
        return runtime

    monkeypatch.setattr("src.server.websocket.create_pipeline_runtime", fake_create_pipeline_runtime)

    await state.initialize()

    assert state.llm is marker
    assert state.pipeline_runtime is runtime


def test_conversation_state_preload_tts_falls_back_to_kokoro(monkeypatch):
    class FakeKokoroProvider:
        def __init__(self, voice: str):
            self.voice = voice

    state = ConversationState()
    state.config = {
        "character": {"voice": {"kokoro_voice": "march-fast"}},
        "tts": {"warmup_on_start": True},
    }
    state.current_language = "fr"

    monkeypatch.setattr("src.server.websocket.KokoroProvider", FakeKokoroProvider)

    qwen_tts = type("Qwen3TTSProvider", (), {})()

    class FakeRuntime:
        def __init__(self):
            self.tts = None
            self.seen_language = None

        def ensure_tts(self, language=None):
            self.seen_language = language
            self.tts = qwen_tts
            return qwen_tts

        def preload_tts(self, *, on_load_error=None):
            assert on_load_error is not None
            return on_load_error(qwen_tts, RuntimeError("warmup failed"))

    runtime = FakeRuntime()
    state.pipeline_runtime = runtime

    tts = state.preload_tts()

    assert isinstance(tts, FakeKokoroProvider)
    assert tts.voice == "march-fast"
    assert state.tts is tts
    assert runtime.tts is tts
