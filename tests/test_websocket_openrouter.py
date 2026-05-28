from types import SimpleNamespace

import pytest

from src.server.websocket import ConversationState
from src.llm.base import Message
from src.assistant.conversation_memory import ConversationMemoryConfig, ConversationMemoryStore


@pytest.mark.asyncio
async def test_conversation_state_initialize_selects_openrouter(monkeypatch):
    state = ConversationState()
    state.config = {
        "mode": "pipeline",
        "character": {"system_prompt": "You are helpful."},
        "llm": {
            "provider": "openrouter",
            "openrouter": {
                "model": "deepseek/deepseek-v4-pro",
                "api_key": "test-key",
            },
        },
    }
    state.mode = "pipeline"

    marker = SimpleNamespace(name="openrouter")
    runtime = SimpleNamespace(ensure_llm=lambda: marker)

    def fake_create_pipeline_runtime(config, *, initial_tts_language=None):
        openrouter_config = config["llm"]["openrouter"]
        assert openrouter_config["model"] == "deepseek/deepseek-v4-pro"
        assert openrouter_config["api_key"] == "test-key"
        assert initial_tts_language == "en"
        return runtime

    monkeypatch.setattr("src.server.websocket.create_pipeline_runtime", fake_create_pipeline_runtime)

    await state.initialize()

    assert state.llm is marker
    assert state.pipeline_runtime is runtime


@pytest.mark.asyncio
async def test_conversation_state_initialize_loads_pipeline_memory(monkeypatch):
    state = ConversationState()
    state.config = {
        "mode": "pipeline",
        "character": {"system_prompt": "You are helpful."},
        "llm": {"provider": "ollama"},
    }
    state.mode = "pipeline"

    memory = SimpleNamespace(
        load_context_messages=lambda: [Message(role="user", content="remembered")]
    )
    runtime = SimpleNamespace(
        ensure_llm=lambda: SimpleNamespace(name="ollama"),
        ensure_memory=lambda: memory,
    )

    monkeypatch.setattr(
        "src.server.websocket.create_pipeline_runtime",
        lambda config, *, initial_tts_language=None: runtime,
    )

    await state.initialize()

    assert state.memory_store is memory
    assert [message.content for message in state.messages] == [
        "You are helpful.",
        "remembered",
    ]


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


@pytest.mark.asyncio
async def test_websocket_pipeline_curation_refreshes_bounded_memory(tmp_path):
    from src.server.websocket import WebSocketManager

    class FakeCuratorLLM:
        async def chat_stream(self, messages):
            yield '{"summary":"- User prefers green accent colors."}'

    manager = WebSocketManager()
    memory_store = ConversationMemoryStore(
        ConversationMemoryConfig(
            history_path=tmp_path / "conversation.jsonl",
            summary_path=tmp_path / "summary.txt",
        )
    )
    llm = FakeCuratorLLM()
    state = SimpleNamespace(
        memory_store=memory_store,
        config={"character": {"system_prompt": "You are helpful."}},
        messages=[],
        get_llm=lambda: llm,
    )

    await manager._curate_pipeline_memory(
        state,
        "I prefer green accent colors.",
        "Noted.",
    )

    assert memory_store.load_summary() == "- User prefers green accent colors."
    assert [message.role for message in state.messages] == ["system", "system"]
    assert state.messages[0].content == "You are helpful."
    assert "User prefers green accent colors" in state.messages[1].content
