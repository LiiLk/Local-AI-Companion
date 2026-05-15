import asyncio
from pathlib import Path
import shutil
from uuid import uuid4

from src.assistant.conversation_memory import (
    ConversationMemoryConfig,
    ConversationMemoryStore,
    build_conversation_memory_config,
    initial_messages,
)
from src.llm.base import Message


def _test_dir(name: str) -> Path:
    path = Path.cwd() / ".codex_test_artifacts" / f"{name}-{uuid4().hex}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def test_memory_store_persists_and_loads_bounded_recent_context():
    test_dir = _test_dir("bounded-context")
    try:
        store = ConversationMemoryStore(
            ConversationMemoryConfig(
                history_path=test_dir / "conversation.jsonl",
                summary_path=test_dir / "summary.txt",
                max_recent_turns=2,
            )
        )

        store.append_exchange("one", "reply one")
        store.append_exchange("two", "reply two")
        store.append_exchange("three", "reply three")

        messages = store.load_context_messages()

        assert [(message.role, message.content) for message in messages] == [
            ("user", "two"),
            ("assistant", "reply two"),
            ("user", "three"),
            ("assistant", "reply three"),
        ]
    finally:
        shutil.rmtree(test_dir, ignore_errors=True)


def test_memory_store_loads_recent_context_from_tail():
    test_dir = _test_dir("tail-context")
    try:
        history_path = test_dir / "conversation.jsonl"
        old_entries = "".join(
            f'{{"role":"user","content":"old {index}"}}\n'
            for index in range(200)
        )
        history_path.write_text(
            old_entries
            + '{"role":"assistant","content":"old ignored"}\n'
            + "not-json\n"
            + '{"role":"user","content":"latest user"}\n'
            + '{"role":"assistant","content":"latest assistant"}\n',
            encoding="utf-8",
        )
        store = ConversationMemoryStore(
            ConversationMemoryConfig(
                history_path=history_path,
                summary_path=test_dir / "summary.txt",
                max_recent_turns=1,
            )
        )

        messages = store.load_context_messages()

        assert [(message.role, message.content) for message in messages] == [
            ("user", "latest user"),
            ("assistant", "latest assistant"),
        ]
    finally:
        shutil.rmtree(test_dir, ignore_errors=True)


def test_memory_store_includes_existing_summary_as_system_context():
    test_dir = _test_dir("summary-context")
    try:
        summary_path = test_dir / "summary.txt"
        summary_path.write_text("The user likes concise answers.", encoding="utf-8")
        store = ConversationMemoryStore(
            ConversationMemoryConfig(
                history_path=test_dir / "conversation.jsonl",
                summary_path=summary_path,
                max_recent_turns=1,
            )
        )
        store.append_exchange("hello", "hi")

        messages = initial_messages("Base system prompt.", store)

        assert messages[0].content == "Base system prompt."
        assert messages[1].role == "system"
        assert "The user likes concise answers." in messages[1].content
        assert messages[-2].content == "hello"
        assert messages[-1].content == "hi"
    finally:
        shutil.rmtree(test_dir, ignore_errors=True)


def test_memory_store_ignores_malformed_jsonl_lines():
    test_dir = _test_dir("malformed-jsonl")
    try:
        history_path = test_dir / "conversation.jsonl"
        history_path.write_text(
            '{"role":"user","content":"valid"}\n'
            "not-json\n"
            '{"role":"tool","content":"ignored"}\n',
            encoding="utf-8",
        )
        store = ConversationMemoryStore(
            ConversationMemoryConfig(
                history_path=history_path,
                summary_path=test_dir / "summary.txt",
            )
        )

        messages = store.load_messages()

        assert [(message.role, message.content) for message in messages] == [("user", "valid")]
    finally:
        shutil.rmtree(test_dir, ignore_errors=True)


def test_memory_store_truncates_large_messages():
    test_dir = _test_dir("truncate")
    try:
        store = ConversationMemoryStore(
            ConversationMemoryConfig(
                history_path=test_dir / "conversation.jsonl",
                summary_path=test_dir / "summary.txt",
                max_message_chars=5,
            )
        )

        assert store.append_exchange("abcdef", "ghijkl") is True

        messages = store.load_messages()

        assert [message.content for message in messages] == ["abcde", "ghijk"]
    finally:
        shutil.rmtree(test_dir, ignore_errors=True)


def test_memory_store_curates_exchange_into_summary():
    class FakeCuratorLLM:
        def __init__(self):
            self.calls = []

        async def chat_stream(self, messages):
            self.calls.append(messages)
            yield '{"summary":"- User prefers concise technical explanations."}'

    test_dir = _test_dir("curate")
    try:
        llm = FakeCuratorLLM()
        store = ConversationMemoryStore(
            ConversationMemoryConfig(
                history_path=test_dir / "conversation.jsonl",
                summary_path=test_dir / "summary.txt",
            )
        )

        updated = asyncio.run(
            store.curate_exchange(
                llm,
                "I prefer concise technical explanations.",
                "Got it.",
            )
        )

        assert updated is True
        assert store.load_summary() == "- User prefers concise technical explanations."
        assert "Current memory summary:" in llm.calls[0][1].content
    finally:
        shutil.rmtree(test_dir, ignore_errors=True)


def test_memory_store_curates_summary_from_streamed_message_chunks():
    class MessageChunkLLM:
        async def chat_stream(self, messages):
            yield Message(role="assistant", content='{"summary":"- User prefers local models')
            yield Message(role="assistant", content=' for private projects."}')

    test_dir = _test_dir("curate-message-chunks")
    try:
        store = ConversationMemoryStore(
            ConversationMemoryConfig(
                history_path=test_dir / "conversation.jsonl",
                summary_path=test_dir / "summary.txt",
            )
        )

        updated = asyncio.run(
            store.curate_exchange(
                MessageChunkLLM(),
                "I prefer local models for private projects.",
                "I will remember that preference.",
            )
        )

        assert updated is True
        assert store.load_summary() == "- User prefers local models for private projects."
    finally:
        shutil.rmtree(test_dir, ignore_errors=True)


def test_memory_store_skips_sensitive_content_during_curation():
    class GuardedLLM:
        def __init__(self):
            self.called = False

        async def chat_stream(self, messages):
            self.called = True
            yield '{"summary":"bad"}'

    test_dir = _test_dir("sensitive")
    try:
        llm = GuardedLLM()
        store = ConversationMemoryStore(
            ConversationMemoryConfig(
                history_path=test_dir / "conversation.jsonl",
                summary_path=test_dir / "summary.txt",
            )
        )

        updated = asyncio.run(
            store.curate_exchange(
                llm,
                "My API key is sk-testsecret.",
                "I will not store that.",
            )
        )

        assert updated is False
        assert llm.called is False
        assert store.load_summary() == ""
    finally:
        shutil.rmtree(test_dir, ignore_errors=True)


def test_memory_store_ignores_malformed_curator_response():
    class MalformedLLM:
        async def chat_stream(self, messages):
            yield "not json"

    test_dir = _test_dir("malformed-curator")
    try:
        store = ConversationMemoryStore(
            ConversationMemoryConfig(
                history_path=test_dir / "conversation.jsonl",
                summary_path=test_dir / "summary.txt",
            )
        )

        updated = asyncio.run(
            store.curate_exchange(
                MalformedLLM(),
                "I like pair programming.",
                "Noted.",
            )
        )

        assert updated is False
        assert store.load_summary() == ""
    finally:
        shutil.rmtree(test_dir, ignore_errors=True)


def test_memory_store_clear_removes_history_and_summary():
    test_dir = _test_dir("clear")
    try:
        history_path = test_dir / "conversation.jsonl"
        summary_path = test_dir / "summary.txt"
        history_path.write_text("x", encoding="utf-8")
        summary_path.write_text("y", encoding="utf-8")
        store = ConversationMemoryStore(
            ConversationMemoryConfig(history_path=history_path, summary_path=summary_path)
        )

        store.clear()

        assert not history_path.exists()
        assert not summary_path.exists()
    finally:
        shutil.rmtree(test_dir, ignore_errors=True)


def test_memory_config_resolves_relative_paths_from_project_root():
    test_dir = _test_dir("config-root")
    try:
        config = {
            "memory": {
                "enabled": True,
                "history_path": "custom/history.jsonl",
                "summary_path": "custom/summary.txt",
                "max_recent_turns": 3,
                "max_message_chars": 44,
                "max_summary_chars": 42,
                "curate_enabled": False,
                "curator_timeout_sec": 7,
                "curator_max_input_chars": 55,
                "curator_max_output_chars": 66,
            }
        }

        memory_config = build_conversation_memory_config(config, project_root=test_dir)

        assert memory_config.history_path == test_dir / "custom/history.jsonl"
        assert memory_config.summary_path == test_dir / "custom/summary.txt"
        assert memory_config.max_recent_turns == 3
        assert memory_config.max_message_chars == 44
        assert memory_config.max_summary_chars == 42
        assert memory_config.curate_enabled is False
        assert memory_config.curator_timeout_sec == 7
        assert memory_config.curator_max_input_chars == 55
        assert memory_config.curator_max_output_chars == 66
    finally:
        shutil.rmtree(test_dir, ignore_errors=True)
