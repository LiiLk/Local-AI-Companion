from pathlib import Path
import shutil
from uuid import uuid4

from src.assistant.conversation_memory import (
    ConversationMemoryConfig,
    ConversationMemoryStore,
    build_conversation_memory_config,
    initial_messages,
)


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
            }
        }

        memory_config = build_conversation_memory_config(config, project_root=test_dir)

        assert memory_config.history_path == test_dir / "custom/history.jsonl"
        assert memory_config.summary_path == test_dir / "custom/summary.txt"
        assert memory_config.max_recent_turns == 3
        assert memory_config.max_message_chars == 44
        assert memory_config.max_summary_chars == 42
    finally:
        shutil.rmtree(test_dir, ignore_errors=True)
