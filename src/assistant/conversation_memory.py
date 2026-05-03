"""Local persistent conversation memory helpers.

The memory layer is intentionally small: append-only JSONL on disk plus a
bounded recent-message view for prompts. Semantic summarization can grow from
the summary file without changing the runtime contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from typing import Any

from src.llm.base import Message

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HISTORY_PATH = Path("data/memory/conversation.jsonl")
DEFAULT_SUMMARY_PATH = Path("data/memory/summary.txt")
VALID_MEMORY_ROLES = {"user", "assistant"}


@dataclass(frozen=True)
class ConversationMemoryConfig:
    enabled: bool = True
    history_path: Path = PROJECT_ROOT / DEFAULT_HISTORY_PATH
    summary_path: Path = PROJECT_ROOT / DEFAULT_SUMMARY_PATH
    max_recent_turns: int = 8
    max_message_chars: int = 2000
    max_summary_chars: int = 2000


class ConversationMemoryStore:
    """Persist conversation messages and build bounded prompt context."""

    def __init__(self, config: ConversationMemoryConfig):
        self.config = config

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    def load_context_messages(self) -> list[Message]:
        """Return summary plus the recent bounded transcript for LLM context."""
        if not self.enabled:
            return []

        context: list[Message] = []
        summary = self.load_summary()
        if summary:
            context.append(
                Message(
                    role="system",
                    content=(
                        "Relevant memory summary from previous conversations:\n"
                        f"{summary}"
                    ),
                )
            )
        context.extend(select_recent_turns(self.load_messages(), self.config.max_recent_turns))
        return context

    def load_messages(self) -> list[Message]:
        if not self.enabled or not self.config.history_path.exists():
            return []

        messages: list[Message] = []
        try:
            with self.config.history_path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        payload = json.loads(line)
                    except json.JSONDecodeError:
                        logger.warning(
                            "Ignoring malformed memory line %s in %s",
                            line_number,
                            self.config.history_path,
                        )
                        continue

                    role = str(payload.get("role") or "")
                    content = _truncate_memory_text(
                        str(payload.get("content") or ""),
                        self.config.max_message_chars,
                    )
                    if role not in VALID_MEMORY_ROLES or not content:
                        continue
                    messages.append(Message(role=role, content=content))
        except OSError as exc:
            logger.warning("Could not read conversation memory: %s", exc)
        return messages

    def load_summary(self) -> str:
        if (
            not self.enabled
            or self.config.max_summary_chars <= 0
            or not self.config.summary_path.exists()
        ):
            return ""

        try:
            summary = self.config.summary_path.read_text(encoding="utf-8").strip()
        except OSError as exc:
            logger.warning("Could not read conversation memory summary: %s", exc)
            return ""
        if not summary:
            return ""
        return summary[-self.config.max_summary_chars :]

    def append_exchange(
        self,
        user_text: str,
        assistant_text: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        if not self.enabled:
            return False

        user_text = _truncate_memory_text(user_text or "", self.config.max_message_chars)
        assistant_text = _truncate_memory_text(assistant_text or "", self.config.max_message_chars)
        if not user_text or not assistant_text:
            return False

        now = datetime.now(timezone.utc).isoformat()
        entries = [
            {
                "timestamp": now,
                "role": "user",
                "content": user_text,
                "metadata": metadata or {},
            },
            {
                "timestamp": now,
                "role": "assistant",
                "content": assistant_text,
                "metadata": metadata or {},
            },
        ]

        try:
            self.config.history_path.parent.mkdir(parents=True, exist_ok=True)
            with self.config.history_path.open("a", encoding="utf-8") as handle:
                for entry in entries:
                    handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except OSError as exc:
            logger.warning("Could not append conversation memory: %s", exc)
            return False
        return True

    def clear(self) -> None:
        if not self.enabled:
            return

        for path in (self.config.history_path, self.config.summary_path):
            try:
                path.unlink(missing_ok=True)
            except OSError as exc:
                logger.warning("Could not clear memory file %s: %s", path, exc)


def build_conversation_memory_config(
    app_config: dict[str, Any],
    *,
    project_root: Path | None = None,
) -> ConversationMemoryConfig:
    memory_config = app_config.get("memory", {}) or {}
    root = project_root or PROJECT_ROOT
    return ConversationMemoryConfig(
        enabled=bool(memory_config.get("enabled", True)),
        history_path=_resolve_memory_path(
            memory_config.get("history_path", DEFAULT_HISTORY_PATH),
            root,
        ),
        summary_path=_resolve_memory_path(
            memory_config.get("summary_path", DEFAULT_SUMMARY_PATH),
            root,
        ),
        max_recent_turns=max(0, int(memory_config.get("max_recent_turns", 8))),
        max_message_chars=max(0, int(memory_config.get("max_message_chars", 2000))),
        max_summary_chars=max(0, int(memory_config.get("max_summary_chars", 2000))),
    )


def create_conversation_memory(
    app_config: dict[str, Any],
    *,
    project_root: Path | None = None,
) -> ConversationMemoryStore | None:
    config = build_conversation_memory_config(app_config, project_root=project_root)
    if not config.enabled:
        return None
    return ConversationMemoryStore(config)


def initial_messages(system_prompt: str, memory_store: ConversationMemoryStore | None) -> list[Message]:
    messages = [Message(role="system", content=system_prompt)]
    if memory_store:
        messages.extend(memory_store.load_context_messages())
    return messages


def select_recent_turns(messages: list[Message], max_recent_turns: int) -> list[Message]:
    if max_recent_turns <= 0:
        return []
    return list(messages[-max_recent_turns * 2 :])


def _resolve_memory_path(value: str | Path, project_root: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return project_root / path


def _truncate_memory_text(text: str, max_chars: int) -> str:
    clean = (text or "").strip()
    if max_chars <= 0:
        return ""
    if len(clean) <= max_chars:
        return clean
    return clean[:max_chars].rstrip()
