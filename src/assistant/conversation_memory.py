"""Local persistent conversation memory helpers.

The memory layer is intentionally small: append-only JSONL on disk plus a
bounded recent-message view for prompts. Semantic summarization can grow from
the summary file without changing the runtime contract.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import re
from typing import Any

from src.llm.base import Message

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HISTORY_PATH = Path("data/memory/conversation.jsonl")
DEFAULT_SUMMARY_PATH = Path("data/memory/summary.txt")
VALID_MEMORY_ROLES = {"user", "assistant"}
SENSITIVE_MEMORY_PATTERN = re.compile(
    r"(?i)\b("
    r"api[_-]?key|token|secret|password|passwd|bearer|private key|"
    r"openrouter_api_key|github_pat|ghp_[a-z0-9_]+|sk-[a-z0-9_-]+"
    r")\b"
)


@dataclass(frozen=True)
class ConversationMemoryConfig:
    enabled: bool = True
    history_path: Path = PROJECT_ROOT / DEFAULT_HISTORY_PATH
    summary_path: Path = PROJECT_ROOT / DEFAULT_SUMMARY_PATH
    max_recent_turns: int = 8
    max_message_chars: int = 2000
    max_summary_chars: int = 2000
    curate_enabled: bool = True
    curator_timeout_sec: float = 20.0
    curator_max_input_chars: int = 1200
    curator_max_output_chars: int = 3000


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

    def write_summary(self, summary: str) -> bool:
        if not self.enabled or self.config.max_summary_chars <= 0:
            return False

        clean = _truncate_memory_text(summary, self.config.max_summary_chars)
        try:
            self.config.summary_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = self.config.summary_path.with_suffix(
                f"{self.config.summary_path.suffix}.tmp"
            )
            tmp_path.write_text(clean, encoding="utf-8")
            tmp_path.replace(self.config.summary_path)
        except OSError as exc:
            logger.warning("Could not write conversation memory summary: %s", exc)
            return False
        return True

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

    async def curate_exchange(
        self,
        llm: Any,
        user_text: str,
        assistant_text: str,
    ) -> bool:
        if not self._can_curate(user_text, assistant_text):
            return False

        current_summary = self.load_summary()
        messages = build_curator_messages(
            current_summary=current_summary,
            user_text=_truncate_memory_text(user_text, self.config.curator_max_input_chars),
            assistant_text=_truncate_memory_text(
                assistant_text,
                self.config.curator_max_input_chars,
            ),
        )
        try:
            raw = await asyncio.wait_for(
                _collect_llm_text(llm, messages, self.config.curator_max_output_chars),
                timeout=self.config.curator_timeout_sec,
            )
        except Exception as exc:
            logger.info("Memory curation skipped after LLM error: %s", exc)
            return False

        curated_summary = parse_curator_summary(raw)
        if curated_summary is None:
            logger.info("Memory curation skipped: malformed curator response")
            return False

        curated_summary = _truncate_memory_text(curated_summary, self.config.max_summary_chars)
        if curated_summary == current_summary:
            return False
        if not curated_summary and not current_summary:
            return False

        updated = self.write_summary(curated_summary)
        if updated:
            logger.info("Memory summary updated (%s chars)", len(curated_summary))
        return updated

    def _can_curate(self, user_text: str, assistant_text: str) -> bool:
        if not self.enabled or not self.config.curate_enabled:
            return False
        combined = f"{user_text}\n{assistant_text}".strip()
        if not combined:
            return False
        if SENSITIVE_MEMORY_PATTERN.search(combined):
            logger.info("Memory curation skipped: sensitive-looking content")
            return False
        if _looks_like_low_signal_blob(user_text):
            logger.info("Memory curation skipped: low-signal user content")
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
        curate_enabled=bool(memory_config.get("curate_enabled", True)),
        curator_timeout_sec=max(0.1, float(memory_config.get("curator_timeout_sec", 20))),
        curator_max_input_chars=max(0, int(memory_config.get("curator_max_input_chars", 1200))),
        curator_max_output_chars=max(0, int(memory_config.get("curator_max_output_chars", 3000))),
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


def build_curator_messages(
    *,
    current_summary: str,
    user_text: str,
    assistant_text: str,
) -> list[Message]:
    return [
        Message(
            role="system",
            content=(
                "You maintain a short long-term memory summary for a local AI companion. "
                "Keep only durable facts that will help future conversations: user preferences, "
                "stable personal facts, project decisions, recurring constraints, and working conventions. "
                "Do NOT store secrets, passwords, API keys, tokens, private credentials, one-off tasks, "
                "temporary details, long logs, code dumps, or speculation. "
                "Return ONLY valid JSON with one string field named summary. "
                "The summary must be concise bullet text or an empty string."
            ),
        ),
        Message(
            role="user",
            content=(
                "Current memory summary:\n"
                f"{current_summary or '(empty)'}\n\n"
                "New exchange:\n"
                f"User: {user_text}\n"
                f"Assistant: {assistant_text}\n\n"
                "Return the complete updated memory summary as JSON."
            ),
        ),
    ]


def parse_curator_summary(raw: str) -> str | None:
    text = (raw or "").strip()
    if not text:
        return None
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict) or "summary" not in payload:
        return None
    summary = payload.get("summary")
    if summary is None:
        return ""
    if not isinstance(summary, str):
        return None
    return summary.strip()


def select_recent_turns(messages: list[Message], max_recent_turns: int) -> list[Message]:
    if max_recent_turns <= 0:
        return []
    return list(messages[-max_recent_turns * 2 :])


async def _collect_llm_text(llm: Any, messages: list[Message], max_chars: int) -> str:
    if max_chars <= 0:
        return ""

    chat_stream = getattr(llm, "chat_stream", None)
    if callable(chat_stream):
        chunks: list[str] = []
        total = 0
        async for chunk in chat_stream(messages):
            chunk_text = _coerce_llm_text_chunk(chunk)
            if not chunk_text:
                continue
            remaining = max_chars - total
            if remaining <= 0:
                break
            chunks.append(chunk_text[:remaining])
            total += min(len(chunk_text), remaining)
        return "".join(chunks)

    chat = getattr(llm, "chat", None)
    if callable(chat):
        response = await chat(messages)
        return _coerce_llm_text_chunk(response)[:max_chars]

    return ""


def _coerce_llm_text_chunk(chunk: Any) -> str:
    if chunk is None:
        return ""
    content = getattr(chunk, "content", None)
    if content is not None:
        return str(content or "")
    if isinstance(chunk, dict):
        content = chunk.get("content")
        if content is not None:
            return str(content or "")
        message = chunk.get("message")
        if isinstance(message, dict):
            return str(message.get("content") or "")
        delta = chunk.get("delta")
        if isinstance(delta, dict):
            return str(delta.get("content") or "")
    return str(chunk or "")


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


def _looks_like_low_signal_blob(text: str) -> bool:
    clean = (text or "").strip()
    if not clean:
        return True
    if "```" in clean:
        return True
    if clean.count("\n") >= 12:
        return True
    return False
