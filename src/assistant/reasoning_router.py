"""Adaptive reasoning router ("fast by default, think when needed").

The first LLM call always runs with the base effort. An extra, configurable
system instruction asks the model to answer with a single routing marker when
the request really needs careful multi-step reasoning. Because the marker is
decided by the model on the first tokens, the simple path pays no extra
round-trip: only the first few characters are held back while we check whether
the reply starts with a marker.

This module is intentionally pure (no I/O, no provider imports) so the marker
state machine and the orchestration can be unit-tested with fake LLMs.
"""

from __future__ import annotations

import inspect
import logging
import random
import time
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Awaitable, Callable, Optional

from src.llm.base import Message
from src.utils.turn_latency import get_turn_latency_tracker

logger = logging.getLogger(__name__)

THINK_MARKER = "<|THINK|>"
THINK_HARD_MARKER = "<|THINK_HARD|>"

# Longest first: a marker is matched by ``str.startswith`` so the hard marker
# must be tested before the short one when they share a prefix.
_MARKERS: tuple[tuple[str, str], ...] = (
    (THINK_HARD_MARKER, "think_hard"),
    (THINK_MARKER, "think"),
)

DEFAULT_FILLER_PHRASES: tuple[str, ...] = (
    "Hmm, let me think about that.",
    "Good question, give me a second.",
    "Okay, let me work that out.",
)

DEFAULT_ROUTING_PROMPT = (
    "Routing instruction: before answering, decide whether the user's request "
    "genuinely requires careful multi-step reasoning, a non-trivial calculation, "
    "planning, or writing code. If it does, reply with exactly <|THINK|> and "
    "nothing else, or <|THINK_HARD|> if it is especially difficult. Otherwise, "
    "answer the user normally. Never output these markers unless you are asking "
    "for deeper reasoning."
)


def strip_reasoning_markers(text: str) -> str:
    """Remove any routing marker that leaked outside the leading position."""
    for marker, _decision in _MARKERS:
        text = text.replace(marker, "")
    return text


class MarkerStrippingFilter:
    """Remove routing markers anywhere in a stream, with a bounded holdback.

    Only a trailing suffix that could still grow into a marker (at most the
    longest marker minus one character) is held between chunks, so a marker
    split across two chunks is still removed.
    """

    def __init__(self, markers: tuple[tuple[str, str], ...] = _MARKERS):
        self._markers = tuple(marker for marker, _decision in markers)
        self._buffer = ""

    def feed(self, chunk: str) -> str:
        self._buffer += chunk
        output: list[str] = []
        index = 0
        while index < len(self._buffer):
            marker = next(
                (
                    candidate
                    for candidate in self._markers
                    if self._buffer.startswith(candidate, index)
                ),
                None,
            )
            if marker is not None:
                index += len(marker)
                continue
            remainder = self._buffer[index:]
            if any(candidate.startswith(remainder) for candidate in self._markers):
                break
            output.append(self._buffer[index])
            index += 1
        self._buffer = self._buffer[index:]
        return "".join(output)

    def flush(self) -> str:
        tail = self._buffer
        self._buffer = ""
        return tail


@dataclass
class AdaptiveReasoningConfig:
    """Configuration for the adaptive reasoning router."""

    enabled: bool = False
    base_effort: str = "none"
    escalate_effort: str = "medium"
    max_effort: str = "high"
    escalated_max_completion_tokens: int = 2048
    filler_phrases: list[str] = field(
        default_factory=lambda: list(DEFAULT_FILLER_PHRASES)
    )
    routing_prompt: str = DEFAULT_ROUTING_PROMPT

    @classmethod
    def from_dict(cls, data: Any) -> Optional["AdaptiveReasoningConfig"]:
        """Build a config from the ``llm.adaptive_reasoning`` mapping.

        A missing mapping means "feature off" and returns ``None`` so the
        pipeline keeps its previous single-call behaviour untouched.
        """
        if not isinstance(data, dict):
            return None

        phrases = data.get("filler_phrases")
        if not isinstance(phrases, (list, tuple)) or not phrases:
            phrases = list(DEFAULT_FILLER_PHRASES)

        routing_prompt = str(data.get("routing_prompt") or DEFAULT_ROUTING_PROMPT)

        try:
            max_tokens = int(data.get("escalated_max_completion_tokens", 2048))
        except (TypeError, ValueError):
            max_tokens = 2048

        return cls(
            enabled=bool(data.get("enabled", False)),
            base_effort=str(data.get("base_effort", "none") or "none"),
            escalate_effort=str(data.get("escalate_effort", "medium") or "medium"),
            max_effort=str(data.get("max_effort", "high") or "high"),
            escalated_max_completion_tokens=max_tokens,
            filler_phrases=[str(phrase) for phrase in phrases],
            routing_prompt=routing_prompt,
        )

    def pick_filler(self) -> str:
        """Return a waiting phrase to speak while the escalation runs."""
        return random.choice(self.filler_phrases)


class ReasoningMarkerRouter:
    """Detect a routing marker at the very start of a stream.

    ``feed`` returns the text that is safe to release downstream. While the
    reply could still be a marker prefix it returns ``""`` and holds the
    characters, so the holdback is bounded by the longest marker length.
    """

    def __init__(self, markers: tuple[tuple[str, str], ...] = _MARKERS):
        self._markers = tuple(sorted(markers, key=lambda item: len(item[0]), reverse=True))
        self._max_len = max(len(marker) for marker, _decision in self._markers)
        self._held = ""
        self._resolved = False
        self.decision: Optional[str] = None

    @property
    def resolved(self) -> bool:
        return self._resolved

    def _classify(self, text: str) -> Optional[str]:
        for marker, decision in self._markers:
            if text.startswith(marker):
                return decision
        if any(marker.startswith(text) for marker, _decision in self._markers):
            return None
        return "direct"

    def feed(self, chunk: str) -> str:
        if self._resolved:
            return chunk

        self._held += chunk
        decision = self._classify(self._held)
        if decision is None:
            return ""

        self._resolved = True
        self.decision = decision
        released = self._held
        self._held = ""
        if decision == "direct":
            return released
        # A marker: the buffered characters are routing noise and are dropped.
        return ""

    def flush(self) -> str:
        """Resolve an undecided stream as a normal (direct) answer."""
        if self._resolved:
            return ""
        self._resolved = True
        self.decision = "direct"
        released = self._held
        self._held = ""
        return released


def build_routing_messages(messages: list[Message], routing_prompt: str) -> list[Message]:
    """Return the first-call messages with the routing instruction appended."""
    return [*messages, Message(role="system", content=routing_prompt)]


def accepts_options_override(llm: Any) -> bool:
    """Return True when ``llm.chat_stream`` can take ``options_override``."""
    chat_stream = getattr(llm, "chat_stream", None)
    if not callable(chat_stream):
        return False
    try:
        signature = inspect.signature(chat_stream)
    except (TypeError, ValueError):
        return False
    parameters = signature.parameters
    if "options_override" in parameters:
        return True
    return any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )


async def _chat_stream(
    llm: Any,
    messages: list[Message],
    options_override: Optional[dict[str, Any]],
) -> AsyncGenerator[str, None]:
    """Call ``llm.chat_stream`` with the override only when it is supported."""
    if options_override is not None and accepts_options_override(llm):
        stream = llm.chat_stream(messages, options_override=options_override)
    else:
        stream = llm.chat_stream(messages)
    async for chunk in stream:
        yield chunk


async def stream_llm_with_adaptive_reasoning(
    llm: Any,
    messages: list[Message],
    config: Optional[AdaptiveReasoningConfig],
    *,
    on_escalation: Optional[Callable[[str, str, str], Awaitable[None]]] = None,
) -> AsyncGenerator[str, None]:
    """Stream the LLM, escalating the reasoning effort when the model asks.

    ``on_escalation`` is awaited with ``(decision, effort, filler)`` just before
    the second call, so the caller can speak the waiting phrase immediately.
    """
    if config is None or not config.enabled:
        async for chunk in _chat_stream(llm, messages, None):
            yield chunk
        return

    routing_messages = build_routing_messages(messages, config.routing_prompt)
    base_override = {"reasoning": {"effort": config.base_effort}}
    started = time.perf_counter()
    first_chunk_ms: Optional[float] = None
    router = ReasoningMarkerRouter()
    stripper = MarkerStrippingFilter()

    async for chunk in _chat_stream(llm, routing_messages, base_override):
        if first_chunk_ms is None:
            first_chunk_ms = (time.perf_counter() - started) * 1000.0
        released = router.feed(chunk)
        if router.resolved and router.decision != "direct":
            break
        if released:
            cleaned = stripper.feed(released)
            if cleaned:
                yield cleaned

    if not router.resolved:
        released = router.flush()
        if released:
            cleaned = stripper.feed(released)
            if cleaned:
                yield cleaned

    decision = router.decision or "direct"
    if decision == "direct":
        tail = stripper.flush()
        if tail:
            yield tail
        logger.info(
            "reasoning_route decision=direct effort=%s first_call_ms=%.0f",
            config.base_effort,
            first_chunk_ms if first_chunk_ms is not None else 0.0,
        )
        return

    effort = config.escalate_effort if decision == "think" else config.max_effort
    get_turn_latency_tracker().mark("reasoning_escalated")
    logger.info(
        "reasoning_route decision=%s effort=%s first_call_ms=%.0f",
        decision,
        effort,
        first_chunk_ms if first_chunk_ms is not None else 0.0,
    )

    if on_escalation is not None:
        await on_escalation(decision, effort, config.pick_filler())

    escalated_override = {
        "reasoning": {"effort": effort},
        "max_completion_tokens": config.escalated_max_completion_tokens,
    }
    escalated_stripper = MarkerStrippingFilter()
    async for chunk in _chat_stream(llm, messages, escalated_override):
        cleaned = escalated_stripper.feed(chunk)
        if cleaned:
            yield cleaned
    tail = escalated_stripper.flush()
    if tail:
        yield tail
