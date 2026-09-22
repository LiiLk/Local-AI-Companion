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
from contextlib import aclosing
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

# How many leading whitespace-only characters the router holds while it waits
# for the first non-whitespace token. Bounds the holdback for a stream that
# never produces anything else.
_MAX_LEADING_WHITESPACE = 16

DEFAULT_FILLER_PHRASES: tuple[str, ...] = (
    "Hmm, let me think about that.",
    "Good question, give me a second.",
    "Okay, let me work that out.",
)

DEFAULT_FILLER_PHRASES_FR: tuple[str, ...] = (
    "Hmm, laisse-moi réfléchir.",
    "Bonne question, une seconde.",
    "D'accord, je regarde ça.",
)

DEFAULT_FILLER_PHRASES_BY_LANGUAGE: dict[str, tuple[str, ...]] = {
    "en": DEFAULT_FILLER_PHRASES,
    "fr": DEFAULT_FILLER_PHRASES_FR,
    "es": (
        "Un momento, déjame pensarlo.",
        "Buena pregunta, dame un segundo.",
    ),
    "de": (
        "Moment, lass mich kurz nachdenken.",
        "Gute Frage, einen Augenblick.",
    ),
    "it": (
        "Un attimo, fammi pensare.",
        "Bella domanda, dammi un secondo.",
    ),
    "pt": (
        "Um momento, deixa eu pensar.",
        "Boa pergunta, só um segundo.",
    ),
    "ja": (
        "ちょっと考えさせて。",
        "いい質問だね、少し待って。",
    ),
    "zh": (
        "让我想一想。",
        "好问题，稍等一下。",
    ),
    "ko": (
        "잠깐 생각해 볼게요.",
        "좋은 질문이네요, 잠시만요.",
    ),
    "ru": (
        "Секунду, дай подумать.",
        "Хороший вопрос, минутку.",
    ),
    "ar": (
        "لحظة، دعني أفكر.",
        "سؤال جيد، أمهلني ثانية.",
    ),
    "hi": (
        "एक पल, मुझे सोचने दो।",
        "अच्छा सवाल है, एक सेकंड।",
    ),
    "tr": (
        "Bir saniye, düşüneyim.",
        "Güzel soru, bir saniye.",
    ),
}

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
    filler_phrases_by_language: dict[str, list[str]] = field(default_factory=dict)
    routing_prompt: str = DEFAULT_ROUTING_PROMPT

    @classmethod
    def from_dict(cls, data: Any) -> Optional["AdaptiveReasoningConfig"]:
        """Build a config from the ``llm.adaptive_reasoning`` mapping.

        A missing mapping means "feature off" and returns ``None`` so the
        pipeline keeps its previous single-call behaviour untouched.

        ``filler_phrases`` accepts two shapes:
        - a mapping ``{language: [phrases]}``: merged over the built-in
          per-language defaults, so the config wins language by language and
          built-ins cover the languages it omits;
        - a plain list ``[phrases]``: applied to every language and replaces
          the built-in mapping entirely (legacy behaviour).
        """
        if not isinstance(data, dict):
            return None

        phrases = data.get("filler_phrases")
        if isinstance(phrases, dict):
            # Merge the configured mapping over the built-in defaults: the
            # config wins language by language, and built-ins fill the
            # languages it does not mention. A per-language list therefore no
            # longer silently re-enables the English fallback for every
            # supported language it omits.
            by_language = cls._default_filler_mapping()
            by_language.update(cls._normalize_filler_mapping(phrases))
            filler_phrases = by_language.get("en") or list(DEFAULT_FILLER_PHRASES)
        elif isinstance(phrases, (list, tuple)) and phrases:
            # A plain list applies to every language (legacy behaviour) and
            # deliberately replaces the built-in mapping.
            by_language = {}
            filler_phrases = [str(phrase) for phrase in phrases]
        else:
            by_language = cls._default_filler_mapping()
            filler_phrases = list(DEFAULT_FILLER_PHRASES)

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
            filler_phrases=filler_phrases,
            filler_phrases_by_language=by_language,
            routing_prompt=routing_prompt,
        )

    @staticmethod
    def _normalize_filler_mapping(mapping: dict) -> dict[str, list[str]]:
        normalized: dict[str, list[str]] = {}
        for language, values in mapping.items():
            if not isinstance(values, (list, tuple)):
                continue
            cleaned = [str(value) for value in values if str(value).strip()]
            if cleaned:
                normalized[str(language).strip().lower()] = cleaned
        return normalized

    @staticmethod
    def _default_filler_mapping() -> dict[str, list[str]]:
        return {
            language: list(phrases)
            for language, phrases in DEFAULT_FILLER_PHRASES_BY_LANGUAGE.items()
        }

    def _filler_phrases_for(self, language_code: Optional[str]) -> list[str]:
        if not self.filler_phrases_by_language:
            return self.filler_phrases
        code = (language_code or "").replace("_", "-").split("-")[0].strip().lower()
        mapping = self.filler_phrases_by_language
        for key in (code, "default", "en"):
            phrases = mapping.get(key)
            if phrases:
                return phrases
        return self.filler_phrases

    def pick_filler(self, language_code: Optional[str] = None) -> str:
        """Return a waiting phrase, preferring the response language."""
        phrases = self._filler_phrases_for(language_code)
        if not phrases:
            phrases = list(DEFAULT_FILLER_PHRASES)
        return random.choice(phrases)


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
        candidate = self._held.lstrip()
        if not candidate:
            # Only whitespace so far: keep holding and keep waiting for the
            # first non-whitespace character, but drop the excess beyond the
            # memory bound. Resolving here as "direct" would swallow a marker
            # that arrives afterwards (the stripper removes it, leaving the
            # user with an empty answer).
            if len(self._held) > _MAX_LEADING_WHITESPACE:
                self._held = self._held[-_MAX_LEADING_WHITESPACE:]
            return ""

        decision = self._classify(candidate)
        if decision is None:
            return ""

        self._resolved = True
        self.decision = decision
        if decision == "direct":
            # Leading whitespace is part of a normal answer, so release it too.
            released = self._held
            self._held = ""
            return released
        # A marker: the buffered characters are routing noise and are dropped.
        self._held = ""
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
    """Return the first-call messages with the routing instruction added.

    The instruction is merged into the *content* of the last user message
    (prefixed), the same per-turn pattern used by
    ``ConversationPipeline._build_llm_messages``. Inserting a second
    ``role="system"`` message mid-conversation breaks providers whose chat
    templates require a single leading system message followed by strict
    user/assistant alternation (e.g. the Gemma adapter's ``_split_messages``
    plus ``apply_chat_template``), so the instruction travels with the user
    turn instead.

    A new list and a new final ``Message`` are returned; the caller's message
    objects are never mutated. When the last message is not a user turn (not
    the case for the normal pipeline, which always ends on a user message) the
    routing instruction is appended as a trailing system message as a
    documented fallback, rather than being dropped.
    """
    if messages and messages[-1].role == "user":
        last_user = messages[-1]
        merged = Message(
            role="user",
            content=f"{routing_prompt}\n\n{last_user.content}",
        )
        return [*messages[:-1], merged]
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


def accepts_first_token_kwarg(callback: Any) -> bool:
    """Return True when ``callback`` can take ``first_token_epoch_ms``."""
    if not callable(callback):
        return False
    try:
        signature = inspect.signature(callback)
    except (TypeError, ValueError):
        return False
    parameters = signature.parameters
    if "first_token_epoch_ms" in parameters:
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
    async with aclosing(stream) as active:
        async for chunk in active:
            yield chunk


async def stream_llm_with_adaptive_reasoning(
    llm: Any,
    messages: list[Message],
    config: Optional[AdaptiveReasoningConfig],
    *,
    on_escalation: Optional[Callable[[str, str, str], Awaitable[None]]] = None,
    language_code: Optional[str] = None,
) -> AsyncGenerator[str, None]:
    """Stream the LLM, escalating the reasoning effort when the model asks.

    ``on_escalation`` is awaited with ``(decision, effort, filler)`` just before
    the second call, so the caller can speak the waiting phrase immediately.
    ``language_code`` selects the filler phrase language when configured.
    """
    if config is None or not config.enabled:
        async with aclosing(_chat_stream(llm, messages, None)) as stream:
            async for chunk in stream:
                yield chunk
        return

    routing_messages = build_routing_messages(messages, config.routing_prompt)
    base_override = {"reasoning": {"effort": config.base_effort}}
    started = time.perf_counter()
    first_chunk_ms: Optional[float] = None
    first_token_epoch_ms: Optional[int] = None
    router = ReasoningMarkerRouter()
    stripper = MarkerStrippingFilter()

    # ``aclosing`` closes the provider stream deterministically on the marker
    # ``break`` (and if the consumer abandons this generator) instead of
    # leaving a live HTTP response behind during the escalated call.
    async with aclosing(_chat_stream(llm, routing_messages, base_override)) as stream:
        async for chunk in stream:
            if first_chunk_ms is None:
                first_chunk_ms = (time.perf_counter() - started) * 1000.0
                first_token_epoch_ms = int(time.time() * 1000)
                # The first routing chunk is the first token of the turn; mark
                # it before any escalation so ``first_sentence`` cannot precede
                # it in the canonical turn_latency order.
                get_turn_latency_tracker().mark("llm_first_token")
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
        filler = config.pick_filler(language_code)
        if accepts_first_token_kwarg(on_escalation):
            await on_escalation(
                decision,
                effort,
                filler,
                first_token_epoch_ms=first_token_epoch_ms,
            )
        else:
            await on_escalation(decision, effort, filler)

    escalated_override = {
        "reasoning": {"effort": effort},
        "max_completion_tokens": config.escalated_max_completion_tokens,
    }
    escalated_stripper = MarkerStrippingFilter()
    async with aclosing(_chat_stream(llm, messages, escalated_override)) as stream:
        async for chunk in stream:
            cleaned = escalated_stripper.feed(chunk)
            if cleaned:
                yield cleaned
    tail = escalated_stripper.flush()
    if tail:
        yield tail
