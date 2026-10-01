"""Unit tests for the pure adaptive reasoning router (no network, no I/O)."""

import pytest

import src.assistant.reasoning_router as reasoning_router
from src.assistant.reasoning_router import (
    AdaptiveReasoningConfig,
    DEFAULT_FILLER_PHRASES_BY_LANGUAGE,
    MarkerStrippingFilter,
    ReasoningMarkerRouter,
    THINK_HARD_MARKER,
    THINK_MARKER,
    accepts_options_override,
    build_routing_messages,
    stream_llm_with_adaptive_reasoning,
    strip_reasoning_markers,
)
from src.llm.base import Message
from src.llm.gemma_text_vision_llm import GemmaTextVisionLLM
from src.utils.language_detection import LANGUAGE_NAMES


class RecordingLLM:
    """Fake LLM that accepts ``options_override`` and records every call."""

    def __init__(self, scripts):
        self._scripts = scripts
        self.calls = []

    async def chat_stream(self, messages, options_override=None):
        self.calls.append((messages, options_override))
        index = min(len(self.calls) - 1, len(self._scripts) - 1)
        for chunk in self._scripts[index]:
            yield chunk


class NoOverrideLLM:
    """Fake provider whose ``chat_stream`` does not support the override."""

    def __init__(self, scripts):
        self._scripts = scripts
        self.calls = []

    async def chat_stream(self, messages):
        self.calls.append(messages)
        index = min(len(self.calls) - 1, len(self._scripts) - 1)
        for chunk in self._scripts[index]:
            yield chunk


class CloseTrackingLLM:
    """Fake provider whose generators record when they are closed."""

    def __init__(self, scripts):
        self._scripts = scripts
        self.events = []
        self.calls = 0

    async def chat_stream(self, messages, options_override=None):
        index = self.calls
        self.calls += 1
        self.events.append(f"start{index}")
        try:
            for chunk in self._scripts[index]:
                yield chunk
        finally:
            self.events.append(f"close{index}")


def _enabled(**overrides) -> AdaptiveReasoningConfig:
    data = {"enabled": True}
    data.update(overrides)
    return AdaptiveReasoningConfig.from_dict(data)


async def _collect(stream):
    return [chunk async for chunk in stream]


def test_router_releases_plain_text_immediately():
    router = ReasoningMarkerRouter()

    assert router.feed("Hello") == "Hello"
    assert router.decision == "direct"
    assert router.resolved is True


def test_router_holds_until_prefix_is_ruled_out():
    router = ReasoningMarkerRouter()

    assert router.feed("<") == ""
    assert router.feed("3 you") == "<3 you"
    assert router.decision == "direct"


def test_router_detects_think_marker_split_across_chunks():
    router = ReasoningMarkerRouter()

    assert router.feed("<|TH") == ""
    assert router.feed("INK|>") == ""
    assert router.decision == "think"
    assert router.feed("ignored") == "ignored"


def test_router_detects_hard_marker():
    router = ReasoningMarkerRouter()

    assert router.feed(THINK_HARD_MARKER) == ""
    assert router.decision == "think_hard"


def test_router_flush_releases_unresolved_prefix():
    router = ReasoningMarkerRouter()

    assert router.feed("<") == ""
    assert router.flush() == "<"
    assert router.decision == "direct"


def test_router_holdback_is_bounded_by_longest_marker():
    router = ReasoningMarkerRouter()

    released = router.feed("This is a normal sentence.")
    assert released == "This is a normal sentence."


def test_router_holds_leading_whitespace_until_it_can_classify():
    router = ReasoningMarkerRouter()

    assert router.feed(" ") == ""
    assert router.resolved is False
    assert router.feed("Hello") == " Hello"
    assert router.decision == "direct"


def test_router_classifies_marker_after_leading_whitespace():
    router = ReasoningMarkerRouter()

    assert router.feed("\n") == ""
    assert router.feed("<|TH") == ""
    assert router.feed("INK|>") == ""
    assert router.decision == "think"


def test_router_bounds_whitespace_only_stream_without_resolving():
    router = ReasoningMarkerRouter()

    released = router.feed(" " * 64)

    assert released == ""
    assert router.resolved is False
    # The holdback is bounded; the excess whitespace may be dropped.
    assert len(router.flush()) <= reasoning_router._MAX_LEADING_WHITESPACE
    assert router.decision == "direct"


def test_router_does_not_resolve_whitespace_cap_then_detects_marker():
    router = ReasoningMarkerRouter()

    released = ""
    for _ in range(20):
        released += router.feed(" ")

    assert released == ""
    assert router.resolved is False
    assert router.feed("<|THINK|>") == ""
    assert router.decision == "think"


def test_router_does_not_resolve_whitespace_cap_then_releases_direct():
    router = ReasoningMarkerRouter()

    for _ in range(20):
        router.feed(" ")

    released = router.feed("Hello")

    assert released.strip() == "Hello"
    assert router.resolved is True
    assert router.decision == "direct"


def test_strip_reasoning_markers_removes_embedded_markers():
    assert strip_reasoning_markers(f"answer {THINK_MARKER} here") == "answer  here"
    assert strip_reasoning_markers(f"{THINK_HARD_MARKER}answer") == "answer"


def test_stripping_filter_removes_marker_split_across_chunks():
    stripper = MarkerStrippingFilter()

    assert stripper.feed("Hello <|TH") == "Hello "
    assert stripper.feed("INK|> world") == " world"
    assert stripper.flush() == ""


def test_stripping_filter_flushes_partial_prefix_at_end():
    stripper = MarkerStrippingFilter()

    assert stripper.feed("5 <") == "5 "
    assert stripper.flush() == "<"


@pytest.mark.asyncio
async def test_marker_split_across_second_call_chunks_is_removed():
    llm = RecordingLLM([["<|THINK|>"], ["The ", "<|TH", "INK|>", "answer."]])
    config = _enabled()

    chunks = await _collect(
        stream_llm_with_adaptive_reasoning(
            llm,
            [Message(role="user", content="hi")],
            config,
        )
    )

    assert "".join(chunks) == "The answer."
    assert len(llm.calls) == 2


def test_from_dict_missing_mapping_disables_feature():
    assert AdaptiveReasoningConfig.from_dict(None) is None
    assert AdaptiveReasoningConfig.from_dict("nope") is None


def test_from_dict_defaults_and_overrides():
    config = AdaptiveReasoningConfig.from_dict({"enabled": True})

    assert config.enabled is True
    assert config.base_effort == "none"
    assert config.escalate_effort == "medium"
    assert config.max_effort == "high"
    assert config.escalated_max_completion_tokens == 2048
    assert config.filler_phrases
    assert THINK_MARKER in config.routing_prompt


def test_from_dict_reads_custom_values():
    config = AdaptiveReasoningConfig.from_dict(
        {
            "enabled": True,
            "base_effort": "low",
            "escalate_effort": "high",
            "max_effort": "high",
            "escalated_max_completion_tokens": 4096,
            "filler_phrases": ["One moment."],
            "routing_prompt": "Route this.",
        }
    )

    assert config.base_effort == "low"
    assert config.escalated_max_completion_tokens == 4096
    assert config.filler_phrases == ["One moment."]
    assert config.routing_prompt == "Route this."


def test_build_routing_messages_merges_instruction_into_last_user():
    messages = [Message(role="system", content="sys"), Message(role="user", content="hi")]

    routed = build_routing_messages(messages, "route")

    assert [message.role for message in routed] == ["system", "user"]
    assert routed[-1].content == "route\n\nhi"
    # The caller's message objects are never mutated.
    assert messages[-1].content == "hi"


def test_build_routing_messages_keeps_single_system_with_full_history():
    messages = [
        Message(role="system", content="sys"),
        Message(role="user", content="hi"),
        Message(role="assistant", content="hello"),
        Message(role="user", content="again"),
    ]

    routed = build_routing_messages(messages, "route")

    assert sum(1 for message in routed if message.role == "system") == 1
    assert [message.role for message in routed] == [
        "system",
        "user",
        "assistant",
        "user",
    ]
    assert routed[-1].content == "route\n\nagain"
    assert messages[-1].content == "again"


def test_build_routing_messages_appends_system_when_no_trailing_user():
    messages = [
        Message(role="system", content="sys"),
        Message(role="assistant", content="a"),
    ]

    routed = build_routing_messages(messages, "route")

    assert [message.role for message in routed] == ["system", "assistant", "system"]
    assert routed[-1].content == "route"


def test_routing_messages_preserve_gemma_current_prompt():
    messages = [Message(role="system", content="sys"), Message(role="user", content="hi")]
    routed = build_routing_messages(messages, "route")

    latest, history = GemmaTextVisionLLM._split_messages(None, routed)

    assert latest == "route\n\nhi"
    assert [entry["role"] for entry in history] == ["system"]


def test_accepts_options_override_detects_support():
    assert accepts_options_override(RecordingLLM([["x"]])) is True
    assert accepts_options_override(NoOverrideLLM([["x"]])) is False


@pytest.mark.asyncio
async def test_direct_reply_uses_single_call_with_base_effort():
    llm = RecordingLLM([["Hello", " world"]])
    config = _enabled()

    chunks = await _collect(
        stream_llm_with_adaptive_reasoning(
            llm,
            [Message(role="user", content="hi")],
            config,
        )
    )

    assert chunks == ["Hello", " world"]
    assert len(llm.calls) == 1
    messages, override = llm.calls[0]
    assert override == {"reasoning": {"effort": "none"}}
    assert len(messages) == 1
    assert messages[-1].role == "user"
    assert messages[-1].content == f"{config.routing_prompt}\n\nhi"


@pytest.mark.asyncio
async def test_think_marker_escalates_and_speaks_filler_first():
    llm = RecordingLLM([["<|THINK|>"], ["The answer."]])
    config = _enabled()
    events = []

    async def on_escalation(decision, effort, filler):
        events.append((decision, effort, filler))

    chunks = await _collect(
        stream_llm_with_adaptive_reasoning(
            llm,
            [Message(role="user", content="hi")],
            config,
            on_escalation=on_escalation,
        )
    )

    assert chunks == ["The answer."]
    assert len(llm.calls) == 2
    first_messages, first_override = llm.calls[0]
    second_messages, second_override = llm.calls[1]
    assert first_override == {"reasoning": {"effort": "none"}}
    assert second_override == {
        "reasoning": {"effort": "medium"},
        "max_completion_tokens": 2048,
    }
    assert second_messages == [Message(role="user", content="hi")]
    assert len(events) == 1
    assert events[0][0] == "think"
    assert events[0][1] == "medium"
    assert events[0][2] in config.filler_phrases


@pytest.mark.asyncio
async def test_think_hard_marker_uses_max_effort():
    llm = RecordingLLM([["<|THINK_HARD|>"], ["Hard answer."]])
    config = _enabled()

    chunks = await _collect(
        stream_llm_with_adaptive_reasoning(
            llm,
            [Message(role="user", content="hi")],
            config,
        )
    )

    assert chunks == ["Hard answer."]
    assert llm.calls[1][1]["reasoning"]["effort"] == "high"


@pytest.mark.asyncio
async def test_marker_in_second_call_is_stripped_without_third_call():
    llm = RecordingLLM([["<|THINK|>"], [f"{THINK_MARKER}The answer."]])
    config = _enabled()

    chunks = await _collect(
        stream_llm_with_adaptive_reasoning(
            llm,
            [Message(role="user", content="hi")],
            config,
        )
    )

    assert chunks == ["The answer."]
    assert len(llm.calls) == 2


@pytest.mark.asyncio
async def test_provider_without_override_still_escalates_without_error():
    llm = NoOverrideLLM([["<|THINK|>"], ["The answer."]])
    config = _enabled()

    chunks = await _collect(
        stream_llm_with_adaptive_reasoning(
            llm,
            [Message(role="user", content="hi")],
            config,
        )
    )

    assert chunks == ["The answer."]
    assert len(llm.calls) == 2


@pytest.mark.asyncio
async def test_disabled_config_uses_single_unmodified_call():
    llm = RecordingLLM([["Hello"]])
    config = AdaptiveReasoningConfig(enabled=False)

    chunks = await _collect(
        stream_llm_with_adaptive_reasoning(
            llm,
            [Message(role="user", content="hi")],
            config,
        )
    )

    assert chunks == ["Hello"]
    assert len(llm.calls) == 1
    assert llm.calls[0][1] is None
    assert llm.calls[0][0] == [Message(role="user", content="hi")]


@pytest.mark.asyncio
async def test_first_stream_is_closed_before_the_escalated_call():
    llm = CloseTrackingLLM([["<|THINK|>"], ["The answer."]])
    config = _enabled()

    chunks = await _collect(
        stream_llm_with_adaptive_reasoning(
            llm,
            [Message(role="user", content="hi")],
            config,
        )
    )

    assert chunks == ["The answer."]
    assert llm.events.index("close0") < llm.events.index("start1")


@pytest.mark.asyncio
async def test_abandoned_router_stream_closes_the_provider_stream():
    llm = CloseTrackingLLM([["a", "b", "c"]])
    config = _enabled()
    stream = stream_llm_with_adaptive_reasoning(
        llm,
        [Message(role="user", content="hi")],
        config,
    )

    assert await stream.__anext__() == "a"
    await stream.aclose()

    assert "close0" in llm.events


@pytest.mark.asyncio
async def test_first_routing_chunk_marks_llm_first_token_before_escalation(monkeypatch):
    class RecordingTracker:
        def __init__(self):
            self.events = []

        def mark(self, stage):
            self.events.append(stage)

    tracker = RecordingTracker()
    monkeypatch.setattr(reasoning_router, "get_turn_latency_tracker", lambda: tracker)
    llm = RecordingLLM([["<|THINK|>"], ["The answer."]])
    config = _enabled()

    async def on_escalation(decision, effort, filler):
        tracker.events.append("escalation")

    chunks = await _collect(
        stream_llm_with_adaptive_reasoning(
            llm,
            [Message(role="user", content="hi")],
            config,
            on_escalation=on_escalation,
        )
    )

    assert chunks == ["The answer."]
    assert tracker.events == ["llm_first_token", "reasoning_escalated", "escalation"]


@pytest.mark.asyncio
async def test_escalation_passes_preexisting_first_token_epoch_ms_to_callback():
    llm = RecordingLLM([["<|THINK|>"], ["The answer."]])
    config = _enabled()
    received = {}

    async def on_escalation(decision, effort, filler, first_token_epoch_ms=None):
        received["value"] = first_token_epoch_ms

    chunks = await _collect(
        stream_llm_with_adaptive_reasoning(
            llm,
            [Message(role="user", content="hi")],
            config,
            on_escalation=on_escalation,
        )
    )

    assert chunks == ["The answer."]
    assert isinstance(received["value"], int)


@pytest.mark.asyncio
async def test_positional_three_arg_escalation_callback_still_supported():
    llm = RecordingLLM([["<|THINK|>"], ["The answer."]])
    config = _enabled()
    events = []

    async def on_escalation(decision, effort, filler):
        events.append((decision, effort, filler))

    chunks = await _collect(
        stream_llm_with_adaptive_reasoning(
            llm,
            [Message(role="user", content="hi")],
            config,
            on_escalation=on_escalation,
        )
    )

    assert chunks == ["The answer."]
    assert len(events) == 1


def test_pick_filler_selects_phrase_by_language():
    config = AdaptiveReasoningConfig.from_dict(
        {
            "enabled": True,
            "filler_phrases": {
                "en": ["thinking"],
                "fr": ["reflexion"],
                "default": ["fallback"],
            },
        }
    )

    assert config.pick_filler("fr-FR") == "reflexion"
    assert config.pick_filler("fr") == "reflexion"
    # Built-in phrases now fill languages the config does not override, so
    # "de" is no longer an unknown language falling back to "default".
    assert config.pick_filler("de") in DEFAULT_FILLER_PHRASES_BY_LANGUAGE["de"]
    # An unknown language still uses the configured "default" fallback.
    assert config.pick_filler("xx") == "fallback"
    assert config.pick_filler(None) == "fallback"


def test_config_mapping_falls_back_to_builtin_filler_per_language():
    config = AdaptiveReasoningConfig.from_dict(
        {"enabled": True, "filler_phrases": {"en": ["thinking"], "fr": ["reflexion"]}}
    )

    assert config.pick_filler("es") in DEFAULT_FILLER_PHRASES_BY_LANGUAGE["es"]
    assert config.pick_filler("ja") in DEFAULT_FILLER_PHRASES_BY_LANGUAGE["ja"]
    # An unknown language still falls back to the (possibly overridden) English.
    assert config.pick_filler("xx") == "thinking"


def test_config_mapping_overrides_builtin_per_language():
    config = AdaptiveReasoningConfig.from_dict(
        {"enabled": True, "filler_phrases": {"es": ["Custom"]}}
    )

    assert config.pick_filler("es") == "Custom"
    assert config.pick_filler("fr") in DEFAULT_FILLER_PHRASES_BY_LANGUAGE["fr"]


def test_default_filler_phrases_cover_all_supported_languages():
    languages = set(LANGUAGE_NAMES)
    config = AdaptiveReasoningConfig.from_dict({"enabled": True})

    assert languages <= set(DEFAULT_FILLER_PHRASES_BY_LANGUAGE)
    for language in languages:
        assert (
            config.pick_filler(language)
            in DEFAULT_FILLER_PHRASES_BY_LANGUAGE[language]
        )


def test_pick_filler_list_stays_language_agnostic():
    config = AdaptiveReasoningConfig.from_dict(
        {"enabled": True, "filler_phrases": ["only"]}
    )

    assert config.pick_filler("fr") == "only"
    assert config.pick_filler(None) == "only"


def test_default_filler_phrases_are_language_aware():
    config = AdaptiveReasoningConfig.from_dict({"enabled": True})

    assert config.pick_filler("fr") in DEFAULT_FILLER_PHRASES_BY_LANGUAGE["fr"]
    assert config.pick_filler("en") in DEFAULT_FILLER_PHRASES_BY_LANGUAGE["en"]


@pytest.mark.asyncio
async def test_escalation_uses_filler_for_the_response_language():
    config = AdaptiveReasoningConfig.from_dict(
        {
            "enabled": True,
            "filler_phrases": {"en": ["thinking"], "fr": ["reflexion"]},
        }
    )
    llm = RecordingLLM([["<|THINK|>"], ["The answer."]])
    fillers = []

    async def on_escalation(decision, effort, filler):
        fillers.append(filler)

    await _collect(
        stream_llm_with_adaptive_reasoning(
            llm,
            [Message(role="user", content="hi")],
            config,
            on_escalation=on_escalation,
            language_code="fr-FR",
        )
    )

    assert fillers == ["reflexion"]
