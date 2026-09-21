"""Unit tests for the pure adaptive reasoning router (no network, no I/O)."""

import pytest

from src.assistant.reasoning_router import (
    AdaptiveReasoningConfig,
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


def test_build_routing_messages_appends_system_instruction():
    messages = [Message(role="system", content="sys"), Message(role="user", content="hi")]

    routed = build_routing_messages(messages, "route")

    assert [message.role for message in routed] == ["system", "user", "system"]
    assert routed[-1].content == "route"


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
    assert messages[-1].role == "system"
    assert messages[-1].content == config.routing_prompt


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
