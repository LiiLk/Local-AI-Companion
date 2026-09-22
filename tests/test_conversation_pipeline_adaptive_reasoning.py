"""Integration tests for adaptive reasoning inside the conversation pipeline."""

import asyncio
import io
import time
import wave

from src.assistant.conversation_pipeline import ConversationConfig, ConversationPipeline
from src.assistant.reasoning_router import AdaptiveReasoningConfig
from src.tts.base import TTSResult


def _make_wav_bytes(sample_rate: int = 24000) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b"\x00\x00" * 2400)
    return buffer.getvalue()


class FakeASRResult:
    def __init__(self, text: str, language: str = "en", confidence=0.99):
        self.text = text
        self.language = language
        self.confidence = confidence
        self.segments = []


class EnglishASR:
    def transcribe(self, audio, language=None):
        return FakeASRResult("Tell me about the stars.", language="en")


class FrenchASR:
    def transcribe(self, audio, language=None):
        return FakeASRResult("Parle-moi des etoiles.", language="fr")


class KokoroProvider:
    def __init__(self):
        self.calls = []

    async def synthesize(self, text, output_path=None):
        self.calls.append(text)
        return TTSResult(audio_data=_make_wav_bytes())

    def set_language(self, language):
        pass


class SlowSynthTTS(KokoroProvider):
    """Records synthesis timing; each synth takes ``delay`` seconds."""

    def __init__(self, delay: float = 0.1):
        super().__init__()
        self.delay = delay
        self.intervals = []

    async def synthesize(self, text, output_path=None):
        started = time.perf_counter()
        await asyncio.sleep(self.delay)
        finished = time.perf_counter()
        self.intervals.append((text, started, finished))
        return await super().synthesize(text, output_path)


class AdaptiveLLM:
    """Fake LLM that records messages and per-request overrides."""

    def __init__(self, scripts):
        self._scripts = scripts
        self.calls = []

    async def chat_stream(self, messages, options_override=None):
        self.calls.append((messages, options_override))
        index = min(len(self.calls) - 1, len(self._scripts) - 1)
        for chunk in self._scripts[index]:
            yield chunk


class SlowSecondCallLLM(AdaptiveLLM):
    """Escalated call delays its first chunk so the filler audio wins the race."""

    async def chat_stream(self, messages, options_override=None):
        self.calls.append((messages, options_override))
        index = min(len(self.calls) - 1, len(self._scripts) - 1)
        if index >= 1:
            await asyncio.sleep(0.05)
        for chunk in self._scripts[index]:
            yield chunk


class TimedAdaptiveLLM(AdaptiveLLM):
    """Records the wall-clock start time of every ``chat_stream`` call."""

    def __init__(self, scripts):
        super().__init__(scripts)
        self.call_started = []

    async def chat_stream(self, messages, options_override=None):
        self.call_started.append(time.perf_counter())
        async for chunk in super().chat_stream(messages, options_override):
            yield chunk


def _run(llm, tts, payloads, chunks, *, stream_tts=True, adaptive=None):
    config = ConversationConfig(
        stream_tts=stream_tts,
        asr_language="auto",
        reply_language=None,
        adaptive_reasoning=adaptive,
    )
    pipeline = ConversationPipeline(llm=llm, tts=tts, asr=EnglishASR(), config=config)

    async def on_audio_ready(payload):
        payloads.append(payload)

    async def on_response_chunk(chunk):
        chunks.append(chunk)

    pipeline.on_audio_ready = on_audio_ready
    pipeline.on_response_chunk = on_response_chunk
    return pipeline, asyncio.run(pipeline.process_speech(b"\x00\x00" * 1600))


def test_think_marker_escalates_and_hides_marker_from_tts_and_history():
    adaptive = AdaptiveReasoningConfig.from_dict({"enabled": True})
    llm = AdaptiveLLM([["<|THINK|>"], ["Paris is the capital of France."]])
    tts = KokoroProvider()
    payloads, chunks = [], []

    pipeline, result = _run(llm, tts, payloads, chunks, adaptive=adaptive)

    assert result == "Paris is the capital of France."
    assert len(llm.calls) == 2
    assert llm.calls[1][1] == {
        "reasoning": {"effort": "medium"},
        "max_completion_tokens": 2048,
    }
    assert tts.calls[0] in adaptive.filler_phrases
    assert "Paris is the capital of France." in tts.calls
    assert all("<|THINK|>" not in text for text in tts.calls)
    assert all("<|THINK|>" not in chunk for chunk in chunks)
    assert pipeline.messages[-1].content == "Paris is the capital of France."


def test_direct_reply_uses_single_call_and_speaks_nothing_extra():
    adaptive = AdaptiveReasoningConfig.from_dict({"enabled": True})
    llm = AdaptiveLLM([["The stars are bright tonight."]])
    tts = KokoroProvider()
    payloads, chunks = [], []

    _pipeline, result = _run(llm, tts, payloads, chunks, adaptive=adaptive)

    assert result == "The stars are bright tonight."
    assert len(llm.calls) == 1
    assert llm.calls[0][1] == {"reasoning": {"effort": "none"}}
    assert tts.calls == ["The stars are bright tonight."]


def test_disabled_adaptive_reasoning_keeps_single_unmodified_call():
    adaptive = AdaptiveReasoningConfig.from_dict({"enabled": False})
    llm = AdaptiveLLM([["The stars are bright tonight."]])
    tts = KokoroProvider()
    payloads, chunks = [], []

    _pipeline, result = _run(llm, tts, payloads, chunks, adaptive=adaptive)

    assert result == "The stars are bright tonight."
    assert len(llm.calls) == 1
    assert llm.calls[0][1] is None
    assert llm.calls[0][0][-1].role == "user"


def test_non_streaming_path_speaks_filler_then_answer():
    adaptive = AdaptiveReasoningConfig.from_dict({"enabled": True})
    llm = AdaptiveLLM([["<|THINK_HARD|>"], ["The hard answer."]])
    tts = KokoroProvider()
    payloads, chunks = [], []

    _pipeline, result = _run(
        llm, tts, payloads, chunks, stream_tts=False, adaptive=adaptive
    )

    assert result == "The hard answer."
    assert llm.calls[1][1]["reasoning"]["effort"] == "high"
    assert tts.calls[0] in adaptive.filler_phrases
    assert tts.calls[1] == "The hard answer."


def test_french_response_escalation_speaks_french_filler():
    adaptive = AdaptiveReasoningConfig.from_dict(
        {
            "enabled": True,
            "filler_phrases": {"en": ["thinking"], "fr": ["reflexion"]},
        }
    )
    llm = AdaptiveLLM([["<|THINK|>"], ["Paris est la capitale de la France."]])
    tts = KokoroProvider()
    payloads, chunks = [], []
    config = ConversationConfig(
        stream_tts=True,
        asr_language="auto",
        reply_language="fr",
        adaptive_reasoning=adaptive,
    )
    pipeline = ConversationPipeline(llm=llm, tts=tts, asr=FrenchASR(), config=config)

    async def on_audio_ready(payload):
        payloads.append(payload)

    pipeline.on_audio_ready = on_audio_ready

    result = asyncio.run(pipeline.process_speech(b"\x00\x00" * 1600))

    assert result == "Paris est la capitale de la France."
    assert tts.calls[0] == "reflexion"
    assert "Paris est la capitale de la France." in " ".join(tts.calls)


def test_language_guard_rewrite_bypasses_adaptive_routing():
    adaptive = AdaptiveReasoningConfig.from_dict({"enabled": True})
    llm = AdaptiveLLM(
        [
            ["Voici une longue reponse en francais qui ne correspond pas du tout."],
            ["Here is a long answer in English that does not match at all."],
        ]
    )
    tts = KokoroProvider()
    payloads, chunks = [], []
    config = ConversationConfig(
        stream_tts=True,
        asr_language="auto",
        reply_language="en",
        adaptive_reasoning=adaptive,
    )
    pipeline = ConversationPipeline(llm=llm, tts=tts, asr=FrenchASR(), config=config)

    async def on_audio_ready(payload):
        payloads.append(payload)

    pipeline.on_audio_ready = on_audio_ready

    result = asyncio.run(pipeline.process_speech(b"\x00\x00" * 1600))

    assert result == "Here is a long answer in English that does not match at all."
    assert len(llm.calls) == 2
    # The internal rewrite must not receive the routing instruction.
    assert llm.calls[1][0][-1].role == "user"
    assert llm.calls[1][1] is None


def test_escalation_filler_payload_carries_llm_first_token_before_tts():
    adaptive = AdaptiveReasoningConfig.from_dict({"enabled": True})
    llm = SlowSecondCallLLM([["<|THINK|>"], ["Paris is the capital of France."]])
    tts = KokoroProvider()
    payloads, chunks = [], []

    _pipeline, result = _run(llm, tts, payloads, chunks, adaptive=adaptive)

    assert result == "Paris is the capital of France."
    assert tts.calls[0] in adaptive.filler_phrases
    assert len(payloads) >= 2
    filler_trace = payloads[0].trace
    answer_trace = payloads[-1].trace

    assert "llm_first_token_epoch_ms" in filler_trace
    assert "tts_first_chunk_epoch_ms" in filler_trace
    assert (
        filler_trace["llm_first_token_epoch_ms"]
        <= filler_trace["tts_first_chunk_epoch_ms"]
    )
    # The second call must not overwrite the routing timestamp.
    assert (
        answer_trace["llm_first_token_epoch_ms"]
        == filler_trace["llm_first_token_epoch_ms"]
    )
    assert (
        answer_trace["llm_first_token_epoch_ms"]
        <= answer_trace["tts_first_chunk_epoch_ms"]
    )


def test_direct_reply_trace_keeps_llm_first_token_before_tts():
    adaptive = AdaptiveReasoningConfig.from_dict({"enabled": True})
    llm = AdaptiveLLM([["The stars are bright tonight."]])
    tts = KokoroProvider()
    payloads, chunks = [], []

    _pipeline, result = _run(llm, tts, payloads, chunks, adaptive=adaptive)

    assert result == "The stars are bright tonight."
    assert len(llm.calls) == 1
    assert len(payloads) == 1
    trace = payloads[0].trace
    assert (
        trace["llm_first_token_epoch_ms"] <= trace["tts_first_chunk_epoch_ms"]
    )


def test_non_streaming_filler_synthesis_does_not_block_escalated_llm_call():
    adaptive = AdaptiveReasoningConfig.from_dict({"enabled": True})
    llm = TimedAdaptiveLLM([["<|THINK|>"], ["The hard answer."]])
    tts = SlowSynthTTS(delay=0.1)
    payloads, chunks = [], []

    _pipeline, result = _run(
        llm, tts, payloads, chunks, stream_tts=False, adaptive=adaptive
    )

    assert result == "The hard answer."
    assert len(llm.call_started) == 2
    filler_text = tts.calls[0]
    filler_end = next(
        finished
        for text, _started, finished in tts.intervals
        if text == filler_text
    )
    # The escalated LLM call starts while the filler is still synthesizing.
    assert llm.call_started[1] < filler_end
    # Filler audio is delivered before the answer audio.
    assert payloads[0].text == filler_text
    assert payloads[-1].text == "The hard answer."
