"""Regression tests for ASR short-clip robustness guards (LIL-ASR).

Covers:
- a segment ending past the real audio duration rejects the whole turn
  before any LLM call
- a retry that is itself rejected by the guard is never accepted
"""

import asyncio

from src.assistant.conversation_pipeline import ConversationConfig, ConversationPipeline
from src.asr.base import ASRResult


class _LLM:
    def __init__(self):
        self.calls = []

    async def chat_stream(self, messages):
        self.calls.append(messages)
        yield "ok"


class _TTS:
    async def synthesize(self, text, output_path=None):
        return None

    def set_language(self, language):
        pass


def _audio_bytes(seconds: float) -> bytes:
    return b"\x00\x00" * int(16000 * seconds)


def _pipeline(asr) -> ConversationPipeline:
    pipeline = ConversationPipeline(
        llm=_LLM(),
        tts=_TTS(),
        asr=asr,
        config=ConversationConfig(
            stream_tts=False, asr_language="auto", reply_language="en"
        ),
    )
    pipeline._last_user_language_code = "fr"
    return pipeline


def test_segment_past_audio_duration_is_rejected_before_llm():
    class OverrunASR:
        def __init__(self):
            self.calls = 0

        def transcribe(self, audio, language=None):
            self.calls += 1
            return ASRResult(
                text="C'est parti.",
                language="fr",
                confidence=0.9,
                duration=30.0,
                segments=[
                    {
                        "text": "C'est parti.",
                        "start": 0.0,
                        "end": 30.0,
                        "confidence": -0.2,
                    }
                ],
            )

    asr = OverrunASR()
    pipeline = _pipeline(asr)

    result = asyncio.run(pipeline.process_speech(_audio_bytes(2.0)))

    assert result is None
    assert pipeline.llm.calls == []
    assert asr.calls == 1


def test_rejected_retry_is_never_accepted():
    class RetryOverrunASR:
        def __init__(self):
            self.calls = 0

        def transcribe(self, audio, language=None):
            self.calls += 1
            if self.calls == 1:
                return ASRResult(
                    text="You",
                    language="en",
                    confidence=0.30,
                    duration=2.0,
                    segments=[
                        {"text": "You", "start": 0.0, "end": 0.9, "confidence": -1.2}
                    ],
                )
            return ASRResult(
                text="C'est parti.",
                language="fr",
                confidence=0.9,
                duration=30.0,
                segments=[
                    {
                        "text": "C'est parti.",
                        "start": 0.0,
                        "end": 30.0,
                        "confidence": -0.2,
                    }
                ],
            )

    asr = RetryOverrunASR()
    pipeline = _pipeline(asr)

    result = asyncio.run(pipeline.process_speech(_audio_bytes(2.0)))

    assert result is None
    assert pipeline.llm.calls == []
    assert asr.calls >= 2