"""Regression tests for the streaming language guard (LIL-67).

Covers:
- short first fragments must not be judged on their own
- short replies under the threshold must not trigger a rewrite
- the wrong-language fallback must synthesize sentence-by-sentence
"""

import asyncio
import io
import wave

from src.assistant import conversation_pipeline
from src.assistant.conversation_pipeline import ConversationConfig, ConversationPipeline
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
    def __init__(self, text: str, language: str = "en", confidence=None):
        self.text = text
        self.language = language
        self.confidence = confidence
        self.segments = []


class FrenchASR:
    def transcribe(self, audio, language=None):
        return FakeASRResult("Parle-moi des etoiles.", language="fr", confidence=0.99)


class EnglishASR:
    def transcribe(self, audio, language=None):
        return FakeASRResult("Tell me about the stars.", language="en", confidence=0.99)


class KokoroProvider:
    def __init__(self):
        self.calls = []

    async def synthesize(self, text, output_path=None):
        self.calls.append(text)
        return TTSResult(audio_data=_make_wav_bytes())

    def set_language(self, language):
        pass


class ScriptedLLM:
    def __init__(self, replies):
        self._replies = replies
        self.calls = []

    async def chat_stream(self, messages):
        self.calls.append(messages)
        index = min(len(self.calls) - 1, len(self._replies) - 1)
        for chunk in self._replies[index]:
            yield chunk


def _run(llm, tts, payloads, asr=None):
    pipeline = ConversationPipeline(
        llm=llm,
        tts=tts,
        asr=asr or FrenchASR(),
        config=ConversationConfig(stream_tts=True, asr_language="auto", reply_language="en"),
    )

    async def on_audio_ready(payload):
        payloads.append(payload)

    pipeline.on_audio_ready = on_audio_ready
    return pipeline, asyncio.run(pipeline.process_speech(b"\x00\x00" * 1600))


def test_short_reply_below_language_threshold_is_not_rewritten():
    llm = ScriptedLLM([["Sure!"]])
    tts = KokoroProvider()
    payloads = []

    _pipeline, result = _run(llm, tts, payloads)

    assert result == "Sure!"
    assert len(llm.calls) == 1
    assert tts.calls == ["Sure!"]
    assert [payload.text for payload in payloads] == ["Sure!"]


def test_short_first_sentence_accumulates_before_language_decision():
    llm = ScriptedLLM([["Sure! ", "Here is the detailed answer about the stars."]])
    tts = KokoroProvider()
    payloads = []

    _pipeline, result = _run(llm, tts, payloads)

    assert result == "Sure! Here is the detailed answer about the stars."
    assert len(llm.calls) == 1
    assert tts.calls == ["Sure!", "Here is the detailed answer about the stars."]
    assert [payload.text for payload in payloads] == tts.calls


def test_wrong_language_fallback_synthesizes_sentence_by_sentence():
    llm = ScriptedLLM(
        [
            ["Les etoiles sont magnifiques. Elles sont infinies."],
            ["Stars are beautiful. They feel endless."],
        ]
    )
    tts = KokoroProvider()
    payloads = []

    _pipeline, result = _run(llm, tts, payloads)

    assert result == "Stars are beautiful. They feel endless."
    assert len(llm.calls) == 2
    assert tts.calls == ["Stars are beautiful.", "They feel endless."]
    assert [payload.text for payload in payloads] == ["Stars are beautiful.", "They feel endless."]


def test_short_english_reply_misdetected_as_romanian_is_not_rewritten():
    llm = ScriptedLLM([["Sure! An electric vehicle."]])
    tts = KokoroProvider()
    payloads = []

    _pipeline, result = _run(llm, tts, payloads)

    assert result == "Sure! An electric vehicle."
    assert len(llm.calls) == 1
    assert tts.calls == ["Sure!", "An electric vehicle."]


def test_short_english_reply_misdetected_as_tagalog_is_not_rewritten():
    llm = ScriptedLLM([["Okay! So, what's up today?"]])
    tts = KokoroProvider()
    payloads = []

    _pipeline, result = _run(llm, tts, payloads)

    assert result == "Okay! So, what's up today?"
    assert len(llm.calls) == 1
    assert tts.calls == ["Okay!", "So, what's up today?"]


def test_longer_english_reply_with_electric_vehicle_is_not_rewritten():
    llm = ScriptedLLM([["Sure! An electric vehicle works by storing energy."]])
    tts = KokoroProvider()
    payloads = []

    _pipeline, result = _run(llm, tts, payloads)

    assert result == "Sure! An electric vehicle works by storing energy."
    assert len(llm.calls) == 1
    assert tts.calls == ["Sure!", "An electric vehicle works by storing energy."]


def test_longer_english_reply_with_casual_greeting_is_not_rewritten():
    llm = ScriptedLLM([["Okay! So, what's up today? Tell me everything."]])
    tts = KokoroProvider()
    payloads = []

    _pipeline, result = _run(llm, tts, payloads)

    assert result == "Okay! So, what's up today? Tell me everything."
    assert len(llm.calls) == 1
    assert tts.calls == ["Okay!", "So, what's up today?", "Tell me everything."]


def test_french_reply_when_user_speaks_french_is_rewritten_to_english():
    llm = ScriptedLLM(
        [
            ["Bien sur ! Une voiture electrique fonctionne grace a une batterie."],
            ["Of course! An electric car runs on a battery."],
        ]
    )
    tts = KokoroProvider()
    payloads = []

    _pipeline, result = _run(llm, tts, payloads)

    assert result == "Of course! An electric car runs on a battery."
    assert len(llm.calls) == 2
    assert tts.calls == ["Of course!", "An electric car runs on a battery."]


def test_english_reply_when_user_speaks_english_is_not_rewritten():
    llm = ScriptedLLM([["Sure! An electric vehicle works by storing energy."]])
    tts = KokoroProvider()
    payloads = []

    _pipeline, result = _run(llm, tts, payloads, asr=EnglishASR())

    assert result == "Sure! An electric vehicle works by storing energy."
    assert len(llm.calls) == 1
    assert tts.calls == ["Sure!", "An electric vehicle works by storing energy."]


class _RecordingTTSManager:
    instances: list["_RecordingTTSManager"] = []

    def __init__(self, **kwargs):
        self.finish_calls = 0
        self.cancel_calls = 0
        self.submitted: list[str] = []
        _RecordingTTSManager.instances.append(self)

    async def start(self):
        return None

    async def submit(self, text, expression=None):
        self.submitted.append(text)

    async def finish(self):
        self.finish_calls += 1

    async def cancel(self):
        self.cancel_calls += 1


def test_fallback_rewrite_failure_cancels_tts_manager(monkeypatch):
    """Regression for LIL-67: a non-cancel exception during the full-response
    rewrite must still cancel the TTSTaskManager, otherwise its worker task is
    left blocked on the queue forever.
    """

    _RecordingTTSManager.instances = []
    monkeypatch.setattr(
        conversation_pipeline, "TTSTaskManager", _RecordingTTSManager
    )

    llm = ScriptedLLM([["Les etoiles sont magnifiques et tres brillantes."]])
    tts = KokoroProvider()
    payloads = []
    pipeline = ConversationPipeline(
        llm=llm,
        tts=tts,
        asr=FrenchASR(),
        config=ConversationConfig(
            stream_tts=True, asr_language="auto", reply_language="en"
        ),
    )

    async def on_audio_ready(payload):
        payloads.append(payload)

    pipeline.on_audio_ready = on_audio_ready

    async def _boom(*args, **kwargs):
        raise RuntimeError("rewrite boom")

    monkeypatch.setattr(pipeline, "_ensure_response_language", _boom)

    result = asyncio.run(pipeline.process_speech(b"\x00\x00" * 1600))

    assert result is None
    manager = _RecordingTTSManager.instances[-1]
    assert manager.cancel_calls == 1
    assert manager.finish_calls == 0