import asyncio
import io
import shutil
import time
import wave
from pathlib import Path
from uuid import uuid4

from src.assistant.conversation_memory import ConversationMemoryConfig, ConversationMemoryStore
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
    def __init__(self, text: str, language: str = "en", confidence=None, segments=None):
        self.text = text
        self.language = language
        self.confidence = confidence
        self.segments = segments or []


class FakeASR:
    def transcribe(self, audio, language=None):
        return FakeASRResult("Hello")


class FakeLLM:
    def __init__(self, chunks):
        self._chunks = chunks

    async def chat_stream(self, messages):
        for chunk in self._chunks:
            yield chunk


class Qwen3TTSProvider:
    def __init__(self):
        self.calls = []

    async def synthesize(self, text, output_path=None):
        self.calls.append(text)
        if output_path is not None:
            output_path.write_bytes(_make_wav_bytes())
            return TTSResult(audio_path=output_path)
        return TTSResult(audio_data=_make_wav_bytes())


class StreamingQwen3TTSProvider(Qwen3TTSProvider):
    prefer_full_response_tts = False


class KokoroProvider:
    def __init__(self):
        self.calls = []
        self.languages = []

    async def synthesize(self, text, output_path=None):
        self.calls.append(text)
        return TTSResult(audio_data=_make_wav_bytes())

    def set_language(self, language):
        self.languages.append(language)


def _test_dir(name: str) -> Path:
    path = Path.cwd() / ".codex_test_artifacts" / f"{name}-{uuid4().hex}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def test_qwen3_uses_single_shot_tts_even_when_streaming_enabled():
    tts = Qwen3TTSProvider()
    payloads = []
    pipeline = ConversationPipeline(
        llm=FakeLLM(["Hello. ", "How are you?"]),
        tts=tts,
        asr=FakeASR(),
        config=ConversationConfig(stream_tts=True),
    )

    async def on_audio_ready(payload):
        payloads.append(payload)

    pipeline.on_audio_ready = on_audio_ready

    result = asyncio.run(pipeline.process_speech(b"\x00\x00" * 1600))

    assert result == "Hello. How are you?"
    assert tts.calls == ["Hello. How are you?"]
    assert len(payloads) == 1


def test_qwen3_can_opt_in_to_sentence_streaming():
    tts = StreamingQwen3TTSProvider()
    payloads = []
    pipeline = ConversationPipeline(
        llm=FakeLLM(["Hello. ", "How are you?"]),
        tts=tts,
        asr=FakeASR(),
        config=ConversationConfig(stream_tts=True),
    )

    async def on_audio_ready(payload):
        payloads.append(payload)

    pipeline.on_audio_ready = on_audio_ready

    result = asyncio.run(pipeline.process_speech(b"\x00\x00" * 1600))

    assert result == "Hello. How are you?"
    assert tts.calls == ["Hello.", "How are you?"]
    assert [payload.text for payload in payloads] == ["Hello.", "How are you?"]


def test_kokoro_keeps_sentence_streaming_when_enabled():
    tts = KokoroProvider()
    payloads = []
    pipeline = ConversationPipeline(
        llm=FakeLLM(["Hello. ", "How are you?"]),
        tts=tts,
        asr=FakeASR(),
        config=ConversationConfig(stream_tts=True),
    )

    async def on_audio_ready(payload):
        payloads.append(payload)

    pipeline.on_audio_ready = on_audio_ready

    result = asyncio.run(pipeline.process_speech(b"\x00\x00" * 1600))

    assert result == "Hello. How are you?"
    assert tts.calls == ["Hello.", "How are you?"]
    assert [payload.text for payload in payloads] == ["Hello.", "How are you?"]


def test_streaming_tts_can_start_before_first_punctuation():
    tts = KokoroProvider()
    pipeline = ConversationPipeline(
        llm=FakeLLM(["Je peux t'aider tout de suite", " sans problème."]),
        tts=tts,
        asr=FakeASR(),
        config=ConversationConfig(stream_tts=True),
    )

    result = asyncio.run(pipeline.process_speech(b"\x00\x00" * 1600))

    assert result == "Je peux t'aider tout de suite sans problème."
    assert len(tts.calls) >= 2
    assert not tts.calls[0].endswith(".")
    assert "sans problème" not in tts.calls[0]
    assert tts.calls[-1].endswith("problème.")


def test_streaming_tts_does_not_block_llm_stream():
    class SlowKokoroProvider(KokoroProvider):
        async def synthesize(self, text, output_path=None):
            self.calls.append(text)
            await asyncio.sleep(0.2)
            return TTSResult(audio_data=_make_wav_bytes())

    tts = SlowKokoroProvider()
    pipeline = ConversationPipeline(
        llm=FakeLLM(["Hello. ", "How are you?"]),
        tts=tts,
        asr=FakeASR(),
        config=ConversationConfig(stream_tts=True),
    )

    chunk_times = {}
    started = time.monotonic()

    async def on_response_chunk(chunk):
        if chunk == "How are you?":
            chunk_times["second_chunk_s"] = time.monotonic() - started

    pipeline.on_response_chunk = on_response_chunk

    result = asyncio.run(pipeline.process_speech(b"\x00\x00" * 1600))

    assert result == "Hello. How are you?"
    assert chunk_times["second_chunk_s"] < 0.1


def test_process_text_streams_tts_in_desktop_pipeline_mode():
    tts = KokoroProvider()
    payloads = []
    pipeline = ConversationPipeline(
        llm=FakeLLM(["Hello. ", "How are you?"]),
        tts=tts,
        asr=FakeASR(),
        config=ConversationConfig(stream_tts=True, asr_language="en"),
    )

    async def on_audio_ready(payload):
        payloads.append(payload)

    pipeline.on_audio_ready = on_audio_ready

    result = asyncio.run(pipeline.process_text("Hello there"))

    assert result == "Hello. How are you?"
    assert tts.calls == ["Hello.", "How are you?"]
    assert [payload.text for payload in payloads] == ["Hello.", "How are you?"]
    assert pipeline.messages[-2].content == "Hello there"


def test_process_text_loads_persistent_memory_and_rebounds_after_turn():
    class CapturingLLM:
        def __init__(self):
            self.calls = []

        async def chat_stream(self, messages):
            self.calls.append(messages)
            yield "Current reply."

    test_dir = _test_dir("pipeline-memory")
    try:
        memory_store = ConversationMemoryStore(
            ConversationMemoryConfig(
                history_path=test_dir / "conversation.jsonl",
                summary_path=test_dir / "summary.txt",
                max_recent_turns=1,
            )
        )
        memory_store.append_exchange("old one", "old reply one")
        memory_store.append_exchange("old two", "old reply two")

        llm = CapturingLLM()
        pipeline = ConversationPipeline(
            llm=llm,
            tts=KokoroProvider(),
            asr=FakeASR(),
            config=ConversationConfig(stream_tts=False, asr_language="en"),
            memory_store=memory_store,
        )

        result = asyncio.run(pipeline.process_text("current question"))

        assert result == "Current reply."
        prompt_contents = [message.content for message in llm.calls[0]]
        assert "old one" not in prompt_contents
        assert "old reply one" not in prompt_contents
        assert "old two" in prompt_contents
        assert "old reply two" in prompt_contents
        assert "current question" in prompt_contents[-1]
        assert [message.content for message in pipeline.messages[-2:]] == [
            "current question",
            "Current reply.",
        ]
    finally:
        shutil.rmtree(test_dir, ignore_errors=True)


def test_process_text_updates_curated_memory_summary_after_response():
    class ScriptedLLM:
        def __init__(self):
            self.calls = []

        async def chat_stream(self, messages):
            self.calls.append(messages)
            if len(self.calls) == 1:
                yield "Sure, I will keep it brief."
            else:
                yield '{"summary":"- User prefers brief implementation notes."}'

    test_dir = _test_dir("pipeline-curation")
    try:
        memory_store = ConversationMemoryStore(
            ConversationMemoryConfig(
                history_path=test_dir / "conversation.jsonl",
                summary_path=test_dir / "summary.txt",
                max_recent_turns=1,
            )
        )
        llm = ScriptedLLM()
        pipeline = ConversationPipeline(
            llm=llm,
            tts=KokoroProvider(),
            asr=FakeASR(),
            config=ConversationConfig(stream_tts=False, asr_language="en"),
            memory_store=memory_store,
        )

        result = asyncio.run(pipeline.process_text("I prefer brief implementation notes."))

        assert result == "Sure, I will keep it brief."
        assert len(llm.calls) == 2
        assert memory_store.load_summary() == "- User prefers brief implementation notes."
        assert any(
            "Relevant memory summary" in message.content
            for message in pipeline.messages
            if message.role == "system"
        )
    finally:
        shutil.rmtree(test_dir, ignore_errors=True)


def test_pipeline_keeps_detected_cjk_language_without_french_fallback():
    class GuardedASR:
        def __init__(self):
            self.calls = []

        def transcribe(self, audio, language=None):
            self.calls.append(language)
            if language in (None, "", "auto"):
                return FakeASRResult("セルクロス", language="ja")
            return FakeASRResult("Salut", language=language)

    class CapturingLLM:
        def __init__(self):
            self.messages = None

        async def chat_stream(self, messages):
            self.messages = messages
            yield "Bonjour."

    llm = CapturingLLM()
    asr = GuardedASR()
    tts = KokoroProvider()
    pipeline = ConversationPipeline(
        llm=llm,
        tts=tts,
        asr=asr,
        config=ConversationConfig(stream_tts=False, asr_language="auto"),
    )

    result = asyncio.run(pipeline.process_speech(b"\x00\x00" * 1600))

    assert result == "Bonjour."
    assert asr.calls == ["auto"]
    assert "Reply ONLY in Japanese" in llm.messages[-1].content


def test_pipeline_retries_low_confidence_language_with_previous_language_hint():
    class GuardedASR:
        def __init__(self):
            self.calls = []

        def transcribe(self, audio, language=None):
            self.calls.append(language)
            if language in (None, "", "auto"):
                return FakeASRResult("Как оно осталось?", language="ru", confidence=0.57)
            return FakeASRResult("Comment ça va ?", language=language, confidence=0.99)

    class CapturingLLM:
        def __init__(self):
            self.messages = None

        async def chat_stream(self, messages):
            self.messages = messages
            yield "Ça va bien."

    llm = CapturingLLM()
    asr = GuardedASR()
    tts = KokoroProvider()
    pipeline = ConversationPipeline(
        llm=llm,
        tts=tts,
        asr=asr,
        config=ConversationConfig(stream_tts=False, asr_language="auto"),
    )
    pipeline._last_user_language_code = "fr"

    result = asyncio.run(pipeline.process_speech(b"\x00\x00" * 1600))

    assert result == "Ça va bien."
    assert asr.calls == ["auto", "fr"]
    assert "Comment ça va ?" in llm.messages[-1].content


def test_pipeline_reply_language_does_not_force_asr_retry_language():
    class GuardedASR:
        def __init__(self):
            self.calls = []

        def transcribe(self, audio, language=None):
            self.calls.append(language)
            if language in (None, "", "auto"):
                return FakeASRResult(
                    "c'est du cobble Ã§a va",
                    language="fr",
                    confidence=0.21,
                    segments=[{"confidence": -0.84}],
                )
            if language == "en":
                return FakeASRResult(
                    "That's a good cobalt sound",
                    language="en",
                    confidence=0.99,
                    segments=[{"confidence": -0.20}],
                )
            if language == "fr":
                return FakeASRResult(
                    "salut comment ca va",
                    language="fr",
                    confidence=0.99,
                    segments=[{"confidence": -0.20}],
                )
            return FakeASRResult("corrected", language=language, confidence=0.99)

    class CapturingLLM:
        def __init__(self):
            self.messages = None
            self.calls = []

        async def chat_stream(self, messages):
            self.messages = messages
            self.calls.append(messages)
            yield "I understand."

    llm = CapturingLLM()
    asr = GuardedASR()
    tts = KokoroProvider()
    pipeline = ConversationPipeline(
        llm=llm,
        tts=tts,
        asr=asr,
        config=ConversationConfig(
            stream_tts=False,
            asr_language="auto",
            reply_language="en",
        ),
    )

    result = asyncio.run(pipeline.process_speech(b"\x00\x00" * 1600))

    assert result == "I understand."
    assert asr.calls == ["auto", "fr"]
    assert "salut comment ca va" in llm.calls[0][-1].content
    assert "reply ONLY in English" in llm.calls[0][-1].content
    assert pipeline._last_user_language_code == "fr"


def test_pipeline_rejects_low_confidence_asr_when_retry_is_not_better():
    class GuardedASR:
        def __init__(self):
            self.calls = []

        def transcribe(self, audio, language=None):
            self.calls.append(language)
            if language == "en":
                return FakeASRResult(
                    "That's a good cobalt sound",
                    language="en",
                    confidence=0.99,
                    segments=[{"confidence": -0.20}],
                )
            return FakeASRResult(
                "c'est du cobble ca va",
                language="fr",
                confidence=0.21,
                segments=[{"confidence": -0.84}],
            )

    class CapturingLLM:
        def __init__(self):
            self.calls = []

        async def chat_stream(self, messages):
            self.calls.append(messages)
            yield "I understand."

    llm = CapturingLLM()
    asr = GuardedASR()
    tts = KokoroProvider()
    pipeline = ConversationPipeline(
        llm=llm,
        tts=tts,
        asr=asr,
        config=ConversationConfig(
            stream_tts=False,
            asr_language="auto",
            reply_language="en",
        ),
    )

    result = asyncio.run(pipeline.process_speech(b"\x00\x00" * 1600))

    assert result is None
    assert asr.calls == ["auto", "fr"]
    assert llm.calls == []


def test_pipeline_accepts_confidence_less_retry_for_low_confidence_asr():
    class GuardedASR:
        def __init__(self):
            self.calls = []

        def transcribe(self, audio, language=None):
            self.calls.append(language)
            if language in (None, "", "auto"):
                return FakeASRResult(
                    "c'est du cobble ca va",
                    language="fr",
                    confidence=0.21,
                    segments=[{"confidence": -0.84}],
                )
            return FakeASRResult("salut comment ca va", language=language)

    class CapturingLLM:
        def __init__(self):
            self.calls = []

        async def chat_stream(self, messages):
            self.calls.append(messages)
            yield "I understand."

    llm = CapturingLLM()
    asr = GuardedASR()
    tts = KokoroProvider()
    pipeline = ConversationPipeline(
        llm=llm,
        tts=tts,
        asr=asr,
        config=ConversationConfig(stream_tts=False, asr_language="auto", reply_language="en"),
    )

    result = asyncio.run(pipeline.process_speech(b"\x00\x00" * 1600))

    assert result == "I understand."
    assert asr.calls == ["auto", "fr"]
    assert "salut comment ca va" in llm.calls[0][-1].content


def test_pipeline_remembers_language_inferred_from_asr_text_without_language():
    class CapturingLLM:
        def __init__(self):
            self.calls = []

        async def chat_stream(self, messages):
            self.calls.append(messages)
            yield "I understand."

    class LanguageLessASR:
        def transcribe(self, audio, language=None):
            return FakeASRResult("Parle-moi des etoiles.", language=None, confidence=0.99)

    llm = CapturingLLM()
    tts = KokoroProvider()
    pipeline = ConversationPipeline(
        llm=llm,
        tts=tts,
        asr=LanguageLessASR(),
        config=ConversationConfig(stream_tts=False, asr_language="auto", reply_language="en"),
    )

    result = asyncio.run(pipeline.process_speech(b"\x00\x00" * 1600))

    assert result == "I understand."
    assert pipeline._last_user_language_code == "fr"
    assert "The user is speaking French." in llm.calls[0][-1].content


def test_pipeline_can_force_english_reply_while_understanding_french_input():
    class CapturingLLM:
        def __init__(self):
            self.messages = None

        async def chat_stream(self, messages):
            self.messages = messages
            yield "Black holes are regions of spacetime with extremely strong gravity."

    class FrenchASR:
        def transcribe(self, audio, language=None):
            return FakeASRResult("Parle-moi des trous noirs.", language="fr", confidence=0.99)

    llm = CapturingLLM()
    tts = KokoroProvider()
    pipeline = ConversationPipeline(
        llm=llm,
        tts=tts,
        asr=FrenchASR(),
        config=ConversationConfig(stream_tts=False, asr_language="auto", reply_language="en"),
    )

    result = asyncio.run(pipeline.process_speech(b"\x00\x00" * 1600))

    assert result == "Black holes are regions of spacetime with extremely strong gravity."
    assert "The user is speaking French." in llm.messages[-1].content
    assert "reply ONLY in English" in llm.messages[-1].content
    assert tts.languages[-1] == "en"


def test_pipeline_rewrites_wrong_language_reply_before_tts():
    class ScriptedLLM:
        def __init__(self):
            self.calls = []

        async def chat_stream(self, messages):
            self.calls.append(messages)
            if len(self.calls) == 1:
                yield "Les etoiles sont magnifiques dans l'espace."
            else:
                yield "Stars are beautiful in space."

    class FrenchASR:
        def transcribe(self, audio, language=None):
            return FakeASRResult("Parle-moi des etoiles dans l'espace.", language="fr", confidence=0.99)

    llm = ScriptedLLM()
    tts = KokoroProvider()
    payloads = []
    chunks = []
    pipeline = ConversationPipeline(
        llm=llm,
        tts=tts,
        asr=FrenchASR(),
        config=ConversationConfig(stream_tts=True, asr_language="auto", reply_language="en"),
    )

    async def on_audio_ready(payload):
        payloads.append(payload)

    async def on_response_chunk(chunk):
        chunks.append(chunk)

    pipeline.on_audio_ready = on_audio_ready
    pipeline.on_response_chunk = on_response_chunk

    result = asyncio.run(pipeline.process_speech(b"\x00\x00" * 1600))

    assert result == "Stars are beautiful in space."
    assert len(llm.calls) == 2
    assert "Never answer in French" in llm.calls[0][-1].content
    assert "Rewrite this assistant reply in English" in llm.calls[1][-1].content
    assert tts.calls == ["Stars are beautiful in space."]
    assert [payload.text for payload in payloads] == ["Stars are beautiful in space."]
    assert chunks == ["Les etoiles sont magnifiques dans l'espace."]


def test_pipeline_keeps_fast_streaming_when_first_sentence_is_already_english():
    class ScriptedLLM:
        def __init__(self):
            self.calls = []

        async def chat_stream(self, messages):
            self.calls.append(messages)
            yield "Stars are beautiful in space. "
            yield "They feel endless."

    class FrenchASR:
        def transcribe(self, audio, language=None):
            return FakeASRResult("Parle-moi des etoiles.", language="fr", confidence=0.99)

    llm = ScriptedLLM()
    tts = KokoroProvider()
    payloads = []
    pipeline = ConversationPipeline(
        llm=llm,
        tts=tts,
        asr=FrenchASR(),
        config=ConversationConfig(stream_tts=True, asr_language="auto", reply_language="en"),
    )

    async def on_audio_ready(payload):
        payloads.append(payload)

    pipeline.on_audio_ready = on_audio_ready

    result = asyncio.run(pipeline.process_speech(b"\x00\x00" * 1600))

    assert result == "Stars are beautiful in space. They feel endless."
    assert len(llm.calls) == 1
    assert tts.calls == ["Stars are beautiful in space.", "They feel endless."]
    assert [payload.text for payload in payloads] == ["Stars are beautiful in space.", "They feel endless."]


def test_pipeline_sets_tts_language_from_reply_language_on_init():
    tts = KokoroProvider()

    ConversationPipeline(
        llm=FakeLLM(["Hello."]),
        tts=tts,
        asr=FakeASR(),
        config=ConversationConfig(stream_tts=False, reply_language="en"),
    )

    assert tts.languages == ["en"]
