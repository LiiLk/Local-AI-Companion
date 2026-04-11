import asyncio
import wave
from pathlib import Path

from src.assistant.conversation_pipeline import ConversationConfig, ConversationPipeline
from src.llm.base import Message


class FakeLLM:
    async def chat_stream(self, messages):
        yield "Bonjour."


class FakeASRResult:
    def __init__(self, text: str):
        self.text = text


class FakeASR:
    def transcribe(self, audio, language=None):
        return FakeASRResult("Salut")


class FakeTTS:
    async def synthesize(self, text, output_path):
        with wave.open(str(output_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(24000)
            wf.writeframes(b"\x00\x00" * 2400)


class FakeRVC:
    def __init__(self):
        self.calls = []

    def convert_file(self, src, dst):
        self.calls.append((Path(src), Path(dst)))
        with wave.open(str(dst), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(44100)
            wf.writeframes(b"\x01\x00" * 4410)
        return Path(dst)


def test_conversation_pipeline_applies_optional_rvc():
    payloads = []
    rvc = FakeRVC()
    pipeline = ConversationPipeline(
        llm=FakeLLM(),
        tts=FakeTTS(),
        asr=FakeASR(),
        config=ConversationConfig(stream_tts=False),
        rvc=rvc,
    )

    async def on_audio_ready(payload):
        payloads.append(payload)

    pipeline.on_audio_ready = on_audio_ready

    result = asyncio.run(pipeline.process_speech(b"\x00\x00" * 1600))

    assert result == "Bonjour."
    assert len(rvc.calls) == 1
    assert len(payloads) == 1
    assert payloads[0].sample_rate == 44100


def test_process_text_can_restart_after_cancel_active_run():
    pipeline = ConversationPipeline(
        llm=FakeLLM(),
        tts=FakeTTS(),
        asr=FakeASR(),
        config=ConversationConfig(stream_tts=False),
    )

    async def cancel_on_start():
        pipeline.cancel_active_run("test restart")

    pipeline.on_response_start = cancel_on_start

    try:
        asyncio.run(pipeline.process_text("Premier tour"))
        raised = False
    except asyncio.CancelledError:
        raised = True

    assert raised is True
    assert pipeline.is_processing is False

    pipeline.on_response_start = None
    result = asyncio.run(pipeline.process_text("Deuxieme tour"))

    assert result == "Bonjour."
    assert pipeline.is_processing is False
