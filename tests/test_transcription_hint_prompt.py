"""The transcription hint must only reach the LLM on spoken turns.

Typing "GI" on purpose must not be reinterpreted as a speech transcription
error, so the hint is injected per-turn for speech input only.
"""

import asyncio
from types import SimpleNamespace

import pytest

from src.assistant.conversation_pipeline import ConversationConfig, ConversationPipeline
from src.asr.base import ASRResult
from src.llm.base import Message
from src.server import websocket as websocket_module
from src.server.websocket import WebSocketManager

HINT = "SPEECH-ONLY-TRANSCRIPTION-HINT"


class _CaptureLLM:
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


def _speech_asr():
    class SpeechASR:
        def transcribe(self, audio, language=None):
            return ASRResult(
                text="hello there",
                language="en",
                confidence=0.95,
                duration=1.0,
                segments=[
                    {"text": "hello there", "start": 0.0, "end": 0.9, "confidence": -0.1}
                ],
            )

    return SpeechASR()


def _pipeline() -> ConversationPipeline:
    return ConversationPipeline(
        llm=_CaptureLLM(),
        tts=_TTS(),
        asr=_speech_asr(),
        config=ConversationConfig(
            stream_tts=False,
            asr_language="en",
            reply_language="en",
            transcription_hint_prompt=HINT,
        ),
    )


def test_speech_turn_injects_transcription_hint_into_llm_messages():
    pipeline = _pipeline()

    asyncio.run(pipeline.process_speech(_audio_bytes(1.0)))

    assert pipeline.llm.calls
    last_user = pipeline.llm.calls[0][-1]
    assert last_user.role == "user"
    assert HINT in last_user.content


def test_text_turn_does_not_inject_transcription_hint():
    pipeline = _pipeline()

    asyncio.run(pipeline.process_text("GI"))

    assert pipeline.llm.calls
    last_user = pipeline.llm.calls[0][-1]
    assert last_user.role == "user"
    assert HINT not in last_user.content


def test_websocket_apply_transcription_hint_prefixes_last_user_message():
    manager = WebSocketManager()
    state = SimpleNamespace(config={"pipeline": {"transcription_hint_prompt": HINT}})
    messages = [
        Message(role="system", content="sys"),
        Message(role="user", content="GI"),
    ]

    result = manager._apply_transcription_hint(state, messages)

    assert result[-1].role == "user"
    assert result[-1].content == f"(System: {HINT})\n\nGI"


def test_websocket_apply_transcription_hint_noop_without_config():
    manager = WebSocketManager()
    state = SimpleNamespace(config={})
    messages = [Message(role="user", content="GI")]

    result = manager._apply_transcription_hint(state, messages)

    assert result[-1].content == "GI"


async def _noop_send_json(*_args, **_kwargs):
    return None


class _FakeTTSManager:
    def __init__(self, **_kwargs):
        pass

    async def start(self):
        pass

    async def submit(self, text, expression=None):
        pass

    async def finish(self):
        pass

    async def cancel(self):
        pass


def _websocket_state(captured):
    class FakeLLM:
        async def chat_stream(self, messages):
            captured.append(messages)
            yield "ok"

    return SimpleNamespace(
        mode="pipeline",
        config={"pipeline": {"transcription_hint_prompt": HINT}},
        messages=[Message(role="system", content="sys")],
        current_language="en",
        current_expression="neutral",
        emotion_detector=None,
        pipeline_runtime=None,
        tts=None,
        get_llm=lambda: FakeLLM(),
        get_tts=lambda: SimpleNamespace(),
        get_rvc=lambda: None,
        memory_store=None,
    )


@pytest.mark.asyncio
async def test_websocket_speech_turn_injects_hint(monkeypatch):
    monkeypatch.setattr(websocket_module, "TTSTaskManager", _FakeTTSManager)
    manager = WebSocketManager()
    client_id = "client-speech-hint"
    captured: list = []
    manager.states[client_id] = _websocket_state(captured)
    manager.send_json = _noop_send_json  # type: ignore[method-assign]

    await manager._handle_text_message_turn(client_id, "hello", speech_origin=True)

    assert captured
    assert HINT in captured[0][-1].content


@pytest.mark.asyncio
async def test_websocket_typed_turn_does_not_inject_hint(monkeypatch):
    monkeypatch.setattr(websocket_module, "TTSTaskManager", _FakeTTSManager)
    manager = WebSocketManager()
    client_id = "client-text-hint"
    captured: list = []
    manager.states[client_id] = _websocket_state(captured)
    manager.send_json = _noop_send_json  # type: ignore[method-assign]

    await manager._handle_text_message_turn(client_id, "GI")

    assert captured
    assert HINT not in captured[0][-1].content