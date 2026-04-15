from src.assistant.pipeline_runtime import (
    build_pipeline_conversation_config,
    close_pipeline_runtime_services,
    preload_pipeline_asr,
    preload_pipeline_tts,
    resolve_initial_tts_language,
)
import pytest


def test_resolve_initial_tts_language_prefers_reply_language():
    config = {
        "pipeline": {"reply_language": "en"},
    }

    assert resolve_initial_tts_language(config, "fr") == "en"


def test_build_pipeline_conversation_config_uses_pipeline_defaults():
    config = {
        "character": {
            "name": "March 7th",
            "system_prompt": "You are March 7th.",
        },
        "tts": {
            "stream_tts": True,
            "auto_detect_language": False,
        },
        "asr": {
            "language": "auto",
        },
        "pipeline": {
            "reply_language": "en",
        },
    }

    conversation_config = build_pipeline_conversation_config(config)

    assert conversation_config.character_name == "March 7th"
    assert conversation_config.system_prompt == "You are March 7th."
    assert conversation_config.stream_tts is True
    assert conversation_config.auto_detect_language is False
    assert conversation_config.asr_language == "auto"
    assert conversation_config.reply_language == "en"


def test_preload_pipeline_asr_uses_get_model_fallback():
    class FakeASR:
        def __init__(self):
            self.calls = []

        def _get_model(self):
            self.calls.append("_get_model")

    asr = FakeASR()

    assert preload_pipeline_asr(asr) is asr
    assert asr.calls == ["_get_model"]


def test_preload_pipeline_tts_uses_load_model_fallback_and_warmup():
    class FakeTTS:
        def __init__(self):
            self.calls = []

        def _load_model(self):
            self.calls.append("_load_model")

        def warmup(self):
            self.calls.append("warmup")

    tts = FakeTTS()

    assert preload_pipeline_tts(tts, warmup=True) is tts
    assert tts.calls == ["_load_model", "warmup"]


def test_preload_pipeline_tts_can_replace_failed_provider():
    class FailingTTS:
        def preload(self):
            raise RuntimeError("boom")

    class FallbackTTS:
        def __init__(self):
            self.calls = []

        def preload(self):
            self.calls.append("preload")

        def warmup(self):
            self.calls.append("warmup")

    primary = FailingTTS()
    fallback = FallbackTTS()

    def on_load_error(tts, exc):
        assert tts is primary
        assert str(exc) == "boom"
        return fallback

    assert preload_pipeline_tts(primary, warmup=True, on_load_error=on_load_error) is fallback
    assert fallback.calls == ["preload", "warmup"]


@pytest.mark.asyncio
async def test_close_pipeline_runtime_services_closes_all_backends():
    class FakeTTS:
        def __init__(self):
            self.cleaned = False

        def cleanup(self):
            self.cleaned = True

    class FakeASR:
        def __init__(self):
            self.cleaned = False

        def cleanup(self):
            self.cleaned = True

    class FakeLLM:
        def __init__(self):
            self.closed = False

        async def close(self):
            self.closed = True

    class FakeRVC:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    tts = FakeTTS()
    asr = FakeASR()
    llm = FakeLLM()
    rvc = FakeRVC()

    await close_pipeline_runtime_services(llm=llm, tts=tts, asr=asr, rvc=rvc)

    assert tts.cleaned is True
    assert asr.cleaned is True
    assert llm.closed is True
    assert rvc.closed is True
