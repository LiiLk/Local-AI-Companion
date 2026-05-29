from pathlib import Path
import shutil
from uuid import uuid4

from src.assistant.pipeline_runtime import (
    build_pipeline_conversation_config,
    close_pipeline_runtime_services,
    create_pipeline_runtime,
    create_pipeline_tts,
    preload_pipeline_asr,
    preload_pipeline_rvc,
    preload_pipeline_tts,
    resolve_initial_tts_language,
)
import pytest


def _test_dir(name: str) -> Path:
    path = Path.cwd() / ".codex_test_artifacts" / f"{name}-{uuid4().hex}"
    path.mkdir(parents=True, exist_ok=False)
    return path


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
            "max_queue_size": 4,
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
    assert conversation_config.tts_max_queue_size == 4
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


def test_preload_pipeline_rvc_fails_open_when_warmup_errors():
    class FakeRVC:
        def __init__(self):
            self.calls = []

        def preload(self):
            self.calls.append("preload")

        def warmup(self):
            self.calls.append("warmup")
            raise TimeoutError("worker hung")

        def close(self):
            self.calls.append("close")

    rvc = FakeRVC()

    assert preload_pipeline_rvc(rvc, warmup=True) is None
    assert rvc.calls == ["preload", "warmup", "close"]


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


def test_pipeline_runtime_uses_reply_language_for_initial_tts(monkeypatch):
    config = {
        "pipeline": {"reply_language": "en"},
        "tts": {"provider": "kokoro"},
    }
    runtime = create_pipeline_runtime(config, initial_tts_language="fr")

    marker = object()

    def fake_create_pipeline_tts(runtime_config, *, initial_language=None):
        assert runtime_config is config
        assert initial_language == "en"
        return marker, "Kokoro"

    monkeypatch.setattr("src.assistant.pipeline_runtime.create_pipeline_tts", fake_create_pipeline_tts)

    assert runtime.ensure_tts() is marker
    assert runtime.tts_summary == "Kokoro"


def test_pipeline_runtime_creates_memory_from_config():
    test_dir = _test_dir("runtime-memory")
    try:
        config = {
            "memory": {
                "enabled": True,
                "history_path": str(test_dir / "conversation.jsonl"),
                "summary_path": str(test_dir / "summary.txt"),
                "max_recent_turns": 2,
            }
        }
        runtime = create_pipeline_runtime(config)

        memory = runtime.ensure_memory()

        assert memory is not None
        assert memory.config.history_path == test_dir / "conversation.jsonl"
        assert memory.config.max_recent_turns == 2
    finally:
        shutil.rmtree(test_dir, ignore_errors=True)


def test_pipeline_runtime_reports_ready_when_all_services_exist():
    runtime = create_pipeline_runtime({"llm": {"provider": "openrouter"}, "tts": {"rvc": {"enabled": True}}})
    runtime.llm = object()
    runtime.tts = object()
    runtime.asr = object()
    runtime.rvc = object()

    assert runtime.is_ready() is True


def test_pipeline_runtime_resolves_degraded_backend_status():
    runtime = create_pipeline_runtime({})
    runtime.tts = type("FakeTTS", (), {"degraded_reason": "fallback active"})()

    status = runtime.resolve_backend_status(
        requested_state="ready",
        runtime_error=None,
        extra_degraded_reason="slow mode",
    )

    assert status.state == "degraded"
    assert status.degraded_reason == "fallback active | slow mode"
    assert status.runtime_error is None


def test_chatterbox_tts_receives_configured_model_revision():
    config = {
        "tts": {
            "provider": "chatterbox",
            "chatterbox": {
                "model_revision": "abc123",
            },
        },
    }

    tts, _summary = create_pipeline_tts(config)

    assert tts.model_revision == "abc123"
    tts.cleanup()


def test_routed_tts_chatterbox_receives_configured_model_revision():
    config = {
        "tts": {
            "provider": "qwen3",
            "chatterbox": {
                "model_revision": "abc123",
            },
            "qwen3": {
                "backend": "worker",
                "python_path": "/definitely/missing/python",
            },
        },
    }

    tts, _summary = create_pipeline_tts(config)

    assert tts._providers["chatterbox"].model_revision == "abc123"
    tts.cleanup()
