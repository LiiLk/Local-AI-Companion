from pathlib import Path
import asyncio
import shutil
import threading
from types import SimpleNamespace
from unittest.mock import Mock
from uuid import uuid4

from src.assistant.pipeline_runtime import (
    build_pipeline_conversation_config,
    close_pipeline_runtime_services,
    create_pipeline_runtime,
    create_pipeline_rvc,
    create_pipeline_tts,
    preload_pipeline_asr,
    preload_pipeline_rvc,
    preload_pipeline_runtime_services,
    preload_pipeline_tts,
    resolve_initial_tts_language,
    resolve_pipeline_system_prompt,
    resolve_transcription_hint_prompt,
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


def test_voice_style_prompt_is_appended_when_configured():
    config = {
        "character": {"name": "March 7th", "system_prompt": "You are March 7th."},
        "pipeline": {"voice_style_prompt": "Speak naturally, 1 to 3 sentences."},
    }

    conversation_config = build_pipeline_conversation_config(config)

    assert conversation_config.system_prompt.startswith("You are March 7th.")
    assert "Speak naturally, 1 to 3 sentences." in conversation_config.system_prompt


def test_voice_style_prompt_is_omitted_when_empty():
    config = {
        "character": {"name": "March 7th", "system_prompt": "You are March 7th."},
        "pipeline": {"voice_style_prompt": ""},
    }

    conversation_config = build_pipeline_conversation_config(config)

    assert conversation_config.system_prompt == "You are March 7th."


def test_build_pipeline_conversation_config_reads_adaptive_reasoning():
    config = {
        "llm": {
            "adaptive_reasoning": {
                "enabled": True,
                "base_effort": "none",
                "escalate_effort": "medium",
            }
        }
    }

    conversation_config = build_pipeline_conversation_config(config)

    assert conversation_config.adaptive_reasoning is not None
    assert conversation_config.adaptive_reasoning.enabled is True
    assert conversation_config.adaptive_reasoning.base_effort == "none"


def test_build_pipeline_conversation_config_without_adaptive_reasoning():
    conversation_config = build_pipeline_conversation_config({})

    assert conversation_config.adaptive_reasoning is None


def test_transcription_hint_prompt_is_not_in_permanent_system_prompt():
    config = {
        "character": {"name": "March 7th", "system_prompt": "You are March 7th."},
        "pipeline": {
            "transcription_hint_prompt": "Transcriptions may contain recognition errors.",
        },
    }

    conversation_config = build_pipeline_conversation_config(config)

    assert conversation_config.system_prompt == "You are March 7th."
    assert "Transcriptions may contain recognition errors." not in conversation_config.system_prompt


def test_transcription_hint_prompt_is_exposed_for_speech_turns_only():
    config = {
        "character": {"name": "March 7th", "system_prompt": "You are March 7th."},
        "pipeline": {
            "transcription_hint_prompt": "Transcriptions may contain recognition errors.",
        },
    }

    assert (
        resolve_transcription_hint_prompt(config)
        == "Transcriptions may contain recognition errors."
    )

    conversation_config = build_pipeline_conversation_config(config)
    assert (
        conversation_config.transcription_hint_prompt
        == "Transcriptions may contain recognition errors."
    )


def test_transcription_hint_prompt_is_omitted_for_text_only_system_prompt():
    config = {
        "character": {"name": "March 7th", "system_prompt": "You are March 7th."},
        "pipeline": {"voice_style_prompt": "Speak naturally."},
    }

    assert resolve_pipeline_system_prompt(config) == "You are March 7th.\n\nSpeak naturally."
    assert resolve_transcription_hint_prompt(config) == ""


def test_transcription_hint_prompt_is_omitted_when_empty():
    config = {
        "character": {"name": "March 7th", "system_prompt": "You are March 7th."},
        "pipeline": {"transcription_hint_prompt": ""},
    }

    conversation_config = build_pipeline_conversation_config(config)

    assert conversation_config.system_prompt == "You are March 7th."


def test_transcription_hint_prompt_is_omitted_in_omni_mode():
    config = {
        "mode": "omni",
        "character": {"name": "March 7th", "system_prompt": "You are March 7th."},
        "pipeline": {
            "transcription_hint_prompt": "Transcriptions may contain recognition errors.",
        },
    }

    conversation_config = build_pipeline_conversation_config(config)

    assert conversation_config.system_prompt == "You are March 7th."


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


@pytest.mark.parametrize("entrypoint", ["runtime", "services"])
@pytest.mark.parametrize("rvc_enabled", [True, False])
def test_preload_spawns_rvc_before_other_services_and_waits_after_tts(
    entrypoint, rvc_enabled,
):
    events = []
    llm = SimpleNamespace(preload=lambda: events.append("llm"))
    asr = SimpleNamespace(preload=lambda: events.append("asr"))
    tts = SimpleNamespace(
        preload=lambda: events.append("tts"),
        warmup=lambda: events.append("tts_warmup"),
    )
    rvc = SimpleNamespace(
        spawn_worker=lambda: events.append("rvc_spawn"),
        preload=lambda: events.append("rvc_ready"),
        warmup=lambda: events.append("rvc_warmup"),
    ) if rvc_enabled else None
    if entrypoint == "runtime":
        runtime = create_pipeline_runtime({
            "tts": {"warmup_on_start": True, "rvc": {"enabled": rvc_enabled}},
        })
        runtime.llm, runtime.asr, runtime.tts, runtime.rvc = llm, asr, tts, rvc
        result = runtime.preload_all()
        assert runtime.collect_degraded_reason() is None
    else:
        result = preload_pipeline_runtime_services(
            llm=llm, asr=asr, tts=tts, rvc=rvc, tts_warmup_on_start=True,
        )

    assert result == (tts, rvc)
    assert events == (
        ["rvc_spawn", "llm", "asr", "tts", "tts_warmup", "rvc_ready", "rvc_warmup"]
        if rvc_enabled else ["llm", "asr", "tts", "tts_warmup"]
    )


@pytest.mark.parametrize("entrypoint", ["runtime", "services"])
def test_preload_non_worker_rvc_keeps_existing_order(entrypoint):
    events = []
    llm = SimpleNamespace(preload=lambda: events.append("llm"))
    asr = SimpleNamespace(preload=lambda: events.append("asr"))
    tts = SimpleNamespace(preload=lambda: events.append("tts"))
    rvc = SimpleNamespace(
        preload=lambda: events.append("rvc_preload"),
        warmup=lambda: events.append("rvc_warmup"),
    )
    if entrypoint == "runtime":
        runtime = create_pipeline_runtime({"tts": {"rvc": {"enabled": True}}})
        runtime.llm, runtime.asr, runtime.tts, runtime.rvc = llm, asr, tts, rvc
        result = runtime.preload_all()
    else:
        result = preload_pipeline_runtime_services(llm=llm, asr=asr, tts=tts, rvc=rvc)
    assert result == (tts, rvc)
    assert events == ["llm", "asr", "tts", "rvc_preload", "rvc_warmup"]


@pytest.mark.parametrize("entrypoint", ["runtime", "services"])
@pytest.mark.parametrize("failed_service", ["llm", "asr", "tts"])
def test_preload_closes_early_worker_when_required_service_fails(entrypoint, failed_service):
    events = []

    def preload(name):
        events.append(name)
        if name == failed_service:
            raise RuntimeError(f"{name} unavailable")

    llm = SimpleNamespace(preload=lambda: preload("llm"))
    asr = SimpleNamespace(preload=lambda: preload("asr"))
    tts = SimpleNamespace(preload=lambda: preload("tts"))
    rvc = SimpleNamespace(
        spawn_worker=lambda: events.append("spawn"),
        preload=lambda: pytest.fail("RVC must not preload after a required service fails"),
        close=lambda: events.append("close"),
    )
    with pytest.raises(RuntimeError, match=f"{failed_service} unavailable"):
        if entrypoint == "runtime":
            runtime = create_pipeline_runtime({"tts": {"rvc": {"enabled": True}}})
            runtime.llm, runtime.asr, runtime.tts, runtime.rvc = llm, asr, tts, rvc
            runtime.preload_all()
        else:
            preload_pipeline_runtime_services(llm=llm, asr=asr, tts=tts, rvc=rvc)
    assert events[0] == "spawn"
    assert events[-2:] == [failed_service, "close"]


@pytest.mark.parametrize("stage", ["spawn_worker", "preload", "warmup"])
@pytest.mark.parametrize("error", [
    TimeoutError("startup timed out after 90s"), RuntimeError("worker crashed"),
])
def test_runtime_rvc_failure_is_degraded_and_stays_disabled(monkeypatch, stage, error):
    events = []
    runtime = create_pipeline_runtime({"tts": {"rvc": {"enabled": True}}})
    runtime.llm = SimpleNamespace(preload=lambda: events.append("llm"))
    runtime.asr = SimpleNamespace(preload=lambda: events.append("asr"))
    runtime.tts = SimpleNamespace(preload=lambda: events.append("tts"))
    rvc = SimpleNamespace(
        spawn_worker=lambda: events.append("spawn_worker"),
        preload=lambda: events.append("preload"),
        warmup=lambda: events.append("warmup"),
        close=lambda: events.append("close"),
    )

    def fail():
        events.append(stage)
        raise error

    setattr(rvc, stage, fail)

    def create_rvc(_config):
        events.append("create")
        return rvc, "test voice"

    monkeypatch.setattr("src.assistant.pipeline_runtime.create_pipeline_rvc", create_rvc)
    runtime.preload_all()
    status = runtime.resolve_backend_status()

    assert runtime.rvc is None
    assert status.state == "degraded"
    assert status.degraded_reason == f"RVC unavailable: voice conversion disabled ({error})"
    assert runtime.is_ready() is True  # Base TTS remains usable in degraded mode.
    assert events.count("close") == 1
    assert [event for event in events if event in {"llm", "asr", "tts"}] == ["llm", "asr", "tts"]
    assert runtime.ensure_rvc() is None
    assert runtime.preload_rvc() is None
    assert runtime.spawn_rvc_worker() is None
    assert events.count("create") == 1


@pytest.mark.parametrize("error", [None, RuntimeError("worker unavailable")])
def test_runtime_rvc_factory_failure_is_degraded_without_retry(monkeypatch, error):
    runtime = create_pipeline_runtime({"tts": {"rvc": {"enabled": True}}})
    attempts = []

    def create_rvc(_config):
        attempts.append("create")
        if error:
            raise error
        return None, None

    monkeypatch.setattr("src.assistant.pipeline_runtime.create_pipeline_rvc", create_rvc)
    assert runtime.ensure_rvc() is None
    assert runtime.resolve_backend_status().state == "degraded"
    assert "RVC unavailable: voice conversion disabled" in runtime.collect_degraded_reason()
    assert runtime.ensure_rvc() is None
    assert attempts == ["create"]


def test_concurrent_runtime_ensure_rvc_creates_one_converter(monkeypatch):
    runtime = create_pipeline_runtime({"tts": {"rvc": {"enabled": True}}})
    factory_entered = threading.Event()
    release_factory = threading.Event()
    contender_attempted = threading.Event()
    created = []
    results = []
    errors = []
    creation_lock = getattr(runtime, "_rvc_creation_lock", None)
    if creation_lock is not None:
        class ObservedLock:
            def __enter__(self):
                if threading.current_thread().name == "RuntimeContender":
                    contender_attempted.set()
                creation_lock.acquire()

            def __exit__(self, *args):
                creation_lock.release()

        monkeypatch.setattr(runtime, "_rvc_creation_lock", ObservedLock())

    def factory(_config):
        converter = Mock()
        created.append(converter)
        if threading.current_thread().name == "RuntimeContender":
            contender_attempted.set()
        factory_entered.set()
        assert release_factory.wait(5)
        return converter, "voice"

    def ensure():
        try:
            results.append(runtime.ensure_rvc())
        except Exception as exc:
            errors.append(exc)

    monkeypatch.setattr("src.assistant.pipeline_runtime.create_pipeline_rvc", factory)
    threads = [threading.Thread(target=ensure, daemon=True, name=name)
               for name in ("RuntimeCreator", "RuntimeContender")]
    threads[0].start()
    try:
        assert factory_entered.wait(5)
        threads[1].start()
        assert contender_attempted.wait(5)
    finally:
        release_factory.set()
        for thread in threads:
            if thread.ident is not None:
                thread.join(timeout=5)
        asyncio.run(runtime.close())

    assert all(not thread.is_alive() for thread in threads)
    assert errors == []
    assert len(created) == 1
    assert results == [created[0], created[0]]


@pytest.mark.parametrize("operation", ["ensure_rvc", "spawn_rvc_worker", "preload_rvc"])
def test_runtime_close_during_rvc_factory_does_not_publish_or_spawn(monkeypatch, operation):
    runtime = create_pipeline_runtime({"tts": {"rvc": {"enabled": True}}})
    factory_entered = threading.Event()
    release_factory = threading.Event()
    closed = threading.Event()
    converter = Mock()
    results = []
    errors = []

    def factory(_config):
        factory_entered.set()
        assert release_factory.wait(5)
        return converter, "voice"

    def run_operation():
        try:
            results.append(getattr(runtime, operation)())
        except Exception as exc:
            errors.append(exc)

    def close():
        try:
            asyncio.run(runtime.close())
        except Exception as exc:
            errors.append(exc)
        finally:
            closed.set()

    monkeypatch.setattr("src.assistant.pipeline_runtime.create_pipeline_rvc", factory)
    creator = threading.Thread(target=run_operation, daemon=True)
    closer = threading.Thread(target=close, daemon=True)
    creator.start()
    try:
        assert factory_entered.wait(5)
        closer.start()
        assert closed.wait(5), "Runtime close must not wait for the RVC factory"
        assert runtime.ensure_rvc() is None
    finally:
        release_factory.set()
        creator.join(timeout=5)
        if closer.ident is not None:
            closer.join(timeout=5)

    assert not creator.is_alive() and not closer.is_alive()
    assert errors == []
    assert results == [None]
    assert runtime.rvc is None
    assert runtime.rvc_summary is None
    converter.close.assert_called_once_with()
    converter.spawn_worker.assert_not_called()
    converter.preload.assert_not_called()
    converter.warmup.assert_not_called()


def test_runtime_ensure_rvc_after_close_never_calls_factory(monkeypatch):
    runtime = create_pipeline_runtime({"tts": {"rvc": {"enabled": True}}})
    factory = Mock(return_value=(Mock(), "voice"))
    monkeypatch.setattr("src.assistant.pipeline_runtime.create_pipeline_rvc", factory)
    asyncio.run(runtime.close())
    assert runtime.ensure_rvc() is None
    factory.assert_not_called()


def test_runtime_discard_rvc_allows_a_new_converter(monkeypatch):
    runtime = create_pipeline_runtime({"tts": {"rvc": {"enabled": True}}})
    first, second = Mock(), Mock()
    factory = Mock(side_effect=[(first, "first"), (second, "second")])
    monkeypatch.setattr("src.assistant.pipeline_runtime.create_pipeline_rvc", factory)
    assert runtime.ensure_rvc() is first

    runtime.discard_rvc()

    first.close.assert_called_once_with()
    assert runtime.rvc is None
    assert runtime.rvc_summary is None
    assert runtime.collect_degraded_reason() is None
    assert runtime.ensure_rvc() is second
    second.close.assert_not_called()
    assert factory.call_count == 2


@pytest.mark.parametrize("operation,stage", [
    ("spawn_rvc_worker", "spawn_worker"), ("preload_rvc", "preload"),
    ("preload_rvc", "warmup"),
])
@pytest.mark.parametrize("cleanup", ["close", "discard"])
@pytest.mark.parametrize("failure", [False, True])
def test_runtime_inflight_rvc_cannot_republish_after_removal(
    monkeypatch, operation, stage, cleanup, failure,
):
    runtime = create_pipeline_runtime({"tts": {"rvc": {"enabled": True}}})
    original, replacement = Mock(), Mock()
    runtime.rvc = original
    runtime.rvc_summary = "original"
    entered = threading.Event()
    release = threading.Event()
    cleanup_done = threading.Event()
    results = []
    errors = []

    def blocked_operation():
        entered.set()
        assert release.wait(5)
        if failure:
            raise RuntimeError("removed converter failed")

    setattr(original, stage, blocked_operation)

    def run_operation():
        try:
            results.append(getattr(runtime, operation)())
        except Exception as exc:
            errors.append(exc)

    def remove():
        try:
            if cleanup == "close":
                asyncio.run(runtime.close())
            else:
                runtime.discard_rvc()
        except Exception as exc:
            errors.append(exc)
        finally:
            cleanup_done.set()

    monkeypatch.setattr(
        "src.assistant.pipeline_runtime.create_pipeline_rvc", lambda _: (replacement, "replacement"),
    )
    worker = threading.Thread(target=run_operation, daemon=True)
    remover = threading.Thread(target=remove, daemon=True)
    worker.start()
    try:
        assert entered.wait(5)
        remover.start()
        assert cleanup_done.wait(5), "RVC lifecycle lock must not cover spawn/handshake/warmup"
        assert runtime.rvc is None
        if cleanup == "discard":
            assert runtime.ensure_rvc() is replacement
    finally:
        release.set()
        worker.join(timeout=5)
        if remover.ident is not None:
            remover.join(timeout=5)

    assert not worker.is_alive() and not remover.is_alive()
    assert errors == []
    assert results == [None]
    assert runtime.rvc is (replacement if cleanup == "discard" else None)
    assert runtime.collect_degraded_reason() is None
    replacement.close.assert_not_called()


def test_runtime_preload_retry_recreates_early_rvc_after_required_failure(monkeypatch):
    runtime = create_pipeline_runtime({"tts": {"rvc": {"enabled": True}}})
    first, second = Mock(), Mock()
    factory = Mock(side_effect=[(first, "first"), (second, "second")])
    monkeypatch.setattr("src.assistant.pipeline_runtime.create_pipeline_rvc", factory)
    runtime.llm = SimpleNamespace(preload=Mock(side_effect=[RuntimeError("LLM failed"), None]))
    runtime.asr = SimpleNamespace(preload=lambda: None)
    runtime.tts = SimpleNamespace(preload=lambda: None)
    with pytest.raises(RuntimeError, match="LLM failed"):
        runtime.preload_all()

    assert runtime.rvc is None
    assert runtime.is_ready() is False
    first.close.assert_called_once_with()
    runtime.preload_all()
    assert runtime.rvc is second
    assert runtime.is_ready() is True
    second.spawn_worker.assert_called_once_with()
    second.preload.assert_called_once_with()
    assert factory.call_count == 2


@pytest.mark.asyncio
async def test_runtime_close_closes_rvc_before_waiting_for_llm():
    runtime = create_pipeline_runtime({"tts": {"rvc": {"enabled": True}}})
    converter = Mock()
    runtime.rvc = converter
    llm_closing = asyncio.Event()
    release_llm = asyncio.Event()

    async def close_llm():
        llm_closing.set()
        await release_llm.wait()

    runtime.llm = SimpleNamespace(close=close_llm)
    task = asyncio.create_task(runtime.close())
    try:
        await asyncio.wait_for(llm_closing.wait(), timeout=5)
        converter.close.assert_called_once_with()
        assert runtime.rvc is None
        assert runtime.ensure_rvc() is None
        assert runtime.is_ready() is False
    finally:
        release_llm.set()
        await task


@pytest.mark.parametrize("configured_timeout", [None, 120])
def test_rvc_factory_passes_startup_timeout(monkeypatch, configured_timeout):
    rvc_config = {"model_path": "voice.pth", "request_timeout_sec": 7}
    if configured_timeout is not None:
        rvc_config["startup_timeout_sec"] = configured_timeout
    monkeypatch.setattr(
        "src.assistant.pipeline_runtime.build_rvc_runtime_config", lambda _: rvc_config,
    )
    kwargs_seen = {}

    class FakeConverter:
        @staticmethod
        def is_available(**_kwargs):
            return True

        def __init__(self, **kwargs):
            kwargs_seen.update(kwargs)

    monkeypatch.setattr("src.tts.rvc_provider.RVCConverter", FakeConverter)
    converter, _summary = create_pipeline_rvc({})
    assert converter is not None
    assert kwargs_seen["startup_timeout_sec"] == (configured_timeout or 90)
    assert kwargs_seen["request_timeout_sec"] == 7


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
