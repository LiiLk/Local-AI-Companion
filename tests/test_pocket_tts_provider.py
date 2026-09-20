import asyncio
import builtins
import copy
import importlib
import io
import json
import os
from pathlib import Path
import socket
import subprocess
import time
import sys
import threading
from types import SimpleNamespace
from unittest.mock import Mock
import wave

import numpy as np
import pytest

from src.tts import pocket_tts_provider as pocket


class FakeTensor:
    def __init__(self, samples):
        self.samples = samples

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.samples


@pytest.fixture
def backend(monkeypatch):
    model = Mock()
    model.sample_rate = 16000
    model.get_state_for_audio_prompt.side_effect = lambda source: {"source": source, "steps": [0]}

    def generate(model_state, text_to_generate, max_tokens=50, frames_after_eos=None, copy_state=True):
        state = copy.deepcopy(model_state) if copy_state else model_state
        state["steps"][0] += 1
        return FakeTensor(np.array([-2, -0.5, 0, 0.5, 2], dtype=np.float32))

    model.generate_audio.side_effect = generate
    model_type = Mock()
    model_type.load_model.return_value = model
    monkeypatch.setitem(sys.modules, "pocket_tts", SimpleNamespace(TTSModel=model_type))
    monkeypatch.setattr(pocket, "version", lambda name: "3.1.0")
    return model_type, model


def test_import_constructor_and_controls_are_lightweight(monkeypatch):
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.split(".")[0] in ("torch", "pocket_tts", "numpy"):
            raise AssertionError(f"Unexpected heavy import: {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    monkeypatch.setattr(socket, "create_connection", Mock(side_effect=AssertionError("Network access")))
    monkeypatch.setattr(socket.socket, "connect", Mock(side_effect=AssertionError("Network access")))
    importlib.reload(pocket)
    provider = pocket.PocketTTSProvider()
    assert provider.voice == "alba"
    assert provider.language == "en"
    assert provider.device == "cpu"
    provider.set_language("en-US")
    provider.set_language(None)
    provider.set_voice("anna")
    provider.cleanup()
    assert provider._model is None


@pytest.mark.asyncio
async def test_wav_encoding_sample_rate_duration_and_cache(backend):
    model_type, model = backend
    provider = pocket.PocketTTSProvider()
    first = await provider.synthesize("Hello.")
    second = await provider.synthesize("Again.")
    model_type.load_model.assert_called_once_with(language="english")
    model.to.assert_called_once_with("cpu")
    model.eval.assert_called_once_with()
    model.get_state_for_audio_prompt.assert_called_once_with("alba")
    assert model.generate_audio.call_args.kwargs == {"copy_state": True}
    assert provider._voice_states["alba"]["steps"] == [0]
    assert first.audio_path is None
    assert first.duration == 5 / 16000
    assert first.metadata["sample_rate"] == 16000
    assert first.metadata["device"] == "cpu"
    assert first.metadata["synth_ms"] >= 0
    assert first.audio_data == second.audio_data
    with wave.open(io.BytesIO(first.audio_data), "rb") as wav:
        assert wav.getnchannels() == 1
        assert wav.getsampwidth() == 2
        assert wav.getframerate() == 16000
        assert wav.getnframes() == 5
        assert np.frombuffer(wav.readframes(5), dtype="<i2").tolist() == [
            -32768, -16384, 0, 16384, 32767,
        ]


@pytest.mark.asyncio
async def test_output_path_and_stream_fallback(backend, tmp_path):
    provider = pocket.PocketTTSProvider()
    path = tmp_path / "speech.wav"
    result = await provider.synthesize("Hello.", path)
    assert result.audio_path == path
    assert result.audio_data is None
    chunks = [chunk async for chunk in provider.synthesize_stream("Hello.")]
    assert chunks == [path.read_bytes()]


def test_hooks_voice_cache_and_cleanup(backend):
    model_type, model = backend
    provider = pocket.PocketTTSProvider()
    provider.preload()
    provider.warmup()
    provider.set_voice("anna")
    provider.preload()
    provider.set_voice("alba")
    provider.preload()
    assert model.get_state_for_audio_prompt.call_count == 2
    model.generate_audio.assert_called_once_with(
        provider._voice_states["alba"], "Hello.", copy_state=True,
    )
    provider.cleanup()
    provider.cleanup()
    assert provider._model is None
    assert provider._voice_states == {}
    provider.preload()
    assert model_type.load_model.call_count == 2


def test_instances_do_not_share_cache(backend):
    model_type, model = backend
    first = pocket.PocketTTSProvider()
    second = pocket.PocketTTSProvider()
    first.preload()
    second.preload()
    assert model_type.load_model.call_count == 2
    assert first._voice_states is not second._voice_states
    first.cleanup()
    assert second._model is model
    assert second._voice_states


@pytest.mark.parametrize("as_voice", [False, True])
def test_local_reference_audio_is_explicit_and_cached(backend, tmp_path, as_voice):
    _, model = backend
    path = tmp_path / "reference.wav"
    path.write_bytes(b"mock reference")
    provider = pocket.PocketTTSProvider(
        **({"voice": str(path)} if as_voice else {"ref_audio_path": path})
    )
    provider.preload()
    provider.preload()
    model.get_state_for_audio_prompt.assert_called_once_with(path.resolve())
    provider.set_language("en")
    assert provider.voice == str(path)
    provider.set_voice("alba")
    provider.preload()
    assert model.get_state_for_audio_prompt.call_args.args == ("alba",)


def test_missing_reference_does_not_load_model(backend, tmp_path):
    model_type, _ = backend
    provider = pocket.PocketTTSProvider(ref_audio_path=tmp_path / "missing.wav")
    with pytest.raises(FileNotFoundError, match="reference audio not found"):
        provider.preload()
    model_type.load_model.assert_not_called()


@pytest.mark.parametrize("source", [
    "https://example.com/voice.wav", "hf://repo/voice.wav", "file:///voice.wav",
    "//server/share/voice.wav", "\\\\server\\share\\voice.wav", "file:voice.wav",
    "voice.safetensors", "",
])
def test_rejects_remote_or_non_audio_reference(source):
    with pytest.raises(ValueError, match="local"):
        pocket.PocketTTSProvider(ref_audio_path=source)


@pytest.mark.parametrize("voice", ["unknown", "https://example.com/voice.wav", "hf://repo/voice.wav"])
def test_invalid_voice_does_not_change_selection(voice):
    provider = pocket.PocketTTSProvider()
    with pytest.raises(ValueError):
        provider.set_voice(voice)
    assert provider.voice == "alba"


@pytest.mark.parametrize("language", ["fr", "de", "ja", "auto", "", "en-invalid"])
def test_unsupported_languages_fail_clearly(language):
    with pytest.raises(ValueError, match="English only"):
        pocket.PocketTTSProvider(language=language)
    provider = pocket.PocketTTSProvider()
    with pytest.raises(ValueError, match="English only"):
        provider.set_language(language)


@pytest.mark.asyncio
async def test_list_voices_needs_no_backend():
    provider = pocket.PocketTTSProvider()
    voices = await provider.list_voices("en-GB")
    assert {voice.id for voice in voices} == set(pocket.VOICE_NAMES)
    assert all(voice.language == "en" for voice in voices)
    with pytest.raises(ValueError, match="English only"):
        await provider.list_voices("fr")


@pytest.mark.parametrize("kwargs, message", [
    ({"device": "cuda"}, "cpu"),
    ({"rate": "+20%"}, "rate adjustment"),
    ({"pitch": "+10Hz"}, "pitch adjustment"),
])
def test_unsupported_controls(kwargs, message):
    with pytest.raises(ValueError, match=message):
        pocket.PocketTTSProvider(**kwargs)


def test_missing_install_and_wrong_version(monkeypatch):
    def missing(name):
        raise pocket.PackageNotFoundError(name)

    provider = pocket.PocketTTSProvider()
    monkeypatch.setattr(pocket, "version", missing)
    with pytest.raises(RuntimeError, match="requirements-optional-pocket.txt"):
        provider.preload()
    monkeypatch.setattr(pocket, "version", lambda name: "1.1.1")
    with pytest.raises(RuntimeError, match="requires pocket-tts==3.1.0"):
        provider.preload()


def test_import_failure_is_actionable(backend, monkeypatch):
    monkeypatch.setitem(sys.modules, "pocket_tts", None)
    with pytest.raises(RuntimeError, match="dependencies could not load"):
        pocket.PocketTTSProvider().preload()


def test_model_load_failure_can_retry(backend):
    model_type, model = backend
    model_type.load_model.side_effect = OSError("offline")
    provider = pocket.PocketTTSProvider()
    with pytest.raises(RuntimeError, match="model access permissions") as error:
        provider.preload()
    assert isinstance(error.value.__cause__, OSError)
    assert provider._model is None
    model_type.load_model.side_effect = None
    provider.preload()
    assert provider._model is model


def test_voice_access_failure_does_not_cache_or_fallback(backend):
    model_type, model = backend
    model.get_state_for_audio_prompt.side_effect = ValueError("cloning unavailable")
    provider = pocket.PocketTTSProvider()
    with pytest.raises(RuntimeError, match="No alternative voice"):
        provider.preload()
    assert provider._voice_states == {}
    model.get_state_for_audio_prompt.side_effect = lambda source: {"steps": [0]}
    provider.preload()
    model_type.load_model.assert_called_once()


@pytest.mark.asyncio
async def test_synthesis_failure_releases_lock(backend):
    _, model = backend
    model.generate_audio.side_effect = RuntimeError("inference failed")
    provider = pocket.PocketTTSProvider()
    with pytest.raises(RuntimeError, match="synthesis failed on CPU"):
        await provider.synthesize("Hello.")
    await asyncio.wait_for(asyncio.to_thread(provider.cleanup), timeout=2)
    assert provider._model is None


@pytest.mark.asyncio
@pytest.mark.parametrize("samples", [[], [float("nan")], [float("inf")], [[0, 1]]])
async def test_invalid_model_audio(backend, samples):
    _, model = backend
    model.generate_audio.side_effect = lambda *args, **kwargs: FakeTensor(np.array(samples))
    with pytest.raises(RuntimeError, match="invalid or empty mono audio"):
        await pocket.PocketTTSProvider().synthesize("Hello.")


@pytest.mark.asyncio
async def test_invalid_sample_rate(backend):
    _, model = backend
    model.sample_rate = 0
    with pytest.raises(RuntimeError, match="invalid sample rate"):
        await pocket.PocketTTSProvider().synthesize("Hello.")


@pytest.mark.asyncio
async def test_output_write_failure_releases_lock(backend, tmp_path):
    provider = pocket.PocketTTSProvider()
    with pytest.raises(OSError):
        await provider.synthesize("Hello.", tmp_path)
    await asyncio.wait_for(asyncio.to_thread(provider.cleanup), 2)
    assert provider._model is None


@pytest.mark.asyncio
async def test_concurrent_first_load_and_synthesis_are_serialized(backend):
    model_type, model = backend
    provider = pocket.PocketTTSProvider()
    entered = threading.Event()
    release = threading.Event()
    calls = []
    original_generate = model.generate_audio.side_effect

    def blocked_generate(*args, **kwargs):
        calls.append(threading.get_ident())
        if len(calls) == 1:
            entered.set()
            if not release.wait(5):
                raise RuntimeError("test inference timed out")
        return original_generate(*args, **kwargs)

    model.generate_audio.side_effect = blocked_generate
    tasks = [asyncio.create_task(provider.synthesize("First."))]
    try:
        assert await asyncio.to_thread(entered.wait, 2)
        tasks.extend(asyncio.create_task(provider.synthesize("Next.")) for _ in range(3))
        await asyncio.sleep(0.05)
        assert len(calls) == 1
        model_type.load_model.assert_called_once()
    finally:
        release.set()
        await asyncio.wait_for(asyncio.gather(*tasks), 3)
    assert len(calls) == 4
    assert all(thread_id != threading.get_ident() for thread_id in calls)
    model.get_state_for_audio_prompt.assert_called_once()
    assert provider._voice_states["alba"]["steps"] == [0]


@pytest.mark.asyncio
async def test_empty_input_does_not_load(backend):
    model_type, _ = backend
    with pytest.raises(ValueError, match="empty text"):
        await pocket.PocketTTSProvider().synthesize("  ")
    model_type.load_model.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["synthesize", "cleanup", "preload", "warmup", "set_voice"])
async def test_cancellation_keeps_worker_lock_until_inference_finishes(backend, operation):
    _, model = backend
    provider = pocket.PocketTTSProvider()
    entered = threading.Event()
    release = threading.Event()
    attempted = threading.Event()
    completed = threading.Event()
    original_generate = model.generate_audio.side_effect
    calls = []

    def blocked_generate(*args, **kwargs):
        calls.append(threading.get_ident())
        if len(calls) == 1:
            entered.set()
            if not release.wait(5):
                raise RuntimeError("test inference timed out")
        return original_generate(*args, **kwargs)

    model.generate_audio.side_effect = blocked_generate
    task = asyncio.create_task(provider.synthesize("First."))
    contender = None
    try:
        assert await asyncio.to_thread(entered.wait, 2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert provider._lock.locked()

        def contend():
            attempted.set()
            if operation == "synthesize":
                provider._synthesize_sync("Second.")
            elif operation == "set_voice":
                provider.set_voice("anna")
            else:
                getattr(provider, operation)()
            completed.set()

        contender = asyncio.create_task(asyncio.to_thread(contend))
        assert await asyncio.to_thread(attempted.wait, 2)
        assert not await asyncio.to_thread(completed.wait, 0.05)
        assert len(calls) == 1
        assert provider._model is model
        assert provider._voice == "alba"
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        if contender is not None:
            await asyncio.wait_for(contender, 3)
    assert completed.is_set()
    assert not provider._lock.locked()
    if operation == "cleanup":
        assert provider._model is None
        assert provider._voice_states == {}


CHILD_BOOTSTRAP = '''
import io, os, runpy, sys, time, types, wave
mode = sys.argv.pop(1)
script = sys.argv.pop(1)
class FakeProvider:
    def __init__(self, **kwargs):
        self.voice = kwargs.get("voice", "alba")
    def preload(self):
        assert os.environ["CUDA_VISIBLE_DEVICES"] == ""
        assert "PYTHONPATH" not in os.environ
        print("private upstream diagnostic", flush=True)
        os.write(1, b"native stdout diagnostic\\n")
        if mode == "startup_timeout": time.sleep(30)
        if mode == "startup_error": raise RuntimeError("private startup secret")
    def set_voice(self, voice): self.voice = voice
    def warmup(self): pass
    def cleanup(self): pass
    def _synthesize_sync(self, text):
        if mode == "timeout": time.sleep(30)
        if mode == "exit": os._exit(3)
        if mode == "error": raise RuntimeError(text)
        if mode == "stderr": os.write(2, b"private text\\n" * 20000)
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(22050)
            wav.writeframes(b"\\x00\\x00" * 16)
        return types.SimpleNamespace(
            audio_data=b"bad WAV" if mode == "bad_wav" else buffer.getvalue(),
            metadata={"synth_ms": 1},
        )
module = types.ModuleType("src.tts.pocket_tts_provider")
module.PocketTTSProvider = FakeProvider
sys.modules["src.tts.pocket_tts_provider"] = module
sys.argv = [script] + sys.argv[1:]
runpy.run_path(script, run_name="__main__")
'''


@pytest.fixture
def child_worker(monkeypatch):
    original_popen = subprocess.Popen
    processes = []
    workers = []
    mode = ["ok"]

    def launch(args, **kwargs):
        if len(args) < 4 or args[3] != str(pocket.WORKER_SCRIPT):
            return original_popen(args, **kwargs)
        assert kwargs["shell"] is False
        assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == ""
        assert args[0] == sys.executable
        command = [sys.executable, "-u", "-B", "-c", CHILD_BOOTSTRAP, mode[0], args[3], *args[4:]]
        process = original_popen(command, **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr(pocket.subprocess, "Popen", launch)
    from src.utils import platform_compat

    def kill(process, timeout=2):
        if process.poll() is None:
            process.kill()
        process.wait(timeout=timeout)

    monkeypatch.setattr(platform_compat, "kill_process_tree", kill)

    def create(selected_mode="ok", **kwargs):
        mode[0] = selected_mode
        provider = pocket.PocketTTSWorkerProvider(
            python_path=sys.executable, startup_timeout_sec=2, request_timeout_sec=0.3, **kwargs,
        )
        workers.append(provider)
        return provider

    yield create, processes, mode
    for provider in workers:
        provider.cleanup()
    assert all(process.poll() is not None for process in processes)
    assert all(process.stdin.closed and process.stdout.closed and process.stderr.closed for process in processes)


def test_worker_defaults_and_lightweight_constructor(monkeypatch):
    original_import = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.split(".")[0] in ("torch", "pocket_tts"):
            raise AssertionError(name)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    provider = pocket.PocketTTSWorkerProvider()
    assert ".venv-pocket-tts" in str(provider.python_path)
    assert provider.startup_timeout_sec == 300
    assert provider.request_timeout_sec == 60
    assert provider.voice == "alba"
    assert provider.device == "cpu"
    provider.set_voice("anna")
    provider.set_language("en-GB")
    with pytest.raises(ValueError):
        provider.set_language("fr")
    with pytest.raises(ValueError):
        provider.set_rate("+10%")
    provider.cleanup()
    with pytest.raises(RuntimeError, match="closed"):
        provider.preload()


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("nan")])
def test_worker_rejects_bad_timeouts(timeout):
    with pytest.raises(ValueError, match="finite and positive"):
        pocket.PocketTTSWorkerProvider(request_timeout_sec=timeout)


@pytest.mark.asyncio
async def test_worker_persistence_wav_voice_and_stream(child_worker, tmp_path):
    create, processes, _ = child_worker
    provider = create()
    await asyncio.to_thread(provider.preload)
    worker = provider._worker
    await asyncio.to_thread(provider.warmup)
    results = await asyncio.gather(*(provider.synthesize("Hello.") for _ in range(3)))
    assert len(processes) == 1
    assert all(result.duration == 16 / 22050 for result in results)
    assert results[0].metadata["backend"] == "worker"
    provider.set_voice("anna")
    path = tmp_path / "out.wav"
    saved = await provider.synthesize("Again.", path)
    assert saved.audio_path == path
    assert saved.audio_data is None
    assert saved.metadata["voice"] == "anna"
    assert [chunk async for chunk in provider.synthesize_stream("Hello")] == [path.read_bytes()]
    provider.cleanup()
    provider.cleanup()
    assert all(not thread.is_alive() for thread in worker.threads)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode, message", [
    ("startup_error", "startup failed"), ("startup_timeout", "startup timed out"),
    ("timeout", "synthesize timed out"), ("error", "synthesize failed"),
    ("exit", "exited or sent invalid JSON"), ("bad_wav", "invalid WAV audio"),
])
async def test_worker_failure_is_bounded_and_kills_child(child_worker, mode, message):
    create, processes, _ = child_worker
    provider = create(mode)
    provider.startup_timeout_sec = 0.3
    started = time.monotonic()
    with pytest.raises(RuntimeError, match=message) as error:
        await provider.synthesize("private user text")
    assert "private" not in str(error.value)
    assert time.monotonic() - started < 4
    assert len(processes) == 1
    assert processes[0].poll() is not None
    assert provider._worker is None


@pytest.mark.asyncio
async def test_worker_stderr_is_drained_not_logged(child_worker, caplog):
    create, _, _ = child_worker
    provider = create("stderr")
    assert (await provider.synthesize("private user text")).audio_data
    assert "private" not in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["cancel_inflight", "cleanup", "task_cancel"])
async def test_worker_cancel_kills_inference_and_invalidates_queued_jobs(child_worker, operation):
    create, processes, mode = child_worker
    provider = create("timeout")
    provider.request_timeout_sec = 20
    await asyncio.to_thread(provider.preload)
    generation, voice, marker = provider._snapshot()
    active = asyncio.create_task(provider.synthesize("Block."))
    await asyncio.sleep(0.05)
    queued = asyncio.create_task(
        asyncio.to_thread(provider._request, "warmup", generation, voice, threading.Event()),
    )
    await asyncio.sleep(0.05)
    started = time.monotonic()
    if operation == "task_cancel":
        active.cancel()
    else:
        getattr(provider, operation)()
    results = await asyncio.wait_for(asyncio.gather(active, queued, return_exceptions=True), 3)
    assert all(isinstance(result, BaseException) for result in results)
    assert time.monotonic() - started < 3
    assert len(processes) == 1
    assert processes[0].poll() is not None
    assert provider._worker is None
    if operation == "cleanup":
        with pytest.raises(RuntimeError, match="closed"):
            await provider.synthesize("Must not restart.")
    else:
        mode[0] = "ok"
        assert (await provider.synthesize("Explicit new request.")).audio_data
        assert len(processes) == 2


@pytest.mark.asyncio
async def test_worker_cleanup_during_startup(child_worker):
    create, processes, _ = child_worker
    provider = create("startup_timeout")
    provider.startup_timeout_sec = 20
    task = asyncio.create_task(provider.synthesize("Hello"))
    deadline = time.monotonic() + 2
    while not processes and time.monotonic() < deadline:
        await asyncio.sleep(0.01)
    assert processes
    provider.cleanup()
    with pytest.raises(RuntimeError, match="closed"):
        await asyncio.wait_for(task, 3)
    assert processes[0].poll() is not None


def test_worker_missing_python_fails_without_fallback(tmp_path):
    provider = pocket.PocketTTSWorkerProvider(python_path=tmp_path / "missing.exe")
    with pytest.raises(RuntimeError, match="dedicated .venv-pocket-tts"):
        provider.preload()
    assert provider._worker is None
    provider.cleanup()
