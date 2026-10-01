"""Tests for the local RVC wrapper."""

import hashlib
import json
import time
import sys
import types
import threading
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from src.tts import rvc_provider
from src.tts.rvc_provider import RVCConverter


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


class FakeModernRVC:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def infer_audio(self, voice_model: str, audio_path: str, audio_format: str = "wav", **kwargs):
        output_dir = Path.cwd() / "output"
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / f"{voice_model}.{audio_format}"
        output_path.write_bytes(Path(audio_path).read_bytes())
        return str(output_path)


class FakeInferRVC:
    def __init__(self, model: str, index: str | None = None, config=None):
        self.model = model
        self.index = index
        self.config = config
        self.outputfreq = 32000

    def __call__(self, audio_path: str, **kwargs):
        return [0.0, 0.1, -0.1, 0.0]


class FakeConfig:
    def __init__(self):
        self.device = "cuda:0"
        self.is_half = True
        self.x_pad = 3
        self.x_query = 10
        self.x_center = 60
        self.x_max = 65


class DurationPreservingInferRVC(FakeInferRVC):
    """InferRVC stand-in that keeps the input duration at tgt_sr."""

    def __call__(self, audio_path: str, **kwargs):
        audio, sample_rate = sf.read(audio_path, dtype="float32")
        target_rate = getattr(self, "tgt_sr", None) or self.outputfreq
        samples = int(round(len(audio) / float(sample_rate) * target_rate))
        return np.zeros(samples, dtype=np.float32)


class _FakeEmptyStream:
    def readline(self):
        return ""

    def close(self):
        return None


class _FakeStdout:
    def __init__(self):
        self.queue = [json.dumps({"status": "ready", "backend": "inferrvc"}) + "\n"]

    def readline(self):
        if not self.queue:
            return ""
        return self.queue.pop(0)


class _FakeStdin:
    def __init__(self, process):
        self.process = process
        self.last_payload = None

    def write(self, data: str):
        self.last_payload = json.loads(data)

    def flush(self):
        if self.last_payload is None:
            return
        command = self.last_payload.get("command")
        if command == "convert":
            output_path = Path(self.last_payload["output_path"])
            output_path.write_bytes(b"worker-output")
            self.process.stdout.queue.append(
                json.dumps({"status": "ok", "output_path": str(output_path)}) + "\n"
            )
        elif command == "shutdown":
            self.process.stdout.queue.append(json.dumps({"status": "bye"}) + "\n")


class _BlockingStdin(_FakeStdin):
    def flush(self):
        if self.last_payload is None:
            return
        command = self.last_payload.get("command")
        if command == "shutdown":
            self.process.stdout.queue.append(json.dumps({"status": "bye"}) + "\n")


class FakePopen:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.stdout = _FakeStdout()
        self.stdin = _FakeStdin(self)
        self.stderr = _FakeEmptyStream()
        self._terminated = False

    def poll(self):
        return None if not self._terminated else 0

    def wait(self, timeout=None):
        self._terminated = True
        return 0

    def kill(self):
        self._terminated = True

    def terminate(self):
        self._terminated = True


class _BlockingStdout(_FakeStdout):
    def readline(self):
        if self.queue:
            return self.queue.pop(0)
        time.sleep(0.2)
        return ""


class FakeBlockingPopen(FakePopen):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stdout = _BlockingStdout()
        self.stdin = _BlockingStdin(self)


class _NeverReadyStdout:
    def readline(self):
        time.sleep(0.2)
        return ""


class FakeNeverReadyPopen(FakePopen):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stdout = _NeverReadyStdout()


@pytest.fixture
def worker_converter(tmp_path, monkeypatch):
    python_path = tmp_path / "python.exe"
    worker_script = tmp_path / "rvc_worker.py"
    model_path = tmp_path / "voice.pth"
    for path in (python_path, worker_script, model_path):
        path.write_bytes(b"fake")
    monkeypatch.setattr(rvc_provider.subprocess, "Popen", FakePopen)
    monkeypatch.setattr("src.utils.platform_compat.kill_process_tree", lambda process: process.terminate())
    converter = RVCConverter(
        model_path=model_path,
        backend="worker",
        python_path=python_path,
        worker_script=worker_script,
        site_packages_dir=tmp_path / "overlay",
        request_timeout_sec=15,
    )
    yield converter
    converter.close()


def test_worker_default_startup_timeout_is_separate(worker_converter):
    assert worker_converter.startup_timeout_sec == 90
    assert worker_converter.request_timeout_sec == 15


def test_concurrent_load_reads_worker_handshake_once(worker_converter, monkeypatch):
    worker_converter.spawn_worker()
    process = worker_converter._worker_process
    first_read_started = threading.Event()
    second_load_entered = threading.Event()
    release_ready = threading.Event()
    original_lock = worker_converter._worker_lock
    original_read = worker_converter._read_worker_response_line
    reads = []
    errors = []
    ready_states = []

    class ObservedLock:
        def __enter__(self):
            if first_read_started.is_set():
                second_load_entered.set()
            original_lock.acquire()

        def __exit__(self, *args):
            original_lock.release()

    def read_ready(timeout_sec, *, operation):
        reads.append(operation)
        if len(reads) == 1:
            first_read_started.set()
            assert release_ready.wait(5)
        else:
            second_load_entered.set()
        return original_read(timeout_sec, operation=operation)

    def load():
        try:
            worker_converter._load()
            ready_states.append(worker_converter._worker_ready)
        except Exception as exc:
            errors.append(exc)

    monkeypatch.setattr(worker_converter, "_worker_lock", ObservedLock())
    monkeypatch.setattr(worker_converter, "_read_worker_response_line", read_ready)
    threads = [threading.Thread(target=load, daemon=True) for _ in range(2)]
    threads[0].start()
    try:
        assert first_read_started.wait(5)
        threads[1].start()
        # The contender reaches either the lock or a second handshake read.
        assert second_load_entered.wait(5)
    finally:
        release_ready.set()
        for thread in threads:
            if thread.ident is not None:
                thread.join(timeout=5)

    assert all(not thread.is_alive() for thread in threads)
    assert errors == []
    assert reads == ["startup"]
    assert ready_states == [True, True]
    assert worker_converter._converter is process
    assert worker_converter._worker_process is process
    assert process.poll() is None


@pytest.mark.parametrize("elapsed, remaining", [(80.0, 10.0), (120.0, 0.1)])
def test_worker_startup_uses_budget_since_spawn(
    worker_converter, monkeypatch, elapsed, remaining,
):
    now = [100.0]
    monkeypatch.setattr(rvc_provider.time, "monotonic", lambda: now[0])
    worker_converter.spawn_worker()
    process = worker_converter._worker_process
    now[0] += elapsed
    # Reusing the early worker must not restart its startup budget.
    worker_converter.spawn_worker()
    timeouts = []
    original_read = worker_converter._read_worker_response_line

    def read_response(timeout_sec, *, operation="response"):
        timeouts.append((operation, timeout_sec))
        return original_read(timeout_sec, operation=operation)

    monkeypatch.setattr(worker_converter, "_read_worker_response_line", read_response)
    worker_converter.preload()

    assert timeouts == [("startup", pytest.approx(remaining))]
    assert worker_converter._worker_process is process
    assert worker_converter._worker_ready is True
    assert process.stdout.queue == []


def test_worker_relaunch_resets_startup_budget(worker_converter, monkeypatch):
    now = [100.0]
    monkeypatch.setattr(rvc_provider.time, "monotonic", lambda: now[0])
    worker_converter.spawn_worker()
    now[0] += 120.0
    worker_converter._terminate_worker_process()
    timeouts = []

    def popen(*args, **kwargs):
        # Time spent launching the replacement also counts toward its budget.
        now[0] += 5.0
        return FakePopen(*args, **kwargs)

    def read_response(timeout_sec, *, operation="response"):
        timeouts.append((operation, timeout_sec))
        return worker_converter._worker_process.stdout.readline()

    monkeypatch.setattr(rvc_provider.subprocess, "Popen", popen)
    monkeypatch.setattr(worker_converter, "_read_worker_response_line", read_response)
    worker_converter._relaunch_worker()

    assert timeouts == [("startup", pytest.approx(85.0))]
    assert worker_converter._worker_ready is True


@pytest.mark.parametrize("startup_timeout", [0.5, 90.0])
def test_worker_startup_and_conversion_use_distinct_timeouts(
    worker_converter, monkeypatch, tmp_path, startup_timeout,
):
    monkeypatch.setattr(rvc_provider.time, "monotonic", lambda: 100.0)
    worker_converter.startup_timeout_sec = startup_timeout
    calls = []

    def read_response(timeout_sec, *, operation="response"):
        calls.append((operation, timeout_sec))
        # Simulate ready arriving within its budget, even beyond the phrase budget.
        if operation == "startup":
            ready_after_sec = startup_timeout / 2
            if timeout_sec < ready_after_sec:
                raise TimeoutError("worker is still starting")
        return worker_converter._worker_process.stdout.readline()

    monkeypatch.setattr(worker_converter, "_read_worker_response_line", read_response)
    worker_converter.convert_file(tmp_path / "input.wav", tmp_path / "output.wav")

    assert calls == [("startup", startup_timeout), ("response", 15)]
    assert worker_converter._worker_ready is True


def test_worker_spawn_defers_ready_and_reuses_process(worker_converter, monkeypatch):
    processes = []
    reads = []

    def popen(*args, **kwargs):
        process = FakePopen(*args, **kwargs)
        original_readline = process.stdout.readline

        def readline():
            reads.append("ready")
            return original_readline()

        process.stdout.readline = readline
        processes.append(process)
        return process

    monkeypatch.setattr(rvc_provider.subprocess, "Popen", popen)
    worker_converter.spawn_worker()
    worker_converter.spawn_worker()
    assert len(processes) == 1
    assert reads == []
    assert worker_converter._converter is None
    assert worker_converter._worker_ready is False
    assert worker_converter._warmed_up is False

    worker_converter.preload()
    worker_converter.preload()
    assert len(processes) == 1
    assert reads == ["ready"]
    assert worker_converter._worker_ready is True
    assert worker_converter._warmed_up is False


@pytest.mark.parametrize("backend", ["inferrvc", "rvc_inferpy"])
def test_non_worker_spawn_does_not_load_backend(worker_converter, monkeypatch, backend):
    worker_converter.backend = backend
    loads = []
    monkeypatch.setattr(
        worker_converter, "_resolve_backend",
        lambda: pytest.fail("Spawn must not resolve in-process backends"),
    )
    worker_converter.spawn_worker()
    assert worker_converter._worker_process is None

    monkeypatch.setattr(worker_converter, "_resolve_backend", lambda: backend)
    loader = "_load_inferrvc" if backend == "inferrvc" else "_load_legacy_backend"
    monkeypatch.setattr(worker_converter, loader, lambda: loads.append(backend))
    worker_converter.preload()
    assert loads == [backend]


def test_worker_relaunch_uses_startup_timeout(worker_converter, monkeypatch):
    monkeypatch.setattr(rvc_provider.time, "monotonic", lambda: 100.0)
    worker_converter.startup_timeout_sec = 75
    timeouts = []

    def read_response(timeout_sec, *, operation="response"):
        timeouts.append((operation, timeout_sec))
        return worker_converter._worker_process.stdout.readline()

    monkeypatch.setattr(worker_converter, "_read_worker_response_line", read_response)
    worker_converter.preload()
    worker_converter._terminate_worker_process()
    worker_converter._relaunch_worker()

    assert timeouts == [("startup", 75), ("startup", 75)]
    assert worker_converter._worker_ready is True


def test_worker_reads_delayed_ready_beyond_phrase_budget(worker_converter, monkeypatch):
    monkeypatch.setattr(
        rvc_provider.subprocess, "Popen",
        _make_scripted_popen(["slow_start"], startup_delay=0.15),
    )
    worker_converter.request_timeout_sec = 0.01
    worker_converter.startup_timeout_sec = 1.0
    worker_converter.preload()
    assert worker_converter._worker_ready is True
    assert worker_converter._worker_relaunch_count == 0


@pytest.mark.parametrize("response", ["", "invalid json", '{"status": "error"}'])
def test_early_worker_handshake_failure_closes_process(worker_converter, response):
    worker_converter.spawn_worker()
    process = worker_converter._worker_process
    process.stdout.queue = [response]

    with pytest.raises(RuntimeError):
        worker_converter.preload()

    assert process.poll() is not None
    assert worker_converter._worker_process is None
    assert worker_converter._worker_ready is False
    assert worker_converter._worker_relaunch_count == 0


def test_worker_close_after_spawn_does_not_wait_for_ready(worker_converter):
    worker_converter.spawn_worker()
    process = worker_converter._worker_process
    worker_converter.close()
    assert process.poll() is not None
    assert worker_converter._worker_process is None


def test_worker_warmup_timeout_does_not_relaunch_disabled_voice(worker_converter, monkeypatch):
    monkeypatch.setattr(rvc_provider.subprocess, "Popen", FakeBlockingPopen)
    relaunches = []
    monkeypatch.setattr(worker_converter, "_schedule_worker_relaunch", lambda: relaunches.append("retry"))
    worker_converter.request_timeout_sec = 0.01
    worker_converter.preload()

    with pytest.raises(TimeoutError):
        worker_converter.warmup()

    assert relaunches == []
    assert worker_converter._worker_process is None
    assert worker_converter._warmed_up is False


def test_convert_file_with_modern_rvc_backend(tmp_path, monkeypatch):
    fake_module = types.ModuleType("rvc_inferpy")
    fake_module.RVCConverter = FakeModernRVC
    monkeypatch.setitem(sys.modules, "rvc_inferpy", fake_module)
    monkeypatch.setattr(rvc_provider, "PROJECT_ROOT", tmp_path)

    model_path = tmp_path / "March-7th.pth"
    index_path = tmp_path / "March-7th.index"
    input_path = tmp_path / "input.wav"
    output_path = tmp_path / "output.wav"

    model_path.write_bytes(b"fake model")
    index_path.write_bytes(b"fake index")
    input_path.write_bytes(b"fake wav")

    converter = RVCConverter(
        model_path=model_path,
        index_path=index_path,
        device="cpu",
        backend="rvc_inferpy",
    )

    result = converter.convert_file(input_path, output_path)

    assert result == output_path
    assert output_path.read_bytes() == b"fake wav"
    assert (tmp_path / "models" / "March-7th" / "March-7th.pth").exists()
    assert (tmp_path / "models" / "March-7th" / "March-7th.index").exists()


def test_convert_file_with_inferrvc_backend(tmp_path, monkeypatch):
    inferrvc_module = types.ModuleType("inferrvc")
    inferrvc_module.RVC = FakeInferRVC
    inferrvc_configs_module = types.ModuleType("inferrvc.configs")
    inferrvc_config_module = types.ModuleType("inferrvc.configs.config")
    inferrvc_config_module.Config = FakeConfig

    monkeypatch.setitem(sys.modules, "inferrvc", inferrvc_module)
    monkeypatch.setitem(sys.modules, "inferrvc.configs", inferrvc_configs_module)
    monkeypatch.setitem(sys.modules, "inferrvc.configs.config", inferrvc_config_module)

    model_path = tmp_path / "March-7th.pth"
    index_path = tmp_path / "March-7th.index"
    input_path = tmp_path / "input.wav"
    output_path = tmp_path / "output.wav"

    model_path.write_bytes(b"fake model")
    index_path.write_bytes(b"fake index")
    sf.write(input_path, [0.0, 0.1, -0.1, 0.0], 24000)

    converter = RVCConverter(
        model_path=model_path,
        index_path=index_path,
        device="cpu",
        backend="inferrvc",
    )

    result = converter.convert_file(input_path, output_path)
    audio, sample_rate = sf.read(output_path, dtype="float32")

    assert result == output_path
    assert sample_rate == 32000
    assert audio.shape[0] == 4


def test_inferrvc_backend_trims_short_input_padding(tmp_path, monkeypatch):
    inferrvc_module = types.ModuleType("inferrvc")
    inferrvc_module.RVC = DurationPreservingInferRVC
    inferrvc_configs_module = types.ModuleType("inferrvc.configs")
    inferrvc_config_module = types.ModuleType("inferrvc.configs.config")
    inferrvc_config_module.Config = FakeConfig

    monkeypatch.setitem(sys.modules, "inferrvc", inferrvc_module)
    monkeypatch.setitem(sys.modules, "inferrvc.configs", inferrvc_configs_module)
    monkeypatch.setitem(sys.modules, "inferrvc.configs.config", inferrvc_config_module)

    model_path = tmp_path / "March-7th.pth"
    input_path = tmp_path / "input.wav"
    output_path = tmp_path / "output.wav"

    model_path.write_bytes(b"fake model")
    sf.write(input_path, np.zeros(8000, dtype=np.float32), 16000)

    converter = RVCConverter(
        model_path=model_path,
        backend="inferrvc",
    )

    result = converter.convert_file(input_path, output_path)
    audio, sample_rate = sf.read(output_path, dtype="float32")

    assert result == output_path
    assert sample_rate == 32000
    assert len(audio) / float(sample_rate) == pytest.approx(0.5, abs=0.06)


def test_convert_file_with_worker_backend(tmp_path, monkeypatch):
    python_path = tmp_path / "python.exe"
    worker_script = tmp_path / "rvc_worker.py"
    model_path = tmp_path / "March-7th.pth"
    index_path = tmp_path / "March-7th.index"
    input_path = tmp_path / "input.wav"
    output_path = tmp_path / "output.wav"

    python_path.write_text("")
    worker_script.write_text("")
    model_path.write_bytes(b"fake model")
    index_path.write_bytes(b"fake index")
    input_path.write_bytes(b"fake wav")

    monkeypatch.setattr(rvc_provider.subprocess, "Popen", FakePopen)

    converter = RVCConverter(
        model_path=model_path,
        index_path=index_path,
        backend="worker",
        python_path=python_path,
        worker_script=worker_script,
        site_packages_dir=tmp_path / ".rvc-site-packages",
    )

    result = converter.convert_file(input_path, output_path)

    assert result == output_path
    assert output_path.read_bytes() == b"worker-output"
    converter.close()


def test_worker_backend_accepts_matching_model_sha256(tmp_path, monkeypatch):
    python_path = tmp_path / "python.exe"
    worker_script = tmp_path / "rvc_worker.py"
    model_path = tmp_path / "March-7th.pth"
    input_path = tmp_path / "input.wav"
    output_path = tmp_path / "output.wav"

    model_bytes = b"fake model"
    python_path.write_text("")
    worker_script.write_text("")
    model_path.write_bytes(model_bytes)
    input_path.write_bytes(b"fake wav")

    monkeypatch.setattr(rvc_provider.subprocess, "Popen", FakePopen)

    converter = RVCConverter(
        model_path=model_path,
        backend="worker",
        python_path=python_path,
        worker_script=worker_script,
        site_packages_dir=tmp_path / ".rvc-site-packages",
        model_sha256=_sha256_bytes(model_bytes),
    )

    assert converter.convert_file(input_path, output_path) == output_path
    converter.close()


def test_worker_backend_rejects_mismatched_model_sha256(tmp_path, monkeypatch):
    python_path = tmp_path / "python.exe"
    worker_script = tmp_path / "rvc_worker.py"
    model_path = tmp_path / "March-7th.pth"
    input_path = tmp_path / "input.wav"
    output_path = tmp_path / "output.wav"

    python_path.write_text("")
    worker_script.write_text("")
    model_path.write_bytes(b"fake model")
    input_path.write_bytes(b"fake wav")

    monkeypatch.setattr(rvc_provider.subprocess, "Popen", FakePopen)

    converter = RVCConverter(
        model_path=model_path,
        backend="worker",
        python_path=python_path,
        worker_script=worker_script,
        site_packages_dir=tmp_path / ".rvc-site-packages",
        model_sha256="0" * 64,
    )

    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        converter.convert_file(input_path, output_path)

    assert converter._worker_process is None


def test_worker_backend_times_out_and_resets_worker(tmp_path, monkeypatch):
    python_path = tmp_path / "python.exe"
    worker_script = tmp_path / "rvc_worker.py"
    model_path = tmp_path / "March-7th.pth"
    index_path = tmp_path / "March-7th.index"
    input_path = tmp_path / "input.wav"
    output_path = tmp_path / "output.wav"

    python_path.write_text("")
    worker_script.write_text("")
    model_path.write_bytes(b"fake model")
    index_path.write_bytes(b"fake index")
    input_path.write_bytes(b"fake wav")

    monkeypatch.setattr(rvc_provider.subprocess, "Popen", FakeBlockingPopen)

    converter = RVCConverter(
        model_path=model_path,
        index_path=index_path,
        backend="worker",
        python_path=python_path,
        worker_script=worker_script,
        site_packages_dir=tmp_path / ".rvc-site-packages",
        request_timeout_sec=0.01,
    )

    with pytest.raises(TimeoutError):
        converter.convert_file(input_path, output_path)

    assert converter._worker_relaunch_count == 1
    assert _wait_until(lambda: converter._worker_ready) is True
    assert converter._worker_process is not None
    converter.close()


def test_worker_startup_times_out_and_resets_worker(tmp_path, monkeypatch):
    python_path = tmp_path / "python.exe"
    worker_script = tmp_path / "rvc_worker.py"
    model_path = tmp_path / "March-7th.pth"
    index_path = tmp_path / "March-7th.index"
    input_path = tmp_path / "input.wav"
    output_path = tmp_path / "output.wav"

    python_path.write_text("")
    worker_script.write_text("")
    model_path.write_bytes(b"fake model")
    index_path.write_bytes(b"fake index")
    input_path.write_bytes(b"fake wav")

    monkeypatch.setattr(rvc_provider.subprocess, "Popen", FakeNeverReadyPopen)

    converter = RVCConverter(
        model_path=model_path,
        index_path=index_path,
        backend="worker",
        python_path=python_path,
        worker_script=worker_script,
        site_packages_dir=tmp_path / ".rvc-site-packages",
        request_timeout_sec=15.0,
        startup_timeout_sec=0.01,
    )
    monkeypatch.setattr(
        converter, "_schedule_worker_relaunch",
        lambda: pytest.fail("A failed startup must not relaunch a disabled worker"),
    )

    with pytest.raises(TimeoutError, match="startup timed out"):
        converter.convert_file(input_path, output_path)

    assert converter._worker_process is None
    assert converter._worker_ready is False
    assert converter._worker_relaunch_count == 0
    converter.close()


def test_convert_file_fast_fails_while_relaunching_without_backend(tmp_path, monkeypatch):
    """Regression for LIL-67: after a first-startup timeout ``_backend_name`` is
    still ``None`` while the relaunch thread is in flight. A concurrent
    conversion must fail fast instead of calling ``_load()`` and spawning a
    second worker in parallel with that relaunch.
    """

    python_path = tmp_path / "python.exe"
    worker_script = tmp_path / "rvc_worker.py"
    model_path = tmp_path / "March-7th.pth"
    input_path = tmp_path / "input.wav"
    output_path = tmp_path / "output.wav"

    python_path.write_text("")
    worker_script.write_text("")
    model_path.write_bytes(b"fake model")
    input_path.write_bytes(b"fake wav")

    spawned: list[FakePopen] = []

    class _RecordingPopen(FakePopen):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            spawned.append(self)

    monkeypatch.setattr(rvc_provider.subprocess, "Popen", _RecordingPopen)

    converter = RVCConverter(
        model_path=model_path,
        backend="worker",
        python_path=python_path,
        worker_script=worker_script,
        site_packages_dir=tmp_path / ".rvc-site-packages",
    )
    # State left behind by a first-startup timeout, with the auto-relaunch
    # thread still running.
    converter._backend_name = None
    converter._relaunching = True

    with pytest.raises(RuntimeError, match="relaunching|not ready"):
        converter.convert_file(input_path, output_path)

    assert spawned == []


def test_convert_array_fast_fails_while_relaunching_without_backend(tmp_path, monkeypatch):
    """Regression for LIL-67 follow-up: ``convert_array`` calls ``_load()``
    before reaching the ``convert_file`` guard, so after a first-startup timeout
    (``_backend_name`` still ``None``) it could spawn a second worker while the
    relaunch thread is in flight.
    """

    python_path = tmp_path / "python.exe"
    worker_script = tmp_path / "rvc_worker.py"
    model_path = tmp_path / "March-7th.pth"

    python_path.write_text("")
    worker_script.write_text("")
    model_path.write_bytes(b"fake model")

    spawned: list[FakePopen] = []

    class _RecordingPopen(FakePopen):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            spawned.append(self)

    monkeypatch.setattr(rvc_provider.subprocess, "Popen", _RecordingPopen)

    converter = RVCConverter(
        model_path=model_path,
        backend="worker",
        python_path=python_path,
        worker_script=worker_script,
        site_packages_dir=tmp_path / ".rvc-site-packages",
    )
    converter._backend_name = None
    converter._relaunching = True

    with pytest.raises(RuntimeError, match="relaunching|not ready"):
        converter.convert_array(np.zeros(1600, dtype=np.float32), 16000)

    assert spawned == []


def test_preload_fast_fails_while_relaunching_without_backend(tmp_path, monkeypatch):
    """Regression for LIL-67 follow-up: ``preload`` must not spawn a second
    worker while a background relaunch is already starting one.
    """

    python_path = tmp_path / "python.exe"
    worker_script = tmp_path / "rvc_worker.py"
    model_path = tmp_path / "March-7th.pth"

    python_path.write_text("")
    worker_script.write_text("")
    model_path.write_bytes(b"fake model")

    spawned: list[FakePopen] = []

    class _RecordingPopen(FakePopen):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            spawned.append(self)

    monkeypatch.setattr(rvc_provider.subprocess, "Popen", _RecordingPopen)

    converter = RVCConverter(
        model_path=model_path,
        backend="worker",
        python_path=python_path,
        worker_script=worker_script,
        site_packages_dir=tmp_path / ".rvc-site-packages",
    )
    converter._backend_name = None
    converter._relaunching = True

    with pytest.raises(RuntimeError, match="relaunching|not ready"):
        converter.preload()

    assert spawned == []


def test_relaunch_defers_worker_ready_until_after_rewarmup(tmp_path, monkeypatch):
    """Regression for LIL-67: the relaunched worker must not be advertised as
    ready until its re-warmup finishes, otherwise concurrent conversions block
    behind the warmup instead of failing fast.
    """

    python_path = tmp_path / "python.exe"
    worker_script = tmp_path / "rvc_worker.py"
    model_path = tmp_path / "March-7th.pth"

    python_path.write_text("")
    worker_script.write_text("")
    model_path.write_bytes(b"fake model")

    converter = RVCConverter(
        model_path=model_path,
        backend="worker",
        python_path=python_path,
        worker_script=worker_script,
        site_packages_dir=tmp_path / ".rvc-site-packages",
    )
    converter._backend_name = "worker"
    converter._warmed_up = True

    observed: dict[str, bool] = {}

    def _fake_start_worker() -> None:
        converter._worker_process = FakePopen()
        converter._backend_name = "worker"
        converter._worker_ready = True

    def _fake_warmup() -> None:
        observed["ready_during_warmup"] = converter._worker_ready
        observed["relaunching_during_warmup"] = converter._relaunching
        converter._warmed_up = True

    monkeypatch.setattr(converter, "_start_worker", _fake_start_worker)
    monkeypatch.setattr(converter, "warmup", _fake_warmup)

    converter._relaunching = True
    converter._relaunch_worker()

    assert observed["ready_during_warmup"] is False
    assert observed["relaunching_during_warmup"] is True
    assert converter._worker_ready is True
    assert converter._relaunching is False


def _wait_until(predicate, timeout: float = 3.0, interval: float = 0.01) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


def _make_scripted_popen(behaviors, startup_delay: float = 0.0):
    """Popen stand-in whose per-instance behavior is driven by a list."""

    class _Stdout:
        def __init__(self, process):
            self.process = process
            self._ready_sent = process.behavior != "slow_start"
            self.queue = (
                [json.dumps({"status": "ready", "backend": "worker"}) + "\n"]
                if self._ready_sent
                else []
            )

        def readline(self):
            if self.queue:
                return self.queue.pop(0)
            if self.process.behavior == "slow_start" and not self._ready_sent:
                time.sleep(startup_delay)
                self._ready_sent = True
                return json.dumps({"status": "ready", "backend": "worker"}) + "\n"
            if self.process.behavior == "block":
                time.sleep(5.0)
            return ""

    class _Stdin:
        def __init__(self, process):
            self.process = process
            self.last_payload = None

        def write(self, data):
            self.last_payload = json.loads(data)

        def flush(self):
            if self.last_payload is None:
                return
            command = self.last_payload.get("command")
            if command == "convert" and self.process.behavior != "block":
                output_path = Path(self.last_payload["output_path"])
                output_path.write_bytes(b"worker-output")
                self.process.stdout.queue.append(
                    json.dumps({"status": "ok", "output_path": str(output_path)}) + "\n"
                )
            elif command == "shutdown":
                self.process.stdout.queue.append(json.dumps({"status": "bye"}) + "\n")

    class _ScriptedPopen:
        _instance_count = 0

        def __init__(self, *args, **kwargs):
            index = _ScriptedPopen._instance_count
            _ScriptedPopen._instance_count += 1
            self.behavior = behaviors[min(index, len(behaviors) - 1)]
            self.stdout = _Stdout(self)
            self.stdin = _Stdin(self)
            self.stderr = _FakeEmptyStream()
            self._terminated = False

        def poll(self):
            return 0 if self._terminated else None

        def wait(self, timeout=None):
            self._terminated = True
            return 0

        def kill(self):
            self._terminated = True

        def terminate(self):
            self._terminated = True

    return _ScriptedPopen


def test_worker_request_timeout_relaunches_and_serves_again(tmp_path, monkeypatch):
    python_path = tmp_path / "python.exe"
    worker_script = tmp_path / "rvc_worker.py"
    model_path = tmp_path / "March-7th.pth"
    index_path = tmp_path / "March-7th.index"
    input_path = tmp_path / "input.wav"
    output_path = tmp_path / "output.wav"

    python_path.write_text("")
    worker_script.write_text("")
    model_path.write_bytes(b"fake model")
    index_path.write_bytes(b"fake index")
    input_path.write_bytes(b"fake wav")

    monkeypatch.setattr(
        rvc_provider.subprocess, "Popen", _make_scripted_popen(["block", "ok"])
    )

    converter = RVCConverter(
        model_path=model_path,
        index_path=index_path,
        backend="worker",
        python_path=python_path,
        worker_script=worker_script,
        site_packages_dir=tmp_path / ".rvc-site-packages",
        request_timeout_sec=0.05,
    )

    with pytest.raises(TimeoutError):
        converter.convert_file(input_path, output_path)

    assert converter._worker_relaunch_count == 1
    assert _wait_until(lambda: converter._worker_ready) is True

    result = converter.convert_file(input_path, output_path)

    assert result == output_path
    assert output_path.read_bytes() == b"worker-output"
    converter.close()


def test_worker_is_fast_fail_while_relaunching(tmp_path, monkeypatch):
    python_path = tmp_path / "python.exe"
    worker_script = tmp_path / "rvc_worker.py"
    model_path = tmp_path / "March-7th.pth"
    index_path = tmp_path / "March-7th.index"
    input_path = tmp_path / "input.wav"
    output_path = tmp_path / "output.wav"

    python_path.write_text("")
    worker_script.write_text("")
    model_path.write_bytes(b"fake model")
    index_path.write_bytes(b"fake index")
    input_path.write_bytes(b"fake wav")

    monkeypatch.setattr(
        rvc_provider.subprocess,
        "Popen",
        _make_scripted_popen(["block", "slow_start", "ok"], startup_delay=0.4),
    )

    converter = RVCConverter(
        model_path=model_path,
        index_path=index_path,
        backend="worker",
        python_path=python_path,
        worker_script=worker_script,
        site_packages_dir=tmp_path / ".rvc-site-packages",
        request_timeout_sec=0.5,
    )

    with pytest.raises(TimeoutError):
        converter.convert_file(input_path, output_path)

    assert converter._relaunching is True

    started = time.perf_counter()
    with pytest.raises(RuntimeError, match="relaunching|not ready"):
        converter.convert_file(input_path, output_path)
    elapsed = time.perf_counter() - started
    assert elapsed < 0.2, f"fast-fail call took {elapsed:.3f}s"

    assert _wait_until(lambda: converter._worker_ready) is True
    assert converter.convert_file(input_path, output_path) == output_path
    converter.close()
