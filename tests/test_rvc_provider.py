"""Tests for the local RVC wrapper."""

import json
import time
import sys
import types
from pathlib import Path

import pytest
import soundfile as sf

from src.tts import rvc_provider
from src.tts.rvc_provider import RVCConverter


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

    assert converter._worker_process is None


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
        request_timeout_sec=0.01,
    )

    with pytest.raises(TimeoutError, match="startup timed out"):
        converter.convert_file(input_path, output_path)

    assert converter._worker_process is None
