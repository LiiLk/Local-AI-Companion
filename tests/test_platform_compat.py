"""Unit tests for LIL-49 cross-platform helpers."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from src.utils import platform_compat


def _executable(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")
    path.chmod(0o755)
    return path


def test_normalize_path_string_windows_separators():
    assert platform_compat.normalize_path_string(".\\venv\\Scripts\\python.exe") == (
        "venv/Scripts/python.exe"
    )
    assert platform_compat.normalize_path_string("./scripts/rvc_worker.py") == (
        "scripts/rvc_worker.py"
    )


def test_resolve_python_null_uses_current_interpreter(tmp_path):
    _executable(tmp_path / "venv" / "bin" / "python")

    resolved = platform_compat.resolve_python_executable(None, project_root=tmp_path)

    assert resolved == Path(sys.executable).absolute()


def test_resolve_python_maps_windows_venv_to_posix(tmp_path, monkeypatch):
    monkeypatch.setattr(platform_compat, "is_windows", lambda: False)
    linux_python = _executable(tmp_path / ".venv-worker" / "bin" / "python")

    resolved = platform_compat.resolve_python_executable(
        ".venv-worker/Scripts/python.exe",
        project_root=tmp_path,
    )

    assert resolved == linux_python.absolute()


def test_resolve_python_maps_posix_venv_to_windows(tmp_path, monkeypatch):
    monkeypatch.setattr(platform_compat, "is_windows", lambda: True)
    windows_python = _executable(tmp_path / ".venv-worker" / "Scripts" / "python.exe")

    resolved = platform_compat.resolve_python_executable(
        ".venv-worker/bin/python",
        project_root=tmp_path,
    )

    assert resolved == windows_python.absolute()


def test_explicit_missing_python_does_not_fall_back(tmp_path, monkeypatch):
    monkeypatch.setattr(platform_compat, "is_windows", lambda: False)
    _executable(tmp_path / "venv" / "bin" / "python")

    resolved = platform_compat.resolve_python_executable(
        ".venv-qwen/Scripts/python.exe",
        project_root=tmp_path,
    )

    assert resolved == (tmp_path / ".venv-qwen" / "bin" / "python").absolute()
    assert not resolved.exists()


def test_resolve_worker_script_accepts_windows_style_relative(tmp_path):
    worker = tmp_path / "scripts" / "rvc_worker.py"
    worker.parent.mkdir(parents=True)
    worker.write_text("", encoding="utf-8")

    resolved = platform_compat.resolve_worker_script(
        ".\\scripts\\rvc_worker.py",
        worker,
        project_root=tmp_path,
    )

    assert resolved == worker.resolve()


def test_explicit_missing_worker_is_not_replaced_by_default(tmp_path):
    default = tmp_path / "scripts" / "default_worker.py"
    default.parent.mkdir(parents=True)
    default.write_text("", encoding="utf-8")

    resolved = platform_compat.resolve_worker_script(
        "scripts/custom_worker.py",
        default,
        project_root=tmp_path,
    )

    assert resolved == (tmp_path / "scripts" / "custom_worker.py").resolve()
    assert not resolved.exists()


def test_platform_label_uses_specific_platform(monkeypatch):
    monkeypatch.setattr(platform_compat, "is_windows", lambda: False)
    monkeypatch.setattr(platform_compat, "is_wsl", lambda: True)
    monkeypatch.setattr(platform_compat, "is_linux", lambda: True)
    assert platform_compat.platform_label() == "wsl"

    monkeypatch.setattr(platform_compat, "is_windows", lambda: True)
    assert platform_compat.platform_label() == "windows"


def test_audio_probe_is_json_serializable(monkeypatch):
    fake_sounddevice = SimpleNamespace(
        default=SimpleNamespace(device=(-1, 3)),
        query_devices=lambda: [
            {"name": "Microphone", "max_input_channels": 1, "max_output_channels": 0},
            {"name": "Speakers", "max_input_channels": 0, "max_output_channels": 2},
        ],
    )
    monkeypatch.setitem(sys.modules, "sounddevice", fake_sounddevice)

    report = platform_compat.probe_audio_devices()

    assert report["default_device"] == [-1, 3]
    assert report["input_devices"] == 1
    assert report["output_devices"] == 1
    json.dumps(report)


def test_wsl_probe_is_json_serializable(monkeypatch):
    monkeypatch.setattr(platform_compat, "is_wsl", lambda: True)
    monkeypatch.setenv("WSL_DISTRO_NAME", "Ubuntu")
    monkeypatch.setenv("PULSE_SERVER", "unix:/mnt/wslg/PulseServer")

    report = platform_compat.probe_wsl_environment()

    assert report["is_wsl"] is True
    assert report["distro"] == "Ubuntu"
    json.dumps(report)


def test_wsl_desktop_env_enables_software_webgl_by_default(monkeypatch):
    monkeypatch.setattr(platform_compat, "is_wsl", lambda: True)
    monkeypatch.setattr(
        platform_compat,
        "ensure_wsl_audio_env",
        lambda: {"PULSE_SERVER": "unix:/mnt/wslg/PulseServer"},
    )
    monkeypatch.delenv("LOCAL_AI_WSL_WEBGL_MODE", raising=False)
    monkeypatch.delenv("QTWEBENGINE_CHROMIUM_FLAGS", raising=False)
    monkeypatch.delenv("QT_OPENGL", raising=False)
    monkeypatch.delenv("LIBGL_ALWAYS_SOFTWARE", raising=False)
    monkeypatch.delenv("QT_SCALE_FACTOR", raising=False)
    monkeypatch.delenv("QT_AUTO_SCREEN_SCALE_FACTOR", raising=False)
    monkeypatch.delenv("QT_ENABLE_HIGHDPI_SCALING", raising=False)

    applied = platform_compat.ensure_wsl_desktop_env()

    flags = " ".join(platform_compat.WSL_SOFTWARE_WEBGL_FLAGS)
    assert applied["PULSE_SERVER"] == "unix:/mnt/wslg/PulseServer"
    assert applied["QTWEBENGINE_CHROMIUM_FLAGS"] == flags
    assert applied["QT_OPENGL"] == "software"
    assert applied["LIBGL_ALWAYS_SOFTWARE"] == "1"
    assert applied["QT_SCALE_FACTOR"] == "1"
    assert applied["QT_AUTO_SCREEN_SCALE_FACTOR"] == "0"
    assert applied["QT_ENABLE_HIGHDPI_SCALING"] == "0"
    assert platform_compat.os.environ["QTWEBENGINE_CHROMIUM_FLAGS"] == flags


def test_wsl_desktop_env_respects_explicit_chromium_flags(monkeypatch):
    monkeypatch.setattr(platform_compat, "is_wsl", lambda: True)
    monkeypatch.setattr(platform_compat, "ensure_wsl_audio_env", lambda: {})
    monkeypatch.setenv("QTWEBENGINE_CHROMIUM_FLAGS", "--use-gl=desktop")

    applied = platform_compat.ensure_wsl_desktop_env()

    assert "QTWEBENGINE_CHROMIUM_FLAGS" not in applied
    assert platform_compat.os.environ["QTWEBENGINE_CHROMIUM_FLAGS"] == "--use-gl=desktop"


def test_wsl_desktop_env_can_opt_out_of_software_webgl(monkeypatch):
    monkeypatch.setattr(platform_compat, "is_wsl", lambda: True)
    monkeypatch.setattr(platform_compat, "ensure_wsl_audio_env", lambda: {})
    monkeypatch.setenv("LOCAL_AI_WSL_WEBGL_MODE", "auto")
    monkeypatch.delenv("QTWEBENGINE_CHROMIUM_FLAGS", raising=False)

    applied = platform_compat.ensure_wsl_desktop_env()

    assert "QTWEBENGINE_CHROMIUM_FLAGS" not in applied


class _FakeProcess:
    def __init__(self, pid: int = 1234):
        self.pid = pid
        self.wait_calls: list[float] = []
        self.terminated = False
        self.killed = False

    def poll(self):
        return None

    def wait(self, timeout):
        self.wait_calls.append(timeout)
        return 0

    def terminate(self):
        self.terminated = True

    def kill(self):
        self.killed = True


def test_kill_process_tree_uses_posix_process_group(monkeypatch):
    process = _FakeProcess()
    signals: list[tuple[int, int]] = []
    monkeypatch.setattr(platform_compat, "is_windows", lambda: False)
    monkeypatch.setattr(
        platform_compat.os,
        "killpg",
        lambda pid, sig: signals.append((pid, sig)),
    )

    platform_compat.kill_process_tree(process)

    assert signals == [(process.pid, platform_compat.signal.SIGTERM)]
    assert process.wait_calls == [5.0]


def test_kill_process_tree_uses_taskkill_on_windows(monkeypatch):
    process = _FakeProcess()
    commands: list[list[str]] = []
    monkeypatch.setattr(platform_compat, "is_windows", lambda: True)

    def fake_run(command, **_kwargs):
        commands.append(command)
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(platform_compat.subprocess, "run", fake_run)

    platform_compat.kill_process_tree(process)

    assert commands[0][-4:] == ["/PID", str(process.pid), "/T", "/F"]
    assert process.wait_calls == [5.0]
