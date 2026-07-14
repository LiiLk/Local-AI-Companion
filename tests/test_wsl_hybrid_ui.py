"""Unit tests for WSL hybrid pet helpers (LIL-49 option A)."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.utils.wsl_hybrid_ui import (
    find_windows_checkout,
    find_windows_python,
    sync_pet_shell_to_windows_checkout,
    windows_path_to_wsl,
    wsl_path_to_windows,
)


def test_wsl_path_to_windows_mnt_style(monkeypatch):
    # Force fallback conversion without calling wslpath
    monkeypatch.setattr("src.utils.wsl_hybrid_ui.shutil.which", lambda name: None)
    win = wsl_path_to_windows(Path("/mnt/c/Users/Khalil/Documents/Local-AI-Companion"))
    assert win.lower().startswith("c:")
    assert "Local-AI-Companion" in win


def test_wsl_path_to_windows_home_fallback(monkeypatch):
    monkeypatch.setattr("src.utils.wsl_hybrid_ui.shutil.which", lambda name: None)
    monkeypatch.setenv("WSL_DISTRO_NAME", "Ubuntu")
    win = wsl_path_to_windows(Path("/home/khalil/Local-AI-Companion"))
    assert "Ubuntu" in win
    assert "Local-AI-Companion" in win.replace("/", "\\") or "Local-AI-Companion" in win


def test_windows_path_to_wsl_drive_fallback(monkeypatch):
    monkeypatch.setattr("src.utils.wsl_hybrid_ui.is_wsl", lambda: True)
    monkeypatch.setattr("src.utils.wsl_hybrid_ui.shutil.which", lambda name: None)

    path = windows_path_to_wsl(r"C:\Users\Ada\Local-AI-Companion")

    assert path == Path("/mnt/c/Users/Ada/Local-AI-Companion")


def test_find_windows_checkout_honors_env(tmp_path, monkeypatch):
    html = tmp_path / "frontend" / "live2d" / "index.html"
    html.parent.mkdir(parents=True)
    html.write_text("<html></html>", encoding="utf-8")
    monkeypatch.setenv("LOCAL_AI_WINDOWS_CHECKOUT", str(tmp_path))
    found = find_windows_checkout()
    assert found == tmp_path


def test_find_windows_python_honors_env(tmp_path, monkeypatch):
    fake = tmp_path / "python.exe"
    fake.write_text("", encoding="utf-8")
    monkeypatch.setenv("LOCAL_AI_WINDOWS_PYTHON", str(fake))
    # Skip the expensive PyQt probe when env path exists
    monkeypatch.setattr(
        "src.utils.wsl_hybrid_ui._windows_python_has_pyqt",
        lambda path: True,
    )
    found = find_windows_python()
    assert found == fake


def test_find_windows_python_accepts_windows_style_env(tmp_path, monkeypatch):
    fake = tmp_path / "python.exe"
    fake.write_text("", encoding="utf-8")
    monkeypatch.setenv("LOCAL_AI_WINDOWS_PYTHON", r"C:\project\venv\Scripts\python.exe")
    monkeypatch.setattr(
        "src.utils.wsl_hybrid_ui.windows_path_to_wsl",
        lambda value: fake,
    )

    assert find_windows_python() == fake


def test_sync_pet_shell_copies_files(tmp_path, monkeypatch):
    # Point PROJECT_ROOT at a temp tree with shell sources
    src_root = tmp_path / "wsl_repo"
    for rel in (
        "scripts/windows_pet_shell.py",
        "src/desktop/bridge_proxy.py",
        "desktop/qt_avatar_shell.py",
        "frontend/live2d/index.html",
        "frontend/live2d/desktop-bridge.js",
        "frontend/live2d/live2d.js",
    ):
        path = src_root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# {rel}\n", encoding="utf-8")

    runtime_asset = (
        src_root
        / "frontend"
        / "live2d"
        / "runtime-assets"
        / "live2d_sdk_web"
        / "Core"
        / "live2dcubismcore.min.js"
    )
    runtime_asset.parent.mkdir(parents=True, exist_ok=True)
    runtime_asset.write_text("// cubism core\n", encoding="utf-8")

    win = tmp_path / "win_repo"
    win.mkdir()
    monkeypatch.setattr("src.utils.wsl_hybrid_ui.PROJECT_ROOT", src_root)
    sync_pet_shell_to_windows_checkout(win)
    assert (win / "scripts" / "windows_pet_shell.py").is_file()
    assert (win / "src" / "desktop" / "bridge_proxy.py").is_file()
    assert (win / "desktop" / "qt_avatar_shell.py").is_file()
    assert (win / "frontend" / "live2d" / "index.html").is_file()
    assert (
        win
        / "frontend"
        / "live2d"
        / "runtime-assets"
        / "live2d_sdk_web"
        / "Core"
        / "live2dcubismcore.min.js"
    ).is_file()


def test_sync_pet_shell_fails_when_required_source_is_missing(tmp_path, monkeypatch):
    src_root = tmp_path / "incomplete_repo"
    src_root.mkdir()
    win = tmp_path / "win_repo"
    win.mkdir()
    monkeypatch.setattr("src.utils.wsl_hybrid_ui.PROJECT_ROOT", src_root)

    with pytest.raises(RuntimeError, match="missing source"):
        sync_pet_shell_to_windows_checkout(win)
