"""
LIL-49 option A — Hybrid: WSL backend + Windows-native desktop pet shell.

Microsoft WSL interop launches Windows executables from Linux:
https://learn.microsoft.com/en-us/windows/wsl/filesystems#run-windows-tools-from-linux

Localhost forwarding lets the Windows shell reach the WSL bridge:
https://learn.microsoft.com/en-us/windows/wsl/networking

Architecture:
  WSL:   pipeline + bridge  ws://127.0.0.1:8765
  Windows Qt pet shell: transparent / topmost / click-through, talks to bridge
"""

from __future__ import annotations

import contextlib
import logging
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Optional

from src.utils.platform_compat import PROJECT_ROOT, is_wsl

logger = logging.getLogger(__name__)

DEFAULT_BRIDGE_PORT = 8765

# Files the Windows shell needs when the active code lives under the WSL home tree.
_SYNC_REL_PATHS = (
    "scripts/windows_pet_shell.py",
    "src/desktop/bridge_proxy.py",
    "desktop/qt_avatar_shell.py",
    "frontend/live2d/index.html",
    "frontend/live2d/desktop-bridge.js",
    "frontend/live2d/live2d.js",
)
_SYNC_REL_DIRS = (
    "frontend/live2d/runtime-assets",
)


def _which_windows(*names: str) -> Optional[str]:
    for name in names:
        path = shutil.which(name)
        if path:
            return path
    return None


def wsl_path_to_windows(path: Path) -> str:
    """Convert a WSL/Linux path to a Windows path via wslpath."""
    path = path.resolve()
    wslpath = shutil.which("wslpath")
    if wslpath:
        try:
            out = subprocess.check_output(  # nosec B603
                [wslpath, "-w", str(path)],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
            if out:
                return out
        except Exception:
            pass
    text = str(path)
    if text.startswith("/mnt/") and len(text) > 5 and text[5].isalpha():
        drive = text[5].upper()
        rest = text[6:].replace("/", "\\")
        return f"{drive}:{rest}"
    # \\wsl$\Distro\home\...
    distro = os.environ.get("WSL_DISTRO_NAME") or "Ubuntu"
    return "\\\\wsl$\\" + distro + text.replace("/", "\\")


def windows_path_to_wsl(value: str | Path) -> Path:
    """Accept either Windows (C:\\...) or WSL (/mnt/c/...) path input."""
    text = str(value).strip().strip('"').strip("'")
    if not is_wsl() or not re.match(r"^[A-Za-z]:[\\/]", text):
        return Path(text)

    wslpath = shutil.which("wslpath")
    if wslpath:
        try:
            converted = subprocess.check_output(  # nosec B603
                [wslpath, "-u", text],
                text=True,
                stderr=subprocess.DEVNULL,
                timeout=5,
            ).strip()
            if converted:
                return Path(converted)
        except Exception:
            pass

    drive = text[0].lower()
    remainder = text[2:].replace("\\", "/").lstrip("/")
    return Path("/mnt") / drive / remainder


def _windows_user_candidates() -> list[str]:
    names: list[str] = []
    for key in ("LOCAL_AI_WINDOWS_USER", "WIN_USER", "USER", "USERNAME"):
        val = (os.environ.get(key) or "").strip()
        if val and val not in names:
            names.append(val)
    users_root = Path("/mnt/c/Users")
    if users_root.is_dir():
        try:
            for candidate in sorted(users_root.iterdir(), key=lambda path: path.name.lower()):
                if candidate.is_dir() and candidate.name not in names:
                    names.append(candidate.name)
        except OSError:
            pass
    return names


def _candidate_windows_checkouts() -> list[Path]:
    """Likely Windows-drive clones of this repo (NTFS is more reliable for Qt)."""
    env = os.environ.get("LOCAL_AI_WINDOWS_CHECKOUT", "").strip()
    out: list[Path] = []
    if env:
        out.append(windows_path_to_wsl(env))
    for user in _windows_user_candidates():
        out.append(Path("/mnt/c/Users") / user / "Documents" / "Local-AI-Companion")
        out.append(Path("/mnt/c/Users") / user / "Local-AI-Companion")
    # Dedup while preserving order
    seen: set[str] = set()
    unique: list[Path] = []
    for path in out:
        key = str(path)
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return unique


def find_windows_checkout() -> Optional[Path]:
    for path in _candidate_windows_checkouts():
        if (path / "frontend" / "live2d" / "index.html").is_file() or (
            path / "desktop" / "qt_avatar_shell.py"
        ).is_file():
            return path
    return None


def find_windows_python() -> Optional[Path]:
    """Locate a Windows CPython usable for the pet shell (PyQt6)."""
    env = os.environ.get("LOCAL_AI_WINDOWS_PYTHON", "").strip()
    if env:
        candidate = windows_path_to_wsl(env)
        if candidate.is_file():
            return candidate

    candidates: list[Path] = []

    for checkout in _candidate_windows_checkouts():
        candidates.append(checkout / "venv" / "Scripts" / "python.exe")

    # py launcher
    py = _which_windows("py.exe", "py")
    if py:
        try:
            out = subprocess.check_output(  # nosec B603
                [py, "-3.12", "-c", "import sys; print(sys.executable)"],
                text=True,
                stderr=subprocess.DEVNULL,
                timeout=8,
            ).strip()
            if out:
                candidates.insert(0, windows_path_to_wsl(out))
        except Exception:
            pass

    which_py = _which_windows("python.exe")
    if which_py and "/WindowsApps/" not in which_py.replace("\\", "/"):
        candidates.append(Path(which_py))

    for path in candidates:
        try:
            if path.is_file() and _windows_python_has_pyqt(path):
                return path
        except Exception:
            continue

    # Return first existing python even without PyQt (caller shows install hint)
    for path in candidates:
        if path.is_file():
            return path
    return None


def _windows_python_has_pyqt(python: Path) -> bool:
    try:
        subprocess.check_call(  # nosec B603
            [
                str(python),
                "-c",
                "from PyQt6.QtWidgets import QApplication; from PyQt6.QtWebEngineWidgets import QWebEngineView",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=20,
        )
        return True
    except Exception:
        return False


def sync_pet_shell_to_windows_checkout(win_checkout: Path) -> None:
    """Copy hybrid shell sources from the active WSL tree into the Windows checkout."""
    failures: list[str] = []
    for rel in _SYNC_REL_PATHS:
        src = PROJECT_ROOT / rel
        dst = win_checkout / rel
        if not src.is_file():
            failures.append(f"missing source: {src}")
            continue
        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
        except Exception as exc:
            failures.append(f"{src} -> {dst}: {exc}")

    for rel in _SYNC_REL_DIRS:
        src = PROJECT_ROOT / rel
        dst = win_checkout / rel
        if not src.is_dir():
            failures.append(f"missing source directory: {src}")
            continue
        try:
            # Merge runtime assets into an existing Windows checkout instead of
            # replacing the directory. Licensed/local model packs may only live
            # in the Windows tree, while the WSL checkout may contain just SDK
            # runtime files; deleting the destination would remove those assets.
            shutil.copytree(src, dst, dirs_exist_ok=True)
        except Exception as exc:
            failures.append(f"{src} -> {dst}: {exc}")

    if failures:
        raise RuntimeError("Windows pet shell sync failed: " + "; ".join(failures))


def launch_windows_pet_shell(
    *,
    bridge_port: int = DEFAULT_BRIDGE_PORT,
    bridge_url: str | None = None,
    page_url: str | None = None,
) -> subprocess.Popen[Any]:
    """
    Start scripts/windows_pet_shell.py with Windows Python (native pet overlay).
    """
    python = find_windows_python()
    if python is None:
        raise FileNotFoundError(
            "No Windows Python found for the pet shell. Install Python 3.12 on Windows, "
            "create a venv with PyQt6+PyQt6-WebEngine, then set LOCAL_AI_WINDOWS_PYTHON "
            "to that python.exe path."
        )

    if not _windows_python_has_pyqt(python):
        raise RuntimeError(
            f"Windows Python at {python} is missing PyQt6/WebEngine. "
            f'Run: "{python}" -m pip install PyQt6 PyQt6-WebEngine websockets'
        )

    script = PROJECT_ROOT / "scripts" / "windows_pet_shell.py"
    if not script.is_file():
        raise FileNotFoundError(f"Missing {script}")

    # Prefer a Windows-drive checkout so Windows Python loads sources from NTFS
    # (more reliable than \\wsl$\). Fall back to wslpath of the WSL tree.
    win_checkout = find_windows_checkout()
    if win_checkout is not None:
        sync_pet_shell_to_windows_checkout(win_checkout)
        linux_root = win_checkout.resolve()
        script_path = win_checkout / "scripts" / "windows_pet_shell.py"
    else:
        linux_root = PROJECT_ROOT.resolve()
        script_path = script

    # Windows Python needs Windows-style paths for argv / PYTHONPATH.
    # Popen cwd from WSL must stay a Linux path (/mnt/c/...), not C:\...
    script_arg = wsl_path_to_windows(script_path) if is_wsl() else str(script_path)
    win_root = wsl_path_to_windows(linux_root) if is_wsl() else str(linux_root)
    popen_cwd = str(linux_root) if linux_root.is_dir() else None

    url = bridge_url or f"ws://127.0.0.1:{int(bridge_port)}"
    cmd = [
        str(python),
        script_arg,
        "--bridge-url",
        url,
    ]
    if page_url:
        cmd.extend(["--page-url", page_url])

    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "windows_pet_shell.log"

    logger.info("Launching Windows pet shell: %s", " ".join(cmd[:4]))
    logger.info("Pet shell log: %s", log_path)

    env = os.environ.copy()
    # Windows uses ; as pathsep for PYTHONPATH
    env["PYTHONPATH"] = win_root

    log_fh = open(log_path, "ab", buffering=0)  # noqa: SIM115 — kept open for child lifetime
    try:
        proc = subprocess.Popen(  # nosec B603
            cmd,
            cwd=popen_cwd,
            env=env,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    except Exception:
        log_fh.close()
        raise

    # Attach handle so GC does not close the fd while the child runs
    proc._pet_log_fh = log_fh  # type: ignore[attr-defined]

    # Detect immediate crash (missing module, bad path, …)
    time.sleep(0.6)
    if proc.poll() is not None:
        tail = ""
        try:
            tail = log_path.read_text(encoding="utf-8", errors="replace")[-2000:]
        except Exception:
            pass
        raise RuntimeError(
            f"Windows pet shell exited immediately (code={proc.returncode}). "
            f"See {log_path}. Tail:\n{tail}"
        )

    return proc


class HybridUiHandle:
    """Tracks the Windows pet process (no static HTTP server required for file://)."""

    def __init__(self, process: subprocess.Popen[Any]):
        self.process = process

    def stop(self) -> None:
        if self.process.poll() is None:
            with contextlib.suppress(Exception):
                self.process.terminate()
            try:
                self.process.wait(timeout=3)
            except Exception:
                with contextlib.suppress(Exception):
                    self.process.kill()
        log_fh = getattr(self.process, "_pet_log_fh", None)
        if log_fh is not None:
            with contextlib.suppress(Exception):
                log_fh.close()


def start_hybrid_windows_ui(
    *,
    bridge_port: int = DEFAULT_BRIDGE_PORT,
    ui_port: int | None = None,  # unused (kept for API compat)
    live2d_dir: Path | None = None,  # unused
) -> HybridUiHandle:
    """
    After the WSL bridge is listening, start the Windows-native pet shell.

    Falls back to opening Edge app-mode only if Windows Python/PyQt is missing.
    """
    del ui_port, live2d_dir
    time.sleep(0.2)
    try:
        proc = launch_windows_pet_shell(bridge_port=bridge_port)
        logger.info(
            "Hybrid pet: Windows Qt shell launched (pid=%s). "
            "Backend remains in WSL on port %s.",
            proc.pid,
            bridge_port,
        )
        return HybridUiHandle(proc)
    except Exception as exc:
        logger.warning(
            "Windows pet shell launch failed (%s); falling back to browser app mode",
            exc,
        )
        return _fallback_browser(bridge_port=bridge_port, error=exc)


def _fallback_browser(*, bridge_port: int, error: Exception) -> HybridUiHandle:
    """Last resort: Edge/Chrome app window (not true desktop-pet transparency)."""
    from functools import partial
    from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
    import socket
    import threading

    def port_free(port: int) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind(("127.0.0.1", port))
                return True
            except OSError:
                return False

    port = 8766
    for p in range(8766, 8786):
        if port_free(p):
            port = p
            break

    directory = (PROJECT_ROOT / "frontend" / "live2d").resolve()
    handler = partial(SimpleHTTPRequestHandler, directory=str(directory))
    httpd = ThreadingHTTPServer(("127.0.0.1", port), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True, name="live2d-static-fallback")
    thread.start()
    url = f"http://127.0.0.1:{port}/index.html?backendPort={int(bridge_port)}"
    logger.error(
        "Could not start Windows Qt pet shell: %s. "
        "Opening browser fallback (not desktop-incrusted): %s. "
        "Install PyQt6 on Windows Python and set LOCAL_AI_WINDOWS_PYTHON.",
        error,
        url,
    )
    edge = _which_windows("msedge.exe", "msedge")
    ps = _which_windows("powershell.exe")
    if edge:
        subprocess.Popen([edge, f"--app={url}"], start_new_session=True)  # nosec B603
    elif ps:
        subprocess.Popen(  # nosec B603
            [ps, "-NoProfile", "-Command", f"Start-Process '{url}'"],
            start_new_session=True,
        )

    class _Fallback:
        process = None

        def stop(self) -> None:
            with contextlib.suppress(Exception):
                httpd.shutdown()

    return _Fallback()  # type: ignore[return-value]
