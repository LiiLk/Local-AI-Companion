"""
Cross-platform helpers for Windows + WSL + native Linux.

The project is Windows-first for the desktop avatar, but agent workflows and
many contributors run under WSL2. Config historically used Windows-only paths
like ``venv\\Scripts\\python.exe``; this module rewrites them safely.
"""

from __future__ import annotations

import logging
import os
import platform
import shutil
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WSL_SOFTWARE_WEBGL_FLAGS = (
    "--use-gl=angle",
    "--use-angle=swiftshader-webgl",
    "--enable-unsafe-swiftshader",
)


def is_windows() -> bool:
    return os.name == "nt" or sys.platform.startswith("win")


def is_linux() -> bool:
    return sys.platform.startswith("linux")


def is_wsl() -> bool:
    """Detect Windows Subsystem for Linux (WSL1/WSL2)."""
    if not is_linux():
        return False
    # Official and common signals
    if os.environ.get("WSL_DISTRO_NAME") or os.environ.get("WSL_INTEROP"):
        return True
    try:
        release = platform.release().lower()
        if "microsoft" in release or "wsl" in release:
            return True
    except Exception:
        pass
    try:
        version = Path("/proc/version").read_text(encoding="utf-8", errors="ignore").lower()
        if "microsoft" in version or "wsl" in version:
            return True
    except Exception:
        pass
    return False


def platform_label() -> str:
    if is_windows():
        return "windows"
    if is_wsl():
        return "wsl"
    if is_linux():
        return "linux"
    return sys.platform


def normalize_path_string(value: str | Path) -> str:
    """Normalize separators; keep relative/absolute form."""
    text = str(value).strip().strip('"').strip("'")
    text = text.replace("\\", "/")
    # Strip leading ./ for cleaner joins
    if text.startswith("./"):
        text = text[2:]
    return text


def _is_windows_drive_path(text: str) -> bool:
    return len(text) >= 3 and text[0].isalpha() and text[1:3] == ":/"


def _windows_drive_path_to_posix(text: str) -> Path:
    drive = text[0].lower()
    rest = text[3:].lstrip("/")
    return Path("/mnt") / drive / rest


def resolve_project_path(
    value: str | Path | None,
    *,
    project_root: Path | None = None,
    must_exist: bool = False,
) -> Path | None:
    """Resolve a config path relative to the project root when needed."""
    if value is None:
        return None
    text = normalize_path_string(value)
    if not text or text.lower() in {"null", "none", "~"}:
        return None

    root = (project_root or PROJECT_ROOT).resolve()
    if not is_windows() and _is_windows_drive_path(text):
        path = _windows_drive_path_to_posix(text)
    else:
        path = Path(text)
        if not path.is_absolute():
            path = root / path
    path = path.resolve()

    if must_exist and not path.exists():
        return None
    return path


def _windows_venv_python_to_posix(path: Path) -> list[Path]:
    """Map ``.../Scripts/python.exe`` style paths to ``.../bin/python``."""
    text = path.as_posix()
    alts: list[Path] = []
    replacements = (
        ("/Scripts/python.exe", "/bin/python"),
        ("/Scripts/python", "/bin/python"),
        ("/scripts/python.exe", "/bin/python"),
    )
    for old, new in replacements:
        if old in text:
            alts.append(Path(text.replace(old, new)))
    # Also try python3
    for alt in list(alts):
        if alt.name == "python":
            alts.append(alt.with_name("python3"))
    return alts


def _posix_venv_python_to_windows(path: Path) -> list[Path]:
    text = path.as_posix()
    alts: list[Path] = []
    if "/bin/python" in text:
        base = text.replace("/bin/python3", "/bin/python")
        base = base.replace("/bin/python", "/Scripts/python.exe")
        alts.append(Path(base))
    return alts


def resolve_python_executable(
    configured: str | Path | None = None,
    *,
    project_root: Path | None = None,
) -> Path:
    """
    Resolve a Python executable from config, with Windows ↔ WSL/Linux mapping.

    ``None`` means the interpreter currently running the application.

    An explicit path only tries that path and its cross-platform venv twin
    (``Scripts/python.exe`` ↔ ``bin/python``). If neither exists, the expected
    path for the current platform is returned so the provider can fail with a
    useful error instead of silently launching an unrelated environment.
    """
    root = (project_root or PROJECT_ROOT).resolve()
    if configured in (None, "", "null", "None"):
        return Path(sys.executable).absolute()

    text = normalize_path_string(configured)
    if not is_windows() and _is_windows_drive_path(text):
        path = _windows_drive_path_to_posix(text)
    else:
        path = Path(text)
        if not path.is_absolute():
            path = root / path

    mapped_candidates = (
        _posix_venv_python_to_windows(path)
        if is_windows()
        else _windows_venv_python_to_posix(path)
    )
    # On the opposite platform, prefer the native venv twin even when both
    # layouts exist in the checkout.
    candidates = [*mapped_candidates, path]

    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        try:
            # Use absolute() (not resolve()) so venv/bin/python keeps its path
            # when it is a symlink into /usr/bin — workers must see the venv.
            resolved = candidate.expanduser().absolute()
            if resolved.is_file() or (resolved.is_symlink() and resolved.exists()):
                return resolved
        except OSError:
            continue

    preferred_missing = mapped_candidates[0] if mapped_candidates else path
    return preferred_missing.expanduser().absolute()


def resolve_worker_script(
    configured: str | Path | None,
    default: Path,
    *,
    project_root: Path | None = None,
) -> Path:
    """Resolve a worker script path without masking an invalid explicit path."""
    if configured in (None, "", "null", "None"):
        return default.resolve()
    path = resolve_project_path(configured, project_root=project_root)
    if path is not None:
        return path
    return default.resolve()


def kill_process_tree(process: subprocess.Popen[Any], *, timeout: float = 5.0) -> None:
    """Terminate a worker process and its children (Windows taskkill / POSIX group)."""
    if process.poll() is not None:
        return

    pid = int(process.pid)
    if pid <= 0:
        return

    if is_windows():
        system_root = Path(os.environ.get("SystemRoot", "C:/Windows"))
        taskkill = system_root / "System32" / "taskkill.exe"
        taskkill_exe = str(taskkill.resolve()) if taskkill.is_file() else "taskkill"
        command = [taskkill_exe, "/PID", str(pid), "/T", "/F"]
        try:
            result = subprocess.run(  # nosec B603
                command,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=10,
                check=False,
                shell=False,
            )
            if result.returncode == 0:
                try:
                    process.wait(timeout=timeout)
                except Exception:
                    pass
                return
            logger.debug(
                "taskkill returned %s for pid=%s: %s",
                result.returncode,
                pid,
                result.stderr.strip(),
            )
        except Exception:
            logger.debug("taskkill failed for pid=%s", pid, exc_info=True)

        try:
            process.terminate()
            process.wait(timeout=timeout)
            return
        except Exception:
            try:
                process.kill()
                process.wait(timeout=timeout)
            except Exception:
                logger.debug("Failed to terminate Windows process pid=%s", pid, exc_info=True)
            return

    # POSIX: try process group first (workers started with start_new_session)
    try:
        os.killpg(pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError, OSError):
        try:
            process.terminate()
        except Exception:
            pass
    try:
        process.wait(timeout=timeout)
        return
    except Exception:
        pass
    try:
        os.killpg(pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        try:
            process.kill()
        except Exception:
            pass
    try:
        process.wait(timeout=timeout)
    except Exception:
        logger.debug("Failed to fully kill process pid=%s", pid, exc_info=True)


def ensure_wsl_audio_env() -> dict[str, str]:
    """
    Best-effort PulseAudio/WSLg env for sounddevice under WSL.

    Returns the env keys that were set (for logging). Does not override
    an already configured PULSE_SERVER.
    """
    applied: dict[str, str] = {}
    if not is_wsl():
        return applied

    pulse = Path("/mnt/wslg/PulseServer")
    if "PULSE_SERVER" not in os.environ and pulse.exists():
        value = f"unix:{pulse}"
        os.environ["PULSE_SERVER"] = value
        applied["PULSE_SERVER"] = value

    # WSLg display is usually already set; only fill if missing
    if not os.environ.get("DISPLAY") and Path("/tmp/.X11-unix").exists():
        os.environ["DISPLAY"] = ":0"
        applied["DISPLAY"] = ":0"

    return applied


def ensure_wsl_desktop_env() -> dict[str, str]:
    """Configure deterministic WSLg audio and local software WebGL defaults."""
    applied = ensure_wsl_audio_env()
    if not is_wsl():
        return applied

    webgl_mode = os.environ.get("LOCAL_AI_WSL_WEBGL_MODE", "software").strip().lower()
    if webgl_mode in {"auto", "native", "off", "0", "false"}:
        return applied

    # QtWebEngine's native EGL probing is unreliable under WSLg on systems
    # without a DRM render node. The bundled Live2D page is local/trusted, so
    # prefer Chromium's deterministic CPU WebGL path unless the user supplied
    # explicit Chromium flags or opted out above.
    if not os.environ.get("QTWEBENGINE_CHROMIUM_FLAGS"):
        flags = " ".join(WSL_SOFTWARE_WEBGL_FLAGS)
        os.environ["QTWEBENGINE_CHROMIUM_FLAGS"] = flags
        applied["QTWEBENGINE_CHROMIUM_FLAGS"] = flags

    for key, value in (
        ("QT_OPENGL", "software"),
        ("LIBGL_ALWAYS_SOFTWARE", "1"),
        # Prevent WSLg + Qt double-scaling that detaches the native HUD from the
        # Live2D webview (character and mute/stop/chat bar looking "dislocated").
        ("QT_AUTO_SCREEN_SCALE_FACTOR", "0"),
        ("QT_ENABLE_HIGHDPI_SCALING", "0"),
        ("QT_SCALE_FACTOR", "1"),
    ):
        if not os.environ.get(key):
            os.environ[key] = value
            applied[key] = value

    return applied


def probe_wsl_environment() -> dict[str, Any]:
    """Return JSON-safe WSL/WSLg host integration diagnostics."""
    return {
        "is_wsl": is_wsl(),
        "distro": os.environ.get("WSL_DISTRO_NAME"),
        "kernel_release": platform.release(),
        "pulse_server": os.environ.get("PULSE_SERVER"),
        "pulse_socket_exists": Path("/mnt/wslg/PulseServer").exists(),
        "display": os.environ.get("DISPLAY"),
        "wayland_display": os.environ.get("WAYLAND_DISPLAY"),
        "nvidia_smi": shutil.which("nvidia-smi"),
        "webgl_mode": os.environ.get("LOCAL_AI_WSL_WEBGL_MODE", "software"),
        "qtwebengine_chromium_flags": os.environ.get("QTWEBENGINE_CHROMIUM_FLAGS"),
        "qt_opengl": os.environ.get("QT_OPENGL"),
        "libgl_always_software": os.environ.get("LIBGL_ALWAYS_SOFTWARE"),
    }


def probe_audio_devices() -> dict[str, Any]:
    """Return a small diagnostic dict for input/output device availability."""
    info: dict[str, Any] = {
        "platform": platform_label(),
        "input_devices": 0,
        "output_devices": 0,
        "default_device": None,
        "error": None,
        "hint": None,
    }
    try:
        import sounddevice as sd

        devices = sd.query_devices()
        default_device = sd.default.device
        try:
            info["default_device"] = [int(value) for value in default_device]
        except (TypeError, ValueError):
            info["default_device"] = default_device
        if isinstance(devices, dict):
            devices = [devices]
        inputs = [d for d in devices if int(d.get("max_input_channels", 0) or 0) > 0]
        outputs = [d for d in devices if int(d.get("max_output_channels", 0) or 0) > 0]
        info["input_devices"] = len(inputs)
        info["output_devices"] = len(outputs)
        info["device_names"] = [d.get("name") for d in devices[:12]]
    except Exception as exc:  # noqa: BLE001
        info["error"] = str(exc)

    if is_wsl() and info["input_devices"] == 0:
        wsl = probe_wsl_environment()
        if not wsl["pulse_socket_exists"]:
            info["hint"] = (
                "WSLg PulseAudio socket is missing. Run `wsl --update` from an "
                "Administrator PowerShell, then `wsl --shutdown` and restart the distro."
            )
        elif not wsl["pulse_server"]:
            info["hint"] = (
                "WSLg is present but PULSE_SERVER is missing. Restart WSL, or set "
                "PULSE_SERVER=unix:/mnt/wslg/PulseServer for this shell."
            )
        else:
            info["hint"] = (
                "WSLg audio is present, but PortAudio exposes no input device. Verify "
                "Windows microphone privacy permissions, install the packages from "
                "scripts/setup_wsl.sh, then run `wsl --shutdown` and retry."
            )
    return info


def probe_cuda() -> dict[str, Any]:
    info: dict[str, Any] = {
        "platform": platform_label(),
        "torch_available": False,
        "cuda_available": False,
        "device_name": None,
        "error": None,
        "hint": None,
    }
    try:
        import torch

        info["torch_available"] = True
        info["cuda_available"] = bool(torch.cuda.is_available())
        if info["cuda_available"]:
            info["device_name"] = torch.cuda.get_device_name(0)
    except Exception as exc:  # noqa: BLE001
        info["error"] = str(exc)
    if is_wsl() and not info["cuda_available"]:
        if shutil.which("nvidia-smi"):
            info["hint"] = (
                "The WSL NVIDIA bridge is visible, but this PyTorch build has no CUDA. "
                "Install a CUDA-enabled PyTorch build in the project venv."
            )
        else:
            info["hint"] = (
                "Install or update the NVIDIA driver on Windows, then run `wsl --update`. "
                "Do not install a Linux NVIDIA display driver inside WSL."
            )
    return info
