#!/usr/bin/env python3
"""LIL-49 — Platform diagnostic for Windows / WSL / Linux.

Usage (repo root):
    python scripts/check_platform.py
    python scripts/check_platform.py --json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.platform_compat import (
    ensure_wsl_audio_env,
    platform_label,
    probe_audio_devices,
    probe_cuda,
    probe_wsl_environment,
    resolve_python_executable,
    resolve_worker_script,
    resolve_project_path,
)
from src.utils.character_loader import resolve_character_config, resolve_live2d_model_config
from src.utils.config_loader import load_yaml_config


def main() -> int:
    parser = argparse.ArgumentParser(description="Local AI Companion platform check (LIL-49)")
    parser.add_argument("--json", action="store_true", help="Machine-readable output")
    args = parser.parse_args()

    applied = ensure_wsl_audio_env()
    cuda = probe_cuda()
    audio = probe_audio_devices()
    wsl = probe_wsl_environment()
    config = resolve_character_config(load_yaml_config(PROJECT_ROOT / "config" / "config.yaml"))

    legacy_win_python = ".\\venv\\Scripts\\python.exe"
    resolved_python = resolve_python_executable(legacy_win_python, project_root=PROJECT_ROOT)
    worker = resolve_worker_script(
        ".\\scripts\\rvc_worker.py",
        PROJECT_ROOT / "scripts" / "rvc_worker.py",
        project_root=PROJECT_ROOT,
    )
    live2d_path, live2d_name = resolve_live2d_model_config(config)
    live2d_dir = resolve_project_path(live2d_path, project_root=PROJECT_ROOT)
    live2d_model = (
        live2d_dir / live2d_name
        if live2d_dir is not None and live2d_name
        else None
    )
    rvc_config = config.get("tts", {}).get("rvc", {}) or {}
    rvc_model = resolve_project_path(rvc_config.get("model_path"), project_root=PROJECT_ROOT)
    rvc_index = resolve_project_path(rvc_config.get("index_path"), project_root=PROJECT_ROOT)
    rvc_overlay = PROJECT_ROOT / ".rvc-overlay"

    report = {
        "ticket": "LIL-49",
        "platform": platform_label(),
        "wsl_env_applied": applied,
        "wsl": wsl,
        "python_sys": sys.executable,
        "resolve_legacy_windows_python": str(resolved_python),
        "resolve_legacy_worker_script": str(worker),
        "worker_script_exists": worker.is_file(),
        "live2d_model": str(live2d_model) if live2d_model else None,
        "live2d_model_exists": bool(live2d_model and live2d_model.is_file()),
        "rvc_model_exists": bool(rvc_model and rvc_model.is_file()),
        "rvc_index_exists": bool(rvc_index and rvc_index.is_file()),
        "rvc_overlay_exists": rvc_overlay.is_dir(),
        "cuda": cuda,
        "audio": audio,
    }

    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0

    print("=== Local AI Companion — platform check (LIL-49) ===")
    print(f"platform:          {report['platform']}")
    print(f"sys.executable:    {report['python_sys']}")
    print(f"legacy win python → {report['resolve_legacy_windows_python']}")
    print(f"legacy worker     → {report['resolve_legacy_worker_script']} (exists={report['worker_script_exists']})")
    print(f"Live2D model:      {report['live2d_model']} (exists={report['live2d_model_exists']})")
    print(
        "RVC voice:         "
        f"model={report['rvc_model_exists']} index={report['rvc_index_exists']} "
        f"overlay={report['rvc_overlay_exists']}"
    )
    if applied:
        print(f"WSL env applied:   {applied}")
    if wsl.get("is_wsl"):
        print(
            f"WSLg:              pulse_socket={wsl.get('pulse_socket_exists')} "
            f"PULSE_SERVER={wsl.get('pulse_server')!r} DISPLAY={wsl.get('display')!r}"
        )
    print(
        f"CUDA:              available={cuda.get('cuda_available')} "
        f"device={cuda.get('device_name')!r} err={cuda.get('error')!r}"
    )
    if cuda.get("hint"):
        print(f"CUDA hint:         {cuda['hint']}")
    print(
        f"Audio:             in={audio.get('input_devices')} out={audio.get('output_devices')} "
        f"default={audio.get('default_device')!r}"
    )
    if audio.get("hint"):
        print(f"Audio hint:        {audio['hint']}")
    if audio.get("error"):
        print(f"Audio error:       {audio['error']}")

    # Soft exit code: 0 always for diagnostics; warnings are textual.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
