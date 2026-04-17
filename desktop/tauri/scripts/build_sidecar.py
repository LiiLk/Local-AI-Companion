from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
TAURI_DIR = ROOT / "desktop" / "tauri"
SRC_TAURI_DIR = TAURI_DIR / "src-tauri"
SPEC_PATH = TAURI_DIR / "pyinstaller" / "assistant_bridge.spec"
BIN_DIR = SRC_TAURI_DIR / "binaries"
DIST_DIR = TAURI_DIR / ".sidecar-dist"
WORK_DIR = TAURI_DIR / ".sidecar-build"


def host_triple() -> str:
    result = subprocess.run(
        ["rustc", "--print", "host-tuple"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def target_binary_path() -> Path:
    suffix = ".exe" if sys.platform.startswith("win") else ""
    return BIN_DIR / f"local-ai-companion-sidecar-{host_triple()}{suffix}"


def build_sidecar() -> None:
    pyinstaller = shutil.which("pyinstaller")
    if pyinstaller is None:
        raise SystemExit(
            "PyInstaller is not installed. Install it with `python -m pip install pyinstaller` "
            "before building the desktop sidecar."
        )

    BIN_DIR.mkdir(parents=True, exist_ok=True)
    DIST_DIR.mkdir(parents=True, exist_ok=True)
    WORK_DIR.mkdir(parents=True, exist_ok=True)

    subprocess.run(
        [
          pyinstaller,
          "--noconfirm",
          "--clean",
          str(SPEC_PATH),
          "--distpath",
          str(DIST_DIR),
          "--workpath",
          str(WORK_DIR),
        ],
        cwd=ROOT,
        check=True,
    )

    built = DIST_DIR / ("local-ai-companion-sidecar.exe" if sys.platform.startswith("win") else "local-ai-companion-sidecar")
    if not built.exists():
        raise SystemExit(f"Expected PyInstaller output not found: {built}")

    target = target_binary_path()
    shutil.copy2(built, target)
    print(f"Sidecar ready: {target}")


if __name__ == "__main__":
    build_sidecar()
