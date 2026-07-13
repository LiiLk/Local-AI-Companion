#!/usr/bin/env bash
# Bootstrap the supported WSL2 backend/CLI development environment.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/venv}"
INSTALL_RVC=0
INSTALL_DEV=0
SKIP_APT=0

usage() {
  cat <<'EOF'
Usage: bash scripts/setup_wsl.sh [options]

Options:
  --with-rvc  Install the optional RVC voice-conversion overlay.
  --dev       Install test/development dependencies.
  --skip-apt  Skip Ubuntu/Debian system packages.
  -h, --help  Show this help.

Environment:
  VENV_DIR    Override the virtualenv directory (default: ./venv).
EOF
}

while (($#)); do
  case "$1" in
    --with-rvc) INSTALL_RVC=1 ;;
    --dev) INSTALL_DEV=1 ;;
    --skip-apt) SKIP_APT=1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

if ! grep -qiE '(microsoft|wsl)' /proc/sys/kernel/osrelease /proc/version 2>/dev/null; then
  echo "This installer targets WSL2. Native Linux may work, but is not validated by this script." >&2
  exit 1
fi

if ((SKIP_APT == 0)); then
  if ! command -v apt-get >/dev/null 2>&1; then
    echo "apt-get was not found. Use Ubuntu/Debian WSL or rerun with --skip-apt." >&2
    exit 1
  fi

  echo "[1/4] Installing WSL host packages..."
  sudo apt-get update
  sudo apt-get install -y \
    build-essential \
    alsa-utils \
    ffmpeg \
    git \
    libasound2-dev \
    libasound2-plugins \
    libegl1 \
    libgbm1 \
    libgl1-mesa-dri \
    libportaudio2 \
    libpulse-dev \
    mesa-utils \
    mesa-vulkan-drivers \
    pkg-config \
    portaudio19-dev \
    pulseaudio-utils \
    python3-dev \
    python3-venv
else
  echo "[1/4] Skipping system packages (--skip-apt)."
fi

echo "[2/4] Creating Python environment at $VENV_DIR..."
if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  python3 -m venv "$VENV_DIR"
fi
PYTHON="$VENV_DIR/bin/python"
"$PYTHON" -m pip install --upgrade pip setuptools wheel

echo "[3/4] Installing Local AI Companion dependencies..."
if ((INSTALL_DEV)); then
  "$PYTHON" -m pip install -r "$ROOT_DIR/requirements-dev.txt"
else
  "$PYTHON" -m pip install -r "$ROOT_DIR/requirements.txt"
fi

if ((INSTALL_RVC)); then
  bash "$ROOT_DIR/scripts/install_rvc_wsl.sh" --python "$PYTHON"
fi

echo "[4/4] Running platform diagnostics..."
"$PYTHON" "$ROOT_DIR/scripts/check_platform.py"

LIVE2D_MODEL="$ROOT_DIR/frontend/live2d/runtime-assets/models/march7th_tauri/march7th.model3.json"
if [[ ! -f "$LIVE2D_MODEL" ]]; then
  echo "WARNING: the March 7th Live2D runtime pack is missing." >&2
  echo "Expected licensed model entrypoint: $LIVE2D_MODEL" >&2
  echo "The backend and microphone can run, but the avatar cannot load without this pack." >&2
fi

if command -v pactl >/dev/null 2>&1; then
  if ! pactl info >/dev/null 2>&1; then
    echo "WARNING: WSLg PulseAudio is not reachable in this session." >&2
    echo "Run 'wsl --shutdown' from PowerShell, reopen WSL, then rerun the diagnostic." >&2
  elif [[ -z "$(pactl list short sources 2>/dev/null)" ]]; then
    echo "WARNING: WSLg PulseAudio is reachable but exposes no microphone source." >&2
    echo "Check Windows Settings > Privacy & security > Microphone." >&2
  fi
fi

cat <<EOF

WSL setup complete.

Desktop assistant (canonical command on Windows and WSL):

  source "$VENV_DIR/bin/activate"
  python run_assistant.py

WSL-only secondary tools:

  bash scripts/run_wsl.sh web       # browser UI
  bash scripts/run_wsl.sh cli       # text CLI
  bash scripts/run_wsl.sh check     # diagnostics

- If audio is missing: PowerShell → wsl --shutdown → reopen WSL → bash scripts/run_wsl.sh check
- CUDA uses the NVIDIA driver on Windows; do not install a Linux NVIDIA display driver in WSL.

Details: docs/lil-49-wsl-compatibility.md
EOF
