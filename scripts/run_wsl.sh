#!/usr/bin/env bash
# LIL-49 — Easy launcher for Local AI Companion under WSL2.
#
# This is the supported *WSL product path*: web UI + voice backend (or CLI).
# The polished Live2D desktop mascot remains a Windows-native target.
#
# Usage:
#   bash scripts/run_wsl.sh              # web server (default)
#   bash scripts/run_wsl.sh web
#   bash scripts/run_wsl.sh cli
#   bash scripts/run_wsl.sh cli-voice
#   bash scripts/run_wsl.sh bridge
#   bash scripts/run_wsl.sh desktop      # experimental Qt avatar (WSLg only)
#   bash scripts/run_wsl.sh check
#
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/venv}"
PYTHON="${VENV_DIR}/bin/python"
MODE="${1:-web}"

usage() {
  cat <<'EOF'
Local AI Companion — WSL launcher (LIL-49)

Usage: bash scripts/run_wsl.sh [mode]

Modes:
  web         FastAPI + browser UI on http://127.0.0.1:8000  (default)
  cli         Text chatbot (main.py)
  cli-voice   Voice CLI (main.py --voice --listen)
  bridge      Backend websocket bridge only (port 8765)
  desktop     Experimental Qt/Live2D inside WSLg only (not desktop-pet parity)
  check       Platform diagnostic only

First-time setup:
  bash scripts/setup_wsl.sh --with-rvc

Desktop assistant (Windows and WSL):
  python run_assistant.py

Open the web UI from Windows browser:
  http://localhost:8000
  (WSL localhost is forwarded to Windows by default)
EOF
}

if [[ "${MODE}" == "-h" || "${MODE}" == "--help" || "${MODE}" == "help" ]]; then
  usage
  exit 0
fi

if [[ ! -x "$PYTHON" ]]; then
  cat >&2 <<EOF
No WSL venv at: $VENV_DIR

Run setup first:
  bash scripts/setup_wsl.sh
  source venv/bin/activate
EOF
  exit 1
fi

# Ensure Pulse/WSLg + software WebGL defaults before any Qt/audio import.
export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:$PYTHONPATH}"
cd "$ROOT_DIR"

# shellcheck disable=SC1091
source <("$PYTHON" - <<'PY'
from src.utils.platform_compat import ensure_wsl_desktop_env, is_wsl
import os, shlex
if is_wsl():
    ensure_wsl_desktop_env()
for k in (
    "PULSE_SERVER", "DISPLAY", "QTWEBENGINE_CHROMIUM_FLAGS",
    "QT_OPENGL", "LIBGL_ALWAYS_SOFTWARE", "QT_SCALE_FACTOR",
    "QT_AUTO_SCREEN_SCALE_FACTOR", "QT_ENABLE_HIGHDPI_SCALING",
):
    v = os.environ.get(k)
    if v:
        print(f"export {k}={shlex.quote(v)}")
PY
)

case "$MODE" in
  check)
    exec "$PYTHON" scripts/check_platform.py
    ;;
  web|server)
    echo "[WSL] Starting FastAPI companion on http://127.0.0.1:8000"
    echo "[WSL] For the desktop pet, use: python run_assistant.py"
    echo "[WSL] Or open http://localhost:8000/web/ in Windows Chrome/Edge"
    echo "[WSL] Ctrl+C to stop."
    # reload=False for a predictable "product" run path
    exec "$PYTHON" - <<'PY'
from pathlib import Path
import uvicorn
from src.utils.config_loader import load_yaml_config
from src.server.settings import resolve_server_host, resolve_server_port

config = load_yaml_config(Path("config/config.yaml"))
host = resolve_server_host(config)
port = resolve_server_port(config)
uvicorn.run("src.server.app:app", host=host, port=port, reload=False)
PY
    ;;
  cli)
    exec "$PYTHON" main.py
    ;;
  cli-voice|voice)
    exec "$PYTHON" main.py --voice --listen
    ;;
  bridge)
    echo "[WSL] Bridge backend on ws://127.0.0.1:8765 (no local window)."
    exec "$PYTHON" run_assistant.py --bridge-server --bridge-port 8765
    ;;
  desktop|wslg)
    echo "[WSL] Experimental desktop avatar inside WSLg (software WebGL)."
    echo "[WSL] For the Windows-native desktop pet, use: python run_assistant.py"
    echo "[WSL] EGL/DRM console noise is expected under WSLg."
    exec env LOCAL_AI_FORCE_WSL_DESKTOP=1 "$PYTHON" run_assistant.py --force-wsl-desktop
    ;;
  *)
    echo "Unknown mode: $MODE" >&2
    usage >&2
    exit 2
    ;;
esac
