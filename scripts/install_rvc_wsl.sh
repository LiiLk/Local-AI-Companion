#!/usr/bin/env bash
# Install the isolated InferRVC dependency overlay for WSL2/Linux.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="$ROOT_DIR/venv/bin/python"
INSTALL_DIR="$ROOT_DIR/.rvc-overlay"
FAIRSEQ_COMMIT="44800430a728c2216fd1cf1e8daa672f50dfacba"

usage() {
  cat <<'EOF'
Usage: bash scripts/install_rvc_wsl.sh [options]

Options:
  --python PATH       Main runtime Python (default: ./venv/bin/python).
  --install-dir PATH  Overlay directory (default: ./.rvc-overlay).
  -h, --help          Show this help.
EOF
}

while (($#)); do
  case "$1" in
    --python) PYTHON="$2"; shift ;;
    --install-dir) INSTALL_DIR="$2"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

if [[ ! -x "$PYTHON" ]]; then
  echo "Python executable not found: $PYTHON" >&2
  exit 1
fi

REQUIREMENTS="$ROOT_DIR/scripts/requirements-rvc-worker.txt"
if [[ ! -f "$REQUIREMENTS" ]]; then
  echo "RVC requirements not found: $REQUIREMENTS" >&2
  exit 1
fi

echo "Installing RVC dependency overlay only. Voice model assets are resolved at runtime from config/characters/<preset>.yaml or explicit RVC config; FAISS index files remain optional when index_rate is 0."

STAGING_DIR="${INSTALL_DIR}.staging"
BACKUP_DIR="${INSTALL_DIR}.backup"

cleanup() {
  rm -rf "$STAGING_DIR"
}
trap cleanup EXIT

rm -rf "$STAGING_DIR" "$BACKUP_DIR"
mkdir -p "$STAGING_DIR"

echo "Installing isolated RVC packages into $STAGING_DIR..."
"$PYTHON" -m pip install --upgrade --target "$STAGING_DIR" -r "$REQUIREMENTS"
echo "Installing RVC packages without a duplicate Torch/CUDA runtime..."
"$PYTHON" -m pip install --upgrade --target "$STAGING_DIR" --no-deps \
  "https://github.com/One-sixth/fairseq/archive/${FAIRSEQ_COMMIT}.zip" \
  "torchcrepe==0.0.24" \
  "inferrvc==1.0"

# Defensive cleanup for upgrades from an older installer. RVC always reuses
# Torch/Torchaudio from the main environment so CUDA stays consistent.
rm -rf \
  "$STAGING_DIR/torch" \
  "$STAGING_DIR/torchaudio" \
  "$STAGING_DIR/torchgen" \
  "$STAGING_DIR/functorch"
find "$STAGING_DIR" -maxdepth 1 -type d \
  \( -name 'torch-*.dist-info' -o -name 'torchaudio-*.dist-info' \) \
  -exec rm -rf {} +

"$PYTHON" "$ROOT_DIR/scripts/rvc_worker.py" \
  --check-imports \
  --site-packages-dir "$STAGING_DIR"

if [[ -e "$INSTALL_DIR" ]]; then
  mv "$INSTALL_DIR" "$BACKUP_DIR"
fi
if ! mv "$STAGING_DIR" "$INSTALL_DIR"; then
  if [[ -e "$BACKUP_DIR" ]]; then
    mv "$BACKUP_DIR" "$INSTALL_DIR"
  fi
  exit 1
fi
rm -rf "$BACKUP_DIR"
trap - EXIT

echo "RVC overlay installed at $INSTALL_DIR"
