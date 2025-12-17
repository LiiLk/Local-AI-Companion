#!/bin/bash
# Script to start the llama.cpp server with Qwen3-VL-8B
# 
# This script uses the pre-compiled Vulkan binaries of llama.cpp
# installed in ~/tools/llama-cpp/
#
# Usage:
#   ./scripts/start_llm_server.sh           # Run in foreground
#   ./scripts/start_llm_server.sh --daemon  # Run in background

set -e

# === Configuration ===
# Resolve project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
LLAMA_CPP_DIR="$HOME/tools/llama-cpp"

# Model paths
MODEL_DIR="$PROJECT_ROOT/models/qwen3-vl-8b"
MODEL_FILE="$MODEL_DIR/Qwen3-VL-8B-Instruct-Q4_K_M.gguf"
MMPROJ_FILE="$MODEL_DIR/mmproj-F16.gguf"
LOG_FILE="/tmp/llama-server.log"

# Server parameters
HOST="0.0.0.0"
PORT=8080
CTX_SIZE=4096  # Reduced from 8192 to save VRAM (~2GB less)
GPU_LAYERS=20  # Reduced to fit with CosyVoice3 (~5GB LLM + 4GB TTS = 9GB total, leaves 3GB margin)

# === Checks ===
if [ ! -d "$LLAMA_CPP_DIR" ]; then
    echo "❌ llama.cpp not found: $LLAMA_CPP_DIR"
    echo ""
    echo "Installation:"
    echo "  mkdir -p ~/tools/llama-cpp && cd ~/tools/llama-cpp"
    echo "  wget https://github.com/ggml-org/llama.cpp/releases/latest/download/llama-<version>-bin-ubuntu-vulkan-x64.tar.gz"
    echo "  tar -xzf llama-*.tar.gz"
    exit 1
fi

if [ ! -f "$MODEL_FILE" ]; then
    echo "❌ Model not found: $MODEL_FILE"
    echo ""
    echo "Download:"
    echo "  mkdir -p $MODEL_DIR && cd $MODEL_DIR"
    echo "  ./scripts/setup_qwen_vl.sh"
    exit 1
fi

if [ ! -f "$MMPROJ_FILE" ]; then
    echo "❌ Vision projector not found: $MMPROJ_FILE"
    exit 1
fi

# === Stop existing instance ===
if pgrep -f "llama-server" > /dev/null; then
    echo "⚠️  Stopping existing instance..."
    pkill -f "llama-server" || true
    sleep 2
fi

# === Display information ===
echo "🦙 Qwen3-VL-8B-Instruct Server"
echo "=============================="
echo "   Model     : $(basename $MODEL_FILE) (4.7 GB)"
echo "   Vision    : $(basename $MMPROJ_FILE) (170 MB)"
echo "   Backend   : Vulkan (GPU)"
echo "   Port      : http://localhost:$PORT"
echo "   Context   : $CTX_SIZE tokens"
echo "   GPU Layers: $GPU_LAYERS (all)"
echo ""

# === Start the server ===
cd "$LLAMA_CPP_DIR"
export LD_LIBRARY_PATH="$LLAMA_CPP_DIR:$LD_LIBRARY_PATH"

if [ "$1" = "--daemon" ] || [ "$1" = "-d" ]; then
    echo "🚀 Starting in background..."
    nohup ./llama-server \
        --model "$MODEL_FILE" \
        --mmproj "$MMPROJ_FILE" \
        --host "$HOST" \
        --port "$PORT" \
        --ctx-size "$CTX_SIZE" \
        --n-gpu-layers "$GPU_LAYERS" \
        --jinja \
        --no-context-shift \
        > "$LOG_FILE" 2>&1 &
    
    PID=$!
    echo "   PID: $PID"
    echo "   Logs: $LOG_FILE"
    echo ""
    echo "Waiting for startup..."
    sleep 8
    
    if curl -s http://localhost:$PORT/health > /dev/null 2>&1; then
        echo "✅ Server ready!"
        echo ""
        echo "Test: curl http://localhost:$PORT/v1/chat/completions -H 'Content-Type: application/json' -d '{\"messages\":[{\"role\":\"user\",\"content\":\"Salut!\"}]}'"
    else
        echo "⚠️  Server is starting (may take ~10s to load the model)"
        echo "   Check the logs: tail -f $LOG_FILE"
    fi
else
    echo "🚀 Starting in foreground (Ctrl+C to stop)..."
    echo ""
    ./llama-server \
        --model "$MODEL_FILE" \
        --mmproj "$MMPROJ_FILE" \
        --host "$HOST" \
        --port "$PORT" \
        --ctx-size "$CTX_SIZE" \
        --n-gpu-layers "$GPU_LAYERS" \
        --jinja \
        --no-context-shift
fi
