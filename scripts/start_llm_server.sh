#!/bin/bash
# Script to start the llama.cpp server with Jan-v2-VL-high
# 
# This script uses the pre-compiled Vulkan binaries of llama.cpp
# installed in ~/tools/llama-cpp/
#
# Usage:
#   ./scripts/start_llm_server.sh           # Run in foreground
#   ./scripts/start_llm_server.sh --daemon  # Run in background

set -e

# === Configuration ===
LLAMA_CPP_DIR="$HOME/tools/llama-cpp"
MODEL_DIR="$HOME/models/jan-v2-vl-high"
MODEL_FILE="$MODEL_DIR/Jan-v2-VL-high-Q4_K_M.gguf"
MMPROJ_FILE="$MODEL_DIR/mmproj-Jan-v2-VL-high.gguf"
LOG_FILE="/tmp/llama-server.log"

# Server parameters
HOST="0.0.0.0"
PORT=8080
CTX_SIZE=8192
GPU_LAYERS=99  # All layers on GPU

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
    echo "  wget https://huggingface.co/janhq/Jan-v2-VL-high-gguf/resolve/main/Jan-v2-VL-high-Q4_K_M.gguf"
    echo "  wget https://huggingface.co/janhq/Jan-v2-VL-high-gguf/resolve/main/mmproj-Jan-v2-VL-high.gguf"
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
echo "🦙 Jan-v2-VL-high Server"
echo "========================"
echo "   Model     : $(basename $MODEL_FILE) (4.7 GB)"
echo "   Vision    : $(basename $MMPROJ_FILE) (1.1 GB)"
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
