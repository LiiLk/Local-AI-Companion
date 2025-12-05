#!/bin/bash
# Script pour lancer le serveur llama.cpp avec Jan-v2-VL-high
# 
# Ce script utilise les binaires Vulkan pré-compilés de llama.cpp
# installés dans ~/tools/llama-cpp/
#
# Usage:
#   ./scripts/start_llm_server.sh           # Lancer au premier plan
#   ./scripts/start_llm_server.sh --daemon  # Lancer en arrière-plan

set -e

# === Configuration ===
LLAMA_CPP_DIR="$HOME/tools/llama-cpp"
MODEL_DIR="$HOME/models/jan-v2-vl-high"
MODEL_FILE="$MODEL_DIR/Jan-v2-VL-high-Q4_K_M.gguf"
MMPROJ_FILE="$MODEL_DIR/mmproj-Jan-v2-VL-high.gguf"
LOG_FILE="/tmp/llama-server.log"

# Paramètres du serveur
HOST="0.0.0.0"
PORT=8080
CTX_SIZE=8192
GPU_LAYERS=99  # Toutes les couches sur GPU

# === Vérifications ===
if [ ! -d "$LLAMA_CPP_DIR" ]; then
    echo "❌ llama.cpp non trouvé: $LLAMA_CPP_DIR"
    echo ""
    echo "Installation:"
    echo "  mkdir -p ~/tools/llama-cpp && cd ~/tools/llama-cpp"
    echo "  wget https://github.com/ggml-org/llama.cpp/releases/latest/download/llama-<version>-bin-ubuntu-vulkan-x64.tar.gz"
    echo "  tar -xzf llama-*.tar.gz"
    exit 1
fi

if [ ! -f "$MODEL_FILE" ]; then
    echo "❌ Modèle non trouvé: $MODEL_FILE"
    echo ""
    echo "Téléchargement:"
    echo "  mkdir -p $MODEL_DIR && cd $MODEL_DIR"
    echo "  wget https://huggingface.co/janhq/Jan-v2-VL-high-gguf/resolve/main/Jan-v2-VL-high-Q4_K_M.gguf"
    echo "  wget https://huggingface.co/janhq/Jan-v2-VL-high-gguf/resolve/main/mmproj-Jan-v2-VL-high.gguf"
    exit 1
fi

if [ ! -f "$MMPROJ_FILE" ]; then
    echo "❌ Vision projector non trouvé: $MMPROJ_FILE"
    exit 1
fi

# === Arrêter l'instance existante ===
if pgrep -f "llama-server" > /dev/null; then
    echo "⚠️  Arrêt de l'instance existante..."
    pkill -f "llama-server" || true
    sleep 2
fi

# === Afficher les informations ===
echo "🦙 Jan-v2-VL-high Server"
echo "========================"
echo "   Modèle    : $(basename $MODEL_FILE) (4.7 GB)"
echo "   Vision    : $(basename $MMPROJ_FILE) (1.1 GB)"
echo "   Backend   : Vulkan (GPU)"
echo "   Port      : http://localhost:$PORT"
echo "   Context   : $CTX_SIZE tokens"
echo "   GPU Layers: $GPU_LAYERS (toutes)"
echo ""

# === Lancer le serveur ===
cd "$LLAMA_CPP_DIR"
export LD_LIBRARY_PATH="$LLAMA_CPP_DIR:$LD_LIBRARY_PATH"

if [ "$1" = "--daemon" ] || [ "$1" = "-d" ]; then
    echo "🚀 Lancement en arrière-plan..."
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
    echo "Attente du démarrage..."
    sleep 8
    
    if curl -s http://localhost:$PORT/health > /dev/null 2>&1; then
        echo "✅ Serveur prêt!"
        echo ""
        echo "Test: curl http://localhost:$PORT/v1/chat/completions -H 'Content-Type: application/json' -d '{\"messages\":[{\"role\":\"user\",\"content\":\"Salut!\"}]}'"
    else
        echo "⚠️  Le serveur démarre (peut prendre ~10s pour charger le modèle)"
        echo "   Vérifiez les logs: tail -f $LOG_FILE"
    fi
else
    echo "🚀 Lancement au premier plan (Ctrl+C pour arrêter)..."
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
