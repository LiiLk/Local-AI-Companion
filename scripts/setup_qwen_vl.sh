#!/bin/bash
# scripts/setup_qwen_vl.sh

echo "🚀 Setting up Qwen2.5-VL-7B-Instruct..."

# Directory for the model
MODEL_DIR="models/qwen3-vl-8b"
mkdir -p "$MODEL_DIR"

# Install huggingface-cli if not present (should be in venv)
if ! ./venv/bin/huggingface-cli --version &> /dev/null; then
    echo "⚠️  huggingface-cli not found. Installing in venv..."
    ./venv/bin/pip install huggingface_hub
fi

# Define CLI path
CLI="./venv/bin/huggingface-cli"

# Repository and Files
REPO="unsloth/Qwen3-VL-8B-Instruct-GGUF"
MODEL_FILE="Qwen3-VL-8B-Instruct-Q4_K_M.gguf"
MMPROJ_FILE="mmproj-F16.gguf"

echo "📥 Downloading model to $MODEL_DIR..."
echo "   Repo: $REPO"
echo "   Files: $MODEL_FILE, $MMPROJ_FILE"

# Download Model
$CLI download "$REPO" "$MODEL_FILE" --local-dir "$MODEL_DIR" --local-dir-use-symlinks False
if [ $? -ne 0 ]; then
    echo "❌ Failed to download model file."
    exit 1
fi

# Download Vision Projector (mmproj)
$CLI download "$REPO" "$MMPROJ_FILE" --local-dir "$MODEL_DIR" --local-dir-use-symlinks False
if [ $? -ne 0 ]; then
    echo "❌ Failed to download mmproj file."
    exit 1
fi

echo "✅ Download complete!"
echo "   Model Path: $(pwd)/$MODEL_DIR/$MODEL_FILE"
echo "   Vision Adapter: $(pwd)/$MODEL_DIR/$MMPROJ_FILE"
