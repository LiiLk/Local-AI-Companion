#!/bin/bash
# CosyVoice3 TTS Server Startup Script
#
# This script manages the CosyVoice3 TTS server which runs in
# a separate conda environment (Python 3.10).
#
# Usage:
#   ./scripts/start_cosyvoice3.sh start    # Start the server
#   ./scripts/start_cosyvoice3.sh stop     # Stop the server
#   ./scripts/start_cosyvoice3.sh status   # Check if running
#   ./scripts/start_cosyvoice3.sh install  # Install dependencies

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
COSYVOICE_DIR="${COSYVOICE_DIR:-$HOME/tools/CosyVoice}"
CONDA_ENV="cosyvoice"
PORT="${COSYVOICE_PORT:-9881}"
HOST="${COSYVOICE_HOST:-127.0.0.1}"
PID_FILE="/tmp/cosyvoice3_server.pid"
LOG_FILE="/tmp/cosyvoice3_server.log"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

print_header() {
    echo -e "${CYAN}"
    echo "╔════════════════════════════════════════════════════════════════╗"
    echo "║           CosyVoice3 TTS Server Manager                        ║"
    echo "║                                                                 ║"
    echo "║   State-of-the-art multilingual TTS with voice cloning        ║"
    echo "║   Model: Fun-CosyVoice3-0.5B (~2.5GB VRAM with FP16)          ║"
    echo "╚════════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

# Get conda base path
get_conda_base() {
    if command -v conda &> /dev/null; then
        conda info --base
    elif [ -d "$HOME/miniconda3" ]; then
        echo "$HOME/miniconda3"
    elif [ -d "$HOME/anaconda3" ]; then
        echo "$HOME/anaconda3"
    else
        echo ""
    fi
}

# Check if cosyvoice conda env exists
check_conda_env() {
    local conda_base=$(get_conda_base)
    if [ -z "$conda_base" ]; then
        return 1
    fi

    if [ -d "$conda_base/envs/$CONDA_ENV" ]; then
        return 0
    fi
    return 1
}

# Check if CosyVoice is installed
check_installation() {
    if [ -d "$COSYVOICE_DIR" ] && [ -d "$COSYVOICE_DIR/cosyvoice" ]; then
        return 0
    fi
    return 1
}

# Check if model is downloaded
check_model() {
    local model_dir="$COSYVOICE_DIR/pretrained_models/Fun-CosyVoice3-0.5B"
    if [ -d "$model_dir" ] && [ -f "$model_dir/llm.pt" ]; then
        return 0
    fi
    return 1
}

# Check if server is running
is_running() {
    if [ -f "$PID_FILE" ]; then
        local pid=$(cat "$PID_FILE")
        if ps -p "$pid" > /dev/null 2>&1; then
            return 0
        fi
    fi

    # Also check by port
    if curl -s "http://$HOST:$PORT/health" > /dev/null 2>&1; then
        return 0
    fi

    return 1
}

# Start the server
start_server() {
    print_header

    if is_running; then
        echo -e "${YELLOW}⚠️  CosyVoice3 server is already running on port $PORT${NC}"
        return 0
    fi

    if ! check_installation; then
        echo -e "${RED}❌ CosyVoice not found at: $COSYVOICE_DIR${NC}"
        echo -e "${YELLOW}Run: git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git $COSYVOICE_DIR${NC}"
        exit 1
    fi

    if ! check_conda_env; then
        echo -e "${RED}❌ Conda environment '$CONDA_ENV' not found${NC}"
        echo -e "${YELLOW}Create it with: conda create -n $CONDA_ENV python=3.10${NC}"
        exit 1
    fi

    if ! check_model; then
        echo -e "${RED}❌ CosyVoice3 model not found${NC}"
        echo -e "${YELLOW}Download from: https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512${NC}"
        exit 1
    fi

    local conda_base=$(get_conda_base)
    local python_bin="$conda_base/envs/$CONDA_ENV/bin/python"

    if [ ! -f "$python_bin" ]; then
        echo -e "${RED}❌ Python not found in conda env: $python_bin${NC}"
        exit 1
    fi

    echo -e "${GREEN}🚀 Starting CosyVoice3 server on $HOST:$PORT...${NC}"
    echo -e "${BLUE}   Python: $python_bin${NC}"
    echo -e "${BLUE}   Model: $COSYVOICE_DIR/pretrained_models/Fun-CosyVoice3-0.5B${NC}"
    echo -e "${BLUE}   Log: $LOG_FILE${NC}"
    echo ""

    # Start server in background
    export COSYVOICE_DIR="$COSYVOICE_DIR"

    nohup "$python_bin" "$SCRIPT_DIR/cosyvoice3_server.py" \
        --host "$HOST" \
        --port "$PORT" \
        --model-dir "$COSYVOICE_DIR/pretrained_models/Fun-CosyVoice3-0.5B" \
        --fp16 \
        > "$LOG_FILE" 2>&1 &

    local pid=$!
    echo $pid > "$PID_FILE"

    echo -e "${YELLOW}⏳ Waiting for server to start (loading model ~30s)...${NC}"

    # Wait for server to be ready
    local max_wait=60
    local wait_time=0
    while [ $wait_time -lt $max_wait ]; do
        if curl -s "http://$HOST:$PORT/health" > /dev/null 2>&1; then
            echo ""
            echo -e "${GREEN}✅ CosyVoice3 server is ready!${NC}"
            echo ""
            echo -e "${BLUE}API Endpoints:${NC}"
            echo "   GET  http://$HOST:$PORT/           - Server info"
            echo "   GET  http://$HOST:$PORT/health     - Health check"
            echo "   POST http://$HOST:$PORT/synthesize - Synthesize speech"
            echo "   POST http://$HOST:$PORT/synthesize_with_ref - Synthesize with ref path"
            echo ""
            echo -e "${YELLOW}Example:${NC}"
            echo "   curl -X POST http://$HOST:$PORT/synthesize_with_ref \\"
            echo "        -F 'text=Bonjour!' \\"
            echo "        -F 'ref_audio_path=$PROJECT_DIR/resources/voices/juri_neutral.wav' \\"
            echo "        --output test.wav"
            return 0
        fi

        # Check if process died
        if ! ps -p $pid > /dev/null 2>&1; then
            echo ""
            echo -e "${RED}❌ Server process died. Check logs: $LOG_FILE${NC}"
            tail -20 "$LOG_FILE"
            rm -f "$PID_FILE"
            exit 1
        fi

        sleep 2
        wait_time=$((wait_time + 2))
        echo -n "."
    done

    echo ""
    echo -e "${RED}❌ Server failed to start within ${max_wait}s${NC}"
    echo -e "${YELLOW}Check logs: tail -f $LOG_FILE${NC}"
    exit 1
}

# Stop the server
stop_server() {
    print_header

    if [ -f "$PID_FILE" ]; then
        local pid=$(cat "$PID_FILE")
        if ps -p "$pid" > /dev/null 2>&1; then
            echo -e "${YELLOW}Stopping CosyVoice3 server (PID: $pid)...${NC}"
            kill "$pid" 2>/dev/null || true
            sleep 2

            # Force kill if still running
            if ps -p "$pid" > /dev/null 2>&1; then
                kill -9 "$pid" 2>/dev/null || true
            fi

            echo -e "${GREEN}✅ Server stopped${NC}"
        else
            echo -e "${YELLOW}Server process not found (stale PID file)${NC}"
        fi
        rm -f "$PID_FILE"
    else
        echo -e "${YELLOW}No PID file found. Server may not be running.${NC}"

        # Try to find and kill by port
        local pid=$(lsof -t -i:$PORT 2>/dev/null || true)
        if [ -n "$pid" ]; then
            echo -e "${YELLOW}Found process on port $PORT (PID: $pid), killing...${NC}"
            kill "$pid" 2>/dev/null || true
            echo -e "${GREEN}✅ Server stopped${NC}"
        fi
    fi
}

# Show server status
show_status() {
    print_header

    echo -e "${BLUE}Configuration:${NC}"
    echo "   COSYVOICE_DIR: $COSYVOICE_DIR"
    echo "   Conda env: $CONDA_ENV"
    echo "   Port: $PORT"
    echo ""

    echo -e "${BLUE}Installation:${NC}"
    if check_installation; then
        echo -e "   CosyVoice: ${GREEN}✅ Installed${NC}"
    else
        echo -e "   CosyVoice: ${RED}❌ Not found${NC}"
    fi

    if check_conda_env; then
        echo -e "   Conda env: ${GREEN}✅ Found${NC}"
    else
        echo -e "   Conda env: ${RED}❌ Not found${NC}"
    fi

    if check_model; then
        echo -e "   Model: ${GREEN}✅ Downloaded${NC}"
    else
        echo -e "   Model: ${RED}❌ Not found${NC}"
    fi

    echo ""
    echo -e "${BLUE}Server Status:${NC}"
    if is_running; then
        echo -e "   Status: ${GREEN}✅ Running${NC}"
        if [ -f "$PID_FILE" ]; then
            echo "   PID: $(cat "$PID_FILE")"
        fi

        # Get server info
        local info=$(curl -s "http://$HOST:$PORT/" 2>/dev/null || true)
        if [ -n "$info" ]; then
            echo "   Info: $info"
        fi
    else
        echo -e "   Status: ${YELLOW}⚠️  Not running${NC}"
    fi
}

# Show logs
show_logs() {
    if [ -f "$LOG_FILE" ]; then
        echo -e "${BLUE}Showing logs from: $LOG_FILE${NC}"
        echo "─────────────────────────────────────────────────────────────────"
        tail -f "$LOG_FILE"
    else
        echo -e "${YELLOW}No log file found at: $LOG_FILE${NC}"
    fi
}

# Install dependencies
install_deps() {
    print_header
    echo -e "${BLUE}Installing CosyVoice3 dependencies...${NC}"
    echo ""

    local conda_base=$(get_conda_base)

    if [ -z "$conda_base" ]; then
        echo -e "${RED}❌ Conda not found. Please install Miniconda first.${NC}"
        exit 1
    fi

    # Check if env exists
    if ! check_conda_env; then
        echo -e "${YELLOW}Creating conda environment '$CONDA_ENV' with Python 3.10...${NC}"
        conda create -n "$CONDA_ENV" python=3.10 -y
    fi

    # Install dependencies
    echo -e "${YELLOW}Installing dependencies in $CONDA_ENV...${NC}"
    source "$conda_base/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV"

    if [ -f "$COSYVOICE_DIR/requirements.txt" ]; then
        pip install -r "$COSYVOICE_DIR/requirements.txt"
    fi

    # Install FastAPI and uvicorn for the server
    pip install fastapi uvicorn python-multipart

    echo -e "${GREEN}✅ Dependencies installed!${NC}"
}

# Show help
show_help() {
    print_header
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  start     Start the CosyVoice3 TTS server"
    echo "  stop      Stop the server"
    echo "  restart   Restart the server"
    echo "  status    Show server status"
    echo "  logs      Show server logs (tail -f)"
    echo "  install   Install dependencies"
    echo "  help      Show this help message"
    echo ""
    echo "Environment variables:"
    echo "  COSYVOICE_DIR   Path to CosyVoice (default: ~/tools/CosyVoice)"
    echo "  COSYVOICE_PORT  Server port (default: 9881)"
    echo "  COSYVOICE_HOST  Server host (default: 127.0.0.1)"
    echo ""
    echo "Examples:"
    echo "  $0 start"
    echo "  $0 status"
    echo "  COSYVOICE_PORT=9882 $0 start"
    echo ""
}

# Main
case "${1:-help}" in
    start)
        start_server
        ;;
    stop)
        stop_server
        ;;
    restart)
        stop_server
        sleep 2
        start_server
        ;;
    status)
        show_status
        ;;
    logs)
        show_logs
        ;;
    install)
        install_deps
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        echo -e "${RED}Unknown command: $1${NC}"
        show_help
        exit 1
        ;;
esac
