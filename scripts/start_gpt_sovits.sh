#!/bin/bash
# GPT-SoVITS Server Setup and Startup Script
# This script helps set up and run the GPT-SoVITS TTS server

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
SOVITS_DIR="${SOVITS_DIR:-$HOME/tools/GPT-SoVITS}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_header() {
    echo -e "${BLUE}"
    echo "╔════════════════════════════════════════════════════════════════╗"
    echo "║           GPT-SoVITS TTS Server Manager                        ║"
    echo "║                                                                 ║"
    echo "║   Best quality voice cloning TTS (53k+ GitHub stars)           ║"
    echo "╚════════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

check_installation() {
    if [ -d "$SOVITS_DIR" ] && [ -f "$SOVITS_DIR/api.py" ]; then
        return 0
    fi
    return 1
}

install_sovits() {
    echo -e "${YELLOW}📦 Installing GPT-SoVITS...${NC}"
    
    mkdir -p "$(dirname "$SOVITS_DIR")"
    
    # Clone repository
    echo -e "${GREEN}Cloning GPT-SoVITS repository...${NC}"
    git clone --recursive https://github.com/RVC-Boss/GPT-SoVITS.git "$SOVITS_DIR"
    
    cd "$SOVITS_DIR"
    
    # Create conda environment
    echo -e "${GREEN}Creating conda environment...${NC}"
    if command -v conda &> /dev/null; then
        conda create -n GPTSoVits python=3.10 -y
        source "$(conda info --base)/etc/profile.d/conda.sh"
        conda activate GPTSoVits
        
        # Install dependencies
        echo -e "${GREEN}Installing dependencies...${NC}"
        pip install -r requirements.txt
        
        # Install ffmpeg
        conda install ffmpeg -y
        
        echo -e "${GREEN}✅ GPT-SoVITS installed successfully!${NC}"
        echo ""
        echo -e "${YELLOW}⚠️  You need to download pretrained models:${NC}"
        echo "   1. Download from: https://huggingface.co/lj1995/GPT-SoVITS"
        echo "   2. Place in: $SOVITS_DIR/GPT_SoVITS/pretrained_models/"
        echo ""
    else
        echo -e "${RED}❌ Conda not found! Please install Miniconda first.${NC}"
        exit 1
    fi
}

start_server() {
    local port="${1:-9880}"
    local ref_audio="${2:-}"
    local ref_text="${3:-}"
    local ref_lang="${4:-zh}"
    
    cd "$SOVITS_DIR"
    
    # Activate conda environment
    if command -v conda &> /dev/null; then
        source "$(conda info --base)/etc/profile.d/conda.sh"
        conda activate GPTSoVits 2>/dev/null || true
    fi
    
    echo -e "${GREEN}🚀 Starting GPT-SoVITS server on port $port...${NC}"
    echo ""
    
    # Build command
    local cmd="python api.py -p $port"
    
    if [ -n "$ref_audio" ]; then
        cmd="$cmd -dr \"$ref_audio\" -dt \"$ref_text\" -dl \"$ref_lang\""
        echo -e "${BLUE}Default reference: $ref_audio ($ref_lang)${NC}"
    fi
    
    echo -e "${YELLOW}Command: $cmd${NC}"
    echo ""
    echo "─────────────────────────────────────────────────────────────────"
    
    eval $cmd
}

show_help() {
    print_header
    echo "Usage: $0 [COMMAND] [OPTIONS]"
    echo ""
    echo "Commands:"
    echo "  install              Install GPT-SoVITS from GitHub"
    echo "  start [OPTIONS]      Start the GPT-SoVITS API server"
    echo "  status               Check if server is running"
    echo "  help                 Show this help message"
    echo ""
    echo "Start Options:"
    echo "  -p, --port PORT      API port (default: 9880)"
    echo "  -r, --ref PATH       Default reference audio path"
    echo "  -t, --text TEXT      Reference audio text"
    echo "  -l, --lang LANG      Reference language (zh/en/ja/ko/yue)"
    echo ""
    echo "Examples:"
    echo "  $0 install"
    echo "  $0 start"
    echo "  $0 start -p 9880 -r /path/to/ref.wav -l ja"
    echo ""
    echo "Configuration:"
    echo "  SOVITS_DIR=$SOVITS_DIR"
    echo ""
}

check_status() {
    if curl -s "http://127.0.0.1:9880/" > /dev/null 2>&1; then
        echo -e "${GREEN}✅ GPT-SoVITS server is running on port 9880${NC}"
        return 0
    else
        echo -e "${YELLOW}⚠️  GPT-SoVITS server is not running${NC}"
        return 1
    fi
}

# Main
case "${1:-help}" in
    install)
        print_header
        if check_installation; then
            echo -e "${YELLOW}GPT-SoVITS is already installed at: $SOVITS_DIR${NC}"
            read -p "Reinstall? (y/N) " -n 1 -r
            echo
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                rm -rf "$SOVITS_DIR"
                install_sovits
            fi
        else
            install_sovits
        fi
        ;;
    start)
        print_header
        if ! check_installation; then
            echo -e "${RED}❌ GPT-SoVITS not found at: $SOVITS_DIR${NC}"
            echo -e "${YELLOW}Run '$0 install' first.${NC}"
            exit 1
        fi
        
        # Parse options
        shift
        port=9880
        ref_audio=""
        ref_text=""
        ref_lang="zh"
        
        while [[ $# -gt 0 ]]; do
            case $1 in
                -p|--port) port="$2"; shift 2 ;;
                -r|--ref) ref_audio="$2"; shift 2 ;;
                -t|--text) ref_text="$2"; shift 2 ;;
                -l|--lang) ref_lang="$2"; shift 2 ;;
                *) shift ;;
            esac
        done
        
        start_server "$port" "$ref_audio" "$ref_text" "$ref_lang"
        ;;
    status)
        check_status
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
