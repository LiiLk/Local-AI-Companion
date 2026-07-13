#!/usr/bin/env python3
"""
🤖 Live2D AI Assistant Launcher

Quick start:
    ./run_assistant.py
    
With options:
    ./run_assistant.py --debug           # Enable debug mode
    ./run_assistant.py --headless        # Run without window (testing)
    ./run_assistant.py --config my.yaml  # Use custom config

Hotkeys:
    F2: Toggle mute/unmute microphone  
    F3: Interrupt current response
    Escape: Quit
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

if __name__ == "__main__":
    # LIL-49: configure WSLg audio/WebGL before sounddevice/Qt imports.
    from src.utils.platform_compat import ensure_wsl_desktop_env

    ensure_wsl_desktop_env()
    from src.assistant.app import main

    main()
