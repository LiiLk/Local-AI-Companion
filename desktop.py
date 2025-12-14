#!/usr/bin/env python3
"""
Quick launcher for AI Desktop Companion.

Usage:
    python desktop.py              # Start companion only (requires backend running)
    python desktop.py --all        # Start backend + companion
    python desktop.py --help       # Show all options
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.desktop.__main__ import main

if __name__ == "__main__":
    main()
