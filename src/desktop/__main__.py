#!/usr/bin/env python3
"""
Desktop Companion Launcher

Starts the AI companion as a desktop overlay (transparent window with Live2D avatar).
Can also start the backend server automatically.

Usage:
    python -m src.desktop                    # Start companion only (backend must be running)
    python -m src.desktop --with-backend     # Start backend + companion
    python -m src.desktop --debug            # Debug mode
    python -m src.desktop --help             # Show help
"""

import argparse
import logging
import os
import subprocess
import sys
import time
import warnings
from pathlib import Path

# Suppress GTK-related warnings (we use Qt as fallback)
os.environ.setdefault('PYWEBVIEW_GUI', 'qt')  # Prefer Qt over GTK
warnings.filterwarnings('ignore', message='.*GTK.*')

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Reduce noise from websockets library
logging.getLogger('websockets').setLevel(logging.WARNING)


def check_dependencies():
    """Check if required dependencies are installed."""
    missing = []
    
    try:
        import webview
    except ImportError:
        missing.append("pywebview")
    
    try:
        import websockets
    except ImportError:
        missing.append("websockets")
    
    try:
        import yaml
    except ImportError:
        missing.append("pyyaml")
    
    # Optional but recommended
    optional_missing = []
    try:
        import pynput
    except ImportError:
        optional_missing.append("pynput (for hotkeys)")
    
    try:
        import pystray
        from PIL import Image
    except ImportError:
        optional_missing.append("pystray pillow (for system tray)")
    
    if missing:
        print("❌ Missing required dependencies:")
        for dep in missing:
            print(f"   - {dep}")
        print(f"\nInstall with: pip install {' '.join(missing)}")
        return False
    
    if optional_missing:
        print("⚠️ Optional dependencies not installed:")
        for dep in optional_missing:
            print(f"   - {dep}")
        print()
    
    return True


def check_backend_running(host: str = "localhost", port: int = 8000) -> bool:
    """Check if the backend server is running."""
    import socket
    
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1)
            s.connect((host, port))
            return True
    except (socket.timeout, ConnectionRefusedError):
        return False


def start_backend(host: str = "0.0.0.0", port: int = 8000):
    """Start the backend server in a subprocess."""
    logger.info(f"Starting backend server on {host}:{port}...")
    
    cmd = [
        sys.executable, "-m", "uvicorn",
        "src.server.app:app",
        "--host", host,
        "--port", str(port)
    ]
    
    process = subprocess.Popen(
        cmd,
        cwd=str(PROJECT_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT
    )
    
    # Wait for server to be ready
    for _ in range(30):
        if check_backend_running("localhost", port):
            logger.info("✅ Backend server is ready")
            return process
        time.sleep(0.5)
    
    logger.error("❌ Backend server failed to start")
    process.terminate()
    return None


def main():
    parser = argparse.ArgumentParser(
        description="AI Desktop Companion - A Live2D avatar assistant"
    )
    parser.add_argument(
        "--with-backend", "-b",
        action="store_true",
        help="Also start the backend server"
    )
    parser.add_argument(
        "--host",
        default="localhost",
        help="Backend host (default: localhost)"
    )
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=8000,
        help="Backend port (default: 8000)"
    )
    parser.add_argument(
        "--size",
        nargs=2,
        type=int,
        default=[400, 600],
        metavar=("W", "H"),
        help="Window size (default: 400 600)"
    )
    parser.add_argument(
        "--position",
        nargs=2,
        type=int,
        default=[-1, -1],
        metavar=("X", "Y"),
        help="Window position (-1 = auto)"
    )
    parser.add_argument(
        "--debug", "-d",
        action="store_true",
        help="Enable debug mode"
    )
    parser.add_argument(
        "--no-tray",
        action="store_true",
        help="Disable system tray icon"
    )
    parser.add_argument(
        "--no-hotkeys",
        action="store_true",
        help="Disable global hotkeys"
    )
    
    args = parser.parse_args()
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Import after dependency check
    from src.desktop.app import DesktopCompanion, DesktopConfig
    
    # Start backend if requested
    backend_process = None
    if args.with_backend:
        backend_process = start_backend("0.0.0.0", args.port)
        if not backend_process:
            sys.exit(1)
    else:
        # Check if backend is running
        if not check_backend_running(args.host, args.port):
            logger.warning(f"⚠️ Backend not detected at {args.host}:{args.port}")
            logger.info("Start with --with-backend or run the server manually:")
            logger.info("   python -m src.server")
            logger.info("")
    
    # Create config
    config = DesktopConfig(
        width=args.size[0],
        height=args.size[1],
        x=args.position[0],
        y=args.position[1],
        backend_host=args.host,
        backend_port=args.port,
        enable_tray=not args.no_tray,
        enable_hotkeys=not args.no_hotkeys,
        debug=args.debug
    )
    
    logger.info("=" * 50)
    logger.info("🎭 AI Desktop Companion")
    logger.info("=" * 50)
    logger.info(f"   Window: {config.width}x{config.height}")
    logger.info(f"   Backend: {config.backend_host}:{config.backend_port}")
    logger.info(f"   Hotkeys: F12=toggle, F11=mic")
    logger.info(f"   Tray: {'enabled' if config.enable_tray else 'disabled'}")
    logger.info("=" * 50)
    
    # Create and start companion
    companion = DesktopCompanion(config)
    
    try:
        companion.start()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Error: {e}")
        if args.debug:
            raise
    finally:
        if backend_process:
            logger.info("Stopping backend server...")
            backend_process.terminate()
            backend_process.wait(timeout=5)


if __name__ == "__main__":
    main()
