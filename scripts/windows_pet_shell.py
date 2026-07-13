#!/usr/bin/env python3
"""
Windows-native desktop pet shell for Local AI Companion (LIL-49 option A).

Runs on Windows Python (PyQt6). Connects to an assistant bridge that may live
in WSL or on the same machine:

  WSL:    python run_assistant.py --bridge-server --bridge-port 8765
  Windows: python scripts/windows_pet_shell.py --bridge-url ws://127.0.0.1:8765

The shell is the same transparent / always-on-top / click-through Qt overlay used
by the full Windows desktop path — only the brain is remote.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description="Windows pet shell (bridge client)")
    parser.add_argument(
        "--bridge-url",
        default="ws://127.0.0.1:8765",
        help="Assistant bridge WebSocket URL (default: ws://127.0.0.1:8765)",
    )
    parser.add_argument(
        "--page-url",
        default=None,
        help="Optional HTTP URL for Live2D UI (default: local frontend/live2d/index.html)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=860,
    )
    parser.add_argument(
        "--height",
        type=int,
        default=760,
    )
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    log = logging.getLogger("windows_pet_shell")

    try:
        from desktop.qt_avatar_shell import QtAvatarShell
    except ImportError as exc:
        log.error(
            "PyQt6 shell unavailable (%s). On Windows install: "
            "pip install PyQt6 PyQt6-WebEngine",
            exc,
        )
        return 1

    from src.desktop.bridge_proxy import BridgeProxyAssistant

    html_path = PROJECT_ROOT / "frontend" / "live2d" / "index.html"
    if not html_path.is_file() and not args.page_url:
        log.error("Live2D HTML missing: %s (or pass --page-url)", html_path)
        return 1

    model = (
        PROJECT_ROOT
        / "frontend"
        / "live2d"
        / "runtime-assets"
        / "models"
        / "march7th_tauri"
        / "march7th.model3.json"
    )
    if not model.is_file() and not args.page_url:
        log.warning(
            "March 7th model pack not found at %s — avatar may not load",
            model,
        )

    log.info("Connecting to bridge %s ...", args.bridge_url)
    try:
        proxy = BridgeProxyAssistant(args.bridge_url)
    except Exception as exc:
        log.error("Bridge connection failed: %s", exc)
        log.error(
            "Is the WSL/backend bridge running? "
            "python run_assistant.py --bridge-server"
        )
        return 1

    shell = QtAvatarShell(
        proxy,
        html_path,
        width=args.width,
        height=args.height,
        page_url=args.page_url,
    )

    def on_event(name: str, event_args: tuple) -> None:
        # Inject backend events into the webview (same as in-process assistant).
        try:
            shell.dispatch_frontend_event(name, *event_args)
        except Exception as exc:  # noqa: BLE001
            log.debug("dispatch %s failed: %s", name, exc)

    proxy.set_event_handler(on_event)

    def on_loaded() -> None:
        log.info("Windows pet shell loaded")
        try:
            runtime = proxy.get_runtime_state()
            shell.dispatch_frontend_event("onBackendReady", runtime)
        except Exception as exc:  # noqa: BLE001
            log.debug("onBackendReady: %s", exc)

    log.info("Starting transparent desktop pet shell (Windows-native)")
    try:
        return int(shell.run(on_loaded) or 0)
    finally:
        proxy.close()


if __name__ == "__main__":
    raise SystemExit(main())
