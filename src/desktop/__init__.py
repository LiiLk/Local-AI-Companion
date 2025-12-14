"""
Desktop Companion Module

Provides a desktop overlay application for the AI companion with:
- Transparent Live2D avatar window (always-on-top)
- System tray integration
- Global hotkeys
- WebSocket connection to backend AI
"""

from .app import DesktopCompanion, DesktopConfig

__all__ = ["DesktopCompanion", "DesktopConfig"]
