"""
Desktop Companion Module

Provides a desktop overlay application for the AI companion with:
- Transparent Live2D avatar window (always-on-top)
- System tray integration
- Global hotkeys
- WebSocket connection to backend AI
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .app import DesktopCompanion, DesktopConfig

__all__ = ["DesktopCompanion", "DesktopConfig"]


def __getattr__(name: str):
    """Load the optional tray application only when its public classes are requested."""
    if name in __all__:
        from .app import DesktopCompanion, DesktopConfig

        return {"DesktopCompanion": DesktopCompanion, "DesktopConfig": DesktopConfig}[name]
    raise AttributeError(name)
