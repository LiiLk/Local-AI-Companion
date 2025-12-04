"""
Server Module - FastAPI backend with WebSocket support
"""

from .app import create_app
from .websocket import WebSocketManager

__all__ = [
    "create_app",
    "WebSocketManager",
]
