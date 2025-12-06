"""
Vision Module - Screen capture and visual perception.

This module provides visual perception capabilities:
- Screen capture (screenshots, screen recording)
- Camera input (webcam)
- Image analysis via multimodal LLMs

Architecture follows the Strategy pattern with abstract base class.
"""

from .base import VisionProvider, CaptureResult
from .screen import ScreenCaptureProvider

__all__ = [
    "VisionProvider",
    "CaptureResult",
    "ScreenCaptureProvider",
]
