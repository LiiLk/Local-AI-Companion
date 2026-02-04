# TTS Module - Text-to-Speech implementations
#
# Providers:
# - KokoroProvider  : Lightweight local 82M params, good quality, no voice cloning
# - EdgeTTSProvider : Microsoft cloud (free fallback)

from .base import BaseTTS, TTSResult, Voice
from .edge_tts_provider import EdgeTTSProvider
from .kokoro_provider import KokoroProvider

__all__ = [
    "BaseTTS",
    "TTSResult",
    "Voice",
    "EdgeTTSProvider",
    "KokoroProvider",
]
