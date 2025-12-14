# TTS Module - Text-to-Speech implementations
#
# Providers:
# - XTTSProvider    : Multilingual voice cloning, 17 languages, ~2.8GB VRAM ⭐ RECOMMENDED
# - KokoroProvider  : Lightweight local 82M params, good quality, no voice cloning
# - EdgeTTSProvider : Microsoft cloud (free fallback)
# - F5TTSProvider   : F5-TTS voice cloning (experimental)

from .base import BaseTTS, TTSResult, Voice
from .edge_tts_provider import EdgeTTSProvider
from .kokoro_provider import KokoroProvider
from .xtts_provider import XTTSProvider
from .f5_tts_provider import F5TTSProvider

__all__ = [
    "BaseTTS", 
    "TTSResult", 
    "Voice", 
    "EdgeTTSProvider", 
    "KokoroProvider",
    "XTTSProvider",
    "F5TTSProvider",
]
