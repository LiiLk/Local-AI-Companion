# TTS Module - Text-to-Speech implementations
#
# ACTIVE Providers (currently used):
# - XTTSProvider   : Multilingual voice cloning, 17 languages, ~2.8GB VRAM ⭐ RECOMMENDED
# - KokoroProvider : Lightweight local 82M params, good quality, no voice cloning
# - EdgeTTSProvider: Microsoft cloud (free fallback)
#
# DEPRECATED Providers (kept for reference, not exported):
# - f5tts_provider.py    : English accent on French text despite "multilingual" claims
# - openaudio_provider.py: Too heavy (~4GB CPU), slow for our use case

from .base import BaseTTS, TTSResult, Voice
from .edge_tts_provider import EdgeTTSProvider
from .kokoro_provider import KokoroProvider
from .xtts_provider import XTTSProvider

__all__ = [
    "BaseTTS", 
    "TTSResult", 
    "Voice", 
    "EdgeTTSProvider", 
    "KokoroProvider",
    "XTTSProvider",
]
