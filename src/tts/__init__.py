# TTS Module - Text-to-Speech implementations
#
# Providers:
# - CosyVoice3Provider: State-of-the-art 0.5B model, 9 languages, zero-shot cloning ⭐ BEST QUALITY
# - XTTSProvider      : Multilingual voice cloning, 17 languages, ~2.8GB VRAM
# - KokoroProvider    : Lightweight local 82M params, good quality, no voice cloning
# - EdgeTTSProvider   : Microsoft cloud (free fallback)
# - F5TTSProvider     : F5-TTS voice cloning (experimental)

from .base import BaseTTS, TTSResult, Voice
from .edge_tts_provider import EdgeTTSProvider
from .kokoro_provider import KokoroProvider
from .xtts_provider import XTTSProvider
from .f5_tts_provider import F5TTSProvider
from .cosyvoice3_provider import CosyVoice3Provider

__all__ = [
    "BaseTTS", 
    "TTSResult", 
    "Voice", 
    "EdgeTTSProvider", 
    "KokoroProvider",
    "XTTSProvider",
    "F5TTSProvider",
    "CosyVoice3Provider",
]
