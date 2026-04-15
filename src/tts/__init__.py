# TTS Module - Text-to-Speech implementations
#
# Providers:
# - KokoroProvider  : Lightweight local 82M params, good quality, no voice cloning
# - EdgeTTSProvider : Microsoft cloud (free fallback)

from .base import BaseTTS, TTSResult, Voice
from .edge_tts_provider import EdgeTTSProvider
from .kokoro_provider import KokoroProvider
from .chatterbox_provider import ChatterboxTTSProvider
from .qwen3_tts_provider import Qwen3TTSProvider
from .routed_provider import RoutedTTSProvider

__all__ = [
    "BaseTTS",
    "TTSResult",
    "Voice",
    "EdgeTTSProvider",
    "KokoroProvider",
    "ChatterboxTTSProvider",
    "Qwen3TTSProvider",
    "RoutedTTSProvider",
]
