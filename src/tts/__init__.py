# TTS Module - Text-to-Speech implementations
from .base import BaseTTS, TTSResult, Voice
from .edge_tts_provider import EdgeTTSProvider
from .kokoro_provider import KokoroProvider

__all__ = ["BaseTTS", "TTSResult", "Voice", "EdgeTTSProvider", "KokoroProvider"]
