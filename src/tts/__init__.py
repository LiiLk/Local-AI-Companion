# TTS Module - Text-to-Speech implementations
from .base import BaseTTS, TTSResult
from .edge_tts_provider import EdgeTTSProvider

__all__ = ["BaseTTS", "TTSResult", "EdgeTTSProvider"]
