"""
ASR (Automatic Speech Recognition) Module
Speech-to-Text conversion using local models
"""

from .base import BaseASR, BaseRealtimeASR, ASRResult, ASRSegment
from .whisper_provider import WhisperProvider, RealtimeWhisperProvider, create_whisper_asr

__all__ = [
    "BaseASR",
    "BaseRealtimeASR",
    "ASRResult",
    "ASRSegment",
    "WhisperProvider",
    "RealtimeWhisperProvider",
    "create_whisper_asr",
]
