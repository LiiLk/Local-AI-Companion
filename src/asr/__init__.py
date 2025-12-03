"""
ASR (Automatic Speech Recognition) Module
Speech-to-Text conversion using local models
"""

from .base import BaseASR, BaseRealtimeASR, ASRResult, ASRSegment
from .whisper_provider import WhisperProvider, RealtimeWhisperProvider, create_whisper_asr
from .canary_provider import CanaryProvider, create_canary_asr
from .parakeet_provider import ParakeetProvider, create_parakeet_asr

__all__ = [
    "BaseASR",
    "BaseRealtimeASR",
    "ASRResult",
    "ASRSegment",
    "WhisperProvider",
    "RealtimeWhisperProvider",
    "create_whisper_asr",
    "CanaryProvider",
    "create_canary_asr",
    "ParakeetProvider",
    "create_parakeet_asr",
]
