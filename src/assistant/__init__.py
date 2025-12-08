"""
Live2D Assistant Package.

Provides the unified application combining:
- Continuous microphone capture with VAD
- ASR → LLM → TTS conversation pipeline  
- Live2D avatar integration
- Hotkey controls
"""

from .audio_service import AudioService, AudioServiceConfig, MicState
from .conversation_pipeline import (
    ConversationPipeline,
    ConversationConfig,
    AudioPayload,
    EmotionDetector,
)
from .app import Live2DAssistant

__all__ = [
    'AudioService',
    'AudioServiceConfig',
    'MicState',
    'ConversationPipeline',
    'ConversationConfig',
    'AudioPayload',
    'EmotionDetector',
    'Live2DAssistant',
]
