"""
Live2D Assistant Package.

Provides the unified application combining:
- Continuous microphone capture with VAD
- ASR -> LLM -> TTS conversation pipeline
- Live2D avatar integration
- Hotkey controls

Keep imports lazy to avoid side effects from the desktop app module when a
caller only needs the pipeline or audio service helpers.
"""

from .audio_service import AudioService, AudioServiceConfig, MicState
from .conversation_pipeline import (
    ConversationPipeline,
    ConversationConfig,
    AudioPayload,
    EmotionDetector,
)

__all__ = [
    "AudioService",
    "AudioServiceConfig",
    "MicState",
    "ConversationPipeline",
    "ConversationConfig",
    "AudioPayload",
    "EmotionDetector",
    "Live2DAssistant",
]


def __getattr__(name):
    if name == "Live2DAssistant":
        from .app import Live2DAssistant

        return Live2DAssistant
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
