"""
Live2D Assistant Package.

Provides the unified application combining:
- Continuous microphone capture with VAD
- ASR -> LLM -> TTS conversation pipeline
- Live2D avatar integration
- Hotkey controls

Keep imports lazy to avoid side effects from optional desktop/audio
dependencies when a caller only needs a lightweight submodule.
"""

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


_EXPORT_MODULES = {
    "AudioService": ".audio_service",
    "AudioServiceConfig": ".audio_service",
    "MicState": ".audio_service",
    "ConversationPipeline": ".conversation_pipeline",
    "ConversationConfig": ".conversation_pipeline",
    "AudioPayload": ".conversation_pipeline",
    "EmotionDetector": ".conversation_pipeline",
    "Live2DAssistant": ".app",
}


def __getattr__(name):
    module_name = _EXPORT_MODULES.get(name)
    if module_name:
        from importlib import import_module

        value = getattr(import_module(module_name, package=__name__), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(__all__))
