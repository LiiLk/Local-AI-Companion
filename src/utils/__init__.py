"""
Utility modules for LLM Assistant.
"""

from .language_detection import detect_language, LanguageCode
from .audio_analysis import analyze_audio_volumes, read_wav_pcm, calculate_audio_duration_ms
from .emotion_detector import (
    EmotionDetector, 
    EmotionConfig, 
    detect_emotion, 
    get_expression_for_text,
    strip_emotion_markers
)
from .character_loader import (
    get_available_characters,
    load_character_preset,
    resolve_character_config,
    get_character_info,
)

__all__ = [
    "detect_language", 
    "LanguageCode",
    "analyze_audio_volumes",
    "read_wav_pcm",
    "calculate_audio_duration_ms",
    "EmotionDetector",
    "EmotionConfig",
    "detect_emotion",
    "get_expression_for_text",
    "strip_emotion_markers",
    # Character management
    "get_available_characters",
    "load_character_preset",
    "resolve_character_config",
    "get_character_info",
]
