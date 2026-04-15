"""
Utility modules for LLM Assistant.
"""

from .language_detection import (
    detect_language,
    get_language_name,
    LanguageCode,
    normalize_language_code,
)
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
from .config_loader import load_yaml_config

__all__ = [
    "detect_language", 
    "get_language_name",
    "LanguageCode",
    "normalize_language_code",
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
    "load_yaml_config",
]
