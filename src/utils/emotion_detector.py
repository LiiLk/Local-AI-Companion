"""
Emotion Detection for Live2D Expressions

Detects emotions from text (LLM output) and maps to Live2D expressions.
Supports explicit markers like (happy), *excited*, etc.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class EmotionConfig:
    """Configuration for emotion detection and expression mapping."""
    
    # Patterns to detect emotions in text
    patterns: list[str] = field(default_factory=lambda: [
        r'\((\w+)\)',     # (happy)
        r'\[(\w+)\]',     # [sad]
        r'\*(\w+)\*',     # *excited*
        r'<(\w+)>',       # <blush>
    ])
    
    # Emotion to expression mapping (model-specific)
    # This maps English emotion names to Live2D expression names
    mapping: dict[str, str] = field(default_factory=lambda: {
        # English → March 7th expressions (Chinese names)
        'happy': '星星',           # Stars (excited/happy)
        'excited': '比耶',         # Peace sign
        'joy': '星星',
        'sad': '哭',               # Cry
        'cry': '哭',
        'crying': '哭',
        'angry': '黑脸',           # Dark face
        'mad': '黑脸',
        'shy': '脸红',             # Blush
        'blush': '脸红',
        'embarrassed': '流汗',     # Sweat
        'nervous': '流汗',
        'sweat': '流汗',
        'worried': '流汗',
        'surprised': '比耶',       # Peace (reuse)
        'peace': '比耶',
        'photo': '照相',           # Camera
        'camera': '照相',
        'cover': '捂脸',           # Face cover
        'facepalm': '捂脸',
        # Neutral
        'neutral': 'neutral',
        'normal': 'neutral',
        'default': 'neutral',
    })
    
    # Default expression when no emotion detected
    default_expression: str = 'neutral'


class EmotionDetector:
    """
    Detects emotions from text and maps to Live2D expressions.
    
    Usage:
        detector = EmotionDetector()
        emotion = detector.detect("I'm so (happy) to see you!")
        expression = detector.get_expression(emotion)  # Returns "星星"
        clean_text = detector.strip_markers(text)  # For TTS
    """
    
    def __init__(self, config: Optional[EmotionConfig] = None):
        self.config = config or EmotionConfig()
        self._compiled_patterns = [
            re.compile(p, re.IGNORECASE) 
            for p in self.config.patterns
        ]
    
    def detect(self, text: str) -> Optional[str]:
        """
        Detect emotion from text using pattern matching.
        
        Args:
            text: Input text (usually LLM output)
            
        Returns:
            Detected emotion name (lowercase) or None
        """
        if not text:
            return None
        
        text_lower = text.lower()
        
        # Check for explicit emotion markers
        for pattern in self._compiled_patterns:
            matches = pattern.findall(text_lower)
            for match in matches:
                # Check if this emotion is in our mapping
                if match in self.config.mapping:
                    logger.debug(f"Detected emotion: {match}")
                    return match
        
        return None
    
    def get_expression(self, emotion: Optional[str]) -> str:
        """
        Get Live2D expression name for an emotion.
        
        Args:
            emotion: Emotion name (e.g., "happy", "sad")
            
        Returns:
            Expression name for Live2D (e.g., "星星", "哭")
        """
        if not emotion:
            return self.config.default_expression
        
        return self.config.mapping.get(
            emotion.lower(), 
            self.config.default_expression
        )
    
    def detect_and_get_expression(self, text: str) -> tuple[Optional[str], str]:
        """
        Detect emotion and return both emotion and expression name.
        
        Args:
            text: Input text
            
        Returns:
            Tuple of (emotion_name, expression_name)
        """
        emotion = self.detect(text)
        expression = self.get_expression(emotion)
        return emotion, expression
    
    def strip_markers(self, text: str) -> str:
        """
        Remove emotion markers from text for TTS.
        
        The TTS should not read "(happy)" or "*excited*" aloud.
        
        Args:
            text: Input text with emotion markers
            
        Returns:
            Clean text without markers
        """
        result = text
        for pattern in self._compiled_patterns:
            result = pattern.sub('', result)
        
        # Clean up extra whitespace
        result = re.sub(r'\s+', ' ', result).strip()
        return result


# Singleton instance for easy import
_default_detector: Optional[EmotionDetector] = None


def get_emotion_detector() -> EmotionDetector:
    """Get the default emotion detector instance."""
    global _default_detector
    if _default_detector is None:
        _default_detector = EmotionDetector()
    return _default_detector


def detect_emotion(text: str) -> Optional[str]:
    """Convenience function to detect emotion from text."""
    return get_emotion_detector().detect(text)


def get_expression_for_text(text: str) -> str:
    """Convenience function to get expression for text."""
    detector = get_emotion_detector()
    emotion = detector.detect(text)
    return detector.get_expression(emotion)


def strip_emotion_markers(text: str) -> str:
    """Convenience function to strip emotion markers."""
    return get_emotion_detector().strip_markers(text)
