"""
Language Detection Utility.

Detects the language of a text to enable automatic voice selection.
Uses a hybrid approach:
1. Fast heuristic detection (patterns, common words)
2. Optional langdetect library for more accuracy

Supported languages: French (fr), English (en)
Default fallback: English
"""

import logging
import re
from enum import Enum
from typing import Literal

logger = logging.getLogger(__name__)


class LanguageCode(str, Enum):
    """Supported language codes."""
    
    FRENCH = "fr"
    ENGLISH = "en"
    
    def __str__(self) -> str:
        return self.value


# ============================================================================
# HEURISTIC PATTERNS
# ============================================================================

# Common French words that rarely appear in English
FRENCH_MARKERS = {
    # Articles and prepositions
    "le", "la", "les", "un", "une", "des", "du", "de", "au", "aux",
    "ce", "cette", "ces", "mon", "ma", "mes", "ton", "ta", "tes",
    "son", "sa", "ses", "notre", "votre", "leur", "leurs",
    # Common words
    "je", "tu", "il", "elle", "nous", "vous", "ils", "elles",
    "suis", "es", "est", "sommes", "êtes", "sont",
    "ai", "as", "avons", "avez", "ont",
    "mais", "ou", "et", "donc", "or", "ni", "car",
    "que", "qui", "quoi", "dont", "où",
    "ne", "pas", "plus", "jamais", "rien",
    "très", "bien", "mal", "peu", "beaucoup", "trop",
    "avec", "sans", "pour", "dans", "sur", "sous",
    "ici", "là", "oui", "non", "peut-être",
    "bonjour", "salut", "merci", "pardon", "excusez",
    "aujourd'hui", "demain", "hier", "maintenant",
    "comment", "pourquoi", "quand", "combien",
    # Verbs
    "faire", "avoir", "être", "aller", "venir", "voir",
    "veux", "peux", "dois", "vais", "voudrais", "pourrais",
    # French-specific contractions
    "c'est", "j'ai", "n'est", "qu'est", "d'accord",
}

# Common English words that rarely appear in French
ENGLISH_MARKERS = {
    # Articles and prepositions
    "the", "a", "an",
    # Pronouns
    "i", "you", "he", "she", "it", "we", "they",
    "my", "your", "his", "her", "its", "our", "their",
    # Common words
    "is", "are", "am", "was", "were", "been", "being",
    "have", "has", "had", "do", "does", "did",
    "will", "would", "could", "should", "might",
    "can", "may", "must", "shall",
    "and", "but", "or", "so", "yet",
    "that", "which", "who", "whom", "whose",
    "not", "no", "yes", "maybe",
    "very", "really", "quite", "too", "much",
    "with", "without", "for", "from", "about",
    "here", "there", "where", "when", "why", "how",
    "hello", "hi", "thanks", "sorry", "please",
    "today", "tomorrow", "yesterday", "now",
    "what", "because", "just", "also",
    # Verbs
    "get", "go", "come", "see", "know", "think",
    "want", "need", "like", "love", "hate",
    # Contractions
    "i'm", "you're", "he's", "she's", "it's", "we're", "they're",
    "i've", "you've", "we've", "they've",
    "i'll", "you'll", "he'll", "she'll", "we'll", "they'll",
    "don't", "doesn't", "didn't", "won't", "wouldn't",
    "can't", "couldn't", "shouldn't", "isn't", "aren't",
}

# French-specific character patterns
FRENCH_CHAR_PATTERN = re.compile(r"[àâäéèêëïîôùûüÿçœæ]", re.IGNORECASE)

# Apostrophe patterns typical of French
FRENCH_APOSTROPHE_PATTERN = re.compile(
    r"\b(c'|d'|j'|l'|m'|n'|qu'|s'|t')", 
    re.IGNORECASE
)


def detect_language(
    text: str,
    default: LanguageCode = LanguageCode.ENGLISH,
    use_langdetect: bool = False,
) -> LanguageCode:
    """
    Detect the language of a text.
    
    Uses fast heuristic detection by default. Can optionally use
    the langdetect library for better accuracy on ambiguous texts.
    
    Args:
        text: Text to analyze
        default: Default language if detection fails
        use_langdetect: Use langdetect library for better accuracy
        
    Returns:
        LanguageCode (fr or en)
        
    Examples:
        >>> detect_language("Bonjour, comment ça va ?")
        LanguageCode.FRENCH
        
        >>> detect_language("Hello, how are you?")
        LanguageCode.ENGLISH
        
        >>> detect_language("Salut!")
        LanguageCode.FRENCH
    """
    if not text or not text.strip():
        return default
    
    text_lower = text.lower().strip()
    
    # Try heuristic detection first (fast)
    result = _detect_heuristic(text_lower)
    
    if result is not None:
        logger.debug(f"Heuristic detection: '{text[:50]}...' -> {result}")
        return result
    
    # Use langdetect if enabled and heuristic is inconclusive
    if use_langdetect:
        result = _detect_with_langdetect(text)
        if result is not None:
            logger.debug(f"Langdetect detection: '{text[:50]}...' -> {result}")
            return result
    
    logger.debug(f"Language detection inconclusive for: '{text[:50]}...', using default: {default}")
    return default


def _detect_heuristic(text_lower: str) -> LanguageCode | None:
    """
    Detect language using heuristic patterns.
    
    Returns None if detection is inconclusive.
    """
    # Check for French-specific characters (very strong signal)
    french_chars = len(FRENCH_CHAR_PATTERN.findall(text_lower))
    if french_chars >= 2:
        return LanguageCode.FRENCH
    
    # Check for French apostrophe patterns (strong signal)
    french_apostrophes = len(FRENCH_APOSTROPHE_PATTERN.findall(text_lower))
    if french_apostrophes >= 1:
        return LanguageCode.FRENCH
    
    # Tokenize and count marker words
    # Simple tokenization: split on non-word characters
    words = set(re.findall(r"\b[\w']+\b", text_lower))
    
    french_score = len(words & FRENCH_MARKERS)
    english_score = len(words & ENGLISH_MARKERS)
    
    # Add weight for French characters
    french_score += french_chars * 2
    french_score += french_apostrophes * 3
    
    logger.debug(f"Scores - FR: {french_score}, EN: {english_score}")
    
    # Determine language based on scores
    if french_score > english_score and french_score >= 1:
        return LanguageCode.FRENCH
    elif english_score > french_score and english_score >= 1:
        return LanguageCode.ENGLISH
    
    # Single word detection for common greetings
    text_stripped = text_lower.strip("!?.,:;\"' ")
    if text_stripped in {"bonjour", "salut", "bonsoir", "coucou", "allô", "allo"}:
        return LanguageCode.FRENCH
    if text_stripped in {"hello", "hi", "hey", "goodbye", "bye", "thanks"}:
        return LanguageCode.ENGLISH
    
    return None  # Inconclusive


def _detect_with_langdetect(text: str) -> LanguageCode | None:
    """
    Detect language using the langdetect library.
    
    Requires: pip install langdetect
    """
    try:
        from langdetect import detect, DetectorFactory
        
        # Set seed for reproducibility
        DetectorFactory.seed = 0
        
        detected = detect(text)
        
        if detected == "fr":
            return LanguageCode.FRENCH
        elif detected == "en":
            return LanguageCode.ENGLISH
        else:
            # Other language detected, return None
            logger.debug(f"Langdetect detected '{detected}', not FR/EN")
            return None
            
    except ImportError:
        logger.warning("langdetect not installed. Install with: pip install langdetect")
        return None
    except Exception as e:
        logger.warning(f"Langdetect error: {e}")
        return None


def get_voice_for_language(
    language: LanguageCode,
    voice_mapping: dict[str, str],
) -> str:
    """
    Get the appropriate voice for a language.
    
    Args:
        language: Detected language code
        voice_mapping: Dict mapping language codes to voice names
                       e.g., {"fr": "ff_siwis", "en": "af_heart"}
    
    Returns:
        Voice name/ID for the given language
        
    Raises:
        KeyError: If no voice mapping exists for the language
    """
    return voice_mapping[str(language)]


# ============================================================================
# QUICK TEST
# ============================================================================

if __name__ == "__main__":
    # Test the detection
    test_texts = [
        ("Bonjour, comment ça va ?", LanguageCode.FRENCH),
        ("Hello, how are you?", LanguageCode.ENGLISH),
        ("Je suis très content de te voir", LanguageCode.FRENCH),
        ("I'm very happy to see you", LanguageCode.ENGLISH),
        ("Salut!", LanguageCode.FRENCH),
        ("Hi!", LanguageCode.ENGLISH),
        ("C'est une belle journée aujourd'hui", LanguageCode.FRENCH),
        ("It's a beautiful day today", LanguageCode.ENGLISH),
        ("Qu'est-ce que tu fais ?", LanguageCode.FRENCH),
        ("What are you doing?", LanguageCode.ENGLISH),
        ("Merci beaucoup pour ton aide", LanguageCode.FRENCH),
        ("Thanks a lot for your help", LanguageCode.ENGLISH),
    ]
    
    print("Language Detection Tests")
    print("=" * 60)
    
    for text, expected in test_texts:
        result = detect_language(text)
        status = "✅" if result == expected else "❌"
        print(f"{status} '{text[:40]:<40}' -> {result} (expected: {expected})")
