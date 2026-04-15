"""
Language detection and normalization helpers.

The runtime needs two different capabilities:
1. Normalize language hints coming from configs, ASR, and locales.
2. Detect the likely response language for short user text when ASR is absent.

The detection path intentionally stays simple:
- fast FR/EN heuristics for short desktop utterances
- langdetect as a fallback for broader multilingual coverage
"""

from __future__ import annotations

import logging
import re
from enum import Enum

logger = logging.getLogger(__name__)


class LanguageCode(str, Enum):
    """Common language codes used across the desktop runtime."""

    ARABIC = "ar"
    CHINESE = "zh"
    ENGLISH = "en"
    FRENCH = "fr"
    GERMAN = "de"
    HINDI = "hi"
    ITALIAN = "it"
    JAPANESE = "ja"
    KOREAN = "ko"
    PORTUGUESE = "pt"
    RUSSIAN = "ru"
    SPANISH = "es"
    TURKISH = "tr"

    def __str__(self) -> str:
        return self.value


LANGUAGE_NAMES: dict[str, str] = {
    "ar": "Arabic",
    "de": "German",
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "hi": "Hindi",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "pt": "Portuguese",
    "ru": "Russian",
    "tr": "Turkish",
    "zh": "Chinese",
}

LANGUAGE_ALIASES: dict[str, str] = {
    "arabic": "ar",
    "chinese": "zh",
    "deutsch": "de",
    "english": "en",
    "espanol": "es",
    "espanol (espana)": "es",
    "french": "fr",
    "francais": "fr",
    "german": "de",
    "hindi": "hi",
    "italian": "it",
    "japanese": "ja",
    "korean": "ko",
    "mandarin": "zh",
    "portuguese": "pt",
    "russian": "ru",
    "spanish": "es",
    "turkish": "tr",
}


FRENCH_MARKERS = {
    "bonjour",
    "cest",
    "comment",
    "merci",
    "oui",
    "pourquoi",
    "salut",
}
ENGLISH_MARKERS = {
    "hello",
    "how",
    "please",
    "thanks",
    "what",
    "why",
}
FRENCH_CHAR_PATTERN = re.compile(r"[a-z]*[àâäéèêëïîôùûüÿçœæ][a-z]*", re.IGNORECASE)
FRENCH_APOSTROPHE_PATTERN = re.compile(r"\b(c'|d'|j'|l'|m'|n'|qu'|s'|t')", re.IGNORECASE)


def _enum_or_code(code: str) -> LanguageCode | str:
    try:
        return LanguageCode(code)
    except ValueError:
        return code


def normalize_language_code(language: str | LanguageCode | None) -> str | None:
    """
    Normalize a language hint to a short ISO-like code.

    Examples:
    - ``fr`` -> ``fr``
    - ``fr-FR`` -> ``fr``
    - ``Portuguese`` -> ``pt``
    - ``auto`` -> ``None``
    """

    if language is None:
        return None

    value = str(language).strip().lower()
    if not value or value == "auto":
        return None

    alias = LANGUAGE_ALIASES.get(value)
    if alias:
        return alias

    if "-" in value:
        value = value.split("-", 1)[0]

    if value in LANGUAGE_NAMES:
        return value

    return value


def get_language_name(language: str | LanguageCode | None) -> str | None:
    code = normalize_language_code(language)
    if not code:
        return None
    return LANGUAGE_NAMES.get(code, code.upper())


def detect_language(
    text: str,
    default: str | LanguageCode = LanguageCode.ENGLISH,
    use_langdetect: bool = True,
) -> LanguageCode | str:
    """
    Detect the likely language of a text snippet.

    The function always returns a normalized short code when possible.
    """

    default_code = normalize_language_code(default) or "en"
    if not text or not text.strip():
        return _enum_or_code(default_code)

    text_lower = text.lower().strip()
    heuristic = _detect_heuristic(text_lower)
    if heuristic is not None:
        return heuristic

    if use_langdetect:
        detected = _detect_with_langdetect(text)
        if detected is not None:
            return detected

    return _enum_or_code(default_code)


def _detect_heuristic(text_lower: str) -> LanguageCode | None:
    french_chars = len(FRENCH_CHAR_PATTERN.findall(text_lower))
    french_apostrophes = len(FRENCH_APOSTROPHE_PATTERN.findall(text_lower))
    if french_chars >= 1 or french_apostrophes >= 1:
        return LanguageCode.FRENCH

    words = set(re.findall(r"\b[\w']+\b", text_lower))
    french_score = len(words & FRENCH_MARKERS) + french_chars * 2 + french_apostrophes * 3
    english_score = len(words & ENGLISH_MARKERS)

    if french_score > english_score and french_score >= 1:
        return LanguageCode.FRENCH
    if english_score > french_score and english_score >= 1:
        return LanguageCode.ENGLISH
    return None


def _detect_with_langdetect(text: str) -> LanguageCode | str | None:
    try:
        from langdetect import DetectorFactory, detect

        DetectorFactory.seed = 0
        detected = normalize_language_code(detect(text))
        if not detected:
            return None
        return _enum_or_code(detected)
    except ImportError:
        logger.debug("langdetect is not installed")
        return None
    except Exception as exc:
        logger.debug("langdetect error: %s", exc)
        return None


def get_voice_for_language(
    language: str | LanguageCode,
    voice_mapping: dict[str, str],
) -> str:
    """Resolve a voice from a language code."""

    code = normalize_language_code(language)
    if not code:
        raise KeyError("No normalized language code")
    return voice_mapping[code]


if __name__ == "__main__":
    samples = [
        ("Bonjour, comment ca va ?", "fr"),
        ("Hello, how are you?", "en"),
        ("Hola, que tal?", "es"),
        ("Guten Morgen", "de"),
    ]
    for text, expected in samples:
        result = str(detect_language(text))
        status = "OK" if result == expected else "FAIL"
        print(f"{status}: {text!r} -> {result} (expected {expected})")
