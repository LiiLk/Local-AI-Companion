"""Tests for plain-text normalization applied to TTS input."""

import pytest

from src.utils.emotion_detector import EmotionDetector
from src.utils.tts_text import (
    has_speakable_content,
    normalize_text_for_tts,
    prepare_text_for_tts,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("**1.", "1."),
        ("> Quoted text", "Quoted text"),
        (">Quoted text", "Quoted text"),
        ("Keep ~~this~~ word.", "Keep this word."),
        (
            "Battery Pack (Energy Storage)**  \nThe battery lasts.",
            "Battery Pack (Energy Storage) The battery lasts.",
        ),
        ("**Formation:** When a star collapses.", "Formation: When a star collapses."),
        (
            "- **Singularity:** The infinitely dense core.",
            "Singularity: The infinitely dense core.",
        ),
        (
            "- **Ergosphere** (for rotating black holes): A region.",
            "Ergosphere (for rotating black holes): A region.",
        ),
        (
            "- **Intermediate-mass** and **primordial** black holes.",
            "Intermediate-mass and primordial black holes.",
        ),
        ("an electric vehicle (EV) .", "an electric vehicle (EV)."),
        (
            "See [this link](https://example.com) now.",
            "See this link now.",
        ),
        ("Use `code` here.", "Use code here."),
        ("# Title\n\nSome text", "Title Some text"),
        ("1. First item\n2. Second item", "First item Second item"),
    ],
)
def test_normalize_text_for_tts_keeps_words(raw, expected):
    assert normalize_text_for_tts(raw) == expected


def test_has_speakable_content_rejects_marker_only_segments():
    assert has_speakable_content(normalize_text_for_tts("#")) is False
    assert has_speakable_content(normalize_text_for_tts("Hello")) is True


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("2026. A new era began.", "2026. A new era began."),
        ("42.", "42."),
        ("10. Tenth item", "Tenth item"),
        ("1. First item\n2. Second item", "First item Second item"),
    ],
)
def test_normalize_text_for_tts_preserves_numeric_sentences(raw, expected):
    assert normalize_text_for_tts(raw) == expected


def test_prepare_text_for_tts_keeps_line_structure_before_markdown_normalization():
    result = prepare_text_for_tts(
        "Intro.\n## Details\n1. One\n2. Two", EmotionDetector()
    )

    assert result == "Intro. Details One Two"


class _Detector:
    def strip_markers(self, text: str) -> str:
        return text.replace("*giggles*", "").replace("*", "")


def test_prepare_text_for_tts_strips_markers_before_normalizing():
    result = prepare_text_for_tts("Heh *giggles* **Ergosphere**", _Detector())

    assert "giggles" not in result
    assert "Ergosphere" in result
    assert "**" not in result