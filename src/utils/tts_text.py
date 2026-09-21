"""Convert LLM rich text into plain spoken text for TTS.

The LLM sometimes emits Markdown (bold, headings, lists, links, code). The TTS
must never read those symbols aloud, and it must never lose words that merely
look like Markdown. This module removes presentation syntax while keeping every
word, and exposes a helper that also strips emotion/action markers first so the
two transformations never fight over the same characters.
"""

from __future__ import annotations

import re
from typing import Any

import emoji

_MARKDOWN_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_BOLD_RE = re.compile(r"\*\*(.+?)\*\*", re.DOTALL)
_BOLD_UNDERSCORE_RE = re.compile(r"__(.+?)__", re.DOTALL)
_ITALIC_RE = re.compile(r"(?<!\*)\*(?!\s)(.+?)(?<!\s)\*(?!\*)", re.DOTALL)
_ITALIC_UNDERSCORE_RE = re.compile(r"(?<!\w)_(?!\s)(.+?)(?<!\s)_(?!\w)", re.DOTALL)
_HEADING_RE = re.compile(r"(?m)^\s{0,3}#{1,6}\s*")
_LIST_MARKER_RE = re.compile(r"(?m)^\s*(?:[-*+]\s+|\d+\.(?!\d)\s*)")
_MULTI_NEWLINE_RE = re.compile(r"\n\s*\n+")
_SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([,.;:!?])")
_WHITESPACE_RE = re.compile(r"\s+")


def normalize_text_for_tts(text: str) -> str:
    """Return ``text`` as plain spoken prose, without Markdown syntax."""
    if not text:
        return ""

    result = _MARKDOWN_LINK_RE.sub(r"\1", text)
    result = _BOLD_RE.sub(r"\1", result)
    result = _BOLD_UNDERSCORE_RE.sub(r"\1", result)
    result = _ITALIC_RE.sub(r"\1", result)
    result = _ITALIC_UNDERSCORE_RE.sub(r"\1", result)
    result = _HEADING_RE.sub("", result)
    result = result.replace("*", "").replace("`", "")
    result = _LIST_MARKER_RE.sub("", result)
    result = emoji.replace_emoji(result, replace="")
    result = _MULTI_NEWLINE_RE.sub(" ", result)
    result = result.replace("\n", " ")
    result = _SPACE_BEFORE_PUNCT_RE.sub(r"\1", result)
    result = _WHITESPACE_RE.sub(" ", result)
    return result.strip()


def has_speakable_content(text: str) -> bool:
    """True when ``text`` still holds something worth synthesizing."""
    return any(char.isalnum() for char in text)


def prepare_text_for_tts(text: str, emotion_detector: Any | None = None) -> str:
    """Strip emotion markers first, then normalize Markdown for TTS."""
    if not text:
        return ""

    if emotion_detector is not None:
        strip_markers = getattr(emotion_detector, "strip_markers", None)
        if callable(strip_markers):
            text = strip_markers(text)

    return normalize_text_for_tts(text)