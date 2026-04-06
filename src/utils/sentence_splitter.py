"""
Streaming Sentence Splitter.

Accumulates tokens and yields complete sentences at boundary characters.
Designed for the streaming LLM -> TTS pipeline.
"""

import re


# Common abbreviations that should NOT trigger a split
ABBREVIATIONS = {"dr", "mr", "mrs", "ms", "prof", "sr", "jr", "st", "etc", "vs", "fig"}

# Sentence-ending pattern: punctuation followed by space or end-of-string
# We look for .!? followed by at least one whitespace character
SENTENCE_BOUNDARY = re.compile(r'([.!?]+)(\s+)')
CLAUSE_BOUNDARY = re.compile(r'([,;:])(\s+)')


class SentenceSplitter:
    """
    Accumulates streamed text and yields complete sentences.

    Usage:
        splitter = SentenceSplitter()
        for token in llm_stream:
            splitter.feed(token)
            for sentence in splitter.get_sentences():
                send_to_tts(sentence)
        # After stream ends:
        remaining = splitter.flush()
        if remaining:
            send_to_tts(remaining)
    """

    def __init__(
        self,
        min_length: int = 1,
        min_clause_length: int = 48,
        min_clause_words: int = 6,
        faster_first_response: bool = False,
    ):
        self._buffer = ""
        self._pending: list[str] = []
        self._min_length = min_length
        self._min_clause_length = min_clause_length
        self._min_clause_words = min_clause_words
        self._faster_first = faster_first_response
        self._sentence_count = 0

    def feed(self, text: str) -> None:
        """Add text to the buffer and extract complete sentences."""
        if not text:
            return

        self._buffer += text
        self._extract_sentences()

    def _is_abbreviation(self, text_before_punct: str) -> bool:
        """Check if the period belongs to an abbreviation."""
        words = text_before_punct.split()
        if not words:
            return False
        last_word = words[-1].lower().rstrip(".")
        return last_word in ABBREVIATIONS

    def _extract_sentences(self) -> None:
        """Extract complete sentences from the buffer."""
        # Strategy: scan for sentence boundaries (punctuation + whitespace).
        # If punctuation is just "." and the word before it is an abbreviation,
        # skip that boundary.
        search_start = 0
        while True:
            match = SENTENCE_BOUNDARY.search(self._buffer, pos=search_start)
            if not match:
                break

            punct = match.group(1)
            # End of the sentence = end of punctuation
            sentence_end = match.start() + len(punct)
            # We consume up to end of whitespace
            consume_end = match.end()

            # Check abbreviation: only for single period
            text_before = self._buffer[:match.start()]
            if punct == "." and self._is_abbreviation(text_before):
                # Not a real boundary, keep scanning past this match
                search_start = match.end()
                continue

            # Extract the sentence (everything up to and including punctuation)
            candidate = self._buffer[:sentence_end].strip()

            if len(candidate) >= self._min_length:
                self._pending.append(candidate)

            # Advance the buffer past the consumed portion
            self._buffer = self._buffer[consume_end:]
            search_start = 0  # Reset since buffer changed

        while True:
            match = CLAUSE_BOUNDARY.search(self._buffer)
            if not match:
                break

            clause_end = match.start() + len(match.group(1))
            consume_end = match.end()
            candidate = self._buffer[:clause_end].strip()
            word_count = len(candidate.split())

            # Lower thresholds for the first sentence to reduce time-to-first-audio
            if self._faster_first and self._sentence_count == 0:
                clause_min_len = 20
                clause_min_words = 3
            else:
                clause_min_len = self._min_clause_length
                clause_min_words = self._min_clause_words

            if len(candidate) < clause_min_len or word_count < clause_min_words:
                break

            self._pending.append(candidate)
            self._buffer = self._buffer[consume_end:]

        # Handle sentence at end of buffer (punctuation at very end, no trailing space).
        # Only emit if the buffer ends with sentence-ending punctuation.
        stripped = self._buffer.rstrip()
        if stripped and stripped[-1] in ".!?":
            # Check it's not an abbreviation
            if stripped[-1] == ".":
                text_before = stripped[:-1]
                if self._is_abbreviation(text_before):
                    return
            candidate = stripped
            if len(candidate) >= self._min_length:
                self._pending.append(candidate)
                self._buffer = self._buffer[self._buffer.rindex(stripped[-1]) + 1:]

    def get_sentences(self) -> list[str]:
        """Return and clear all complete sentences found so far."""
        sentences = self._pending
        self._sentence_count += len(sentences)
        self._pending = []
        return sentences

    def flush(self) -> str:
        """Return any remaining text in the buffer (incomplete sentence)."""
        remaining = self._buffer.strip()
        self._buffer = ""
        return remaining
