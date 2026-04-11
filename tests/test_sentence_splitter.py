"""Tests for streaming sentence splitter."""

from src.utils.sentence_splitter import SentenceSplitter


class TestSentenceSplitter:
    def test_single_sentence(self):
        splitter = SentenceSplitter()
        splitter.feed("Hello world.")
        assert splitter.get_sentences() == ["Hello world."]

    def test_multiple_sentences(self):
        splitter = SentenceSplitter()
        splitter.feed("First. Second. Third.")
        assert splitter.get_sentences() == ["First.", "Second.", "Third."]

    def test_incremental_feeding(self):
        splitter = SentenceSplitter()
        splitter.feed("Hello ")
        assert splitter.get_sentences() == []
        splitter.feed("world. ")
        assert splitter.get_sentences() == ["Hello world."]
        splitter.feed("Next!")
        assert splitter.get_sentences() == ["Next!"]

    def test_question_mark(self):
        splitter = SentenceSplitter()
        splitter.feed("How are you? I'm fine.")
        sentences = splitter.get_sentences()
        assert sentences == ["How are you?", "I'm fine."]

    def test_exclamation(self):
        splitter = SentenceSplitter()
        splitter.feed("Wow! Amazing!")
        assert splitter.get_sentences() == ["Wow!", "Amazing!"]

    def test_preserves_chatterbox_tags(self):
        splitter = SentenceSplitter()
        splitter.feed("Ha [laugh] that's funny!")
        sentences = splitter.get_sentences()
        assert len(sentences) == 1
        assert "[laugh]" in sentences[0]

    def test_flush_incomplete(self):
        splitter = SentenceSplitter()
        splitter.feed("Hello world")
        assert splitter.get_sentences() == []
        remaining = splitter.flush()
        assert remaining == "Hello world"

    def test_french_text(self):
        splitter = SentenceSplitter()
        splitter.feed("Bonjour ! Comment ça va ? Très bien.")
        sentences = splitter.get_sentences()
        assert len(sentences) == 3

    def test_abbreviations_not_split(self):
        splitter = SentenceSplitter()
        splitter.feed("Dr. Smith is here.")
        sentences = splitter.get_sentences()
        # Should not split on "Dr."
        assert len(sentences) == 1

    def test_faster_first_response_emits_fragment_without_punctuation(self):
        splitter = SentenceSplitter(
            faster_first_response=True,
            first_fragment_length=18,
            first_fragment_words=4,
        )
        splitter.feed("Je peux t'aider tout de suite maintenant")
        assert splitter.get_sentences() == ["Je peux t'aider"]
        assert splitter.flush() == "tout de suite maintenant"

    def test_empty_input(self):
        splitter = SentenceSplitter()
        splitter.feed("")
        assert splitter.get_sentences() == []
