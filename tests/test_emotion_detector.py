"""Tests for EmotionDetector — especially Chatterbox tag preservation."""

from src.utils.emotion_detector import EmotionDetector, get_emotion_detector


class TestStripMarkersForTTS:
    """strip_markers_for_tts must preserve Chatterbox tags."""

    def test_preserves_laugh_tag(self):
        detector = EmotionDetector()
        result = detector.strip_markers_for_tts("Ha [laugh] that was funny!")
        assert "[laugh]" in result

    def test_preserves_chuckle_tag(self):
        detector = EmotionDetector()
        result = detector.strip_markers_for_tts("Well [chuckle] okay then")
        assert "[chuckle]" in result

    def test_preserves_sigh_tag(self):
        detector = EmotionDetector()
        result = detector.strip_markers_for_tts("Oh [sigh] fine")
        assert "[sigh]" in result

    def test_preserves_cough_tag(self):
        detector = EmotionDetector()
        result = detector.strip_markers_for_tts("Excuse me [cough]")
        assert "[cough]" in result

    def test_strips_emotion_brackets(self):
        detector = EmotionDetector()
        result = detector.strip_markers_for_tts("I'm [sad] today")
        assert "[sad]" not in result
        assert "today" in result

    def test_strips_parentheses_markers(self):
        detector = EmotionDetector()
        result = detector.strip_markers_for_tts("Hello (happy) world")
        assert "(happy)" not in result
        assert "Hello" in result
        assert "world" in result

    def test_strips_asterisk_markers(self):
        detector = EmotionDetector()
        result = detector.strip_markers_for_tts("Wow *excited* amazing")
        assert "*excited*" not in result

    def test_strips_angle_bracket_markers(self):
        detector = EmotionDetector()
        result = detector.strip_markers_for_tts("Oh <blush> hi")
        assert "<blush>" not in result

    def test_mixed_markers_and_chatterbox_tags(self):
        detector = EmotionDetector()
        text = "C'est genial *excited* j'adore [laugh] !"
        result = detector.strip_markers_for_tts(text)
        assert "[laugh]" in result
        assert "*excited*" not in result
        assert "genial" in result

    def test_cleans_whitespace(self):
        detector = EmotionDetector()
        result = detector.strip_markers_for_tts("Hello  (happy)  world")
        assert "  " not in result

    def test_case_insensitive_chatterbox_tags(self):
        detector = EmotionDetector()
        result = detector.strip_markers_for_tts("Ha [LAUGH] funny")
        assert "[LAUGH]" in result


class TestDetectEmotion:
    """Existing detect/get_expression behavior should not break."""

    def test_detect_happy(self):
        detector = EmotionDetector()
        emotion = detector.detect("I'm so (happy) today!")
        assert emotion == "happy"

    def test_detect_returns_none_for_no_emotion(self):
        detector = EmotionDetector()
        assert detector.detect("Just a normal sentence.") is None

    def test_get_expression_happy(self):
        detector = EmotionDetector()
        expr = detector.get_expression("happy")
        assert expr == "星星"

    def test_get_expression_none_returns_default(self):
        detector = EmotionDetector()
        expr = detector.get_expression(None)
        assert expr == "neutral"
