from src.tts.qwen3_tts_provider import Qwen3TTSProvider


def test_qwen3_tts_detects_french_from_text():
    provider = Qwen3TTSProvider(language="auto")

    assert provider._resolve_request_language("Bonjour, comment ça va ?") == "French"


def test_qwen3_tts_uses_language_hint_for_short_text():
    provider = Qwen3TTSProvider(language="auto")

    provider.set_language("fr")

    assert provider._resolve_request_language("OK.") == "French"


def test_qwen3_tts_respects_explicit_language_setting():
    provider = Qwen3TTSProvider(language="fr")

    assert provider._resolve_request_language("Hello there.") == "French"
