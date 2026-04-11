from src.tts.kokoro_provider import KokoroProvider


def test_kokoro_set_language_switches_to_recommended_french_voice():
    provider = KokoroProvider(voice="af_heart")

    provider.set_language("fr")

    assert provider.voice == "ff_siwis"
    assert provider.lang_code == "f"


def test_kokoro_set_language_switches_back_to_english_voice():
    provider = KokoroProvider(voice="ff_siwis")

    provider.set_language("en")

    assert provider.voice == "af_heart"
    assert provider.lang_code == "a"
