from src.server.routes import public_voice_label


def test_public_voice_label_keeps_plain_voice_ids():
    assert public_voice_label("ff_siwis", "kokoro") == "ff_siwis"


def test_public_voice_label_strips_windows_paths():
    assert (
        public_voice_label(
            r"C:\Users\Khalil\voices\private-reference.wav",
            "qwen3",
        )
        == "private-reference.wav"
    )


def test_public_voice_label_strips_posix_paths():
    assert (
        public_voice_label("/home/user/voices/private-reference.wav", "chatterbox")
        == "private-reference.wav"
    )


def test_public_voice_label_uses_fallback_for_empty_values():
    assert public_voice_label(None, "qwen3") == "qwen3"
