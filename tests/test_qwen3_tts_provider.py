import tempfile
from pathlib import Path

import pytest

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


def test_qwen3_tts_customvoice_model_resolves_invalid_voice_clone_mode():
    mode, reason = Qwen3TTSProvider.resolve_mode_for_model(
        "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
        "voice_clone",
    )

    assert mode == "custom_voice"
    assert reason == "Qwen3-TTS CustomVoice model does not support voice_clone; using custom_voice."


def test_qwen3_tts_warmup_uses_speaker_native_language_for_custom_voice():
    provider = Qwen3TTSProvider(mode="custom_voice", speaker="Ryan", language="fr")

    assert provider._warmup_text() == "Hello."


def test_qwen3_tts_warmup_uses_french_when_router_hint_is_french():
    provider = Qwen3TTSProvider(mode="voice_clone", language="auto")
    provider.set_language("fr")

    assert provider._warmup_text() == "Bonjour."


def test_qwen3_tts_worker_import_check_rejects_missing_python():
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as handle:
        worker_script = Path(handle.name)
        handle.write(b"print('ok')")

    try:
        assert Qwen3TTSProvider._worker_import_check(
            worker_script.with_name("missing-python.exe"),
            worker_script,
        ) is False
    finally:
        worker_script.unlink(missing_ok=True)


def test_qwen3_tts_worker_validation_rejects_non_python_script():
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as handle:
        worker_script = Path(handle.name)

    try:
        with pytest.raises(RuntimeError, match="must be a Python file"):
            Qwen3TTSProvider._validate_worker_process_inputs(Path(__file__), worker_script)
    finally:
        worker_script.unlink(missing_ok=True)
