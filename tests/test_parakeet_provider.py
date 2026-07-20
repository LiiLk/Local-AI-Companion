"""Unit tests for the Parakeet ASR provider wiring (LIL-48).

These do not download or load the ONNX model — they cover factory dispatch,
automatic language handling, text coercion, and mocked transcribe paths.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.asr.parakeet_provider import (
    INSTALL_HINT,
    SUPPORTED_LANGUAGES,
    TARGET_SAMPLE_RATE,
    ParakeetASRProvider,
)
from src.assistant.pipeline_runtime import create_pipeline_asr


def test_factory_builds_parakeet_provider_when_available():
    with patch.object(ParakeetASRProvider, "is_available", return_value=True):
        asr, summary = create_pipeline_asr({"asr": {"provider": "parakeet"}})
    assert isinstance(asr, ParakeetASRProvider)
    assert asr.model_name == "nemo-parakeet-tdt-0.6b-v3"
    assert asr.quantization == "int8"
    assert "Parakeet" in summary
    assert "opt-in" in summary.lower()


def test_factory_honors_parakeet_config():
    with patch.object(ParakeetASRProvider, "is_available", return_value=True):
        asr, _ = create_pipeline_asr(
            {
                "asr": {
                    "provider": "parakeet",
                    "parakeet": {
                        "quantization": "fp16",
                        "model_name": "nemo-parakeet-tdt-0.6b-v3",
                        "providers": ["CUDAExecutionProvider"],
                        "sample_rate": 8000,
                    },
                }
            }
        )
    assert asr.quantization == "fp16"
    assert asr.providers == ["CUDAExecutionProvider"]
    assert asr.sample_rate == TARGET_SAMPLE_RATE


def test_factory_errors_clearly_when_onnx_asr_missing():
    with patch.object(ParakeetASRProvider, "is_available", return_value=False):
        with pytest.raises(ImportError, match="onnx-asr"):
            create_pipeline_asr({"asr": {"provider": "parakeet"}})


def test_default_factory_still_whisper():
    asr, summary = create_pipeline_asr({"asr": {}})
    assert "Whisper" in summary
    assert asr.model_size == "small"


def test_default_providers_are_cpu():
    """Default to CPU so onnxruntime does not probe missing CUDA/TensorRT DLLs."""
    asr = ParakeetASRProvider()
    assert asr.providers == ["CPUExecutionProvider"]


def test_supported_languages_cover_project_scope():
    langs = ParakeetASRProvider().get_supported_languages()
    assert {"fr", "en", "es"}.issubset(set(langs))
    assert langs == SUPPORTED_LANGUAGES
    assert langs is not SUPPORTED_LANGUAGES  # returns a copy


def test_model_info_before_load():
    info = ParakeetASRProvider().get_model_info()
    assert info["provider"] == "parakeet"
    assert info["loaded"] is False
    assert info["sample_rate"] == TARGET_SAMPLE_RATE
    assert info["status"].startswith("validated opt-in")


def test_cleanup_releases_model():
    asr = ParakeetASRProvider()
    asr._model = MagicMock()
    asr.cleanup()
    assert asr._model is None


def test_coerce_text_variants():
    assert ParakeetASRProvider._coerce_text(None) == ""
    assert ParakeetASRProvider._coerce_text("  hi  ") == "hi"
    assert ParakeetASRProvider._coerce_text(["a", "b"]) == "a b"
    obj = MagicMock()
    obj.text = " from-attr "
    assert ParakeetASRProvider._coerce_text(obj) == "from-attr"


def test_transcribe_numpy_uses_model_recognize():
    asr = ParakeetASRProvider()
    fake_model = MagicMock()
    fake_model.recognize.return_value = "  bonjour  "
    asr._model = fake_model

    audio = np.zeros(16000, dtype=np.float32)
    result = asr.transcribe(audio, language="fr")

    assert result.text == "bonjour"
    assert result.language is None
    assert result.duration == pytest.approx(1.0)
    fake_model.recognize.assert_called_once()
    called_audio = fake_model.recognize.call_args.args[0]
    assert np.array_equal(called_audio, audio)
    assert fake_model.recognize.call_args.kwargs == {"sample_rate": TARGET_SAMPLE_RATE}


def test_transcribe_file_does_not_forward_language(tmp_path):
    asr = ParakeetASRProvider()
    fake_model = MagicMock()
    fake_model.recognize.return_value = "bonjour"
    asr._model = fake_model
    audio_path = tmp_path / "sample.wav"
    audio_path.touch()

    result = asr.transcribe(audio_path, language="fr")

    assert result.text == "bonjour"
    assert result.language is None
    fake_model.recognize.assert_called_once_with(str(audio_path))


def test_transcribe_empty_buffer():
    asr = ParakeetASRProvider()
    asr._model = MagicMock()
    result = asr.transcribe(np.array([], dtype=np.float32))
    assert result.text == ""
    asr._model.recognize.assert_not_called()


def test_transcribe_missing_file():
    asr = ParakeetASRProvider()
    asr._model = MagicMock()
    with pytest.raises(FileNotFoundError):
        asr.transcribe(Path("/nonexistent/audio.wav"))


def test_get_model_raises_install_hint():
    asr = ParakeetASRProvider()

    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "onnx_asr" or name.startswith("onnx_asr."):
            raise ImportError("nope")
        return real_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=fake_import):
        with pytest.raises(ImportError, match="requirements-optional-parakeet"):
            asr._get_model()
    assert "onnx-asr is not installed" in INSTALL_HINT


def test_to_mono_float32_averages_channels():
    stereo = np.stack([np.ones(8), np.zeros(8)], axis=1).astype(np.float32)
    mono = ParakeetASRProvider._to_mono_float32(stereo)
    assert mono.shape == (8,)
    assert np.allclose(mono, 0.5)
