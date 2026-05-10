import sys
from types import SimpleNamespace

import numpy as np

from src.asr.whisper_provider import WhisperProvider


def test_whisper_provider_model_info_exposes_effective_device_after_cuda_fallback(monkeypatch):
    calls = []

    class FakeWhisperModel:
        def __init__(self, model_path, device, compute_type, cpu_threads):
            calls.append(
                {
                    "model_path": model_path,
                    "device": device,
                    "compute_type": compute_type,
                    "cpu_threads": cpu_threads,
                }
            )
            if device == "cuda":
                raise RuntimeError("cuDNN failed to initialize")

    monkeypatch.setitem(
        sys.modules,
        "faster_whisper",
        SimpleNamespace(WhisperModel=FakeWhisperModel),
    )

    provider = WhisperProvider(model_size="small", device="cuda", compute_type="float16")

    info_before = provider.get_model_info()
    assert info_before["device"] == "cuda"
    assert info_before["compute_type"] == "float16"
    assert info_before["effective_device"] is None
    assert info_before["effective_compute_type"] is None

    provider._get_model()

    info_after = provider.get_model_info()
    assert calls == [
        {
            "model_path": "small",
            "device": "cuda",
            "compute_type": "float16",
            "cpu_threads": 8,
        },
        {
            "model_path": "small",
            "device": "cpu",
            "compute_type": "int8",
            "cpu_threads": 8,
        },
    ]
    assert info_after["device"] == "cuda"
    assert info_after["compute_type"] == "float16"
    assert info_after["effective_device"] == "cpu"
    assert info_after["effective_compute_type"] == "int8"
    assert info_after["loaded"] is True


def test_whisper_provider_rejects_low_confidence_repetitive_hallucination():
    provider = WhisperProvider(model_size="small")

    class FakeModel:
        def transcribe(self, *_args, **_kwargs):
            segment = SimpleNamespace(
                start=0.0,
                end=2.0,
                text="ව" * 80,
                avg_logprob=-0.22,
                no_speech_prob=0.47,
            )
            info = SimpleNamespace(language="nn", language_probability=0.16, duration=2.0)
            return iter([segment]), info

    provider._model = FakeModel()

    result = provider.transcribe(np.zeros(16000, dtype=np.float32), language=None)

    assert result.text == ""
    assert result.language == "nn"
    assert result.confidence == 0.16


def test_whisper_provider_keeps_normal_low_confidence_short_text():
    provider = WhisperProvider(model_size="small")

    class FakeModel:
        def transcribe(self, *_args, **_kwargs):
            segment = SimpleNamespace(
                start=0.0,
                end=2.0,
                text="Salut comment ca va",
                avg_logprob=-0.22,
                no_speech_prob=0.20,
            )
            info = SimpleNamespace(language="fr", language_probability=0.22, duration=2.0)
            return iter([segment]), info

    provider._model = FakeModel()

    result = provider.transcribe(np.zeros(16000, dtype=np.float32), language=None)

    assert result.text == "Salut comment ca va"


def test_whisper_provider_keeps_segment_under_configured_no_speech_threshold():
    provider = WhisperProvider(model_size="small")

    class FakeModel:
        def transcribe(self, *_args, **_kwargs):
            segment = SimpleNamespace(
                start=0.0,
                end=1.2,
                text="I'm still here",
                avg_logprob=-0.20,
                no_speech_prob=0.55,
            )
            info = SimpleNamespace(language="en", language_probability=0.82, duration=1.2)
            return iter([segment]), info

    provider._model = FakeModel()

    result = provider.transcribe(np.zeros(16000, dtype=np.float32), language=None)

    assert result.text == "I'm still here"


def test_whisper_provider_rejects_repeated_clause_loop():
    provider = WhisperProvider(model_size="small")

    class FakeModel:
        def transcribe(self, *_args, **_kwargs):
            segment = SimpleNamespace(
                start=0.0,
                end=1.8,
                text=("C'est le plus... " * 14).strip(),
                avg_logprob=-0.16,
                no_speech_prob=0.43,
            )
            info = SimpleNamespace(language="fr", language_probability=0.39, duration=1.8)
            return iter([segment]), info

    provider._model = FakeModel()

    result = provider.transcribe(np.zeros(16000, dtype=np.float32), language=None)

    assert result.text == ""
