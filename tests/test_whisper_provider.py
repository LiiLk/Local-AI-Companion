from types import SimpleNamespace

import numpy as np

from src.asr.whisper_provider import WhisperProvider


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
