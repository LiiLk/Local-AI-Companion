import sys
from types import SimpleNamespace

import numpy as np

from src.assistant.audio_service import AudioService


def test_audio_service_resample_uses_soxr_when_available(monkeypatch):
    calls = []

    def fake_resample(audio, src_rate, dst_rate):
        calls.append((audio.dtype, src_rate, dst_rate))
        return np.ones(5, dtype=np.float64)

    monkeypatch.setitem(sys.modules, "soxr", SimpleNamespace(resample=fake_resample))
    service = object.__new__(AudioService)

    result = service._resample_audio(np.zeros(10, dtype=np.float32), 48000, 16000)

    assert calls == [(np.dtype("float32"), 48000, 16000)]
    assert result.dtype == np.float32
    assert result.shape == (5,)


def test_audio_service_resample_falls_back_to_linear_interpolation(monkeypatch):
    monkeypatch.setitem(sys.modules, "soxr", None)
    service = object.__new__(AudioService)

    result = service._resample_audio(np.linspace(-1, 1, 9, dtype=np.float32), 48000, 16000)

    assert result.dtype == np.float32
    assert result.shape == (3,)
