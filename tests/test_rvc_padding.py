"""Tests for trimming the silence RVC padding adds to short clips."""

import types

import numpy as np
import pytest
import soundfile as sf

import scripts.rvc_worker as rvc_worker
from src.rvc_padding import compute_trim_bounds


def _duration_sec(path) -> float:
    audio, sample_rate = sf.read(path, dtype="float32")
    return len(audio) / float(sample_rate)


class _FakeWorkerRVC:
    """Convert without a GPU while preserving the input duration."""

    def __init__(self, *, x_pad: int = 3, tgt_sr: int = 16000):
        self.config = types.SimpleNamespace(x_pad=x_pad)
        self.tgt_sr = tgt_sr
        self.outputfreq = tgt_sr
        self.NO_CHANGE = 2

    def __call__(self, input_path, **kwargs):
        audio, sample_rate = sf.read(input_path, dtype="float32")
        samples = int(round(len(audio) / float(sample_rate) * self.tgt_sr))
        return np.zeros(samples, dtype=np.float32)


def _run_worker_conversion(tmp_path, *, duration_sec, input_sr, tgt_sr):
    input_path = tmp_path / "input.wav"
    output_path = tmp_path / "output.wav"
    samples = int(round(duration_sec * input_sr))
    sf.write(input_path, np.zeros(samples, dtype=np.float32), input_sr)

    rvc_worker.run_conversion(
        _FakeWorkerRVC(tgt_sr=tgt_sr),
        {"input_path": str(input_path), "output_path": str(output_path)},
    )
    return output_path


def test_short_input_is_trimmed_back_to_original_duration(tmp_path):
    output_path = _run_worker_conversion(
        tmp_path, duration_sec=0.5, input_sr=16000, tgt_sr=16000
    )

    assert _duration_sec(output_path) == pytest.approx(0.5, abs=0.06)


def test_long_input_is_left_unchanged(tmp_path):
    output_path = _run_worker_conversion(
        tmp_path, duration_sec=5.0, input_sr=16000, tgt_sr=16000
    )

    assert _duration_sec(output_path) == pytest.approx(5.0, abs=0.06)


def test_trim_accounts_for_different_output_sample_rate(tmp_path):
    output_path = _run_worker_conversion(
        tmp_path, duration_sec=0.5, input_sr=16000, tgt_sr=32000
    )

    audio, sample_rate = sf.read(output_path, dtype="float32")
    assert sample_rate == 32000
    assert len(audio) / float(sample_rate) == pytest.approx(0.5, abs=0.06)


def test_compute_trim_bounds_converts_padding_to_output_samples():
    start, end = compute_trim_bounds(
        pad_left_samples=16000,
        pad_right_samples=16000,
        input_sample_rate=16000,
        output_sample_rate=32000,
        output_length=80000,
    )

    assert start == 32000
    assert end == 48000


def test_compute_trim_bounds_skips_when_output_too_short(caplog):
    start, end = compute_trim_bounds(
        pad_left_samples=40000,
        pad_right_samples=40000,
        input_sample_rate=16000,
        output_sample_rate=16000,
        output_length=100,
    )

    assert (start, end) == (0, 100)
    assert "too short" in caplog.text


def test_compute_trim_bounds_is_noop_without_padding():
    assert compute_trim_bounds(
        pad_left_samples=0,
        pad_right_samples=0,
        input_sample_rate=16000,
        output_sample_rate=24000,
        output_length=1234,
    ) == (0, 1234)