import wave
from pathlib import Path

from scripts.benchmark_asr import (
    BenchmarkResult,
    LiveSegment,
    WarmupResult,
    audio_duration_from_bytes,
    char_error_rate,
    normalize_text,
    summarize_results,
    wav_duration_seconds,
    word_error_rate,
    write_markdown_report,
    write_pcm16_wav,
)


def test_normalize_text_is_case_punctuation_and_accent_insensitive():
    assert normalize_text("Salut, comment \u00e7a va ?") == "salut comment ca va"


def test_word_and_char_error_rate():
    assert word_error_rate("salut comment ca va", "salut comment va") == 0.25
    assert char_error_rate("abc", "adc") == 1 / 3


def test_audio_duration_from_pcm16_bytes():
    assert audio_duration_from_bytes(b"\x00\x00" * 8000, 16000) == 0.5


def test_write_pcm16_wav_round_trips_duration(tmp_path: Path):
    audio_path = tmp_path / "segment.wav"
    write_pcm16_wav(audio_path, b"\x00\x00" * 8000, 16000)

    with wave.open(str(audio_path), "rb") as handle:
        assert handle.getnchannels() == 1
        assert handle.getsampwidth() == 2
        assert handle.getframerate() == 16000

    assert wav_duration_seconds(audio_path) == 0.5


def test_summarize_results_and_markdown_report(tmp_path: Path):
    segment = LiveSegment(
        id="live_001",
        path=tmp_path / "live_001.wav",
        sample_rate=16000,
        bytes_len=32000,
        audio_duration_sec=1.0,
        capture_start_ms=100.0,
        capture_end_ms=1300.0,
        gap_from_previous_end_ms=None,
        starts_before_previous_commit=None,
        vad_wall_ms=1200.0,
        commit_delay_ms=700,
        commit_ready_ms=1900.0,
        reference="Salut comment ca va",
        was_cut_off=False,
    )
    result = BenchmarkResult(
        segment_id="live_001",
        provider="whisper",
        model="small",
        language="fr",
        requested_language="fr",
        reference="Salut comment ca va",
        hypothesis="Salut comment va",
        audio_duration_sec=1.0,
        transcribe_ms=250.0,
        vad_end_to_transcript_ready_ms=950.0,
        realtime_factor=4.0,
        wer=0.25,
        cer=0.1,
        confidence=0.9,
        effective_device="cuda",
        effective_compute_type="float16",
        was_cut_off=False,
    )

    summary = summarize_results([result])
    output_path = tmp_path / "report.md"
    warmup = WarmupResult(
        provider="whisper",
        model="small",
        segment_id="live_001",
        run_index=1,
        transcribe_ms=1200.0,
    )
    write_markdown_report(output_path, [segment], [result], [warmup])
    output = output_path.read_text(encoding="utf-8")

    assert summary[0]["mean_wer"] == 0.25
    assert summary[0]["mean_wer_without_cutoffs"] == 0.25
    assert summary[0]["median_vad_end_to_transcript_ready_ms"] == 950.0
    assert "Live Voice Benchmark Results" in output
    assert "Segment Timing" in output
    assert "Warmup Runs" in output
    assert "live_001" in output
