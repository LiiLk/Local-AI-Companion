"""Unit tests for the LIL-67 RVC hang diagnostic decision helper."""

from scripts.diag_rvc_hang import whisper_warmup_ok


def test_whisper_warmup_error_status_is_rejected():
    assert whisper_warmup_ok("error:CUDA out of memory") is False


def test_whisper_warmup_success_statuses_are_accepted():
    assert whisper_warmup_ok("warmup:' Hello there.'") is True
    assert whisper_warmup_ok("ok") is True
