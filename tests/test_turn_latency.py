"""Tests for per-turn latency instrumentation (src/utils/turn_latency.py)."""

import asyncio
import contextvars
import logging

import pytest

from src.utils.turn_latency import TurnLatencyTracker


class FakeClock:
    """Deterministic, manually advanced clock for latency tests."""

    def __init__(self, start: float = 0.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def test_records_first_occurrence_only():
    clock = FakeClock()
    tracker = TurnLatencyTracker(clock=clock)
    tracker.start(turn_id=1)

    clock.advance(0.1)
    tracker.mark("asr_done")
    clock.advance(0.5)
    tracker.mark("asr_done")  # ignored: first occurrence wins
    clock.advance(0.2)
    tracker.mark("llm_first_token")

    result = tracker.finish()
    assert result["asr_done"] == pytest.approx(100.0)
    assert result["llm_first_token"] == pytest.approx(800.0)


def test_mark_without_start_is_noop():
    clock = FakeClock()
    tracker = TurnLatencyTracker(clock=clock)

    tracker.mark("asr_done")  # must not raise

    assert tracker.finish() == {}


def test_finish_without_start_returns_empty():
    tracker = TurnLatencyTracker(clock=FakeClock())
    assert tracker.finish() == {}


def test_missing_stages_absent_from_line(caplog):
    clock = FakeClock()
    tracker = TurnLatencyTracker(clock=clock)
    tracker.start(turn_id=3)

    clock.advance(0.2)
    tracker.mark("asr_done")
    clock.advance(0.5)
    tracker.mark("first_audio_out")

    with caplog.at_level(logging.INFO, logger="src.utils.turn_latency"):
        tracker.finish()

    line = caplog.records[-1].message
    assert "first_sentence" not in line
    assert "rvc_done" not in line
    assert "asr_done=200" in line
    assert "first_audio_out=700" in line


def test_log_line_format_and_order(caplog):
    clock = FakeClock()
    tracker = TurnLatencyTracker(clock=clock)
    tracker.start(turn_id=12)

    for stage, delta in (
        ("asr_done", 0.18),
        ("llm_first_token", 0.46),
        ("first_sentence", 0.27),
        ("tts_first_audio", 0.24),
        ("rvc_done", 0.14),
        ("first_audio_out", 0.02),
    ):
        clock.advance(delta)
        tracker.mark(stage)

    with caplog.at_level(logging.INFO, logger="src.utils.turn_latency"):
        result = tracker.finish()

    expected = (
        "turn_latency turn=12 asr_done=180 llm_first_token=640 "
        "first_sentence=910 tts_first_audio=1150 rvc_done=1290 "
        "first_audio_out=1310 total_to_first_audio_ms=1310"
    )
    assert caplog.records[-1].message == expected
    assert result["total_to_first_audio_ms"] == pytest.approx(1310.0)
    assert len(caplog.records) == 1


def test_total_falls_back_to_tts_first_audio():
    clock = FakeClock()
    tracker = TurnLatencyTracker(clock=clock)
    tracker.start(turn_id=5)

    clock.advance(0.9)
    tracker.mark("tts_first_audio")

    result = tracker.finish()
    assert result["total_to_first_audio_ms"] == pytest.approx(900.0)


def test_summary_avg_and_p95():
    clock = FakeClock()
    tracker = TurnLatencyTracker(clock=clock)

    totals_ms = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    for index, total_ms in enumerate(totals_ms):
        tracker.start(turn_id=index)
        clock.advance(total_ms / 1000.0)
        tracker.mark("first_audio_out")
        tracker.finish()

    summary = tracker.summary()
    assert summary["count"] == 10
    assert summary["avg_ms"] == pytest.approx(550.0)
    # Nearest-rank p95 of 10 values: ceil(0.95 * 10) - 1 = index 9 -> 1000 ms.
    assert summary["p95_ms"] == pytest.approx(1000.0)


def test_summary_empty():
    tracker = TurnLatencyTracker(clock=FakeClock())
    summary = tracker.summary()
    assert summary["count"] == 0
    assert summary["avg_ms"] == 0.0
    assert summary["p95_ms"] == 0.0


def test_history_is_bounded():
    clock = FakeClock()
    tracker = TurnLatencyTracker(clock=clock, history_size=3)

    for index in range(5):
        tracker.start(turn_id=index)
        clock.advance(0.1)
        tracker.mark("first_audio_out")
        tracker.finish()

    assert tracker.summary()["count"] == 3


def test_log_summary(caplog):
    clock = FakeClock()
    tracker = TurnLatencyTracker(clock=clock)
    tracker.start(turn_id=1)
    clock.advance(0.42)
    tracker.mark("first_audio_out")
    tracker.finish()

    with caplog.at_level(logging.INFO, logger="src.utils.turn_latency"):
        tracker.log_summary()

    assert "turn_latency_summary turns=1" in caplog.records[-1].message
    assert "avg_ms=420" in caplog.records[-1].message


def test_explicit_t0_is_respected():
    clock = FakeClock(start=100.0)
    tracker = TurnLatencyTracker(clock=clock)
    tracker.start(turn_id=1, t0=99.5)

    clock.advance(0.1)
    tracker.mark("asr_done")

    assert tracker.finish()["asr_done"] == pytest.approx(600.0)


def test_auto_turn_id(caplog):
    clock = FakeClock()
    tracker = TurnLatencyTracker(clock=clock)
    tracker.start()
    clock.advance(0.05)
    tracker.mark("first_audio_out")

    with caplog.at_level(logging.INFO, logger="src.utils.turn_latency"):
        tracker.finish()

    assert "turn=1" in caplog.records[-1].message


def test_ensure_started_does_not_reset_active_turn():
    clock = FakeClock()
    tracker = TurnLatencyTracker(clock=clock)
    tracker.start(turn_id=1)
    clock.advance(0.3)

    tracker.ensure_started(turn_id=99)  # must be ignored while active
    clock.advance(0.2)
    tracker.mark("asr_done")

    assert tracker.finish()["asr_done"] == pytest.approx(500.0)


def test_ensure_started_starts_when_idle():
    clock = FakeClock()
    tracker = TurnLatencyTracker(clock=clock)
    tracker.ensure_started(turn_id=7)
    clock.advance(0.25)
    tracker.mark("asr_done")

    assert tracker.finish()["asr_done"] == pytest.approx(250.0)


def test_overlapping_turns_are_isolated_per_context():
    """Two concurrent turns (one per context) must not clobber each other."""
    clock = FakeClock()
    tracker = TurnLatencyTracker(clock=clock)

    context_a = contextvars.copy_context()
    context_b = contextvars.copy_context()

    context_a.run(lambda: tracker.start(turn_id="A"))

    def _run_b():
        tracker.start(turn_id="B")
        clock.advance(0.2)
        tracker.mark("asr_done")
        return tracker.finish()

    result_b = context_b.run(_run_b)

    def _mark_a():
        clock.advance(0.1)
        tracker.mark("asr_done")

    context_a.run(_mark_a)
    result_a = context_a.run(tracker.finish)

    assert result_a["asr_done"] == pytest.approx(300.0)
    assert result_b["asr_done"] == pytest.approx(200.0)


def test_mark_from_child_task_targets_the_starting_turn():
    """A mark emitted from a child asyncio task belongs to the parent turn."""
    clock = FakeClock()
    tracker = TurnLatencyTracker(clock=clock)

    async def main():
        tracker.start(turn_id="desktop")

        async def worker():
            clock.advance(0.3)
            tracker.mark("tts_first_audio")

        await asyncio.create_task(worker())
        clock.advance(0.2)
        tracker.mark("first_audio_out")
        return tracker.finish()

    result = asyncio.run(main())

    assert result["tts_first_audio"] == pytest.approx(300.0)
    assert result["first_audio_out"] == pytest.approx(500.0)
