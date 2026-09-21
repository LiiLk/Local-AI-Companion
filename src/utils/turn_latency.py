"""Per-turn conversation latency instrumentation.

Pure stdlib helper that timestamps the stages of a single conversation turn
from the end of user speech up to the first audio played/sent.

Typical stage order::

    asr_done -> llm_first_token -> first_sentence
             -> tts_first_audio -> rvc_done -> first_audio_out

The tracker is intentionally dependency-free so any runtime path (desktop
pipeline, websocket server) can import and mark stages with a single call.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections import deque
from typing import Callable, Optional

logger = logging.getLogger(__name__)

# Canonical stage order used for the structured log line. Only stages that
# were actually marked appear in the line.
STAGE_ORDER: tuple[str, ...] = (
    "asr_done",
    "llm_first_token",
    "first_sentence",
    "tts_first_audio",
    "rvc_done",
    "first_audio_out",
)

# Stage used as the turn total; tts_first_audio is a fallback when the audio
# was synthesized but the "played/sent" marker never fired.
_TOTAL_STAGE = "first_audio_out"
_TOTAL_FALLBACK = "tts_first_audio"

_DEFAULT_HISTORY_SIZE = 200


def _percentile(values: list[float], percentile: float) -> float:
    """Nearest-rank percentile (no interpolation)."""
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = math.ceil(percentile / 100.0 * len(ordered)) - 1
    rank = max(0, min(rank, len(ordered) - 1))
    return ordered[rank]


class TurnLatencyTracker:
    """Collect the first occurrence of each latency stage for a turn.

    A single instance tracks one active turn at a time and keeps a bounded
    history of completed turns so a session summary can be reported later.
    """

    def __init__(
        self,
        *,
        clock: Callable[[], float] = time.perf_counter,
        history_size: int = _DEFAULT_HISTORY_SIZE,
    ) -> None:
        self._clock = clock
        self._lock = threading.Lock()
        self._history: deque[float] = deque(maxlen=max(1, int(history_size)))
        self._counter = 0
        self._turn_id: Optional[object] = None
        self._t0: Optional[float] = None
        self._marks: dict[str, float] = {}

    def start(self, turn_id: object = None, t0: Optional[float] = None) -> None:
        """Begin a new turn at speech end (``t0``) or now."""
        with self._lock:
            self._counter += 1
            self._turn_id = turn_id if turn_id is not None else self._counter
            self._t0 = t0 if t0 is not None else self._clock()
            self._marks = {}

    def ensure_started(self, turn_id: object = None, t0: Optional[float] = None) -> None:
        """Start a turn only if none is currently active."""
        with self._lock:
            if self._t0 is not None:
                return
            self._counter += 1
            self._turn_id = turn_id if turn_id is not None else self._counter
            self._t0 = t0 if t0 is not None else self._clock()
            self._marks = {}

    def mark(self, stage: str) -> None:
        """Record the first occurrence of ``stage``; later calls are ignored.

        Safe to call without an active turn (no-op).
        """
        with self._lock:
            if self._t0 is None or stage in self._marks:
                return
            self._marks[stage] = (self._clock() - self._t0) * 1000.0

    def finish(self) -> dict[str, float]:
        """Close the active turn, log one structured line and return the marks."""
        with self._lock:
            if self._t0 is None:
                return {}
            turn_id = self._turn_id
            marks = self._ordered_marks()
            self._turn_id = None
            self._t0 = None
            self._marks = {}

        if not marks:
            return {}

        total = marks.get(_TOTAL_STAGE)
        if total is None:
            total = marks.get(_TOTAL_FALLBACK)
        if total is not None:
            marks["total_to_first_audio_ms"] = total
            with self._lock:
                self._history.append(total)

        logger.info("%s", self._format_line(turn_id, marks))
        return marks

    def summary(self) -> dict[str, float]:
        """Return count, average and p95 of ``total_to_first_audio_ms``."""
        with self._lock:
            totals = list(self._history)
        if not totals:
            return {"count": 0, "avg_ms": 0.0, "p95_ms": 0.0}
        return {
            "count": len(totals),
            "avg_ms": sum(totals) / len(totals),
            "p95_ms": _percentile(totals, 95.0),
        }

    def log_summary(
        self,
        log: Optional[logging.Logger] = None,
        *,
        level: int = logging.INFO,
    ) -> None:
        """Log the accumulated session summary (no-op when empty)."""
        summary = self.summary()
        if not summary["count"]:
            return
        (log or logger).log(
            level,
            "turn_latency_summary turns=%d avg_ms=%.0f p95_ms=%.0f",
            summary["count"],
            summary["avg_ms"],
            summary["p95_ms"],
        )

    def _ordered_marks(self) -> dict[str, float]:
        ordered = {stage: self._marks[stage] for stage in STAGE_ORDER if stage in self._marks}
        for stage, value in self._marks.items():
            if stage not in ordered:
                ordered[stage] = value
        return ordered

    @staticmethod
    def _format_line(turn_id: object, marks: dict[str, float]) -> str:
        parts = [f"turn={turn_id}"]
        parts.extend(f"{stage}={value:.0f}" for stage, value in marks.items())
        return "turn_latency " + " ".join(parts)


_default_tracker: Optional[TurnLatencyTracker] = None
_default_lock = threading.Lock()


def get_turn_latency_tracker() -> TurnLatencyTracker:
    """Return the process-wide tracker shared by all turn paths."""
    global _default_tracker
    if _default_tracker is None:
        with _default_lock:
            if _default_tracker is None:
                _default_tracker = TurnLatencyTracker()
    return _default_tracker
