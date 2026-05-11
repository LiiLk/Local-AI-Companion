"""Small startup timing helper for human-readable boot diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import threading
import time
from typing import Callable


@dataclass(frozen=True)
class StartupStep:
    name: str
    elapsed_ms: float
    delta_ms: float


class StartupProfiler:
    """Collect lightweight checkpoints from multiple startup threads."""

    def __init__(
        self,
        label: str,
        *,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.label = label
        self._clock = clock
        now = self._clock()
        self._started_at = now
        self._last_mark_at = now
        self._steps: list[StartupStep] = []
        self._lock = threading.Lock()

    def mark(self, name: str) -> StartupStep:
        with self._lock:
            now = self._clock()
            step = StartupStep(
                name=name,
                elapsed_ms=(now - self._started_at) * 1000.0,
                delta_ms=(now - self._last_mark_at) * 1000.0,
            )
            self._last_mark_at = now
            self._steps.append(step)
            return step

    def total_ms(self) -> float:
        return (self._clock() - self._started_at) * 1000.0

    def steps(self) -> list[StartupStep]:
        with self._lock:
            return list(self._steps)

    def format_summary(self, *, status: str | None = None) -> str:
        with self._lock:
            total_ms = (self._clock() - self._started_at) * 1000.0
            parts = [
                f"{step.name}=+{step.delta_ms:.0f}ms/@{step.elapsed_ms:.0f}ms"
                for step in self._steps
            ]

        status_text = f" status={status}" if status else ""
        detail_text = " | " + " | ".join(parts) if parts else ""
        return f"{self.label} startup profile{status_text} total={total_ms:.0f}ms{detail_text}"

    def log_summary(
        self,
        logger: logging.Logger,
        *,
        status: str | None = None,
        level: int = logging.INFO,
    ) -> None:
        logger.log(level, self.format_summary(status=status))
