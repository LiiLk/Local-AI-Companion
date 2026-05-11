import logging

from src.utils.startup_profiler import StartupProfiler


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def test_startup_profiler_records_elapsed_and_delta_times():
    clock = FakeClock()
    profiler = StartupProfiler("desktop", clock=clock)

    clock.advance(0.125)
    first = profiler.mark("config_loaded")
    clock.advance(1.5)
    second = profiler.mark("preload_asr_done")

    assert first.elapsed_ms == 125
    assert first.delta_ms == 125
    assert second.elapsed_ms == 1625
    assert second.delta_ms == 1500
    assert profiler.steps() == [first, second]


def test_startup_profiler_summary_is_single_line(caplog):
    clock = FakeClock()
    profiler = StartupProfiler("desktop", clock=clock)

    clock.advance(0.1)
    profiler.mark("config_loaded")
    clock.advance(0.2)

    with caplog.at_level(logging.INFO):
        profiler.log_summary(logging.getLogger("startup-test"), status="ready")

    assert len(caplog.records) == 1
    message = caplog.records[0].message
    assert "\n" not in message
    assert "desktop startup profile status=ready total=300ms" in message
    assert "config_loaded=+100ms/@100ms" in message
