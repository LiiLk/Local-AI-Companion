"""Tests for speculative ASR on speech end (LIL-69, desktop path only).

The desktop commit window delays ASR by 160-900 ms after VAD reports speech
end. When the assistant is at rest we start the Whisper transcription as soon
as the window opens, so the ASR overlaps the window, then hand the finished
result to ``ConversationPipeline.process_speech`` when the audio is unchanged.
The WebSocket path is out of scope for this PR.
"""

import asyncio
import contextlib
import logging
import threading
import time
from concurrent.futures import Future as ConcurrentFuture

import pytest

from src.assistant.app import Live2DAssistant, _speculative_saved_ms
from src.assistant.conversation_pipeline import ConversationConfig, ConversationPipeline
from src.asr.base import ASRResult
from src.vad.smart_turn import SmartTurnConfig


class _LLM:
    def __init__(self):
        self.calls = []

    async def chat_stream(self, messages):
        self.calls.append(messages)
        yield "ok"


class _TTS:
    async def synthesize(self, text, output_path=None):
        return None

    def set_language(self, language):
        pass


class _CountingASR:
    """ASR stub that counts how many times the model was actually called."""

    def __init__(self, text="hello", language="en", confidence=0.9):
        self.calls = 0
        self.text = text
        self.language = language
        self.confidence = confidence

    def transcribe(self, audio, language=None):
        self.calls += 1
        return ASRResult(
            text=self.text,
            language=self.language,
            confidence=self.confidence,
            duration=2.0,
            segments=[
                {
                    "text": self.text,
                    "start": 0.0,
                    "end": 1.0,
                    "confidence": self.confidence,
                }
            ],
        )


def _audio_bytes(seconds=2.0):
    return b"\x00\x00" * int(16000 * seconds)


def _pipeline(asr):
    return ConversationPipeline(
        llm=_LLM(),
        tts=_TTS(),
        asr=asr,
        config=ConversationConfig(
            stream_tts=False, asr_language="en", reply_language="en"
        ),
    )


# ---------------------------------------------------------------------------
# ConversationPipeline
# ---------------------------------------------------------------------------


def test_transcribe_speech_delegates_to_transcribe():
    asr = _CountingASR()
    pipeline = _pipeline(asr)

    result = asyncio.run(pipeline.transcribe_speech(_audio_bytes()))

    assert result.text == "hello"
    assert asr.calls == 1


def test_process_speech_uses_speculative_transcription():
    asr = _CountingASR()
    pipeline = _pipeline(asr)

    async def scenario():
        speculative = asyncio.get_running_loop().create_future()
        speculative.set_result(
            ASRResult(
                text="hello",
                language="en",
                confidence=0.9,
                duration=2.0,
                segments=[
                    {"text": "hello", "start": 0.0, "end": 1.0, "confidence": 0.9}
                ],
            )
        )
        return await pipeline.process_speech(
            _audio_bytes(), speculative_transcription=speculative
        )

    result = asyncio.run(scenario())

    assert result == "ok"
    assert asr.calls == 0
    assert pipeline.llm.calls


def test_process_speech_falls_back_when_speculation_raises(caplog):
    asr = _CountingASR()
    pipeline = _pipeline(asr)

    async def scenario():
        speculative = asyncio.get_running_loop().create_future()
        speculative.set_exception(RuntimeError("speculation exploded"))
        return await pipeline.process_speech(
            _audio_bytes(), speculative_transcription=speculative
        )

    with caplog.at_level(logging.WARNING):
        result = asyncio.run(scenario())

    assert result == "ok"
    assert asr.calls == 1
    assert any(
        "speculative" in record.getMessage().lower() for record in caplog.records
    )


def test_process_speech_abandons_turn_when_speculation_is_none():
    asr = _CountingASR()
    pipeline = _pipeline(asr)

    async def scenario():
        speculative = asyncio.get_running_loop().create_future()
        speculative.set_result(None)
        return await pipeline.process_speech(
            _audio_bytes(), speculative_transcription=speculative
        )

    result = asyncio.run(scenario())

    assert result is None
    assert asr.calls == 0
    assert pipeline.llm.calls == []


def test_process_speech_accepts_concurrent_future():
    asr = _CountingASR()
    pipeline = _pipeline(asr)
    speculative = ConcurrentFuture()
    speculative.set_result(
        ASRResult(
            text="hello",
            language="en",
            confidence=0.9,
            duration=2.0,
            segments=[{"text": "hello", "start": 0.0, "end": 1.0, "confidence": 0.9}],
        )
    )

    result = asyncio.run(
        pipeline.process_speech(_audio_bytes(), speculative_transcription=speculative)
    )

    assert result == "ok"
    assert asr.calls == 0


class _OverlapASR:
    """ASR stub that detects two concurrent ``transcribe`` calls."""

    def __init__(self):
        self._lock = threading.Lock()
        self.active = 0
        self.max_active = 0
        self.calls = 0

    def transcribe(self, audio, language=None):
        with self._lock:
            self.active += 1
            self.calls += 1
            self.max_active = max(self.max_active, self.active)
        time.sleep(0.05)
        with self._lock:
            self.active -= 1
        return ASRResult(
            text="hello",
            language="en",
            confidence=0.9,
            duration=2.0,
            segments=[{"text": "hello", "start": 0.0, "end": 1.0, "confidence": 0.9}],
        )


def test_concurrent_transcribe_once_is_serialized():
    asr = _OverlapASR()
    pipeline = _pipeline(asr)

    async def scenario():
        await asyncio.gather(
            pipeline._transcribe_once(_audio_bytes(), "en"),
            pipeline._transcribe_once(_audio_bytes(), "en"),
        )

    asyncio.run(scenario())

    assert asr.calls == 2
    assert asr.max_active == 1


class _GatedASR:
    """ASR stub that blocks inside ``transcribe`` until it is released."""

    def __init__(self):
        self._lock = threading.Lock()
        self.entered = threading.Event()
        self.release = threading.Event()
        self.active = 0
        self.max_active = 0
        self.calls = 0

    def transcribe(self, audio, language=None):
        with self._lock:
            self.active += 1
            self.calls += 1
            self.max_active = max(self.max_active, self.active)
        self.entered.set()
        self.release.wait(timeout=5.0)
        with self._lock:
            self.active -= 1
        return ASRResult(
            text="hello",
            language="en",
            confidence=0.9,
            duration=2.0,
            segments=[{"text": "hello", "start": 0.0, "end": 1.0, "confidence": 0.9}],
        )


def test_cancelled_transcribe_keeps_asr_lock_until_executor_finishes():
    """A barge-in cancelling the awaited speculation must not let the next
    pass enter the GPU while the first executor thread is still running."""
    asr = _GatedASR()
    pipeline = _pipeline(asr)

    async def scenario():
        loop = asyncio.get_running_loop()
        first = asyncio.create_task(pipeline._transcribe_once(_audio_bytes(), "en"))
        assert await loop.run_in_executor(None, asr.entered.wait, 2.0)

        first.cancel()
        second = asyncio.create_task(pipeline._transcribe_once(_audio_bytes(), "en"))
        await asyncio.sleep(0.1)

        # The cancelled pass still holds the lock while its executor thread
        # runs, so the second pass must not have reached ``transcribe``.
        assert asr.calls == 1

        asr.release.set()
        with contextlib.suppress(asyncio.CancelledError):
            await first
        await second

    asyncio.run(scenario())

    assert asr.calls == 2
    assert asr.max_active == 1


@pytest.mark.parametrize("cancel_transcription", [False, True])
def test_wait_for_asr_idle_waits_for_executor(cancel_transcription):
    asr = _GatedASR()
    pipeline = _pipeline(asr)

    async def scenario():
        transcription = asyncio.create_task(pipeline._transcribe_once(_audio_bytes(), "en"))
        try:
            assert await asyncio.to_thread(asr.entered.wait, 2.0)
            if cancel_transcription:
                transcription.cancel()
            idle = asyncio.create_task(pipeline.wait_for_asr_idle())
            await asyncio.sleep(0.05)
            assert not idle.done()
            assert asr.active == 1
        finally:
            asr.release.set()
            with contextlib.suppress(asyncio.CancelledError):
                await transcription
        await idle
        assert asr.active == 0

    asyncio.run(scenario())


def test_wait_for_asr_idle_timeout_leaves_transcription_running():
    asr = _GatedASR()
    pipeline = _pipeline(asr)

    async def scenario():
        transcription = asyncio.create_task(pipeline._transcribe_once(_audio_bytes(), "en"))
        try:
            assert await asyncio.to_thread(asr.entered.wait, 2.0)
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(pipeline.wait_for_asr_idle(), timeout=0.01)
            assert not transcription.done()
            assert asr.active == 1
            assert pipeline._get_asr_lock().locked()
        finally:
            asr.release.set()
            await transcription
        await pipeline.wait_for_asr_idle()

    asyncio.run(scenario())


def test_process_text_waits_for_speculative_asr_before_llm():
    asr = _GatedASR()
    pipeline = _pipeline(asr)

    async def scenario():
        transcription = asyncio.create_task(pipeline.transcribe_speech(_audio_bytes()))
        text_turn = None
        try:
            assert await asyncio.to_thread(asr.entered.wait, 2.0)
            text_turn = asyncio.create_task(pipeline.process_text("hello"))
            await asyncio.sleep(0.05)
            assert not text_turn.done()
            assert pipeline.llm.calls == []
        finally:
            asr.release.set()
            await transcription
            if text_turn is not None:
                assert await text_turn == "ok"
        assert pipeline.llm.calls

    asyncio.run(scenario())


# ---------------------------------------------------------------------------
# Language-state pollution from rejected speculation
# ---------------------------------------------------------------------------


def test_transcribe_speech_does_not_update_language_state():
    asr = _CountingASR(language="fr")
    pipeline = _pipeline(asr)
    assert pipeline._last_user_language_code == "en"

    result = asyncio.run(pipeline.transcribe_speech(_audio_bytes()))

    assert result.text == "hello"
    assert result.language == "fr"
    assert pipeline._last_user_language_code == "en"


def test_resolve_transcription_updates_language_state_when_consumed():
    asr = _CountingASR(language="en")
    pipeline = _pipeline(asr)

    async def scenario():
        speculative = asyncio.get_running_loop().create_future()
        speculative.set_result(
            ASRResult(
                text="hello",
                language="fr",
                confidence=0.9,
                duration=2.0,
                segments=[
                    {"text": "hello", "start": 0.0, "end": 1.0, "confidence": 0.9}
                ],
            )
        )
        return await pipeline._resolve_transcription(_audio_bytes(), speculative)

    result = asyncio.run(scenario())

    assert result.language == "fr"
    assert pipeline._last_user_language_code == "fr"


def test_resolve_transcription_fallback_still_updates_language_state():
    asr = _CountingASR(language="fr")
    pipeline = _pipeline(asr)

    async def scenario():
        speculative = asyncio.get_running_loop().create_future()
        speculative.set_exception(RuntimeError("speculation exploded"))
        return await pipeline._resolve_transcription(_audio_bytes(), speculative)

    result = asyncio.run(scenario())

    assert result.text == "hello"
    assert pipeline._last_user_language_code == "fr"


# ---------------------------------------------------------------------------
# Desktop Live2DAssistant
# ---------------------------------------------------------------------------


class FakeHandle:
    def __init__(self, callback=None):
        self.callback = callback
        self.cancelled = False

    def cancel(self):
        self.cancelled = True


class FakeLoop:
    def __init__(self):
        self.scheduled = []

    def call_later(self, delay, callback):
        handle = FakeHandle(callback)
        self.scheduled.append((delay, handle))
        return handle

    def call_soon_threadsafe(self, callback, *args):
        callback(*args)

    def is_closed(self):
        return False


class QueuedLoop(FakeLoop):
    """Defer thread-safe callbacks to expose races before timer re-arming."""

    def __init__(self):
        super().__init__()
        self.ready = []

    def call_soon_threadsafe(self, callback, *args):
        self.ready.append((callback, args))

    def drain(self):
        while self.ready:
            callback, args = self.ready.pop(0)
            callback(*args)


def _make_assistant(pipeline):
    assistant = Live2DAssistant.__new__(Live2DAssistant)
    assistant._window = None
    assistant._bridge_server = None
    assistant._active_response_future = None
    assistant._active_turn_id = None
    assistant._latest_audio_turn_id = None
    assistant._playback_deadline = 0.0
    assistant._playback_release_handle = None
    assistant._audio_processing_owned = False
    assistant._drop_current_speech = False
    assistant._speech_active = False
    assistant._pending_speech_audio = bytearray()
    assistant._pending_speech_commit_handle = None
    assistant._pending_speech_lock = threading.Lock()
    assistant._speech_commit_delay_ms = 700
    assistant._pending_speech_commit_delay_ms = 700
    assistant._pending_speech_end_monotonic = None
    assistant._pending_speech_generation = 0
    assistant._pending_speech_detection_thread = None
    assistant._speculative_asr = None
    assistant._speculative_asr_finished_at = None
    assistant._speculative_asr_started_at = None
    assistant._speculative_asr_skip_reason = None
    assistant._turn_detection_config = SmartTurnConfig(enabled=False)
    assistant._smart_turn = None
    assistant._turn_counter = 0
    assistant.config = {
        "mode": "pipeline",
        "audio": {},
        "asr": {"min_audio_ms": 0},
    }
    assistant.pipeline = pipeline
    assistant._omni_pipeline = None
    assistant._gemma_pipeline = None
    assistant._loop = FakeLoop()
    assistant._arm_pending_speech_commit = lambda *args, **kwargs: None
    assistant._dispatch_frontend_event = lambda *args, **kwargs: None
    assistant._start_turn = lambda *args, **kwargs: None
    return assistant


def _install_inline_speculation(monkeypatch):
    """Run speculative coroutines inline and return a completed future."""
    started = []

    def fake_run_coroutine_threadsafe(coro, loop):
        started.append(coro)
        future = ConcurrentFuture()
        try:
            future.set_result(asyncio.run(coro))
        except BaseException as exc:  # pragma: no cover - defensive
            future.set_exception(exc)
        return future

    monkeypatch.setattr(asyncio, "run_coroutine_threadsafe", fake_run_coroutine_threadsafe)
    return started


def _pending_question(monkeypatch, probability=0.3, question_mark_commit=True):
    assistant = _make_assistant(_pipeline(_CountingASR()))
    del assistant._arm_pending_speech_commit
    assistant._loop = QueuedLoop()
    assistant._turn_detection_config = SmartTurnConfig.from_config(
        {"turn_detection": {"question_mark_commit": question_mark_commit}}
    )
    assistant._pending_speech_audio.extend(b"A" * 3200)
    assistant._pending_speech_end_monotonic = 10.0
    assistant._pending_speech_generation = 1
    now = [10.0]
    monkeypatch.setattr(time, "perf_counter", lambda: now[0])
    future = ConcurrentFuture()

    def launch(coro, loop):
        coro.close()
        return future

    monkeypatch.setattr(asyncio, "run_coroutine_threadsafe", launch)
    assistant._maybe_start_speculative_asr(bytes(assistant._pending_speech_audio))
    assistant._arm_pending_speech_commit(2500)
    assistant._loop.drain()
    assistant._apply_turn_verdict(1, (probability >= 0.5, probability), 0.0, 10.0)
    assistant._loop.drain()
    return assistant, future, now


@pytest.mark.parametrize("text", ["  Salut, comment ça va ?  ", "Tu me proposes quoi？\n"])
@pytest.mark.parametrize("elapsed_ms, expected_delay_ms, saved_ms", [(125, 125, 650), (500, 0, 400)])
def test_uncertain_question_rearms_complete_delay_on_loop(
    monkeypatch, caplog, text, elapsed_ms, expected_delay_ms, saved_ms
):
    assistant, future, now = _pending_question(monkeypatch)
    previous = assistant._pending_speech_commit_handle
    now[0] += elapsed_ms / 1000.0

    with caplog.at_level(logging.INFO):
        worker = threading.Thread(target=future.set_result, args=(ASRResult(text=text),))
        worker.start()
        worker.join(timeout=2.0)
        assert not worker.is_alive()
        # The future callback runs off-loop; timers must only change on drain.
        assert assistant._pending_speech_commit_handle is previous
        assert not previous.cancelled
        assistant._loop.drain()

    assert previous.cancelled
    assert assistant._loop.scheduled[-1][0] == pytest.approx(expected_delay_ms / 1000.0)
    messages = [record.getMessage() for record in caplog.records]
    assert messages.count(
        f"turn_detection question_mark_commit tier=uncertain saved_ms={saved_ms}"
    ) == 1


@pytest.mark.parametrize(
    "case", ["no_question", "internal_question", "incomplete", "complete", "audio_changed",
             "disabled", "error", "cancelled", "empty"]
)
def test_question_commit_keeps_window_when_ineligible(monkeypatch, caplog, case):
    probability = {"incomplete": 0.05, "complete": 0.9}.get(case, 0.3)
    assistant, future, now = _pending_question(
        monkeypatch, probability=probability, question_mark_commit=case != "disabled"
    )
    previous = assistant._pending_speech_commit_handle
    scheduled = list(assistant._loop.scheduled)
    now[0] = 10.5
    if case == "audio_changed":
        assistant._pending_speech_audio.extend(b"B" * 1600)

    with caplog.at_level(logging.INFO):
        if case == "error":
            future.set_exception(RuntimeError("speculation exploded"))
        elif case == "cancelled":
            future.cancel()
        else:
            text = {"no_question": "Bonjour.", "internal_question": "Quoi ? Attends."}.get(case, "Ça va ?")
            future.set_result(None if case == "empty" else ASRResult(text=text))
        assistant._loop.drain()

    assert assistant._loop.scheduled == scheduled
    assert assistant._pending_speech_commit_handle is previous
    assert not previous.cancelled
    assert not any("question_mark_commit" in record.getMessage() for record in caplog.records)


@pytest.mark.parametrize("change", ["resume", "commit", "replace"])
def test_question_callback_rechecks_pending_turn_on_loop(monkeypatch, change):
    assistant, future, now = _pending_question(monkeypatch)
    now[0] = 10.5
    future.set_result(ASRResult(text="Ça va ?"))
    scheduled = list(assistant._loop.scheduled)
    if change == "resume":
        assistant._on_speech_start()
    elif change == "commit":
        assistant._commit_pending_speech()
    else:
        replacement = ConcurrentFuture()
        replacement.set_result(ASRResult(text="Ça va ?"))
        assistant._speculative_asr = (bytes(assistant._pending_speech_audio), replacement)

    assistant._loop.drain()

    assert assistant._loop.scheduled == scheduled


@pytest.mark.parametrize("change", ["resume", "commit"])
def test_pending_turn_tier_is_cleared(monkeypatch, change):
    assistant, future, now = _pending_question(monkeypatch)
    assert assistant._pending_speech_commit_tier == "uncertain"
    if change == "resume":
        assistant._on_speech_start()
    else:
        assistant._commit_pending_speech()
    assert assistant._pending_speech_commit_tier is None


@pytest.mark.parametrize("change", ["resume", "commit", "shutdown"])
def test_queued_verdict_arm_cannot_restore_cleared_tier(monkeypatch, change):
    assistant, future, now = _pending_question(monkeypatch)
    assistant._apply_turn_verdict(1, (False, 0.3), 0.0, 10.0)
    if change == "resume":
        assistant._on_speech_start()
    elif change == "commit":
        assistant._commit_pending_speech()
    else:
        assistant.audio_service = None
        asyncio.run(assistant._cancel_active_turn_for_shutdown())

    assistant._loop.drain()

    assert assistant._pending_speech_commit_tier is None


def test_question_completed_before_verdict_can_shorten_uncertain_window(monkeypatch):
    assistant, future, now = _pending_question(monkeypatch)
    assistant._pending_speech_commit_tier = None
    future.set_result(ASRResult(text="Ça va ?"))
    assistant._loop.drain()
    now[0] = 10.5

    assistant._apply_turn_verdict(1, (False, 0.3), 500.0, 10.0)
    assistant._loop.drain()

    assert assistant._loop.scheduled[-1][0] == 0.0


def test_question_and_verdict_queued_together_keep_short_delay(monkeypatch, caplog):
    assistant, future, now = _pending_question(monkeypatch)
    assistant._pending_speech_commit_tier = None
    assistant._arm_pending_speech_commit(2500)
    assistant._loop.drain()
    now[0] = 10.5
    assistant._loop.call_soon_threadsafe(
        assistant._apply_turn_verdict, 1, (False, 0.3), 500.0, 10.0
    )

    with caplog.at_level(logging.INFO):
        future.set_result(ASRResult(text="Ça va ?"))
        assistant._loop.drain()

    assert assistant._loop.scheduled[-1][0] == 0.0
    assert caplog.messages.count(
        "turn_detection question_mark_commit tier=uncertain saved_ms=900"
    ) == 1


def test_question_commit_reports_remaining_time_since_uncertain_arm(monkeypatch, caplog):
    assistant, future, now = _pending_question(monkeypatch)
    now[0] = 10.125
    assistant._apply_turn_verdict(1, (False, 0.3), 125.0, 10.0)
    assistant._loop.drain()
    now[0] = 10.5

    with caplog.at_level(logging.INFO):
        future.set_result(ASRResult(text="Ça va ?"))
        assistant._loop.drain()

    assert "turn_detection question_mark_commit tier=uncertain saved_ms=525" in caplog.messages


def test_shutdown_waits_for_asr_started_after_commit():
    asr = _GatedASR()
    pipeline = _pipeline(asr)
    assistant = _make_assistant(pipeline)
    assistant.audio_service = None

    async def scenario():
        assistant._loop = asyncio.get_running_loop()

        def start_turn(turn_id, runner, source):
            assistant._active_turn_id = turn_id
            assistant._active_response_future = asyncio.run_coroutine_threadsafe(
                runner(), assistant._loop
            )

        assistant._start_turn = start_turn
        assistant._pending_speech_audio.extend(_audio_bytes())
        assistant._commit_pending_speech()
        shutdown = None
        try:
            assert await asyncio.to_thread(asr.entered.wait, 2.0)
            assert assistant._speculative_asr is None
            shutdown = asyncio.create_task(assistant._cancel_active_turn_for_shutdown())
            await asyncio.sleep(0.05)
            assert assistant._active_response_future.cancelled()
            assert not shutdown.done()
            assert asr.active == 1
        finally:
            asr.release.set()
            if shutdown is not None:
                await shutdown
        assert asr.active == 0

    asyncio.run(scenario())


def test_speech_end_starts_speculation_when_assistant_idle(monkeypatch):
    asr = _CountingASR()
    pipeline = _pipeline(asr)
    assistant = _make_assistant(pipeline)
    started = _install_inline_speculation(monkeypatch)

    assistant._on_speech_start()
    assistant._on_speech_detected(b"A" * 3200)
    assistant._on_speech_end()

    assert len(started) == 1
    snapshot, future = assistant._speculative_asr
    assert snapshot == b"A" * 3200
    assert future.done()
    assert asr.calls == 1


def test_speech_end_skips_speculation_while_turn_active(monkeypatch, caplog):
    asr = _CountingASR()
    pipeline = _pipeline(asr)
    assistant = _make_assistant(pipeline)
    started = _install_inline_speculation(monkeypatch)

    assistant._on_speech_start()
    assistant._on_speech_detected(b"A" * 3200)
    # A turn became active before VAD confirmed speech end.
    assistant._active_response_future = ConcurrentFuture()
    assistant._on_speech_end()

    assert started == []
    assert assistant._speculative_asr is None
    assert asr.calls == 0

    with caplog.at_level(logging.INFO):
        assistant._commit_pending_speech()

    assert any(
        "speculative_asr miss reason=turn_active" in record.getMessage()
        for record in caplog.records
    )


def test_speech_end_skips_speculation_when_disabled(monkeypatch, caplog):
    asr = _CountingASR()
    pipeline = _pipeline(asr)
    assistant = _make_assistant(pipeline)
    assistant.config["asr"]["speculative_on_speech_end"] = False
    started = _install_inline_speculation(monkeypatch)

    assistant._on_speech_start()
    assistant._on_speech_detected(b"A" * 3200)
    assistant._on_speech_end()

    assert started == []
    assert assistant._speculative_asr is None
    assert asr.calls == 0

    with caplog.at_level(logging.INFO):
        assistant._commit_pending_speech()

    assert any(
        "speculative_asr miss reason=disabled" in record.getMessage()
        for record in caplog.records
    )


def test_commit_reuses_speculation_and_runs_asr_once(monkeypatch, caplog):
    asr = _CountingASR()
    pipeline = _pipeline(asr)
    assistant = _make_assistant(pipeline)
    _install_inline_speculation(monkeypatch)
    captured = {}
    assistant._start_turn = lambda turn_id, runner, source: captured.update(
        runner=runner, source=source
    )

    assistant._on_speech_start()
    assistant._on_speech_detected(b"A" * 3200)
    assistant._on_speech_end()
    assert asr.calls == 1

    with caplog.at_level(logging.INFO):
        assistant._commit_pending_speech()

    assert assistant._speculative_asr is None
    assert captured["source"] == "speech"
    result = asyncio.run(captured["runner"]())
    assert result == "ok"
    assert asr.calls == 1
    assert pipeline.llm.calls
    assert any(
        "speculative_asr hit" in record.getMessage() for record in caplog.records
    )


def test_commit_ignores_stale_speculation_when_audio_changed(monkeypatch, caplog):
    asr = _CountingASR()
    pipeline = _pipeline(asr)
    assistant = _make_assistant(pipeline)
    _install_inline_speculation(monkeypatch)
    captured = {}
    assistant._start_turn = lambda turn_id, runner, source: captured.update(
        runner=runner, source=source
    )

    assistant._on_speech_start()
    assistant._on_speech_detected(b"A" * 3200)
    assistant._on_speech_end()
    assert asr.calls == 1

    # User resumed before the window expired; the snapshot no longer matches.
    assistant._on_speech_detected(b"B" * 1600)
    with caplog.at_level(logging.INFO):
        assistant._commit_pending_speech()

    assert assistant._speculative_asr is None
    result = asyncio.run(captured["runner"]())
    assert result == "ok"
    assert asr.calls == 2
    assert any(
        "speculative_asr miss reason=audio_changed" in record.getMessage()
        for record in caplog.records
    )


def test_speech_end_skips_speculation_below_min_audio_ms(monkeypatch):
    asr = _CountingASR()
    pipeline = _pipeline(asr)
    assistant = _make_assistant(pipeline)
    assistant.config["asr"]["min_audio_ms"] = 700
    started = _install_inline_speculation(monkeypatch)

    assistant._on_speech_start()
    # 3200 bytes = 100 ms of 16 kHz PCM16, below the 700 ms floor: the commit
    # would drop it, so speculating would only occupy the GPU for nothing.
    assistant._on_speech_detected(b"A" * 3200)
    assistant._on_speech_end()

    assert started == []
    assert assistant._speculative_asr is None
    assert asr.calls == 0


# ---------------------------------------------------------------------------
# saved_ms overlaps
# ---------------------------------------------------------------------------


def test_speculative_saved_ms_is_the_real_overlap():
    # Finishes before the commit: gain is launch -> finish.
    assert _speculative_saved_ms(1.0, 1.2, 1.5) == 200
    # Still running at commit: gain is launch -> commit.
    assert _speculative_saved_ms(1.0, None, 1.4) == 400
    # Closed after the commit instant (clock oddity): clamp to commit.
    assert _speculative_saved_ms(1.0, 2.0, 1.5) == 500
    # No speculation data.
    assert _speculative_saved_ms(None, 1.2, 1.5) == 0


def test_commit_reports_overlap_not_finish_to_commit(monkeypatch, caplog):
    asr = _CountingASR()
    pipeline = _pipeline(asr)
    assistant = _make_assistant(pipeline)
    _install_inline_speculation(monkeypatch)
    assistant._start_turn = lambda *args, **kwargs: None
    completed = ConcurrentFuture()
    completed.set_result(None)
    assistant._pending_speech_audio.extend(b"A" * 3200)
    assistant._speculative_asr = (b"A" * 3200, completed)
    assistant._speculative_asr_started_at = 10.0
    assistant._speculative_asr_finished_at = 10.2
    monkeypatch.setattr(time, "perf_counter", lambda: 10.5)

    with caplog.at_level(logging.INFO):
        assistant._commit_pending_speech()

    assert any(
        "speculative_asr hit saved_ms=200" in record.getMessage()
        for record in caplog.records
    )
    assert assistant._speculative_asr_started_at is None


@pytest.mark.parametrize("outcome", ["success", "error", "cancelled"])
@pytest.mark.parametrize("done_before_commit", [False, True])
def test_commit_logs_speculation_outcome_when_done(monkeypatch, caplog, outcome, done_before_commit):
    assistant = _make_assistant(_pipeline(_CountingASR()))
    speculation = ConcurrentFuture()
    assistant._pending_speech_audio.extend(b"A" * 3200)
    assistant._speculative_asr = (b"A" * 3200, speculation)
    assistant._speculative_asr_started_at = 10.0
    monkeypatch.setattr(time, "perf_counter", lambda: 10.4)

    def complete():
        if outcome == "success":
            speculation.set_result(ASRResult(text="hello", language="en"))
        elif outcome == "error":
            speculation.set_exception(RuntimeError("speculation exploded"))
        else:
            speculation.cancel()

    with caplog.at_level(logging.INFO):
        if done_before_commit:
            complete()
        assistant._commit_pending_speech()
        if not done_before_commit:
            assert not any("speculative_asr hit" in record.getMessage() for record in caplog.records)
            monkeypatch.setattr(time, "perf_counter", lambda: 11.0)
            complete()

    messages = [record.getMessage() for record in caplog.records]
    if outcome == "success":
        assert messages.count("speculative_asr hit saved_ms=400") == 1
        assert "speculative_asr miss reason=error" not in messages
    else:
        assert messages.count(f"speculative_asr miss reason={outcome}") == 1
        other = "error" if outcome == "cancelled" else "cancelled"
        assert f"speculative_asr miss reason={other}" not in messages
        assert not any("speculative_asr hit" in message for message in messages)
