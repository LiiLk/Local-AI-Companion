"""Tests for TTSTaskManager — decoupled LLM/TTS pipeline."""

import asyncio
import io
import wave
from pathlib import Path

import pytest

from src.tts.base import TTSResult
from src.tts.tts_task_manager import TTSTaskManager


class FakeTTS:
    """Fake TTS that writes minimal WAV files."""

    def __init__(self, delay: float = 0.01):
        self.delay = delay
        self.call_count = 0
        self.cancelled = False

    async def synthesize(self, text, output_path=None):
        await asyncio.sleep(self.delay)
        self.call_count += 1
        samples = b"\x00\x00" * 2400  # 100ms silence at 24kHz
        if output_path:
            with wave.open(str(output_path), "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(24000)
                wf.writeframes(samples)
            return TTSResult(audio_path=output_path)

        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(24000)
            wf.writeframes(samples)
        return TTSResult(audio_data=buffer.getvalue(), metadata={"file_write_ms": 0.0, "file_read_ms": 0.0})

    def cancel_inflight(self):
        self.cancelled = True


@pytest.mark.asyncio
async def test_delivers_in_order():
    """Sentences submitted in order are delivered in order."""
    tts = FakeTTS()
    delivered = []

    async def on_audio(payload):
        delivered.append(payload["text"])

    mgr = TTSTaskManager(tts=tts, on_audio_ready=on_audio)
    await mgr.start()
    await mgr.submit("Hello.")
    await mgr.submit("How are you?")
    await mgr.submit("Goodbye.")
    await mgr.finish()

    assert delivered == ["Hello.", "How are you?", "Goodbye."]
    assert tts.call_count == 3


@pytest.mark.asyncio
async def test_skips_empty_text():
    tts = FakeTTS()
    delivered = []

    async def on_audio(payload):
        delivered.append(payload["text"])

    mgr = TTSTaskManager(tts=tts, on_audio_ready=on_audio)
    await mgr.start()
    await mgr.submit("")
    await mgr.submit("  ")
    await mgr.submit("Real sentence.")
    await mgr.finish()

    assert delivered == ["Real sentence."]


@pytest.mark.asyncio
async def test_handles_tts_error_gracefully():
    """TTS failure on one sentence doesn't crash the pipeline."""

    class FailingTTS(FakeTTS):
        async def synthesize(self, text, output_path=None):
            if "fail" in text:
                raise RuntimeError("TTS boom")
            return await super().synthesize(text, output_path)

    tts = FailingTTS()
    delivered = []

    async def on_audio(payload):
        delivered.append(payload["text"])

    mgr = TTSTaskManager(tts=tts, on_audio_ready=on_audio)
    await mgr.start()
    await mgr.submit("Before.")
    await mgr.submit("fail here")
    await mgr.submit("After.")
    await mgr.finish()

    assert "Before." in delivered
    assert "After." in delivered
    assert "fail here" not in delivered


@pytest.mark.asyncio
async def test_cancel_stops_worker():
    """Cancelling the manager stops processing."""
    tts = FakeTTS(delay=0.1)
    delivered = []

    async def on_audio(payload):
        delivered.append(payload["text"])

    mgr = TTSTaskManager(tts=tts, on_audio_ready=on_audio)
    await mgr.start()
    await mgr.submit("First.")
    # Give worker time to pick up first item
    await asyncio.sleep(0.05)
    await mgr.cancel()

    # Should not crash, worker stopped cleanly
    assert isinstance(delivered, list)
    assert tts.cancelled is True


@pytest.mark.asyncio
async def test_payload_has_expected_keys():
    """Audio payload contains all required fields."""
    tts = FakeTTS()
    payloads = []

    async def on_audio(payload):
        payloads.append(payload)

    mgr = TTSTaskManager(tts=tts, on_audio_ready=on_audio)
    await mgr.start()
    await mgr.submit("Test sentence.")
    await mgr.finish()

    assert len(payloads) == 1
    p = payloads[0]
    assert "audio_base64" in p
    assert "wav_bytes" in p
    assert "volumes" in p
    assert "duration_ms" in p
    assert "sample_rate" in p
    assert "text" in p
    assert p["text"] == "Test sentence."
    assert p["sample_rate"] == 24000
    assert p["duration_ms"] > 0


@pytest.mark.asyncio
async def test_llm_stream_not_blocked():
    """
    Verify that submitting to the queue returns immediately
    while TTS processes in the background.
    """
    tts = FakeTTS(delay=0.2)  # Slow TTS
    delivered = []

    async def on_audio(payload):
        delivered.append(payload["text"])

    mgr = TTSTaskManager(tts=tts, on_audio_ready=on_audio)
    await mgr.start()

    # Simulate rapid LLM output — these should queue instantly
    import time

    t0 = time.monotonic()
    await mgr.submit("Sentence one.")
    await mgr.submit("Sentence two.")
    await mgr.submit("Sentence three.")
    queue_time = time.monotonic() - t0

    # Queuing should be near-instant (< 50ms), not 3 * 200ms
    assert queue_time < 0.1, f"Queuing took {queue_time:.3f}s — LLM stream is blocked!"

    await mgr.finish()
    assert len(delivered) == 3


@pytest.mark.asyncio
async def test_submit_backpressures_when_queue_is_full():
    tts = FakeTTS()

    async def on_audio(payload):
        pass

    mgr = TTSTaskManager(tts=tts, on_audio_ready=on_audio, max_queue_size=1)
    await mgr.submit("Queued sentence.")

    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(mgr.submit("Blocked sentence."), timeout=0.02)

    await mgr.cancel()
