"""
TTS Task Manager -- Decouples LLM streaming from TTS synthesis.

Accepts sentences via submit(), synthesizes them sequentially
(GPU-safe), and delivers audio payloads via callback in order.
The queue is bounded so very long LLM streams apply backpressure instead of
growing memory without limit while TTS is working.
"""

import asyncio
import base64
import io
import logging
import tempfile
import time
import wave
from pathlib import Path
from typing import Any, Callable, Coroutine, Optional

import numpy as np

logger = logging.getLogger(__name__)


def _read_wav_bytes(wav_bytes: bytes) -> tuple[bytes, int]:
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        return wf.readframes(wf.getnframes()), wf.getframerate()


def _read_wav_file(path: Path) -> tuple[bytes, int]:
    with wave.open(str(path), "rb") as wf:
        return wf.readframes(wf.getnframes()), wf.getframerate()


def _analyze_volumes(pcm: bytes, sr: int, chunk_ms: int = 50) -> list[float]:
    samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
    chunk_sz = int(sr * chunk_ms / 1000)
    vols: list[float] = []
    for i in range(0, len(samples), chunk_sz):
        c = samples[i : i + chunk_sz]
        if len(c) == 0:
            continue
        rms = float(np.sqrt(np.mean(c**2)))
        n = min(1.0, rms / 8000)
        vols.append(0.0 if n < 0.05 else n)
    return vols


class TTSTaskManager:
    """
    Async TTS pipeline that runs independently of the LLM stream.
    """

    def __init__(
        self,
        tts: Any,
        on_audio_ready: Callable[..., Coroutine],
        rvc: Optional[Any] = None,
        sample_rate: int = 24000,
        lip_sync_chunk_ms: int = 50,
        on_expression: Optional[Callable[..., Coroutine]] = None,
        emotion_detector: Optional[Any] = None,
        max_queue_size: int = 8,
    ):
        self._tts = tts
        self._on_audio_ready = on_audio_ready
        self._rvc = rvc
        self._sample_rate = sample_rate
        self._lip_sync_chunk_ms = lip_sync_chunk_ms
        self._on_expression = on_expression
        self._emotion_detector = emotion_detector
        queue_maxsize = max(0, int(max_queue_size))
        self._queue: asyncio.Queue[tuple[str, Optional[str], float] | None] = asyncio.Queue(
            maxsize=queue_maxsize
        )
        self._worker_task: Optional[asyncio.Task] = None

    async def start(self):
        self._worker_task = asyncio.create_task(self._worker())

    async def submit(self, text: str, expression: Optional[str] = None):
        await self._queue.put((text, expression, time.perf_counter()))

    async def finish(self):
        await self._queue.put(None)
        if self._worker_task:
            await self._worker_task

    async def cancel(self):
        self._abort_inflight_tts()
        if self._worker_task and not self._worker_task.done():
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break

    def _abort_inflight_tts(self) -> None:
        cancel_fn = getattr(self._tts, "cancel_inflight", None)
        if not callable(cancel_fn):
            return
        try:
            cancel_fn()
        except Exception as exc:
            logger.debug("TTS cancel_inflight failed: %s", exc)

    async def _worker(self):
        while True:
            item = await self._queue.get()
            if item is None:
                break

            text, expression, queued_at = item
            if not text or not text.strip():
                continue

            try:
                await self._synthesize_one(text, expression, queued_at)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error("TTS worker error for %r: %s", text[:40], e)

    async def _synthesize_one(self, text: str, expression: Optional[str], queued_at: float):
        if self._emotion_detector:
            detected = self._emotion_detector.detect(text)
            if detected and self._on_expression:
                await self._on_expression(detected)
            text = self._emotion_detector.strip_markers(text)
            if not text.strip():
                return

        full_wav: Optional[bytes] = None
        pcm: Optional[bytes] = None
        sr = self._sample_rate
        metadata: dict = {}

        synth_started = time.perf_counter()
        queue_wait_ms = (synth_started - queued_at) * 1000
        try:
            result = await self._tts.synthesize(text)
        except asyncio.CancelledError:
            self._abort_inflight_tts()
            raise
        synth_elapsed_ms = (time.perf_counter() - synth_started) * 1000

        if result.metadata:
            metadata.update(result.metadata)

        file_read_ms = float(metadata.get("file_read_ms", 0.0) or 0.0)
        file_write_ms = float(metadata.get("file_write_ms", 0.0) or 0.0)
        provider_roundtrip_ms = float(metadata.get("provider_roundtrip_ms", synth_elapsed_ms) or synth_elapsed_ms)
        provider_wait_ms = max(0.0, provider_roundtrip_ms - float(metadata.get("synth_ms", synth_elapsed_ms) or synth_elapsed_ms))

        if result.audio_data:
            full_wav = result.audio_data
            pcm, sr = _read_wav_bytes(full_wav)
        elif result.audio_path:
            read_started = time.perf_counter()
            full_wav = Path(result.audio_path).read_bytes()
            pcm, sr = _read_wav_file(Path(result.audio_path))
            file_read_ms += (time.perf_counter() - read_started) * 1000

        if not full_wav or pcm is None:
            return

        rvc_ms = 0.0
        if self._rvc:
            rvc_started = time.perf_counter()
            full_wav, pcm, sr = await self._apply_rvc(full_wav)
            rvc_ms = (time.perf_counter() - rvc_started) * 1000

        volumes = _analyze_volumes(pcm, sr, self._lip_sync_chunk_ms)
        duration_ms = int(len(pcm) / (sr * 2) * 1000)
        total_tts_ms = (time.perf_counter() - synth_started) * 1000
        attn_used = metadata.get("attn_implementation") or metadata.get("attn_implementation_actual")

        logger.info(
            "TTS sentence metrics: text=%r queue_wait_ms=%.1f synth_ms=%.1f provider_wait_ms=%.1f total_ms=%.1f file_write_ms=%.1f file_read_ms=%.1f rvc_ms=%.1f attn=%s",
            text[:80],
            queue_wait_ms,
            float(metadata.get("synth_ms", synth_elapsed_ms) or synth_elapsed_ms),
            provider_wait_ms,
            total_tts_ms,
            file_write_ms,
            file_read_ms,
            rvc_ms,
            attn_used or "unknown",
        )

        payload = {
            "audio_base64": base64.b64encode(full_wav).decode("utf-8"),
            "wav_bytes": full_wav,
            "pcm_bytes": pcm,
            "volumes": volumes,
            "duration_ms": duration_ms,
            "sample_rate": sr,
            "text": text,
            "expression": expression,
            "tts_metrics": {
                "synth_ms": float(metadata.get("synth_ms", synth_elapsed_ms) or synth_elapsed_ms),
                "provider_roundtrip_ms": provider_roundtrip_ms,
                "provider_wait_ms": provider_wait_ms,
                "total_ms": total_tts_ms,
                "file_write_ms": file_write_ms,
                "file_read_ms": file_read_ms,
                "rvc_ms": rvc_ms,
                "attn_implementation": attn_used,
            },
        }

        await self._on_audio_ready(payload)

    async def _apply_rvc(self, wav_bytes: bytes) -> tuple[bytes, bytes, int]:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as src_file:
            src = Path(src_file.name)
        with tempfile.NamedTemporaryFile(suffix=".rvc.wav", delete=False) as dst_file:
            dst = Path(dst_file.name)
        try:
            src.write_bytes(wav_bytes)
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._rvc.convert_file, src, dst)
            out_wav = dst.read_bytes()
            out_pcm, out_sr = _read_wav_file(dst)
            return out_wav, out_pcm, out_sr
        except Exception as e:
            logger.warning("RVC failed, using original: %s", e)
            pcm, sr = _read_wav_bytes(wav_bytes)
            return wav_bytes, pcm, sr
        finally:
            src.unlink(missing_ok=True)
            dst.unlink(missing_ok=True)
