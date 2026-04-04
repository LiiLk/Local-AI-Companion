"""
Chatterbox Multilingual ONNX Q4 TTS Provider.

Extends BaseTTS with voice cloning and emotion tag support.
Uses ONNX Runtime for inference (separate CUDA allocator from PyTorch).

Features:
- Voice cloning from 5-10s reference audio
- Emotion tags: [laugh], [chuckle], [cough], [sigh]
- 23 languages including French
- Exaggeration slider (0.0-1.0)
- ~2 GB VRAM (ONNX Q4)
"""

import asyncio
import io
import logging
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import AsyncGenerator, Optional

import numpy as np
import soundfile as sf

from .base import BaseTTS, TTSResult, Voice

logger = logging.getLogger(__name__)


# Available emotion tags that Chatterbox interprets natively
CHATTERBOX_EMOTION_TAGS = {"laugh", "chuckle", "cough", "sigh"}

CHATTERBOX_VOICES = [
    Voice(id="default", name="Default", language="multi", gender="Female"),
]


class ChatterboxTTSProvider(BaseTTS):
    """
    TTS provider using Chatterbox Multilingual ONNX Q4.

    Extends BaseTTS and implements all abstract methods.
    Model is lazily loaded on first use.

    Args:
        model_id: HuggingFace model ID for ONNX Q4 variant.
        ref_audio_path: Path to reference WAV for voice cloning.
        exaggeration: Emotion exaggeration (0.0-1.0).
        cfg_weight: Classifier-free guidance weight.
        language: Target language code (e.g., "fr", "en").
    """

    SAMPLE_RATE = 24000

    def __init__(
        self,
        model_id: str = "onnx-community/chatterbox-multilingual-ONNX",
        ref_audio_path: str | Path | None = None,
        exaggeration: float = 0.5,
        cfg_weight: float = 0.5,
        language: str = "fr",
    ):
        self.model_id = model_id
        self.ref_audio_path = Path(ref_audio_path) if ref_audio_path else None
        self.exaggeration = exaggeration
        self.cfg_weight = cfg_weight
        self.language = language
        self._speed = 1.0

        self._model = None
        self._ref_audio_data = None
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tts")

    def _load_model(self):
        """Load Chatterbox ONNX Q4 model (lazy)."""
        if self._model is not None:
            return

        logger.info(f"Loading Chatterbox from {self.model_id}...")

        # Import and load the ONNX model
        from chatterbox_onnx import ChatterboxONNX

        self._model = ChatterboxONNX.from_pretrained(self.model_id)

        # Pre-load reference audio if configured
        if self.ref_audio_path and self.ref_audio_path.exists():
            self._ref_audio_data = self._load_reference(self.ref_audio_path)
            logger.info(f"Voice reference loaded: {self.ref_audio_path}")

        logger.info("Chatterbox loaded successfully")

    def _load_reference(self, path: Path) -> np.ndarray:
        """Load and preprocess reference audio for voice cloning."""
        audio, sr = sf.read(str(path))
        # Resample to 24kHz if needed
        if sr != self.SAMPLE_RATE:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.SAMPLE_RATE)
        # Convert to mono if stereo
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        return audio.astype(np.float32)

    def _synthesize_sync(self, text: str) -> tuple[np.ndarray, int]:
        """Synchronous synthesis. Returns (audio_array, sample_rate)."""
        self._load_model()

        kwargs = {
            "text": text,
            "exaggeration": self.exaggeration,
            "cfg_weight": self.cfg_weight,
        }

        if self._ref_audio_data is not None:
            kwargs["audio_prompt"] = self._ref_audio_data

        audio = self._model.generate(**kwargs)

        # audio is a numpy array at 24kHz
        if isinstance(audio, (list, tuple)):
            audio = audio[0]
        if hasattr(audio, "numpy"):
            audio = audio.numpy()

        return audio.astype(np.float32), self.SAMPLE_RATE

    async def synthesize(
        self,
        text: str,
        output_path: Path | None = None,
    ) -> TTSResult:
        """Synthesize text to audio."""
        loop = asyncio.get_running_loop()
        audio, sr = await loop.run_in_executor(
            self._executor, self._synthesize_sync, text
        )

        # Convert to 16-bit PCM WAV
        audio_int16 = (audio * 32767).astype(np.int16)

        if output_path:
            sf.write(str(output_path), audio_int16, sr, subtype="PCM_16")
            duration = len(audio) / sr
            return TTSResult(
                audio_path=output_path,
                duration=duration,
            )
        else:
            buf = io.BytesIO()
            sf.write(buf, audio_int16, sr, format="WAV", subtype="PCM_16")
            audio_bytes = buf.getvalue()
            duration = len(audio) / sr
            return TTSResult(
                audio_data=audio_bytes,
                duration=duration,
            )

    async def synthesize_stream(self, text: str) -> AsyncGenerator[bytes, None]:
        """
        Streaming synthesis — yields audio chunks.

        Note: Chatterbox ONNX may not support true streaming.
        Falls back to generating full audio then chunking.
        """
        result = await self.synthesize(text)
        data = result.audio_data
        if data is None and result.audio_path:
            data = result.audio_path.read_bytes()

        # Yield in ~100ms chunks
        chunk_size = self.SAMPLE_RATE * 2 * 100 // 1000  # 16-bit = 2 bytes/sample
        for i in range(0, len(data), chunk_size):
            yield data[i : i + chunk_size]

    async def list_voices(self, language: str | None = None) -> list[Voice]:
        """List available voices."""
        return CHATTERBOX_VOICES

    def set_voice(self, voice_id: str) -> None:
        """Set voice by reference audio path."""
        path = Path(voice_id)
        if path.exists():
            self.ref_audio_path = path
            self._ref_audio_data = self._load_reference(path)
            logger.info(f"Voice reference set: {path}")

    def set_rate(self, rate: str) -> None:
        """Adjust speech rate. Accepts '+20%' style strings."""
        # Parse percentage string to float multiplier
        try:
            pct = int(rate.replace("%", "").replace("+", ""))
            self._speed = 1.0 + pct / 100.0
        except (ValueError, AttributeError):
            pass

    def set_pitch(self, pitch: str) -> None:
        """No-op — Chatterbox handles pitch internally."""
        pass

    def set_reference_audio(self, ref_audio_path: str | Path) -> None:
        """Set reference audio for voice cloning."""
        self.set_voice(str(ref_audio_path))

    def cleanup(self):
        """Unload model and free resources."""
        if self._model is not None:
            del self._model
            self._model = None
        self._ref_audio_data = None
        logger.info("ChatterboxTTSProvider cleaned up")
