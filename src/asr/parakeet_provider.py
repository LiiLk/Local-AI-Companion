"""
Parakeet ASR Provider using onnx-asr (NVIDIA Parakeet-TDT-0.6B-v3, ONNX).

Runs the NeMo Parakeet model through ONNX Runtime *without* installing NeMo.
This is intentional for Windows-first installs: full NeMo is heavy and fragile;
`onnx-asr` is a thin pure-Python stack (numpy + onnxruntime + optional hub).

Status (LIL-48 / LIL-45):
- Opt-in provider only. Public default remains Whisper (`asr.provider: whisper`).
- Validated on the Windows and WSL desktop pipeline paths.

Languages: 25 European (incl. fr / en / es). Not for zh / ja / ar / ko / hi —
use Whisper for full multilingual coverage.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any, AsyncGenerator, List, Optional, Union

import numpy as np

from .base import ASRResult, ASRSegment, BaseASR

logger = logging.getLogger(__name__)

# The 25 European languages Parakeet-TDT-0.6b-v3 is trained on (ISO 639-1).
SUPPORTED_LANGUAGES = [
    "bg",
    "hr",
    "cs",
    "da",
    "nl",
    "en",
    "et",
    "fi",
    "fr",
    "de",
    "el",
    "hu",
    "it",
    "lv",
    "lt",
    "mt",
    "pl",
    "pt",
    "ro",
    "sk",
    "sl",
    "es",
    "sv",
    "ru",
    "uk",
]

TARGET_SAMPLE_RATE = 16000
DEFAULT_MODEL_NAME = "nemo-parakeet-tdt-0.6b-v3"
INSTALL_HINT = (
    "onnx-asr is not installed. Install the optional Parakeet stack with:\n"
    "  pip install -r requirements-optional-parakeet.txt\n"
    "Then set `asr.provider: parakeet` in config/config.local.yaml."
)


class ParakeetASRProvider(BaseASR):
    """ASR provider using onnx-asr to run Parakeet-TDT-0.6b-v3 (ONNX).

    Args:
        model_name: onnx-asr model id (default nemo-parakeet-tdt-0.6b-v3).
        quantization: ONNX quantization ("int8" recommended on CPU; "fp16" for GPU).
        providers: onnxruntime execution providers. Defaults to CPU only so
            runtime does not probe missing CUDA/TensorRT DLLs.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        quantization: Optional[str] = "int8",
        providers: Optional[List[str]] = None,
    ):
        self.model_name = model_name
        self.quantization = quantization
        self.sample_rate = TARGET_SAMPLE_RATE
        # Default to CPU: int8 is a CPU path and avoids onnxruntime probing
        # CUDA/TensorRT when GPU DLLs are absent. GPU users can pass
        # providers=["CUDAExecutionProvider"] (or DirectML on Windows).
        self.providers = list(providers) if providers else ["CPUExecutionProvider"]
        self._model: Any | None = None

    @staticmethod
    def is_available() -> bool:
        """Return True when the optional onnx-asr package is importable."""
        try:
            import onnx_asr  # noqa: F401
        except ImportError:
            return False
        return True

    def _get_model(self):
        """Lazy load the ONNX model (downloads on first use via huggingface hub)."""
        if self._model is None:
            try:
                import onnx_asr
            except ImportError as exc:
                raise ImportError(INSTALL_HINT) from exc

            logger.info(
                "Loading Parakeet ASR (%s, quantization=%s, providers=%s)...",
                self.model_name,
                self.quantization,
                self.providers,
            )
            self._model = onnx_asr.load_model(
                self.model_name,
                quantization=self.quantization,
                providers=self.providers,
            )
            logger.info("Parakeet ASR loaded (%s)", self.model_name)
        return self._model

    def preload(self):
        """Eagerly load the model (used by the pipeline preload path)."""
        self._get_model()
        return self

    def cleanup(self) -> None:
        """Drop the model reference so runtime shutdown can reclaim memory."""
        self._model = None

    @staticmethod
    def _to_mono_float32(waveform: np.ndarray) -> np.ndarray:
        arr = np.asarray(waveform, dtype=np.float32)
        if arr.ndim > 1:
            # Average channels → mono
            arr = arr.mean(axis=-1 if arr.shape[-1] <= 8 else 0)
        return np.ascontiguousarray(arr.reshape(-1), dtype=np.float32)

    @staticmethod
    def _coerce_text(raw: Any) -> str:
        """Normalize onnx-asr recognize() output to a single stripped string."""
        if raw is None:
            return ""
        if isinstance(raw, str):
            return raw.strip()
        if isinstance(raw, (list, tuple)):
            parts = [str(x).strip() for x in raw if x is not None and str(x).strip()]
            return " ".join(parts).strip()
        # Some versions may return an object with .text
        text_attr = getattr(raw, "text", None)
        if isinstance(text_attr, str):
            return text_attr.strip()
        return str(raw).strip()

    def transcribe(
        self,
        audio_input: Union[str, Path, np.ndarray],
        language: Optional[str] = None,
        initial_prompt: Optional[str] = None,  # interface parity; unused by Parakeet
    ) -> ASRResult:
        """Transcribe an audio file or float32 numpy array to text."""
        # onnx-asr performs its own language detection and does not accept a
        # language hint. Keep these arguments only for BaseASR compatibility.
        del language, initial_prompt
        model = self._get_model()

        if isinstance(audio_input, (str, Path)):
            audio_path = Path(audio_input)
            if not audio_path.exists():
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
            # Prefer path-based recognize so onnx-asr can resample as needed.
            logger.info("Transcribing %s", audio_path.name)
            raw = model.recognize(str(audio_path))
            # Duration best-effort via soundfile when available.
            duration = None
            try:
                import soundfile as sf

                info = sf.info(str(audio_path))
                duration = float(info.duration)
            except Exception:  # noqa: BLE001
                duration = None
            text = self._coerce_text(raw)
            return ASRResult(
                text=text,
                language=None,
                confidence=None,
                duration=duration,
                segments=[],
            )

        waveform = self._to_mono_float32(audio_input)
        sample_rate = self.sample_rate
        if waveform.size == 0:
            return ASRResult(
                text="",
                language=None,
                duration=0.0,
            )

        duration = float(len(waveform) / sample_rate) if sample_rate else None
        logger.info(
            "Transcribing audio buffer (%s samples, %.2fs)",
            waveform.shape[0],
            duration or 0.0,
        )

        raw = model.recognize(waveform, sample_rate=sample_rate)
        text = self._coerce_text(raw)
        return ASRResult(
            text=text,
            language=None,
            confidence=None,
            duration=duration,
            segments=[],
        )

    async def transcribe_stream(
        self,
        audio_path: str | Path,
        language: Optional[str] = None,
    ) -> AsyncGenerator[ASRSegment, None]:
        """Minimal streaming shim: full utterance, one segment.

        Parakeet is used offline after VAD segmentation, so chunked streaming
        is not required for the conversational pipeline.
        """
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None, lambda: self.transcribe(audio_path, language=language)
        )
        if result.text:
            yield ASRSegment(
                text=result.text,
                start=0.0,
                end=result.duration or 0.0,
                confidence=None,
            )

    def get_supported_languages(self) -> List[str]:
        return SUPPORTED_LANGUAGES.copy()

    def get_model_info(self) -> dict:
        return {
            "provider": "parakeet",
            "model_name": self.model_name,
            "quantization": self.quantization,
            "sample_rate": self.sample_rate,
            "providers": list(self.providers),
            "languages": "25 European (auto-detect)",
            "loaded": self._model is not None,
            "optional_dependency": "onnx-asr",
            "status": "validated opt-in (LIL-48)",
        }
