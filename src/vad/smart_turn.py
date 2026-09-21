"""Smart Turn v3.2 semantic end-of-turn detection.

Shared by the desktop assistant (``src/assistant/app.py``) and the WebSocket
server (``src/server/websocket.py``). The detector predicts whether the speaker
finished their turn from the raw waveform only (no transcript), so the commit
delay can be shortened when the utterance is clearly complete and lengthened
when it looks like a mid-sentence pause.

Reference preprocessing (pipecat-ai/smart-turn, inference.py):

* 16 kHz mono float32 input.
* Keep the last 8 seconds (truncate from the beginning), zero-pad at the
  beginning when shorter.
* ``WhisperFeatureExtractor(chunk_length=8)`` with ``do_normalize=True``.
* ONNX input ``"input_features"``, shape ``(1, 80, 800)``.
* Output is already a sigmoid probability; official threshold is 0.5.

Everything degrades cleanly: if onnxruntime/transformers/huggingface_hub are
unavailable, the model cannot be downloaded, or ONNX inference raises, the
detector disables itself (one WARNING) and callers fall back to the existing
fixed ``speech_commit_delay_ms`` delay.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Pinned Hugging Face revision for the int8 CPU model (smart-turn-v3 repo).
SMART_TURN_MODEL_REPO = "pipecat-ai/smart-turn-v3"
SMART_TURN_REVISION = "f766f81d3cfdf7737ac64aad813d91bbfd56bf93"
SMART_TURN_TARGET_SAMPLE_RATE = 16000
SMART_TURN_MAX_SECONDS = 8
# Official decision threshold from inference.py (`prediction = 1 if probability > 0.5`).
SMART_TURN_THRESHOLD = 0.5

# Optional dependencies are imported lazily so the module stays importable in
# environments that only run the fixed-delay path. Tests inject fakes here.
onnxruntime = None
hf_hub_download = None
WhisperFeatureExtractor = None


def _ensure_optional_deps() -> bool:
    global onnxruntime, hf_hub_download, WhisperFeatureExtractor

    if onnxruntime is None:
        try:
            import onnxruntime as _onnxruntime

            onnxruntime = _onnxruntime
        except Exception:
            logger.debug("onnxruntime unavailable for Smart Turn", exc_info=True)
            return False
    if hf_hub_download is None:
        try:
            from huggingface_hub import hf_hub_download as _hf_hub_download

            hf_hub_download = _hf_hub_download
        except Exception:
            logger.debug("huggingface_hub unavailable for Smart Turn", exc_info=True)
            return False
    if WhisperFeatureExtractor is None:
        try:
            from transformers import WhisperFeatureExtractor as _feature_extractor

            WhisperFeatureExtractor = _feature_extractor
        except Exception:
            logger.debug("transformers unavailable for Smart Turn", exc_info=True)
            return False
    return True


@dataclass
class SmartTurnConfig:
    """Configuration for the Smart Turn end-of-turn detector."""

    enabled: bool = True
    model: str = "smart-turn-v3.2-cpu"
    filename: str = "smart-turn-v3.2-cpu.onnx"
    revision: str = SMART_TURN_REVISION
    threshold: float = SMART_TURN_THRESHOLD
    complete_delay_ms: int = 250
    incomplete_delay_ms: int = 2500
    fallback_delay_ms: int = 700
    sample_rate: int = SMART_TURN_TARGET_SAMPLE_RATE
    max_seconds: int = SMART_TURN_MAX_SECONDS
    num_threads: int = 1

    @classmethod
    def from_config(cls, audio_config: Optional[dict], fallback_delay_ms: int = 700) -> "SmartTurnConfig":
        """Build the config from the ``audio`` section of config.yaml."""
        section = audio_config.get("turn_detection", {}) if isinstance(audio_config, dict) else {}
        if not isinstance(section, dict):
            section = {}

        def _int(key: str, default: int) -> int:
            try:
                return max(0, int(section.get(key, default)))
            except (TypeError, ValueError):
                return default

        def _float(key: str, default: float) -> float:
            try:
                return float(section.get(key, default))
            except (TypeError, ValueError):
                return default

        model = str(section.get("model", cls.model) or cls.model)
        try:
            fallback = max(0, int(fallback_delay_ms))
        except (TypeError, ValueError):
            fallback = 700

        return cls(
            enabled=bool(section.get("enabled", True)),
            model=model,
            filename=f"{model}.onnx",
            revision=str(section.get("revision", SMART_TURN_REVISION) or SMART_TURN_REVISION),
            threshold=_float("threshold", SMART_TURN_THRESHOLD),
            complete_delay_ms=_int("complete_delay_ms", 250),
            incomplete_delay_ms=_int("incomplete_delay_ms", 2500),
            fallback_delay_ms=fallback,
            sample_rate=_int("sample_rate", SMART_TURN_TARGET_SAMPLE_RATE),
            max_seconds=_int("max_seconds", SMART_TURN_MAX_SECONDS),
            num_threads=max(1, _int("num_threads", 1)),
        )


def resolve_turn_commit_delay_ms(
    verdict: Optional[bool],
    *,
    enabled: bool = True,
    complete_delay_ms: int = 250,
    incomplete_delay_ms: int = 2500,
    fallback_delay_ms: int = 700,
) -> int:
    """Pure delay policy shared by every turn path.

    ``verdict`` is ``True``/``False`` when the detector produced a decision and
    ``None`` when it is disabled or unavailable. Unavailable always falls back
    to the legacy fixed delay.
    """
    if not enabled or verdict is None:
        return max(0, int(fallback_delay_ms))
    delay = complete_delay_ms if verdict else incomplete_delay_ms
    return max(0, int(delay))


def resolve_commit_delay_for_turn(
    config: "SmartTurnConfig",
    verdict: Optional[bool],
    *,
    fallback_delay_ms: Optional[int] = None,
    elapsed_ms: float = 0.0,
) -> int:
    """Single source of truth for the three turn cases.

    * disabled / unavailable (``verdict is None``) -> fallback delay
    * complete -> ``complete_delay_ms`` minus time already spent
    * incomplete -> ``incomplete_delay_ms``

    The fallback is clamped to 0 when the elapsed inference time already
    exceeds the configured fixed delay.
    """
    complete_delay = max(0, int(config.complete_delay_ms - max(0.0, elapsed_ms)))
    fallback = config.fallback_delay_ms if fallback_delay_ms is None else fallback_delay_ms
    return resolve_turn_commit_delay_ms(
        verdict,
        enabled=config.enabled,
        complete_delay_ms=complete_delay,
        incomplete_delay_ms=config.incomplete_delay_ms,
        fallback_delay_ms=fallback,
    )


def prepare_audio(
    audio,
    sample_rate: int,
    *,
    max_seconds: int = SMART_TURN_MAX_SECONDS,
    target_sample_rate: int = SMART_TURN_TARGET_SAMPLE_RATE,
) -> Optional[np.ndarray]:
    """Normalize raw audio to the model's expected 16 kHz mono float32 buffer.

    Accepts PCM16 ``bytes``/``bytearray``/``memoryview`` or a numpy array.
    Returns ``None`` (so callers fall back) when the sample rate is not the
    model's 16 kHz, rather than silently feeding wrong-rate audio.
    """
    if audio is None:
        return None
    if int(sample_rate) != int(target_sample_rate):
        logger.debug(
            "Smart Turn refuses non-%sHz audio (got %sHz)", target_sample_rate, sample_rate
        )
        return None

    if isinstance(audio, (bytes, bytearray, memoryview)):
        samples = np.frombuffer(bytes(audio), dtype=np.int16).astype(np.float32) / 32768.0
    else:
        array = np.asarray(audio)
        if array.size == 0:
            return None
        if np.issubdtype(array.dtype, np.integer):
            samples = array.astype(np.float32) / 32768.0
        else:
            samples = array.astype(np.float32, copy=False)

    max_samples = int(max_seconds * target_sample_rate)
    if max_samples <= 0:
        return None
    if samples.shape[0] > max_samples:
        samples = samples[-max_samples:]
    elif samples.shape[0] < max_samples:
        samples = np.pad(samples, (max_samples - samples.shape[0], 0), mode="constant")
    return np.ascontiguousarray(samples, dtype=np.float32)


class SmartTurnDetector:
    """Lazy-loading Smart Turn v3.2 ONNX detector (CPU)."""

    def __init__(self, config: Optional[SmartTurnConfig] = None):
        self.config = config or SmartTurnConfig()
        self._session = None
        self._feature_extractor = None
        self._load_failed = False
        self._warned = False
        self._lock = threading.Lock()

    @property
    def available(self) -> bool:
        """False once the detector has permanently disabled itself."""
        return not self._load_failed

    def warmup(self) -> bool:
        """Force the lazy load so the first turn does not pay for it."""
        return self._ensure_loaded()

    def _disable(self, exc: Exception) -> None:
        self._load_failed = True
        self._session = None
        self._feature_extractor = None
        if not self._warned:
            logger.warning(
                "Smart Turn detector disabled, falling back to fixed commit delay: %s", exc
            )
            self._warned = True

    def _ensure_loaded(self) -> bool:
        if self._session is not None:
            return True
        if self._load_failed:
            return False
        with self._lock:
            if self._session is not None:
                return True
            if self._load_failed:
                return False
            try:
                if not _ensure_optional_deps():
                    raise RuntimeError(
                        "onnxruntime/transformers/huggingface_hub unavailable"
                    )
                model_path = hf_hub_download(
                    repo_id=SMART_TURN_MODEL_REPO,
                    filename=self.config.filename,
                    revision=self.config.revision,
                )
                session_options = onnxruntime.SessionOptions()
                session_options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
                session_options.inter_op_num_threads = 1
                session_options.intra_op_num_threads = self.config.num_threads
                session_options.graph_optimization_level = (
                    onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
                )
                self._session = onnxruntime.InferenceSession(
                    model_path,
                    sess_options=session_options,
                    providers=["CPUExecutionProvider"],
                )
                self._feature_extractor = WhisperFeatureExtractor(
                    chunk_length=self.config.max_seconds
                )
                logger.info("Smart Turn detector ready (model=%s)", self.config.model)
                return True
            except Exception as exc:
                self._disable(exc)
                return False

    def predict(self, audio, sample_rate: Optional[int] = None):
        """Return ``(is_complete, probability)`` or ``None`` when unavailable."""
        if not self._ensure_loaded():
            return None

        sr = int(sample_rate or self.config.sample_rate)
        try:
            prepared = prepare_audio(
                audio,
                sr,
                max_seconds=self.config.max_seconds,
                target_sample_rate=self.config.sample_rate,
            )
            if prepared is None:
                return None

            max_samples = self.config.max_seconds * self.config.sample_rate
            started = time.perf_counter()
            inputs = self._feature_extractor(
                prepared,
                sampling_rate=self.config.sample_rate,
                return_tensors="np",
                padding="max_length",
                max_length=max_samples,
                truncation=True,
                do_normalize=True,
            )
            input_features = inputs.input_features.squeeze(0).astype(np.float32)
            input_features = np.expand_dims(input_features, axis=0)
            outputs = self._session.run(None, {"input_features": input_features})
            probability = float(outputs[0][0].item())
            infer_ms = (time.perf_counter() - started) * 1000.0
            logger.debug(
                "Smart Turn inference %.1fms probability=%.4f", infer_ms, probability
            )
            return (probability > self.config.threshold, probability)
        except Exception as exc:
            self._disable(exc)
            return None