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
import re
import threading
import time
import wave
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
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
# Below this probability the turn is treated as a genuine mid-sentence pause
# instead of merely uncertain. Splitting the old binary policy into three tiers
# keeps a hesitant "hmm..." from waiting as long as a full stop.
SMART_TURN_UNCERTAIN_THRESHOLD = 0.15
# Maximum number of debug WAVs kept when debug_save_dir is enabled.
SMART_TURN_DEBUG_MAX_FILES = 200
# Only files the detector itself writes (see ``save_debug_wav``) may be pruned.
_DEBUG_WAV_NAME_PATTERN = re.compile(
    r"\d{8}-\d{6}_(?:complete|uncertain|incomplete)_p\d\.\d{2}(?:_\d{3})?\.wav"
)

# Optional dependencies are imported lazily so the module stays importable in
# environments that only run the fixed-delay path. Tests inject fakes here.
onnxruntime = None
hf_hub_download = None
WhisperFeatureExtractor = None
# The missing-dependency WARNING is emitted at most once per process.
_missing_deps_warned = False


def _ensure_optional_deps() -> bool:
    global onnxruntime, hf_hub_download, WhisperFeatureExtractor, _missing_deps_warned

    missing = []
    if onnxruntime is None:
        try:
            import onnxruntime as _onnxruntime

            onnxruntime = _onnxruntime
        except Exception:
            missing.append("onnxruntime")
    if hf_hub_download is None:
        try:
            from huggingface_hub import hf_hub_download as _hf_hub_download

            hf_hub_download = _hf_hub_download
        except Exception:
            missing.append("huggingface_hub")
    if WhisperFeatureExtractor is None:
        try:
            from transformers import WhisperFeatureExtractor as _feature_extractor

            WhisperFeatureExtractor = _feature_extractor
        except Exception:
            missing.append("transformers")
    if missing:
        if not _missing_deps_warned:
            _missing_deps_warned = True
            logger.warning(
                "Smart Turn (turn_detection) is enabled but its optional "
                "dependencies are missing (%s); falling back to the fixed speech "
                "commit delay. Install them with: pip install -r requirements.txt",
                ", ".join(missing),
            )
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
    uncertain_threshold: float = SMART_TURN_UNCERTAIN_THRESHOLD
    complete_delay_ms: int = 250
    uncertain_delay_ms: int = 900
    incomplete_delay_ms: int = 2500
    fallback_delay_ms: int = 700
    # Shorter VAD end-of-speech silence threshold used only while turn detection
    # is active, so Smart Turn can decide instead of waiting for the long
    # silence window. Ignored when the detector is disabled or unavailable.
    vad_required_misses: int = 8
    # When set, every evaluated turn is written as a 16 kHz mono WAV for tuning.
    debug_save_dir: Optional[str] = None
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
        debug_save_dir = section.get("debug_save_dir")
        debug_save_dir = str(debug_save_dir) if debug_save_dir else None

        return cls(
            enabled=bool(section.get("enabled", True)),
            model=model,
            filename=f"{model}.onnx",
            revision=str(section.get("revision", SMART_TURN_REVISION) or SMART_TURN_REVISION),
            threshold=_float("threshold", SMART_TURN_THRESHOLD),
            uncertain_threshold=_float("uncertain_threshold", SMART_TURN_UNCERTAIN_THRESHOLD),
            complete_delay_ms=_int("complete_delay_ms", 250),
            uncertain_delay_ms=_int("uncertain_delay_ms", 900),
            incomplete_delay_ms=_int("incomplete_delay_ms", 2500),
            fallback_delay_ms=fallback,
            vad_required_misses=_int("vad_required_misses", 8),
            debug_save_dir=debug_save_dir,
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


def resolve_turn_tier(config: "SmartTurnConfig", probability: float) -> str:
    """Classify a detector probability into ``complete``/``uncertain``/``incomplete``."""
    value = float(probability)
    if value >= config.threshold:
        return "complete"
    if value >= config.uncertain_threshold:
        return "uncertain"
    return "incomplete"


def resolve_commit_delay_for_turn(
    config: "SmartTurnConfig",
    verdict,
    *,
    fallback_delay_ms: Optional[int] = None,
    elapsed_ms: float = 0.0,
) -> int:
    """Single source of truth for the three turn cases.

    ``verdict`` is ``None`` when the detector is disabled or unavailable, a
    probability float otherwise (``True``/``False`` still work and map to
    ``1.0``/``0.0`` for backwards compatibility).

    * disabled / unavailable (``verdict is None``) -> fallback delay
    * probability >= ``threshold`` -> ``complete_delay_ms`` minus time already spent
    * ``uncertain_threshold`` <= probability < ``threshold`` -> ``uncertain_delay_ms``
    * probability < ``uncertain_threshold`` -> ``incomplete_delay_ms``

    The fallback is clamped to 0 when the elapsed inference time already
    exceeds the configured fixed delay.
    """
    complete_delay = max(0, int(config.complete_delay_ms - max(0.0, elapsed_ms)))
    fallback = config.fallback_delay_ms if fallback_delay_ms is None else fallback_delay_ms
    if not config.enabled or verdict is None:
        return max(0, int(fallback))

    tier = resolve_turn_tier(config, verdict)
    if tier == "complete":
        delay = complete_delay
    elif tier == "uncertain":
        delay = config.uncertain_delay_ms
    else:
        delay = config.incomplete_delay_ms
    return max(0, int(delay))


def resolve_vad_required_misses(
    config: "SmartTurnConfig",
    *,
    detector_available: bool,
    default_misses: int,
) -> int:
    """Effective VAD end-of-speech silence threshold.

    The shorter turn-detection threshold only applies while Smart Turn is both
    enabled and usable; otherwise the caller's existing value is preserved.
    """
    if config.enabled and detector_available:
        return max(0, int(config.vad_required_misses))
    return max(0, int(default_misses))


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


def save_debug_wav(
    directory,
    samples,
    sample_rate: int,
    tier: str,
    probability: float,
    *,
    max_files: int = SMART_TURN_DEBUG_MAX_FILES,
) -> Optional[str]:
    """Write the evaluated audio as a 16 kHz mono WAV for offline tuning.

    Returns the written path, or ``None`` when disabled. Filenames carry the
    timestamp, the tier and the probability, e.g.
    ``20260921-173933_uncertain_p0.30.wav``. Oldest files beyond ``max_files``
    are deleted so a long session cannot fill the disk.
    """
    if not directory:
        return None

    directory_path = Path(directory)
    directory_path.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    stem = f"{stamp}_{tier}_p{probability:.2f}"
    path = directory_path / f"{stem}.wav"
    if path.exists():
        path = directory_path / f"{stem}_{int(time.time() * 1000) % 1000:03d}.wav"

    audio = np.asarray(samples, dtype=np.float32)
    if audio.size == 0:
        return None
    pcm16 = (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(int(sample_rate))
        handle.writeframes(pcm16.tobytes())

    _prune_debug_wavs(directory_path, max_files)
    return str(path)


def _prune_debug_wavs(directory: Path, max_files: int) -> None:
    try:
        files = sorted(
            (item for item in directory.glob("*.wav") if _DEBUG_WAV_NAME_PATTERN.fullmatch(item.name)),
            key=lambda item: item.name,
        )
    except OSError:
        return
    if len(files) <= max_files:
        return
    for stale in files[: len(files) - max_files]:
        try:
            stale.unlink()
        except OSError:
            pass


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
            self._save_debug_audio(prepared, probability)
            return (probability > self.config.threshold, probability)
        except Exception as exc:
            self._disable(exc)
            return None

    def _save_debug_audio(self, samples: np.ndarray, probability: float) -> None:
        """Persist the evaluated audio when ``debug_save_dir`` is configured."""
        directory = self.config.debug_save_dir
        if not directory:
            return
        try:
            save_debug_wav(
                directory,
                samples,
                self.config.sample_rate,
                resolve_turn_tier(self.config, probability),
                probability,
            )
        except Exception:
            logger.debug("Smart Turn debug audio save failed", exc_info=True)