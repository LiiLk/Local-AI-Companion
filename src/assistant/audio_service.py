"""
Audio Service - Continuous microphone capture with VAD.

This service captures audio from the microphone continuously,
uses VAD to detect speech, and provides callbacks when speech is detected.

Features:
- Continuous mic capture in background thread
- VAD-based speech detection
- Hotkey to mute/unmute
- Callbacks for speech events
"""

import asyncio
import logging
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional

import numpy as np

try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
except ImportError:
    SOUNDDEVICE_AVAILABLE = False
    print("⚠️ sounddevice not installed. Run: pip install sounddevice")

from src.vad import SileroVAD

logger = logging.getLogger(__name__)

# Suppress input overflow warnings (common during model loading/inference)
logging.getLogger("sounddevice").setLevel(logging.ERROR)


class MicState(Enum):
    """Microphone state."""
    LISTENING = "listening"
    MUTED = "muted"
    PROCESSING = "processing"  # Processing speech, ignoring new audio


@dataclass
class AudioServiceConfig:
    """Configuration for audio service."""
    sample_rate: int = 16000
    channels: int = 1
    chunk_duration_ms: int = 32  # 32ms chunks for VAD
    device: Optional[int] = None  # None = auto-select best input device

    # VAD settings
    vad_prob_threshold: float = 0.5
    vad_db_threshold: float = -50
    vad_required_hits: int = 3
    vad_required_misses: int = 30
    vad_min_speech_ms: int = 450
    vad_min_voiced_ms: int = 180

    # Behavior
    auto_start: bool = True
    start_muted: bool = False


class AudioService:
    """
    Continuous audio capture service with VAD.

    Usage:
        service = AudioService()
        service.on_speech_detected = my_callback
        service.start()
        # ... later ...
        service.toggle_mute()  # Mute/unmute
        service.stop()
    """

    def __init__(self, config: Optional[AudioServiceConfig] = None):
        self.config = config or AudioServiceConfig()

        # State
        self._muted_by_user = bool(self.config.start_muted)
        self._processing_blocked = False
        self._state = self._effective_state()
        self._running = False
        self._stream: Optional[object] = None
        self._capture_thread: Optional[threading.Thread] = None
        self._stream_sample_rate = self.config.sample_rate

        # VAD
        from src.vad.silero_vad import VADConfig
        vad_config = VADConfig(
            sample_rate=self.config.sample_rate,
            prob_threshold=self.config.vad_prob_threshold,
            db_threshold=self.config.vad_db_threshold,
            required_hits=self.config.vad_required_hits,
            required_misses=self.config.vad_required_misses,
            min_speech_ms=self.config.vad_min_speech_ms,
            min_voiced_ms=self.config.vad_min_voiced_ms,
        )
        self._vad = SileroVAD(config=vad_config)

        # Callbacks
        self.on_speech_start: Optional[Callable[[], None]] = None
        self.on_speech_end: Optional[Callable[[], None]] = None
        self.on_speech_detected: Optional[Callable[[bytes], None]] = None
        self.on_state_change: Optional[Callable[[MicState], None]] = None

        # Async event loop reference (for calling async callbacks)
        self._loop: Optional[asyncio.AbstractEventLoop] = None

        logger.info(f"AudioService initialized (sample_rate={self.config.sample_rate})")

    @property
    def state(self) -> MicState:
        return self._state

    @property
    def is_muted(self) -> bool:
        return self._state == MicState.MUTED

    @property
    def is_listening(self) -> bool:
        return self._state == MicState.LISTENING

    def _effective_state(self) -> MicState:
        if self._muted_by_user:
            return MicState.MUTED
        if self._processing_blocked:
            return MicState.PROCESSING
        return MicState.LISTENING

    def _set_state(self, new_state: MicState):
        """Set state and notify callback."""
        if self._state != new_state:
            old_state = self._state
            self._state = new_state
            logger.info(f"Mic state: {old_state.value} → {new_state.value}")

            if self.on_state_change:
                self.on_state_change(new_state)

    def start(self, loop: Optional[asyncio.AbstractEventLoop] = None):
        """Start audio capture."""
        if not SOUNDDEVICE_AVAILABLE:
            raise RuntimeError("sounddevice is not installed")

        if self._running:
            logger.warning("AudioService already running")
            return

        self._loop = loop
        self._running = True

        # Start capture thread
        self._capture_thread = threading.Thread(
            target=self._capture_loop,
            daemon=True,
            name="AudioCapture"
        )
        self._capture_thread.start()

        logger.info("🎤 AudioService started")

        self._set_state(self._effective_state())

    def stop(self):
        """Stop audio capture."""
        self._running = False

        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None

        if self._capture_thread:
            self._capture_thread.join(timeout=2.0)
            self._capture_thread = None

        self._vad.reset()
        logger.info("🎤 AudioService stopped")

    def toggle_mute(self) -> bool:
        """Toggle mute state. Returns True if now muted."""
        self._muted_by_user = not self._muted_by_user
        self._set_state(self._effective_state())
        self._vad.reset()
        return self._muted_by_user

    def mute(self):
        """Mute microphone."""
        if not self._muted_by_user:
            self._muted_by_user = True
            self._set_state(self._effective_state())
            self._vad.reset()

    def unmute(self):
        """Unmute microphone."""
        if self._muted_by_user:
            self._muted_by_user = False
            self._set_state(self._effective_state())
            self._vad.reset()

    def set_processing(self, processing: bool):
        """Set processing state (ignores audio while AI is speaking)."""
        if self._processing_blocked != processing:
            self._processing_blocked = processing
            self._set_state(self._effective_state())
            self._vad.reset()

    def _list_input_devices(self) -> list[tuple[int, dict]]:
        devices = []
        for index, info in enumerate(sd.query_devices()):
            if info.get("max_input_channels", 0) > 0:
                devices.append((index, info))
        return devices

    def _get_hostapi_name(self, device_info: dict) -> str:
        try:
            return str(sd.query_hostapis(device_info["hostapi"])["name"])
        except Exception:
            return "Unknown"

    def _iter_input_device_candidates(self):
        if self.config.device is not None:
            yield self.config.device
            return

        input_devices = self._list_input_devices()
        if not input_devices:
            return

        default_devices = sd.default.device
        default_input = None
        try:
            default_input = int(default_devices[0])
        except Exception:
            if isinstance(default_devices, int):
                default_input = default_devices

        default_name = None
        if default_input is not None:
            try:
                default_name = str(sd.query_devices(default_input).get("name", "")).lower()
            except Exception:
                default_name = None

        def sort_key(item: tuple[int, dict]) -> tuple[int, int]:
            index, info = item
            hostapi = self._get_hostapi_name(info).lower()
            name = str(info.get("name", "")).lower()

            priority = 50
            if "wasapi" in hostapi:
                priority = 0
            elif "wdm-ks" in hostapi:
                priority = 10
            elif "directsound" in hostapi:
                priority = 20
            elif "mme" in hostapi:
                priority = 30

            if index == default_input:
                priority -= 1000
            elif default_name and name == default_name:
                priority -= 200

            if "mappeur de sons microsoft" in name or "microsoft sound mapper" in name:
                priority += 100

            return (priority, index)

        seen: set[int] = set()
        for index, _ in sorted(input_devices, key=sort_key):
            if index not in seen:
                seen.add(index)
                yield index

    def _iter_sample_rates(self, device_info: dict):
        seen: set[int] = set()
        for candidate in (
            int(self.config.sample_rate),
            int(round(device_info.get("default_samplerate") or 0)),
        ):
            if candidate > 0 and candidate not in seen:
                seen.add(candidate)
                yield candidate

    def _build_stream_kwargs(self, device_index: int, sample_rate: int, blocksize: int, callback):
        kwargs = {
            "samplerate": sample_rate,
            "channels": self.config.channels,
            "dtype": np.float32,
            "blocksize": blocksize,
            "device": device_index,
            "latency": "low",
            "callback": callback,
        }

        device_info = sd.query_devices(device_index)
        hostapi_name = self._get_hostapi_name(device_info)
        if hasattr(sd, "WasapiSettings") and "WASAPI" in hostapi_name.upper():
            kwargs["extra_settings"] = sd.WasapiSettings(exclusive=False)

        return kwargs

    def _resample_audio(self, audio: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
        if src_rate == dst_rate or audio.size == 0:
            return audio

        try:
            import soxr

            return soxr.resample(
                audio.astype(np.float32, copy=False),
                src_rate,
                dst_rate,
            ).astype(np.float32, copy=False)
        except Exception as exc:
            logger.debug(
                "soxr resampling unavailable, falling back to linear interpolation: %s",
                exc,
            )

        target_length = max(1, int(round(audio.size * dst_rate / src_rate)))
        src_positions = np.arange(audio.size, dtype=np.float32)
        dst_positions = np.linspace(0, audio.size - 1, num=target_length, dtype=np.float32)
        return np.interp(dst_positions, src_positions, audio).astype(np.float32)

    def _open_input_stream(self, callback) -> None:
        last_error = None
        attempts: list[str] = []

        for device_index in self._iter_input_device_candidates():
            device_info = sd.query_devices(device_index)
            hostapi_name = self._get_hostapi_name(device_info)
            device_name = device_info.get("name", f"device {device_index}")

            for sample_rate in self._iter_sample_rates(device_info):
                blocksize = max(1, int(sample_rate * self.config.chunk_duration_ms / 1000))
                kwargs = self._build_stream_kwargs(device_index, sample_rate, blocksize, callback)

                try:
                    self._stream = sd.InputStream(**kwargs)
                    self._stream.start()
                    self._stream_sample_rate = sample_rate
                    logger.info(
                        "🎤 Audio stream started (device=%s: %s [%s], samplerate=%s)",
                        device_index,
                        device_name,
                        hostapi_name,
                        sample_rate,
                    )
                    return
                except Exception as exc:
                    last_error = exc
                    attempts.append(f"{device_name} [{hostapi_name}] @ {sample_rate}Hz -> {exc}")
                    logger.warning(
                        "Audio device open failed for %s [%s] @ %sHz: %s",
                        device_name,
                        hostapi_name,
                        sample_rate,
                        exc,
                    )

        if last_error is None:
            raise RuntimeError("No input devices with capture channels were found")

        attempt_summary = "; ".join(attempts[:4])
        raise RuntimeError(f"Unable to open any input device. Attempts: {attempt_summary}") from last_error

    def _capture_loop(self):
        """Main capture loop running in separate thread."""

        def audio_callback(indata, frames, time_info, status):
            if status:
                # Input overflow is common during heavy GPU processing, only log at debug
                if "overflow" in str(status).lower():
                    logger.debug(f"Audio status: {status}")
                else:
                    logger.warning(f"Audio status: {status}")

            # Only process if listening
            if self._state != MicState.LISTENING:
                return

            # Convert to float32 and resample if stream had to open at a different rate
            audio_float = indata[:, 0].astype(np.float32)
            if self._stream_sample_rate != self.config.sample_rate:
                audio_float = self._resample_audio(
                    audio_float,
                    self._stream_sample_rate,
                    self.config.sample_rate,
                )
                if audio_float.size == 0:
                    return

            # Process through VAD without an intermediate Python list conversion.
            for event in self._vad.process_audio(audio_float):
                if event == b"<|START|>":
                    logger.debug("Speech started")
                    if self.on_speech_start:
                        self._call_callback(self.on_speech_start)

                elif event == b"<|END|>":
                    logger.debug("Speech ended")
                    if self.on_speech_end:
                        self._call_callback(self.on_speech_end)

                elif len(event) > 100:
                    # This is audio data
                    logger.info(f"🎤 Speech detected: {len(event)} bytes")
                    if self.on_speech_detected:
                        self._call_callback(self.on_speech_detected, event)

        try:
            self._open_input_stream(audio_callback)

            # Keep thread alive
            while self._running:
                time.sleep(0.1)

        except Exception as exc:
            logger.error(f"Audio capture error: {exc}")
            self._running = False

    def _call_callback(self, callback: Callable, *args):
        """Call callback, handling async if needed."""
        try:
            if asyncio.iscoroutinefunction(callback):
                if self._loop and self._loop.is_running():
                    asyncio.run_coroutine_threadsafe(callback(*args), self._loop)
                else:
                    # Create new loop for this call
                    asyncio.run(callback(*args))
            else:
                callback(*args)
        except Exception as exc:
            logger.error(f"Callback error: {exc}")


# Convenience function
def create_audio_service(config_dict: Optional[dict] = None) -> AudioService:
    """Create an AudioService from config dict."""
    if config_dict:
        config = AudioServiceConfig(**config_dict)
    else:
        config = AudioServiceConfig()
    return AudioService(config)
