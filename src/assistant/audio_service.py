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
import queue
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
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
    device: Optional[int] = None  # None = default device
    
    # VAD settings
    vad_prob_threshold: float = 0.5
    vad_db_threshold: float = -50
    vad_required_hits: int = 3
    vad_required_misses: int = 30
    
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
        self._state = MicState.MUTED if self.config.start_muted else MicState.LISTENING
        self._running = False
        self._stream: Optional[sd.InputStream] = None
        self._capture_thread: Optional[threading.Thread] = None
        
        # VAD
        from src.vad.silero_vad import VADConfig
        vad_config = VADConfig(
            sample_rate=self.config.sample_rate,
            prob_threshold=self.config.vad_prob_threshold,
            db_threshold=self.config.vad_db_threshold,
            required_hits=self.config.vad_required_hits,
            required_misses=self.config.vad_required_misses,
        )
        self._vad = SileroVAD(config=vad_config)
        
        # Audio queue for processing
        self._audio_queue: queue.Queue = queue.Queue()
        
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
        
        if not self.config.start_muted:
            self._set_state(MicState.LISTENING)
    
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
        if self._state == MicState.MUTED:
            self._set_state(MicState.LISTENING)
            self._vad.reset()
            return False
        elif self._state == MicState.LISTENING:
            self._set_state(MicState.MUTED)
            self._vad.reset()
            return True
        # If processing, don't change state
        return self._state == MicState.MUTED
    
    def mute(self):
        """Mute microphone."""
        if self._state != MicState.MUTED:
            self._set_state(MicState.MUTED)
            self._vad.reset()
    
    def unmute(self):
        """Unmute microphone."""
        if self._state == MicState.MUTED:
            self._set_state(MicState.LISTENING)
            self._vad.reset()
    
    def set_processing(self, processing: bool):
        """Set processing state (ignores audio while AI is speaking)."""
        if processing:
            self._set_state(MicState.PROCESSING)
            self._vad.reset()
        else:
            self._set_state(MicState.LISTENING)
    
    def _capture_loop(self):
        """Main capture loop running in separate thread."""
        chunk_samples = int(self.config.sample_rate * self.config.chunk_duration_ms / 1000)
        
        def audio_callback(indata, frames, time_info, status):
            if status:
                logger.warning(f"Audio status: {status}")
            
            # Only process if listening
            if self._state != MicState.LISTENING:
                return
            
            # Convert to float32 and normalize
            audio_float = indata[:, 0].astype(np.float32)
            
            # Process through VAD
            for event in self._vad.process_audio(audio_float.tolist()):
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
            self._stream = sd.InputStream(
                samplerate=self.config.sample_rate,
                channels=self.config.channels,
                dtype=np.float32,
                blocksize=chunk_samples,
                device=self.config.device,
                callback=audio_callback
            )
            self._stream.start()
            logger.info(f"🎤 Audio stream started (device={self.config.device or 'default'})")
            
            # Keep thread alive
            while self._running:
                time.sleep(0.1)
                
        except Exception as e:
            logger.error(f"Audio capture error: {e}")
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
        except Exception as e:
            logger.error(f"Callback error: {e}")


# Convenience function
def create_audio_service(config_dict: Optional[dict] = None) -> AudioService:
    """Create an AudioService from config dict."""
    if config_dict:
        config = AudioServiceConfig(**config_dict)
    else:
        config = AudioServiceConfig()
    return AudioService(config)
