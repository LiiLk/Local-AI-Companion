"""
Silero VAD - Voice Activity Detection using Silero model.

Based on Open-LLM-VTuber's implementation.
Uses a state machine to detect speech start/end with smoothing.
"""

import numpy as np
import torch
from collections import deque
from enum import Enum
from dataclasses import dataclass
from typing import Generator, Optional
from silero_vad import load_silero_vad


class State(Enum):
    """VAD State Machine states."""
    IDLE = 1      # Waiting for speech
    ACTIVE = 2    # Speech detected
    INACTIVE = 3  # Speech ended, waiting for more or timeout


@dataclass
class VADConfig:
    """Configuration for Silero VAD."""
    sample_rate: int = 16000
    prob_threshold: float = 0.5      # Probability threshold for speech
    db_threshold: float = -50        # dB threshold (more sensitive)
    required_hits: int = 3           # Consecutive frames to confirm speech start (~100ms)
    required_misses: int = 24        # Consecutive frames to confirm speech end (~800ms)
    smoothing_window: int = 5        # Smoothing window for probability


class SileroVAD:
    """
    Silero Voice Activity Detection.
    
    Processes audio in chunks and yields speech segments.
    """
    
    def __init__(self, config: Optional[VADConfig] = None):
        self.config = config or VADConfig()
        
        print("🔄 Loading Silero VAD model...")
        self.model = load_silero_vad()
        print("✅ Silero VAD loaded!")
        
        # State machine
        self.state = State.IDLE
        self.hit_count = 0
        self.miss_count = 0
        
        # Buffers
        self.audio_buffer = bytearray()
        self.pre_buffer = deque(maxlen=20)  # Keep last ~640ms before speech
        
        # Smoothing windows
        self.prob_window = deque(maxlen=self.config.smoothing_window)
        self.db_window = deque(maxlen=self.config.smoothing_window)
        
        # Chunk size: 512 samples @ 16kHz = 32ms
        self.chunk_size = 512 if self.config.sample_rate == 16000 else 256
    
    def reset(self):
        """Reset VAD state."""
        self.state = State.IDLE
        self.hit_count = 0
        self.miss_count = 0
        self.audio_buffer.clear()
        self.pre_buffer.clear()
        self.prob_window.clear()
        self.db_window.clear()
        self.model.reset_states()
    
    @staticmethod
    def calculate_db(audio: np.ndarray) -> float:
        """Calculate dB level of audio chunk."""
        rms = np.sqrt(np.mean(np.square(audio.astype(np.float32))))
        if rms > 0:
            return 20 * np.log10(rms + 1e-7)
        return -100.0
    
    def _get_smoothed(self, prob: float, db: float) -> tuple[float, float]:
        """Get smoothed probability and dB values."""
        self.prob_window.append(prob)
        self.db_window.append(db)
        return np.mean(self.prob_window), np.mean(self.db_window)
    
    def process_chunk(self, audio_float: np.ndarray) -> Generator[bytes, None, None]:
        """
        Process a single audio chunk through the state machine.
        
        Args:
            audio_float: Audio samples as float32 numpy array (normalized -1 to 1)
            
        Yields:
            b"<|START|>" when speech starts
            b"<|END|>" when speech ends  
            Audio bytes when speech segment is complete
        """
        # Convert to int16 for storage
        audio_int16 = (audio_float * 32767).astype(np.int16)
        chunk_bytes = audio_int16.tobytes()
        
        # Calculate dB
        db = self.calculate_db(audio_int16)
        
        # Get speech probability from Silero
        with torch.no_grad():
            tensor = torch.from_numpy(audio_float)
            prob = self.model(tensor, self.config.sample_rate).item()
        
        # Smooth values
        smoothed_prob, smoothed_db = self._get_smoothed(prob, db)
        
        # Check thresholds
        is_speech = (smoothed_prob >= self.config.prob_threshold and 
                     smoothed_db >= self.config.db_threshold)
        
        # State machine transitions
        if self.state == State.IDLE:
            self.pre_buffer.append(chunk_bytes)
            
            if is_speech:
                self.hit_count += 1
                if self.hit_count >= self.config.required_hits:
                    # Speech started!
                    self.state = State.ACTIVE
                    self.hit_count = 0
                    
                    # Include pre-buffer audio
                    for pre_chunk in self.pre_buffer:
                        self.audio_buffer.extend(pre_chunk)
                    self.audio_buffer.extend(chunk_bytes)
                    self.pre_buffer.clear()
                    
                    yield b"<|START|>"
            else:
                self.hit_count = 0
                
        elif self.state == State.ACTIVE:
            self.audio_buffer.extend(chunk_bytes)
            
            if is_speech:
                self.miss_count = 0
            else:
                self.miss_count += 1
                if self.miss_count >= self.config.required_misses:
                    # Speech ended!
                    self.state = State.IDLE
                    self.miss_count = 0
                    
                    # Yield the complete audio segment
                    if len(self.audio_buffer) > 1024:  # At least some audio
                        yield bytes(self.audio_buffer)
                    
                    self.audio_buffer.clear()
                    yield b"<|END|>"
    
    def process_audio(self, audio_data: list[float]) -> Generator[bytes, None, None]:
        """
        Process audio data (list of float samples).
        
        Args:
            audio_data: List of float samples (-1 to 1)
            
        Yields:
            Speech events and audio segments
        """
        audio_np = np.array(audio_data, dtype=np.float32)
        
        # Process in chunks
        for i in range(0, len(audio_np), self.chunk_size):
            chunk = audio_np[i:i + self.chunk_size]
            if len(chunk) < self.chunk_size:
                # Pad last chunk
                chunk = np.pad(chunk, (0, self.chunk_size - len(chunk)))
            
            yield from self.process_chunk(chunk)
    
    def force_end(self) -> Optional[bytes]:
        """Force end current speech segment and return audio if any."""
        if self.state == State.ACTIVE and len(self.audio_buffer) > 1024:
            audio = bytes(self.audio_buffer)
            self.reset()
            return audio
        self.reset()
        return None
