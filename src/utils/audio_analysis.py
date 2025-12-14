"""
Audio Analysis Utilities for Live2D Lip-Sync

Provides fast audio volume extraction from WAV files for real-time lip-sync.
"""

import logging
import wave
from pathlib import Path
from typing import Union

import numpy as np

logger = logging.getLogger(__name__)


def analyze_audio_volumes(
    audio_data: Union[bytes, Path, str],
    sample_rate: int = 24000,
    chunk_ms: int = 50,
    normalize_divisor: float = 8000.0,
    threshold: float = 0.05
) -> list[float]:
    """
    Analyze audio and return volume levels per chunk for lip-sync.
    
    Args:
        audio_data: Raw PCM bytes (16-bit signed) or path to WAV file
        sample_rate: Audio sample rate in Hz
        chunk_ms: Chunk duration in milliseconds (50ms ~= 20fps)
        normalize_divisor: Divisor for RMS normalization (lower = more sensitive)
        threshold: Minimum volume threshold (values below are set to 0)
        
    Returns:
        List of normalized volume values (0.0 - 1.0) per chunk
    """
    # Load audio data
    if isinstance(audio_data, (Path, str)):
        audio_bytes, sample_rate = read_wav_pcm(Path(audio_data))
    else:
        audio_bytes = audio_data
    
    # Convert bytes to samples
    samples = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
    
    if len(samples) == 0:
        return []
    
    # Calculate chunk size in samples
    chunk_samples = int(sample_rate * chunk_ms / 1000)
    
    volumes = []
    for i in range(0, len(samples), chunk_samples):
        chunk = samples[i:i + chunk_samples]
        if len(chunk) == 0:
            continue
        
        # Calculate RMS (Root Mean Square)
        rms = np.sqrt(np.mean(chunk ** 2))
        
        # Normalize
        normalized = min(1.0, rms / normalize_divisor)
        
        # Apply threshold to filter silence
        if normalized < threshold:
            normalized = 0.0
        
        volumes.append(round(float(normalized), 3))
    
    return volumes


def read_wav_pcm(wav_path: Path) -> tuple[bytes, int]:
    """
    Read raw PCM data from a WAV file.
    
    Args:
        wav_path: Path to WAV file
        
    Returns:
        Tuple of (raw PCM bytes, sample rate)
    """
    with wave.open(str(wav_path), 'rb') as wf:
        sample_rate = wf.getframerate()
        frames = wf.readframes(wf.getnframes())
        return frames, sample_rate


def calculate_audio_duration_ms(audio_bytes: bytes, sample_rate: int) -> int:
    """
    Calculate audio duration in milliseconds.
    
    Args:
        audio_bytes: Raw PCM bytes (16-bit = 2 bytes per sample)
        sample_rate: Sample rate in Hz
        
    Returns:
        Duration in milliseconds
    """
    num_samples = len(audio_bytes) // 2  # 16-bit = 2 bytes per sample
    duration_sec = num_samples / sample_rate
    return int(duration_sec * 1000)
