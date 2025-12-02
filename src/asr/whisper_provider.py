"""
Whisper ASR Provider using faster-whisper.

faster-whisper is a reimplementation of OpenAI's Whisper using CTranslate2,
which is up to 4x faster than the original with the same accuracy.

Features:
- Local processing (100% private)
- Multiple model sizes (tiny to large-v3)
- Automatic language detection
- Word-level timestamps
- GPU acceleration with CUDA
"""

import asyncio
from pathlib import Path
from typing import Optional, List, AsyncGenerator
from dataclasses import dataclass

from .base import BaseASR, BaseRealtimeASR, ASRResult, ASRSegment


# Model size configurations
MODEL_SIZES = {
    "tiny": {"params": "39M", "vram": "~1GB", "speed": "fastest"},
    "base": {"params": "74M", "vram": "~1GB", "speed": "fast"},
    "small": {"params": "244M", "vram": "~2GB", "speed": "medium"},
    "medium": {"params": "769M", "vram": "~5GB", "speed": "slow"},
    "large-v3": {"params": "1.5B", "vram": "~10GB", "speed": "slowest"},
}

# Whisper supported languages (subset of most common)
SUPPORTED_LANGUAGES = [
    "en", "fr", "de", "es", "it", "pt", "nl", "pl", "ru", "ja", 
    "ko", "zh", "ar", "hi", "tr", "vi", "th", "id", "uk", "cs"
]


class WhisperProvider(BaseASR):
    """
    ASR provider using faster-whisper for local speech-to-text.
    
    Uses lazy loading to only load the model when first needed.
    
    Args:
        model_size: Size of the Whisper model (tiny, base, small, medium, large-v3)
        device: Device to run on ("cuda", "cpu", or "auto")
        compute_type: Precision ("float16", "int8", "float32")
        
    Example:
        >>> asr = WhisperProvider(model_size="small")
        >>> result = asr.transcribe("audio.wav")
        >>> print(result.text)
        "Bonjour, comment ça va ?"
    """
    
    def __init__(
        self,
        model_size: str = "base",
        device: str = "auto",
        compute_type: str = "float16"
    ):
        if model_size not in MODEL_SIZES:
            raise ValueError(
                f"Invalid model size '{model_size}'. "
                f"Choose from: {list(MODEL_SIZES.keys())}"
            )
        
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        
        # Lazy loading - model loaded on first use
        self._model = None
        
    def _get_model(self):
        """
        Lazy load the Whisper model.
        
        Only loads the model on first use to save memory and startup time.
        """
        if self._model is None:
            try:
                from faster_whisper import WhisperModel
            except ImportError:
                raise ImportError(
                    "faster-whisper is not installed. "
                    "Install it with: pip install faster-whisper"
                )
            
            print(f"🔄 Chargement de Whisper ({self.model_size})...")
            
            # Determine device
            if self.device == "auto":
                # Try CUDA first, fallback to CPU if issues
                try:
                    import torch
                    if torch.cuda.is_available():
                        device = "cuda"
                    else:
                        device = "cpu"
                except:
                    device = "cpu"
            else:
                device = self.device
                
            # Adjust compute type for CPU
            compute_type = self.compute_type
            if device == "cpu":
                compute_type = "int8"  # float16 not supported on CPU
            
            # Try CUDA first, fallback to CPU if cuDNN issues
            try:
                self._model = WhisperModel(
                    self.model_size,
                    device=device,
                    compute_type=compute_type
                )
            except Exception as e:
                if "cudnn" in str(e).lower() or device == "cuda":
                    print(f"⚠️ CUDA error, falling back to CPU: {e}")
                    device = "cpu"
                    compute_type = "int8"
                    self._model = WhisperModel(
                        self.model_size,
                        device=device,
                        compute_type=compute_type
                    )
                else:
                    raise
            
            print(f"✅ Whisper chargé ! (device={device}, compute={compute_type})")
            
        return self._model
    
    def transcribe(
        self,
        audio_path: str | Path,
        language: Optional[str] = None
    ) -> ASRResult:
        """
        Transcribe an audio file to text.
        
        Args:
            audio_path: Path to audio file (WAV, MP3, FLAC, etc.)
            language: Language code (e.g., "fr"). None for auto-detection.
            
        Returns:
            ASRResult with transcribed text and metadata
        """
        model = self._get_model()
        audio_path = Path(audio_path)
        
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        # Transcribe with faster-whisper
        segments, info = model.transcribe(
            str(audio_path),
            language=language,
            beam_size=5,
            word_timestamps=True,
            vad_filter=True,  # Filter out silence
        )
        
        # Collect all segments
        all_segments = []
        full_text_parts = []
        
        for segment in segments:
            full_text_parts.append(segment.text)
            all_segments.append({
                "text": segment.text.strip(),
                "start": segment.start,
                "end": segment.end,
                "confidence": getattr(segment, 'avg_logprob', None)
            })
        
        full_text = " ".join(full_text_parts).strip()
        
        return ASRResult(
            text=full_text,
            language=info.language,
            confidence=info.language_probability,
            duration=info.duration,
            segments=all_segments
        )
    
    async def transcribe_stream(
        self,
        audio_path: str | Path,
        language: Optional[str] = None
    ) -> AsyncGenerator[ASRSegment, None]:
        """
        Transcribe audio and yield segments as they're processed.
        
        Useful for displaying transcription in real-time.
        """
        model = self._get_model()
        audio_path = Path(audio_path)
        
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        # Transcribe with faster-whisper
        segments, info = model.transcribe(
            str(audio_path),
            language=language,
            beam_size=5,
            word_timestamps=True,
            vad_filter=True,
        )
        
        for segment in segments:
            yield ASRSegment(
                text=segment.text.strip(),
                start=segment.start,
                end=segment.end,
                confidence=getattr(segment, 'avg_logprob', None)
            )
            # Small delay to simulate streaming
            await asyncio.sleep(0.01)
    
    def get_supported_languages(self) -> List[str]:
        """Get list of supported language codes."""
        return SUPPORTED_LANGUAGES.copy()
    
    def get_model_info(self) -> dict:
        """Get information about the loaded model."""
        info = MODEL_SIZES[self.model_size].copy()
        info["model_size"] = self.model_size
        info["device"] = self.device
        info["compute_type"] = self.compute_type
        info["loaded"] = self._model is not None
        return info


class RealtimeWhisperProvider(BaseRealtimeASR, WhisperProvider):
    """
    Real-time Whisper ASR with microphone input.
    
    Extends WhisperProvider with the ability to listen from the microphone
    and perform voice activity detection.
    
    Requires: sounddevice, webrtcvad (optional for better VAD)
    """
    
    def __init__(
        self,
        model_size: str = "base",
        device: str = "auto",
        compute_type: str = "float16",
        sample_rate: int = 16000,
        silence_threshold: float = 0.5,  # seconds of silence to stop
    ):
        super().__init__(model_size, device, compute_type)
        
        self.sample_rate = sample_rate
        self.silence_threshold = silence_threshold
        self._is_listening = False
        self._audio_buffer = []
        
    def start_listening(self) -> None:
        """Start listening from the microphone."""
        self._is_listening = True
        self._audio_buffer = []
        
    def stop_listening(self) -> None:
        """Stop listening from the microphone."""
        self._is_listening = False
        
    def is_listening(self) -> bool:
        """Check if currently listening."""
        return self._is_listening
    
    async def listen_once(
        self, 
        timeout: Optional[float] = None,
        save_path: Optional[str] = None
    ) -> ASRResult:
        """
        Listen for a single utterance and transcribe it.
        
        Records audio until silence is detected, then transcribes.
        
        Args:
            timeout: Maximum recording time in seconds
            save_path: Optional path to save the recorded audio
            
        Returns:
            ASRResult with the transcribed speech
        """
        try:
            import sounddevice as sd
            import numpy as np
            import tempfile
            import soundfile as sf
        except ImportError:
            raise ImportError(
                "Required packages not installed. Run:\n"
                "pip install sounddevice soundfile numpy"
            )
        
        print("🎤 Écoute... (parlez maintenant)")
        
        # Recording parameters
        duration = timeout or 10.0  # Default 10 seconds max
        channels = 1
        
        # Record audio
        recording = sd.rec(
            int(duration * self.sample_rate),
            samplerate=self.sample_rate,
            channels=channels,
            dtype=np.float32
        )
        
        # Wait for recording or timeout
        try:
            sd.wait()
        except KeyboardInterrupt:
            sd.stop()
        
        print("✅ Enregistrement terminé, transcription...")
        
        # Trim silence from end (simple energy-based)
        audio = recording.flatten()
        
        # Find last non-silent sample
        energy = np.abs(audio)
        threshold = 0.01
        non_silent = np.where(energy > threshold)[0]
        
        if len(non_silent) > 0:
            last_sound = non_silent[-1]
            # Add a small buffer after last sound
            end_idx = min(last_sound + int(0.3 * self.sample_rate), len(audio))
            audio = audio[:end_idx]
        
        # Save to temporary file
        if save_path:
            audio_path = save_path
        else:
            temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            audio_path = temp_file.name
            temp_file.close()
            
        sf.write(audio_path, audio, self.sample_rate)
        
        # Transcribe
        result = self.transcribe(audio_path)
        
        # Clean up temp file if not saving
        if not save_path:
            Path(audio_path).unlink(missing_ok=True)
        
        return result


# Convenience function
def create_whisper_asr(
    model_size: str = "base",
    realtime: bool = False,
    **kwargs
) -> BaseASR:
    """
    Factory function to create a Whisper ASR provider.
    
    Args:
        model_size: Size of Whisper model
        realtime: If True, create RealtimeWhisperProvider with mic support
        **kwargs: Additional arguments passed to provider
        
    Returns:
        WhisperProvider or RealtimeWhisperProvider instance
    """
    if realtime:
        return RealtimeWhisperProvider(model_size=model_size, **kwargs)
    return WhisperProvider(model_size=model_size, **kwargs)
