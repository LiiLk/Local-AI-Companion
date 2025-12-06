"""
Whisper ASR Provider using faster-whisper.

faster-whisper is a reimplementation of OpenAI's Whisper using CTranslate2,
which is up to 4x faster than the original with the same accuracy.

Features:
- Local processing (100% private)
- Multiple model sizes (tiny to large-v3)
- French-optimized distilled models from bofenghuang
- Automatic language detection
- Word-level timestamps
- GPU acceleration with CUDA
"""

import asyncio
import os
import numpy as np
from pathlib import Path
from typing import Optional, List, AsyncGenerator, Union
from dataclasses import dataclass

from .base import BaseASR, BaseRealtimeASR, ASRResult, ASRSegment


# Directory for locally cached models
MODELS_DIR = Path(__file__).parent.parent.parent / "models"


# Model size configurations
MODEL_SIZES = {
    "tiny": {"params": "39M", "vram": "~1GB", "speed": "fastest", "quality": "low"},
    "base": {"params": "74M", "vram": "~1GB", "speed": "fast", "quality": "medium"},
    "small": {"params": "244M", "vram": "~2GB", "speed": "medium", "quality": "good"},
    "medium": {"params": "769M", "vram": "~5GB", "speed": "slow", "quality": "very good"},
    "large-v2": {"params": "1.5B", "vram": "~10GB", "speed": "slowest", "quality": "excellent"},
    "large-v3": {"params": "1.5B", "vram": "~10GB", "speed": "slowest", "quality": "best"},
    # Turbo: pruned large-v3 (4 decoder layers instead of 32) - 6x faster!
    "turbo": {"params": "809M", "vram": "~6GB", "speed": "fast", "quality": "excellent"},
    "large-v3-turbo": {"params": "809M", "vram": "~6GB", "speed": "fast", "quality": "excellent"},
    # Distil models (English only - not recommended for French)
    "distil-large-v3": {"params": "756M", "vram": "~4GB", "speed": "fastest", "quality": "good (EN only)"},
    # ============================================================
    # 🇫🇷 FRENCH-OPTIMIZED MODELS (from bofenghuang)
    # Fine-tuned on 2500+ hours of French data, much better than base Whisper!
    # Uses CTranslate2 format for Faster-Whisper compatibility.
    # ============================================================
    "french-distil-dec4": {
        "params": "0.8B", "vram": "~3GB", "speed": "fast", "quality": "excellent (FR)",
        "hf_repo": "bofenghuang/whisper-large-v3-french-distil-dec4",
        "local_path": "whisper-french-distil-dec4",
    },
    "french-distil-dec2": {
        "params": "0.8B", "vram": "~3GB", "speed": "fastest", "quality": "very good (FR)",
        "hf_repo": "bofenghuang/whisper-large-v3-french-distil-dec2",
        "local_path": "whisper-french-distil-dec2",
    },
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
        initial_prompt: Prompt to guide language detection and transcription style.
                       Should be in the expected language of the audio.
        
    Example:
        >>> asr = WhisperProvider(model_size="small", initial_prompt="Transcription en français.")
        >>> result = asr.transcribe("audio.wav")
        >>> print(result.text)
        "Bonjour, comment ça va ?"
    """
    
    # Beam search provides better accuracy at slight speed cost
    # Set to 1 for fastest speed (greedy decoding)
    BEAM_SIZE = 1
    
    def __init__(
        self,
        model_size: str = "base",
        device: str = "auto",
        compute_type: str = "float16",
        initial_prompt: Optional[str] = None
    ):
        if model_size not in MODEL_SIZES:
            raise ValueError(
                f"Invalid model size '{model_size}'. "
                f"Choose from: {list(MODEL_SIZES.keys())}"
            )
        
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        # initial_prompt helps Whisper detect the correct language
        # Should be in the expected language (e.g. French for French audio)
        self.initial_prompt = initial_prompt
        
        # Lazy loading - model loaded on first use
        self._model = None
        
    def _download_french_model(self, model_config: dict) -> Path:
        """
        Download French-optimized model from HuggingFace to local cache.
        
        Only downloads the CTranslate2 format files needed for faster-whisper.
        """
        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            raise ImportError(
                "huggingface_hub is not installed. "
                "Install it with: pip install huggingface_hub"
            )
        
        hf_repo = model_config["hf_repo"]
        local_path = MODELS_DIR / model_config["local_path"]
        ctranslate2_path = local_path / "ctranslate2"
        
        # Check if already downloaded
        if ctranslate2_path.exists() and (ctranslate2_path / "model.bin").exists():
            print(f"✅ French model already cached: {local_path}")
            return ctranslate2_path
        
        print(f"📥 Downloading French model from {hf_repo}...")
        print(f"   (This will only be done once)")
        
        # Create models directory
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        
        # Download only CTranslate2 files (smaller than full model)
        snapshot_download(
            repo_id=hf_repo,
            local_dir=str(local_path),
            allow_patterns="ctranslate2/*",
            local_dir_use_symlinks=False,
        )
        
        print(f"✅ Model downloaded: {local_path}")
        return ctranslate2_path
    
    def _get_model(self):
        """
        Lazy load the Whisper model.
        
        Only loads the model on first use to save memory and startup time.
        For French models, downloads from HuggingFace if not cached locally.
        """
        if self._model is None:
            try:
                from faster_whisper import WhisperModel
            except ImportError:
                raise ImportError(
                    "faster-whisper is not installed. "
                    "Install it with: pip install faster-whisper"
                )
            
            model_config = MODEL_SIZES.get(self.model_size, {})
            is_french_model = "hf_repo" in model_config
            
            if is_french_model:
                print(f"🔄 Loading French model ({self.model_size})...")
            else:
                print(f"🔄 Loading Whisper ({self.model_size})...")
            
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
            
            # Get model path (download French model if needed)
            if is_french_model:
                model_path = str(self._download_french_model(model_config))
            else:
                model_path = self.model_size
            
            # Try CUDA first, fallback to CPU if cuDNN issues
            try:
                self._model = WhisperModel(
                    model_path,
                    device=device,
                    compute_type=compute_type,
                    cpu_threads=4  # Optimize for CPU
                )
            except Exception as e:
                if "cudnn" in str(e).lower() or device == "cuda":
                    print(f"⚠️ CUDA error, falling back to CPU: {e}")
                    device = "cpu"
                    compute_type = "int8"
                    self._model = WhisperModel(
                        model_path,
                        device=device,
                        compute_type=compute_type,
                        cpu_threads=4  # Optimize for CPU
                    )
                else:
                    raise
            
            model_name = self.model_size if not is_french_model else f"{self.model_size} 🇫🇷"
            print(f"✅ {model_name} loaded! (device={device}, compute={compute_type})")
            
        return self._model
    
    def transcribe(
        self,
        audio_input: Union[str, Path, np.ndarray],
        language: Optional[str] = None,
        initial_prompt: Optional[str] = None
    ) -> ASRResult:
        """
        Transcribe an audio file or numpy array to text.
        
        Args:
            audio_input: Path to audio file or numpy array of samples
            language: Language code (e.g., "fr"). None or "" or "auto" for auto-detection.
            initial_prompt: Override instance prompt for this transcription.
                           Helps with language detection and style.
            
        Returns:
            ASRResult with transcribed text and metadata
        """
        model = self._get_model()
        
        # Handle input type
        if isinstance(audio_input, (str, Path)):
            audio_path = Path(audio_input)
            if not audio_path.exists():
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
            
            # Log audio file info for debugging
            import os
            file_size = os.path.getsize(audio_path)
            print(f"🎤 Transcribing {audio_path.name} ({file_size/1024:.1f} KB)")
            transcribe_input = str(audio_path)
        else:
            # Numpy array
            print(f"🎤 Transcribing audio buffer ({len(audio_input)} samples)")
            transcribe_input = audio_input
        
        # Normalize language setting
        # Empty string or "auto" means auto-detection
        effective_language = language if language and language.lower() != "auto" else None
        
        # Use provided prompt or instance default
        effective_prompt = initial_prompt or self.initial_prompt
        
        # Transcribe with faster-whisper (same approach as Open-LLM-VTuber)
        segments, info = model.transcribe(
            transcribe_input,
            language=effective_language,
            beam_size=self.BEAM_SIZE,
            word_timestamps=True,
            vad_filter=True,  # Filter out silence using Silero VAD
            vad_parameters=dict(
                min_silence_duration_ms=500,  # Minimum silence to split
                speech_pad_ms=400,  # Padding around speech
            ),
            # Anti-hallucination settings:
            condition_on_previous_text=False,  # Prevents repeated hallucinations
            initial_prompt=effective_prompt,  # Guides language detection
            no_speech_threshold=0.6,  # Higher = more strict (default 0.6)
            log_prob_threshold=-1.0,  # Filter low confidence (default -1.0)
            compression_ratio_threshold=2.4,  # Filter repetitive text (default 2.4)
            temperature=0.0,  # Deterministic output (no sampling)
        )
        
        # Collect all segments
        all_segments = []
        full_text_parts = []
        
        for segment in segments:
            # Log each segment for debugging
            avg_logprob = getattr(segment, 'avg_logprob', 0)
            no_speech_prob = getattr(segment, 'no_speech_prob', 0)
            print(f"   📝 [{segment.start:.1f}s-{segment.end:.1f}s] "
                  f"'{segment.text.strip()}' "
                  f"(logprob={avg_logprob:.2f}, no_speech={no_speech_prob:.2f})")
            
            # Filter out low-confidence segments
            if avg_logprob < -1.0 or no_speech_prob > 0.5:
                print(f"   ⚠️ Segment filtered (low confidence)")
                continue
                
            full_text_parts.append(segment.text)
            all_segments.append({
                "text": segment.text.strip(),
                "start": segment.start,
                "end": segment.end,
                "confidence": avg_logprob
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
        
        print("🎤 Listening... (speak now)")
        
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
        
        print("✅ Recording complete, transcribing...")
        
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
