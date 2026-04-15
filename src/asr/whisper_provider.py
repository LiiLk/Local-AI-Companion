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
import logging
import re
import numpy as np
from pathlib import Path
from typing import Optional, List, AsyncGenerator, Union
from dataclasses import dataclass

from .base import BaseASR, BaseRealtimeASR, ASRResult, ASRSegment

logger = logging.getLogger(__name__)

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
    
    # Beam search provides better accuracy at slight speed cost.
    # Default stays conservative for conversational latency.
    DEFAULT_BEAM_SIZE = 1
    MIN_AUTO_LANGUAGE_CONFIDENCE = 0.35
    REPETITIVE_HALLUCINATION_LANGUAGE_CONFIDENCE = 0.55
    
    def __init__(
        self,
        model_size: str = "base",
        device: str = "auto",
        compute_type: str = "float16",
        initial_prompt: Optional[str] = None,
        beam_size: int = DEFAULT_BEAM_SIZE,
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
        self.beam_size = max(1, int(beam_size or self.DEFAULT_BEAM_SIZE))
        
        # Lazy loading - model loaded on first use
        self._model = None

    @staticmethod
    def _looks_repetitive_hallucination(text: str) -> bool:
        cleaned = (text or "").strip()
        if len(cleaned) < 24:
            return False

        chars = [char for char in cleaned if not char.isspace()]
        if len(chars) < 24:
            return False

        unique_ratio = len(set(chars)) / max(len(chars), 1)
        if unique_ratio <= 0.08:
            return True

        repeated_char_run = re.search(r"(.)\1{11,}", cleaned)
        if repeated_char_run:
            return True

        two_char_run = re.search(r"(.{1,2})\1{8,}", cleaned)
        if two_char_run:
            return True

        # Repeated clause loops like "c'est le plus... c'est le plus..."
        clause_parts = [
            part.strip().lower()
            for part in re.split(r"(?:\.{2,}|[.!?…])", cleaned)
            if part and part.strip()
        ]
        if len(clause_parts) >= 4 and len(set(clause_parts)) == 1:
            return True

        tokens = re.findall(r"\b[\w'-]+\b", cleaned.lower())
        if len(tokens) >= 12:
            for n in (2, 3, 4):
                if len(tokens) < n * 5:
                    continue
                repeated_run = 1
                previous = tuple(tokens[:n])
                for index in range(n, len(tokens) - n + 1, n):
                    current = tuple(tokens[index:index + n])
                    if len(current) < n:
                        break
                    if current == previous:
                        repeated_run += 1
                        if repeated_run >= 5:
                            return True
                    else:
                        repeated_run = 1
                        previous = current

        return False
        
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
            logger.info("French model already cached: %s", local_path)
            return ctranslate2_path
        
        logger.info("Downloading French model from %s...", hf_repo)
        logger.info("This will only be done once.")
        
        # Create models directory
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        
        # Download only CTranslate2 files (smaller than full model)
        snapshot_download(
            repo_id=hf_repo,
            local_dir=str(local_path),
            allow_patterns="ctranslate2/*",
            local_dir_use_symlinks=False,
        )
        
        logger.info("French model downloaded: %s", local_path)
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
                logger.info("Loading French model (%s)...", self.model_size)
            else:
                logger.info("Loading Whisper (%s)...", self.model_size)
            
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
                    cpu_threads=8
                )
            except Exception as e:
                if "cudnn" in str(e).lower() or device == "cuda":
                    logger.warning("CUDA error, falling back to CPU: %s", e)
                    device = "cpu"
                    compute_type = "int8"
                    self._model = WhisperModel(
                        model_path,
                        device=device,
                        compute_type=compute_type,
                        cpu_threads=8
                    )
                else:
                    raise
            
            model_name = self.model_size if not is_french_model else f"{self.model_size} 🇫🇷"
            logger.info("%s loaded (device=%s, compute=%s)", model_name, device, compute_type)
            
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
            logger.info("Transcribing %s (%.1f KB)", audio_path.name, file_size / 1024)
            transcribe_input = str(audio_path)
        else:
            # Numpy array
            logger.info("Transcribing audio buffer (%s samples)", len(audio_input))
            transcribe_input = audio_input
        
        # Normalize language setting
        # Empty string or "auto" means auto-detection
        effective_language = language if language and language.lower() != "auto" else None
        
        # Use provided prompt or instance default
        effective_prompt = initial_prompt or self.initial_prompt
        
        # Transcribe with faster-whisper
        # NOTE: vad_filter disabled — Silero VAD already runs upstream in the pipeline.
        # word_timestamps disabled — not needed for conversational use, saves ~30-50% time.
        segments, info = model.transcribe(
            transcribe_input,
            language=effective_language,
            beam_size=self.beam_size,
            word_timestamps=False,
            vad_filter=False,
            # Anti-hallucination settings:
            condition_on_previous_text=False,
            initial_prompt=effective_prompt,
            no_speech_threshold=0.6,
            log_prob_threshold=-1.0,
            compression_ratio_threshold=2.4,
            temperature=0.0,
        )
        
        # Collect all segments
        all_segments = []
        full_text_parts = []
        no_speech_values: list[float] = []

        for segment in segments:
            # Log each segment for debugging
            avg_logprob = getattr(segment, 'avg_logprob', 0)
            no_speech_prob = getattr(segment, 'no_speech_prob', 0)
            no_speech_values.append(float(no_speech_prob))
            logger.info(
                "Segment [%.1fs-%.1fs] '%s' (logprob=%.2f, no_speech=%.2f)",
                segment.start,
                segment.end,
                segment.text.strip(),
                avg_logprob,
                no_speech_prob,
            )
            
            # Filter out low-confidence segments
            if avg_logprob < -1.0 or no_speech_prob > 0.5:
                logger.warning("Segment filtered (low confidence)")
                continue
                
            full_text_parts.append(segment.text)
            all_segments.append({
                "text": segment.text.strip(),
                "start": segment.start,
                "end": segment.end,
                "confidence": avg_logprob
            })
        
        full_text = " ".join(full_text_parts).strip()
        language_confidence = getattr(info, "language_probability", None)
        avg_no_speech = (
            sum(no_speech_values) / len(no_speech_values)
            if no_speech_values
            else 0.0
        )

        if (
            effective_language is None
            and full_text
            and self._looks_repetitive_hallucination(full_text)
            and (
                language_confidence is None
                or float(language_confidence) < self.REPETITIVE_HALLUCINATION_LANGUAGE_CONFIDENCE
                or avg_no_speech >= 0.35
            )
        ):
            logger.warning(
                "Rejecting likely Whisper hallucination: language=%s probability=%.2f avg_no_speech=%.2f text=%r",
                getattr(info, "language", None),
                float(language_confidence or 0.0),
                avg_no_speech,
                full_text[:120],
            )
            full_text = ""

        return ASRResult(
            text=full_text,
            language=info.language,
            confidence=language_confidence,
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
        info["beam_size"] = self.beam_size
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
        
        logger.info("Listening... (speak now)")
        
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
        
        logger.info("Recording complete, transcribing...")
        
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
