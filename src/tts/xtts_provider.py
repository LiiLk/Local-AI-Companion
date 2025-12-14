"""
XTTS v2 Provider - Multilingual Voice Cloning TTS by Coqui.

XTTS v2 is a state-of-the-art TTS model with:
- 17 supported languages (including French!)
- Voice cloning with only 6 seconds of audio
- Streaming with latency < 200ms
- Natural and expressive quality

Specs:
- ~2.8GB VRAM on GPU
- 1.9GB model (auto-download)
- Sample rate: 24kHz
"""

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncGenerator

import numpy as np
import soundfile as sf

from .base import BaseTTS, TTSResult
import re

# Emotion map (using March 7th's optimized voice - 22050Hz, normalized)
EMOTION_REFS = {
    "neutral": "resources/voices/f5_refs/march7th_optimized.wav",
    "happy": "resources/voices/f5_refs/march7th_optimized.wav",
    "sad": "resources/voices/f5_refs/march7th_optimized.wav",
    "angry": "resources/voices/f5_refs/march7th_optimized.wav"
}

logger = logging.getLogger(__name__)


# Languages supported by XTTS v2
SUPPORTED_LANGUAGES = [
    "en", "es", "fr", "de", "it", "pt", "pl", "tr", 
    "ru", "nl", "cs", "ar", "zh-cn", "ja", "hu", "ko", "hi"
]

# Built-in speakers (some examples)
DEFAULT_SPEAKERS = [
    "Claribel Dervla",      # Clear female voice
    "Daisy Studious",       # Studious female voice
    "Gracie Wise",          # Wise female voice
    "Tammie Ema",           # Energetic female voice
    "Alison Dietlinde",     # Soft female voice
    "Ana Florence",         # Natural female voice
    "Annmarie Nele",        # Expressive female voice
    "Asya Anara",           # Mysterious female voice
    "Brenda Stern",         # Serious female voice
    "Gitta Nikolina",       # European female voice
    "Henriette Usha",       # Warm female voice
    "Sofia Hellen",         # Elegant female voice
    "Tammy Grit",           # Determined female voice
    "Tanja Adelina",        # Modern female voice
    "Vjollca Johnnie",      # Unique female voice
    "Andrew Chipper",       # Cheerful male voice
    "Badr Odhiambo",        # Deep male voice
    "Dionisio Schuyler",    # Classic male voice
    "Royston Min",          # Asian male voice
    "Viktor Eka",           # European male voice
    "Abrahan Mack",         # American male voice
    "Adde Michal",          # Scandinavian male voice
    "Baldur Sansen",        # Nordic male voice
    "Craig Gutsy",          # Energetic male voice
    "Damien Black",         # Dark male voice
    "Gilberto Mathias",     # Latin male voice
    "Ilkin Urbano",         # Mediterranean male voice
    "Kazuhiko Atallah",     # Japanese male voice
    "Ludvig Milivoj",       # Slavic male voice
    "Suad Qasim",           # Arabic male voice
    "Torcull Diarmuid",     # Celtic male voice
    "Viktor Menelaos",      # Greek male voice
    "Zacharie Aimilios",    # French male voice
    "Nova Hogarth",         # Non-binary voice
    "Maja Ruoho",           # Finnish female voice
    "Uta Obando",           # German female voice
    "Lidiya Szekeres",      # Hungarian female voice
    "Chandra MacFarland",   # Indian female voice
    "Szofi Granger",        # British female voice
    "Camilla Holmström",    # Swedish female voice
    "Lilya Stainthorpe",    # Russian female voice
    "Zofija Kendrick",      # Polish female voice
    "Narelle Moon",         # Australian female voice
    "Barbora MacLean",      # Scottish female voice
    "Alexandra Hisakawa",   # Japanese female voice
    "Alma María",           # Spanish female voice
    "Rosemary Okafor",      # African female voice
    "Ige Behringer",        # German female voice
    "Filip Traverse",       # French male voice
    "Damjan Chapman",       # British male voice
    "Wulf Carlevaro",       # Italian male voice
    "Aaron Dreschner",      # American male voice
    "Kumar Dahl",           # Indian male voice
    "Eugenio Matarese",     # Italian male voice
    "Ferran Sansen",        # Catalan male voice
    "Xavier Hayasaka",      # Japanese male voice
    "Luis Moray",           # Spanish male voice
    "Marcos Rudaski",       # Polish male voice
]


@dataclass
class XTTSConfig:
    """Configuration for XTTS v2."""
    
    # Default language
    language: str = "fr"
    
    # Built-in speaker (if no voice cloning)
    speaker: str = "Claribel Dervla"
    
    # Voice cloning: reference audio
    speaker_wav: str | None = None
    
    # Device: "cuda" or "cpu"
    device: str | None = None  # None = auto-detect


class XTTSProvider(BaseTTS):
    """
    TTS Provider using XTTS v2 from Coqui.
    
    XTTS v2 offers:
    - 17 languages including French with native accent
    - Voice cloning with 6 seconds of audio
    - 58 built-in speakers
    - ~2.8GB VRAM, fast generation (~1.7s per sentence)
    - Automatic language detection support
    - Speaker embedding caching for faster synthesis
    
    Example:
        # With built-in speaker
        tts = XTTSProvider(language="fr", speaker="Claribel Dervla")
        await tts.synthesize("Bonjour !", Path("output.wav"))
        
        # With voice cloning
        tts = XTTSProvider(language="fr", speaker_wav="~/voices/my_voice.wav")
        await tts.synthesize("Bonjour !", Path("output.wav"))
        
        # With auto-detection (language=None)
        tts = XTTSProvider(language=None, speaker="Claribel Dervla")
        await tts.synthesize("Bonjour !", Path("output.wav"))  # Auto-detects French
    """
    
    def __init__(
        self,
        language: str | None = "en",
        speaker: str = "Claribel Dervla",
        speaker_wav: str | Path | None = None,
        device: str | None = None,
        auto_detect_language: bool = False,
    ):
        """
        Initialize the XTTS v2 provider.
        
        Args:
            language: Language code (fr, en, de, etc.) or None for auto-detect
            speaker: Built-in speaker name (ignored if speaker_wav provided)
            speaker_wav: Path to reference audio for voice cloning
            device: "cuda", "cpu" or None (auto-detect)
            auto_detect_language: Enable automatic language detection
        """
        self.language = language
        self.speaker = speaker
        self.speaker_wav = Path(speaker_wav).expanduser() if speaker_wav else None
        self.device = device
        self.auto_detect_language = auto_detect_language or (language is None)
        
        # Emotion state
        self.current_emotion = "neutral"
        self.ref_dir = Path("resources/voices/f5_refs")  # Reuse existing refs
        
        # Lazy loading
        self._model = None
        self._TTS = None
        
        # Cache for speaker embeddings (significant speedup!)
        self._gpt_cond_latent = None
        self._speaker_embedding = None
        self._cached_speaker = None  # Track which speaker is cached
        
        # Validation
        if language is not None and language not in SUPPORTED_LANGUAGES:
            logger.warning(
                f"Language '{language}' not officially supported. "
                f"Supported languages: {SUPPORTED_LANGUAGES}"
            )
    
    @property
    def model_name(self) -> str:
        """Model name for display."""
        return "XTTS v2"
    
    def set_language(self, language: str) -> None:
        """
        Change the synthesis language.
        
        Args:
            language: Language code (fr, en, de, etc.)
        """
        if language not in SUPPORTED_LANGUAGES:
            logger.warning(
                f"Language '{language}' not officially supported. "
                f"Supported languages: {SUPPORTED_LANGUAGES}"
            )
        self.language = language
        logger.debug(f"XTTS language set to: {language}")
    
    def _get_language_for_text(self, text: str) -> str:
        """
        Get the language to use for synthesis.
        
        If auto_detect_language is enabled, detects the language.
        Otherwise returns the configured language.
        
        Args:
            text: Text to analyze
            
        Returns:
            Language code (e.g., "fr", "en")
        """
        if self.auto_detect_language:
            from ..utils.language_detection import detect_language, LanguageCode
            
            detected = detect_language(text)
            logger.info(f"🌐 Auto-detected language: {detected} for text: '{text[:50]}...'")
            return str(detected)
        
        return self.language or "en"
    
    def _detect_emotion(self, text: str) -> tuple[str, str]:
        """
        Detect emotion tag in text and return (emotion, cleaned_text).
        Tags: [happy], [sad], [angry], [neutral]
        """
        text_lower = text.lower()
        tag_map = {
            "[happy]": "happy", "[joie]": "happy", "[joyeux]": "happy",
            "[sad]": "sad", "[triste]": "sad",
            "[angry]": "angry", "[colere]": "angry", "[fache]": "angry",
            "[neutral]": "neutral", "[neutre]": "neutral"
        }
        
        detected_emotion = self.current_emotion
        for tag, emotion in tag_map.items():
            if tag in text_lower:
                detected_emotion = emotion
                text = re.sub(re.escape(tag), "", text, flags=re.IGNORECASE)
        
        return detected_emotion, text.strip()
    
    def _load_model(self):
        """
        Load the XTTS v2 model (lazy loading).
        
        The model is automatically downloaded from HuggingFace
        on first call (~1.9GB).
        """
        if self._model is not None:
            return self._model
        
        logger.info(f"🔄 Loading {self.model_name}...")
        
        from TTS.api import TTS
        import torch
        
        # Auto-detect device
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self._model = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(self.device)
        
        logger.info(f"✅ {self.model_name} loaded on {self.device}!")
        return self._model
    
    def _get_speaker_embedding(self):
        """
        Get cached speaker embedding or compute it.
        
        Caching the speaker embedding provides significant speedup
        for subsequent synthesis calls (saves ~0.5-1s per call).
        
        Returns:
            Tuple of (gpt_cond_latent, speaker_embedding)
        """
        # Check if we need to recompute (speaker changed)
        current_speaker = str(self.speaker_wav) if self.speaker_wav else self.speaker
        
        if self._gpt_cond_latent is not None and self._cached_speaker == current_speaker:
            return self._gpt_cond_latent, self._speaker_embedding
        
        model = self._load_model()
        
        # Access the underlying XTTS model
        xtts_model = model.synthesizer.tts_model
        
        if self.speaker_wav and self.speaker_wav.exists():
            # Voice cloning - compute embedding from audio
            # Optimized parameters for better voice similarity:
            # - gpt_cond_len=6: Shorter context for more focused voice capture
            # - gpt_cond_chunk_len=6: Match the conditioning length
            # - max_ref_length=30: Use up to 30s of reference (our file is ~12s)
            logger.info(f"🔄 Computing speaker embedding from: {self.speaker_wav}")
            gpt_cond_latent, speaker_embedding = xtts_model.get_conditioning_latents(
                audio_path=str(self.speaker_wav),
                gpt_cond_len=6,
                gpt_cond_chunk_len=6,
                max_ref_length=30
            )
        else:
            # Built-in speaker - get from speaker manager
            logger.info(f"🔄 Loading speaker embedding for: {self.speaker}")
            # For built-in speakers, we use the high-level API which handles this
            # internally, but we can't cache it easily. Fall back to no cache.
            self._cached_speaker = current_speaker
            return None, None
        
        # Cache the embeddings
        self._gpt_cond_latent = gpt_cond_latent
        self._speaker_embedding = speaker_embedding
        self._cached_speaker = current_speaker
        
        logger.info("✅ Speaker embedding cached!")
        return gpt_cond_latent, speaker_embedding
    
    async def synthesize(
        self,
        text: str,
        output_path: Path | None = None
    ) -> TTSResult:
        """
        Convert text to WAV audio file.
        
        If auto_detect_language is enabled, the language is automatically
        detected from the text.
        
        Args:
        Args:
            text: Text to synthesize
            output_path: Output path (optional)
            
        Returns:
            TTSResult with the audio file path
        """
        # 1. Detect and separate emotion tags
        emotion, clean_text = self._detect_emotion(text)
        if not clean_text:
            return TTSResult(audio_data=b"")

        # 2. Update speaker_wav if emotion detected (and ref exists)
        # Only override if we are in "cloning mode" or forced emotion mode
        # If user didn't specify a base speaker_wav, we normally use built-in.
        # But for emotion, we must use cloning.
        
        # Resolve reference audio for this emotion
        ref_path = Path(EMOTION_REFS.get(emotion, EMOTION_REFS["neutral"]))
        if ref_path.exists():
            # Temporarily switch speaker_wav for this synthesis
            original_wav = self.speaker_wav
            self.speaker_wav = ref_path
            # Log change
            if emotion != "neutral":
                logger.info(f"🎭 XTTS Switching emotion to: {emotion} (using {ref_path})")
        else:
            # Fallback
            original_wav = self.speaker_wav
        
        try:
            # Create temp file if no path specified
            if output_path is None:
                import tempfile
                output_path = Path(tempfile.mktemp(suffix=".wav"))
            
            output_path = Path(output_path)
            
            # Get language (auto-detect if enabled)
            language = self._get_language_for_text(clean_text)
            
            # Synthesize in thread
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                self._synthesize_sync,
                clean_text,
                output_path,
                language
            )
            
            # Calculate duration
            info = sf.info(str(output_path))
            duration = info.duration
            
            return TTSResult(audio_path=output_path, duration=duration)
            
        finally:
            # Restore original speaker_wav
            if 'original_wav' in locals():
                self.speaker_wav = original_wav
    
    def _synthesize_sync(self, text: str, output_path: Path, language: str | None = None) -> None:
        """
        Synchronous synthesis (called in a thread).
        
        Uses cached speaker embeddings when available for faster synthesis.
        
        Args:
            text: Text to synthesize
            output_path: Output file path
            language: Language code (if None, uses self.language)
        """
        model = self._load_model()
        lang = language or self.language or "en"
        
        # Try to use cached embeddings for voice cloning
        if self.speaker_wav and self.speaker_wav.exists():
            gpt_cond_latent, speaker_embedding = self._get_speaker_embedding()
            
            if gpt_cond_latent is not None:
                # Use low-level API with cached embeddings (faster!)
                xtts_model = model.synthesizer.tts_model
                
                # Optimized inference parameters for natural voice cloning:
                # - temperature=0.75: Slightly higher for more natural variation
                # - repetition_penalty=5.0: Less aggressive, more natural flow
                # - top_p=0.8: Allow more variation for expressiveness
                # - speed=1.0: Normal speed (can adjust 0.8-1.2)
                out = xtts_model.inference(
                    text=text,
                    language=lang,
                    gpt_cond_latent=gpt_cond_latent,
                    speaker_embedding=speaker_embedding,
                    temperature=0.75,
                    length_penalty=1.0,
                    repetition_penalty=5.0,
                    top_k=50,
                    top_p=0.8,
                    speed=1.0,
                    enable_text_splitting=True
                )
                
                # Save audio
                import torchaudio
                import torch
                import numpy as np
                
                wav_data = out["wav"]
                
                # Convert to tensor if it's numpy
                if isinstance(wav_data, np.ndarray):
                    wav_tensor = torch.from_numpy(wav_data)
                else:
                    wav_tensor = wav_data.cpu()

                if wav_tensor.dim() == 1:
                    wav_tensor = wav_tensor.unsqueeze(0)
                
                torchaudio.save(
                    str(output_path),
                    wav_tensor,
                    24000  # XTTS sample rate
                )
                return
        
        # Fall back to high-level API for built-in speakers
        if self.speaker_wav and self.speaker_wav.exists():
            model.tts_to_file(
                text=text,
                speaker_wav=str(self.speaker_wav),
                language=lang,
                file_path=str(output_path)
            )
        else:
            # Built-in speaker
            model.tts_to_file(
                text=text,
                speaker=self.speaker,
                language=lang,
                file_path=str(output_path)
            )
    
    async def synthesize_stream(
        self,
        text: str
    ) -> AsyncGenerator[bytes, None]:
        """
        Generate audio in streaming mode.
        
        XTTS v2 supports native streaming with latency < 200ms.
        This implementation uses XTTS internal streaming.
        If auto_detect_language is enabled, language is auto-detected.
        
        Args:
            text: Text to synthesize
            
        Yields:
            Audio chunks in bytes (WAV format)
        """
        import io
        import wave
        
        # Get language (auto-detect if enabled)
        language = self._get_language_for_text(text)
        
        # For streaming, we generate full audio then split it
        # A more advanced implementation would use model.inference_stream()
        loop = asyncio.get_event_loop()
        
        # Generate full audio
        import tempfile
        temp_path = Path(tempfile.mktemp(suffix=".wav"))
        
        await loop.run_in_executor(
            None,
            self._synthesize_sync,
            text,
            temp_path,
            language
        )
        
        # Read and stream by chunks
        chunk_size = 4096  # ~85ms at 24kHz
        
        with open(temp_path, "rb") as f:
            # Send WAV header first
            header = f.read(44)
            yield header
            
            # Then audio data by chunks
            while chunk := f.read(chunk_size):
                yield chunk
        
        # Cleanup
        temp_path.unlink(missing_ok=True)
    
    async def synthesize_to_bytes(self, text: str) -> bytes:
        """
        Convert text directly to audio bytes.
        
        Args:
            text: Text to synthesize
            
        Returns:
            Audio in bytes (WAV format)
        """
        import tempfile
        
        temp_path = Path(tempfile.mktemp(suffix=".wav"))
        
        try:
            await self.synthesize(text, temp_path)
            
            with open(temp_path, "rb") as f:
                return f.read()
        finally:
            temp_path.unlink(missing_ok=True)
    
    async def list_voices(self, language: str | None = None) -> list:
        """
        List available voices (speakers).
        
        For XTTS, "voices" are the built-in speakers.
        The language parameter is ignored as all speakers
        can speak all languages.
        
        Args:
            language: Ignored for XTTS (all speakers are multilingual)
            
        Returns:
            List of Voice objects
        """
        from .base import Voice
        
        speakers = self.list_speakers()
        voices = []
        
        for speaker in speakers:
            # XTTS speakers are all multilingual
            voices.append(Voice(
                id=speaker,
                name=speaker,
                language="multilingual",
                gender="Unknown"  # XTTS doesn't specify gender
            ))
        
        return voices
    
    def set_voice(self, voice_id: str) -> None:
        """
        Change the speaker used.
        
        Args:
            voice_id: Speaker name (e.g., "Claribel Dervla")
        """
        self.speaker = voice_id
    
    def set_rate(self, rate: str) -> None:
        """
        Not supported by XTTS v2.
        
        XTTS generates audio at natural speed.
        To modify speed, use audio post-processing.
        """
        pass  # Not supported
    
    def set_pitch(self, pitch: str) -> None:
        """
        Not supported by XTTS v2.
        
        XTTS generates audio with the speaker's natural pitch.
        To modify pitch, use audio post-processing.
        """
        pass  # Not supported
    
    def list_speakers(self) -> list[str]:
        """
        List available built-in speakers.
        
        Returns:
            List of speaker names
        """
        model = self._load_model()
        return model.speakers
    
    @staticmethod
    def list_languages() -> list[str]:
        """
        List supported languages.
        
        Returns:
            List of language codes
        """
        return SUPPORTED_LANGUAGES.copy()
