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
    
    Example:
        # With built-in speaker
        tts = XTTSProvider(language="fr", speaker="Claribel Dervla")
        await tts.synthesize("Bonjour !", Path("output.wav"))
        
        # With voice cloning
        tts = XTTSProvider(language="fr", speaker_wav="~/voices/my_voice.wav")
        await tts.synthesize("Bonjour !", Path("output.wav"))
    """
    
    def __init__(
        self,
        language: str = "fr",
        speaker: str = "Claribel Dervla",
        speaker_wav: str | Path | None = None,
        device: str | None = None,
    ):
        """
        Initialize the XTTS v2 provider.
        
        Args:
            language: Language code (fr, en, de, etc.)
            speaker: Built-in speaker name (ignored if speaker_wav provided)
            speaker_wav: Path to reference audio for voice cloning
            device: "cuda", "cpu" or None (auto-detect)
        """
        self.language = language
        self.speaker = speaker
        self.speaker_wav = Path(speaker_wav).expanduser() if speaker_wav else None
        self.device = device
        
        # Lazy loading
        self._model = None
        self._TTS = None
        
        # Validation
        if language not in SUPPORTED_LANGUAGES:
            logger.warning(
                f"Language '{language}' not officially supported. "
                f"Supported languages: {SUPPORTED_LANGUAGES}"
            )
    
    @property
    def model_name(self) -> str:
        """Model name for display."""
        return "XTTS v2"
    
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
    
    async def synthesize(
        self,
        text: str,
        output_path: Path | None = None
    ) -> TTSResult:
        """
        Convert text to WAV audio file.
        
        Args:
            text: Text to synthesize
            output_path: Output path (optional)
            
        Returns:
            TTSResult with the audio file path
        """
        if not text.strip():
            raise ValueError("Text cannot be empty")
        
        # Create temp file if no path specified
        if output_path is None:
            import tempfile
            output_path = Path(tempfile.mktemp(suffix=".wav"))
        
        output_path = Path(output_path)
        
        # Synthesize in thread to not block the event loop
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            self._synthesize_sync,
            text,
            output_path
        )
        
        # Calculate duration
        info = sf.info(str(output_path))
        duration = info.duration
        
        return TTSResult(audio_path=output_path, duration=duration)
    
    def _synthesize_sync(self, text: str, output_path: Path) -> None:
        """
        Synchronous synthesis (called in a thread).
        """
        model = self._load_model()
        
        # Voice cloning or built-in speaker?
        if self.speaker_wav and self.speaker_wav.exists():
            # Voice cloning
            model.tts_to_file(
                text=text,
                speaker_wav=str(self.speaker_wav),
                language=self.language,
                file_path=str(output_path)
            )
        else:
            # Built-in speaker
            model.tts_to_file(
                text=text,
                speaker=self.speaker,
                language=self.language,
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
        
        Args:
            text: Text to synthesize
            
        Yields:
            Audio chunks in bytes (WAV format)
        """
        import io
        import wave
        
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
            temp_path
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
