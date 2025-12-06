"""
TTS implementation using Kokoro - High quality local TTS.

Kokoro is an open-source TTS model with 82M parameters that offers
quality comparable to ElevenLabs while being 100% local.

Advantages:
- 100% local (no internet required)
- Exceptional and natural voice quality
- Realistic speech rate
- Multilingual support (FR, EN, JA, ZH, etc.)
- Lightweight (82M parameters, ~300MB)
- Works on CPU (GPU optional for more speed)

Disadvantages:
- First load is slower (model download)
- Consumes more RAM than Edge TTS
"""

import io
import tempfile
from pathlib import Path
from typing import AsyncGenerator
import asyncio

import soundfile as sf
import numpy as np

from .base import BaseTTS, TTSResult, Voice


# Kokoro language code mapping
# Kokoro uses single-letter codes for languages
LANG_CODES = {
    "en-US": "a",  # American English
    "en-GB": "b",  # British English
    "es-ES": "e",  # Spanish
    "fr-FR": "f",  # French
    "hi-IN": "h",  # Hindi
    "it-IT": "i",  # Italian
    "ja-JP": "j",  # Japanese
    "pt-BR": "p",  # Brazilian Portuguese
    "zh-CN": "z",  # Mandarin Chinese
}

# Recommended voices by language
# Format: voice_id (used by Kokoro)
RECOMMENDED_VOICES = {
    "fr-FR": "ff_siwis",      # French female voice (SIWIS dataset)
    "en-US": "af_heart",      # American English female voice
    "en-GB": "bf_emma",       # British English female voice
    "ja-JP": "jf_alpha",      # Japanese female voice
    "zh-CN": "zf_xiaobei",    # Chinese female voice
    "es-ES": "ef_dora",       # Spanish female voice
    "it-IT": "if_sara",       # Italian female voice
}

# Complete list of available voices
AVAILABLE_VOICES = [
    # French
    Voice(id="ff_siwis", name="Siwis (French Female)", language="fr-FR", gender="Female"),
    
    # American English
    Voice(id="af_heart", name="Heart (US Female)", language="en-US", gender="Female"),
    Voice(id="af_bella", name="Bella (US Female)", language="en-US", gender="Female"),
    Voice(id="af_nicole", name="Nicole (US Female)", language="en-US", gender="Female"),
    Voice(id="af_sarah", name="Sarah (US Female)", language="en-US", gender="Female"),
    Voice(id="af_sky", name="Sky (US Female)", language="en-US", gender="Female"),
    Voice(id="am_adam", name="Adam (US Male)", language="en-US", gender="Male"),
    Voice(id="am_michael", name="Michael (US Male)", language="en-US", gender="Male"),
    
    # British English
    Voice(id="bf_emma", name="Emma (UK Female)", language="en-GB", gender="Female"),
    Voice(id="bf_isabella", name="Isabella (UK Female)", language="en-GB", gender="Female"),
    Voice(id="bm_george", name="George (UK Male)", language="en-GB", gender="Male"),
    Voice(id="bm_lewis", name="Lewis (UK Male)", language="en-GB", gender="Male"),
    
    # Japanese
    Voice(id="jf_alpha", name="Alpha (JP Female)", language="ja-JP", gender="Female"),
    Voice(id="jf_gongitsune", name="Gongitsune (JP Female)", language="ja-JP", gender="Female"),
    Voice(id="jm_kumo", name="Kumo (JP Male)", language="ja-JP", gender="Male"),
    
    # Chinese
    Voice(id="zf_xiaobei", name="Xiaobei (CN Female)", language="zh-CN", gender="Female"),
    Voice(id="zf_xiaoni", name="Xiaoni (CN Female)", language="zh-CN", gender="Female"),
    Voice(id="zm_yunjian", name="Yunjian (CN Male)", language="zh-CN", gender="Male"),
]


class KokoroProvider(BaseTTS):
    """
    TTS provider using Kokoro - High quality local model.
    
    Kokoro generates 24kHz audio with natural speech rate.
    The model is loaded on first call (lazy loading).
    
    Attributes:
        voice: Kokoro voice identifier
        lang_code: Language code for phonetization
        speed: Speech speed (1.0 = normal)
        _pipeline: Kokoro pipeline (loaded on demand)
    
    Example:
        tts = KokoroProvider(voice="ff_siwis")
        result = await tts.synthesize("Hello world!")
    """
    
    # Sample rate de Kokoro (fixe)
    SAMPLE_RATE = 24000
    
    def __init__(
        self,
        voice: str = "ff_siwis",
        lang_code: str | None = None,
        speed: float = 1.0
    ):
        """
        Initialize the Kokoro provider.
        
        Args:
            voice: Voice identifier (e.g., "ff_siwis", "af_heart")
                   The prefix indicates the language (ff=French, af=US English, etc.)
            lang_code: Explicit language code (auto-detected from voice if None)
            speed: Speech speed (0.5 to 2.0, 1.0 = normal)
        """
        self.voice = voice
        self.speed = speed
        self._pipeline = None
        
        # Auto-detect language code from voice prefix
        # ff_siwis -> f (French), af_heart -> a (US English)
        if lang_code:
            self.lang_code = lang_code
        else:
            voice_prefix = voice[:2] if len(voice) >= 2 else "a"
            # Voice prefix to Kokoro language code mapping
            prefix_to_lang = {
                "ff": "f",  # French female
                "fm": "f",  # French male
                "af": "a",  # American female
                "am": "a",  # American male
                "bf": "b",  # British female
                "bm": "b",  # British male
                "jf": "j",  # Japanese female
                "jm": "j",  # Japanese male
                "zf": "z",  # Chinese female
                "zm": "z",  # Chinese male
                "ef": "e",  # Spanish female
                "em": "e",  # Spanish male
                "if": "i",  # Italian female
                "im": "i",  # Italian male
            }
            self.lang_code = prefix_to_lang.get(voice_prefix, "a")
    
    def _load_pipeline(self):
        """
        Load the Kokoro pipeline (lazy loading).
        
        The model is automatically downloaded on first call
        from HuggingFace (~300MB).
        """
        if self._pipeline is None:
            from kokoro import KPipeline
            
            print(f"🔄 Loading Kokoro (lang={self.lang_code})...")
            self._pipeline = KPipeline(
                lang_code=self.lang_code,
                repo_id="hexgrad/Kokoro-82M"  # Explicit to suppress warning
            )
            print("✅ Kokoro loaded!")
        
        return self._pipeline
    
    async def synthesize(
        self,
        text: str,
        output_path: Path | None = None
    ) -> TTSResult:
        """
        Convert text to WAV audio file.
        
        Args:
            text: Text to synthesize
            output_path: Output path (default: temp file)
            
        Returns:
            TTSResult with audio file path
        """
        # Kokoro is synchronous, run in a thread
        loop = asyncio.get_event_loop()
        audio_data = await loop.run_in_executor(
            None, 
            self._synthesize_sync, 
            text
        )
        
        # Define output path
        if output_path is None:
            # Create temporary WAV file
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            output_path = Path(tmp.name)
            tmp.close()
        
        # Save as WAV
        sf.write(str(output_path), audio_data, self.SAMPLE_RATE)
        
        return TTSResult(audio_path=output_path)
    
    def _synthesize_sync(self, text: str) -> np.ndarray:
        """
        Synchronous synthesis (called in a thread).
        
        Returns:
            Numpy array containing audio data
        """
        pipeline = self._load_pipeline()
        
        # Generate audio
        # Generator returns (graphemes, phonemes, audio) for each segment
        audio_segments = []
        
        for _, _, audio in pipeline(text, voice=self.voice, speed=self.speed):
            audio_segments.append(audio)
        
        # Concatenate all segments
        if audio_segments:
            return np.concatenate(audio_segments)
        else:
            return np.array([], dtype=np.float32)
    
    async def synthesize_stream(
        self,
        text: str
    ) -> AsyncGenerator[bytes, None]:
        """
        Generate audio in streaming mode (segment by segment).
        
        Kokoro naturally generates by segments (sentences),
        allowing playback to start before completion.
        
        Args:
            text: Text to synthesize
            
        Yields:
            Audio chunks in bytes (WAV PCM format)
        """
        loop = asyncio.get_event_loop()
        
        # Synchronous generator to async
        def generate_segments():
            pipeline = self._load_pipeline()
            for _, _, audio in pipeline(text, voice=self.voice, speed=self.speed):
                yield audio
        
        # Convert to async
        for audio in await loop.run_in_executor(None, list, generate_segments()):
            # Convert numpy array to WAV bytes
            buffer = io.BytesIO()
            sf.write(buffer, audio, self.SAMPLE_RATE, format='WAV')
            buffer.seek(0)
            yield buffer.read()
    
    async def synthesize_to_bytes(self, text: str) -> bytes:
        """
        Synthesize and return audio bytes directly.
        
        Args:
            text: Text to synthesize
            
        Returns:
            Audio data in bytes (WAV format)
        """
        loop = asyncio.get_event_loop()
        audio_data = await loop.run_in_executor(
            None,
            self._synthesize_sync,
            text
        )
        
        # Convert to WAV bytes
        buffer = io.BytesIO()
        sf.write(buffer, audio_data, self.SAMPLE_RATE, format='WAV')
        buffer.seek(0)
        return buffer.read()
    
    async def list_voices(self, language: str | None = None) -> list[Voice]:
        """
        List available voices.
        
        Args:
            language: Filter by language (e.g., "fr-FR", "en")
            
        Returns:
            List of available voices
        """
        voices = AVAILABLE_VOICES.copy()
        
        if language:
            # Filter by language
            voices = [v for v in voices if v.language.startswith(language)]
        
        return voices
    
    def set_voice(self, voice_id: str) -> None:
        """
        Change the voice used.
        
        Note: If the language changes, the pipeline will be reloaded.
        """
        old_lang = self.lang_code
        self.voice = voice_id
        
        # Recalculate language code
        voice_prefix = voice_id[:2] if len(voice_id) >= 2 else "a"
        prefix_to_lang = {
            "ff": "f", "fm": "f",
            "af": "a", "am": "a",
            "bf": "b", "bm": "b",
            "jf": "j", "jm": "j",
            "zf": "z", "zm": "z",
            "ef": "e", "em": "e",
            "if": "i", "im": "i",
        }
        self.lang_code = prefix_to_lang.get(voice_prefix, "a")
        
        # If language changed, force pipeline reload
        if old_lang != self.lang_code:
            self._pipeline = None
    
    def set_speed(self, speed: float) -> None:
        """Change speech speed (0.5 to 2.0)."""
        self.speed = max(0.5, min(2.0, speed))
    
    def set_rate(self, rate: str) -> None:
        """
        Change speech rate (interface compatibility).
        
        Converts Edge TTS format ("+20%") to float for Kokoro.
        
        Args:
            rate: Speed modification (e.g., "+20%", "-10%")
        """
        # Convert "+20%" -> 1.2, "-10%" -> 0.9
        try:
            rate_clean = rate.replace("%", "").replace("+", "")
            rate_value = float(rate_clean) / 100
            self.speed = 1.0 + rate_value
            self.speed = max(0.5, min(2.0, self.speed))
        except ValueError:
            self.speed = 1.0
    
    def set_pitch(self, pitch: str) -> None:
        """
        Change voice pitch (not supported by Kokoro).
        
        This method exists for interface compatibility,
        but Kokoro does not support pitch changes.
        
        Args:
            pitch: Ignored (Kokoro does not support pitch)
        """
        # Kokoro does not support pitch, silently ignore
        pass
    
    @staticmethod
    def get_recommended_voice(language: str) -> str:
        """
        Returns a recommended voice for a language.
        
        Args:
            language: Language code (e.g., "fr-FR", "en-US")
            
        Returns:
            Recommended voice identifier
        """
        return RECOMMENDED_VOICES.get(language, "af_heart")
