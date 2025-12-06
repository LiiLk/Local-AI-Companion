"""
TTS implementation using Microsoft Edge TTS.

Edge TTS uses Microsoft Edge's speech synthesis API.
It's free, good quality, and supports 50+ languages.

Advantages:
- Free and unlimited
- Very natural voices (neural voices)
- No GPU required
- Native async support

Disadvantages:
- Requires internet connection
- No voice cloning
"""

import edge_tts
from pathlib import Path
from typing import AsyncGenerator

from .base import BaseTTS, TTSResult, Voice


# Recommended voices by language
RECOMMENDED_VOICES = {
    "fr-FR": "fr-FR-DeniseNeural",      # French, natural
    "fr-CA": "fr-CA-SylvieNeural",      # Quebec French
    "en-US": "en-US-JennyNeural",       # American English
    "en-GB": "en-GB-SoniaNeural",       # British English
    "ja-JP": "ja-JP-NanamiNeural",      # Japanese
    "es-ES": "es-ES-ElviraNeural",      # Spanish
    "de-DE": "de-DE-KatjaNeural",       # German
    "it-IT": "it-IT-ElsaNeural",        # Italian
    "zh-CN": "zh-CN-XiaoxiaoNeural",    # Chinese
    "ko-KR": "ko-KR-SunHiNeural",       # Korean
}


class EdgeTTSProvider(BaseTTS):
    """
    TTS provider using Microsoft Edge TTS.
    
    Attributes:
        voice: Current voice identifier
        rate: Speech speed (e.g., "+0%")
        pitch: Voice pitch (e.g., "+0Hz")
    
    Example:
        tts = EdgeTTSProvider(voice="en-US-JennyNeural")
        result = await tts.synthesize("Hello!", Path("hello.mp3"))
    """
    
    def __init__(
        self,
        voice: str = "fr-FR-DeniseNeural",
        rate: str = "+20%",
        pitch: str = "+0Hz"
    ):
        """
        Initialize the Edge TTS provider.
        
        Args:
            voice: Voice identifier (see RECOMMENDED_VOICES)
            rate: Speech speed (e.g., "+20%" for faster)
            pitch: Voice pitch (e.g., "+10Hz" for higher)
        """
        self.voice = voice
        self.rate = rate
        self.pitch = pitch
    
    async def synthesize(
        self,
        text: str,
        output_path: Path | None = None
    ) -> TTSResult:
        """
        Convert text to MP3 audio file.
        
        Args:
            text: Text to synthesize
            output_path: Output path (default: temp file)
            
        Returns:
            TTSResult with audio file path
        """
        # Create Edge TTS communicator
        communicate = edge_tts.Communicate(
            text=text,
            voice=self.voice,
            rate=self.rate,
            pitch=self.pitch
        )
        
        # Define output path
        if output_path is None:
            output_path = Path(f"/tmp/tts_output_{hash(text)}.mp3")
        
        # Generate and save audio
        await communicate.save(str(output_path))
        
        return TTSResult(audio_path=output_path)
    
    async def synthesize_stream(
        self,
        text: str
    ) -> AsyncGenerator[bytes, None]:
        """
        Generate audio in streaming mode (chunk by chunk).
        
        Allows starting playback before the entire
        synthesis is complete.
        
        Args:
            text: Text to synthesize
            
        Yields:
            Audio chunks (bytes) in MP3 format
        """
        communicate = edge_tts.Communicate(
            text=text,
            voice=self.voice,
            rate=self.rate,
            pitch=self.pitch
        )
        
        # Edge TTS sends audio chunks + metadata
        async for chunk in communicate.stream():
            # We only keep audio chunks (not metadata)
            if chunk["type"] == "audio":
                yield chunk["data"]
    
    async def synthesize_to_bytes(self, text: str) -> bytes:
        """
        Helper method: synthesize and return bytes directly.
        
        Useful when you want audio in memory without saving.
        
        Args:
            text: Text to synthesize
            
        Returns:
            Complete audio data in bytes (MP3)
        """
        audio_chunks = []
        async for chunk in self.synthesize_stream(text):
            audio_chunks.append(chunk)
        return b"".join(audio_chunks)
    
    async def list_voices(self, language: str | None = None) -> list[Voice]:
        """
        List all available voices.
        
        Args:
            language: Filter by language (e.g., "fr-FR", "en")
            
        Returns:
            List of available voices
        """
        # Get all Edge TTS voices
        voices_data = await edge_tts.list_voices()
        
        voices = []
        for v in voices_data:
            # Filter by language if specified
            if language:
                # Supports "fr" or "fr-FR"
                if not v["Locale"].startswith(language):
                    continue
            
            voices.append(Voice(
                id=v["ShortName"],
                name=v["DisplayName"],
                language=v["Locale"],
                gender=v["Gender"]
            ))
        
        return voices
    
    def set_voice(self, voice_id: str) -> None:
        """Change the voice used."""
        self.voice = voice_id
    
    def set_rate(self, rate: str) -> None:
        """Change speech speed."""
        self.rate = rate
    
    def set_pitch(self, pitch: str) -> None:
        """Change voice pitch."""
        self.pitch = pitch
    
    @staticmethod
    def get_recommended_voice(language: str) -> str:
        """
        Return a recommended voice for a language.
        
        Args:
            language: Language code (e.g., "fr-FR", "en-US")
            
        Returns:
            Recommended voice identifier
        """
        return RECOMMENDED_VOICES.get(language, "en-US-JennyNeural")
