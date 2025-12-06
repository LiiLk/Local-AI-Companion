"""
Base module for TTS (Text-to-Speech).

This file defines the INTERFACE that all TTS providers must implement.
Same principle as for LLM: you can switch providers
(Edge TTS → Coqui → Piper) without modifying the main code.

TTS converts text to audio. It can:
1. Generate a complete audio file
2. Stream audio chunk by chunk (for real-time)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import AsyncGenerator
from pathlib import Path


@dataclass
class TTSResult:
    """
    Result of a voice synthesis.
    
    Attributes:
        audio_path: Path to generated audio file (if saved)
        audio_data: Raw audio data in bytes (if in memory)
        duration: Audio duration in seconds (if known)
    """
    audio_path: Path | None = None
    audio_data: bytes | None = None
    duration: float | None = None


@dataclass
class Voice:
    """
    Represents an available voice.
    
    Attributes:
        id: Unique voice identifier (e.g., "en-US-JennyNeural")
        name: Human-readable name (e.g., "Jenny")
        language: Language code (e.g., "en-US")
        gender: "Male" or "Female"
    """
    id: str
    name: str
    language: str
    gender: str


class BaseTTS(ABC):
    """
    Abstract base class for all TTS providers.
    
    Every TTS implementation must inherit from this class
    and implement the abstract methods.
    
    Example:
        class EdgeTTSProvider(BaseTTS):
            async def synthesize(self, text, voice):
                # Edge TTS specific implementation
                ...
    """
    
    @abstractmethod
    async def synthesize(
        self, 
        text: str, 
        output_path: Path | None = None
    ) -> TTSResult:
        """
        Convert text to audio.
        
        Args:
            text: The text to convert to speech
            output_path: Path to save the audio (optional)
                        If None, audio is returned in memory
        
        Returns:
            TTSResult containing the path or audio data
        
        Example:
            result = await tts.synthesize("Hello!", Path("output.mp3"))
            # result.audio_path contains the file path
        """
        pass
    
    @abstractmethod
    async def synthesize_stream(
        self, 
        text: str
    ) -> AsyncGenerator[bytes, None]:
        """
        Convert text to audio in streaming mode.
        
        Useful to start playing audio before the entire
        synthesis is complete (reduced latency).
        
        Args:
            text: The text to convert
            
        Yields:
            Audio chunks (bytes) progressively
            
        Example:
            async for chunk in tts.synthesize_stream("Hello!"):
                audio_player.feed(chunk)  # Play immediately
        """
        pass
    
    @abstractmethod
    async def list_voices(self, language: str | None = None) -> list[Voice]:
        """
        List available voices.
        
        Args:
            language: Filter by language (e.g., "fr-FR", "en-US")
                     If None, returns all voices
        
        Returns:
            List of available voices
            
        Example:
            voices = await tts.list_voices("en-US")
            for v in voices:
                print(f"{v.name} ({v.gender})")
        """
        pass
    
    @abstractmethod
    def set_voice(self, voice_id: str) -> None:
        """
        Change the voice used for synthesis.
        
        Args:
            voice_id: Voice identifier (e.g., "en-US-JennyNeural")
        """
        pass
    
    @abstractmethod
    def set_rate(self, rate: str) -> None:
        """
        Change speech speed.
        
        Args:
            rate: Speed modification (e.g., "+20%", "-10%", "+0%")
        """
        pass
    
    @abstractmethod
    def set_pitch(self, pitch: str) -> None:
        """
        Change voice pitch.
        
        Args:
            pitch: Pitch modification (e.g., "+10Hz", "-5Hz")
        """
        pass
