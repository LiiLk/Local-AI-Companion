"""
Base classes for ASR (Automatic Speech Recognition) providers.

This module defines the abstract interface that all ASR providers must implement.
Following the same pattern as our TTS module for consistency.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, List, AsyncGenerator
from pathlib import Path


@dataclass
class ASRResult:
    """
    Result from speech-to-text transcription.
    
    Attributes:
        text: The transcribed text
        language: Detected or specified language code (e.g., "fr", "en")
        confidence: Confidence score (0.0 to 1.0) if available
        duration: Duration of the audio in seconds
        segments: List of transcription segments with timestamps (optional)
    """
    text: str
    language: Optional[str] = None
    confidence: Optional[float] = None
    duration: Optional[float] = None
    segments: List[dict] = field(default_factory=list)
    
    def __str__(self) -> str:
        return self.text


@dataclass  
class ASRSegment:
    """
    A segment of transcribed speech with timing information.
    
    Useful for subtitles or word-level timestamps.
    
    Attributes:
        text: The transcribed text for this segment
        start: Start time in seconds
        end: End time in seconds
        confidence: Confidence score for this segment
    """
    text: str
    start: float
    end: float
    confidence: Optional[float] = None


class BaseASR(ABC):
    """
    Abstract base class for all ASR (Speech-to-Text) providers.
    
    All ASR implementations must inherit from this class and implement
    the abstract methods. This ensures consistent interface across
    different ASR backends (Whisper, Vosk, etc.)
    
    Example:
        >>> class MyASRProvider(BaseASR):
        ...     def transcribe(self, audio_path):
        ...         # Implementation here
        ...         return ASRResult(text="Hello world")
    """
    
    @abstractmethod
    def transcribe(
        self, 
        audio_path: str | Path,
        language: Optional[str] = None
    ) -> ASRResult:
        """
        Transcribe an audio file to text.
        
        Args:
            audio_path: Path to the audio file (WAV, MP3, etc.)
            language: Optional language code (e.g., "fr", "en"). 
                     If None, auto-detect.
        
        Returns:
            ASRResult containing the transcribed text and metadata
        """
        pass
    
    @abstractmethod
    def transcribe_stream(
        self,
        audio_path: str | Path,
        language: Optional[str] = None
    ) -> AsyncGenerator[ASRSegment, None]:
        """
        Transcribe audio and yield segments as they're processed.
        
        Useful for real-time transcription display.
        
        Args:
            audio_path: Path to the audio file
            language: Optional language code
            
        Yields:
            ASRSegment objects as they're transcribed
        """
        pass
    
    @abstractmethod
    def get_supported_languages(self) -> List[str]:
        """
        Get list of supported language codes.
        
        Returns:
            List of ISO 639-1 language codes (e.g., ["en", "fr", "es"])
        """
        pass
    
    @abstractmethod
    def get_model_info(self) -> dict:
        """
        Get information about the loaded model.
        
        Returns:
            Dict with model name, size, and other metadata
        """
        pass


class BaseRealtimeASR(BaseASR):
    """
    Extended base class for real-time ASR with microphone input.
    
    This adds methods for continuous listening and voice activity detection.
    """
    
    @abstractmethod
    def start_listening(self) -> None:
        """Start listening from the microphone."""
        pass
    
    @abstractmethod
    def stop_listening(self) -> None:
        """Stop listening from the microphone."""
        pass
    
    @abstractmethod
    def is_listening(self) -> bool:
        """Check if currently listening."""
        pass
    
    @abstractmethod
    async def listen_once(self, timeout: Optional[float] = None) -> ASRResult:
        """
        Listen for a single utterance and transcribe it.
        
        Args:
            timeout: Maximum time to wait for speech (seconds)
            
        Returns:
            ASRResult with the transcribed speech
        """
        pass
