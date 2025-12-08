"""
Conversation Pipeline - Orchestrates ASR → LLM → TTS flow.

This module handles the full conversation flow with callbacks
for Live2D integration (expressions, lip-sync).

Features:
- Speech-to-text with ASR
- LLM response generation (streaming)
- Text-to-speech with volume analysis for lip-sync
- Emotion detection from LLM output
- Callbacks for UI/Live2D updates
"""

import asyncio
import base64
import logging
import re
import struct
import tempfile
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import AsyncGenerator, Callable, Optional, Any

import numpy as np

from src.llm.base import BaseLLM, Message
from src.tts.base import BaseTTS
from src.asr.base import BaseASR

logger = logging.getLogger(__name__)


@dataclass
class ConversationConfig:
    """Configuration for conversation pipeline."""
    # Character
    character_name: str = "March 7th"
    system_prompt: str = "You are a helpful AI assistant."
    
    # TTS
    tts_sample_rate: int = 24000
    lip_sync_chunk_ms: int = 50  # Chunk size for volume analysis
    
    # Behavior
    stream_tts: bool = True  # Synthesize sentence-by-sentence
    auto_detect_language: bool = True


@dataclass
class AudioPayload:
    """Payload containing audio data with lip-sync info."""
    audio_bytes: bytes
    audio_base64: str
    volumes: list[float]  # Volume per chunk for lip-sync
    duration_ms: int
    sample_rate: int
    text: str  # The text that was spoken
    expression: Optional[str] = None  # Detected expression


class EmotionDetector:
    """Detects emotions from text and maps to expressions."""
    
    # Patterns to detect emotions in text
    PATTERNS = [
        r'\((\w+)\)',     # (happy)
        r'\[(\w+)\]',     # [sad]
        r'\*(\w+)\*',     # *excited*
        r'<(\w+)>',       # <blush>
    ]
    
    # Emotion to expression mapping (for March 7th model)
    EMOTION_MAP = {
        'happy': '星星',
        'excited': '比耶',
        'sad': '哭',
        'cry': '哭',
        'crying': '哭',
        'angry': '黑脸',
        'shy': '脸红',
        'blush': '脸红',
        'embarrassed': '流汗',
        'nervous': '流汗',
        'sweat': '流汗',
        'surprised': '比耶',
        'peace': '比耶',
        'photo': '照相',
        'cover': '捂脸',
        'facepalm': '捂脸',
    }
    
    def __init__(self, custom_map: Optional[dict] = None):
        self._compiled_patterns = [re.compile(p, re.IGNORECASE) for p in self.PATTERNS]
        self._map = {**self.EMOTION_MAP}
        if custom_map:
            self._map.update(custom_map)
    
    def detect(self, text: str) -> Optional[str]:
        """Detect emotion from text and return expression name."""
        text_lower = text.lower()
        
        for pattern in self._compiled_patterns:
            matches = pattern.findall(text_lower)
            for match in matches:
                if match in self._map:
                    return self._map[match]
        
        return None
    
    def strip_markers(self, text: str) -> str:
        """Remove emotion markers from text for TTS."""
        result = text
        for pattern in self._compiled_patterns:
            result = pattern.sub('', result)
        return result.strip()


def analyze_audio_volumes(audio_bytes: bytes, sample_rate: int, chunk_ms: int = 50) -> list[float]:
    """
    Analyze audio and return volume levels per chunk.
    
    Args:
        audio_bytes: Raw PCM audio bytes (16-bit signed)
        sample_rate: Audio sample rate
        chunk_ms: Chunk duration in milliseconds
        
    Returns:
        List of normalized volume values (0.0 - 1.0)
    """
    # Convert bytes to samples
    samples = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
    
    # Calculate chunk size in samples
    chunk_samples = int(sample_rate * chunk_ms / 1000)
    
    volumes = []
    for i in range(0, len(samples), chunk_samples):
        chunk = samples[i:i + chunk_samples]
        if len(chunk) == 0:
            continue
        
        # Calculate RMS
        rms = np.sqrt(np.mean(chunk ** 2))
        
        # Normalize (16-bit max = 32767)
        normalized = min(1.0, rms / 8000)  # Lower divisor for more sensitivity
        
        # Apply threshold
        if normalized < 0.05:
            normalized = 0.0
        
        volumes.append(float(normalized))
    
    return volumes


def read_wav_data(wav_path: Path) -> tuple[bytes, int]:
    """Read raw PCM data from WAV file."""
    with wave.open(str(wav_path), 'rb') as wf:
        sample_rate = wf.getframerate()
        frames = wf.readframes(wf.getnframes())
        return frames, sample_rate


class ConversationPipeline:
    """
    Orchestrates the full conversation pipeline.
    
    Flow:
    1. Receive audio bytes from mic
    2. Transcribe with ASR
    3. Generate response with LLM (streaming)
    4. Synthesize speech with TTS
    5. Analyze audio for lip-sync
    6. Send payload to frontend
    """
    
    def __init__(
        self,
        llm: BaseLLM,
        tts: BaseTTS,
        asr: BaseASR,
        config: Optional[ConversationConfig] = None
    ):
        self.llm = llm
        self.tts = tts
        self.asr = asr
        self.config = config or ConversationConfig()
        
        # Emotion detection
        self.emotion_detector = EmotionDetector()
        
        # Conversation history
        self.messages: list[Message] = [
            Message(role="system", content=self.config.system_prompt)
        ]
        
        # Callbacks
        self.on_transcription: Optional[Callable[[str], None]] = None
        self.on_response_start: Optional[Callable[[], None]] = None
        self.on_response_chunk: Optional[Callable[[str], None]] = None
        self.on_response_end: Optional[Callable[[str], None]] = None
        self.on_audio_ready: Optional[Callable[[AudioPayload], None]] = None
        self.on_expression_change: Optional[Callable[[str], None]] = None
        self.on_error: Optional[Callable[[str], None]] = None
        
        # State
        self._is_processing = False
        self._current_expression: Optional[str] = None
        
        logger.info(f"ConversationPipeline initialized (character={self.config.character_name})")
    
    @property
    def is_processing(self) -> bool:
        return self._is_processing
    
    async def process_speech(self, audio_bytes: bytes) -> Optional[str]:
        """
        Process speech audio through the full pipeline.
        
        Args:
            audio_bytes: Raw PCM audio from VAD (16-bit, 16kHz)
            
        Returns:
            The full response text, or None on error
        """
        if self._is_processing:
            logger.warning("Already processing, ignoring new speech")
            return None
        
        self._is_processing = True
        
        try:
            # 1. Transcribe audio
            transcription = await self._transcribe(audio_bytes)
            if not transcription:
                logger.info("No speech detected in audio")
                return None
            
            logger.info(f"📝 Transcription: {transcription}")
            if self.on_transcription:
                await self._call_async(self.on_transcription, transcription)
            
            # 2. Add to conversation history
            self.messages.append(Message(role="user", content=transcription))
            
            # 3. Generate response
            if self.on_response_start:
                await self._call_async(self.on_response_start)
            
            full_response = ""
            
            if self.config.stream_tts:
                # Stream TTS sentence-by-sentence
                full_response = await self._stream_response_with_tts()
            else:
                # Get full response first, then TTS
                full_response = await self._get_full_response()
                await self._synthesize_and_send(full_response)
            
            # 4. Add to history
            self.messages.append(Message(role="assistant", content=full_response))
            
            if self.on_response_end:
                await self._call_async(self.on_response_end, full_response)
            
            return full_response
            
        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            if self.on_error:
                await self._call_async(self.on_error, str(e))
            return None
            
        finally:
            self._is_processing = False
    
    async def _transcribe(self, audio_bytes: bytes) -> Optional[str]:
        """Transcribe audio bytes to text."""
        # Convert int16 bytes to float32 for Whisper
        audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
        audio_float = audio_int16.astype(np.float32) / 32767.0
        
        # Save to temp file (ASR expects file path)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_path = Path(f.name)
        
        try:
            # Write WAV file
            with wave.open(str(temp_path), 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(16000)
                wf.writeframes(audio_bytes)
            
            # Transcribe
            result = self.asr.transcribe(temp_path)
            
            if result.text and result.text.strip():
                return result.text.strip()
            return None
            
        finally:
            temp_path.unlink(missing_ok=True)
    
    async def _get_full_response(self) -> str:
        """Get full LLM response (non-streaming TTS mode)."""
        full_response = ""
        
        async for chunk in self.llm.chat_stream(self.messages):
            full_response += chunk
            if self.on_response_chunk:
                await self._call_async(self.on_response_chunk, chunk)
        
        return full_response
    
    async def _stream_response_with_tts(self) -> str:
        """Stream LLM response and synthesize TTS sentence-by-sentence."""
        full_response = ""
        current_sentence = ""
        sentence_delimiters = '.!?。！？\n'
        
        async for chunk in self.llm.chat_stream(self.messages):
            full_response += chunk
            current_sentence += chunk
            
            if self.on_response_chunk:
                await self._call_async(self.on_response_chunk, chunk)
            
            # Check for complete sentence
            if any(d in chunk for d in sentence_delimiters):
                # Split by delimiters
                parts = re.split(r'([.!?。！？\n]+)', current_sentence)
                
                # Process complete sentences
                for i in range(0, len(parts) - 1, 2):
                    if i + 1 < len(parts):
                        sentence = parts[i] + parts[i + 1]
                        if sentence.strip():
                            await self._synthesize_and_send(sentence)
                
                # Keep remainder
                current_sentence = parts[-1] if parts else ""
        
        # Process any remaining text
        if current_sentence.strip():
            await self._synthesize_and_send(current_sentence)
        
        return full_response
    
    async def _synthesize_and_send(self, text: str):
        """Synthesize text to speech and send payload."""
        if not text.strip():
            return
        
        # 1. Detect emotion
        expression = self.emotion_detector.detect(text)
        if expression and expression != self._current_expression:
            self._current_expression = expression
            logger.info(f"😊 Expression: {expression}")
            if self.on_expression_change:
                await self._call_async(self.on_expression_change, expression)
        
        # 2. Clean text for TTS
        clean_text = self.emotion_detector.strip_markers(text)
        if not clean_text.strip():
            return
        
        # 3. Synthesize audio
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_path = Path(f.name)
        
        try:
            # Run TTS
            if asyncio.iscoroutinefunction(self.tts.synthesize):
                await self.tts.synthesize(clean_text, temp_path)
            else:
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, self.tts.synthesize, clean_text, temp_path)
            
            # 4. Read FULL WAV file (with header) for browser playback
            with open(temp_path, 'rb') as f:
                full_wav_bytes = f.read()
            
            # Also read raw PCM for volume analysis
            audio_bytes, sample_rate = read_wav_data(temp_path)
            volumes = analyze_audio_volumes(audio_bytes, sample_rate, self.config.lip_sync_chunk_ms)
            
            # 5. Create payload with FULL WAV (not just PCM)
            audio_base64 = base64.b64encode(full_wav_bytes).decode('utf-8')
            duration_ms = int(len(audio_bytes) / (sample_rate * 2) * 1000)  # 16-bit = 2 bytes
            
            payload = AudioPayload(
                audio_bytes=audio_bytes,
                audio_base64=audio_base64,
                volumes=volumes,
                duration_ms=duration_ms,
                sample_rate=sample_rate,
                text=clean_text,
                expression=expression
            )
            
            logger.info(f"🔊 Audio ready: {duration_ms}ms, {len(volumes)} volume chunks")
            
            # 6. Send to callback
            if self.on_audio_ready:
                await self._call_async(self.on_audio_ready, payload)
                
        finally:
            temp_path.unlink(missing_ok=True)
    
    async def _call_async(self, callback: Callable, *args):
        """Call callback, awaiting if async."""
        try:
            if asyncio.iscoroutinefunction(callback):
                await callback(*args)
            else:
                callback(*args)
        except Exception as e:
            logger.error(f"Callback error: {e}")
    
    def clear_history(self):
        """Clear conversation history (keep system prompt)."""
        self.messages = [self.messages[0]]  # Keep system prompt
        logger.info("Conversation history cleared")
    
    def add_message(self, role: str, content: str):
        """Manually add a message to history."""
        self.messages.append(Message(role=role, content=content))
