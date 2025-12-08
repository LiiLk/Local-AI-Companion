"""
Live2D Integration with AI Pipeline

This module connects the Live2D avatar with the AI assistant's
LLM output and TTS audio for synchronized expressions and lip-sync.

Features:
- Parse LLM output for emotion markers → trigger expressions
- Analyze TTS audio → drive lip-sync in real-time
- Provide hooks for WebSocket integration
"""

import asyncio
import logging
import re
import struct
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncGenerator, Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class EmotionConfig:
    """Configuration for emotion detection and expression mapping."""
    
    # Patterns to detect emotions in text
    patterns: list[str] = None
    
    # Emotion to expression mapping
    mapping: dict[str, str] = None
    
    # Default expression when no emotion detected
    default_expression: str = None
    
    # Expression duration in seconds (0 = permanent until changed)
    expression_duration: float = 0
    
    def __post_init__(self):
        if self.patterns is None:
            self.patterns = [
                r'\((\w+)\)',     # (happy)
                r'\[(\w+)\]',     # [sad]
                r'\*(\w+)\*',     # *excited*
                r'<(\w+)>',       # <blush>
            ]
        
        if self.mapping is None:
            # Map common emotion names to March 7th expressions
            self.mapping = {
                # English → Chinese expression names
                'happy': '星星',      # Stars (excited/happy)
                'excited': '比耶',     # Peace sign
                'sad': '哭',          # Cry
                'cry': '哭',
                'crying': '哭',
                'angry': '黑脸',       # Dark face
                'shy': '脸红',         # Blush
                'blush': '脸红',
                'embarrassed': '流汗',  # Sweat
                'nervous': '流汗',
                'sweat': '流汗',
                'surprised': '比耶',
                'peace': '比耶',
                'photo': '照相',       # Camera
                'cover': '捂脸',       # Face cover
                'facepalm': '捂脸',
            }


class EmotionDetector:
    """
    Detects emotions from text using pattern matching.
    
    Can be extended with sentiment analysis for more accurate detection.
    """
    
    def __init__(self, config: Optional[EmotionConfig] = None):
        self.config = config or EmotionConfig()
        self._compiled_patterns = [
            re.compile(p, re.IGNORECASE) 
            for p in self.config.patterns
        ]
    
    def detect(self, text: str) -> Optional[str]:
        """
        Detect emotion from text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Detected emotion name or None
        """
        text_lower = text.lower()
        
        # Check for explicit emotion markers
        for pattern in self._compiled_patterns:
            matches = pattern.findall(text_lower)
            for match in matches:
                if match in self.config.mapping:
                    return match
        
        return None
    
    def get_expression(self, emotion: str) -> Optional[str]:
        """Get the Live2D expression name for an emotion."""
        return self.config.mapping.get(emotion.lower())
    
    def strip_emotion_markers(self, text: str) -> str:
        """Remove emotion markers from text for TTS."""
        result = text
        for pattern in self._compiled_patterns:
            result = pattern.sub('', result)
        return result.strip()


class AudioAnalyzer:
    """
    Analyzes audio data to generate lip-sync values.
    
    Works with raw PCM audio data (16-bit, mono or stereo).
    """
    
    def __init__(
        self,
        sample_rate: int = 24000,
        frame_duration_ms: int = 16,  # ~60fps
        smoothing: float = 0.3
    ):
        self.sample_rate = sample_rate
        self.frame_samples = int(sample_rate * frame_duration_ms / 1000)
        self.smoothing = smoothing
        self._last_value = 0.0
        
        # Voice frequency range (100-1000 Hz)
        self.min_freq = 100
        self.max_freq = 1000
    
    def analyze_chunk(self, audio_data: bytes) -> float:
        """
        Analyze an audio chunk and return lip-sync value.
        
        Args:
            audio_data: Raw PCM audio bytes (16-bit signed)
            
        Returns:
            Lip-sync value (0.0 - 1.0)
        """
        try:
            # Convert bytes to numpy array
            samples = np.frombuffer(audio_data, dtype=np.int16)
            
            if len(samples) == 0:
                return self._smooth(0.0)
            
            # Calculate RMS (volume)
            rms = np.sqrt(np.mean(samples.astype(np.float32) ** 2))
            
            # Normalize (16-bit audio max ~32767)
            # Using lower threshold for more responsive lip-sync
            normalized = min(1.0, rms / 6000)
            
            # Apply threshold to filter noise
            if normalized < 0.05:
                normalized = 0.0
            else:
                # Scale up above threshold
                normalized = (normalized - 0.05) * 1.1
            
            return self._smooth(min(1.0, normalized))
            
        except Exception as e:
            logger.error(f"Audio analysis error: {e}")
            return self._smooth(0.0)
    
    def _smooth(self, value: float) -> float:
        """Apply smoothing to avoid jittery lip-sync."""
        self._last_value = (
            self._last_value * (1 - self.smoothing) + 
            value * self.smoothing
        )
        return self._last_value
    
    def reset(self):
        """Reset the analyzer state."""
        self._last_value = 0.0


class Live2DPipelineIntegration:
    """
    Integrates Live2D avatar with the AI pipeline.
    
    Handles:
    - Expression changes based on LLM output
    - Lip-sync from TTS audio streaming
    - WebSocket communication for real-time updates
    """
    
    def __init__(
        self,
        overlay = None,  # Live2DOverlay instance
        emotion_config: Optional[EmotionConfig] = None
    ):
        """
        Initialize the integration.
        
        Args:
            overlay: Live2DOverlay instance (can be set later)
            emotion_config: Configuration for emotion detection
        """
        self.overlay = overlay
        self.emotion_detector = EmotionDetector(emotion_config)
        self.audio_analyzer = AudioAnalyzer()
        
        self._lip_sync_active = False
        self._lip_sync_thread = None
        self._audio_queue = asyncio.Queue()
        self._current_expression = None
    
    def set_overlay(self, overlay):
        """Set the Live2D overlay instance."""
        self.overlay = overlay
    
    def process_llm_output(self, text: str) -> str:
        """
        Process LLM output for emotions and return cleaned text.
        
        Args:
            text: Raw LLM output text
            
        Returns:
            Cleaned text (emotion markers removed)
        """
        # Detect emotion
        emotion = self.emotion_detector.detect(text)
        
        if emotion:
            expression = self.emotion_detector.get_expression(emotion)
            if expression:
                logger.info(f"Detected emotion '{emotion}' → expression '{expression}'")
                self.set_expression(expression)
        
        # Return cleaned text for TTS
        return self.emotion_detector.strip_emotion_markers(text)
    
    def set_expression(self, expression: str):
        """Set avatar expression."""
        if self.overlay and self.overlay.wait_ready(timeout=1.0):
            self.overlay.set_expression(expression)
            self._current_expression = expression
    
    def start_lip_sync_from_stream(
        self,
        audio_generator: AsyncGenerator[bytes, None],
        sample_rate: int = 24000
    ):
        """
        Start lip-sync from an async audio stream.
        
        Args:
            audio_generator: Async generator yielding audio chunks
            sample_rate: Audio sample rate in Hz
        """
        self.audio_analyzer.sample_rate = sample_rate
        self._lip_sync_active = True
        
        async def process_stream():
            try:
                async for chunk in audio_generator:
                    if not self._lip_sync_active:
                        break
                    
                    value = self.audio_analyzer.analyze_chunk(chunk)
                    if self.overlay:
                        self.overlay.set_lip_sync(value)
                    
                    # Small delay to match audio playback
                    await asyncio.sleep(0.016)  # ~60fps
                    
            except Exception as e:
                logger.error(f"Lip sync stream error: {e}")
            finally:
                self.stop_lip_sync()
        
        # Run in event loop
        asyncio.create_task(process_stream())
    
    def process_audio_chunk(self, audio_data: bytes) -> float:
        """
        Process a single audio chunk for lip-sync.
        
        Call this repeatedly as audio is being played.
        
        Args:
            audio_data: Raw PCM audio bytes
            
        Returns:
            Lip-sync value (0.0 - 1.0)
        """
        value = self.audio_analyzer.analyze_chunk(audio_data)
        
        if self.overlay:
            self.overlay.set_lip_sync(value)
        
        return value
    
    def start_lip_sync_from_bytes(
        self,
        audio_data: bytes,
        sample_rate: int = 24000,
        blocking: bool = False
    ):
        """
        Start lip-sync from audio bytes.
        
        Simulates real-time lip-sync by processing audio in chunks.
        
        Args:
            audio_data: Complete audio data as bytes
            sample_rate: Audio sample rate
            blocking: If True, blocks until complete
        """
        self.audio_analyzer.sample_rate = sample_rate
        self._lip_sync_active = True
        
        def process_audio():
            try:
                # Process audio in chunks
                chunk_size = self.audio_analyzer.frame_samples * 2  # 16-bit = 2 bytes
                total_chunks = len(audio_data) // chunk_size
                
                for i in range(0, len(audio_data), chunk_size):
                    if not self._lip_sync_active:
                        break
                    
                    chunk = audio_data[i:i + chunk_size]
                    value = self.audio_analyzer.analyze_chunk(chunk)
                    
                    if self.overlay:
                        self.overlay.set_lip_sync(value)
                    
                    # Wait to match audio playback timing
                    time.sleep(self.audio_analyzer.frame_samples / sample_rate)
                
            except Exception as e:
                logger.error(f"Lip sync error: {e}")
            finally:
                self.stop_lip_sync()
        
        if blocking:
            process_audio()
        else:
            self._lip_sync_thread = threading.Thread(
                target=process_audio, 
                daemon=True
            )
            self._lip_sync_thread.start()
    
    def stop_lip_sync(self):
        """Stop lip-sync and reset to neutral."""
        self._lip_sync_active = False
        self.audio_analyzer.reset()
        
        if self.overlay:
            self.overlay.set_lip_sync(0.0)


class WebSocketLive2DHandler:
    """
    WebSocket handler for Live2D avatar control.
    
    Provides real-time avatar control over WebSocket connections.
    Can be integrated with the existing WebSocket server.
    """
    
    def __init__(self, integration: Live2DPipelineIntegration):
        self.integration = integration
    
    async def handle_message(self, message: dict) -> dict:
        """
        Handle incoming WebSocket messages.
        
        Message format:
            {
                "type": "live2d_command",
                "command": "set_expression" | "set_lip_sync" | "play_motion",
                "params": { ... }
            }
        
        Args:
            message: Parsed JSON message
            
        Returns:
            Response dict
        """
        msg_type = message.get('type')
        
        if msg_type != 'live2d_command':
            return {'status': 'ignored'}
        
        command = message.get('command')
        params = message.get('params', {})
        
        try:
            if command == 'set_expression':
                expression = params.get('expression')
                if expression:
                    self.integration.set_expression(expression)
                    return {'status': 'ok', 'expression': expression}
            
            elif command == 'set_lip_sync':
                value = params.get('value', 0.0)
                if self.integration.overlay:
                    self.integration.overlay.set_lip_sync(value)
                return {'status': 'ok', 'lip_sync': value}
            
            elif command == 'play_motion':
                group = params.get('group')
                index = params.get('index', 0)
                if group and self.integration.overlay:
                    self.integration.overlay.play_motion(group, index)
                return {'status': 'ok', 'motion': f'{group}_{index}'}
            
            elif command == 'get_expressions':
                if self.integration.overlay:
                    expressions = self.integration.overlay.get_expressions()
                    return {'status': 'ok', 'expressions': expressions}
            
            else:
                return {'status': 'error', 'message': f'Unknown command: {command}'}
                
        except Exception as e:
            logger.error(f"WebSocket handler error: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def create_audio_chunk_handler(self) -> Callable[[bytes], None]:
        """
        Create a handler for audio chunks (for TTS integration).
        
        Returns:
            Function that processes audio chunks for lip-sync
        """
        def handle_audio_chunk(audio_data: bytes):
            self.integration.process_audio_chunk(audio_data)
        
        return handle_audio_chunk


# Convenience function for quick integration
def create_live2d_integration(overlay=None) -> Live2DPipelineIntegration:
    """
    Create a Live2D integration instance with default settings.
    
    Args:
        overlay: Optional Live2DOverlay instance
        
    Returns:
        Configured Live2DPipelineIntegration instance
    """
    config = EmotionConfig()
    return Live2DPipelineIntegration(overlay=overlay, emotion_config=config)
