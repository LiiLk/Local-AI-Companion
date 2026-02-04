"""
Omni Pipeline - Single-model conversation pipeline using MiniCPM-o.

Replaces the traditional ASR -> LLM -> TTS pipeline with a unified
omni model that handles all three tasks internally.

Exposes the SAME callback interface as ConversationPipeline so that
Live2D integration, WebSocket handlers, and other consumers can
switch between pipeline and omni mode without code changes.
"""

import asyncio
import base64
import logging
import re
import tempfile
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from src.assistant.conversation_pipeline import (
    AudioPayload,
    ConversationConfig,
    EmotionDetector,
    analyze_audio_volumes,
    read_wav_data,
)
from src.llm.base import Message
from src.omni.minicpmo_provider import MiniCPMoProvider

logger = logging.getLogger(__name__)


class OmniPipeline:
    """
    Single-model conversation pipeline using MiniCPM-o.

    Flow:
    1. Receive audio bytes from mic
    2. Transcribe with omni model (ASR)
    3. Generate response with omni model (LLM)
    4. Synthesize speech with omni model (TTS)
    5. Analyze audio for lip-sync
    6. Send payload to frontend via callbacks

    This class exposes the same callback interface as ConversationPipeline:
        on_transcription, on_response_start, on_response_chunk,
        on_response_end, on_audio_ready, on_expression_change, on_error
    """

    def __init__(
        self,
        omni: MiniCPMoProvider,
        config: Optional[ConversationConfig] = None,
    ):
        self.omni = omni
        self.config = config or ConversationConfig()

        # Emotion detection (reuse from conversation_pipeline)
        self.emotion_detector = EmotionDetector()

        # Conversation history
        self.messages: list[Message] = [
            Message(role="system", content=self.config.system_prompt)
        ]

        # Callbacks (same interface as ConversationPipeline)
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

        logger.info(
            f"OmniPipeline initialized (character={self.config.character_name})"
        )

    @property
    def is_processing(self) -> bool:
        return self._is_processing

    async def process_speech(self, audio_bytes: bytes) -> Optional[str]:
        """
        Process speech audio through the omni pipeline.

        Args:
            audio_bytes: Raw PCM audio from VAD (16-bit, 16kHz mono).

        Returns:
            The full response text, or None on error.
        """
        if self._is_processing:
            logger.warning("Already processing, ignoring new speech")
            return None

        self._is_processing = True
        loop = asyncio.get_event_loop()

        try:
            # 1. Transcribe audio (ASR)
            transcription = await self._transcribe(audio_bytes, loop)
            if not transcription:
                logger.info("No speech detected in audio")
                return None

            logger.info(f"Transcription: {transcription}")
            if self.on_transcription:
                await self._call_async(self.on_transcription, transcription)

            # 2. Add user message to history
            self.messages.append(Message(role="user", content=transcription))

            # 3. Generate LLM response
            if self.on_response_start:
                await self._call_async(self.on_response_start)

            full_response = await self._generate_response(loop)

            # 4. Add assistant message to history
            self.messages.append(
                Message(role="assistant", content=full_response)
            )

            if self.on_response_end:
                await self._call_async(self.on_response_end, full_response)

            # 5. Synthesize TTS and send audio
            if self.config.stream_tts:
                await self._stream_tts(full_response, loop)
            else:
                await self._synthesize_and_send(full_response, loop)

            return full_response

        except Exception as e:
            logger.error(f"OmniPipeline error: {e}")
            if self.on_error:
                await self._call_async(self.on_error, str(e))
            return None

        finally:
            self._is_processing = False

    async def _transcribe(
        self, audio_bytes: bytes, loop: asyncio.AbstractEventLoop
    ) -> Optional[str]:
        """Transcribe raw PCM int16 audio bytes."""
        audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
        audio_float = audio_int16.astype(np.float32) / 32767.0

        text = await loop.run_in_executor(
            None, self.omni.transcribe, audio_float
        )
        if text and text.strip():
            return text.strip()
        return None

    async def _generate_response(
        self, loop: asyncio.AbstractEventLoop
    ) -> str:
        """Generate a text response via the omni LLM."""
        # Convert internal Message list to the dict format MiniCPM-o expects
        chat_msgs = []
        for m in self.messages:
            chat_msgs.append({"role": m.role, "content": [m.content]})

        full_response = ""

        # Try streaming first
        def _stream():
            chunks = []
            for chunk in self.omni.chat_stream(chat_msgs):
                chunks.append(chunk)
            return chunks

        chunks = await loop.run_in_executor(None, _stream)
        for chunk in chunks:
            full_response += chunk
            if self.on_response_chunk:
                await self._call_async(self.on_response_chunk, chunk)

        return full_response

    async def _stream_tts(
        self, text: str, loop: asyncio.AbstractEventLoop
    ):
        """Split text into sentences and synthesize each one."""
        sentence_delimiters = ".!?;\n"
        parts = re.split(r"([.!?;\n]+)", text)

        i = 0
        while i < len(parts):
            sentence = parts[i]
            # Append delimiter if it follows
            if i + 1 < len(parts):
                sentence += parts[i + 1]
                i += 2
            else:
                i += 1

            if sentence.strip():
                await self._synthesize_and_send(sentence, loop)

    async def _synthesize_and_send(
        self, text: str, loop: asyncio.AbstractEventLoop
    ):
        """Synthesize a piece of text and send audio payload via callback."""
        if not text.strip():
            return

        # 1. Detect emotion
        expression = self.emotion_detector.detect(text)
        if expression and expression != self._current_expression:
            self._current_expression = expression
            logger.info(f"Expression: {expression}")
            if self.on_expression_change:
                await self._call_async(self.on_expression_change, expression)

        # 2. Clean text for TTS
        clean_text = self.emotion_detector.strip_markers(text)
        if not clean_text.strip():
            return

        # 3. Synthesize via omni model
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_path = Path(f.name)

        try:
            await loop.run_in_executor(
                None, self.omni.synthesize, clean_text, temp_path
            )

            # 4. Read WAV for browser playback and volume analysis
            with open(temp_path, "rb") as f:
                full_wav_bytes = f.read()

            audio_bytes, sample_rate = read_wav_data(temp_path)
            volumes = analyze_audio_volumes(
                audio_bytes, sample_rate, self.config.lip_sync_chunk_ms
            )

            audio_base64 = base64.b64encode(full_wav_bytes).decode("utf-8")
            duration_ms = int(
                len(audio_bytes) / (sample_rate * 2) * 1000
            )  # 16-bit = 2 bytes/sample

            payload = AudioPayload(
                audio_bytes=audio_bytes,
                audio_base64=audio_base64,
                volumes=volumes,
                duration_ms=duration_ms,
                sample_rate=sample_rate,
                text=clean_text,
                expression=expression,
            )

            logger.info(
                f"Audio ready: {duration_ms}ms, {len(volumes)} volume chunks"
            )

            if self.on_audio_ready:
                await self._call_async(self.on_audio_ready, payload)

        finally:
            temp_path.unlink(missing_ok=True)

    async def _call_async(self, callback: Callable, *args):
        """Call a callback, awaiting if it is a coroutine function."""
        try:
            if asyncio.iscoroutinefunction(callback):
                await callback(*args)
            else:
                callback(*args)
        except Exception as e:
            logger.error(f"Callback error: {e}")

    def clear_history(self):
        """Clear conversation history, keeping only the system prompt."""
        self.messages = [self.messages[0]]
        logger.info("Conversation history cleared")

    def add_message(self, role: str, content: str):
        """Manually add a message to history."""
        self.messages.append(Message(role=role, content=content))
