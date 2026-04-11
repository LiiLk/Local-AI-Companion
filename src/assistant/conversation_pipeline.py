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
import io
import base64
import logging
import re
import struct
import tempfile
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import AsyncGenerator, Callable, Optional, Any

import numpy as np

from src.llm.base import BaseLLM, Message
from src.tts.base import BaseTTS, prefers_full_response_tts
from src.tts.tts_task_manager import TTSTaskManager
from src.asr.base import BaseASR, ASRResult
from src.utils.language_detection import LanguageCode, detect_language as detect_text_language
from src.utils.sentence_splitter import SentenceSplitter

logger = logging.getLogger(__name__)

LANGUAGE_NAMES = {
    "fr": "French",
    "en": "English",
}


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
    asr_language: Optional[str] = None

    # Omni mode (MiniCPM-o)
    omni_use_single_pass: bool = True  # Single omni call for speech -> response
    omni_transcribe_for_history: bool = False  # Extra ASR pass for history/UI
    omni_generate_audio: bool = True  # Generate response audio in omni mode
    omni_max_tokens: int = 512
    omni_temperature: float = 0.7


@dataclass
class AudioPayload:
    """Payload containing audio data with lip-sync info."""
    audio_bytes: bytes
    volumes: list[float]  # Volume per chunk for lip-sync
    duration_ms: int
    sample_rate: int
    text: str  # The text that was spoken
    expression: Optional[str] = None  # Detected expression
    audio_base64: Optional[str] = None
    wav_bytes: Optional[bytes] = None
    tts_metrics: Optional[dict[str, float | str | None]] = None


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


def read_wav_bytes(wav_bytes: bytes) -> tuple[bytes, int]:
    """Read raw PCM data from a WAV blob already in memory."""
    with wave.open(io.BytesIO(wav_bytes), 'rb') as wf:
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
        config: Optional[ConversationConfig] = None,
        rvc: Optional[Any] = None,
    ):
        self.llm = llm
        self.tts = tts
        self.asr = asr
        self.config = config or ConversationConfig()
        self.rvc = rvc
        
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
        self._current_language_code: str = self._normalize_supported_language(self.config.asr_language) or "fr"
        self._active_run_id: int = 0
        self._run_counter: int = 0
        
        logger.info(f"ConversationPipeline initialized (character={self.config.character_name})")
    
    @property
    def is_processing(self) -> bool:
        return self._is_processing

    def _begin_run(self, source: str) -> Optional[int]:
        if self._is_processing:
            logger.warning("Already processing, ignoring new %s", source)
            return None

        self._is_processing = True
        self._run_counter += 1
        self._active_run_id = self._run_counter
        return self._active_run_id

    def _finish_run(self, run_id: int) -> None:
        if self._active_run_id == run_id:
            self._is_processing = False

    def _run_is_active(self, run_id: int) -> bool:
        return self._is_processing and self._active_run_id == run_id

    def _ensure_run_active(self, run_id: int) -> None:
        if not self._run_is_active(run_id):
            raise asyncio.CancelledError()

    def cancel_active_run(self, reason: str = "interrupt") -> bool:
        was_active = self._is_processing
        self._run_counter += 1
        self._active_run_id = self._run_counter
        self._is_processing = False
        if was_active:
            logger.info("ConversationPipeline cancel_active_run(%s)", reason)
        return was_active

    def _should_stream_tts_by_sentence(self) -> bool:
        """Return True when sentence-level streaming is appropriate for the provider."""
        return self.config.stream_tts and not prefers_full_response_tts(self.tts)

    @staticmethod
    def _normalize_supported_language(language: Optional[str]) -> Optional[str]:
        if not language:
            return None

        value = language.strip().lower()
        if not value or value == "auto":
            return None
        if value.startswith("fr"):
            return "fr"
        if value.startswith("en"):
            return "en"
        return None

    def _language_default(self) -> LanguageCode:
        return LanguageCode.ENGLISH if self._current_language_code == "en" else LanguageCode.FRENCH

    async def _transcribe_once(self, audio_bytes: bytes, language: Optional[str]) -> ASRResult:
        audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
        audio_float = audio_int16.astype(np.float32) / 32767.0

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.asr.transcribe(audio_float, language=language),
        )

    def _resolve_response_language(self, transcription: str, asr_language: Optional[str]) -> str:
        text_language = str(detect_text_language(transcription, default=self._language_default()))
        asr_code = self._normalize_supported_language(asr_language)

        if asr_code == text_language:
            return asr_code
        if text_language in LANGUAGE_NAMES:
            return text_language
        if asr_code in LANGUAGE_NAMES:
            return asr_code
        return self._current_language_code

    def _apply_language_hint(self, language_code: str) -> None:
        self._current_language_code = language_code
        if self.config.auto_detect_language and hasattr(self.tts, "set_language"):
            self.tts.set_language(language_code)

    def _build_llm_messages(self, language_code: str) -> list[Message]:
        llm_messages = list(self.messages)
        if not llm_messages or llm_messages[-1].role != "user":
            return llm_messages

        language_name = LANGUAGE_NAMES.get(language_code)
        if not language_name:
            return llm_messages

        last_user = llm_messages[-1]
        llm_messages[-1] = Message(
            role="user",
            content=(
                f"(System: The user is speaking {language_name}. "
                f"Reply ONLY in {language_name}. Do not switch languages.)\n\n"
                f"{last_user.content}"
            ),
        )
        return llm_messages
    
    async def process_speech(self, audio_bytes: bytes) -> Optional[str]:
        """
        Process speech audio through the full pipeline.
        
        Args:
            audio_bytes: Raw PCM audio from VAD (16-bit, 16kHz)
            
        Returns:
            The full response text, or None on error
        """
        run_id = self._begin_run("speech")
        if run_id is None:
            return None
        
        try:
            # 1. Transcribe audio
            transcription_result = await self._transcribe(audio_bytes)
            self._ensure_run_active(run_id)
            if not transcription_result or not transcription_result.text:
                logger.info("No speech detected in audio")
                return None
            transcription = transcription_result.text.strip()

            response_language = self._resolve_response_language(
                transcription,
                getattr(transcription_result, "language", None),
            )
            self._apply_language_hint(response_language)
            
            logger.info(f"📝 Transcription: {transcription}")
            if self.on_transcription:
                await self._call_async(self.on_transcription, transcription)
            self._ensure_run_active(run_id)
            
            # 2. Add to conversation history
            self.messages.append(Message(role="user", content=transcription))
            llm_messages = self._build_llm_messages(response_language)
            
            # 3. Generate response
            if self.on_response_start:
                await self._call_async(self.on_response_start)
            self._ensure_run_active(run_id)
            
            full_response = ""
            
            if self._should_stream_tts_by_sentence():
                # Stream TTS sentence-by-sentence
                full_response = await self._stream_response_with_tts(llm_messages, run_id)
            else:
                # Get full response first, then TTS
                full_response = await self._get_full_response(llm_messages, run_id)
                await self._synthesize_and_send(full_response, run_id)
            self._ensure_run_active(run_id)
            
            # 4. Add to history
            self.messages.append(Message(role="assistant", content=full_response))
            
            if self.on_response_end:
                await self._call_async(self.on_response_end, full_response)
            
            return full_response
            
        except asyncio.CancelledError:
            logger.info("Speech pipeline run %s cancelled", run_id)
            raise
        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            if self.on_error:
                await self._call_async(self.on_error, str(e))
            return None
            
        finally:
            self._finish_run(run_id)

    async def process_text(self, text: str) -> Optional[str]:
        """Process direct text input through the LLM -> TTS pipeline."""
        run_id = self._begin_run("text input")
        if run_id is None:
            return None

        user_text = (text or "").strip()
        if not user_text:
            self._finish_run(run_id)
            return None

        try:
            response_language = self._resolve_response_language(user_text, self.config.asr_language)
            self._apply_language_hint(response_language)

            logger.info("⌨️ Desktop text input: %s", user_text[:120])
            if self.on_transcription:
                await self._call_async(self.on_transcription, user_text)
            self._ensure_run_active(run_id)

            self.messages.append(Message(role="user", content=user_text))
            llm_messages = self._build_llm_messages(response_language)

            if self.on_response_start:
                await self._call_async(self.on_response_start)
            self._ensure_run_active(run_id)

            if self._should_stream_tts_by_sentence():
                full_response = await self._stream_response_with_tts(llm_messages, run_id)
            else:
                full_response = await self._get_full_response(llm_messages, run_id)
                await self._synthesize_and_send(full_response, run_id)
            self._ensure_run_active(run_id)

            self.messages.append(Message(role="assistant", content=full_response))

            if self.on_response_end:
                await self._call_async(self.on_response_end, full_response)

            return full_response
        except asyncio.CancelledError:
            logger.info("Text pipeline run %s cancelled", run_id)
            raise
        except Exception as e:
            logger.error(f"Pipeline text error: {e}")
            if self.on_error:
                await self._call_async(self.on_error, str(e))
            return None
        finally:
            self._finish_run(run_id)
    
    async def _transcribe(self, audio_bytes: bytes) -> Optional[ASRResult]:
        """Transcribe audio bytes with a retry guard for bad auto-detection."""
        result = await self._transcribe_once(audio_bytes, self.config.asr_language)
        if not result.text or not result.text.strip():
            return None

        detected_lang = self._normalize_supported_language(getattr(result, "language", None))
        if (
            self._normalize_supported_language(self.config.asr_language) is None
            and getattr(result, "language", None)
            and detected_lang is None
        ):
            retry_language = self._current_language_code or "fr"
            logger.warning(
                "ASR auto-detect returned out-of-scope language=%s, retrying with forced language=%s",
                getattr(result, "language", None),
                retry_language,
            )
            retry_result = await self._transcribe_once(audio_bytes, retry_language)
            if retry_result.text and retry_result.text.strip():
                result = retry_result

        normalized_lang = self._normalize_supported_language(getattr(result, "language", None))
        if normalized_lang:
            result.language = normalized_lang
        return result
    
    async def _get_full_response(self, messages: list[Message], run_id: int) -> str:
        """Get full LLM response (non-streaming TTS mode)."""
        full_response = ""
        
        async for chunk in self.llm.chat_stream(messages):
            self._ensure_run_active(run_id)
            full_response += chunk
            if self.on_response_chunk:
                await self._call_async(self.on_response_chunk, chunk)
            self._ensure_run_active(run_id)
        
        return full_response
    
    async def _stream_response_with_tts(self, messages: list[Message], run_id: int) -> str:
        """Stream the LLM while TTS runs independently in the background."""
        full_response = ""
        splitter = SentenceSplitter(faster_first_response=True)
        llm_started = time.perf_counter()
        first_llm_chunk_logged = False
        first_sentence_submitted = False
        first_audio_sent = False

        async def _on_audio_ready(payload: dict):
            nonlocal first_audio_sent
            if not self._run_is_active(run_id):
                return
            if not self.on_audio_ready:
                if not first_audio_sent:
                    first_audio_sent = True
                    logger.info(
                        "First TTS audio latency: %.1f ms",
                        (time.perf_counter() - llm_started) * 1000,
                    )
                return

            if not first_audio_sent:
                first_audio_sent = True
                logger.info(
                    "First TTS audio latency: %.1f ms",
                    (time.perf_counter() - llm_started) * 1000,
                )

            audio_payload = AudioPayload(
                audio_bytes=payload["pcm_bytes"],
                audio_base64=payload["audio_base64"],
                wav_bytes=payload["wav_bytes"],
                volumes=payload["volumes"],
                duration_ms=payload["duration_ms"],
                sample_rate=payload["sample_rate"],
                text=payload["text"],
                expression=payload.get("expression"),
                tts_metrics=payload.get("tts_metrics"),
            )
            await self._call_async(self.on_audio_ready, audio_payload)

        async def _on_expression(expression: str):
            if not self._run_is_active(run_id):
                return
            if expression and expression != self._current_expression:
                self._current_expression = expression
                if self.on_expression_change:
                    await self._call_async(self.on_expression_change, expression)

        tts_mgr = TTSTaskManager(
            tts=self.tts,
            on_audio_ready=_on_audio_ready,
            rvc=self.rvc,
            sample_rate=self.config.tts_sample_rate,
            lip_sync_chunk_ms=self.config.lip_sync_chunk_ms,
            on_expression=_on_expression,
            emotion_detector=self.emotion_detector,
        )
        await tts_mgr.start()

        try:
            async for chunk in self.llm.chat_stream(messages):
                self._ensure_run_active(run_id)
                full_response += chunk

                if not first_llm_chunk_logged:
                    first_llm_chunk_logged = True
                    logger.info(
                        "First LLM chunk after %.1f ms: %r",
                        (time.perf_counter() - llm_started) * 1000,
                        chunk[:80],
                    )

                if self.on_response_chunk:
                    await self._call_async(self.on_response_chunk, chunk)
                self._ensure_run_active(run_id)

                splitter.feed(chunk)
                for sentence in splitter.get_sentences():
                    if sentence.strip():
                        self._ensure_run_active(run_id)
                        if not first_sentence_submitted:
                            first_sentence_submitted = True
                            logger.info(
                                "First TTS chunk queued after %.1f ms: %r",
                                (time.perf_counter() - llm_started) * 1000,
                                sentence[:80],
                            )
                        await tts_mgr.submit(sentence)

            remaining = splitter.flush()
            if remaining.strip():
                self._ensure_run_active(run_id)
                if not first_sentence_submitted:
                    first_sentence_submitted = True
                    logger.info(
                        "First TTS chunk queued after %.1f ms: %r",
                        (time.perf_counter() - llm_started) * 1000,
                        remaining[:80],
                    )
                await tts_mgr.submit(remaining)

            await tts_mgr.finish()
            return full_response
        except asyncio.CancelledError:
            await tts_mgr.cancel()
            raise
    
    async def _synthesize_and_send(self, text: str, run_id: int):
        """Synthesize text to speech and send payload."""
        if not text.strip():
            return
        self._ensure_run_active(run_id)

        # 1. Detect emotion
        expression = self.emotion_detector.detect(text)
        if expression and expression != self._current_expression:
            self._current_expression = expression
            logger.info(f"😊 Expression: {expression}")
            if self.on_expression_change:
                await self._call_async(self.on_expression_change, expression)
        self._ensure_run_active(run_id)

        # 2. Clean text for TTS
        clean_text = self.emotion_detector.strip_markers(text)
        if not clean_text.strip():
            return

        # 3. Synthesize audio.
        # Providers may either write to the supplied path or return audio_data in memory.
        full_wav_bytes: bytes | None = None
        audio_bytes: bytes | None = None
        sample_rate = self.config.tts_sample_rate

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_path = Path(f.name)

        try:
            if asyncio.iscoroutinefunction(self.tts.synthesize):
                tts_result = await self.tts.synthesize(clean_text, temp_path)
            else:
                loop = asyncio.get_event_loop()
                tts_result = await loop.run_in_executor(None, self.tts.synthesize, clean_text, temp_path)

            if tts_result and tts_result.audio_data:
                full_wav_bytes = tts_result.audio_data
                audio_bytes, sample_rate = read_wav_bytes(full_wav_bytes)
            else:
                audio_path = None
                if tts_result and tts_result.audio_path:
                    audio_path = Path(tts_result.audio_path)
                elif temp_path.exists() and temp_path.stat().st_size > 0:
                    audio_path = temp_path

                if audio_path and audio_path.exists() and audio_path.stat().st_size > 0:
                    full_wav_bytes = audio_path.read_bytes()
                    audio_bytes, sample_rate = read_wav_data(audio_path)
        finally:
            temp_path.unlink(missing_ok=True)

        if not full_wav_bytes or audio_bytes is None:
            return
        self._ensure_run_active(run_id)

        full_wav_bytes, audio_bytes, sample_rate = await self._maybe_apply_rvc(
            full_wav_bytes, audio_bytes, sample_rate
        )
        self._ensure_run_active(run_id)

        volumes = analyze_audio_volumes(audio_bytes, sample_rate, self.config.lip_sync_chunk_ms)

        # 5. Create payload with FULL WAV (not just PCM)
        audio_base64 = base64.b64encode(full_wav_bytes).decode('utf-8')
        duration_ms = int(len(audio_bytes) / (sample_rate * 2) * 1000)

        payload = AudioPayload(
            audio_bytes=audio_bytes,
            audio_base64=audio_base64,
            wav_bytes=full_wav_bytes,
            volumes=volumes,
            duration_ms=duration_ms,
            sample_rate=sample_rate,
            text=clean_text,
            expression=expression
        )

        logger.info(f"🔊 Audio ready: {duration_ms}ms, {len(volumes)} volume chunks")

        if self.on_audio_ready:
            self._ensure_run_active(run_id)
            await self._call_async(self.on_audio_ready, payload)

    async def _maybe_apply_rvc(
        self,
        wav_bytes: bytes,
        audio_bytes: bytes,
        sample_rate: int,
    ) -> tuple[bytes, bytes, int]:
        """Run optional RVC post-processing and fall back to original audio on failure."""
        if not self.rvc:
            return wav_bytes, audio_bytes, sample_rate

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as src:
            src_path = Path(src.name)
            src.write(wav_bytes)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as dst:
            dst_path = Path(dst.name)

        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self.rvc.convert_file, src_path, dst_path)
            converted_wav = dst_path.read_bytes()
            converted_audio, converted_sr = read_wav_data(dst_path)
            return converted_wav, converted_audio, converted_sr
        except Exception as exc:
            logger.warning("RVC conversion failed, using original audio: %s", exc)
            return wav_bytes, audio_bytes, sample_rate
        finally:
            src_path.unlink(missing_ok=True)
            dst_path.unlink(missing_ok=True)

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
