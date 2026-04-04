"""
Gemma-Omni Pipeline — Orchestrator for Gemma E4B + Chatterbox TTS.

Wires together GemmaProvider + ChatterboxTTSProvider to provide the same
callback interface as ConversationPipeline and OmniPipeline.

Flow:
1. Receive audio bytes from mic (VAD)
2. Send audio to Gemma E4B for transcription + response
3. Parse emotions from response (dual output)
4. Send clean text to Chatterbox TTS for voice synthesis
5. Analyze audio for lip-sync
6. Fire callbacks (same interface as existing pipelines)
"""

import asyncio
import base64
import gc
import io
import logging
import wave
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from src.assistant.conversation_pipeline import (
    AudioPayload,
    ConversationConfig,
    analyze_audio_volumes,
)
from src.llm.base import Message
from src.omni.gemma_provider import GemmaProvider
from src.tts.chatterbox_provider import ChatterboxTTSProvider
from src.utils.emotion_detector import EmotionDetector

logger = logging.getLogger(__name__)

MAX_HISTORY_TURNS = 10


class GemmaOmniPipeline:
    """
    Orchestrator: Gemma E4B (ASR+LLM) + Chatterbox (TTS).

    Exposes the same callback interface as OmniPipeline:
        on_transcription, on_response_start, on_response_chunk,
        on_response_end, on_audio_ready, on_expression_change, on_error
    """

    def __init__(
        self,
        gemma: GemmaProvider,
        tts: ChatterboxTTSProvider,
        config: Optional[ConversationConfig] = None,
    ):
        self.gemma = gemma
        self.tts = tts
        self.config = config or ConversationConfig()

        # Emotion detection (canonical implementation)
        self.emotion_detector = EmotionDetector()

        # Conversation history (Gemma message format)
        self.system_prompt = self.config.system_prompt
        self.history: list[dict] = []

        # Callbacks (same interface as ConversationPipeline/OmniPipeline)
        self.on_transcription: Optional[Callable[[str], None]] = None
        self.on_response_start: Optional[Callable[[], None]] = None
        self.on_response_chunk: Optional[Callable[[str], None]] = None
        self.on_response_end: Optional[Callable[[str], None]] = None
        self.on_audio_ready: Optional[Callable[[AudioPayload], None]] = None
        self.on_expression_change: Optional[Callable[[str], None]] = None
        self.on_error: Optional[Callable[[str], None]] = None

        # Screen capture (optional)
        self.screen_buffer: Optional['ScreenBuffer'] = None

        # State
        self._is_processing = False

        # VRAM monitoring
        from src.utils.vram_monitor import VRAMMonitor
        self._vram = VRAMMonitor()

        logger.info(
            f"GemmaOmniPipeline initialized "
            f"(character={self.config.character_name})"
        )

    async def _call_async(self, callback: Callable, *args):
        """Call a callback, awaiting if it is a coroutine function."""
        try:
            if asyncio.iscoroutinefunction(callback):
                await callback(*args)
            else:
                callback(*args)
        except Exception as e:
            logger.error(f"Callback error: {e}")

    def preload(self):
        """Pre-load both models."""
        logger.info("Pre-loading Gemma + Chatterbox...")
        self._vram.log("before_gemma_load")
        self.gemma.preload()
        self._vram.log("after_gemma_load")
        self.tts._load_model()
        self._vram.log("after_chatterbox_load")
        logger.info("Both models loaded")

    async def process_speech(self, audio_bytes: bytes) -> Optional[str]:
        """Process speech — uses streaming if configured."""
        if getattr(self.config, 'stream_tts', False):
            return await self.process_speech_streaming(audio_bytes)
        return await self._process_speech_basic(audio_bytes)

    async def process_speech_streaming(self, audio_bytes: bytes) -> Optional[str]:
        """
        Streaming variant: sentences are sent to TTS as they complete.
        Gemma continues generating while Chatterbox synthesizes.
        """
        if self._is_processing:
            logger.warning("Already processing, skipping")
            return None

        self._is_processing = True

        try:
            from src.utils.sentence_splitter import SentenceSplitter

            system_msg = {"role": "system", "content": [{"type": "text", "text": self.system_prompt}]}
            history_with_system = [system_msg] + self.history

            if self.on_response_start:
                await self._call_async(self.on_response_start)

            if self.on_transcription:
                await self._call_async(self.on_transcription, "[audio input]")

            # Stream tokens from Gemma
            splitter = SentenceSplitter()
            full_response = ""
            first_audio_sent = False

            images = self._get_screen_context()
            stream_kwargs = dict(
                text="",
                history=history_with_system,
                audio=audio_bytes,
                images=images if images else None,
            )

            async for token in self.gemma.chat_stream(**stream_kwargs):
                full_response += token
                splitter.feed(token)

                if self.on_response_chunk:
                    await self._call_async(self.on_response_chunk, token)

                # Check for complete sentences
                for sentence in splitter.get_sentences():
                    await self._synthesize_and_send(sentence, first_audio_sent)
                    first_audio_sent = True

            # Flush remaining text
            remaining = splitter.flush()
            if remaining:
                await self._synthesize_and_send(remaining, first_audio_sent)

            if self.on_response_end:
                await self._call_async(self.on_response_end, full_response)

            # Update history
            self.history.append(
                {"role": "user", "content": [{"type": "audio", "audio": audio_bytes}]}
            )
            self.history.append(
                {"role": "assistant", "content": [{"type": "text", "text": full_response}]}
            )
            max_msgs = MAX_HISTORY_TURNS * 2
            if len(self.history) > max_msgs:
                self.history = self.history[-max_msgs:]

            return full_response

        except Exception as e:
            # OOM during streaming: clean up VRAM before reporting error
            _is_oom = "out of memory" in str(e).lower()
            try:
                import torch
                _is_oom = _is_oom or isinstance(e, torch.cuda.OutOfMemoryError)
                if _is_oom:
                    torch.cuda.empty_cache()
            except ImportError:
                pass
            if _is_oom:
                gc.collect()
                logger.warning("OOM during streaming Gemma inference — cleared VRAM cache")

            logger.error(f"Streaming pipeline error: {e}", exc_info=True)
            if self.on_error:
                await self._call_async(self.on_error, str(e))
            return None
        finally:
            self._is_processing = False

    async def _synthesize_and_send(self, text: str, is_continuation: bool = False) -> None:
        """Synthesize a sentence and fire audio callback."""
        # Emotion detection
        emotion, expression = self.emotion_detector.detect_and_get_expression(text)

        if expression != "neutral" and self.on_expression_change:
            await self._call_async(self.on_expression_change, expression)

        # Clean for TTS
        tts_text = self.emotion_detector.strip_markers_for_tts(text)
        if not tts_text.strip():
            return

        # Synthesize
        tts_result = await self.tts.synthesize(tts_text)
        self._vram.log("after_tts_inference")

        if tts_result.audio_data:
            pcm_data = self._extract_pcm_from_wav(tts_result.audio_data)
            volumes = analyze_audio_volumes(pcm_data, self.tts.SAMPLE_RATE, chunk_ms=50)
            audio_b64 = base64.b64encode(tts_result.audio_data).decode("utf-8")
            duration_ms = int((tts_result.duration or 0) * 1000)

            payload = AudioPayload(
                audio_bytes=pcm_data,
                audio_base64=audio_b64,
                volumes=volumes,
                duration_ms=duration_ms,
                sample_rate=self.tts.SAMPLE_RATE,
                text=tts_text,
                expression=expression,
            )

            if self.on_audio_ready:
                await self._call_async(self.on_audio_ready, payload)

    async def _process_speech_basic(self, audio_bytes: bytes) -> Optional[str]:
        """
        Non-streaming speech processing pipeline.

        Args:
            audio_bytes: Raw PCM 16-bit 16kHz mono audio from VAD.

        Returns:
            The assistant's text response, or None on error.
        """
        if self._is_processing:
            logger.warning("Already processing, skipping")
            return None

        self._is_processing = True

        try:
            # Step 1: Build conversation context
            system_msg = {"role": "system", "content": [{"type": "text", "text": self.system_prompt}]}
            history_with_system = [system_msg] + self.history

            # Step 2: Send audio to Gemma for transcription + response
            if self.on_response_start:
                await self._call_async(self.on_response_start)

            images = self._get_screen_context()
            chat_kwargs = dict(
                text="",
                history=history_with_system,
                audio=audio_bytes,
                images=images if images else None,
            )

            try:
                response = await self.gemma.chat(**chat_kwargs)
            except Exception as e:
                # Check if this is a CUDA OOM error
                _is_oom = "out of memory" in str(e).lower()
                if not _is_oom:
                    try:
                        import torch
                        _is_oom = isinstance(e, torch.cuda.OutOfMemoryError)
                    except ImportError:
                        pass
                if not _is_oom:
                    raise

                logger.warning("OOM during Gemma inference — reducing tokens and retrying")
                try:
                    import torch
                    torch.cuda.empty_cache()
                except ImportError:
                    pass
                gc.collect()

                original_max = self.gemma.max_new_tokens
                self.gemma.max_new_tokens = min(128, original_max)
                try:
                    response = await self.gemma.chat(**chat_kwargs)
                finally:
                    self.gemma.max_new_tokens = original_max

            if not response:
                logger.warning("Empty response from Gemma")
                return None

            # Fire transcription callback (Gemma does ASR implicitly)
            if self.on_transcription:
                await self._call_async(self.on_transcription, "[audio input]")

            # Fire response chunks
            if self.on_response_chunk:
                await self._call_async(self.on_response_chunk, response)

            if self.on_response_end:
                await self._call_async(self.on_response_end, response)

            # Step 3: Detect emotion and get expression
            emotion, expression = self.emotion_detector.detect_and_get_expression(response)

            if expression != "neutral" and self.on_expression_change:
                await self._call_async(self.on_expression_change, expression)

            # Step 4: Clean text for TTS (keep Chatterbox tags)
            tts_text = self.emotion_detector.strip_markers_for_tts(response)

            # Step 5: Synthesize speech with Chatterbox
            tts_result = await self.tts.synthesize(tts_text)

            # Step 6: Build AudioPayload
            if tts_result.audio_data:
                audio_data = tts_result.audio_data

                # Analyze volumes for lip-sync
                pcm_data = self._extract_pcm_from_wav(audio_data)
                volumes = analyze_audio_volumes(
                    pcm_data, self.tts.SAMPLE_RATE, chunk_ms=50
                )

                # Base64 encode the full WAV for browser playback
                audio_b64 = base64.b64encode(audio_data).decode("utf-8")

                duration_ms = int((tts_result.duration or 0) * 1000)

                payload = AudioPayload(
                    audio_bytes=pcm_data,
                    audio_base64=audio_b64,
                    volumes=volumes,
                    duration_ms=duration_ms,
                    sample_rate=self.tts.SAMPLE_RATE,
                    text=tts_text,
                    expression=expression,
                )

                if self.on_audio_ready:
                    await self._call_async(self.on_audio_ready, payload)

            # Step 7: Update history
            self.history.append(
                {"role": "user", "content": [{"type": "audio", "audio": audio_bytes}]}
            )
            self.history.append(
                {"role": "assistant", "content": [{"type": "text", "text": response}]}
            )

            # Trim history
            max_msgs = MAX_HISTORY_TURNS * 2
            if len(self.history) > max_msgs:
                self.history = self.history[-max_msgs:]

            return response

        except Exception as e:
            logger.error(f"Pipeline error: {e}", exc_info=True)
            if self.on_error:
                await self._call_async(self.on_error, str(e))
            return None

        finally:
            self._is_processing = False

    def _extract_pcm_from_wav(self, wav_bytes: bytes) -> bytes:
        """Extract raw PCM data from a WAV file in memory."""
        buf = io.BytesIO(wav_bytes)
        with wave.open(buf, "rb") as wf:
            return wf.readframes(wf.getnframes())

    async def startup(self):
        """Initialize models with progress callbacks."""
        logger.info("Starting GemmaOmniPipeline...")
        self.preload()

    async def shutdown(self):
        """Clean shutdown of all components."""
        logger.info("Shutting down GemmaOmniPipeline...")
        if self.screen_buffer:
            self.screen_buffer.stop()
        self.tts.cleanup()
        self.gemma.cleanup()
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        logger.info("GemmaOmniPipeline shut down")

    async def health_check(self) -> dict:
        """Return pipeline health status."""
        info = {"gemma_loaded": self.gemma._model is not None, "tts_loaded": self.tts._model is not None}
        try:
            import torch
            if torch.cuda.is_available():
                info["vram_allocated_mb"] = torch.cuda.memory_allocated() / 1024 / 1024
                info["vram_reserved_mb"] = torch.cuda.memory_reserved() / 1024 / 1024
        except ImportError:
            pass
        return info

    def clear_history(self):
        """Clear conversation history."""
        self.history.clear()

    def enable_screen_capture(self, config: dict) -> None:
        """Start screen capture if configured."""
        from src.vision.screen_buffer import ScreenBuffer

        self.screen_buffer = ScreenBuffer(
            capture_interval=config.get("interval", 2.0),
            max_buffer=config.get("max_buffer", 30),
            change_threshold=config.get("change_threshold", 0.05),
        )
        self.screen_buffer.start()
        logger.info("Screen capture enabled")

    def _get_screen_context(self) -> list:
        """Get current screen frame(s) for vision context."""
        if not self.screen_buffer:
            return []

        frame = self.screen_buffer.get_latest()
        return [frame] if frame else []
