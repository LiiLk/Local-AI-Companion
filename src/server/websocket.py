"""
WebSocket Handler - Real-time bidirectional communication.

Handles:
- Text messages (chat)
- Audio streaming with VAD (Voice Activity Detection)
- Configuration updates
- Omni mode (MiniCPM-o unified speech-to-speech)
"""

import json
import asyncio
import base64
import emoji
import tempfile
import time
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, Coroutine
from dataclasses import dataclass, field

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import yaml

from src.llm.base import Message
from src.tts import ChatterboxTTSProvider, KokoroProvider
from src.tts.base import prefers_full_response_tts
from src.asr import WhisperProvider
from src.vad import SileroVAD
from src.assistant.pipeline_runtime import (
    close_pipeline_runtime_services,
    create_pipeline_runtime,
)
from src.assistant.conversation_memory import (
    ConversationMemoryStore,
    initial_messages,
)
from src.utils.audio_analysis import analyze_audio_volumes, read_wav_pcm, calculate_audio_duration_ms
from src.utils.character_loader import resolve_character_config
from src.utils.config_loader import load_yaml_config
from src.utils.emotion_detector import EmotionDetector, strip_emotion_markers
from src.utils.language_detection import (
    detect_language as detect_text_language,
    get_language_name,
    normalize_language_code,
)
from src.tts.tts_task_manager import TTSTaskManager
from src.utils.sentence_splitter import SentenceSplitter

websocket_router = APIRouter()


def load_config() -> dict:
    """Load configuration from config.yaml"""
    config_path = Path(__file__).parent.parent.parent / "config" / "config.yaml"
    return load_yaml_config(config_path)


@dataclass
class ConversationState:
    """
    State for a single WebSocket conversation.

    Each connected client has their own state.
    """
    messages: list = field(default_factory=list)
    llm: Optional[Any] = None
    tts: Optional[Any] = None
    asr: Optional[WhisperProvider] = None
    vad: Optional[SileroVAD] = None
    config: dict = field(default_factory=dict)
    is_recording: bool = False
    audio_buffer: list = field(default_factory=list)
    current_language: str = "en"
    emotion_detector: Optional[EmotionDetector] = None
    current_expression: str = "neutral"
    mode: str = "pipeline"  # "pipeline", "omni", or "gemma-omni"
    omni_model: Optional[Any] = None
    omni_pipeline: Optional[Any] = None
    gemma_model: Optional[Any] = None
    gemma_pipeline: Optional[Any] = None
    rvc: Optional[Any] = None
    memory_store: Optional[ConversationMemoryStore] = None
    pipeline_runtime: Optional[Any] = None
    response_task: Optional[asyncio.Task] = None
    response_lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    def __post_init__(self):
        self.config = load_config()
        self.config = resolve_character_config(self.config)
        self.emotion_detector = EmotionDetector()
        self.mode = self.config.get("mode", "pipeline")

    async def initialize(self):
        """Initialize models (lazy loading)."""
        if self.mode == "omni":
            # Omni mode: defer model loading to get_omni()
            pass
        elif self.mode == "gemma-omni":
            # Defer model loading to get_gemma_omni()
            pass
        else:
            runtime = self._get_pipeline_runtime()
            self.llm = runtime.ensure_llm()
            ensure_memory = getattr(runtime, "ensure_memory", None)
            if callable(ensure_memory):
                self.memory_store = ensure_memory()

        # Initialize conversation with system prompt
        if not self.messages:
            character = self.config.get("character", {})
            system_prompt = character.get("system_prompt", "You are a helpful assistant.")
            self.messages = initial_messages(system_prompt, self.memory_store)

    def _get_pipeline_runtime(self):
        if self.pipeline_runtime is None:
            self.pipeline_runtime = create_pipeline_runtime(
                self.config,
                initial_tts_language=self.current_language,
            )
        return self.pipeline_runtime

    def get_llm(self):
        """Get or create the pipeline LLM."""
        self.llm = self._get_pipeline_runtime().ensure_llm()
        return self.llm

    def get_tts(self):
        """Get or create TTS provider (lazy loading). Pipeline mode only."""
        self.tts = self._get_pipeline_runtime().ensure_tts(self.current_language)
        return self.tts

    def get_rvc(self):
        """Get or create RVC voice converter (lazy loading). Returns None if disabled."""
        self.rvc = self._get_pipeline_runtime().ensure_rvc()
        return self.rvc

    def get_asr(self):
        """Get or create ASR provider (lazy loading). Pipeline mode only."""
        self.asr = self._get_pipeline_runtime().ensure_asr()
        asr_config = self.config.get("asr", {})
        self.asr_language = asr_config.get("language", "")
        return self.asr

    def preload_rvc(self):
        """Preload RVC backend/model when enabled."""
        self.rvc = self._get_pipeline_runtime().preload_rvc()
        return self.rvc

    def get_vad(self):
        """Get or create VAD engine (lazy loading)."""
        if self.vad is None:
            from src.vad.silero_vad import VADConfig

            llm_provider = self.config.get("llm", {}).get("provider", "ollama")
            # Read VAD settings: gemma-omni uses gemma config, pipeline uses pipeline config
            if self.mode == "gemma-omni" or (self.mode == "pipeline" and llm_provider == "gemma"):
                vad_source = self.config.get("gemma", {})
            else:
                vad_source = {}
            pipeline_config = self.config.get("pipeline", {})
            # Pipeline-level vad_required_misses overrides the default 30
            default_misses = pipeline_config.get("vad_required_misses", 30)
            vad_config = VADConfig(
                sample_rate=16000,
                prob_threshold=vad_source.get("vad_prob_threshold", 0.5),
                db_threshold=vad_source.get("vad_db_threshold", -50),
                required_hits=vad_source.get("vad_required_hits", 3),
                required_misses=vad_source.get("vad_required_misses", default_misses),
            )
            self.vad = SileroVAD(config=vad_config)
        return self.vad

    def preload_llm(self):
        """Preload the configured pipeline LLM when it supports it."""
        self.llm = self._get_pipeline_runtime().preload_llm()
        return self.llm

    def preload_asr(self):
        """Preload the configured ASR provider when it supports it."""
        self.asr = self._get_pipeline_runtime().preload_asr()
        return self.asr

    def preload_tts(self):
        """Preload and optionally warm up the configured TTS provider."""
        self.tts = self._get_pipeline_runtime().preload_tts(
            on_load_error=self._fallback_tts_after_preload_error,
        )
        return self.tts

    def _fallback_tts_after_preload_error(self, tts: Any, exc: Exception) -> Any:
        if tts.__class__.__name__ != "Qwen3TTSProvider":
            raise exc

        voice_config = self.config.get("character", {}).get("voice", {})
        fallback_voice = voice_config.get("kokoro_voice") or self.config.get("tts", {}).get("kokoro_voice", "ff_siwis")
        print(f"   Qwen3-TTS preload failed, falling back to Kokoro: {exc}")
        self.tts = KokoroProvider(voice=fallback_voice)
        if self.pipeline_runtime is not None:
            self.pipeline_runtime.tts = self.tts
        return self.tts

    def pipeline_ready(self) -> bool:
        runtime = self.pipeline_runtime
        return bool(runtime and runtime.is_ready())

    def get_omni(self):
        """Get or create the omni model and pipeline (lazy loading)."""
        if self.omni_model is None:
            from src.omni import MiniCPMoProvider, OmniPipeline
            from src.assistant.conversation_pipeline import ConversationConfig

            omni_config = self.config.get("omni", {}).get("minicpmo", {})
            character = self.config.get("character", {})

            # Resolve ref_audio_path from character preset or omni config
            character = self.config.get("character", {})
            voice_config = character.get("voice", {})
            ref_audio = (
                voice_config.get("omni_ref_audio")
                or omni_config.get("ref_audio_path")
            )

            print("Loading MiniCPM-o omni model...")
            self.omni_model = MiniCPMoProvider(
                model_name=omni_config.get("model_id", "openbmb/MiniCPM-o-4_5"),
                device=omni_config.get("device", "cuda"),
                quantization=omni_config.get("quantization"),
                ref_audio_path=ref_audio,
            )
            # Trigger lazy load
            self.omni_model._get_model()

            pipeline_config = ConversationConfig(
                character_name=character.get("name", "AI"),
                system_prompt=character.get("system_prompt", "You are a helpful assistant."),
                reply_language=self.config.get("pipeline", {}).get("reply_language"),
            )
            self.omni_pipeline = OmniPipeline(
                omni=self.omni_model,
                config=pipeline_config,
            )
            print("MiniCPM-o loaded successfully")

        return self.omni_model, self.omni_pipeline

    def get_gemma_omni(self):
        """Get or create the Gemma-Omni pipeline (lazy loading)."""
        if self.gemma_model is None:
            from src.omni import GemmaProvider, GemmaOmniPipeline
            from src.tts import ChatterboxTTSProvider
            from src.assistant.conversation_pipeline import ConversationConfig

            gemma_config = self.config.get("gemma", {})
            tts_config = self.config.get("tts", {})
            chatterbox_config = tts_config.get("chatterbox", {})
            character = self.config.get("character", {})
            voice_config = character.get("voice", {})

            ref_audio = voice_config.get("chatterbox_ref_audio")
            exaggeration = voice_config.get("chatterbox_exaggeration", 0.5)
            language = voice_config.get("chatterbox_language", "en")

            print("Loading Gemma E2B + Chatterbox...")
            self.gemma_model = GemmaProvider(
                model_id=gemma_config.get("model_id", "google/gemma-4-E2B-it"),
                device=gemma_config.get("device", "cuda"),
                quantization=gemma_config.get("quantization", "int4"),
                max_new_tokens=gemma_config.get("max_new_tokens", 256),
                temperature=gemma_config.get("temperature", 0.7),
                top_p=gemma_config.get("top_p", 0.95),
                context_max_turns=gemma_config.get("context_max_turns", 10),
                cpu_offload=gemma_config.get("cpu_offload", True),
                offload_dir=gemma_config.get("offload_dir"),
            )
            self.gemma_model.preload()

            chatterbox = ChatterboxTTSProvider(
                model_id=chatterbox_config.get("model_id", "onnx-community/chatterbox-multilingual-ONNX"),
                ref_audio_path=ref_audio,
                exaggeration=exaggeration,
                language=language,
                prefer_full_gpu=chatterbox_config.get("prefer_full_gpu", True),
            )
            chatterbox._load_model()  # Preload TTS so first response is fast

            pipeline_config = ConversationConfig(
                character_name=character.get("name", "AI"),
                system_prompt=character.get("system_prompt", "You are a helpful assistant."),
                stream_tts=tts_config.get("stream_tts", True),
                reply_language=self.config.get("pipeline", {}).get("reply_language"),
            )
            self.gemma_pipeline = GemmaOmniPipeline(
                gemma=self.gemma_model,
                tts=chatterbox,
                config=pipeline_config,
            )

            screen_config = gemma_config.get("screen", {})
            if screen_config.get("enabled", False):
                self.gemma_pipeline.enable_screen_capture(screen_config)
            print("Gemma + Chatterbox loaded successfully")

        return self.gemma_model, self.gemma_pipeline

    async def cleanup(self):
        """Cleanup resources."""
        if self.response_task and not self.response_task.done():
            self.response_task.cancel()
            try:
                await self.response_task
            except asyncio.CancelledError:
                pass
        if self.pipeline_runtime is not None:
            await self.pipeline_runtime.close()
        else:
            await close_pipeline_runtime_services(
                llm=self.llm,
                tts=self.tts,
                asr=self.asr,
                rvc=self.rvc,
            )


class WebSocketManager:
    """
    Manages WebSocket connections and message handling.
    """

    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.states: Dict[str, ConversationState] = {}
        self._preloading: Dict[str, bool] = {}

    async def connect(self, websocket: WebSocket, client_id: str):
        """Accept a new WebSocket connection."""
        await websocket.accept()
        self.active_connections[client_id] = websocket
        self.states[client_id] = ConversationState()
        await self.states[client_id].initialize()
        print(f"Client connected: {client_id}")

        state = self.states[client_id]

        # Send mode_info to client
        await self.send_json(client_id, {
            "type": "mode_info",
            "mode": state.mode
        })

        # Start preloading models in background
        asyncio.create_task(self._preload_models_progressive(client_id))

    async def disconnect(self, client_id: str):
        """Handle client disconnection."""
        if client_id in self.states:
            await self.states[client_id].cleanup()
            del self.states[client_id]
        if client_id in self.active_connections:
            del self.active_connections[client_id]
        print(f"Client disconnected: {client_id}")

    async def send_json(self, client_id: str, data: dict):
        """Send JSON message to a client."""
        if client_id in self.active_connections:
            try:
                await self.active_connections[client_id].send_json(data)
            except Exception as e:
                print(f"Failed to send to {client_id}: {e}")

    async def send_audio_bytes(self, client_id: str, audio_bytes: bytes):
        """Send raw audio bytes to a client."""
        if client_id in self.active_connections:
            try:
                await self.active_connections[client_id].send_bytes(audio_bytes)
            except Exception as e:
                print(f"Failed to send audio to {client_id}: {e}")

    def _get_state(self, client_id: str) -> Optional[ConversationState]:
        """Safely get client state, returns None if client disconnected."""
        return self.states.get(client_id)

    async def _stop_client_audio(self, client_id: str) -> None:
        """Tell the client to stop queued and currently playing audio."""
        await self.send_json(client_id, {"type": "stop_audio"})

    async def _run_turn(
        self,
        client_id: str,
        turn_coro: Coroutine[Any, Any, None],
    ) -> None:
        """Run a single client turn and clear the active task when it finishes."""
        try:
            await turn_coro
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            print(f"Turn error for {client_id}: {exc}")
            await self.send_json(client_id, {
                "type": "error",
                "message": f"Turn error: {str(exc)}",
            })
        finally:
            state = self._get_state(client_id)
            if state and state.response_task is asyncio.current_task():
                state.response_task = None

    async def _schedule_turn(
        self,
        client_id: str,
        turn_coro: Coroutine[Any, Any, None],
    ) -> Optional[asyncio.Task]:
        """
        Ensure a single active response per client.

        Any new turn interrupts the previous one before starting.
        """
        state = self._get_state(client_id)
        if not state:
            turn_coro.close()
            return None

        async with state.response_lock:
            existing_task = state.response_task
            if existing_task and not existing_task.done():
                existing_task.cancel()
                await self._stop_client_audio(client_id)
                try:
                    await existing_task
                except asyncio.CancelledError:
                    pass
                except Exception as exc:
                    print(f"Cancelled turn cleanup error for {client_id}: {exc}")

            task = asyncio.create_task(self._run_turn(client_id, turn_coro))
            state.response_task = task
            return task

    async def handle_interrupt(self, client_id: str) -> None:
        """Interrupt any active turn for a client and stop current playback."""
        state = self._get_state(client_id)
        if not state:
            return

        active_task = state.response_task
        if active_task and not active_task.done():
            active_task.cancel()
            try:
                await active_task
            except asyncio.CancelledError:
                pass
        state.response_task = None
        if state.vad:
            state.vad.reset()
        await self._stop_client_audio(client_id)

    def _clean_text_for_tts(self, text: str) -> str:
        """
        Clean text before TTS:
        1. Remove emojis
        2. Remove markdown symbols (*, #, _, etc.) that TTS reads out loud
        """
        text = emoji.replace_emoji(text, replace="")

        import re
        text = re.sub(r'\*[^*]+\*', '', text)
        text = re.sub(r'[\*\#\_\`\~\>]+', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    @staticmethod
    def _normalize_supported_language(language: Optional[str]) -> Optional[str]:
        return normalize_language_code(language)

    def _apply_language_hint(self, state: ConversationState, language: Optional[str]) -> Optional[str]:
        normalized = self._normalize_supported_language(language)
        if not normalized:
            return None

        state.current_language = normalized
        if state.pipeline_runtime is not None:
            state.pipeline_runtime.set_tts_language(normalized)
            state.tts = state.pipeline_runtime.tts
        elif state.tts and hasattr(state.tts, "set_language"):
            state.tts.set_language(normalized)
        return normalized

    def _build_llm_messages(
        self,
        messages: list[Message],
        language_code: Optional[str],
    ) -> list[Message]:
        llm_messages = list(messages)
        normalized = self._normalize_supported_language(language_code)
        if not normalized or not llm_messages or llm_messages[-1].role != "user":
            return llm_messages

        language_name = get_language_name(normalized)
        if not language_name:
            return llm_messages

        last_msg = llm_messages[-1]
        llm_messages[-1] = Message(
            role="user",
            content=(
                f"(System: The user is speaking {language_name}. "
                f"Reply ONLY in {language_name}. Do not switch languages.)\n\n"
                f"{last_msg.content}"
            ),
        )
        return llm_messages

    def _resolve_turn_language(
        self,
        state: ConversationState,
        content: str,
        explicit_language: Optional[str] = None,
    ) -> str:
        explicit = self._normalize_supported_language(explicit_language)
        detected = self._normalize_supported_language(
            str(detect_text_language(content, default=state.current_language or "en"))
        )
        return explicit or detected or state.current_language or "en"

    async def _transcribe_with_guard(self, state: ConversationState, audio_input) -> Any:
        """
        Retry once with the current session language when ASR auto-detect
        returns a code we do not normalize locally.
        """
        asr = state.get_asr()
        requested_language = getattr(state, "asr_language", "")
        loop = asyncio.get_event_loop()

        def _do_transcribe(language):
            return asr.transcribe(audio_input, language=language)

        result = await loop.run_in_executor(None, lambda: _do_transcribe(requested_language))
        if not result.text or not result.text.strip():
            return result

        detected_lang = self._normalize_supported_language(getattr(result, "language", None))
        if (
            self._normalize_supported_language(requested_language) is None
            and getattr(result, "language", None)
            and detected_lang is None
        ):
            retry_language = state.current_language or None
            if retry_language:
                print(
                    f"ASR auto-detect returned out-of-scope language={getattr(result, 'language', None)}, "
                    f"retrying with forced language={retry_language}"
                )
                retry = await loop.run_in_executor(None, lambda: _do_transcribe(retry_language))
                if retry.text and retry.text.strip():
                    result = retry

        normalized_lang = self._normalize_supported_language(getattr(result, "language", None))
        if normalized_lang:
            result.language = normalized_lang
        else:
            result.language = state.current_language or "en"

        return result

    async def _synthesize_tts_to_path(self, tts: Any, text: str, output_path: Path) -> Path:
        """Resolve TTS output to a concrete file path even when the provider returns bytes."""
        if asyncio.iscoroutinefunction(tts.synthesize):
            result = await tts.synthesize(text, output_path)
        else:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, tts.synthesize, text, output_path)

        if result and getattr(result, "audio_data", None):
            output_path.write_bytes(result.audio_data)
            return output_path

        returned_path = getattr(result, "audio_path", None) if result else None
        if returned_path:
            candidate = Path(returned_path)
            if candidate.exists() and candidate.stat().st_size > 0:
                return candidate

        if output_path.exists() and output_path.stat().st_size > 0:
            return output_path

        raise RuntimeError("TTS provider returned no audio output")

    async def _update_voice_for_language(self, state: ConversationState, text: str):
        """Detect language and update TTS voice if needed."""
        if not text.strip() or len(text) < 5:
            return

        try:
            lang_code = self._normalize_supported_language(
                str(detect_text_language(text, default=state.current_language or "en"))
            )
            if not lang_code:
                return

            if state.current_language != lang_code:
                print(f"Language switch detected: {state.current_language} -> {lang_code}")
                state.current_language = lang_code

                if state.pipeline_runtime is not None:
                    state.pipeline_runtime.set_tts_language(lang_code)
                    state.tts = state.pipeline_runtime.tts
                elif state.tts and hasattr(state.tts, "set_language"):
                    state.tts.set_language(lang_code)

                tts_config = state.config.get("tts", {})
                provider_name = tts_config.get("provider", "kokoro")
                voice_mapping = tts_config.get("voice_mapping", {}).get(provider_name, {})

                new_voice = voice_mapping.get(lang_code)
                if new_voice and state.tts:
                    if hasattr(state.tts, 'set_voice'):
                        state.tts.set_voice(new_voice)
                        print(f"   Switched {provider_name} voice to: {new_voice}")

        except Exception as e:
            print(f"Voice switch error: {e}")

    async def _process_tts_chunk(self, client_id: str, text: str):
        """Generate and send audio for a text chunk with lip-sync data."""
        if not text.strip():
            return

        state = self._get_state(client_id)
        if not state:
            return

        try:
            emotion = None
            expression = None
            if state.emotion_detector:
                emotion = state.emotion_detector.detect(text)
                if emotion:
                    expression = state.emotion_detector.get_expression(emotion)
                    if expression != state.current_expression:
                        state.current_expression = expression
                        print(f"Expression change: {expression}")
                        await self.send_json(client_id, {
                            "type": "expression_change",
                            "expression": expression,
                            "emotion": emotion
                        })

            await self._update_voice_for_language(state, text)

            if state.emotion_detector:
                text = state.emotion_detector.strip_markers(text)
            text = self._clean_text_for_tts(text)

            if not text.strip():
                return

            tts = state.get_tts()
        except Exception as e:
            print(f"TTS init error: {e}")
            await self.send_json(client_id, {
                "type": "error",
                "message": f"TTS unavailable: {e}"
            })
            return

        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                temp_path = Path(f.name)
            loop = asyncio.get_event_loop()
            temp_path = await self._synthesize_tts_to_path(tts, text, temp_path)

            # RVC voice conversion (optional post-processing)
            rvc = state.get_rvc()
            if rvc:
                try:
                    rvc_out = temp_path.with_suffix(".rvc.wav")
                    await loop.run_in_executor(
                        None, rvc.convert_file, temp_path, rvc_out
                    )
                    temp_path.unlink(missing_ok=True)
                    temp_path = rvc_out
                except Exception as e:
                    print(f"RVC conversion error (using original): {e}")

            with open(temp_path, "rb") as f:
                audio_data = f.read()

            volumes = []
            duration_ms = 0
            try:
                pcm_data, sample_rate = read_wav_pcm(temp_path)
                volumes = analyze_audio_volumes(
                    pcm_data,
                    sample_rate=sample_rate,
                    chunk_ms=50
                )
                duration_ms = calculate_audio_duration_ms(pcm_data, sample_rate)
            except Exception as e:
                print(f"Volume analysis error: {e}")

            audio_base64 = base64.b64encode(audio_data).decode("utf-8")
            await self.send_json(client_id, {
                "type": "audio_data",
                "data": audio_base64,
                "format": "wav",
                "lip_sync": {
                    "volumes": volumes,
                    "duration_ms": duration_ms,
                    "chunk_ms": 50
                },
                "expression": expression,
                "text": text
            })

            temp_path.unlink(missing_ok=True)

        except Exception as e:
            print(f"TTS chunk error: {e}")

    # ------------------------------------------------------------------
    # Omni mode handlers
    # ------------------------------------------------------------------

    async def _handle_text_omni(self, client_id: str, content: str):
        """Handle a text message in omni mode using MiniCPM-o."""
        state = self._get_state(client_id)
        if not state:
            return

        omni_model, omni_pipeline = state.get_omni()

        state.messages.append(Message(role="user", content=content))

        await self.send_json(client_id, {"type": "text_start"})
        await self.send_json(client_id, {"type": "audio_start"})

        loop = asyncio.get_event_loop()
        full_response = ""

        try:
            # Build chat messages in MiniCPM-o format
            chat_msgs = []
            for m in state.messages:
                chat_msgs.append({"role": m.role, "content": [m.content]})

            def _stream():
                chunks = []
                for chunk in omni_model.chat_stream(chat_msgs):
                    chunks.append(chunk)
                return chunks

            chunks = await loop.run_in_executor(None, _stream)
            for chunk in chunks:
                full_response += chunk
                await self.send_json(client_id, {
                    "type": "text_chunk",
                    "content": chunk
                })

            state.messages.append(Message(role="assistant", content=full_response))

            # Detect emotion
            expression = None
            if state.emotion_detector:
                emotion = state.emotion_detector.detect(full_response)
                if emotion:
                    expression = state.emotion_detector.get_expression(emotion)
                    if expression != state.current_expression:
                        state.current_expression = expression
                        await self.send_json(client_id, {
                            "type": "expression_change",
                            "expression": expression,
                            "emotion": emotion
                        })

            # Synthesize audio via omni model
            clean_text = full_response
            if state.emotion_detector:
                clean_text = state.emotion_detector.strip_markers(clean_text)
            clean_text = self._clean_text_for_tts(clean_text)

            if clean_text.strip():
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    temp_path = Path(f.name)

                await loop.run_in_executor(None, omni_model.synthesize, clean_text, temp_path)

                with open(temp_path, "rb") as f:
                    audio_data = f.read()

                volumes = []
                duration_ms = 0
                try:
                    pcm_data, sample_rate = read_wav_pcm(temp_path)
                    volumes = analyze_audio_volumes(pcm_data, sample_rate=sample_rate, chunk_ms=50)
                    duration_ms = calculate_audio_duration_ms(pcm_data, sample_rate)
                except Exception as e:
                    print(f"Volume analysis error: {e}")

                audio_base64 = base64.b64encode(audio_data).decode("utf-8")
                await self.send_json(client_id, {
                    "type": "audio_data",
                    "data": audio_base64,
                    "format": "wav",
                    "lip_sync": {
                        "volumes": volumes,
                        "duration_ms": duration_ms,
                        "chunk_ms": 50
                    },
                    "expression": expression,
                    "text": clean_text
                })

                temp_path.unlink(missing_ok=True)

        except Exception as e:
            print(f"Omni text error: {e}")
            await self.send_json(client_id, {
                "type": "error",
                "message": f"Omni processing error: {str(e)}"
            })

        await self.send_json(client_id, {
            "type": "text_end",
            "full_text": full_response
        })
        await self.send_json(client_id, {"type": "audio_end"})
        await self._curate_pipeline_memory(state, content, full_response)

    async def _handle_audio_omni(self, client_id: str, audio_bytes: bytes):
        """Handle audio input in omni mode (no separate ASR step)."""
        state = self._get_state(client_id)
        if not state:
            return

        omni_model, omni_pipeline = state.get_omni()
        loop = asyncio.get_event_loop()

        try:
            audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
            audio_float = audio_int16.astype(np.float32) / 32767.0

            duration_sec = len(audio_int16) / 16000
            print(f"Omni audio received: {len(audio_bytes)} bytes, {duration_sec:.2f}s")

            if duration_sec < 0.5:
                print("Audio too short (< 0.5s), ignored")
                await self.send_json(client_id, {
                    "type": "transcription",
                    "text": "",
                    "message": "Audio too short"
                })
                return

            await self.send_json(client_id, {"type": "transcribing"})
            transcription = await loop.run_in_executor(
                None, omni_model.transcribe, audio_float
            )

            if not transcription or not transcription.strip():
                await self.send_json(client_id, {
                    "type": "transcription",
                    "text": "",
                    "message": "No speech detected"
                })
                return

            await self.send_json(client_id, {
                "type": "transcription",
                "text": transcription,
            })

            await self._handle_text_omni(client_id, transcription)

        except Exception as e:
            print(f"Omni audio error: {e}")
            await self.send_json(client_id, {
                "type": "error",
                "message": f"Omni audio error: {str(e)}"
            })

    # ------------------------------------------------------------------
    # Gemma-omni mode handlers
    # ------------------------------------------------------------------

    def _wire_gemma_callbacks(self, client_id: str, pipeline):
        """Wire GemmaOmniPipeline callbacks to WebSocket sends."""
        if pipeline.on_audio_ready is not None:
            return  # Already wired

        async def send_json(msg):
            try:
                if client_id in self.active_connections:
                    await self.active_connections[client_id].send_json(msg)
            except Exception:
                pass

        async def on_transcription(text):
            await send_json({"type": "transcription", "text": text})

        async def on_response_start():
            await send_json({"type": "text_start"})

        async def on_response_chunk(text):
            await send_json({"type": "text_chunk", "content": text})

        async def on_response_end(text):
            await send_json({"type": "text_end", "full_text": text})

        async def on_audio_ready(payload):
            await send_json({
                "type": "audio_meta",
                "format": "wav",
                "lip_sync": {
                    "volumes": payload.volumes,
                    "duration_ms": payload.duration_ms,
                    "chunk_ms": 50
                },
                "expression": payload.expression,
                "text": payload.text,
            })
            if payload.wav_bytes:
                await self.send_audio_bytes(client_id, payload.wav_bytes)

        async def on_expression_change(expr):
            await send_json({"type": "expression_change", "expression": expr})
            state = self._get_state(client_id)
            if state:
                state.current_expression = expr

        async def on_error(error_msg):
            await send_json({"type": "error", "message": f"Gemma error: {error_msg}"})

        pipeline.on_transcription = on_transcription
        pipeline.on_response_start = on_response_start
        pipeline.on_response_chunk = on_response_chunk
        pipeline.on_response_end = on_response_end
        pipeline.on_audio_ready = on_audio_ready
        pipeline.on_expression_change = on_expression_change
        pipeline.on_error = on_error

    async def _handle_text_gemma(self, client_id: str, content: str):
        """Handle a text message in gemma-omni mode."""
        state = self._get_state(client_id)
        if not state:
            return

        _, pipeline = state.get_gemma_omni()
        self._wire_gemma_callbacks(client_id, pipeline)
        from src.utils.sentence_splitter import SentenceSplitter

        await self.send_json(client_id, {"type": "text_start"})
        await self.send_json(client_id, {"type": "audio_start"})

        full_response = ""
        try:
            splitter = SentenceSplitter()
            async for chunk in pipeline.gemma.chat_stream(
                text=content,
                history=[{"role": "system", "content": [{"type": "text", "text": pipeline.system_prompt}]}] + pipeline.history,
            ):
                if not chunk:
                    continue
                full_response += chunk
                splitter.feed(chunk)
                await self.send_json(client_id, {
                    "type": "text_chunk",
                    "content": chunk
                })

                for sentence in splitter.get_sentences():
                    await pipeline._synthesize_and_send(sentence)

            remaining = splitter.flush()
            if remaining:
                await pipeline._synthesize_and_send(remaining)

            if full_response:
                pipeline.history.append(
                    {"role": "user", "content": [{"type": "text", "text": content}]}
                )
                pipeline.history.append(
                    {"role": "assistant", "content": [{"type": "text", "text": full_response}]}
                )
                max_msgs = pipeline.gemma.context_max_turns * 2
                if len(pipeline.history) > max_msgs:
                    pipeline.history = pipeline.history[-max_msgs:]

        except Exception as e:
            print(f"Gemma text error: {e}")
            await self.send_json(client_id, {
                "type": "error",
                "message": f"Gemma processing error: {str(e)}"
            })

        await self.send_json(client_id, {
            "type": "text_end",
            "full_text": full_response
        })
        await self.send_json(client_id, {"type": "audio_end"})

    async def _handle_audio_gemma(self, client_id: str, audio_bytes: bytes):
        """Handle audio input in gemma-omni mode (Gemma does ASR+LLM together)."""
        state = self._get_state(client_id)
        if not state:
            return

        _, pipeline = state.get_gemma_omni()
        self._wire_gemma_callbacks(client_id, pipeline)

        try:
            audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
            duration_sec = len(audio_int16) / 16000
            print(f"Gemma audio received: {len(audio_bytes)} bytes, {duration_sec:.2f}s")

            if duration_sec < 0.5:
                print("Audio too short (< 0.5s), ignored")
                await self.send_json(client_id, {
                    "type": "transcription",
                    "text": "",
                    "message": "Audio too short"
                })
                return

            # Gemma handles ASR + LLM in one pass via process_speech
            await pipeline.process_speech(audio_bytes)

        except Exception as e:
            print(f"Gemma audio error: {e}")
            await self.send_json(client_id, {
                "type": "error",
                "message": f"Gemma audio error: {str(e)}"
            })

    # ------------------------------------------------------------------
    # Pipeline mode handlers
    # ------------------------------------------------------------------

    async def _handle_text_message_turn(
        self,
        client_id: str,
        content: str,
        language: str | None = None,
        trace: Optional[dict[str, Any]] = None,
    ):
        """Handle a text message from the client."""
        state = self._get_state(client_id)
        if not state:
            return

        # Branch on mode
        if state.mode == "omni":
            await self._handle_text_omni(client_id, content)
            return

        elif state.mode == "gemma-omni":
            await self._handle_text_gemma(client_id, content)
            return

        # Pipeline mode
        trace_data = dict(trace or {})
        now_ms = int(time.time() * 1000)
        trace_data.setdefault("turn_start_epoch_ms", now_ms)
        trace_data.setdefault("text_submit_epoch_ms", now_ms)
        trace_data.setdefault("asr_done_epoch_ms", now_ms)
        state.messages.append(Message(role="user", content=content))

        await self.send_json(client_id, {"type": "text_start"})
        await self.send_json(client_id, {"type": "audio_start"})

        full_response = ""
        llm_started = time.perf_counter()
        first_sentence_logged = False
        first_audio_logged = False

        response_language = self._resolve_turn_language(state, content, explicit_language=language)
        self._apply_language_hint(state, response_language)
        llm_messages = self._build_llm_messages(state.messages, response_language)
        trace_data["llm_start_epoch_ms"] = int(time.time() * 1000)

        # --- Decoupled TTS pipeline ---
        # LLM tokens stream into SentenceSplitter, complete sentences are
        # queued to TTSTaskManager which synthesizes independently.
        # The LLM stream never blocks while TTS is working.

        splitter = SentenceSplitter(faster_first_response=True)
        manager = self  # capture for closure

        async def _on_ws_audio(payload: dict):
            nonlocal first_audio_logged
            if not first_audio_logged:
                first_audio_logged = True
                print(f"First TTS audio latency for {client_id}: {(time.perf_counter() - llm_started) * 1000:.1f} ms")

            payload_trace = dict(trace_data)
            payload_trace["backend_audio_ready_epoch_ms"] = int(time.time() * 1000)

            expression = payload.get("expression")
            if expression and state.emotion_detector:
                if expression != state.current_expression:
                    state.current_expression = expression
                    await manager.send_json(client_id, {
                        "type": "expression_change",
                        "expression": expression,
                    })

            await manager.send_json(client_id, {
                "type": "audio_data",
                "data": payload["audio_base64"],
                "format": "wav",
                "lip_sync": {
                    "volumes": payload["volumes"],
                    "duration_ms": payload["duration_ms"],
                    "chunk_ms": 50,
                },
                "expression": expression,
                "text": payload["text"],
                "trace": payload_trace,
            })

        tts_obj = state.get_tts()
        rvc_obj = state.get_rvc()
        single_shot_tts = prefers_full_response_tts(tts_obj)

        tts_mgr = TTSTaskManager(
            tts=tts_obj,
            on_audio_ready=_on_ws_audio,
            rvc=rvc_obj,
            emotion_detector=state.emotion_detector,
        )
        await tts_mgr.start()

        try:
            llm = state.get_llm()
            async for chunk in llm.chat_stream(llm_messages):
                if "llm_first_token_epoch_ms" not in trace_data:
                    trace_data["llm_first_token_epoch_ms"] = int(time.time() * 1000)
                full_response += chunk

                await self.send_json(client_id, {
                    "type": "text_chunk",
                    "content": chunk,
                })

                if not single_shot_tts:
                    splitter.feed(chunk)
                    for sentence in splitter.get_sentences():
                        if not first_sentence_logged:
                            first_sentence_logged = True
                            trace_data["tts_first_chunk_epoch_ms"] = int(time.time() * 1000)
                            print(f"LLM first sentence latency for {client_id}: {(time.perf_counter() - llm_started) * 1000:.1f} ms")
                        await self._update_voice_for_language(state, sentence)
                        clean = self._clean_text_for_tts(sentence)
                        if clean.strip():
                            await tts_mgr.submit(clean)

            if single_shot_tts:
                clean = self._clean_text_for_tts(full_response)
                if clean.strip():
                    if not first_sentence_logged:
                        first_sentence_logged = True
                        trace_data["tts_first_chunk_epoch_ms"] = int(time.time() * 1000)
                    await self._update_voice_for_language(state, clean)
                    await tts_mgr.submit(clean)
            else:
                remaining = splitter.flush()
                if remaining:
                    if not first_sentence_logged:
                        first_sentence_logged = True
                        trace_data["tts_first_chunk_epoch_ms"] = int(time.time() * 1000)
                        print(f"LLM first sentence latency for {client_id}: {(time.perf_counter() - llm_started) * 1000:.1f} ms")
                    clean = self._clean_text_for_tts(remaining)
                    if clean.strip():
                        await self._update_voice_for_language(state, clean)
                        await tts_mgr.submit(clean)

            print(f"LLM total generation time for {client_id}: {(time.perf_counter() - llm_started) * 1000:.1f} ms")
            await tts_mgr.finish()
        except asyncio.CancelledError:
            await tts_mgr.cancel()
            raise

        state.messages.append(Message(role="assistant", content=full_response))
        self._persist_pipeline_exchange(state, content, full_response)

        await self.send_json(client_id, {
            "type": "text_end",
            "full_text": full_response
        })
        await self.send_json(client_id, {"type": "audio_end"})
        await self._curate_pipeline_memory(state, content, full_response)

    async def handle_text_message(
        self,
        client_id: str,
        content: str,
        language: str | None = None,
        trace: Optional[dict[str, Any]] = None,
    ):
        """Schedule a text turn, interrupting any active turn for this client."""
        await self._schedule_turn(
            client_id,
            self._handle_text_message_turn(client_id, content, language=language, trace=trace),
        )

    async def generate_and_send_audio(self, client_id: str, text: str):
        """Generate TTS audio and send to client."""
        text = self._clean_text_for_tts(text)

        if not text.strip():
            return

        state = self._get_state(client_id)
        if not state:
            return

        tts = state.get_tts()

        await self.send_json(client_id, {"type": "audio_start"})

        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                temp_path = Path(f.name)

            temp_path = await self._synthesize_tts_to_path(tts, text, temp_path)

            with open(temp_path, "rb") as f:
                audio_data = f.read()

            audio_base64 = base64.b64encode(audio_data).decode("utf-8")
            await self.send_json(client_id, {
                "type": "audio_data",
                "data": audio_base64,
                "format": "wav"
            })

            temp_path.unlink(missing_ok=True)

        except Exception as e:
            await self.send_json(client_id, {
                "type": "error",
                "message": f"TTS error: {str(e)}"
            })

        await self.send_json(client_id, {"type": "audio_end"})

    async def _handle_audio_message_turn(self, client_id: str, audio_data: str):
        """Handle audio data from the client (WebM blob)."""
        state = self._get_state(client_id)
        if not state:
            return

        asr = state.get_asr()
        trace = {
            "turn_start_epoch_ms": int(time.time() * 1000),
            "speech_end_epoch_ms": int(time.time() * 1000),
        }

        try:
            audio_bytes = base64.b64decode(audio_data)

            with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as f:
                f.write(audio_bytes)
                webm_path = Path(f.name)

            wav_path = webm_path.with_suffix(".wav")

            import subprocess
            try:
                result = subprocess.run(
                    [
                        "ffmpeg", "-y", "-i", str(webm_path),
                        "-ar", "16000",
                        "-ac", "1",
                        "-f", "wav",
                        str(wav_path)
                    ],
                    capture_output=True,
                    timeout=30
                )
                if result.returncode != 0:
                    raise Exception(f"ffmpeg error: {result.stderr.decode()}")
            except FileNotFoundError:
                raise Exception("ffmpeg not found. Please install ffmpeg.")

            webm_path.unlink(missing_ok=True)

            await self.send_json(client_id, {"type": "transcribing"})
            result = await self._transcribe_with_guard(state, wav_path)
            trace["asr_done_epoch_ms"] = int(time.time() * 1000)

            wav_path.unlink(missing_ok=True)

            if result.text.strip():
                await self.send_json(client_id, {
                    "type": "transcription",
                    "text": result.text,
                    "language": result.language
                })
                await self._handle_text_message_turn(
                    client_id,
                    result.text,
                    language=result.language,
                    trace=trace,
                )
            else:
                await self.send_json(client_id, {
                    "type": "transcription",
                    "text": "",
                    "message": "No speech detected"
                })

        except Exception as e:
            print(f"ASR error: {e}")
            await self.send_json(client_id, {
                "type": "error",
                "message": f"ASR error: {str(e)}"
            })

    async def handle_audio_message(self, client_id: str, audio_data: str):
        """Schedule an uploaded-audio turn, interrupting any active turn for this client."""
        await self._schedule_turn(
            client_id,
            self._handle_audio_message_turn(client_id, audio_data),
        )

    async def handle_audio_stream(self, client_id: str, audio_samples: list):
        """Handle streaming audio data with VAD."""
        state = self._get_state(client_id)
        if not state:
            return

        vad = state.get_vad()

        try:
            for event in vad.process_audio(audio_samples):
                if event == b"<|START|>":
                    state.is_recording = True
                    await self.send_json(client_id, {"type": "vad_start"})

                elif event == b"<|END|>":
                    state.is_recording = False
                    await self.send_json(client_id, {"type": "vad_end"})

                elif len(event) > 100:
                    if state.mode == "omni":
                        await self._schedule_turn(client_id, self._handle_audio_omni(client_id, event))
                    elif state.mode == "gemma-omni":
                        await self._schedule_turn(client_id, self._handle_audio_gemma(client_id, event))
                    else:
                        await self._schedule_turn(client_id, self._transcribe_and_respond_turn(client_id, event))

        except Exception as e:
            print(f"VAD error: {e}")
            await self.send_json(client_id, {
                "type": "error",
                "message": f"VAD error: {str(e)}"
            })

    async def handle_audio_stream_bytes(self, client_id: str, audio_bytes: bytes):
        """Handle PCM16 streaming audio frames sent as binary WebSocket messages."""
        if not audio_bytes:
            return

        audio_samples = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32767.0
        await self.handle_audio_stream(client_id, audio_samples)

    async def _transcribe_and_respond_turn(self, client_id: str, audio_bytes: bytes):
        """Transcribe audio bytes and generate response. Pipeline mode only."""
        state = self._get_state(client_id)
        if not state:
            return

        asr = state.get_asr()
        trace = {
            "turn_start_epoch_ms": int(time.time() * 1000),
            "speech_end_epoch_ms": int(time.time() * 1000),
        }

        try:
            audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
            audio_float = audio_int16.astype(np.float32) / 32767.0

            duration_sec = len(audio_int16) / 16000
            print(f"Audio received: {len(audio_bytes)} bytes, {duration_sec:.2f}s, {len(audio_int16)} samples")

            if duration_sec < 0.5:
                print("Audio too short (< 0.5s), ignored")
                await self.send_json(client_id, {
                    "type": "transcription",
                    "text": "",
                    "message": "Audio too short"
                })
                return

            await self.send_json(client_id, {"type": "transcribing"})
            result = await self._transcribe_with_guard(state, audio_float)
            trace["asr_done_epoch_ms"] = int(time.time() * 1000)

            if result.text.strip():
                await self.send_json(client_id, {
                    "type": "transcription",
                    "text": result.text,
                    "language": result.language
                })
                await self._handle_text_message_turn(
                    client_id,
                    result.text,
                    language=result.language,
                    trace=trace,
                )
            else:
                await self.send_json(client_id, {
                    "type": "transcription",
                    "text": "",
                    "message": "No speech detected"
                })

        except Exception as e:
            print(f"Transcription error: {e}")
            await self.send_json(client_id, {
                "type": "error",
                "message": f"Transcription error: {str(e)}"
            })

    async def handle_clear(self, client_id: str):
        """Clear conversation history."""
        state = self._get_state(client_id)
        if not state:
            return

        if state.memory_store:
            state.memory_store.clear()
        character = state.config.get("character", {})
        system_prompt = character.get("system_prompt", "You are a helpful assistant.")
        state.messages = initial_messages(system_prompt, state.memory_store)

        active_task = state.response_task
        if active_task and not active_task.done():
            active_task.cancel()
            try:
                await active_task
            except asyncio.CancelledError:
                pass
        state.response_task = None
        await self._stop_client_audio(client_id)

        if state.omni_pipeline:
            state.omni_pipeline.clear_history()

        if state.gemma_pipeline:
            state.gemma_pipeline.clear_history()

        await self.send_json(client_id, {
            "type": "cleared",
            "message": "Conversation history cleared"
        })

    @staticmethod
    def _persist_pipeline_exchange(
        state: ConversationState,
        user_text: str,
        assistant_text: str,
    ) -> None:
        if not state.memory_store:
            return
        if not state.memory_store.append_exchange(user_text, assistant_text):
            return
        character = state.config.get("character", {})
        system_prompt = character.get("system_prompt", "You are a helpful assistant.")
        state.messages = initial_messages(system_prompt, state.memory_store)

    async def _curate_pipeline_memory(
        self,
        state: ConversationState,
        user_text: str,
        assistant_text: str,
    ) -> None:
        if not state.memory_store:
            return
        if await state.memory_store.curate_exchange(state.get_llm(), user_text, assistant_text):
            character = state.config.get("character", {})
            system_prompt = character.get("system_prompt", "You are a helpful assistant.")
            state.messages = initial_messages(system_prompt, state.memory_store)

    async def _preload_models_progressive(self, client_id: str):
        """Preload models progressively on connection."""
        state = self.states.get(client_id)
        if not state:
            return

        if self._preloading.get(client_id, False):
            return
        self._preloading[client_id] = True

        loop = asyncio.get_event_loop()

        def is_connected() -> bool:
            return client_id in self.active_connections

        async def safe_send(data: dict):
            if is_connected():
                await self.send_json(client_id, data)

        try:
            await asyncio.sleep(0.2)

            if not is_connected():
                return

            await safe_send({
                "type": "models_loading",
                "message": "Preparing models...",
                "progress": 0
            })

            if state.mode == "omni":
                # Omni mode: single model to load + VAD
                await safe_send({
                    "type": "model_loading",
                    "model": "omni",
                    "message": "Loading MiniCPM-o omni model...",
                    "progress": 10
                })
                try:
                    await loop.run_in_executor(None, state.get_omni)
                    print(f"   Omni model loaded for {client_id}")
                    await safe_send({
                        "type": "model_loaded",
                        "model": "omni",
                        "message": "Omni model ready!",
                        "progress": 90
                    })
                except Exception as e:
                    print(f"   Omni model load error: {e}")
                    await safe_send({
                        "type": "models_error",
                        "message": f"Omni model failed: {str(e)}"
                    })

                if not state.vad and is_connected():
                    try:
                        await loop.run_in_executor(None, state.get_vad)
                        print(f"   VAD loaded for {client_id}")
                    except Exception as e:
                        print(f"   VAD load warning: {e}")

            elif state.mode == "gemma-omni":
                await safe_send({
                    "type": "model_loading",
                    "model": "gemma",
                    "message": "Loading Gemma E2B + Chatterbox...",
                    "progress": 10
                })
                try:
                    await loop.run_in_executor(None, state.get_gemma_omni)
                    print(f"   Gemma + Chatterbox loaded for {client_id}")
                    await safe_send({
                        "type": "model_loaded",
                        "model": "gemma",
                        "message": "Gemma + Chatterbox ready!",
                        "progress": 90
                    })
                except Exception as e:
                    print(f"   Gemma load error: {e}")
                    await safe_send({
                        "type": "models_error",
                        "message": f"Gemma model failed: {str(e)}"
                    })

                if not state.vad and is_connected():
                    try:
                        await loop.run_in_executor(None, state.get_vad)
                        print(f"   VAD loaded for {client_id}")
                    except Exception as e:
                        print(f"   VAD load warning: {e}")

            else:
                # Pipeline mode: load VAD, brain, TTS, ASR
                if not state.vad and is_connected():
                    await safe_send({
                        "type": "model_loading",
                        "model": "vad",
                        "message": "Loading VAD (voice detection)...",
                        "progress": 5
                    })
                    try:
                        await loop.run_in_executor(None, state.get_vad)
                        print(f"   VAD loaded for {client_id}")
                        await safe_send({
                            "type": "model_loaded",
                            "model": "vad",
                            "message": "VAD ready!",
                            "progress": 15
                        })
                    except Exception as e:
                        print(f"   VAD load warning: {e}")

                llm_provider = state.config.get("llm", {}).get("provider", "ollama")
                if llm_provider == "gemma" and is_connected():
                    await safe_send({
                        "type": "model_loading",
                        "model": "llm",
                        "message": "Loading Gemma brain...",
                        "progress": 20
                    })
                    try:
                        await loop.run_in_executor(None, state.preload_llm)
                        print(f"   Gemma brain loaded for {client_id}")
                        await safe_send({
                            "type": "model_loaded",
                            "model": "llm",
                            "message": "Gemma brain ready!",
                            "progress": 55
                        })
                    except Exception as e:
                        print(f"   LLM load error: {e}")
                        await safe_send({
                            "type": "models_error",
                            "message": f"LLM failed: {str(e)}"
                        })

                if not state.tts and is_connected():
                    await safe_send({
                        "type": "model_loading",
                        "model": "tts",
                        "message": "Loading TTS (voice synthesis)...",
                        "progress": 60
                    })
                    try:
                        await loop.run_in_executor(None, state.preload_tts)
                        print(f"   TTS loaded for {client_id}")
                        await safe_send({
                            "type": "model_loaded",
                            "model": "tts",
                            "message": "TTS ready!",
                            "progress": 80
                        })
                    except Exception as e:
                        print(f"   TTS load error: {e}")
                        await safe_send({
                            "type": "models_error",
                            "message": f"TTS failed: {str(e)}"
                        })

                if not state.asr and is_connected():
                    await safe_send({
                        "type": "model_loading",
                        "model": "asr",
                        "message": "Loading ASR (speech recognition)...",
                        "progress": 82
                    })
                    try:
                        await loop.run_in_executor(None, state.preload_asr)
                        print(f"   ASR loaded for {client_id}")
                        await safe_send({
                            "type": "model_loaded",
                            "model": "asr",
                            "message": "ASR ready!",
                            "progress": 100
                        })
                    except Exception as e:
                        print(f"   ASR load error: {e}")
                        await safe_send({
                            "type": "models_error",
                            "message": f"ASR failed: {str(e)}"
                        })

                rvc_config = state.config.get("tts", {}).get("rvc", {})
                if rvc_config.get("enabled", False) and is_connected():
                    await safe_send({
                        "type": "model_loading",
                        "model": "rvc",
                        "message": "Loading RVC voice conversion...",
                        "progress": 90
                    })
                    try:
                        await loop.run_in_executor(None, state.preload_rvc)
                        print(f"   RVC loaded for {client_id}")
                        await safe_send({
                            "type": "model_loaded",
                            "model": "rvc",
                            "message": "RVC ready!",
                            "progress": 98
                        })
                    except Exception as e:
                        print(f"   RVC load warning: {e}")

            if is_connected():
                await safe_send({
                    "type": "models_ready",
                    "message": "All models loaded!",
                    "progress": 100
                })
                print(f"All models preloaded for {client_id}")

        except asyncio.CancelledError:
            print(f"Preloading cancelled for {client_id} (client disconnected)")
        except Exception as e:
            print(f"Preloading error for {client_id}: {e}")
            import traceback
            traceback.print_exc()
            if is_connected():
                await safe_send({
                    "type": "models_error",
                    "message": f"Failed to preload models: {str(e)}"
                })
        finally:
            self._preloading[client_id] = False

    async def preload_models(self, client_id: str):
        """Preload all models when user clicks mic."""
        if self._preloading.get(client_id, False):
            return

        state = self.states.get(client_id)
        if not state:
            return

        if state.mode == "omni":
            if state.omni_model and state.vad:
                await self.send_json(client_id, {
                    "type": "models_ready",
                    "message": "Models already loaded"
                })
                return
        elif state.mode == "gemma-omni":
            if state.gemma_model and state.vad:
                await self.send_json(client_id, {
                    "type": "models_ready",
                    "message": "Models already loaded"
                })
                return
        else:
            if state.vad and state.pipeline_ready():
                await self.send_json(client_id, {
                    "type": "models_ready",
                    "message": "Models already loaded"
                })
                return

        self._preloading[client_id] = True

        await self.send_json(client_id, {
            "type": "models_loading",
            "message": "Loading voice models..."
        })

        try:
            loop = asyncio.get_event_loop()

            if state.mode == "omni":
                async def load_omni():
                    if not state.omni_model:
                        await loop.run_in_executor(None, state.get_omni)

                async def load_vad():
                    if not state.vad:
                        await loop.run_in_executor(None, state.get_vad)

                await asyncio.gather(load_omni(), load_vad())

            elif state.mode == "gemma-omni":
                async def load_gemma():
                    if not state.gemma_model:
                        await loop.run_in_executor(None, state.get_gemma_omni)

                async def load_vad():
                    if not state.vad:
                        await loop.run_in_executor(None, state.get_vad)

                await asyncio.gather(load_gemma(), load_vad())

            else:
                async def load_vad():
                    if not state.vad:
                        await loop.run_in_executor(None, state.get_vad)

                async def load_asr():
                    if not state.asr:
                        await loop.run_in_executor(None, state.preload_asr)

                async def load_tts():
                    if not state.tts:
                        await loop.run_in_executor(None, state.preload_tts)

                async def load_llm():
                    if state.config.get("llm", {}).get("provider", "ollama") == "gemma":
                        await loop.run_in_executor(None, state.preload_llm)

                async def load_rvc():
                    if state.config.get("tts", {}).get("rvc", {}).get("enabled", False) and not state.rvc:
                        await loop.run_in_executor(None, state.preload_rvc)

                await asyncio.gather(load_vad(), load_asr(), load_tts(), load_llm(), load_rvc())

            await self.send_json(client_id, {
                "type": "models_ready",
                "message": "Voice models loaded!"
            })

        except Exception as e:
            print(f"Model preloading error: {e}")
            await self.send_json(client_id, {
                "type": "error",
                "message": f"Failed to load models: {str(e)}"
            })
        finally:
            self._preloading[client_id] = False


# Global manager instance
manager = WebSocketManager()


@websocket_router.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    """
    WebSocket endpoint for real-time conversation.

    Protocol:
    - Client sends: {"type": "text|audio|audio_stream|clear|duplex_audio", ...}
    - Server sends: {"type": "text_start|text_chunk|text_end|audio_data|mode_info|error", ...}
    """
    await manager.connect(websocket, client_id)

    try:
        while True:
            message = await websocket.receive()

            if message.get("type") == "websocket.disconnect":
                raise WebSocketDisconnect()

            if message.get("bytes") is not None:
                await manager.handle_audio_stream_bytes(client_id, message["bytes"])
                continue

            raw_text = message.get("text")
            if raw_text is None:
                continue

            data = json.loads(raw_text)
            msg_type = data.get("type", "text")

            if msg_type == "text":
                content = data.get("content", "")
                if content.strip():
                    state = manager.states.get(client_id)
                    default_language = getattr(state, "current_language", "en") if state else "en"
                    lang = normalize_language_code(
                        str(detect_text_language(content, default=default_language))
                    )
                    await manager.handle_text_message(client_id, content, language=lang)

            elif msg_type == "audio":
                audio_data = data.get("data", "")
                if audio_data:
                    await manager.handle_audio_message(client_id, audio_data)

            elif msg_type == "audio_segment":
                pcm16 = data.get("pcm16", "")
                if pcm16:
                    state = manager.states.get(client_id)
                    if state and state.vad:
                        state.vad.reset()

                    audio_bytes = base64.b64decode(pcm16)
                    if state and state.mode == "omni":
                        await manager._schedule_turn(client_id, manager._handle_audio_omni(client_id, audio_bytes))
                    elif state and state.mode == "gemma-omni":
                        await manager._schedule_turn(client_id, manager._handle_audio_gemma(client_id, audio_bytes))
                    else:
                        await manager._schedule_turn(client_id, manager._transcribe_and_respond_turn(client_id, audio_bytes))

            elif msg_type == "audio_stream":
                audio_samples = data.get("samples", [])
                if audio_samples:
                    await manager.handle_audio_stream(client_id, audio_samples)

            elif msg_type == "duplex_audio":
                # Full-duplex audio for omni/gemma-omni mode
                audio_samples = data.get("samples", [])
                if audio_samples:
                    state = manager.states.get(client_id)
                    if state and state.mode in ("omni", "gemma-omni"):
                        await manager.handle_audio_stream(client_id, audio_samples)

            elif msg_type == "mic_stop":
                state = manager.states.get(client_id)
                if state and state.vad:
                    audio_bytes = state.vad.force_end()
                    if audio_bytes:
                        if state.mode == "omni":
                            await manager._schedule_turn(client_id, manager._handle_audio_omni(client_id, audio_bytes))
                        elif state.mode == "gemma-omni":
                            await manager._schedule_turn(client_id, manager._handle_audio_gemma(client_id, audio_bytes))
                        else:
                            await manager._schedule_turn(client_id, manager._transcribe_and_respond_turn(client_id, audio_bytes))

            elif msg_type == "interrupt":
                await manager.handle_interrupt(client_id)

            elif msg_type == "clear":
                await manager.handle_clear(client_id)

            elif msg_type == "ping":
                await manager.send_json(client_id, {"type": "pong"})

            elif msg_type == "preload_models":
                await manager.preload_models(client_id)

    except WebSocketDisconnect:
        await manager.disconnect(client_id)
    except Exception as e:
        import traceback
        print(f"WebSocket error for {client_id}: {type(e).__name__}: {e}")
        traceback.print_exc()
        await manager.disconnect(client_id)
