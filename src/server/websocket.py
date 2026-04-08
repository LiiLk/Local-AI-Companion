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
import numpy as np
from langdetect import detect, LangDetectException
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass, field

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import yaml

from src.llm import OllamaLLM, GemmaTextVisionLLM
from src.llm.base import Message
from src.tts import KokoroProvider, EdgeTTSProvider, ChatterboxTTSProvider, Qwen3TTSProvider
from src.asr import WhisperProvider
from src.vad import SileroVAD
from src.utils.audio_analysis import analyze_audio_volumes, read_wav_pcm, calculate_audio_duration_ms
from src.utils.character_loader import resolve_character_config
from src.utils.emotion_detector import EmotionDetector, strip_emotion_markers
from src.utils.rvc_config import build_rvc_runtime_config

websocket_router = APIRouter()


def load_config() -> dict:
    """Load configuration from config.yaml"""
    config_path = Path(__file__).parent.parent.parent / "config" / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


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
    current_language: str = "fr"
    emotion_detector: Optional[EmotionDetector] = None
    current_expression: str = "neutral"
    mode: str = "pipeline"  # "pipeline", "omni", or "gemma-omni"
    omni_model: Optional[Any] = None
    omni_pipeline: Optional[Any] = None
    gemma_model: Optional[Any] = None
    gemma_pipeline: Optional[Any] = None
    rvc: Optional[Any] = None

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
            # Pipeline mode: initialize LLM
            if self.llm is None:
                llm_config = self.config.get("llm", {})
                llm_provider = llm_config.get("provider", "ollama")
                if llm_provider == "gemma":
                    from src.omni import GemmaProvider

                    gemma_config = self.config.get("gemma", {})
                    gemma_model = GemmaProvider(
                        model_id=gemma_config.get("model_id", "google/gemma-4-E2B-it"),
                        device=gemma_config.get("device", "cuda"),
                        quantization=gemma_config.get("quantization", "int4"),
                        max_new_tokens=gemma_config.get("max_new_tokens", 96),
                        temperature=gemma_config.get("temperature", 0.7),
                        top_p=gemma_config.get("top_p", 0.95),
                        context_max_turns=gemma_config.get("context_max_turns", 10),
                    )
                    self.llm = GemmaTextVisionLLM(
                        gemma=gemma_model,
                        screen_config=gemma_config.get("screen", {}),
                    )
                else:
                    ollama_config = llm_config.get("ollama", {})
                    self.llm = OllamaLLM(
                        model=ollama_config.get("model", "llama3.2:3b"),
                        base_url=ollama_config.get("base_url", "http://localhost:11434")
                    )

        # Initialize conversation with system prompt
        if not self.messages:
            character = self.config.get("character", {})
            system_prompt = character.get("system_prompt", "You are a helpful assistant.")
            self.messages.append(Message(role="system", content=system_prompt))

    def get_tts(self):
        """Get or create TTS provider (lazy loading). Pipeline mode only."""
        if self.tts is None:
            tts_config = self.config.get("tts", {})
            provider = tts_config.get("provider", "kokoro")
            auto_detect = tts_config.get("auto_detect_language", False)

            print(f"Loading TTS provider: {provider} (auto_detect={auto_detect})")

            if provider == "kokoro":
                voice = tts_config.get("kokoro_voice", "ff_siwis")
                print(f"   Kokoro config: voice={voice}")
                self.tts = KokoroProvider(voice=voice)
            elif provider == "qwen3":
                voice_config = self.config.get("character", {}).get("voice", {})
                qwen3_config = tts_config.get("qwen3", {})
                ref_audio = (
                    voice_config.get("qwen_ref_audio")
                    or voice_config.get("chatterbox_ref_audio")
                    or voice_config.get("omni_ref_audio")
                    or qwen3_config.get("ref_audio_path")
                )
                ref_text = voice_config.get("qwen_ref_text") or qwen3_config.get("ref_text")
                print(
                    f"   Qwen3-TTS config: model={qwen3_config.get('model_id', 'Qwen/Qwen3-TTS-12Hz-0.6B-Base')}, "
                    f"ref={ref_audio}"
                )
                try:
                    if not Qwen3TTSProvider.is_available(
                        backend=qwen3_config.get("backend", "worker"),
                        python_path=qwen3_config.get("python_path"),
                        site_packages_dir=qwen3_config.get("site_packages_dir"),
                        worker_script=qwen3_config.get("worker_script"),
                    ):
                        raise RuntimeError(
                            "Qwen3-TTS runtime is not installed. "
                            "Run scripts/install_qwen3_tts_windows.ps1 first."
                        )
                    self.tts = Qwen3TTSProvider(
                        model_id=qwen3_config.get("model_id", "Qwen/Qwen3-TTS-12Hz-0.6B-Base"),
                        mode=qwen3_config.get("mode", "voice_clone"),
                        language=qwen3_config.get("language", "auto"),
                        speaker=qwen3_config.get("speaker"),
                        instruct=qwen3_config.get("instruct"),
                        ref_audio_path=ref_audio,
                        ref_text=ref_text,
                        x_vector_only_mode=qwen3_config.get("x_vector_only_mode"),
                        device=qwen3_config.get("device", "cuda:0"),
                        dtype=qwen3_config.get("dtype", "bfloat16"),
                        attn_implementation=qwen3_config.get("attn_implementation", "flash_attention_2"),
                        backend=qwen3_config.get("backend", "worker"),
                        python_path=qwen3_config.get("python_path"),
                        site_packages_dir=qwen3_config.get("site_packages_dir"),
                        worker_script=qwen3_config.get("worker_script"),
                    )
                except Exception as exc:
                    print(f"   Qwen3-TTS init failed, falling back to Kokoro: {exc}")
                    voice = tts_config.get("kokoro_voice", "ff_siwis")
                    self.tts = KokoroProvider(voice=voice)
            elif provider == "chatterbox":
                voice_config = self.config.get("character", {}).get("voice", {})
                chatterbox_config = tts_config.get("chatterbox", {})
                ref_audio = voice_config.get("chatterbox_ref_audio")
                exaggeration = voice_config.get("chatterbox_exaggeration", 0.5)
                language = voice_config.get("chatterbox_language", "fr")
                print(f"   Chatterbox config: ref={ref_audio}, language={language}")
                self.tts = ChatterboxTTSProvider(
                    model_id=chatterbox_config.get("model_id", "onnx-community/chatterbox-multilingual-ONNX"),
                    ref_audio_path=ref_audio,
                    exaggeration=exaggeration,
                    cfg_weight=chatterbox_config.get("cfg_weight", 0.5),
                    language=language,
                    prefer_full_gpu=chatterbox_config.get("prefer_full_gpu", True),
                )
            else:
                voice = tts_config.get("voice", "fr-FR-DeniseNeural")
                print(f"   Edge TTS config: voice={voice}")
                self.tts = EdgeTTSProvider(voice=voice)

        return self.tts

    def get_rvc(self):
        """Get or create RVC voice converter (lazy loading). Returns None if disabled."""
        if self.rvc is None:
            rvc_config = build_rvc_runtime_config(self.config)
            if not rvc_config:
                return None
            try:
                from src.tts.rvc_provider import RVCConverter
                if not RVCConverter.is_available(
                    backend=rvc_config.get("backend", "auto"),
                    python_path=rvc_config.get("python_path"),
                    site_packages_dir=rvc_config.get("site_packages_dir"),
                    worker_script=rvc_config.get("worker_script"),
                ):
                    print("   RVC dependencies not installed, skipping voice conversion")
                    return None
                self.rvc = RVCConverter(
                    model_path=rvc_config.get("model_path"),
                    index_path=rvc_config.get("index_path"),
                    device=rvc_config.get("device", "cuda:0"),
                    f0_method=rvc_config.get("f0_method", "rmvpe"),
                    index_rate=rvc_config.get("index_rate", 0.75),
                    protect=rvc_config.get("protect", 0.33),
                    backend=rvc_config.get("backend", "auto"),
                    python_path=rvc_config.get("python_path"),
                    site_packages_dir=rvc_config.get("site_packages_dir"),
                    worker_script=rvc_config.get("worker_script"),
                    f0_up_key=rvc_config.get("f0_up_key", 0.0),
                    output_freq=rvc_config.get("output_freq"),
                )
                print(f"   RVC loaded: {rvc_config.get('model_path')}")
            except Exception as e:
                print(f"   RVC init error (voice conversion disabled): {e}")
                return None
        return self.rvc

    def get_asr(self):
        """Get or create ASR provider (lazy loading). Pipeline mode only."""
        if self.asr is None:
            asr_config = self.config.get("asr", {})
            device = asr_config.get("device", "cpu")

            model_size = asr_config.get("model_size", "base")
            initial_prompt = asr_config.get("prompt", None)
            self.asr = WhisperProvider(
                model_size=model_size,
                device=device,
                initial_prompt=initial_prompt
            )
            self.asr_language = asr_config.get("language", "")

        return self.asr

    def preload_rvc(self):
        """Preload RVC backend/model when enabled."""
        rvc = self.get_rvc()
        if rvc and hasattr(rvc, "preload"):
            rvc.preload()
        if rvc and hasattr(rvc, "warmup"):
            rvc.warmup()
        return rvc

    def get_vad(self):
        """Get or create VAD engine (lazy loading)."""
        if self.vad is None:
            from src.vad.silero_vad import VADConfig

            llm_provider = self.config.get("llm", {}).get("provider", "ollama")
            gemma_config = self.config.get("gemma", {}) if self.mode == "gemma-omni" or (self.mode == "pipeline" and llm_provider == "gemma") else {}
            vad_config = VADConfig(
                sample_rate=16000,
                prob_threshold=gemma_config.get("vad_prob_threshold", 0.5),
                db_threshold=gemma_config.get("vad_db_threshold", -50),
                required_hits=gemma_config.get("vad_required_hits", 3),
                required_misses=gemma_config.get("vad_required_misses", 30),
            )
            self.vad = SileroVAD(config=vad_config)
        return self.vad

    def preload_llm(self):
        """Preload the configured pipeline LLM when it supports it."""
        if self.llm is None:
            return None
        preload = getattr(self.llm, "preload", None)
        if callable(preload):
            preload()
        return self.llm

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
            language = voice_config.get("chatterbox_language", "fr")

            print("Loading Gemma E2B + Chatterbox...")
            self.gemma_model = GemmaProvider(
                model_id=gemma_config.get("model_id", "google/gemma-4-E2B-it"),
                device=gemma_config.get("device", "cuda"),
                quantization=gemma_config.get("quantization", "int4"),
                max_new_tokens=gemma_config.get("max_new_tokens", 256),
                temperature=gemma_config.get("temperature", 0.7),
                top_p=gemma_config.get("top_p", 0.95),
                context_max_turns=gemma_config.get("context_max_turns", 10),
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
        if self.llm:
            await self.llm.close()
        if self.rvc and hasattr(self.rvc, "close"):
            self.rvc.close()


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

    async def _update_voice_for_language(self, state: ConversationState, text: str):
        """Detect language and update TTS voice if needed."""
        if not text.strip() or len(text) < 5:
            return

        try:
            lang = detect(text)

            if lang in ['fr', 'fr-fr']:
                lang_code = 'fr'
            elif lang in ['en', 'en-us', 'en-gb']:
                lang_code = 'en'
            else:
                return

            if state.current_language != lang_code:
                print(f"Language switch detected: {state.current_language} -> {lang_code}")
                state.current_language = lang_code

                tts_config = state.config.get("tts", {})
                provider_name = tts_config.get("provider", "kokoro")
                voice_mapping = tts_config.get("voice_mapping", {}).get(provider_name, {})

                new_voice = voice_mapping.get(lang_code)
                if new_voice and state.tts:
                    if hasattr(state.tts, 'set_voice'):
                        state.tts.set_voice(new_voice)
                        print(f"   Switched {provider_name} voice to: {new_voice}")

        except LangDetectException:
            pass
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

            if asyncio.iscoroutinefunction(tts.synthesize):
                await tts.synthesize(text, temp_path)
            else:
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, tts.synthesize, text, temp_path)

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

    async def handle_text_message(self, client_id: str, content: str, language: str | None = None):
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
        state.messages.append(Message(role="user", content=content))

        await self.send_json(client_id, {"type": "text_start"})
        await self.send_json(client_id, {"type": "audio_start"})

        full_response = ""
        current_sentence = ""

        llm_messages = list(state.messages)

        if language:
            lang_map = {"fr": "French", "en": "English", "es": "Spanish", "de": "German", "it": "Italian", "ja": "Japanese"}
            lang_name = lang_map.get(language, language)

            if llm_messages and llm_messages[-1].role == "user":
                last_msg = llm_messages[-1]
                new_content = f"(System: The user is speaking {lang_name}. Reply in {lang_name}.)\n\n{last_msg.content}"
                llm_messages[-1] = Message(role="user", content=new_content)
                print(f"Enforcing language: {lang_name}")

        async for chunk in state.llm.chat_stream(llm_messages):
            full_response += chunk
            current_sentence += chunk

            await self.send_json(client_id, {
                "type": "text_chunk",
                "content": chunk
            })

            if any(punct in chunk for punct in ".!?\n"):
                import re
                parts = re.split(r'([.!?\n]+)', current_sentence)

                if len(parts) > 1:
                    for i in range(0, len(parts) - 1, 2):
                        sentence = parts[i] + parts[i+1]
                        if sentence.strip():
                            await self._process_tts_chunk(client_id, sentence)

                    current_sentence = parts[-1]

        if current_sentence.strip():
            await self._process_tts_chunk(client_id, current_sentence)

        state.messages.append(Message(role="assistant", content=full_response))

        await self.send_json(client_id, {
            "type": "text_end",
            "full_text": full_response
        })
        await self.send_json(client_id, {"type": "audio_end"})

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

            await tts.synthesize(text, temp_path)

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

    async def handle_audio_message(self, client_id: str, audio_data: str):
        """Handle audio data from the client (WebM blob)."""
        state = self._get_state(client_id)
        if not state:
            return

        asr = state.get_asr()

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
            language = getattr(state, 'asr_language', 'fr')
            result = asr.transcribe(wav_path, language=language)

            wav_path.unlink(missing_ok=True)

            if result.text.strip():
                await self.send_json(client_id, {
                    "type": "transcription",
                    "text": result.text,
                    "language": result.language
                })
                await self.handle_text_message(client_id, result.text, language=result.language)
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
                        asyncio.create_task(self._handle_audio_omni(client_id, event))
                    elif state.mode == "gemma-omni":
                        asyncio.create_task(self._handle_audio_gemma(client_id, event))
                    else:
                        asyncio.create_task(self._transcribe_and_respond(client_id, event))

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

    async def _transcribe_and_respond(self, client_id: str, audio_bytes: bytes):
        """Transcribe audio bytes and generate response. Pipeline mode only."""
        state = self._get_state(client_id)
        if not state:
            return

        asr = state.get_asr()

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
            language = getattr(state, 'asr_language', 'fr')

            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: asr.transcribe(audio_float, language=language)
            )

            if result.text.strip():
                await self.send_json(client_id, {
                    "type": "transcription",
                    "text": result.text,
                    "language": result.language
                })
                await self.handle_text_message(client_id, result.text, language=result.language)
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

        character = state.config.get("character", {})
        system_prompt = character.get("system_prompt", "You are a helpful assistant.")
        state.messages = [Message(role="system", content=system_prompt)]

        if state.omni_pipeline:
            state.omni_pipeline.clear_history()

        if state.gemma_pipeline:
            state.gemma_pipeline.clear_history()

        await self.send_json(client_id, {
            "type": "cleared",
            "message": "Conversation history cleared"
        })

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
                        def _preload_tts():
                            tts = state.get_tts()
                            # Preload ONNX model so first synthesis is fast
                            if hasattr(tts, '_load_model'):
                                try:
                                    tts._load_model()
                                except Exception as exc:
                                    if tts.__class__.__name__ == "Qwen3TTSProvider":
                                        fallback_voice = state.config.get("tts", {}).get("kokoro_voice", "ff_siwis")
                                        print(f"   Qwen3-TTS preload failed, falling back to Kokoro: {exc}")
                                        state.tts = KokoroProvider(voice=fallback_voice)
                                    else:
                                        raise
                        await loop.run_in_executor(None, _preload_tts)
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
                        await loop.run_in_executor(None, state.get_asr)
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
            llm_ready = True
            if state.config.get("llm", {}).get("provider", "ollama") == "gemma":
                llm_ready = bool(getattr(getattr(state.llm, "gemma", None), "_model", None))
            rvc_enabled = state.config.get("tts", {}).get("rvc", {}).get("enabled", False)
            rvc_ready = state.rvc is not None or not rvc_enabled
            if state.vad and state.asr and state.tts and llm_ready and rvc_ready:
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
                        await loop.run_in_executor(None, state.get_asr)

                async def load_tts():
                    if not state.tts:
                        await loop.run_in_executor(None, state.get_tts)

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
                    lang = None
                    try:
                        from langdetect import detect
                        detected = detect(content)
                        if detected.startswith('fr'): lang = 'fr'
                        elif detected.startswith('en'): lang = 'en'
                    except:
                        pass

                    await manager.handle_text_message(client_id, content, language=lang)

            elif msg_type == "audio":
                audio_data = data.get("data", "")
                if audio_data:
                    await manager.handle_audio_message(client_id, audio_data)

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
                            await manager._handle_audio_omni(client_id, audio_bytes)
                        elif state.mode == "gemma-omni":
                            await manager._handle_audio_gemma(client_id, audio_bytes)
                        else:
                            await manager._transcribe_and_respond(client_id, audio_bytes)

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
