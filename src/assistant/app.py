"""
Live2D Assistant Application - Main entry point.

This is the unified application that combines:
- Live2D avatar overlay
- Continuous microphone capture with VAD
- ASR → LLM → TTS conversation pipeline
- Hotkey controls

Usage:
    python -m src.assistant.app
    
Hotkeys:
    F2: Toggle mute/unmute microphone
    F3: Interrupt current response
    F12: Toggle overlay visibility
    Escape: Quit
"""

# Disable Hub telemetry globally, but do not force offline mode here.
# Some providers (Qwen3-TTS, Chatterbox, Gemma first-run downloads) still need
# normal Hugging Face access unless they explicitly opt into offline mode.
import os
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

import argparse
import asyncio
import base64
from concurrent.futures import CancelledError as FutureCancelledError
from contextvars import ContextVar
import contextlib
import json
import logging
import signal
import time
import sys
import threading
from pathlib import Path
from typing import Optional

import yaml

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.assistant.audio_service import AudioService, AudioServiceConfig, MicState
from src.assistant.conversation_pipeline import ConversationPipeline, ConversationConfig, AudioPayload
from src.utils.character_loader import resolve_character_config
from src.utils.rvc_config import build_rvc_runtime_config

# Conditional imports
try:
    from pynput import keyboard
    PYNPUT_AVAILABLE = True
except ImportError:
    PYNPUT_AVAILABLE = False
    print("⚠️ pynput not installed. Hotkeys disabled. Run: pip install pynput")

try:
    import webview
    WEBVIEW_AVAILABLE = True
except ImportError:
    WEBVIEW_AVAILABLE = False
    print("⚠️ pywebview not installed. Run: pip install pywebview")

try:
    from websockets.asyncio.server import serve
    from websockets.exceptions import ConnectionClosed

    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False
    ConnectionClosed = Exception  # type: ignore[assignment]
    print("⚠️ websockets not installed. Bridge server disabled. Run: pip install websockets")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
CURRENT_DESKTOP_TURN_ID: ContextVar[Optional[int]] = ContextVar("desktop_turn_id", default=None)


def load_config(config_path: Path) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


class DesktopBridgeApi:
    """Minimal pywebview bridge for the desktop Live2D shell."""

    def __init__(self, assistant: "Live2DAssistant"):
        self._assistant = assistant

    def send_text(self, text: str) -> dict:
        return self._assistant.submit_text(text)

    def interrupt(self) -> dict:
        return self._assistant.request_interrupt("ui")

    def toggle_mute(self) -> dict:
        return self._assistant.toggle_mute()

    def get_runtime_state(self) -> dict:
        return self._assistant.get_runtime_state()

    def toggle_debug(self) -> dict:
        return self._assistant.toggle_debug()


class DesktopBridgeServer:
    """Tiny desktop-only websocket bridge for the Tauri shell."""

    def __init__(self, assistant: "Live2DAssistant", host: str = "127.0.0.1", port: int = 8765):
        self._assistant = assistant
        self._host = host
        self._port = port
        self._server = None
        self._clients = set()

    async def start(self) -> None:
        if not WEBSOCKETS_AVAILABLE:
            raise RuntimeError("websockets is required for bridge-server mode")
        if self._server is not None:
            return

        self._server = await serve(self._handle_client, self._host, self._port, max_size=8 * 1024 * 1024)
        logger.info("✅ Desktop bridge listening on ws://%s:%s", self._host, self._port)

    async def stop(self) -> None:
        clients = list(self._clients)
        self._clients.clear()

        for client in clients:
            with contextlib.suppress(Exception):
                await client.close()

        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None

    def emit_frontend_event_sync(self, event_name: str, *args) -> None:
        loop = self._assistant._loop
        if not loop or loop.is_closed():
            return
        asyncio.run_coroutine_threadsafe(self._broadcast_event(event_name, args), loop)

    async def _handle_client(self, websocket) -> None:
        self._clients.add(websocket)
        try:
            await self._send_backend_ready(websocket)
            async for raw_message in websocket:
                await self._handle_message(websocket, raw_message)
        except ConnectionClosed:
            pass
        finally:
            self._clients.discard(websocket)

    async def _handle_message(self, websocket, raw_message: str) -> None:
        try:
            message = json.loads(raw_message)
        except json.JSONDecodeError:
            await self._send_json(websocket, {"type": "error", "message": "Invalid JSON payload"})
            return

        if message.get("type") != "command":
            await self._send_json(websocket, {"type": "error", "message": "Unsupported bridge message"})
            return

        name = str(message.get("name") or "")
        request_id = message.get("request_id")

        try:
            if name == "send_text":
                result = self._assistant.submit_text(str(message.get("text") or ""))
            elif name == "interrupt":
                result = self._assistant.request_interrupt("tauri")
            elif name == "toggle_mute":
                result = self._assistant.toggle_mute()
            elif name == "get_runtime_state":
                result = self._assistant.get_runtime_state()
            elif name == "toggle_debug":
                result = self._assistant.toggle_debug()
            else:
                raise ValueError(f"Unknown bridge command: {name}")
        except Exception as exc:
            await self._send_json(
                websocket,
                {
                    "type": "command_result",
                    "request_id": request_id,
                    "name": name,
                    "ok": False,
                    "error": str(exc),
                },
            )
            return

        await self._send_json(
            websocket,
            {
                "type": "command_result",
                "request_id": request_id,
                "name": name,
                "ok": True,
                "result": self._with_backend(result),
            },
        )

    async def _send_backend_ready(self, websocket) -> None:
        await self._send_json(
            websocket,
            {
                "type": "backend_ready",
                "runtime": self._with_backend(self._assistant.get_runtime_state()),
            },
        )

    async def _broadcast_event(self, event_name: str, args: tuple) -> None:
        if not self._clients:
            return

        payload = {
            "type": "frontend_event",
            "name": event_name,
            "args": list(args),
            "runtime": self._with_backend(self._assistant.get_runtime_state()),
        }
        disconnected = []
        for client in list(self._clients):
            try:
                await self._send_json(client, payload)
            except Exception:
                disconnected.append(client)

        for client in disconnected:
            self._clients.discard(client)

    async def _send_json(self, websocket, payload: dict) -> None:
        await websocket.send(json.dumps(payload, ensure_ascii=False))

    @staticmethod
    def _with_backend(runtime: dict) -> dict:
        return {**runtime, "backend": "assistant-bridge"}


class Live2DAssistant:
    """
    Main application class that orchestrates all components.
    
    Architecture:
    - AudioService: Captures mic, runs VAD, yields speech segments
    - ConversationPipeline: ASR → LLM → TTS processing
    - Live2D Overlay: WebView window with Live2D model
    - Hotkey Listener: Global keyboard shortcuts
    """
    
    def __init__(self, config_path: Optional[Path] = None):
        # Load config
        self.config_path = config_path or PROJECT_ROOT / "config" / "config.yaml"
        self.config = load_config(self.config_path)
        self.config = resolve_character_config(self.config)

        # State
        self._running = False
        self._window: Optional[webview.Window] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._loop_thread: Optional[threading.Thread] = None
        self._hotkey_listener = None
        self._preload_thread: Optional[threading.Thread] = None
        self._bridge_server: Optional[DesktopBridgeServer] = None
        self._bridge_only = False
        self._active_response_future = None
        self._turn_timeout_sec = int(self.config.get('gemma', {}).get('turn_timeout_sec', 75))
        self._turn_counter = 0
        self._active_turn_id: Optional[int] = None
        self._latest_audio_turn_id: Optional[int] = None
        self._playback_deadline: float = 0.0
        self._debug_visible = False
        self._js_api = DesktopBridgeApi(self)
        
        # Components (initialized in start())
        self.audio_service: Optional[AudioService] = None
        self.pipeline: Optional[ConversationPipeline] = None
        
        # Current audio playback state
        self._current_volumes: list[float] = []
        self._volume_index: int = 0
        self._lip_sync_task: Optional[asyncio.Task] = None
        
        logger.info("Live2DAssistant created")
    
    def _create_components(self):
        """Create ASR, LLM, TTS, and pipeline components."""
        character_config = self.config.get('character', {})
        mode = self.config.get('mode', 'pipeline')

        logger.info("=" * 50)
        logger.info("STACK CONFIGURATION")
        logger.info("=" * 50)
        logger.info(f"Mode: {mode}")

        if mode == "omni":
            from src.omni import MiniCPMoProvider, OmniPipeline

            omni_config = self.config.get('omni', {}).get('minicpmo', {})

            # Resolve ref_audio_path from character preset or omni config
            voice_config = character_config.get('voice', {})
            ref_audio = (
                voice_config.get('omni_ref_audio')
                or omni_config.get('ref_audio_path')
            )

            self._omni_model = MiniCPMoProvider(
                model_name=omni_config.get('model_id', 'openbmb/MiniCPM-o-4_5'),
                device=omni_config.get('device', 'cuda'),
                dtype=omni_config.get('dtype', 'bfloat16'),
                quantization=omni_config.get('quantization'),
                attn_implementation=omni_config.get('attn_implementation', 'sdpa'),
                ref_audio_path=ref_audio,
                init_vision=omni_config.get('init_vision', True),
                init_audio=omni_config.get('init_audio', True),
                init_tts=omni_config.get('init_tts', True),
            )
            logger.info(f"Omni: MiniCPM-o ({omni_config.get('model_id')}, q={omni_config.get('quantization', 'none')})")

            pipeline_config = ConversationConfig(
                character_name=character_config.get('name', 'AI'),
                system_prompt=character_config.get('system_prompt', 'You are a helpful assistant.'),
                stream_tts=omni_config.get('stream_tts', False),
                omni_use_single_pass=omni_config.get('use_single_pass', True),
                omni_transcribe_for_history=omni_config.get('transcribe_for_history', False),
                omni_generate_audio=omni_config.get('generate_audio', True),
                omni_max_tokens=omni_config.get('max_tokens', 512),
                omni_temperature=omni_config.get('temperature', 0.7),
            )
            self._omni_pipeline = OmniPipeline(
                omni=self._omni_model,
                config=pipeline_config,
            )

            # Wire callbacks (same interface as ConversationPipeline)
            self._omni_pipeline.on_transcription = self._on_transcription
            self._omni_pipeline.on_response_start = self._on_response_start
            self._omni_pipeline.on_response_chunk = self._on_response_chunk
            self._omni_pipeline.on_response_end = self._on_response_end
            self._omni_pipeline.on_audio_ready = self._on_audio_ready
            self._omni_pipeline.on_expression_change = self._on_expression_change
            self._omni_pipeline.on_error = self._on_error

            # Pipeline is not used in omni mode
            self.pipeline = None

        elif mode == "gemma-omni":
            from src.omni import GemmaProvider, GemmaOmniPipeline
            from src.tts import ChatterboxTTSProvider

            gemma_config = self.config.get('gemma', {})
            tts_config = self.config.get('tts', {})
            chatterbox_config = tts_config.get('chatterbox', {})

            # Resolve voice config from character preset
            voice_config = character_config.get('voice', {})
            ref_audio = voice_config.get('chatterbox_ref_audio')
            exaggeration = voice_config.get('chatterbox_exaggeration', 0.5)
            language = voice_config.get('chatterbox_language', 'fr')

            # Create Gemma provider
            self._gemma_model = GemmaProvider(
                model_id=gemma_config.get('model_id', 'google/gemma-4-E4B-it'),
                device=gemma_config.get('device', 'cuda'),
                quantization=gemma_config.get('quantization', 'int4'),
                max_new_tokens=gemma_config.get('max_new_tokens', 256),
                temperature=gemma_config.get('temperature', 0.7),
                top_p=gemma_config.get('top_p', 0.95),
                context_max_turns=gemma_config.get('context_max_turns', 10),
                cpu_offload=gemma_config.get('cpu_offload', True),
                offload_dir=gemma_config.get('offload_dir'),
            )
            logger.info(f"Gemma: {gemma_config.get('model_id')} (q={gemma_config.get('quantization', 'int4')})")

            # Create Chatterbox TTS provider
            self._chatterbox_tts = ChatterboxTTSProvider(
                model_id=chatterbox_config.get('model_id', 'onnx-community/chatterbox-multilingual-ONNX'),
                ref_audio_path=ref_audio,
                exaggeration=exaggeration,
                cfg_weight=chatterbox_config.get('cfg_weight', 0.5),
                language=language,
                prefer_full_gpu=chatterbox_config.get('prefer_full_gpu', True),
            )
            logger.info(f"TTS: Chatterbox (ref={ref_audio}, exag={exaggeration})")

            # Create pipeline
            pipeline_config = ConversationConfig(
                character_name=character_config.get('name', 'AI'),
                system_prompt=character_config.get('system_prompt', 'You are a helpful assistant.'),
                stream_tts=tts_config.get('stream_tts', True),
            )
            self._gemma_pipeline = GemmaOmniPipeline(
                gemma=self._gemma_model,
                tts=self._chatterbox_tts,
                config=pipeline_config,
            )

            # Enable screen capture if configured
            screen_config = gemma_config.get('screen', {})
            if screen_config.get('enabled', False):
                self._gemma_pipeline.enable_screen_capture(screen_config)

            # Wire callbacks (same interface as other pipelines)
            self._gemma_pipeline.on_transcription = self._on_transcription
            self._gemma_pipeline.on_response_start = self._on_response_start
            self._gemma_pipeline.on_response_chunk = self._on_response_chunk
            self._gemma_pipeline.on_response_end = self._on_response_end
            self._gemma_pipeline.on_audio_ready = self._on_audio_ready
            self._gemma_pipeline.on_expression_change = self._on_expression_change
            self._gemma_pipeline.on_error = self._on_error

            self.pipeline = None

        else:
            from src.llm import OllamaLLM, GemmaTextVisionLLM
            from src.tts import KokoroProvider, EdgeTTSProvider, ChatterboxTTSProvider, Qwen3TTSProvider
            from src.asr import WhisperProvider
            from src.omni import GemmaProvider
            from src.tts.rvc_provider import RVCConverter

            llm_config = self.config.get('llm', {})
            tts_config = self.config.get('tts', {})
            asr_config = self.config.get('asr', {})
            gemma_config = self.config.get('gemma', {})
            voice_config = character_config.get('voice', {})

            # Create LLM
            llm_provider = llm_config.get('provider', 'ollama')
            if llm_provider == 'gemma':
                gemma_model = GemmaProvider(
                    model_id=gemma_config.get('model_id', 'google/gemma-4-E4B-it'),
                    device=gemma_config.get('device', 'cuda'),
                    quantization=gemma_config.get('quantization', 'int4'),
                    max_new_tokens=gemma_config.get('max_new_tokens', 96),
                    temperature=gemma_config.get('temperature', 0.7),
                    top_p=gemma_config.get('top_p', 0.95),
                    context_max_turns=gemma_config.get('context_max_turns', 10),
                    cpu_offload=gemma_config.get('cpu_offload', True),
                    offload_dir=gemma_config.get('offload_dir'),
                )
                llm = GemmaTextVisionLLM(
                    gemma=gemma_model,
                    screen_config=gemma_config.get('screen', {}),
                )
                logger.info(
                    "LLM: Gemma text+vision (%s)",
                    gemma_config.get('model_id', 'google/gemma-4-E4B-it'),
                )
            else:
                ollama_cfg = llm_config.get('ollama', {})
                llm = OllamaLLM(
                    model=ollama_cfg.get('model', 'llama3.2:3b'),
                    base_url=ollama_cfg.get('base_url', 'http://localhost:11434')
                )
                logger.info(f"LLM: Ollama ({ollama_cfg.get('model')})")

            # Create TTS
            tts_provider = tts_config.get('provider', 'kokoro')
            if tts_provider == 'kokoro':
                voice = voice_config.get('kokoro_voice') or tts_config.get('kokoro_voice', 'ff_siwis')
                tts = KokoroProvider(voice=voice)
                logger.info(f"TTS: Kokoro ({voice})")
            elif tts_provider == 'qwen3':
                qwen3_config = tts_config.get('qwen3', {})
                ref_audio = (
                    voice_config.get('qwen_ref_audio')
                    or voice_config.get('chatterbox_ref_audio')
                    or voice_config.get('omni_ref_audio')
                    or qwen3_config.get('ref_audio_path')
                )
                ref_text = voice_config.get('qwen_ref_text') or qwen3_config.get('ref_text')
                try:
                    if not Qwen3TTSProvider.is_available(
                        backend=qwen3_config.get('backend', 'worker'),
                        python_path=qwen3_config.get('python_path'),
                        site_packages_dir=qwen3_config.get('site_packages_dir'),
                        worker_script=qwen3_config.get('worker_script'),
                    ):
                        raise RuntimeError(
                            "Qwen3-TTS runtime is not installed. "
                            "Run scripts/install_qwen3_tts_windows.ps1 first."
                        )
                    tts = Qwen3TTSProvider(
                        model_id=qwen3_config.get('model_id', 'Qwen/Qwen3-TTS-12Hz-0.6B-Base'),
                        mode=qwen3_config.get('mode', 'voice_clone'),
                        language=qwen3_config.get('language', 'auto'),
                        speaker=qwen3_config.get('speaker'),
                        instruct=qwen3_config.get('instruct'),
                        ref_audio_path=ref_audio,
                        ref_text=ref_text,
                        x_vector_only_mode=qwen3_config.get('x_vector_only_mode'),
                        device=qwen3_config.get('device', 'cuda:0'),
                        dtype=qwen3_config.get('dtype', 'bfloat16'),
                        attn_implementation=qwen3_config.get('attn_implementation', 'flash_attention_2'),
                        backend=qwen3_config.get('backend', 'worker'),
                        python_path=qwen3_config.get('python_path'),
                        site_packages_dir=qwen3_config.get('site_packages_dir'),
                        worker_script=qwen3_config.get('worker_script'),
                    )
                    logger.info(
                        "TTS: Qwen3-TTS (%s, mode=%s, ref=%s)",
                        qwen3_config.get('model_id', 'Qwen/Qwen3-TTS-12Hz-0.6B-Base'),
                        qwen3_config.get('mode', 'voice_clone'),
                        ref_audio,
                    )
                except Exception as exc:
                    logger.warning("Qwen3-TTS init failed, falling back to Kokoro: %s", exc)
                    voice = tts_config.get('kokoro_voice', 'ff_siwis')
                    tts = KokoroProvider(voice=voice)
                    logger.info(f"TTS fallback: Kokoro ({voice})")
            elif tts_provider == 'chatterbox':
                chatterbox_config = tts_config.get('chatterbox', {})
                ref_audio = voice_config.get('chatterbox_ref_audio')
                exaggeration = voice_config.get('chatterbox_exaggeration', 0.5)
                language = voice_config.get('chatterbox_language', 'fr')
                tts = ChatterboxTTSProvider(
                    model_id=chatterbox_config.get('model_id', 'onnx-community/chatterbox-multilingual-ONNX'),
                    ref_audio_path=ref_audio,
                    exaggeration=exaggeration,
                    cfg_weight=chatterbox_config.get('cfg_weight', 0.5),
                    language=language,
                    prefer_full_gpu=chatterbox_config.get('prefer_full_gpu', False),
                )
                logger.info(f"TTS: Chatterbox (ref={ref_audio}, exag={exaggeration})")
            else:
                voice = tts_config.get('voice', 'en-US-JennyNeural')
                tts = EdgeTTSProvider(voice=voice)
                logger.info(f"TTS: Edge ({voice})")

            # Create ASR
            device = asr_config.get('device', 'cpu')
            model_size = asr_config.get('model_size', 'base')
            prompt = asr_config.get('prompt')
            asr = WhisperProvider(
                model_size=model_size,
                device=device,
                initial_prompt=prompt
            )
            logger.info(f"ASR: Whisper {model_size} on {device}")

            # Create optional RVC post-processor
            rvc = None
            rvc_config = build_rvc_runtime_config(self.config)
            if rvc_config:
                if RVCConverter.is_available(
                    backend=rvc_config.get('backend', 'auto'),
                    python_path=rvc_config.get('python_path'),
                    site_packages_dir=rvc_config.get('site_packages_dir'),
                    worker_script=rvc_config.get('worker_script'),
                ):
                    try:
                        rvc = RVCConverter(
                            model_path=rvc_config.get('model_path'),
                            index_path=rvc_config.get('index_path'),
                            device=rvc_config.get('device', 'cuda:0'),
                            f0_method=rvc_config.get('f0_method', 'rmvpe'),
                            index_rate=rvc_config.get('index_rate', 0.75),
                            protect=rvc_config.get('protect', 0.33),
                            backend=rvc_config.get('backend', 'auto'),
                            python_path=rvc_config.get('python_path'),
                            site_packages_dir=rvc_config.get('site_packages_dir'),
                            worker_script=rvc_config.get('worker_script'),
                            f0_up_key=rvc_config.get('f0_up_key', 0.0),
                            output_freq=rvc_config.get('output_freq'),
                        )
                        logger.info("RVC: %s", rvc_config.get('model_path'))
                    except Exception as exc:
                        logger.warning("RVC init failed, voice conversion disabled: %s", exc)
                else:
                    logger.warning("RVC requested but backend is not usable in this environment")

            # Create pipeline
            pipeline_config = ConversationConfig(
                character_name=character_config.get('name', 'AI'),
                system_prompt=character_config.get('system_prompt', 'You are a helpful assistant.'),
                stream_tts=tts_config.get('stream_tts', True),
                auto_detect_language=tts_config.get('auto_detect_language', True),
                asr_language=asr_config.get('language', 'auto'),
            )

            self.pipeline = ConversationPipeline(
                llm=llm,
                tts=tts,
                asr=asr,
                config=pipeline_config,
                rvc=rvc,
            )

            # Set pipeline callbacks
            self.pipeline.on_transcription = self._on_transcription
            self.pipeline.on_response_start = self._on_response_start
            self.pipeline.on_response_chunk = self._on_response_chunk
            self.pipeline.on_response_end = self._on_response_end
            self.pipeline.on_audio_ready = self._on_audio_ready
            self.pipeline.on_expression_change = self._on_expression_change
            self.pipeline.on_error = self._on_error

        logger.info("=" * 50)

        # Create audio service (shared by both modes)
        llm_provider = self.config.get('llm', {}).get('provider', 'ollama')
        gemma_vad_config = self.config.get('gemma', {}) if mode == 'gemma-omni' or (mode == 'pipeline' and llm_provider == 'gemma') else {}
        audio_config = AudioServiceConfig(
            sample_rate=16000,
            start_muted=False,
            vad_prob_threshold=gemma_vad_config.get('vad_prob_threshold', 0.5),
            vad_db_threshold=gemma_vad_config.get('vad_db_threshold', -50),
            vad_required_hits=gemma_vad_config.get('vad_required_hits', 3),
            vad_required_misses=gemma_vad_config.get('vad_required_misses', 30),
        )
        self.audio_service = AudioService(audio_config)
        self.audio_service.on_speech_start = self._on_speech_start
        self.audio_service.on_speech_end = self._on_speech_end
        self.audio_service.on_speech_detected = self._on_speech_detected
        self.audio_service.on_state_change = self._on_mic_state_change

        logger.info("All components created")
    
    # ==================== Callbacks ====================

    def _next_turn_id(self) -> int:
        self._turn_counter += 1
        return self._turn_counter

    def _resolve_turn_id(self) -> Optional[int]:
        return CURRENT_DESKTOP_TURN_ID.get() or self._active_turn_id

    def _dispatch_frontend_event(self, event_name: str, *args):
        js_args = ", ".join(json.dumps(arg, ensure_ascii=False) for arg in args)
        self._evaluate_js(f"window.{event_name}?.({js_args})")
        if self._bridge_server:
            self._bridge_server.emit_frontend_event_sync(event_name, *args)

    def _has_active_playback(self) -> bool:
        return self._latest_audio_turn_id is not None and self._playback_deadline > time.monotonic()

    def _assistant_busy(self) -> bool:
        return bool(
            (self._active_response_future and not self._active_response_future.done())
            or self._has_active_playback()
        )

    def _interrupt_current_turn(self, reason: str = "interrupt") -> dict:
        future = self._active_response_future
        turn_id = self._active_turn_id if self._active_turn_id is not None else self._latest_audio_turn_id
        interrupted = False
        active_pipeline = self._get_active_pipeline()
        cancel_active_run = getattr(active_pipeline, "cancel_active_run", None)

        if future and not future.done():
            if callable(cancel_active_run):
                cancel_active_run(reason)
            future.cancel()
            interrupted = True
            logger.info("⏹️ Interrupt requested (%s): active response cancelled", reason)
        elif turn_id is not None and self._has_active_playback():
            interrupted = True
            logger.info("⏹️ Interrupt requested (%s): stopping queued playback", reason)
        else:
            logger.info("⏹️ Interrupt requested (%s): no active turn", reason)

        self._playback_deadline = 0.0
        self._latest_audio_turn_id = None

        if turn_id is not None:
            self._dispatch_frontend_event("onPlaybackStop", turn_id)

        return {
            "status": "ok",
            "interrupted": interrupted,
            "turn_id": turn_id,
            **self.get_runtime_state(),
        }

    def request_interrupt(self, reason: str = "ui") -> dict:
        return self._interrupt_current_turn(reason)

    def toggle_mute(self) -> dict:
        if not self.audio_service:
            return {"status": "error", "message": "Audio service unavailable"}
        self.audio_service.toggle_mute()
        return {"status": "ok", **self.get_runtime_state()}

    def toggle_debug(self) -> dict:
        self._debug_visible = not self._debug_visible
        self._evaluate_js("window.Live2DAPI?.toggleDebug?.()")
        return {"status": "ok", **self.get_runtime_state()}

    def get_runtime_state(self) -> dict:
        mic_state = self.audio_service.state.value if self.audio_service else "loading"
        active_future = self._active_response_future
        return {
            "mode": self.config.get('mode', 'pipeline'),
            "mic_state": mic_state,
            "active_turn_id": self._active_turn_id,
            "response_active": bool(active_future and not active_future.done()),
            "playback_active": self._has_active_playback(),
            "debug_visible": self._debug_visible,
            "character_name": self.config.get('character', {}).get('name', 'AI'),
        }

    def _get_active_pipeline(self):
        return (
            getattr(self, '_gemma_pipeline', None)
            or getattr(self, '_omni_pipeline', None)
            or self.pipeline
        )

    def _start_turn(self, turn_id: int, runner, source: str) -> None:
        if not self._loop:
            raise RuntimeError("Event loop not available")

        previous_future = self._active_response_future
        previous_turn_id = self._active_turn_id if previous_future and not previous_future.done() else self._latest_audio_turn_id
        active_pipeline = self._get_active_pipeline()
        cancel_active_run = getattr(active_pipeline, "cancel_active_run", None)

        if previous_turn_id is not None and previous_turn_id != turn_id:
            self._dispatch_frontend_event("onPlaybackStop", previous_turn_id)

        self._playback_deadline = 0.0
        self._latest_audio_turn_id = None
        self._active_turn_id = turn_id

        async def run_turn_with_context():
            token = CURRENT_DESKTOP_TURN_ID.set(turn_id)
            try:
                if previous_future and not previous_future.done():
                    if callable(cancel_active_run):
                        cancel_active_run(f"superseded by {source} turn {turn_id}")
                    previous_future.cancel()

                return await asyncio.wait_for(runner(), timeout=self._turn_timeout_sec)
            finally:
                CURRENT_DESKTOP_TURN_ID.reset(token)

        started_at = time.perf_counter()
        future = asyncio.run_coroutine_threadsafe(run_turn_with_context(), self._loop)
        self._active_response_future = future

        def on_done(f):
            elapsed = time.perf_counter() - started_at
            try:
                result = f.result()
                if result:
                    preview = result[:80].replace("\n", " ")
                    logger.info("%s turn %s completed in %.1fs: %s...", source, turn_id, elapsed, preview)
                else:
                    logger.warning("%s turn %s completed in %.1fs without spoken response", source, turn_id, elapsed)
            except FutureCancelledError:
                logger.info("%s turn %s cancelled after %.1fs", source, turn_id, elapsed)
            except asyncio.TimeoutError:
                logger.error("%s turn %s timed out after %.1fs", source, turn_id, elapsed)
            except Exception as e:
                logger.error("%s turn %s failed after %.1fs: %s", source, turn_id, elapsed, e, exc_info=True)
            finally:
                if self._active_response_future is f:
                    self._active_response_future = None

        future.add_done_callback(on_done)

    def submit_text(self, text: str) -> dict:
        user_text = (text or "").strip()
        if not user_text:
            return {"status": "ignored", "message": "Empty text"}

        if not self._loop:
            return {"status": "error", "message": "Event loop not available"}

        active_pipeline = self._get_active_pipeline()
        process_text = getattr(active_pipeline, "process_text", None)
        if not callable(process_text):
            message = "Text input is only supported in pipeline mode for the desktop stable path."
            logger.warning(message)
            self._dispatch_frontend_event("onError", message, None)
            return {"status": "error", "message": message}

        turn_id = self._next_turn_id()
        self._start_turn(turn_id, lambda: process_text(user_text), source="text")
        return {"status": "ok", "turn_id": turn_id, **self.get_runtime_state()}

    def _on_speech_start(self):
        """Called when VAD confirms speech start."""
        interrupted_turn_id = self._active_turn_id if self._assistant_busy() else None
        if interrupted_turn_id is not None:
            self._interrupt_current_turn("barge-in")
        self._dispatch_frontend_event("onSpeechStart", interrupted_turn_id)

    def _on_speech_end(self):
        """Called when VAD confirms speech end."""
        self._dispatch_frontend_event("onSpeechEnd", self._active_turn_id)

    def _on_speech_detected(self, audio_bytes: bytes):
        """Called when VAD detects complete speech."""
        if not self._loop:
            logger.error("Event loop not available!")
            return
        audio_ms = int(len(audio_bytes) / 32) if audio_bytes else 0
        logger.info(f"Processing {len(audio_bytes)} bytes of speech (~{audio_ms} ms)...")

        active_pipeline = self._get_active_pipeline()
        if active_pipeline is None:
            logger.error("No pipeline available!")
            return

        turn_id = self._next_turn_id()
        self._start_turn(turn_id, lambda: active_pipeline.process_speech(audio_bytes), source="speech")
    
    def _on_mic_state_change(self, state: MicState):
        """Called when mic state changes."""
        self._dispatch_frontend_event("onMicStateChange", state.value)
    
    async def _on_transcription(self, text: str):
        """Called when ASR produces transcription."""
        self._dispatch_frontend_event("onTranscription", text, self._resolve_turn_id())
    
    async def _on_response_start(self):
        """Called when LLM starts generating."""
        self._dispatch_frontend_event("onResponseStart", self._resolve_turn_id())
    
    async def _on_response_chunk(self, chunk: str):
        """Called for each LLM output chunk."""
        self._dispatch_frontend_event("onResponseChunk", chunk, self._resolve_turn_id())
    
    async def _on_response_end(self, full_text: str):
        """Called when LLM finishes generating."""
        self._dispatch_frontend_event("onResponseEnd", full_text, self._resolve_turn_id())
    
    async def _on_audio_ready(self, payload: AudioPayload):
        """Called when TTS audio is ready with volume data."""
        logger.info("Audio payload ready (%sms, text=%r)", payload.duration_ms, (payload.text or '')[:80])
        audio_base64 = payload.audio_base64
        if not audio_base64 and payload.wav_bytes:
            audio_base64 = base64.b64encode(payload.wav_bytes).decode("utf-8")
        emitted_at_ms = int(time.time() * 1000)

        turn_id = self._resolve_turn_id()
        if turn_id is not None:
            self._latest_audio_turn_id = turn_id
        self._playback_deadline = max(
            self._playback_deadline,
            time.monotonic() + max(payload.duration_ms, 0) / 1000.0,
        )

        data = {
            'audio': audio_base64,
            'volumes': payload.volumes,
            'duration': payload.duration_ms,
            'text': payload.text,
            'expression': payload.expression,
            'turn_id': turn_id,
            'tts_metrics': payload.tts_metrics,
            'trace': {
                'backend_audio_ready_epoch_ms': emitted_at_ms,
            },
        }
        self._dispatch_frontend_event("onAudioReady", data)
    
    async def _on_expression_change(self, expression: str):
        """Called when emotion is detected."""
        self._evaluate_js(f"window.Live2DAPI?.setExpression?.({json.dumps(expression, ensure_ascii=False)})")

    async def _on_error(self, error_msg: str):
        """Called when the active pipeline surfaces an error."""
        logger.error(f"Pipeline surfaced error: {error_msg}")
        self._dispatch_frontend_event("onError", error_msg, self._resolve_turn_id())
    
    def _evaluate_js(self, code: str):
        """Evaluate JavaScript in the webview window."""
        if self._window:
            try:
                self._window.evaluate_js(code)
            except Exception as e:
                logger.debug(f"JS eval error: {e}")
    
    def _preload_models_and_start_audio(self):
        """Load heavy models in the background, then start audio capture."""
        try:
            if hasattr(self, '_omni_pipeline') and self._omni_pipeline:
                logger.info("⏳ Pre-loading omni model in background...")
                self._omni_pipeline.preload()
                logger.info("✅ Omni model ready")

            if hasattr(self, '_gemma_pipeline') and self._gemma_pipeline:
                logger.info("⏳ Pre-loading Gemma + Chatterbox in background...")
                self._gemma_pipeline.preload()
                logger.info("✅ Gemma + Chatterbox ready")

            if self.pipeline:
                logger.info("⏳ Pre-loading pipeline models in background...")
                llm = getattr(self.pipeline, 'llm', None)
                asr = getattr(self.pipeline, 'asr', None)
                tts = getattr(self.pipeline, 'tts', None)
                rvc = getattr(self.pipeline, 'rvc', None)
                if hasattr(llm, 'preload'):
                    llm.preload()
                if hasattr(asr, '_get_model'):
                    asr._get_model()
                if hasattr(tts, 'preload'):
                    tts.preload()
                elif hasattr(tts, '_load_model'):
                    try:
                        tts._load_model()
                    except Exception as exc:
                        if tts.__class__.__name__ == "Qwen3TTSProvider":
                            from src.tts import KokoroProvider

                            fallback_voice = self.config.get('tts', {}).get('kokoro_voice', 'ff_siwis')
                            logger.warning("Qwen3-TTS preload failed, falling back to Kokoro: %s", exc)
                            fallback_tts = KokoroProvider(voice=fallback_voice)
                            fallback_tts._load_pipeline()
                            self.pipeline.tts = fallback_tts
                            tts = fallback_tts
                        else:
                            raise
                tts_warmup_on_start = bool(self.config.get('tts', {}).get('warmup_on_start', False))
                if tts_warmup_on_start and hasattr(tts, 'warmup'):
                    tts.warmup()
                if rvc and hasattr(rvc, 'preload'):
                    rvc.preload()
                    if hasattr(rvc, 'warmup'):
                        rvc.warmup()
                logger.info("✅ Pipeline models ready")

            if self.audio_service and self._loop and not self.audio_service._running:
                self.audio_service.start(self._loop)
                logger.info("✅ Audio capture enabled after model preload")

        except Exception as e:
            logger.error(f"Background preload failed: {e}", exc_info=True)

    def _start_background_preload(self):
        """Start model preload without blocking the UI."""
        if self._preload_thread and self._preload_thread.is_alive():
            return

        self._preload_thread = threading.Thread(
            target=self._preload_models_and_start_audio,
            daemon=True,
            name="ModelPreload",
        )
        self._preload_thread.start()

    # ==================== Hotkeys ====================
    
    def _setup_hotkeys(self):
        """Setup global hotkey listener."""
        if not PYNPUT_AVAILABLE:
            logger.warning("pynput not available, hotkeys disabled")
            return
        
        def on_press(key):
            try:
                if key == keyboard.Key.f2:
                    runtime = self.toggle_mute()
                    logger.info("F2 toggle mute -> %s", runtime.get("mic_state"))
                    
                elif key == keyboard.Key.f3:
                    runtime = self.request_interrupt("hotkey")
                    logger.info("F3 interrupt -> turn=%s", runtime.get("turn_id"))
                    
                elif key == keyboard.Key.f12:
                    runtime = self.toggle_debug()
                    logger.info("F12 debug -> visible=%s", runtime.get("debug_visible"))
                        
                elif key == keyboard.Key.esc:
                    # Quit
                    logger.info("👋 Quit requested")
                    self.stop()
                    
            except Exception as e:
                logger.error(f"Hotkey error: {e}")
        
        self._hotkey_listener = keyboard.Listener(on_press=on_press)
        self._hotkey_listener.start()
        logger.info("⌨️ Hotkeys enabled: F2=mute, F3=interrupt, F12=toggle, Esc=quit")
    
    # ==================== Window ====================
    
    def _create_window(self):
        """Create the Live2D overlay window."""
        if not WEBVIEW_AVAILABLE:
            logger.error("pywebview not available")
            return None
        
        live2d_config = self.config.get('live2d', {})
        window_config = live2d_config.get('window', {})
        
        # HTML path
        html_path = PROJECT_ROOT / "frontend" / "live2d" / "index.html"
        
        window = webview.create_window(
            title="AI Assistant",
            url=f"file://{html_path}",
            width=window_config.get('width', 400),
            height=window_config.get('height', 600),
            x=window_config.get('x', -1) if window_config.get('x', -1) >= 0 else None,
            y=window_config.get('y', -1) if window_config.get('y', -1) >= 0 else None,
            frameless=window_config.get('frameless', True),
            easy_drag=True,
            on_top=window_config.get('on_top', True),
            transparent=window_config.get('transparent', True),
            background_color="#10141d",
            js_api=self._js_api,
        )
        
        return window
    
    def _on_window_loaded(self):
        """Called when the webview window is loaded."""
        logger.info("🖼️ Window loaded")
        self._dispatch_frontend_event("onBackendReady", self.get_runtime_state())
    
    # ==================== Main ====================
    
    def start(self, bridge_only: bool = False, bridge_port: int = 8765):
        """Start the assistant application."""
        logger.info("🚀 Starting Live2D Assistant...")
        self._bridge_only = bridge_only
        self._running = True
        
        # Create components
        self._create_components()
        
        # Setup hotkeys
        self._setup_hotkeys()
        
        # Create event loop and run it in a background thread
        self._loop = asyncio.new_event_loop()
        
        def run_loop():
            """Run the event loop in a background thread."""
            asyncio.set_event_loop(self._loop)
            self._loop.run_forever()
        
        # Start the event loop thread
        import threading
        self._loop_thread = threading.Thread(target=run_loop, daemon=True)
        self._loop_thread.start()
        logger.info("✅ Event loop started in background thread")

        if bridge_only:
            self._bridge_server = DesktopBridgeServer(self, port=bridge_port)
            asyncio.run_coroutine_threadsafe(self._bridge_server.start(), self._loop).result(timeout=5)
            logger.info("Bridge-only desktop backend enabled on port %s", bridge_port)
        
        # Pre-load models without blocking the window, then start audio capture
        self._start_background_preload()

        # Create and start window
        if bridge_only:
            logger.info("Running in bridge-only mode (no pywebview window)")
            try:
                while self._running:
                    time.sleep(0.2)
            except KeyboardInterrupt:
                pass
        elif WEBVIEW_AVAILABLE:
            self._window = self._create_window()

            def on_loaded():
                self._on_window_loaded()

            # Start webview (blocking)
            # Use 'edgechromium' (WebView2) for transparency support on Windows
            # Requires pywebview>=6.0.0 for transparency with mouse events
            webview.start(
                on_loaded,
                debug=logging.getLogger().isEnabledFor(logging.DEBUG),
                gui='edgechromium',
            )
        else:
            # No GUI mode - just run the loop
            logger.info("Running in headless mode (no GUI)")
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                pass
        
        self.stop()
    
    def stop(self):
        """Stop the assistant application."""
        logger.info("🛑 Stopping Live2D Assistant...")
        if not self._running and not self._loop:
            return
        
        self._running = False
        
        if self.audio_service:
            self.audio_service.stop()
        
        if self._hotkey_listener:
            self._hotkey_listener.stop()
        
        # Shutdown omni pipeline if used
        if hasattr(self, '_omni_pipeline') and self._omni_pipeline:
            self._omni_pipeline.shutdown()

        # Shutdown gemma pipeline if used
        if hasattr(self, '_gemma_pipeline') and self._gemma_pipeline:
            asyncio.run_coroutine_threadsafe(
                self._gemma_pipeline.shutdown(), self._loop
            ).result(timeout=10)

        if self.pipeline:
            llm = getattr(self.pipeline, 'llm', None)
            tts = getattr(self.pipeline, 'tts', None)
            rvc = getattr(self.pipeline, 'rvc', None)
            if hasattr(tts, 'cleanup'):
                tts.cleanup()
            if hasattr(llm, 'cleanup'):
                llm.cleanup()
            if rvc and hasattr(rvc, 'close'):
                rvc.close()
        
        if self._bridge_server and self._loop:
            try:
                asyncio.run_coroutine_threadsafe(self._bridge_server.stop(), self._loop).result(timeout=5)
            except Exception as exc:
                logger.debug("Bridge shutdown error: %s", exc)
            self._bridge_server = None

        if self._loop:
            self._loop.call_soon_threadsafe(self._loop.stop)

        if self._loop_thread and self._loop_thread.is_alive():
            self._loop_thread.join(timeout=2)
            self._loop_thread = None
        self._loop = None
        
        logger.info("👋 Goodbye!")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Live2D AI Assistant")
    parser.add_argument(
        '--config', '-c',
        type=Path,
        default=None,
        help='Path to config file'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging'
    )
    parser.add_argument(
        '--bridge-server',
        action='store_true',
        help='Run the desktop backend without pywebview and expose a local websocket bridge for Tauri'
    )
    parser.add_argument(
        '--bridge-port',
        type=int,
        default=8765,
        help='Port for the local desktop bridge websocket server'
    )
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Handle Ctrl+C gracefully
    app = Live2DAssistant(config_path=args.config)
    
    def signal_handler(sig, frame):
        app.stop()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    app.start(bridge_only=args.bridge_server, bridge_port=args.bridge_port)


if __name__ == "__main__":
    main()
