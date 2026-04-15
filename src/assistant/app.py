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
from src.assistant.pipeline_runtime import (
    close_pipeline_runtime_services,
    create_pipeline_runtime_components,
    preload_pipeline_runtime_services,
)
from src.utils.character_loader import resolve_character_config
from src.utils.config_loader import load_yaml_config

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
_HEALTH_UNSET = object()


def _describe_exception(exc: BaseException) -> str:
    message = str(exc).strip()
    if message:
        return message
    return exc.__class__.__name__


def resolve_turn_timeout_sec(config: dict) -> int:
    llm_config = config.get("llm", {})
    provider = llm_config.get("provider", "ollama")

    if provider == "ollama":
        ollama_config = llm_config.get("ollama", {})
        return int(
            ollama_config.get(
                "turn_timeout_sec",
                max(int(ollama_config.get("request_timeout_sec", 180)) + 30, 90),
            )
        )

    if provider == "openrouter":
        openrouter_config = llm_config.get("openrouter", {})
        return int(
            openrouter_config.get(
                "turn_timeout_sec",
                max(int(openrouter_config.get("request_timeout_sec", 180)) + 30, 90),
            )
        )

    return int(config.get("gemma", {}).get("turn_timeout_sec", 75))


def load_config(config_path: Path) -> dict:
    """Load configuration from YAML file."""
    return load_yaml_config(config_path)


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
        self._turn_timeout_sec = resolve_turn_timeout_sec(self.config)
        self._turn_counter = 0
        self._active_turn_id: Optional[int] = None
        self._latest_audio_turn_id: Optional[int] = None
        self._playback_deadline: float = 0.0
        self._debug_visible = False
        self._backend_state = "warming_up"
        self._degraded_reason: Optional[str] = None
        self._runtime_error: Optional[str] = None
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
                reply_language=self.config.get('pipeline', {}).get('reply_language'),
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
            language = voice_config.get('chatterbox_language', 'en')

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
                reply_language=self.config.get('pipeline', {}).get('reply_language'),
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
            runtime = create_pipeline_runtime_components(self.config)
            logger.info("LLM: %s", runtime.llm_summary)
            logger.info("TTS: %s", runtime.tts_summary)
            logger.info("ASR: %s", runtime.asr_summary)
            if runtime.rvc_summary:
                logger.info("RVC: %s", runtime.rvc_summary)

            self.pipeline = ConversationPipeline(
                llm=runtime.llm,
                tts=runtime.tts,
                asr=runtime.asr,
                config=runtime.conversation_config,
                rvc=runtime.rvc,
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
        pipeline_config = self.config.get('pipeline', {})
        default_misses = pipeline_config.get('vad_required_misses', 30)
        start_muted = self._resolve_audio_start_muted()
        audio_config = AudioServiceConfig(
            sample_rate=16000,
            start_muted=start_muted,
            vad_prob_threshold=gemma_vad_config.get('vad_prob_threshold', 0.5),
            vad_db_threshold=gemma_vad_config.get('vad_db_threshold', -50),
            vad_required_hits=gemma_vad_config.get('vad_required_hits', 3),
            vad_required_misses=gemma_vad_config.get('vad_required_misses', default_misses),
        )
        self.audio_service = AudioService(audio_config)
        self.audio_service.on_speech_start = self._on_speech_start
        self.audio_service.on_speech_end = self._on_speech_end
        self.audio_service.on_speech_detected = self._on_speech_detected
        self.audio_service.on_state_change = self._on_mic_state_change

        logger.info("All components created")

    def _resolve_audio_start_muted(self) -> bool:
        audio_config = self.config.get("audio", {})
        bridge_start_muted = audio_config.get("bridge_start_muted")
        if bridge_start_muted is not None:
            return bool(bridge_start_muted)
        if self._bridge_only:
            return True
        start_muted = audio_config.get("start_muted")
        if start_muted is not None:
            return bool(start_muted)
        return False
    
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

    def _active_tts_provider_name(self) -> Optional[str]:
        pipeline = self._get_active_pipeline()
        tts = getattr(pipeline, "tts", None)
        if tts is None:
            return None

        provider_name = getattr(tts, "active_provider_name", None)
        if provider_name:
            return provider_name

        class_name = tts.__class__.__name__
        if class_name.endswith("Provider"):
            class_name = class_name[:-8]
        return class_name.lower()

    def _active_llm_model_name(self) -> Optional[str]:
        pipeline = self._get_active_pipeline()
        llm = getattr(pipeline, "llm", None)
        if llm is None:
            return None

        model_name = getattr(llm, "model", None)
        if model_name:
            return str(model_name)

        gemma = getattr(llm, "gemma", None)
        if gemma is not None:
            return str(getattr(gemma, "model_id", "gemma"))

        return llm.__class__.__name__

    def _collect_degraded_reason(self) -> Optional[str]:
        pipeline = self._get_active_pipeline()
        parts: list[str] = []
        llm = getattr(pipeline, "llm", None)
        tts = getattr(pipeline, "tts", None)

        llm_reason = getattr(llm, "degraded_reason", None)
        if llm_reason:
            parts.append(str(llm_reason))

        tts_reason = getattr(tts, "degraded_reason", None)
        if tts_reason:
            parts.append(str(tts_reason))

        if self._degraded_reason:
            parts.append(self._degraded_reason)

        unique_parts: list[str] = []
        for part in parts:
            if part and part not in unique_parts:
                unique_parts.append(part)
        return " | ".join(unique_parts) if unique_parts else None

    def _set_backend_health(
        self,
        *,
        state: Optional[str] = None,
        degraded_reason: Optional[str] | object = _HEALTH_UNSET,
        runtime_error: Optional[str] | object = _HEALTH_UNSET,
    ) -> None:
        if state is not None:
            self._backend_state = state
        if degraded_reason is not _HEALTH_UNSET:
            self._degraded_reason = degraded_reason
        if runtime_error is not _HEALTH_UNSET:
            self._runtime_error = runtime_error

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
        degraded_reason = self._collect_degraded_reason()
        backend_state = self._backend_state
        if self._runtime_error:
            backend_state = "error"
        elif backend_state != "warming_up" and degraded_reason:
            backend_state = "degraded"
        return {
            "mode": self.config.get('mode', 'pipeline'),
            "backend_state": backend_state,
            "mic_state": mic_state,
            "active_turn_id": self._active_turn_id,
            "response_active": bool(active_future and not active_future.done()),
            "playback_active": self._has_active_playback(),
            "debug_visible": self._debug_visible,
            "character_name": self.config.get('character', {}).get('name', 'AI'),
            "active_language": getattr(self._get_active_pipeline(), "_current_language_code", None),
            "active_llm_model": self._active_llm_model_name(),
            "active_tts_provider": self._active_tts_provider_name(),
            "degraded_reason": degraded_reason,
            "runtime_error": self._runtime_error,
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
                if callable(cancel_active_run):
                    cancel_active_run(f"{source} turn {turn_id} timeout")
                logger.error("%s turn %s timed out after %.1fs", source, turn_id, elapsed)
            except Exception as e:
                logger.error(
                    "%s turn %s failed after %.1fs: %s",
                    source,
                    turn_id,
                    elapsed,
                    _describe_exception(e),
                    exc_info=True,
                )
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

        runtime = self.get_runtime_state()
        if runtime.get("backend_state") == "warming_up":
            return {"status": "warming_up", "message": "Backend is still warming up.", **runtime}
        if runtime.get("backend_state") == "error":
            return {"status": "error", "message": runtime.get("runtime_error") or "Backend failed to start.", **runtime}

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
        logger.info("LLM response complete (%s chars): %r", len(full_text or ""), (full_text or "")[:160])
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
                **(payload.trace or {}),
                'backend_audio_ready_epoch_ms': emitted_at_ms,
            },
        }
        self._dispatch_frontend_event("onAudioReady", data)
    
    async def _on_expression_change(self, expression: str):
        """Called when emotion is detected."""
        self._evaluate_js(f"window.Live2DAPI?.setExpression?.({json.dumps(expression, ensure_ascii=False)})")

    async def _on_error(self, error_msg: str):
        """Called when the active pipeline surfaces an error."""
        logger.error("Pipeline surfaced error: %s", error_msg or "Unknown pipeline error")
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
            self._set_backend_health(state="warming_up", runtime_error=None)
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
                tts, rvc = preload_pipeline_runtime_services(
                    llm=llm,
                    asr=asr,
                    tts=tts,
                    rvc=rvc,
                    tts_warmup_on_start=bool(self.config.get('tts', {}).get('warmup_on_start', False)),
                    rvc_warmup_on_start=True,
                )
                self.pipeline.tts = tts
                self.pipeline.rvc = rvc
                logger.info("✅ Pipeline models ready")

            degraded_reason = self._collect_degraded_reason()
            self._set_backend_health(
                state="degraded" if degraded_reason else "ready",
                degraded_reason=degraded_reason,
                runtime_error=None,
            )
            if self.audio_service and self._loop and not self.audio_service._running:
                self.audio_service.start(self._loop)
                logger.info("✅ Audio capture enabled after model preload")
            self._dispatch_frontend_event("onBackendReady", self.get_runtime_state())

        except Exception as e:
            self._set_backend_health(state="error", runtime_error=str(e))
            logger.error(f"Background preload failed: {e}", exc_info=True)
            self._dispatch_frontend_event("onError", str(e), self._resolve_turn_id())

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
            asr = getattr(self.pipeline, 'asr', None)
            rvc = getattr(self.pipeline, 'rvc', None)
            try:
                if self._loop and self._loop.is_running():
                    asyncio.run_coroutine_threadsafe(
                        close_pipeline_runtime_services(llm=llm, tts=tts, asr=asr, rvc=rvc),
                        self._loop,
                    ).result(timeout=10)
                else:
                    asyncio.run(close_pipeline_runtime_services(llm=llm, tts=tts, asr=asr, rvc=rvc))
            except Exception as exc:
                logger.debug("Pipeline cleanup error: %s", exc, exc_info=True)
        
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
