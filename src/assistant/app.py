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

# Set HuggingFace offline mode BEFORE any imports to prevent network requests
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

import argparse
import asyncio
import base64
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

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: Path) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


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
        self._hotkey_listener = None
        self._preload_thread: Optional[threading.Thread] = None
        self._active_response_future = None
        self._turn_timeout_sec = int(self.config.get('gemma', {}).get('turn_timeout_sec', 75))
        
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
            from src.llm import OllamaLLM
            from src.tts import KokoroProvider, EdgeTTSProvider
            from src.asr import WhisperProvider

            llm_config = self.config.get('llm', {})
            tts_config = self.config.get('tts', {})
            asr_config = self.config.get('asr', {})

            # Create LLM
            ollama_cfg = llm_config.get('ollama', {})
            llm = OllamaLLM(
                model=ollama_cfg.get('model', 'llama3.2:3b'),
                base_url=ollama_cfg.get('base_url', 'http://localhost:11434')
            )
            logger.info(f"LLM: Ollama ({ollama_cfg.get('model')})")

            # Create TTS
            tts_provider = tts_config.get('provider', 'kokoro')
            if tts_provider == 'kokoro':
                voice = tts_config.get('kokoro_voice', 'ff_siwis')
                tts = KokoroProvider(voice=voice)
                logger.info(f"TTS: Kokoro ({voice})")
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

            # Create pipeline
            pipeline_config = ConversationConfig(
                character_name=character_config.get('name', 'AI'),
                system_prompt=character_config.get('system_prompt', 'You are a helpful assistant.'),
                stream_tts=tts_config.get('stream_tts', True),
                auto_detect_language=tts_config.get('auto_detect_language', True),
            )

            self.pipeline = ConversationPipeline(
                llm=llm,
                tts=tts,
                asr=asr,
                config=pipeline_config
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
        gemma_vad_config = self.config.get('gemma', {}) if mode == 'gemma-omni' else {}
        audio_config = AudioServiceConfig(
            sample_rate=16000,
            start_muted=False,
            vad_prob_threshold=gemma_vad_config.get('vad_prob_threshold', 0.5),
            vad_db_threshold=gemma_vad_config.get('vad_db_threshold', -50),
            vad_required_hits=gemma_vad_config.get('vad_required_hits', 3),
            vad_required_misses=gemma_vad_config.get('vad_required_misses', 30),
        )
        self.audio_service = AudioService(audio_config)
        self.audio_service.on_speech_detected = self._on_speech_detected
        self.audio_service.on_state_change = self._on_mic_state_change

        logger.info("All components created")
    
    # ==================== Callbacks ====================
    
    def _on_speech_detected(self, audio_bytes: bytes):
        """Called when VAD detects complete speech."""
        if not self._loop:
            logger.error("Event loop not available!")
            return

        if self._active_response_future and not self._active_response_future.done():
            logger.warning("A response is already in progress, ignoring new speech")
            return

        self.audio_service.set_processing(True)
        audio_ms = int(len(audio_bytes) / 32) if audio_bytes else 0
        logger.info(f"Processing {len(audio_bytes)} bytes of speech (~{audio_ms} ms)...")

        active_pipeline = (
            getattr(self, '_gemma_pipeline', None)
            or getattr(self, '_omni_pipeline', None)
            or self.pipeline
        )
        if active_pipeline is None:
            logger.error("No pipeline available!")
            self._on_response_complete()
            return

        async def run_pipeline_with_timeout():
            return await asyncio.wait_for(
                active_pipeline.process_speech(audio_bytes),
                timeout=self._turn_timeout_sec,
            )

        started_at = time.perf_counter()
        future = asyncio.run_coroutine_threadsafe(run_pipeline_with_timeout(), self._loop)
        self._active_response_future = future

        def on_done(f):
            elapsed = time.perf_counter() - started_at
            try:
                result = f.result()
                if result:
                    preview = result[:80].replace("\n", " ")
                    logger.info("Pipeline completed in %.1fs: %s...", elapsed, preview)
                else:
                    logger.warning("Pipeline completed in %.1fs without a spoken response", elapsed)
            except asyncio.TimeoutError:
                logger.error("Pipeline timed out after %.1fs", elapsed)
            except Exception as e:
                logger.error(f"Pipeline error after {elapsed:.1f}s: {e}", exc_info=True)
            finally:
                if self._active_response_future is f:
                    self._active_response_future = None
                self._on_response_complete()

        future.add_done_callback(on_done)
    
    def _on_response_complete(self):
        """Called when pipeline finishes processing."""
        if self._active_response_future and self._active_response_future.done():
            self._active_response_future = None
        if self.audio_service:
            self.audio_service.set_processing(False)
    
    def _on_mic_state_change(self, state: MicState):
        """Called when mic state changes."""
        self._evaluate_js(f"window.onMicStateChange?.('{state.value}')")
    
    async def _on_transcription(self, text: str):
        """Called when ASR produces transcription."""
        # Escape text for JS
        escaped = text.replace('\\', '\\\\').replace("'", "\\'").replace('\n', '\\n')
        self._evaluate_js(f"window.onTranscription?.('{escaped}')")
    
    async def _on_response_start(self):
        """Called when LLM starts generating."""
        self._evaluate_js("window.onResponseStart?.()")
    
    async def _on_response_chunk(self, chunk: str):
        """Called for each LLM output chunk."""
        escaped = chunk.replace('\\', '\\\\').replace("'", "\\'").replace('\n', '\\n')
        self._evaluate_js(f"window.onResponseChunk?.('{escaped}')")
    
    async def _on_response_end(self, full_text: str):
        """Called when LLM finishes generating."""
        escaped = full_text.replace('\\', '\\\\').replace("'", "\\'").replace('\n', '\\n')
        self._evaluate_js(f"window.onResponseEnd?.('{escaped}')")
    
    async def _on_audio_ready(self, payload: AudioPayload):
        """Called when TTS audio is ready with volume data."""
        logger.info("Audio payload ready (%sms, text=%r)", payload.duration_ms, (payload.text or '')[:80])
        # Send audio to frontend for playback
        audio_base64 = payload.audio_base64
        if not audio_base64 and payload.wav_bytes:
            audio_base64 = base64.b64encode(payload.wav_bytes).decode("utf-8")

        data = {
            'audio': audio_base64,
            'volumes': payload.volumes,
            'duration': payload.duration_ms,
            'text': payload.text,
            'expression': payload.expression,
        }
        json_str = json.dumps(data)
        self._evaluate_js(f"window.onAudioReady?.({json_str})")
    
    async def _on_expression_change(self, expression: str):
        """Called when emotion is detected."""
        self._evaluate_js(f"Live2DManager?.setExpression?.('{expression}')")

    async def _on_error(self, error_msg: str):
        """Called when the active pipeline surfaces an error."""
        logger.error(f"Pipeline surfaced error: {error_msg}")
        escaped = error_msg.replace("\\", "\\\\").replace("\'", "\\\'").replace("\n", "\\n")
        self._evaluate_js(f"window.onError?.('{escaped}')")
    
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
                    # Toggle mute
                    is_muted = self.audio_service.toggle_mute()
                    status = "🔇 Muted" if is_muted else "🎤 Listening"
                    logger.info(status)
                    
                elif key == keyboard.Key.f3:
                    if self._active_response_future and not self._active_response_future.done():
                        self._active_response_future.cancel()
                        logger.info("⏹️ Interrupt requested: current response cancelled")
                        self._on_response_complete()
                    else:
                        logger.info("⏹️ Interrupt requested (no active response)")
                    
                elif key == keyboard.Key.f12:
                    # Toggle visibility
                    if self._window:
                        # Toggle window visibility
                        pass
                        
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
        )
        
        return window
    
    def _on_window_loaded(self):
        """Called when the webview window is loaded."""
        logger.info("🖼️ Window loaded")
        
        # Inject assistant callbacks
        js_code = """
        // Audio queue for sequential playback
        window._audioQueue = [];
        window._isPlaying = false;
        window._audioContext = null;
        
        // Initialize AudioContext on first interaction (or try immediately)
        function initAudioContext() {
            if (!window._audioContext) {
                window._audioContext = new (window.AudioContext || window.webkitAudioContext)();
                console.log('AudioContext created, state:', window._audioContext.state);
            }
            if (window._audioContext.state === 'suspended') {
                window._audioContext.resume();
            }
        }
        
        // Play next audio in queue
        async function playNextAudio() {
            if (window._isPlaying || window._audioQueue.length === 0) return;
            
            window._isPlaying = true;
            const data = window._audioQueue.shift();
            
            try {
                initAudioContext();
                
                // Set expression if present
                if (data.expression && window.Live2DManager) {
                    Live2DManager.setExpression(data.expression);
                }
                
                // Decode base64 to ArrayBuffer
                const binaryString = atob(data.audio);
                const bytes = new Uint8Array(binaryString.length);
                for (let i = 0; i < binaryString.length; i++) {
                    bytes[i] = binaryString.charCodeAt(i);
                }
                
                // Decode audio
                const audioBuffer = await window._audioContext.decodeAudioData(bytes.buffer);
                
                // Create source node
                const source = window._audioContext.createBufferSource();
                source.buffer = audioBuffer;
                source.connect(window._audioContext.destination);
                
                // Start lip-sync
                const volumes = data.volumes;
                const chunkMs = (audioBuffer.duration * 1000) / volumes.length;
                let index = 0;
                
                const lipSyncInterval = setInterval(() => {
                    if (index < volumes.length && window.Live2DManager) {
                        Live2DManager.setLipSync(volumes[index]);
                        index++;
                    } else {
                        clearInterval(lipSyncInterval);
                        if (window.Live2DManager) {
                            Live2DManager.setLipSync(0);
                        }
                    }
                }, chunkMs);
                
                // When audio ends
                source.onended = function() {
                    clearInterval(lipSyncInterval);
                    if (window.Live2DManager) {
                        Live2DManager.setLipSync(0);
                    }
                    window._isPlaying = false;
                    playNextAudio();  // Play next in queue
                };
                
                // Play!
                source.start(0);
                console.log('🔊 Playing audio:', Math.round(audioBuffer.duration * 1000), 'ms');
                
            } catch (e) {
                console.error('Audio decode/play error:', e);
                window._isPlaying = false;
                playNextAudio();  // Try next
            }
        }
        
        // Audio playback with lip-sync
        window.onAudioReady = function(data) {
            console.log('Audio queued:', data.duration, 'ms, queue size:', window._audioQueue.length + 1);
            window._audioQueue.push(data);
            playNextAudio();
        };
        
        // Mic state indicator
        window.onMicStateChange = function(state) {
            console.log('Mic state:', state);
        };
        
        // Transcription display
        window.onTranscription = function(text) {
            console.log('👤 User:', text);
        };
        
        // Response handling
        window.onResponseStart = function() {
            console.log('🤔 AI thinking...');
        };
        
        window.onResponseChunk = function(chunk) {
            // Could display streaming text
        };
        
        window.onResponseEnd = function(text) {
            console.log('✅ AI response complete');
        };
        
        // Try to init audio context immediately (may fail without user interaction)
        try { initAudioContext(); } catch(e) {}
        
        // Also init on any click
        document.addEventListener('click', initAudioContext, { once: true });
        
        console.log('✅ Assistant callbacks injected');
        """
        
        if self._window:
            self._window.evaluate_js(js_code)
    
    # ==================== Main ====================
    
    def start(self):
        """Start the assistant application."""
        logger.info("🚀 Starting Live2D Assistant...")
        
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
        
        # Pre-load models without blocking the window, then start audio capture
        self._start_background_preload()

        # Create and start window
        if WEBVIEW_AVAILABLE:
            self._window = self._create_window()

            def on_loaded():
                self._on_window_loaded()

            # Start webview (blocking)
            # Use 'edgechromium' (WebView2) for transparency support on Windows
            # Requires pywebview>=6.0.0 for transparency with mouse events
            webview.start(on_loaded, debug=True, gui='edgechromium')
        else:
            # No GUI mode - just run the loop
            logger.info("Running in headless mode (no GUI)")
            try:
                while True:
                    import time
                    time.sleep(1)
            except KeyboardInterrupt:
                pass
        
        self.stop()
    
    def stop(self):
        """Stop the assistant application."""
        logger.info("🛑 Stopping Live2D Assistant...")
        
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
        
        if self._loop:
            self._loop.call_soon_threadsafe(self._loop.stop)
        
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
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Handle Ctrl+C gracefully
    app = Live2DAssistant(config_path=args.config)
    
    def signal_handler(sig, frame):
        app.stop()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    app.start()


if __name__ == "__main__":
    main()
