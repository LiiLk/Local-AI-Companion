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

import argparse
import asyncio
import base64
import json
import logging
import signal
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
        
        # State
        self._running = False
        self._window: Optional[webview.Window] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._hotkey_listener = None
        
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
        from src.llm import OllamaLLM
        from src.llm.llamacpp_provider import LlamaCppProvider
        from src.tts import KokoroProvider, EdgeTTSProvider, XTTSProvider, GPTSoVITSProvider
        from src.asr import WhisperProvider, CanaryProvider, ParakeetProvider
        
        llm_config = self.config.get('llm', {})
        tts_config = self.config.get('tts', {})
        asr_config = self.config.get('asr', {})
        character_config = self.config.get('character', {})
        
        logger.info("=" * 50)
        logger.info("🚀 STACK CONFIGURATION")
        logger.info("=" * 50)
        
        # Create LLM
        llm_provider = llm_config.get('provider', 'ollama')
        if llm_provider == 'llamacpp':
            llamacpp_cfg = llm_config.get('llamacpp', {})
            llm = LlamaCppProvider(
                base_url=llamacpp_cfg.get('base_url', 'http://localhost:8080'),
                model_name=llamacpp_cfg.get('model_name', 'qwen3-vl-8b'),
                max_tokens=llamacpp_cfg.get('max_tokens', 4096),
                temperature=llamacpp_cfg.get('temperature', 0.7),
            )
            logger.info(f"🧠 LLM: Qwen3-VL-8B via llama.cpp ({llamacpp_cfg.get('base_url')})")
        else:
            ollama_cfg = llm_config.get('ollama', {})
            llm = OllamaLLM(
                model=ollama_cfg.get('model', 'llama3.2:3b'),
                base_url=ollama_cfg.get('base_url', 'http://localhost:11434')
            )
            logger.info(f"🧠 LLM: Ollama ({ollama_cfg.get('model')})")
        
        # Create TTS
        tts_provider = tts_config.get('provider', 'kokoro')
        if tts_provider == 'gpt_sovits':
            sovits_cfg = tts_config.get('gpt_sovits', {})
            tts = GPTSoVITSProvider(tts_config)
            logger.info(f"🔊 TTS: GPT-SoVITS ({sovits_cfg.get('api_url', 'http://127.0.0.1:9880')})")
        elif tts_provider == 'xtts':
            xtts_cfg = tts_config.get('xtts', {})
            speaker_wav = xtts_cfg.get('speaker_wav')
            tts = XTTSProvider(
                language=xtts_cfg.get('language', 'fr'),
                speaker=xtts_cfg.get('speaker', 'Claribel Dervla'),
                speaker_wav=speaker_wav,
                device=xtts_cfg.get('device', 'cuda'),
            )
            voice_info = f"voice clone: {speaker_wav}" if speaker_wav else f"speaker: {xtts_cfg.get('speaker')}"
            logger.info(f"🔊 TTS: XTTS v2 on {xtts_cfg.get('device', 'cuda')} ({voice_info})")
        elif tts_provider == 'kokoro':
            voice = tts_config.get('kokoro_voice', 'ff_siwis')
            tts = KokoroProvider(voice=voice)
            logger.info(f"🔊 TTS: Kokoro ({voice})")
        else:
            voice = tts_config.get('voice', 'en-US-JennyNeural')
            tts = EdgeTTSProvider(voice=voice)
            logger.info(f"🔊 TTS: Edge ({voice})")
        
        # Create ASR
        asr_provider = asr_config.get('provider', 'whisper')
        device = asr_config.get('device', 'cpu')
        if asr_provider == 'canary':
            asr = CanaryProvider(device=device)
            logger.info(f"🎤 ASR: Canary on {device}")
        elif asr_provider == 'parakeet':
            asr = ParakeetProvider(device=device)
            logger.info(f"🎤 ASR: Parakeet on {device}")
        else:
            model_size = asr_config.get('model_size', 'base')
            prompt = asr_config.get('prompt')
            asr = WhisperProvider(
                model_size=model_size,
                device=device,
                initial_prompt=prompt
            )
            logger.info(f"🎤 ASR: Whisper {model_size} on {device}")
        
        logger.info("=" * 50)
        
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
        
        # Create audio service
        audio_config = AudioServiceConfig(
            sample_rate=16000,
            start_muted=False,
        )
        self.audio_service = AudioService(audio_config)
        self.audio_service.on_speech_detected = self._on_speech_detected
        self.audio_service.on_state_change = self._on_mic_state_change
        
        logger.info("✅ All components created")
    
    # ==================== Callbacks ====================
    
    def _on_speech_detected(self, audio_bytes: bytes):
        """Called when VAD detects complete speech."""
        if self._loop:
            # Set processing state (mute mic while AI responds)
            self.audio_service.set_processing(True)
            
            logger.info(f"🎯 Processing {len(audio_bytes)} bytes of speech...")
            
            # Process in event loop
            future = asyncio.run_coroutine_threadsafe(
                self.pipeline.process_speech(audio_bytes),
                self._loop
            )
            
            def on_done(f):
                try:
                    result = f.result()
                    logger.info(f"✅ Pipeline completed: {result[:50] if result else 'No result'}...")
                except Exception as e:
                    logger.error(f"❌ Pipeline error: {e}", exc_info=True)
                finally:
                    self._on_response_complete()
            
            future.add_done_callback(on_done)
        else:
            logger.error("❌ Event loop not available!")
    
    def _on_response_complete(self):
        """Called when pipeline finishes processing."""
        # Re-enable listening
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
        # Send audio to frontend for playback
        data = {
            'audio': payload.audio_base64,
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
    
    def _evaluate_js(self, code: str):
        """Evaluate JavaScript in the webview window."""
        if self._window:
            try:
                self._window.evaluate_js(code)
            except Exception as e:
                logger.debug(f"JS eval error: {e}")
    
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
                    # Interrupt (TODO: implement interruption)
                    logger.info("⏹️ Interrupt requested")
                    
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
        
        # Start audio service (now the loop is running!)
        self.audio_service.start(self._loop)
        
        # Create and start window
        if WEBVIEW_AVAILABLE:
            self._window = self._create_window()
            
            def on_loaded():
                self._on_window_loaded()
            
            # Start webview (blocking)
            webview.start(on_loaded, debug=True)
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
