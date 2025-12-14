"""
Desktop Companion Application

A desktop pet/companion that displays a Live2D avatar as a transparent
overlay window on Linux/Windows/macOS. Connects to the AI backend via
WebSocket for voice interaction.

Features:
- Transparent, frameless, always-on-top window
- Live2D avatar with lip-sync and expressions
- System tray with quick actions
- Global hotkeys (F12 toggle, F11 settings)
- Draggable window
- Microphone input with VAD
"""

import asyncio
import base64
import json
import logging
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional
from queue import Queue

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)

# Optional imports with fallbacks
try:
    import webview
    WEBVIEW_AVAILABLE = True
except ImportError:
    WEBVIEW_AVAILABLE = False
    logger.warning("pywebview not installed. Install with: pip install pywebview")

try:
    from pynput import keyboard
    PYNPUT_AVAILABLE = True
except ImportError:
    PYNPUT_AVAILABLE = False
    logger.warning("pynput not installed. Hotkeys disabled. Install with: pip install pynput")

try:
    import pystray
    from PIL import Image
    TRAY_AVAILABLE = True
except ImportError:
    TRAY_AVAILABLE = False
    logger.warning("pystray/PIL not installed. System tray disabled. Install with: pip install pystray pillow")

try:
    import websockets
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False
    logger.warning("websockets not installed. Install with: pip install websockets")


@dataclass  
class DesktopConfig:
    """Configuration for the desktop companion."""
    # Window settings
    width: int = 400
    height: int = 600
    x: int = -1  # -1 = auto (right side)
    y: int = -1  # -1 = auto (bottom)
    opacity: float = 1.0
    frameless: bool = True
    on_top: bool = True
    transparent: bool = True
    
    # Backend connection
    backend_host: str = "localhost"
    backend_port: int = 8000
    
    # Features
    enable_hotkeys: bool = True
    enable_tray: bool = True
    enable_microphone: bool = True
    
    # Paths
    html_path: str = ""
    icon_path: str = ""
    
    # Debug
    debug: bool = False
    
    def __post_init__(self):
        if not self.html_path:
            self.html_path = str(PROJECT_ROOT / "frontend" / "desktop" / "index.html")
        if not self.icon_path:
            self.icon_path = str(PROJECT_ROOT / "assets" / "icon.png")


class WebSocketClient:
    """
    WebSocket client for communication with the AI backend.
    
    Handles sending audio/text and receiving responses with lip-sync data.
    """
    
    def __init__(self, host: str, port: int, client_id: str = "desktop"):
        self.host = host
        self.port = port
        self.client_id = client_id
        self.ws = None
        self._running = False
        self._loop = None
        self._thread = None
        self._message_queue = Queue()
        
        # Callbacks
        self.on_connected: Optional[Callable] = None
        self.on_disconnected: Optional[Callable] = None
        self.on_text_chunk: Optional[Callable[[str], None]] = None
        self.on_text_end: Optional[Callable[[str], None]] = None
        self.on_audio_data: Optional[Callable[[dict], None]] = None
        self.on_expression_change: Optional[Callable[[str], None]] = None
        self.on_transcription: Optional[Callable[[str], None]] = None
        self.on_vad_start: Optional[Callable] = None
        self.on_vad_end: Optional[Callable] = None
        self.on_error: Optional[Callable[[str], None]] = None
    
    @property
    def ws_url(self) -> str:
        return f"ws://{self.host}:{self.port}/ws/{self.client_id}"
    
    def start(self):
        """Start the WebSocket client in a background thread."""
        if self._running:
            return
        
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
    
    def stop(self):
        """Stop the WebSocket client gracefully."""
        self._running = False
        
        # Close WebSocket connection
        if self.ws and self._loop:
            try:
                future = asyncio.run_coroutine_threadsafe(
                    self.ws.close(),
                    self._loop
                )
                future.result(timeout=2)  # Wait up to 2s for close
            except Exception:
                pass  # Ignore errors during shutdown
        
        # Stop the event loop
        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
        
        # Wait for thread to finish
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2)
    
    def _run_loop(self):
        """Run the asyncio event loop in background thread."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        
        try:
            self._loop.run_until_complete(self._connect_and_listen())
        except Exception as e:
            if self._running:  # Only log if not intentionally stopped
                logger.error(f"WebSocket loop error: {e}")
        finally:
            # Cancel all pending tasks
            try:
                pending = asyncio.all_tasks(self._loop)
                for task in pending:
                    task.cancel()
                # Wait for tasks to complete cancellation
                if pending:
                    self._loop.run_until_complete(
                        asyncio.gather(*pending, return_exceptions=True)
                    )
            except Exception:
                pass
            
            try:
                self._loop.close()
            except Exception:
                pass
    
    async def _connect_and_listen(self):
        """Connect to WebSocket and listen for messages."""
        if not WEBSOCKETS_AVAILABLE:
            logger.error("websockets not available")
            return
        
        retry_delay = 1
        max_retry_delay = 30
        
        while self._running:
            try:
                logger.info(f"Connecting to {self.ws_url}...")
                async with websockets.connect(self.ws_url) as ws:
                    self.ws = ws
                    retry_delay = 1
                    
                    logger.info("WebSocket connected!")
                    if self.on_connected:
                        self.on_connected()
                    
                    # Process outgoing messages and incoming messages
                    await asyncio.gather(
                        self._send_loop(),
                        self._receive_loop()
                    )
                    
            except Exception as e:
                logger.warning(f"WebSocket error: {e}")
                if self.on_disconnected:
                    self.on_disconnected()
            
            # Retry with exponential backoff
            if self._running:
                logger.info(f"Reconnecting in {retry_delay}s...")
                await asyncio.sleep(retry_delay)
                retry_delay = min(retry_delay * 2, max_retry_delay)
    
    async def _send_loop(self):
        """Send queued messages."""
        while self._running and self.ws:
            try:
                # Check queue with timeout
                await asyncio.sleep(0.05)
                
                while not self._message_queue.empty():
                    message = self._message_queue.get_nowait()
                    await self.ws.send(json.dumps(message))
                    
            except Exception as e:
                logger.error(f"Send error: {e}")
                break
    
    async def _receive_loop(self):
        """Receive and handle messages."""
        while self._running and self.ws:
            try:
                message = await self.ws.recv()
                data = json.loads(message)
                self._handle_message(data)
                
            except Exception as e:
                logger.error(f"Receive error: {e}")
                break
    
    def _handle_message(self, data: dict):
        """Handle incoming WebSocket message."""
        msg_type = data.get("type", "")
        
        if msg_type == "text_chunk":
            if self.on_text_chunk:
                self.on_text_chunk(data.get("content", ""))
                
        elif msg_type == "text_end":
            if self.on_text_end:
                self.on_text_end(data.get("full_text", ""))
                
        elif msg_type == "audio_data":
            if self.on_audio_data:
                self.on_audio_data(data)
                
        elif msg_type == "expression_change":
            if self.on_expression_change:
                self.on_expression_change(data.get("expression", "neutral"))
                
        elif msg_type == "transcription":
            if self.on_transcription:
                self.on_transcription(data.get("text", ""))
                
        elif msg_type == "vad_start":
            if self.on_vad_start:
                self.on_vad_start()
                
        elif msg_type == "vad_end":
            if self.on_vad_end:
                self.on_vad_end()
                
        elif msg_type == "error":
            if self.on_error:
                self.on_error(data.get("message", "Unknown error"))
    
    def send_text(self, text: str):
        """Send a text message."""
        self._message_queue.put({
            "type": "text",
            "content": text
        })
    
    def send_audio_stream(self, samples: list[float]):
        """Send audio samples for VAD processing."""
        self._message_queue.put({
            "type": "audio_stream",
            "samples": samples
        })
    
    def send_mic_stop(self):
        """Notify backend that microphone was stopped."""
        self._message_queue.put({"type": "mic_stop"})
    
    def send_clear(self):
        """Clear conversation history."""
        self._message_queue.put({"type": "clear"})
    
    def send_preload(self):
        """Request model preloading."""
        self._message_queue.put({"type": "preload_models"})


class DesktopCompanion:
    """
    Main desktop companion application.
    
    Manages the Live2D overlay window, system tray, hotkeys,
    and WebSocket connection to the AI backend.
    """
    
    def __init__(self, config: Optional[DesktopConfig] = None):
        self.config = config or DesktopConfig()
        
        # Components
        self.window: Optional[webview.Window] = None
        self.ws_client: Optional[WebSocketClient] = None
        self.tray: Optional[pystray.Icon] = None
        self.hotkey_listener = None
        
        # State
        self._running = False
        self._visible = True
        self._ready = threading.Event()
        self._lock = threading.Lock()
        
        # Audio state
        self._is_recording = False
        self._audio_queue = []
        
        # Calculate window position
        self._calculate_position()
        
        logger.info("DesktopCompanion initialized")
    
    def _calculate_position(self):
        """Calculate default window position (bottom-right)."""
        if self.config.x >= 0 and self.config.y >= 0:
            return
        
        # Default fallback values
        screen_w = 1920
        screen_h = 1080
        
        try:
            # Try tkinter first
            import tkinter as tk
            root = tk.Tk()
            root.withdraw()
            screen_w = root.winfo_screenwidth()
            screen_h = root.winfo_screenheight()
            root.destroy()
        except ImportError:
            # tkinter not available, try xrandr on Linux
            try:
                import subprocess
                result = subprocess.run(
                    ['xrandr', '--query'],
                    capture_output=True, text=True
                )
                for line in result.stdout.split('\n'):
                    if ' connected' in line and 'primary' in line:
                        # Parse resolution like "1920x1080+0+0"
                        import re
                        match = re.search(r'(\d+)x(\d+)', line)
                        if match:
                            screen_w = int(match.group(1))
                            screen_h = int(match.group(2))
                            break
            except Exception:
                pass
        except Exception as e:
            logger.warning(f"Could not get screen size: {e}")
        
        if self.config.x < 0:
            self.config.x = screen_w - self.config.width - 20
        if self.config.y < 0:
            self.config.y = screen_h - self.config.height - 50
    
    def start(self):
        """Start the desktop companion."""
        if not WEBVIEW_AVAILABLE:
            logger.error("pywebview is required! Install with: pip install pywebview")
            return
        
        self._running = True
        
        # Start WebSocket client
        self._start_ws_client()
        
        # Start system tray (in background thread)
        if self.config.enable_tray and TRAY_AVAILABLE:
            self._start_tray()
        
        # Start hotkeys
        if self.config.enable_hotkeys and PYNPUT_AVAILABLE:
            self._start_hotkeys()
        
        # Start webview (blocking)
        self._start_window()
    
    def stop(self):
        """Stop the desktop companion."""
        self._running = False
        
        if self.ws_client:
            self.ws_client.stop()
        
        if self.tray:
            self.tray.stop()
        
        if self.hotkey_listener:
            self.hotkey_listener.stop()
        
        if self.window:
            self.window.destroy()
    
    def _start_ws_client(self):
        """Initialize and start WebSocket client."""
        self.ws_client = WebSocketClient(
            self.config.backend_host,
            self.config.backend_port,
            client_id=f"desktop_{int(time.time())}"
        )
        
        # Set up callbacks
        self.ws_client.on_connected = self._on_ws_connected
        self.ws_client.on_disconnected = self._on_ws_disconnected
        self.ws_client.on_audio_data = self._on_audio_data
        self.ws_client.on_expression_change = self._on_expression_change
        self.ws_client.on_transcription = self._on_transcription
        self.ws_client.on_vad_start = self._on_vad_start
        self.ws_client.on_vad_end = self._on_vad_end
        
        self.ws_client.start()
    
    def _start_window(self):
        """Create and start the webview window."""
        # Use HTTP URL to load from backend (ensures scripts load correctly)
        backend_url = f"http://{self.config.backend_host}:{self.config.backend_port}"
        html_url = f"{backend_url}/frontend/desktop/index.html"
        logger.info(f"Loading: {html_url}")
        
        # Create API object for JS <-> Python communication
        api = DesktopAPI(self)
        
        # Create window
        self.window = webview.create_window(
            title="AI Companion",
            url=html_url,
            js_api=api,
            width=self.config.width,
            height=self.config.height,
            x=self.config.x,
            y=self.config.y,
            frameless=self.config.frameless,
            easy_drag=True,
            on_top=self.config.on_top,
            transparent=self.config.transparent,
            # Note: On Linux GTK, transparency is handled differently
            background_color='#1a1a24'  # Fallback for non-transparent mode
        )
        
        self.window.events.loaded += self._on_window_loaded
        self.window.events.closed += self._on_window_closed
        
        # Start webview (blocks until closed)
        webview.start(
            debug=self.config.debug,
            gui='gtk' if sys.platform == 'linux' else None
        )
    
    def _start_tray(self):
        """Start the system tray icon."""
        def create_tray():
            # Load or create icon
            icon_path = Path(self.config.icon_path)
            if icon_path.exists():
                image = Image.open(icon_path)
            else:
                # Create a simple icon
                image = Image.new('RGB', (64, 64), color=(99, 102, 241))
            
            # Create menu
            menu = pystray.Menu(
                pystray.MenuItem("Show/Hide (F12)", self.toggle_visibility),
                pystray.MenuItem("Toggle Microphone", self._toggle_microphone),
                pystray.Menu.SEPARATOR,
                pystray.MenuItem("Clear History", self._clear_history),
                pystray.MenuItem("Settings...", self._show_settings),
                pystray.Menu.SEPARATOR,
                pystray.MenuItem("Quit", self.stop)
            )
            
            self.tray = pystray.Icon(
                "AI Companion",
                image,
                "AI Companion",
                menu
            )
            
            self.tray.run()
        
        thread = threading.Thread(target=create_tray, daemon=True)
        thread.start()
    
    def _start_hotkeys(self):
        """Start global hotkey listener."""
        def on_press(key):
            try:
                if key == keyboard.Key.f12:
                    self.toggle_visibility()
                elif key == keyboard.Key.f11:
                    self._toggle_microphone()
            except Exception as e:
                logger.error(f"Hotkey error: {e}")
        
        self.hotkey_listener = keyboard.Listener(on_press=on_press)
        self.hotkey_listener.daemon = True
        self.hotkey_listener.start()
        logger.info("Hotkeys: F12=toggle visibility, F11=toggle microphone")
    
    # ==================== Event Handlers ====================
    
    def _on_window_loaded(self):
        """Called when webview is loaded."""
        self._ready.set()
        logger.info("Window loaded")
    
    def _on_window_closed(self):
        """Called when window is closed."""
        logger.info("Window closed")
        self.stop()
    
    def _on_ws_connected(self):
        """Called when WebSocket connects."""
        logger.info("WebSocket connected")
        self._eval_js("DesktopBridge.showStatus('Connected!')")
        self._eval_js("setTimeout(() => DesktopBridge.hideStatus(), 2000)")
        
        # Request model preload
        if self.ws_client:
            self.ws_client.send_preload()
    
    def _on_ws_disconnected(self):
        """Called when WebSocket disconnects."""
        logger.warning("WebSocket disconnected")
        self._eval_js("DesktopBridge.showStatus('Disconnected - Reconnecting...')")
    
    def _on_audio_data(self, data: dict):
        """Called when audio data is received."""
        audio_b64 = data.get("data", "")
        lip_sync = data.get("lip_sync", {})
        expression = data.get("expression", "")
        
        # Pass to JavaScript for playback
        lip_sync_json = json.dumps(lip_sync)
        js = f"DesktopBridge.playAudioWithLipSync('{audio_b64}', {lip_sync_json}, '{expression}')"
        self._eval_js(js)
    
    def _on_expression_change(self, expression: str):
        """Called when expression should change."""
        self._eval_js(f"Live2DManager.setExpression('{expression}')")
    
    def _on_transcription(self, text: str):
        """Called when transcription is received."""
        if text:
            # Show what user said (briefly)
            escaped = text.replace("'", "\\'").replace("\n", " ")
            self._eval_js(f"DesktopBridge.showSpeechBubble('🎤 {escaped}')")
    
    def _on_vad_start(self):
        """Called when VAD detects speech start."""
        # Visual indicator handled in frontend via vad_start message
        pass
    
    def _on_vad_end(self):
        """Called when VAD detects speech end."""
        # Visual indicator handled in frontend via vad_end message
        pass
    
    # ==================== Public Methods ====================
    
    def toggle_visibility(self):
        """Toggle window visibility."""
        if not self.window:
            return
        
        with self._lock:
            if self._visible:
                self.window.hide()
                self._visible = False
            else:
                self.window.show()
                self._visible = True
    
    def _toggle_microphone(self):
        """Toggle microphone recording."""
        self._is_recording = not self._is_recording
        
        # Toggle mic button style in frontend
        if self._is_recording:
            self._eval_js("document.getElementById('mic-btn')?.classList.add('recording')")
        else:
            self._eval_js("document.getElementById('mic-btn')?.classList.remove('recording')")
            if self.ws_client:
                self.ws_client.send_mic_stop()
    
    def _clear_history(self):
        """Clear conversation history."""
        if self.ws_client:
            self.ws_client.send_clear()
        self._eval_js("DesktopBridge.showSpeechBubble('🗑️ History cleared')")
    
    def _show_settings(self):
        """Show settings dialog."""
        # TODO: Implement settings dialog
        logger.info("Settings requested")
    
    def _eval_js(self, script: str):
        """Safely evaluate JavaScript in the window."""
        if self.window and self._ready.is_set():
            try:
                self.window.evaluate_js(script)
            except Exception as e:
                logger.error(f"JS eval error: {e}")
    
    def send_text(self, text: str):
        """Send text message to AI."""
        if self.ws_client:
            self.ws_client.send_text(text)
    
    def set_expression(self, expression: str):
        """Set avatar expression."""
        self._eval_js(f"Live2DManager.setExpression('{expression}')")


class DesktopAPI:
    """
    Python API exposed to JavaScript via pywebview.
    
    Methods here can be called from JS: pywebview.api.method_name()
    
    NOTE: Only simple methods are exposed. We don't store complex objects
    as attributes to avoid pywebview serialization errors with __weakref__.
    """
    
    def __init__(self, companion: DesktopCompanion):
        # Store weak reference to avoid serialization issues
        import weakref
        self._companion_ref = weakref.ref(companion)
    
    @property
    def _companion(self) -> Optional[DesktopCompanion]:
        return self._companion_ref()
    
    def on_ready(self):
        """Called by JS when Live2D is ready."""
        logger.info("Live2D ready (from JS)")
    
    def send_text(self, text: str):
        """Send text message to AI."""
        if self._companion:
            self._companion.send_text(text)
    
    def toggle_visibility(self):
        """Toggle window visibility."""
        if self._companion:
            self._companion.toggle_visibility()
    
    def clear_history(self):
        """Clear conversation history."""
        if self._companion:
            self._companion._clear_history()
    
    def quit(self):
        """Quit the application."""
        if self._companion:
            self._companion.stop()
