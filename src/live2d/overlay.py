#!/usr/bin/env python3
"""
Live2D Overlay for Local AI Companion

A transparent, frameless, always-on-top window displaying a Live2D avatar
using pywebview. Integrates with the AI pipeline for lip-sync and expressions.

Features:
- Transparent background with GPU-accelerated WebGL rendering
- Hotkeys for visibility toggle (F12) and window mode (F11)
- Expression control based on LLM output
- Lip-sync from TTS audio

Usage:
    python overlay.py                    # Start overlay with default settings
    python overlay.py --debug            # Enable debug mode
    python overlay.py --position 100 100 # Set initial position
    python overlay.py --size 400 600     # Set window size
"""

import argparse
import logging
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add project root to path (go up from src/live2d/ to project root)
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import webview
except ImportError:
    logger.error("pywebview is not installed. Install with: pip install pywebview")
    sys.exit(1)

try:
    from pynput import keyboard
except ImportError:
    logger.warning("pynput is not installed. Hotkeys will be disabled.")
    keyboard = None


@dataclass
class OverlayConfig:
    """Configuration for the Live2D overlay window."""
    width: int = 400
    height: int = 600
    x: int = -1  # -1 means auto-position (right side of screen)
    y: int = -1  # -1 means auto-position (bottom)
    opacity: float = 0.95
    frameless: bool = True
    on_top: bool = True
    transparent: bool = True
    debug: bool = False
    html_path: str = field(default_factory=lambda: str(
        PROJECT_ROOT / 'frontend' / 'live2d' / 'index.html'
    ))


class Live2DOverlay:
    """
    Live2D overlay window manager.
    
    Handles the pywebview window lifecycle, hotkeys, and JavaScript API
    for controlling the Live2D avatar from Python.
    """
    
    def __init__(self, config: Optional[OverlayConfig] = None):
        """Initialize the overlay with configuration."""
        self.config = config or OverlayConfig()
        self.window: Optional[webview.Window] = None
        self._visible = True
        self._frameless = self.config.frameless
        self._hotkey_listener = None
        self._ready = threading.Event()
        self._lock = threading.Lock()
        
        # Callbacks for external integration
        self._on_ready_callbacks: list[Callable] = []
        self._on_close_callbacks: list[Callable] = []
        
        # Calculate default position (bottom-right corner)
        if self.config.x < 0 or self.config.y < 0:
            self._calculate_default_position()
    
    def _calculate_default_position(self):
        """Calculate default position for the window."""
        try:
            # Try to get screen size
            import tkinter as tk
            root = tk.Tk()
            root.withdraw()
            screen_width = root.winfo_screenwidth()
            screen_height = root.winfo_screenheight()
            root.destroy()
            
            if self.config.x < 0:
                self.config.x = screen_width - self.config.width - 50
            if self.config.y < 0:
                self.config.y = screen_height - self.config.height - 100
                
        except Exception as e:
            logger.warning(f"Could not get screen size: {e}")
            self.config.x = 100 if self.config.x < 0 else self.config.x
            self.config.y = 100 if self.config.y < 0 else self.config.y
    
    def _on_loaded(self):
        """Called when the webview content is loaded."""
        logger.info("Live2D overlay loaded")
        self._ready.set()
        
        # Execute ready callbacks
        for callback in self._on_ready_callbacks:
            try:
                callback(self)
            except Exception as e:
                logger.error(f"Ready callback error: {e}")
    
    def _on_closed(self):
        """Called when the window is closed."""
        logger.info("Live2D overlay closed")
        self._stop_hotkeys()
        
        # Execute close callbacks
        for callback in self._on_close_callbacks:
            try:
                callback(self)
            except Exception as e:
                logger.error(f"Close callback error: {e}")
    
    def _setup_hotkeys(self):
        """Set up global hotkeys using pynput."""
        if keyboard is None:
            logger.warning("pynput not available, hotkeys disabled")
            return
        
        def on_press(key):
            try:
                if key == keyboard.Key.f12:
                    self.toggle_visibility()
                elif key == keyboard.Key.f11:
                    self.toggle_frameless()
            except Exception as e:
                logger.error(f"Hotkey error: {e}")
        
        self._hotkey_listener = keyboard.Listener(on_press=on_press)
        self._hotkey_listener.daemon = True
        self._hotkey_listener.start()
        logger.info("Hotkeys enabled: F12=toggle visibility, F11=toggle frameless")
    
    def _stop_hotkeys(self):
        """Stop the hotkey listener."""
        if self._hotkey_listener:
            self._hotkey_listener.stop()
            self._hotkey_listener = None
    
    def start(self, blocking: bool = True):
        """
        Start the overlay window.
        
        Args:
            blocking: If True, blocks until window is closed.
                      If False, starts in a background thread.
        """
        if blocking:
            self._start_window()
        else:
            thread = threading.Thread(target=self._start_window, daemon=True)
            thread.start()
            # Wait for window to be ready
            self._ready.wait(timeout=10)
    
    def _start_window(self):
        """Create and start the webview window."""
        # Resolve HTML path
        html_path = Path(self.config.html_path)
        if not html_path.is_absolute():
            html_path = PROJECT_ROOT / html_path
        
        if not html_path.exists():
            # Try relative to this file
            html_path = Path(__file__).parent.parent / 'frontend' / 'live2d' / 'index.html'
        
        if not html_path.exists():
            raise FileNotFoundError(f"HTML file not found: {html_path}")
        
        html_url = html_path.as_uri()
        logger.info(f"Loading: {html_url}")
        
        # Create window
        self.window = webview.create_window(
            title='Live2D Avatar',
            url=html_url,
            width=self.config.width,
            height=self.config.height,
            x=self.config.x,
            y=self.config.y,
            frameless=self.config.frameless,
            easy_drag=True,
            on_top=self.config.on_top,
            transparent=self.config.transparent
            # Note: background_color not needed when transparent=True
        )
        
        # Set up event handlers
        self.window.events.loaded += self._on_loaded
        self.window.events.closed += self._on_closed
        
        # Start hotkeys
        self._setup_hotkeys()
        
        # Start webview (blocking)
        webview.start(
            debug=self.config.debug,
            gui='gtk' if sys.platform == 'linux' else None,  # GTK for Linux transparency
            private_mode=False
        )
    
    def wait_ready(self, timeout: float = 10.0) -> bool:
        """Wait for the overlay to be ready."""
        return self._ready.wait(timeout=timeout)
    
    def toggle_visibility(self):
        """Toggle window visibility."""
        if self.window is None:
            return
        
        with self._lock:
            if self._visible:
                self.window.hide()
                self._visible = False
                logger.info("Overlay hidden")
            else:
                self.window.show()
                self._visible = True
                logger.info("Overlay shown")
    
    def toggle_frameless(self):
        """Toggle between frameless and windowed mode."""
        if self.window is None:
            return
        
        # Note: pywebview doesn't support changing frameless at runtime
        # We need to recreate the window
        logger.info("Toggle frameless requested (requires window restart)")
        self._frameless = not self._frameless
        # For now, just log - full implementation would save state and recreate window
    
    def set_visible(self, visible: bool):
        """Set window visibility."""
        if self.window is None:
            return
        
        with self._lock:
            if visible and not self._visible:
                self.window.show()
                self._visible = True
            elif not visible and self._visible:
                self.window.hide()
                self._visible = False
    
    def close(self):
        """Close the overlay window."""
        if self.window:
            self.window.destroy()
    
    # ==================== JavaScript API ====================
    
    def evaluate_js(self, script: str) -> any:
        """
        Execute JavaScript in the webview.
        
        Args:
            script: JavaScript code to execute
            
        Returns:
            Result from the JavaScript execution
        """
        if self.window is None:
            logger.warning("Window not ready, cannot evaluate JS")
            return None
        
        try:
            return self.window.evaluate_js(script)
        except Exception as e:
            logger.error(f"JS evaluation error: {e}")
            return None
    
    def set_expression(self, expression: str):
        """
        Set the avatar's expression.
        
        Args:
            expression: Expression name (e.g., "happy", "sad", "blush")
        """
        self.evaluate_js(f"Live2DAPI.setExpression('{expression}')")
        logger.debug(f"Expression set: {expression}")
    
    def set_lip_sync(self, value: float):
        """
        Set lip sync value directly.
        
        Args:
            value: Lip sync value (0.0 - 1.0)
        """
        value = max(0.0, min(1.0, value))
        self.evaluate_js(f"Live2DAPI.setLipSync({value})")
    
    def start_lip_sync_from_audio(self):
        """Start lip sync from microphone audio analysis."""
        self.evaluate_js("Live2DAPI.startLipSyncFromAudio()")
    
    def stop_lip_sync(self):
        """Stop lip sync."""
        self.evaluate_js("Live2DAPI.stopLipSync()")
    
    def set_parameter(self, name: str, value: float):
        """
        Set a model parameter directly.
        
        Args:
            name: Parameter name (e.g., "ParamAngleX")
            value: Parameter value
        """
        self.evaluate_js(f"Live2DAPI.setParameter('{name}', {value})")
    
    def play_motion(self, group: str, index: int = 0):
        """
        Play a motion animation.
        
        Args:
            group: Motion group name
            index: Motion index within the group
        """
        self.evaluate_js(f"Live2DAPI.playMotion('{group}', {index})")
    
    def toggle_debug(self):
        """Toggle debug overlay in the Live2D view."""
        self.evaluate_js("Live2DAPI.toggleDebug()")
    
    def get_expressions(self) -> list:
        """Get list of available expressions."""
        result = self.evaluate_js("Live2DAPI.getExpressions()")
        return result if result else []
    
    # ==================== Callbacks ====================
    
    def on_ready(self, callback: Callable):
        """Register a callback for when the overlay is ready."""
        self._on_ready_callbacks.append(callback)
    
    def on_close(self, callback: Callable):
        """Register a callback for when the overlay is closed."""
        self._on_close_callbacks.append(callback)


class ExpressionMapper:
    """
    Maps LLM output emotions to Live2D expressions.
    
    Parses text for emotion markers like (happy), (sad), etc.
    and maps them to model-specific expression names.
    """
    
    # Default emotion to expression mapping
    DEFAULT_MAPPING = {
        # English emotions
        'happy': 'happy',
        'excited': 'excited',
        'sad': 'sad',
        'cry': 'cry',
        'crying': 'cry',
        'angry': 'angry',
        'shy': 'shy',
        'blush': 'blush',
        'blushing': 'blush',
        'embarrassed': 'sweat',
        'nervous': 'sweat',
        'surprised': 'excited',
        'thinking': 'sweat',
        'peace': 'peace',
        'photo': 'photo',
        'neutral': None,  # Reset to default
    }
    
    def __init__(self, custom_mapping: Optional[dict] = None):
        """Initialize with optional custom mapping."""
        self.mapping = {**self.DEFAULT_MAPPING}
        if custom_mapping:
            self.mapping.update(custom_mapping)
    
    def parse_emotion_from_text(self, text: str) -> Optional[str]:
        """
        Parse emotion markers from text.
        
        Looks for patterns like (happy), [sad], *excited*
        
        Args:
            text: Input text to parse
            
        Returns:
            Emotion name if found, None otherwise
        """
        import re
        
        # Pattern for emotion markers
        patterns = [
            r'\((\w+)\)',    # (happy)
            r'\[(\w+)\]',    # [sad]
            r'\*(\w+)\*',    # *excited*
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text.lower())
            for match in matches:
                if match in self.mapping:
                    return match
        
        return None
    
    def get_expression(self, emotion: str) -> Optional[str]:
        """
        Get the expression name for an emotion.
        
        Args:
            emotion: Emotion name
            
        Returns:
            Expression name or None
        """
        return self.mapping.get(emotion.lower())


class AudioLipSyncController:
    """
    Controller for syncing lip movements with audio playback.
    
    Analyzes audio data in real-time to generate lip sync values.
    """
    
    def __init__(self, overlay: Live2DOverlay):
        """Initialize with overlay reference."""
        self.overlay = overlay
        self._running = False
        self._thread = None
    
    def start_from_file(self, audio_path: str):
        """
        Start lip sync from an audio file.
        
        This uses the Web Audio API in the browser.
        """
        # Load and analyze audio in JS
        js_code = f"""
        (async () => {{
            const audio = new Audio('{audio_path}');
            Live2DManager.connectAudioElement(audio);
            audio.play();
        }})();
        """
        self.overlay.evaluate_js(js_code)
    
    def start_from_bytes(self, audio_data: bytes, sample_rate: int = 24000):
        """
        Start lip sync from audio bytes (e.g., from TTS).
        
        Analyzes audio volume to generate lip sync values.
        """
        self._running = True
        
        def analyze_audio():
            import numpy as np
            try:
                # Convert bytes to numpy array
                audio = np.frombuffer(audio_data, dtype=np.int16)
                
                # Calculate chunk size (~60fps update)
                chunk_samples = sample_rate // 60
                
                for i in range(0, len(audio), chunk_samples):
                    if not self._running:
                        break
                    
                    chunk = audio[i:i + chunk_samples]
                    if len(chunk) == 0:
                        break
                    
                    # Calculate RMS (root mean square) for volume
                    rms = np.sqrt(np.mean(chunk.astype(np.float32) ** 2))
                    
                    # Normalize to 0-1 range (assuming 16-bit audio)
                    normalized = min(1.0, rms / 8000)
                    
                    # Apply to overlay
                    self.overlay.set_lip_sync(normalized)
                    
                    # Wait for next frame
                    time.sleep(1.0 / 60)
                
                # Reset lip sync when done
                self.overlay.set_lip_sync(0)
                
            except Exception as e:
                logger.error(f"Audio analysis error: {e}")
        
        self._thread = threading.Thread(target=analyze_audio, daemon=True)
        self._thread.start()
    
    def stop(self):
        """Stop lip sync analysis."""
        self._running = False
        self.overlay.stop_lip_sync()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Live2D Avatar Overlay for Local AI Companion'
    )
    parser.add_argument(
        '--debug', action='store_true',
        help='Enable debug mode'
    )
    parser.add_argument(
        '--size', nargs=2, type=int, default=[400, 600],
        metavar=('WIDTH', 'HEIGHT'),
        help='Window size (default: 400 600)'
    )
    parser.add_argument(
        '--position', nargs=2, type=int, default=[-1, -1],
        metavar=('X', 'Y'),
        help='Window position (default: bottom-right)'
    )
    parser.add_argument(
        '--opacity', type=float, default=0.95,
        help='Window opacity 0.0-1.0 (default: 0.95)'
    )
    parser.add_argument(
        '--no-frameless', action='store_true',
        help='Start in windowed mode (with title bar)'
    )
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    config = OverlayConfig(
        width=args.size[0],
        height=args.size[1],
        x=args.position[0],
        y=args.position[1],
        opacity=args.opacity,
        frameless=not args.no_frameless,
        debug=args.debug
    )
    
    logger.info("Starting Live2D Overlay...")
    logger.info(f"  Size: {config.width}x{config.height}")
    logger.info(f"  Position: {config.x}, {config.y}")
    logger.info(f"  Frameless: {config.frameless}")
    logger.info(f"  Debug: {config.debug}")
    
    overlay = Live2DOverlay(config)
    
    def on_ready(overlay):
        logger.info("Overlay ready! Press F12 to toggle, F11 to change mode")
        # Example: Set initial expression
        # overlay.set_expression('happy')
    
    overlay.on_ready(on_ready)
    
    try:
        overlay.start(blocking=True)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Error: {e}")
        raise


if __name__ == '__main__':
    main()
