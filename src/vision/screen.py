"""
Screen Capture Provider - Local screen capture with multimodal LLM analysis.

Uses:
- mss (cross-platform screen capture, fast)
- PIL for image processing
- Multimodal LLM (Jan-v2-VL, LLaVA, etc.) for analysis

This is a 100% local solution - no data leaves your machine.
"""

import time
from pathlib import Path
from typing import Optional
from io import BytesIO
import logging

from .base import VisionProvider, CaptureResult, CaptureType

logger = logging.getLogger(__name__)


class ScreenCaptureProvider(VisionProvider):
    """
    Screen capture provider using mss (cross-platform).
    
    Integrates with multimodal LLMs for visual understanding:
    - Jan-v2-VL (via llama.cpp server)
    - LLaVA
    - Any OpenAI-compatible vision API
    
    Example:
        provider = ScreenCaptureProvider(
            llm_base_url="http://localhost:8080",  # llama.cpp with Jan-v2-VL
            llm_model="jan-v2-vl-high"
        )
        capture, analysis = provider.capture_and_analyze("What application is open?")
    """
    
    def __init__(
        self,
        llm_base_url: str = "http://localhost:8080",
        llm_model: str = "jan-v2-vl-high",
        llm_api_key: Optional[str] = None,
        image_format: str = "png",
        image_quality: int = 85,
        max_dimension: int = 1920,  # Resize large screens for faster processing
    ):
        """
        Initialize screen capture provider.
        
        Args:
            llm_base_url: URL of multimodal LLM server
            llm_model: Model name for vision tasks
            llm_api_key: Optional API key
            image_format: Output format (png, jpeg, webp)
            image_quality: JPEG/WebP quality (1-100)
            max_dimension: Max width/height (resize if larger)
        """
        self.llm_base_url = llm_base_url.rstrip('/')
        self.llm_model = llm_model
        self.llm_api_key = llm_api_key
        self.image_format = image_format
        self.image_quality = image_quality
        self.max_dimension = max_dimension
        
        # Lazy imports
        self._mss = None
        self._Image = None
        self._httpx = None
    
    def _ensure_imports(self):
        """Lazy load heavy dependencies."""
        if self._mss is None:
            try:
                import mss
                self._mss = mss.mss()
            except ImportError:
                raise ImportError(
                    "mss is required for screen capture. "
                    "Install with: pip install mss"
                )
        
        if self._Image is None:
            try:
                from PIL import Image
                self._Image = Image
            except ImportError:
                raise ImportError(
                    "Pillow is required for image processing. "
                    "Install with: pip install Pillow"
                )
        
        if self._httpx is None:
            import httpx
            self._httpx = httpx
    
    @property
    def name(self) -> str:
        return "ScreenCapture"
    
    @property
    def supports_screen(self) -> bool:
        return True
    
    @property
    def supports_camera(self) -> bool:
        return False  # Use CameraProvider for webcam
    
    def capture(
        self,
        region: Optional[tuple[int, int, int, int]] = None,
        monitor: int = 0
    ) -> CaptureResult:
        """
        Capture screen content.
        
        Args:
            region: Optional (x, y, width, height) tuple
            monitor: Monitor index (0 = all monitors, 1+ = specific)
            
        Returns:
            CaptureResult with image data
        """
        self._ensure_imports()
        
        timestamp = time.time()
        
        # Capture screen
        if region:
            x, y, width, height = region
            capture_region = {
                "left": x,
                "top": y,
                "width": width,
                "height": height
            }
            shot = self._mss.grab(capture_region)
            capture_type = CaptureType.SCREEN_REGION
        else:
            # Capture specific monitor or all
            monitors = self._mss.monitors
            if monitor > 0 and monitor < len(monitors):
                shot = self._mss.grab(monitors[monitor])
            else:
                shot = self._mss.grab(monitors[0])  # Primary monitor
            capture_type = CaptureType.SCREENSHOT
        
        # Convert to PIL Image
        img = self._Image.frombytes("RGB", shot.size, shot.bgra, "raw", "BGRX")
        
        # Resize if too large
        if max(img.size) > self.max_dimension:
            ratio = self.max_dimension / max(img.size)
            new_size = (int(img.width * ratio), int(img.height * ratio))
            img = img.resize(new_size, self._Image.Resampling.LANCZOS)
            logger.debug(f"Resized image to {new_size}")
        
        # Convert to bytes
        buffer = BytesIO()
        save_kwargs = {}
        if self.image_format in ("jpeg", "webp"):
            save_kwargs["quality"] = self.image_quality
        img.save(buffer, format=self.image_format.upper(), **save_kwargs)
        image_data = buffer.getvalue()
        
        return CaptureResult(
            image_data=image_data,
            width=img.width,
            height=img.height,
            format=self.image_format,
            capture_type=capture_type,
            timestamp=timestamp,
            metadata={
                "monitor": monitor,
                "region": region,
                "original_size": shot.size,
            }
        )
    
    def analyze(
        self,
        capture: CaptureResult,
        prompt: str,
        context: Optional[str] = None
    ) -> str:
        """
        Analyze image using multimodal LLM.
        
        Sends image to llama.cpp server with Jan-v2-VL or similar model.
        
        Args:
            capture: CaptureResult from capture()
            prompt: Question about the image
            context: Optional conversation context
            
        Returns:
            LLM's analysis text
        """
        self._ensure_imports()
        
        # Build messages for vision API (OpenAI format)
        messages = []
        
        # Add context if provided
        if context:
            messages.append({
                "role": "system",
                "content": context
            })
        
        # Add user message with image
        messages.append({
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": capture.to_data_url()
                    }
                },
                {
                    "type": "text",
                    "text": prompt
                }
            ]
        })
        
        # Send request to LLM
        headers = {"Content-Type": "application/json"}
        if self.llm_api_key:
            headers["Authorization"] = f"Bearer {self.llm_api_key}"
        
        payload = {
            "model": self.llm_model,
            "messages": messages,
            "max_tokens": 1024,
            "temperature": 0.7,
        }
        
        try:
            response = self._httpx.post(
                f"{self.llm_base_url}/v1/chat/completions",
                json=payload,
                headers=headers,
                timeout=60.0  # Vision models can be slow
            )
            response.raise_for_status()
            
            result = response.json()
            return result["choices"][0]["message"]["content"]
            
        except Exception as e:
            logger.error(f"Vision analysis failed: {e}")
            raise RuntimeError(f"Failed to analyze image: {e}")
    
    def list_monitors(self) -> list[dict]:
        """
        List available monitors.
        
        Returns:
            List of monitor info dicts with x, y, width, height
        """
        self._ensure_imports()
        return [
            {
                "index": i,
                "x": m["left"],
                "y": m["top"],
                "width": m["width"],
                "height": m["height"],
            }
            for i, m in enumerate(self._mss.monitors)
        ]
