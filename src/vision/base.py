"""
Vision Base Classes - Abstract interfaces for visual perception.

Following the project's architecture patterns:
- Abstract Base Class (ABC) for provider interface
- Dataclass for structured results
- Type hints throughout
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Any
from enum import Enum
import base64


class CaptureType(Enum):
    """Type of visual capture."""
    SCREENSHOT = "screenshot"      # Single frame capture
    SCREEN_REGION = "region"       # Capture specific region
    WEBCAM = "webcam"              # Camera input
    SCREEN_RECORD = "recording"    # Video recording (future)


@dataclass
class CaptureResult:
    """
    Result from a visual capture operation.
    
    Attributes:
        image_data: Raw image bytes (PNG/JPEG)
        width: Image width in pixels
        height: Image height in pixels
        format: Image format (png, jpeg, webp)
        capture_type: Type of capture performed
        timestamp: Unix timestamp of capture
        base64_data: Base64 encoded image for LLM APIs
        description: Optional AI-generated description
        metadata: Additional capture metadata
    """
    image_data: bytes
    width: int
    height: int
    format: str = "png"
    capture_type: CaptureType = CaptureType.SCREENSHOT
    timestamp: float = 0.0
    base64_data: str = ""
    description: Optional[str] = None
    metadata: dict = field(default_factory=dict)
    
    def __post_init__(self):
        """Generate base64 data if not provided."""
        if not self.base64_data and self.image_data:
            self.base64_data = base64.b64encode(self.image_data).decode('utf-8')
    
    def to_data_url(self) -> str:
        """Convert to data URL for HTML/API use."""
        mime_type = f"image/{self.format}"
        return f"data:{mime_type};base64,{self.base64_data}"
    
    def save(self, path: Path) -> Path:
        """Save capture to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(self.image_data)
        return path


class VisionProvider(ABC):
    """
    Abstract base class for vision providers.
    
    Implements Strategy pattern for interchangeable vision backends.
    Supports screen capture, camera input, and visual analysis.
    
    Example usage:
        provider = ScreenCaptureProvider()
        result = provider.capture()
        description = provider.analyze(result, "What do you see?")
    """
    
    @abstractmethod
    def capture(
        self,
        region: Optional[tuple[int, int, int, int]] = None,
        monitor: int = 0
    ) -> CaptureResult:
        """
        Capture visual input (screen or camera).
        
        Args:
            region: Optional (x, y, width, height) for partial capture
            monitor: Monitor index for multi-monitor setups
            
        Returns:
            CaptureResult with image data and metadata
        """
        pass
    
    @abstractmethod
    def analyze(
        self,
        capture: CaptureResult,
        prompt: str,
        context: Optional[str] = None
    ) -> str:
        """
        Analyze captured image using multimodal LLM.
        
        Args:
            capture: CaptureResult from capture()
            prompt: Question/instruction about the image
            context: Optional conversation context
            
        Returns:
            LLM's analysis/description of the image
        """
        pass
    
    def capture_and_analyze(
        self,
        prompt: str,
        region: Optional[tuple[int, int, int, int]] = None,
        context: Optional[str] = None
    ) -> tuple[CaptureResult, str]:
        """
        Convenience method: capture then analyze in one call.
        
        Args:
            prompt: Question about what's on screen
            region: Optional capture region
            context: Optional conversation context
            
        Returns:
            Tuple of (CaptureResult, analysis_text)
        """
        capture = self.capture(region=region)
        analysis = self.analyze(capture, prompt, context)
        capture.description = analysis
        return capture, analysis
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name for logging."""
        pass
    
    @property
    def supports_camera(self) -> bool:
        """Whether this provider supports camera input."""
        return False
    
    @property
    def supports_screen(self) -> bool:
        """Whether this provider supports screen capture."""
        return True
