"""
NVIDIA Canary ASR Provider using NeMo toolkit.

Canary-1b-v2 is a powerful 1-billion parameter model built for high-quality 
speech transcription and translation across 25 European languages.

Features:
- State-of-the-art ASR performance (WER ~5% for French)
- 25 European languages supported
- Automatic punctuation and capitalization
- Word-level and segment-level timestamps
- GPU accelerated (requires NVIDIA GPU with 6GB+ VRAM)

Requirements:
- NVIDIA GPU with CUDA support
- nemo_toolkit[asr] package
"""

import asyncio
import numpy as np
from pathlib import Path
from typing import Optional, List, Union, AsyncGenerator
import tempfile
import wave

from .base import BaseASR, ASRResult, ASRSegment


# Supported languages for Canary 1B v2
CANARY_LANGUAGES = [
    "bg", "hr", "cs", "da", "nl", "en", "et", "fi", "fr", "de",
    "el", "hu", "it", "lv", "lt", "mt", "pl", "pt", "ro", "sk",
    "sl", "es", "sv", "ru", "uk"
]


class CanaryProvider(BaseASR):
    """
    ASR provider using NVIDIA Canary 1B v2 for speech-to-text.
    
    This is a state-of-the-art multilingual ASR model that provides
    excellent accuracy with automatic punctuation and capitalization.
    
    Args:
        model_name: HuggingFace model name (default: nvidia/canary-1b-v2)
        device: Device to run on ("cuda" or "cpu") - cuda strongly recommended
        
    Example:
        >>> asr = CanaryProvider()
        >>> result = asr.transcribe("audio.wav", language="fr")
        >>> print(result.text)
        "Bonjour, comment ça va ?"
    """
    
    def __init__(
        self,
        model_name: str = "nvidia/canary-1b-v2",
        device: str = "cuda"
    ):
        self.model_name = model_name
        self.device = device
        self._model = None
        
    def _get_model(self):
        """Lazy load the Canary model."""
        if self._model is None:
            try:
                from nemo.collections.asr.models import ASRModel
            except ImportError:
                raise ImportError(
                    "NeMo toolkit is not installed. "
                    "Install it with: pip install nemo_toolkit[asr]"
                )
            
            print(f"🔄 Loading Canary 1B v2 (this may take a moment)...")
            
            # Load model from HuggingFace
            self._model = ASRModel.from_pretrained(model_name=self.model_name)
            
            # Move to appropriate device
            if self.device == "cuda":
                import torch
                if torch.cuda.is_available():
                    self._model = self._model.cuda()
                    print(f"✅ Canary 1B v2 loaded on GPU!")
                else:
                    print("⚠️ CUDA not available, using CPU (slow)")
                    self._model = self._model.cpu()
            else:
                self._model = self._model.cpu()
                print(f"✅ Canary 1B v2 loaded on CPU (slow)")
            
            # Set to evaluation mode
            self._model.eval()
            
        return self._model
    
    def transcribe(
        self,
        audio_input: Union[str, Path, np.ndarray],
        language: Optional[str] = None,
        initial_prompt: Optional[str] = None  # Not used but kept for API compatibility
    ) -> ASRResult:
        """
        Transcribe audio to text using Canary 1B v2.
        
        Args:
            audio_input: Path to audio file or numpy array of samples (float32, 16kHz)
            language: Source language code (e.g., "fr"). Default: "fr"
            initial_prompt: Not used (kept for API compatibility)
            
        Returns:
            ASRResult with transcribed text and metadata
        """
        model = self._get_model()
        
        # Default to French
        source_lang = language if language and language.lower() != "auto" else "fr"
        
        # Validate language
        if source_lang not in CANARY_LANGUAGES:
            print(f"⚠️ Language '{source_lang}' not supported by Canary, using 'fr'")
            source_lang = "fr"
        
        # Handle numpy array input - save to temp file
        temp_file = None
        if isinstance(audio_input, np.ndarray):
            # Create temp WAV file
            temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            audio_path = Path(temp_file.name)
            temp_file.close()
            
            # Convert float32 to int16 if needed
            if audio_input.dtype == np.float32:
                audio_int16 = (audio_input * 32767).astype(np.int16)
            else:
                audio_int16 = audio_input.astype(np.int16)
            
            # Write WAV file
            with wave.open(str(audio_path), 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(16000)
                wf.writeframes(audio_int16.tobytes())
            
            print(f"🎤 Transcribing audio buffer ({len(audio_input)} samples)")
        else:
            audio_path = Path(audio_input)
            if not audio_path.exists():
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
            print(f"🎤 Transcribing {audio_path.name}")
        
        try:
            # Transcribe with Canary
            # For ASR (same source and target language)
            output = model.transcribe(
                [str(audio_path)],
                source_lang=source_lang,
                target_lang=source_lang,  # Same for ASR
            )
            
            # Extract text from output
            if output and len(output) > 0:
                # NeMo returns a list of hypothesis objects
                if hasattr(output[0], 'text'):
                    text = output[0].text
                else:
                    text = str(output[0])
            else:
                text = ""
            
            print(f"   📝 '{text}'")
            
            return ASRResult(
                text=text.strip(),
                language=source_lang,
                confidence=1.0,  # Canary doesn't provide confidence scores directly
                duration=None,
                segments=[]
            )
            
        finally:
            # Cleanup temp file
            if temp_file:
                Path(temp_file.name).unlink(missing_ok=True)
    
    def get_supported_languages(self) -> List[str]:
        """Get list of supported language codes."""
        return CANARY_LANGUAGES.copy()
    
    async def transcribe_stream(
        self,
        audio_path: Union[str, Path],
        language: Optional[str] = None
    ) -> AsyncGenerator[ASRSegment, None]:
        """
        Transcribe audio and yield segments as they're processed.
        
        Note: Canary processes audio in one pass, so this is simulated streaming.
        For true streaming, consider using Whisper with word timestamps.
        """
        # Transcribe the full audio first
        result = self.transcribe(audio_path, language=language)
        
        # Yield the result as a single segment
        if result.text:
            yield ASRSegment(
                text=result.text,
                start=0.0,
                end=result.duration or 0.0,
                confidence=result.confidence
            )
        
        # Small delay to make it feel async
        await asyncio.sleep(0.01)
    
    def get_model_info(self) -> dict:
        """Get information about the model."""
        return {
            "model_name": self.model_name,
            "params": "1B",
            "languages": len(CANARY_LANGUAGES),
            "device": self.device,
            "loaded": self._model is not None,
            "features": [
                "Automatic punctuation",
                "Automatic capitalization", 
                "Word-level timestamps",
                "25 European languages"
            ]
        }


def create_canary_asr(device: str = "cuda") -> CanaryProvider:
    """
    Factory function to create a Canary ASR provider.
    
    Args:
        device: "cuda" (recommended) or "cpu"
        
    Returns:
        CanaryProvider instance
    """
    return CanaryProvider(device=device)
