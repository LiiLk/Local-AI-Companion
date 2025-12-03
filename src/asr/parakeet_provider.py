"""
NVIDIA Parakeet ASR Provider using NeMo toolkit.

Parakeet-TDT-0.6B-v3 is a 600-million-parameter multilingual ASR model
designed for high-throughput speech-to-text transcription.

Features:
- State-of-the-art ASR performance (WER ~5% for French)
- 25 European languages with auto-detection
- Automatic punctuation and capitalization
- Word-level and segment-level timestamps
- Smaller and faster than Canary (600M vs 1B params)
- GPU accelerated (requires NVIDIA GPU with ~2GB VRAM)

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


# Supported languages for Parakeet TDT v3 (auto-detected)
PARAKEET_LANGUAGES = [
    "bg", "hr", "cs", "da", "nl", "en", "et", "fi", "fr", "de",
    "el", "hu", "it", "lv", "lt", "mt", "pl", "pt", "ro", "sk",
    "sl", "es", "sv", "ru", "uk"
]


class ParakeetProvider(BaseASR):
    """
    ASR provider using NVIDIA Parakeet TDT 0.6B v3 for speech-to-text.
    
    This is a fast and accurate multilingual ASR model that automatically
    detects language and provides punctuation/capitalization.
    
    Args:
        model_name: HuggingFace model name (default: nvidia/parakeet-tdt-0.6b-v3)
        device: Device to run on ("cuda" or "cpu") - cuda strongly recommended
        
    Example:
        >>> asr = ParakeetProvider()
        >>> result = asr.transcribe("audio.wav")
        >>> print(result.text)
        "Bonjour, comment ça va ?"
    """
    
    def __init__(
        self,
        model_name: str = "nvidia/parakeet-tdt-0.6b-v3",
        device: str = "cuda"
    ):
        self.model_name = model_name
        self.device = device
        self._model = None
        
    def _get_model(self):
        """Lazy load the Parakeet model."""
        if self._model is None:
            # Suppress NeMo warnings about CUDA graphs (not critical)
            import warnings
            import logging
            warnings.filterwarnings("ignore", message=".*cuda-python.*")
            logging.getLogger("nemo_logger").setLevel(logging.ERROR)
            
            try:
                import nemo.collections.asr as nemo_asr
            except ImportError:
                raise ImportError(
                    "NeMo toolkit is not installed. "
                    "Install it with: pip install nemo_toolkit[asr]"
                )
            
            print(f"🔄 Chargement de Parakeet TDT 0.6B v3...")
            
            # Load model from HuggingFace
            self._model = nemo_asr.models.ASRModel.from_pretrained(
                model_name=self.model_name
            )
            
            # Move to appropriate device
            if self.device == "cuda":
                import torch
                if torch.cuda.is_available():
                    self._model = self._model.cuda()
                    print(f"✅ Parakeet chargé sur GPU!")
                else:
                    print("⚠️ CUDA non disponible, utilisation du CPU (lent)")
                    self._model = self._model.cpu()
            else:
                self._model = self._model.cpu()
                print(f"✅ Parakeet chargé sur CPU (lent)")
            
            # Set to evaluation mode
            self._model.eval()
            
            # Restore logging level
            logging.getLogger("nemo_logger").setLevel(logging.INFO)
            
        return self._model
    
    def transcribe(
        self,
        audio_input: Union[str, Path, np.ndarray],
        language: Optional[str] = None,
        initial_prompt: Optional[str] = None  # Not used but kept for API compatibility
    ) -> ASRResult:
        """
        Transcribe audio to text using Parakeet TDT.
        
        Args:
            audio_input: Path to audio file or numpy array of samples (float32, 16kHz)
            language: Not used - Parakeet auto-detects language
            initial_prompt: Not used (kept for API compatibility)
            
        Returns:
            ASRResult with transcribed text and metadata
        """
        model = self._get_model()
        
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
            
            print(f"🎤 Transcription de buffer audio ({len(audio_input)} samples)")
        else:
            audio_path = Path(audio_input)
            if not audio_path.exists():
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
            print(f"🎤 Transcription de {audio_path.name}")
        
        try:
            # Transcribe with Parakeet (auto language detection)
            output = model.transcribe([str(audio_path)])
            
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
                language="auto",  # Parakeet auto-detects
                confidence=1.0,
                duration=None,
                segments=[]
            )
            
        finally:
            # Cleanup temp file
            if temp_file:
                Path(temp_file.name).unlink(missing_ok=True)
    
    async def transcribe_stream(
        self,
        audio_path: Union[str, Path],
        language: Optional[str] = None
    ) -> AsyncGenerator[ASRSegment, None]:
        """
        Transcribe audio and yield segments as they're processed.
        
        Note: Parakeet processes audio in one pass, so this is simulated streaming.
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
    
    def get_supported_languages(self) -> List[str]:
        """Get list of supported language codes."""
        return PARAKEET_LANGUAGES.copy()
    
    def get_model_info(self) -> dict:
        """Get information about the model."""
        return {
            "model_name": self.model_name,
            "params": "600M",
            "languages": len(PARAKEET_LANGUAGES),
            "device": self.device,
            "loaded": self._model is not None,
            "features": [
                "Automatic language detection",
                "Automatic punctuation",
                "Automatic capitalization", 
                "Word-level timestamps",
                "25 European languages",
                "Fast inference (~2GB VRAM)"
            ]
        }


def create_parakeet_asr(device: str = "cuda") -> ParakeetProvider:
    """
    Factory function to create a Parakeet ASR provider.
    
    Args:
        device: "cuda" (recommended) or "cpu"
        
    Returns:
        ParakeetProvider instance
    """
    return ParakeetProvider(device=device)
