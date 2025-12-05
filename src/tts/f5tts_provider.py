"""
TTS implementation using F5-TTS - Lightweight and fast Voice Cloning.

F5-TTS is a Flow Matching based TTS model with ~300M parameters.
It offers voice cloning with only 10-30s of reference audio.

Advantages:
- Lightweight (~2-3GB VRAM) - can coexist with a 7B LLM
- Very fast (RTF ~0.04x on GPU = real-time!)
- Voice cloning with 10-30s of reference audio
- Native multilingual support (FR, EN, ZH, JA, etc.)
- Automatic model download from HuggingFace
- Simple and clean Python API

Disadvantages:
- Quality slightly lower than OpenAudio S1-mini
- CC-BY-NC license (non-commercial)

Usage:
    # Without voice cloning (default voice)
    tts = F5TTSProvider()
    result = await tts.synthesize("Hello world!")
    
    # With voice cloning
    tts = F5TTSProvider(
        ref_audio="reference.wav",
        ref_text="Exact transcription of the reference audio."
    )
"""

import asyncio
import tempfile
from pathlib import Path
from typing import AsyncGenerator

import numpy as np
import soundfile as sf

from .base import BaseTTS, TTSResult, Voice


# Default voices (without voice cloning)
AVAILABLE_VOICES = [
    Voice(id="default", name="Default F5-TTS", language="multi", gender="Unknown"),
    Voice(id="cloned", name="Cloned Voice", language="multi", gender="Unknown"),
]


class F5TTSProvider(BaseTTS):
    """
    TTS Provider using F5-TTS - Lightweight Voice Cloning.
    
    F5-TTS uses a voice cloning system: you provide a reference audio
    sample and its transcription, and the model generates speech in that voice.
    
    Without a reference, the model uses a default voice.
    
    Attributes:
        ref_audio: Path to reference audio for voice cloning
        ref_text: Transcription of the reference audio
        _model: F5TTS instance (loaded on demand)
    
    Example:
        # Without voice cloning
        tts = F5TTSProvider()
        result = await tts.synthesize("Hello world!")
        
        # With voice cloning
        tts = F5TTSProvider(
            ref_audio=Path("reference.wav"),
            ref_text="Hello, I am the reference voice."
        )
    """
    
    # Sample rate de F5-TTS (24kHz via Vocos vocoder)
    SAMPLE_RATE = 24000
    
    def __init__(
        self,
        ref_audio: str | Path | None = None,
        ref_text: str | None = None,
        model: str = "F5TTS_v1_Base",
        device: str | None = None,
        seed: int | None = None,
    ):
        """
        Initialize the F5-TTS provider.
        
        Args:
            ref_audio: Path to reference audio (10-30s recommended)
            ref_text: Exact transcription of the reference audio
                      If empty, F5-TTS will use ASR to transcribe (+ VRAM)
            model: Model to use ("F5TTS_v1_Base" or "E2TTS_Base")
            device: Device for inference (None = auto-detect cuda/cpu)
            seed: Seed for reproducibility (None = random)
        """
        # Voice cloning configuration
        self.ref_audio = Path(ref_audio) if ref_audio else None
        self.ref_text = ref_text or ""  # Empty = auto-transcription
        
        if self.ref_audio and not self.ref_audio.exists():
            raise FileNotFoundError(f"Reference audio not found: {self.ref_audio}")
        
        # Model configuration
        self.model_name = model
        self.device = device
        self.seed = seed
        
        # Model loaded on demand (lazy loading)
        self._model = None
    
    def _load_model(self):
        """
        Load the F5-TTS model (lazy loading).
        
        The model is automatically downloaded from HuggingFace
        on first call (~1.4GB).
        """
        if self._model is not None:
            return self._model
        
        print(f"🔄 Loading F5-TTS ({self.model_name})...")
        
        from f5_tts.api import F5TTS
        
        self._model = F5TTS(
            model=self.model_name,
            device=self.device,
        )
        
        print("✅ F5-TTS loaded!")
        return self._model
    
    def _get_default_ref(self) -> tuple[str, str]:
        """
        Return the default reference audio and text from F5-TTS.
        
        F5-TTS includes a default English reference example.
        """
        from importlib.resources import files
        
        default_audio = str(files("f5_tts").joinpath("infer/examples/basic/basic_ref_en.wav"))
        default_text = "Some call me nature, others call me mother nature."
        
        return default_audio, default_text
    
    async def synthesize(
        self,
        text: str,
        output_path: Path | None = None
    ) -> TTSResult:
        """
        Convert text to WAV audio file.
        
        Args:
            text: Text to synthesize
            output_path: Output path (default: temp file)
            
        Returns:
            TTSResult with the audio file path
        """
        # Inference is synchronous, run it in a thread
        loop = asyncio.get_event_loop()
        wav, sr = await loop.run_in_executor(
            None, 
            self._synthesize_sync, 
            text
        )
        
        # Define output path
        if output_path is None:
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            output_path = Path(tmp.name)
            tmp.close()
        
        # Save as WAV
        sf.write(str(output_path), wav, sr)
        
        # Calculate duration
        duration = len(wav) / sr
        
        return TTSResult(audio_path=output_path, duration=duration)
    
    def _synthesize_sync(self, text: str) -> tuple[np.ndarray, int]:
        """
        Synchronous synthesis (called in a thread).
        
        Returns:
            Tuple (wav_array, sample_rate)
        """
        model = self._load_model()
        
        # Determine which reference to use
        if self.ref_audio:
            ref_file = str(self.ref_audio)
            ref_text = self.ref_text
        else:
            ref_file, ref_text = self._get_default_ref()
        
        # Generate audio
        wav, sr, _ = model.infer(
            ref_file=ref_file,
            ref_text=ref_text,
            gen_text=text,
            seed=self.seed,
        )
        
        return wav, sr
    
    async def synthesize_stream(
        self,
        text: str
    ) -> AsyncGenerator[bytes, None]:
        """
        Generate audio in streaming mode.
        
        Note: F5-TTS supports chunk inference internally,
        but the current API returns complete audio.
        We simulate streaming by splitting the audio.
        
        Args:
            text: Text to synthesize
            
        Yields:
            Audio chunks in bytes (WAV format)
        """
        import io
        
        loop = asyncio.get_event_loop()
        wav, sr = await loop.run_in_executor(
            None, 
            self._synthesize_sync, 
            text
        )
        
        # Split into ~0.5s chunks
        chunk_size = sr // 2  # 0.5 second
        
        for i in range(0, len(wav), chunk_size):
            chunk = wav[i:i + chunk_size]
            
            # Convert to WAV bytes
            buffer = io.BytesIO()
            sf.write(buffer, chunk, sr, format='WAV')
            buffer.seek(0)
            yield buffer.read()
    
    async def synthesize_to_bytes(self, text: str) -> bytes:
        """
        Synthesize and return audio bytes directly.
        
        Args:
            text: Text to synthesize
            
        Returns:
            Audio data in bytes (WAV format)
        """
        import io
        
        loop = asyncio.get_event_loop()
        wav, sr = await loop.run_in_executor(
            None,
            self._synthesize_sync,
            text
        )
        
        # Convert to WAV bytes
        buffer = io.BytesIO()
        sf.write(buffer, wav, sr, format='WAV')
        buffer.seek(0)
        return buffer.read()
    
    async def list_voices(self, language: str | None = None) -> list[Voice]:
        """
        List available voices.
        
        F5-TTS uses voice cloning, so "voices" are defined
        by the reference audio, not by presets.
        
        Args:
            language: Ignored (F5-TTS is multilingual)
            
        Returns:
            List of available voices
        """
        voices = AVAILABLE_VOICES.copy()
        
        # Add custom voice if configured
        if self.ref_audio:
            voices.append(Voice(
                id="custom",
                name=f"Custom ({self.ref_audio.stem})",
                language="multi",
                gender="Unknown"
            ))
        
        return voices
    
    def set_voice(self, voice_id: str) -> None:
        """
        F5-TTS has no preset voices.
        
        To change voice, use set_reference() with
        a new reference audio.
        """
        pass  # No-op as F5-TTS uses voice cloning
    
    def set_reference(
        self,
        ref_audio: str | Path,
        ref_text: str = ""
    ) -> None:
        """
        Configure the reference voice for voice cloning.
        
        Args:
            ref_audio: Path to reference audio (10-30s recommended)
            ref_text: Exact transcription of the audio (empty = auto-transcription)
            
        Example:
            tts.set_reference(
                "reference.wav",
                "Hello, I am a clear and natural reference voice."
            )
        """
        ref_path = Path(ref_audio)
        if not ref_path.exists():
            raise FileNotFoundError(f"Reference audio not found: {ref_path}")
        
        self.ref_audio = ref_path
        self.ref_text = ref_text
    
    def set_rate(self, rate: str) -> None:
        """
        Change speech rate (not directly supported).
        
        F5-TTS does not support rate changes.
        This method exists for interface compatibility.
        
        Args:
            rate: Ignored
        """
        pass  # F5-TTS doesn't support rate
    
    def set_pitch(self, pitch: str) -> None:
        """
        Change voice pitch (not supported).
        
        F5-TTS does not support pitch changes.
        This method exists for interface compatibility.
        
        Args:
            pitch: Ignored
        """
        pass  # F5-TTS doesn't support pitch
    
    def set_seed(self, seed: int | None) -> None:
        """
        Change the generation seed.
        
        A fixed seed allows reproducible results.
        
        Args:
            seed: Seed (None = random)
        """
        self.seed = seed
