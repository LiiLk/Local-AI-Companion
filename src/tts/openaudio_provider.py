"""
TTS implementation using OpenAudio S1-mini (Fish Speech).

OpenAudio S1-mini is the #1 TTS model on TTS-Arena2 (May 2025) with 0.5B parameters.
It offers exceptional quality, voice cloning and native multilingual support.

Advantages:
- #1 quality on TTS-Arena2 (better than ElevenLabs)
- Voice cloning with 10-30 seconds of reference audio
- Native multilingual support (FR, EN, JA, ZH, etc.)
- Emotions via tags: (excited), (whispering), (sad), (laughing)
- Audio streaming for reduced latency
- 100% local and free

Disadvantages:
- Heavier model (0.5B params, ~3.5GB)
- First load is slow (~30s on CPU)
- Requires ~4GB RAM on CPU or ~2GB VRAM on GPU

Required configuration:
- Checkpoints in ~/models/openaudio-s1-mini/
  - model.pth (1.7GB) - Text-to-semantic model
  - codec.pth (1.8GB) - DAC audio decoder

Voice cloning usage:
    Prepare a reference audio file (10-30s of clear speech)
    and its transcription. The model will clone that voice.
"""

import io
import os
import sys
import queue
import tempfile
from pathlib import Path
from typing import AsyncGenerator
import asyncio
import threading

import numpy as np
import soundfile as sf

from .base import BaseTTS, TTSResult, Voice


# Available voices (default styles without voice cloning)
AVAILABLE_VOICES = [
    Voice(id="default", name="Default (Neural)", language="multi", gender="Unknown"),
    Voice(id="cloned", name="Cloned Voice", language="multi", gender="Unknown"),
]


class OpenAudioProvider(BaseTTS):
    """
    TTS provider using OpenAudio S1-mini (Fish Speech).
    
    OpenAudio uses a voice cloning system: you provide
    a reference audio sample and its transcription, and the model
    generates speech in that voice.
    
    Without reference, the model uses a neutral default voice.
    
    Attributes:
        checkpoint_path: Path to the checkpoints folder
        device: Device for inference ("cpu", "cuda", "mps")
        speaker_wav: Path to reference audio for voice cloning
        speaker_text: Transcription of the reference audio
        _engine: TTS inference engine (loaded on demand)
    
    Example:
        # Without voice cloning
        tts = OpenAudioProvider()
        result = await tts.synthesize("Hello world!")
        
        # With voice cloning
        tts = OpenAudioProvider(
            speaker_wav=Path("reference.wav"),
            speaker_text="Hello, I am the reference voice."
        )
    """
    
    # Sample rate de OpenAudio (44.1kHz)
    SAMPLE_RATE = 44100
    
    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        device: str = "cpu",
        speaker_wav: str | Path | None = None,
        speaker_text: str | None = None,
        compile_model: bool = False,
        half_precision: bool = False,
    ):
        """
        Initialize the OpenAudio S1-mini provider.
        
        Args:
            checkpoint_path: Path to OpenAudio S1-mini checkpoints
                            (default: ~/models/openaudio-s1-mini)
            device: Device for inference ("cpu", "cuda", "mps")
                    Note: "cpu" is recommended to free GPU for LLM
            speaker_wav: Path to reference audio for voice cloning
            speaker_text: Exact transcription of the reference audio
            compile_model: Compile model with torch.compile (slower startup)
            half_precision: Use half precision (fp16/bf16) - recommended on GPU
        """
        # Path configuration
        if checkpoint_path is None:
            checkpoint_path = Path.home() / "models" / "openaudio-s1-mini"
        self.checkpoint_path = Path(checkpoint_path)
        
        # Verify checkpoints exist
        self._validate_checkpoints()
        
        # Configuration device
        self.device = device
        self.compile_model = compile_model
        self.half_precision = half_precision
        
        # Voice cloning configuration
        self.speaker_wav = Path(speaker_wav) if speaker_wav else None
        self.speaker_text = speaker_text
        
        if self.speaker_wav and not self.speaker_wav.exists():
            raise FileNotFoundError(f"Speaker WAV not found: {self.speaker_wav}")
        
        # Engine loaded on demand (lazy loading)
        self._engine = None
        self._lock = threading.Lock()
        
        # Default generation parameters
        self.temperature = 0.8
        self.top_p = 0.8
        self.repetition_penalty = 1.1
        self.max_new_tokens = 1024
        
    def _validate_checkpoints(self) -> None:
        """Verify that checkpoint files are present."""
        required_files = ["model.pth", "codec.pth"]
        
        for filename in required_files:
            filepath = self.checkpoint_path / filename
            if not filepath.exists():
                raise FileNotFoundError(
                    f"Missing checkpoint: {filepath}\n"
                    f"Download with: huggingface-cli download fishaudio/openaudio-s1-mini "
                    f"--local-dir {self.checkpoint_path}"
                )
    
    def _load_engine(self):
        """
        Load the OpenAudio inference engine (lazy loading).
        
        This method is thread-safe and only loads the model once.
        Loading can take ~30s on CPU.
        """
        if self._engine is not None:
            return self._engine
        
        with self._lock:
            # Double-check after acquiring lock
            if self._engine is not None:
                return self._engine
            
            print(f"🔄 Loading OpenAudio S1-mini (device={self.device})...")
            print("   ⏳ This may take ~30 seconds on CPU...")
            
            # Add fish-speech to path if needed
            fish_speech_path = Path.home() / "tools" / "fish-speech"
            if str(fish_speech_path) not in sys.path:
                sys.path.insert(0, str(fish_speech_path))
            
            # Force CPU if requested (hide CUDA)
            if self.device == "cpu":
                os.environ["CUDA_VISIBLE_DEVICES"] = ""
            
            import torch
            from fish_speech.inference_engine import TTSInferenceEngine
            from fish_speech.models.dac.inference import load_model as load_decoder_model
            from fish_speech.models.text2semantic.inference import launch_thread_safe_queue
            
            # Determine precision
            if self.half_precision:
                precision = torch.float16
            else:
                precision = torch.bfloat16 if torch.cuda.is_available() else torch.float32
            
            # Load LLM model (text-to-semantic)
            llama_checkpoint = str(self.checkpoint_path)
            llama_queue = launch_thread_safe_queue(
                checkpoint_path=llama_checkpoint,
                device=self.device,
                precision=precision,
                compile=self.compile_model,
            )
            
            # Load audio decoder (DAC)
            decoder_checkpoint = str(self.checkpoint_path / "codec.pth")
            decoder_model = load_decoder_model(
                config_name="modded_dac_vq",
                checkpoint_path=decoder_checkpoint,
                device=self.device,
            )
            
            # Create inference engine
            self._engine = TTSInferenceEngine(
                llama_queue=llama_queue,
                decoder_model=decoder_model,
                precision=precision,
                compile=self.compile_model,
            )
            
            # Get actual sample rate
            if hasattr(decoder_model, "spec_transform"):
                self.SAMPLE_RATE = decoder_model.spec_transform.sample_rate
            elif hasattr(decoder_model, "sample_rate"):
                self.SAMPLE_RATE = decoder_model.sample_rate
            
            print(f"✅ OpenAudio S1-mini loaded! (sample_rate={self.SAMPLE_RATE}Hz)")
            
            return self._engine
    
    def _get_reference_audio(self) -> list:
        """
        Prépare l'audio de référence pour le voice cloning.
        
        Returns:
            Liste de ServeReferenceAudio ou liste vide si pas de référence
        """
        if self.speaker_wav is None or self.speaker_text is None:
            return []
        
        from fish_speech.utils.schema import ServeReferenceAudio
        
        with open(self.speaker_wav, "rb") as f:
            audio_bytes = f.read()
        
        return [ServeReferenceAudio(audio=audio_bytes, text=self.speaker_text)]
    
    async def synthesize(
        self,
        text: str,
        output_path: Path | None = None
    ) -> TTSResult:
        """
        Convertit du texte en fichier audio WAV.
        
        Supports OpenAudio emotion tags:
        - (excited) for excitement
        - (whispering) for whispering
        - (sad) for sadness
        - (laughing) for laughing
        
        Args:
            text: Text to synthesize (may contain emotion tags)
            output_path: Output path (default: temp file)
            
        Returns:
            TTSResult with audio file path
            
        Example:
            result = await tts.synthesize("Hello (excited) world!")
        """
        # Inference is synchronous, run in a thread
        loop = asyncio.get_event_loop()
        audio_data, sample_rate = await loop.run_in_executor(
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
        sf.write(str(output_path), audio_data, sample_rate)
        
        # Calculate duration
        duration = len(audio_data) / sample_rate
        
        return TTSResult(audio_path=output_path, duration=duration)
    
    def _synthesize_sync(self, text: str) -> tuple[np.ndarray, int]:
        """
        Synchronous synthesis (called in a thread).
        
        Returns:
            Tuple (audio_data, sample_rate)
        """
        from fish_speech.utils.schema import ServeTTSRequest
        
        engine = self._load_engine()
        
        # Prepare request
        request = ServeTTSRequest(
            text=text,
            references=self._get_reference_audio(),
            reference_id=None,
            max_new_tokens=self.max_new_tokens,
            chunk_length=200,
            top_p=self.top_p,
            repetition_penalty=self.repetition_penalty,
            temperature=self.temperature,
            streaming=False,
            format="wav",
        )
        
        # Execute inference
        for result in engine.inference(request):
            if result.code == "final":
                sample_rate, audio = result.audio
                return audio, sample_rate
            elif result.code == "error":
                raise RuntimeError(f"OpenAudio error: {result.error}")
        
        raise RuntimeError("Aucun audio généré")
    
    async def synthesize_stream(
        self,
        text: str
    ) -> AsyncGenerator[bytes, None]:
        """
        Generate audio in streaming mode (segment by segment).
        
        Allows starting playback before the entire
        synthesis is complete (reduced latency).
        
        Args:
            text: Text to synthesize
            
        Yields:
            Audio chunks in bytes (WAV format)
        """
        from fish_speech.utils.schema import ServeTTSRequest
        
        loop = asyncio.get_event_loop()
        
        def generate_segments():
            """Synchronous audio segment generator."""
            engine = self._load_engine()
            
            request = ServeTTSRequest(
                text=text,
                references=self._get_reference_audio(),
                reference_id=None,
                max_new_tokens=self.max_new_tokens,
                chunk_length=200,
                top_p=self.top_p,
                repetition_penalty=self.repetition_penalty,
                temperature=self.temperature,
                streaming=True,
                format="wav",
            )
            
            for result in engine.inference(request):
                if result.code == "segment":
                    sample_rate, audio = result.audio
                    yield audio, sample_rate
                elif result.code == "final":
                    sample_rate, audio = result.audio
                    yield audio, sample_rate
                elif result.code == "error":
                    raise RuntimeError(f"OpenAudio error: {result.error}")
        
        # Collect all segments (run_in_executor doesn't support generators)
        segments = await loop.run_in_executor(None, list, generate_segments())
        
        # Yield chaque segment converti en WAV bytes
        for audio, sample_rate in segments:
            buffer = io.BytesIO()
            sf.write(buffer, audio, sample_rate, format='WAV')
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
        loop = asyncio.get_event_loop()
        audio_data, sample_rate = await loop.run_in_executor(
            None,
            self._synthesize_sync,
            text
        )
        
        # Convert to WAV bytes
        buffer = io.BytesIO()
        sf.write(buffer, audio_data, sample_rate, format='WAV')
        buffer.seek(0)
        return buffer.read()
    
    async def list_voices(self, language: str | None = None) -> list[Voice]:
        """
        List available voices.
        
        OpenAudio uses voice cloning, so "voices" are
        defined by reference audio, not by presets.
        
        Args:
            language: Ignored (OpenAudio is multilingual)
            
        Returns:
            List of available voices
        """
        voices = AVAILABLE_VOICES.copy()
        
        # Add custom voice if configured
        if self.speaker_wav:
            voices.append(Voice(
                id="custom",
                name=f"Custom ({self.speaker_wav.stem})",
                language="multi",
                gender="Unknown"
            ))
        
        return voices
    
    def set_voice(self, voice_id: str) -> None:
        """
        OpenAudio has no preset voices.
        
        To change voice, use set_speaker() with
        a new reference audio.
        """
        pass  # No-op because OpenAudio uses voice cloning
    
    def set_speaker(
        self,
        speaker_wav: str | Path,
        speaker_text: str
    ) -> None:
        """
        Configure the reference voice for voice cloning.
        
        Args:
            speaker_wav: Path to reference audio (10-30s recommended)
            speaker_text: Exact transcription of the audio
            
        Example:
            tts.set_speaker(
                "reference.wav",
                "Hello, I am a clear and natural reference voice."
            )
        """
        speaker_path = Path(speaker_wav)
        if not speaker_path.exists():
            raise FileNotFoundError(f"Speaker WAV not found: {speaker_path}")
        
        self.speaker_wav = speaker_path
        self.speaker_text = speaker_text
    
    def set_temperature(self, temperature: float) -> None:
        """
        Change the generation temperature.
        
        Higher = more variety/creativity
        Lower = more stability/consistency
        
        Args:
            temperature: Value between 0.1 and 1.0 (default: 0.8)
        """
        self.temperature = max(0.1, min(1.0, temperature))
    
    def set_rate(self, rate: str) -> None:
        """
        Change speech rate (not directly supported).
        
        OpenAudio does not support speed changes.
        This method exists for interface compatibility.
        
        Args:
            rate: Ignored
        """
        # OpenAudio does not support rate
        pass
    
    def set_pitch(self, pitch: str) -> None:
        """
        Change voice pitch (not supported).
        
        OpenAudio does not support pitch changes.
        This method exists for interface compatibility.
        
        Args:
            pitch: Ignored
        """
        # OpenAudio does not support pitch
        pass


# Alias for consistency with rest of project
OpenAudioS1Provider = OpenAudioProvider
