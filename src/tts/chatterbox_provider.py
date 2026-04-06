"""
Chatterbox Multilingual ONNX Q4 TTS Provider.

Extends BaseTTS with voice cloning and emotion tag support.
Uses ONNX Runtime for inference (separate CUDA allocator from PyTorch).

Features:
- Voice cloning from 5-10s reference audio
- Emotion tags: [laugh], [chuckle], [cough], [sigh]
- 23 languages including French
- Exaggeration slider (0.0-1.0)
- ~2 GB VRAM (ONNX Q4)
"""

import asyncio
import io
import logging
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import AsyncGenerator, Optional

import numpy as np
import soundfile as sf

from .base import BaseTTS, TTSResult, Voice

logger = logging.getLogger(__name__)


# Available emotion tags that Chatterbox interprets natively
CHATTERBOX_EMOTION_TAGS = {"laugh", "chuckle", "cough", "sigh"}

CHATTERBOX_VOICES = [
    Voice(id="default", name="Default", language="multi", gender="Female"),
]


class ChatterboxTTSProvider(BaseTTS):
    """
    TTS provider using Chatterbox Multilingual ONNX Q4.

    Extends BaseTTS and implements all abstract methods.
    Model is lazily loaded on first use.

    Args:
        model_id: HuggingFace model ID for ONNX Q4 variant.
        ref_audio_path: Path to reference WAV for voice cloning.
        exaggeration: Emotion exaggeration (0.0-1.0).
        cfg_weight: Classifier-free guidance weight.
        language: Target language code (e.g., "fr", "en").
    """

    SAMPLE_RATE = 24000

    def __init__(
        self,
        model_id: str = "onnx-community/chatterbox-multilingual-ONNX",
        ref_audio_path: str | Path | None = None,
        exaggeration: float = 0.5,
        cfg_weight: float = 0.5,
        language: str = "fr",
        prefer_full_gpu: bool = False,
    ):
        self.model_id = model_id
        self.ref_audio_path = Path(ref_audio_path) if ref_audio_path else None
        self.exaggeration = exaggeration
        self.cfg_weight = cfg_weight
        self.language = language
        self.prefer_full_gpu = prefer_full_gpu
        self._speed = 1.0

        self._model = None
        self._ref_audio_data = None
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tts")
        self._load_lock = threading.Lock()

    def _load_model(self):
        """Load Chatterbox ONNX Q4 model (lazy, thread-safe)."""
        if self._model is not None:
            return
        with self._load_lock:
            if self._model is not None:
                return

            logger.info(f"Loading Chatterbox from {self.model_id}...")

            from chatterbox_onnx import ChatterboxOnnx
            import onnxruntime
            import os

            # Add PyTorch CUDA DLLs to PATH so ONNX Runtime can find them
            import torch
            torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
            if torch_lib not in os.environ.get("PATH", ""):
                os.environ["PATH"] = torch_lib + os.pathsep + os.environ.get("PATH", "")
                logger.info(f"Added PyTorch CUDA libs to PATH: {torch_lib}")

            available_providers = set(onnxruntime.get_available_providers())
            can_use_cuda = "CUDAExecutionProvider" in available_providers

            _original_get_session = ChatterboxOnnx._download_and_get_session
            _original_load_tokenizer = ChatterboxOnnx._load_tokenizer
            _target_model_id = self.model_id  # our multilingual model ID

            def _gpu_session(self_inner, filename: str) -> onnxruntime.InferenceSession:
                from huggingface_hub import hf_hub_download
                local_files_only = os.environ.get('HF_HUB_OFFLINE') == '1'
                # Use OUR model_id (multilingual), not the hardcoded English one
                path = hf_hub_download(
                    repo_id=_target_model_id,
                    filename=filename,
                    local_dir=self_inner.output_dir,
                    subfolder='onnx',
                    local_files_only=local_files_only,
                )
                hf_hub_download(
                    repo_id=_target_model_id,
                    filename=filename.replace(".onnx", ".onnx_data"),
                    local_dir=self_inner.output_dir,
                    subfolder='onnx',
                    local_files_only=local_files_only,
                )
                if self.prefer_full_gpu and can_use_cuda:
                    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
                elif can_use_cuda and "language_model" in filename:
                    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
                else:
                    providers = ["CPUExecutionProvider"]

                opts = onnxruntime.SessionOptions()
                # conditional_decoder has 24K nodes — skip graph optimization (saves ~3min)
                if "conditional_decoder" in filename:
                    opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL

                logger.info(f"Loading {filename} with providers: {providers}")
                return onnxruntime.InferenceSession(path, sess_options=opts, providers=providers)

            ChatterboxOnnx._download_and_get_session = _gpu_session

            # Patch tokenizer to use our multilingual model ID
            def _patched_load_tokenizer(self_inner):
                from huggingface_hub import hf_hub_download
                from tokenizers import Tokenizer
                tokenizer_path = hf_hub_download(
                    repo_id=_target_model_id,
                    filename="tokenizer.json",
                    local_dir=self_inner.output_dir,
                )
                return Tokenizer.from_file(tokenizer_path)

            ChatterboxOnnx._load_tokenizer = _patched_load_tokenizer

            # Patch embed_speaker to use soundfile+soxr instead of librosa
            # (librosa imports numba which hangs due to LLVM conflict with TensorRT)
            _original_embed_speaker = ChatterboxOnnx.embed_speaker

            def _embed_speaker_no_librosa(self_inner, source_audio_path: str):
                import soxr
                audio, sr = sf.read(source_audio_path, dtype="float32")
                if audio.ndim > 1:
                    audio = audio.mean(axis=1)
                if sr != 24000:
                    audio = soxr.resample(audio, sr, 24000)
                audio = audio[np.newaxis, :].astype(np.float32)
                return self_inner.speech_encoder_session.run(
                    None, {"audio_values": audio}
                )

            ChatterboxOnnx.embed_speaker = _embed_speaker_no_librosa

            try:
                self._model = ChatterboxOnnx(quantized=True)
            finally:
                ChatterboxOnnx._download_and_get_session = _original_get_session
                ChatterboxOnnx._load_tokenizer = _original_load_tokenizer
                # Keep embed_speaker patched (librosa.load hangs due to numba/LLVM conflict)

            # Pre-load reference audio if configured
            if self.ref_audio_path and self.ref_audio_path.exists():
                self._ref_audio_data = self._load_reference(self.ref_audio_path)
                logger.info(f"Voice reference loaded: {self.ref_audio_path}")

            logger.info("Chatterbox loaded successfully (GPU)" if can_use_cuda else "Chatterbox loaded (CPU)")

    def _load_reference(self, path: Path) -> np.ndarray:
        """Load and preprocess reference audio for voice cloning."""
        audio, sr = sf.read(str(path), dtype="float32")
        # Convert to mono if stereo
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        # Resample to 24kHz if needed (soxr instead of librosa to avoid numba)
        if sr != self.SAMPLE_RATE:
            import soxr
            audio = soxr.resample(audio, sr, self.SAMPLE_RATE)
        return audio.astype(np.float32)

    # Supported language tags for Chatterbox Multilingual tokenizer
    LANG_TAGS = {"fr", "en", "es", "de", "it", "nl", "pt", "el", "tr", "sv",
                 "no", "da", "ru", "pl", "sk", "cs", "hu", "ar", "hi", "ja",
                 "ko", "zh", "ro", "bg", "fi", "ta", "ms", "he", "vi"}

    @staticmethod
    def _detect_lang(text: str) -> str:
        """Simple heuristic language detection based on character patterns."""
        # Common French indicators
        fr_chars = set("àâéèêëïîôùûüÿçœæ")
        en_indicators = {"the ", "is ", "are ", "was ", "have ", "has ", " of ", " and ", " to "}
        lower = text.lower()
        if any(c in fr_chars for c in lower):
            return "fr"
        if any(ind in lower for ind in en_indicators):
            return "en"
        # Default to French (our primary language)
        return "fr"

    def _synthesize_sync(self, text: str) -> tuple[np.ndarray, int]:
        """Synchronous synthesis. Returns (audio_array, sample_rate)."""
        self._load_model()

        # Chatterbox ONNX API: synthesize(text, target_voice_path, ..., output_file_name)
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.close()

        ref_path = str(self.ref_audio_path) if self.ref_audio_path and self.ref_audio_path.exists() else None

        # Auto-detect language and prepend tag for correct accent
        lang = self._detect_lang(text)
        tagged_text = f"[{lang}] {text}"

        # Scale max_new_tokens to text length (~3 tokens per char is generous)
        # Avoids generating hundreds of filler tokens ("euhhhh") on short sentences
        max_tokens = min(max(len(text) * 3, 64), 300)

        logger.info("Chatterbox synth start (%s chars, lang=%s, max_tokens=%d)", len(text), lang, max_tokens)
        self._model.synthesize(
            text=tagged_text,
            target_voice_path=ref_path,
            exaggeration=self.exaggeration,
            max_new_tokens=max_tokens,
            output_file_name=tmp.name,
        )

        # Read back the generated WAV
        audio, sr = sf.read(tmp.name, dtype="float32")
        import os
        os.unlink(tmp.name)

        return audio, sr

    async def synthesize(
        self,
        text: str,
        output_path: Path | None = None,
    ) -> TTSResult:
        """Synthesize text to audio."""
        loop = asyncio.get_running_loop()
        audio, sr = await loop.run_in_executor(
            self._executor, self._synthesize_sync, text
        )

        # Convert to 16-bit PCM WAV
        audio_int16 = (audio * 32767).astype(np.int16)

        if output_path:
            sf.write(str(output_path), audio_int16, sr, subtype="PCM_16")
            duration = len(audio) / sr
            return TTSResult(
                audio_path=output_path,
                duration=duration,
            )
        else:
            buf = io.BytesIO()
            sf.write(buf, audio_int16, sr, format="WAV", subtype="PCM_16")
            audio_bytes = buf.getvalue()
            duration = len(audio) / sr
            return TTSResult(
                audio_data=audio_bytes,
                duration=duration,
            )

    async def synthesize_stream(self, text: str) -> AsyncGenerator[bytes, None]:
        """
        Streaming synthesis — yields audio chunks.

        Note: Chatterbox ONNX may not support true streaming.
        Falls back to generating full audio then chunking.
        """
        result = await self.synthesize(text)
        data = result.audio_data
        if data is None and result.audio_path:
            data = result.audio_path.read_bytes()

        # Yield in ~100ms chunks
        chunk_size = self.SAMPLE_RATE * 2 * 100 // 1000  # 16-bit = 2 bytes/sample
        for i in range(0, len(data), chunk_size):
            yield data[i : i + chunk_size]

    async def list_voices(self, language: str | None = None) -> list[Voice]:
        """List available voices."""
        return CHATTERBOX_VOICES

    def set_voice(self, voice_id: str) -> None:
        """Set voice by reference audio path."""
        path = Path(voice_id)
        if path.exists():
            self.ref_audio_path = path
            self._ref_audio_data = self._load_reference(path)
            logger.info(f"Voice reference set: {path}")

    def set_rate(self, rate: str) -> None:
        """Adjust speech rate. Accepts '+20%' style strings."""
        # Parse percentage string to float multiplier
        try:
            pct = int(rate.replace("%", "").replace("+", ""))
            self._speed = 1.0 + pct / 100.0
        except (ValueError, AttributeError):
            pass

    def set_pitch(self, pitch: str) -> None:
        """No-op — Chatterbox handles pitch internally."""
        pass

    def set_reference_audio(self, ref_audio_path: str | Path) -> None:
        """Set reference audio for voice cloning."""
        self.set_voice(str(ref_audio_path))

    def cleanup(self):
        """Unload model and free resources."""
        if self._model is not None:
            del self._model
            self._model = None
        self._ref_audio_data = None
        self._executor.shutdown(wait=False)
        logger.info("ChatterboxTTSProvider cleaned up")
