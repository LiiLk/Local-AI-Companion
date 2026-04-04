"""
MiniCPM-o 4.5 Omni Model Provider.

A single model that handles ASR, LLM, and TTS natively.
This replaces the need for separate ASR + LLM + TTS providers.

Features:
- Speech-to-text (ASR) via audio input
- Text generation / chat (LLM)
- Text-to-speech with voice cloning (TTS)
- Vision understanding (images)
- Omni mode (audio + video + text simultaneously)

Requirements:
- transformers==4.51.0
- minicpmo-utils[all]>=1.0.2
- librosa>=0.10.0
- bitsandbytes>=0.43.0 (for int4/int8 quantization)
- ~11GB VRAM (int4), ~15GB (int8), ~19GB (bfloat16)
"""

import asyncio
import logging
import os
import tempfile
import threading
import warnings
import wave
from pathlib import Path
from typing import AsyncGenerator, Optional

# NOTE: Offline mode is set in _get_model() to avoid affecting other providers
# (e.g., GemmaProvider needs to download models on first use)
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import numpy as np

# Suppress third-party deprecation warnings (torch, diffusers, etc.)
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")
warnings.filterwarnings("ignore", category=FutureWarning, module="diffusers")
warnings.filterwarnings("ignore", category=UserWarning, message=".*weights_only.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*use_fast.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*Sliding Window.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*seen_tokens.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*clone.*detach.*")

# Suppress noisy INFO logs from transformers modules
logging.getLogger("transformers_modules").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


class MiniCPMoProvider:
    """
    Unified provider wrapping MiniCPM-o 4.5 for ASR + LLM + TTS.

    The model is lazily loaded on first use. All three capabilities
    (transcribe, chat, synthesize) share the same underlying model.

    Args:
        model_name: HuggingFace model ID.
        device: Device string, e.g. "cuda" or "cpu".
        dtype: Torch dtype string. "bfloat16" recommended for GPU.
        quantization: Quantization mode - "int4" (~11GB), "int8" (~15GB),
                      or None for full precision (uses dtype).
        attn_implementation: Attention backend ("sdpa" or "flash_attention_2").
        ref_audio_path: Optional path to a reference WAV for voice cloning TTS.
        init_vision: Initialize vision components.
        init_audio: Initialize audio components.
        init_tts: Initialize TTS components.
    """

    def __init__(
        self,
        model_name: str = "openbmb/MiniCPM-o-4_5",
        device: str = "cuda",
        dtype: str = "bfloat16",
        quantization: Optional[str] = None,
        attn_implementation: str = "sdpa",
        ref_audio_path: Optional[str] = None,
        init_vision: bool = True,
        init_audio: bool = True,
        init_tts: bool = True,
    ):
        self.model_name = model_name
        self.device = device
        self.dtype = dtype
        self.quantization = quantization
        self.attn_implementation = attn_implementation
        self.ref_audio_path = ref_audio_path
        self.init_vision = init_vision
        self.init_audio = init_audio
        self.init_tts = init_tts

        self._model = None
        self._ref_audio: Optional[np.ndarray] = None
        self._loading_lock = threading.Lock()
        self._is_loading = False
        self._is_ready = False

    @property
    def is_ready(self) -> bool:
        """Check if the model is loaded and ready for inference."""
        return self._is_ready

    @property
    def is_loading(self) -> bool:
        """Check if the model is currently loading."""
        return self._is_loading

    def preload(self):
        """Pre-load the model (blocking). Call this at startup to avoid lazy loading delays."""
        self._get_model()

    def _is_model_cached(self) -> bool:
        """Check if the model is already in the Hugging Face cache."""
        try:
            from huggingface_hub import try_to_load_from_cache
            # Check for config.json as a proxy for the full model
            cached_path = try_to_load_from_cache(self.model_name, "config.json")
            return cached_path is not None and cached_path != "NOT_FOUND"
        except Exception:
            return False

    def _get_model(self):
        """Lazy-load the MiniCPM-o model."""
        if self._model is not None:
            return self._model

        # Prevent concurrent loading attempts
        with self._loading_lock:
            if self._model is not None:
                return self._model

            self._is_loading = True
            try:
                return self._load_model_impl()
            finally:
                self._is_loading = False
                self._is_ready = self._model is not None

    def _load_model_impl(self):
        """Internal model loading implementation."""
        # Force offline mode for MiniCPM-o only (already downloaded)
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

        import torch
        from transformers import AutoModel

        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map.get(self.dtype, torch.bfloat16)

        # Build quantization config if requested
        quantization_config = None
        if self.quantization in ("int4", "int8"):
            from transformers import BitsAndBytesConfig

            if self.quantization == "int4":
                import torch
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch_dtype,
                    bnb_4bit_quant_storage=torch.uint8,  # Per official docs
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    llm_int8_skip_modules=["tts"],  # Keep TTS in full precision for voice quality
                )
                logger.info(f"Loading MiniCPM-o 4.5 (int4, ~11GB) on {self.device}...")
            else:  # int8
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_skip_modules=["tts"],  # Skip TTS from quantization
                )
                logger.info(f"Loading MiniCPM-o 4.5 (int8, ~15GB) on {self.device}...")
        else:
            logger.info(f"Loading MiniCPM-o 4.5 ({self.dtype}) on {self.device}...")

        attn_impl = self.attn_implementation
        if attn_impl == "flash_attention_2":
            try:
                import flash_attn  # noqa: F401
            except Exception:
                logger.warning(
                    "flash_attention_2 requested but flash-attn is not installed; "
                    "falling back to sdpa"
                )
                attn_impl = "sdpa"

        load_kwargs = dict(
            trust_remote_code=True,
            attn_implementation=attn_impl,
            torch_dtype=torch_dtype,
            init_vision=self.init_vision,
            init_audio=self.init_audio,
            init_tts=self.init_tts,
        )

        if quantization_config is not None:
            load_kwargs["quantization_config"] = quantization_config
            load_kwargs["device_map"] = "auto"
            # Fix: Required for bitsandbytes FP4/NF4 layers to initialize properly
            # Without this, TTS modules throw AssertionError in LinearFP4
            load_kwargs["low_cpu_mem_usage"] = False

        # Try loading from local cache first to avoid network issues
        # (HTTP 502, timeouts) when HuggingFace servers are slow
        use_local = self._is_model_cached()
        if use_local:
            logger.info("Model found in cache, loading offline...")
            load_kwargs["local_files_only"] = True
            # Also set env var for any sub-components (processor, tokenizer, etc.)
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"

        try:
            self._model = AutoModel.from_pretrained(
                self.model_name,
                **load_kwargs,
            )
        except Exception as e:
            # If local load failed, retry with network access
            if use_local:
                logger.warning(f"Local load failed ({e}), retrying with network...")
                load_kwargs["local_files_only"] = False
                os.environ.pop("HF_HUB_OFFLINE", None)
                os.environ.pop("TRANSFORMERS_OFFLINE", None)
                self._model = AutoModel.from_pretrained(
                    self.model_name,
                    **load_kwargs,
                )
            else:
                raise
        self._model.eval()

        # Convert TTS modules to bfloat16 to avoid quantization conflicts
        # Must match the compute dtype (bfloat16) to avoid dtype mismatch with LLM outputs
        if quantization_config is not None and hasattr(self._model, 'tts'):
            try:
                self._model.tts.to(torch_dtype)
                logger.info(f"TTS modules converted to {torch_dtype} for quantization compatibility")
            except Exception as e:
                logger.warning(f"Failed to convert TTS modules to {torch_dtype}: {e}")

        # Only manually move to device if not using quantization (device_map handles it)
        if quantization_config is None and self.device == "cuda":
            self._model = self._model.cuda()

        # Initialise TTS internals
        self._model.init_tts(streaming=False)

        # Pre-load reference audio for voice cloning if provided
        if self.ref_audio_path:
            self._load_ref_audio(self.ref_audio_path)
        else:
            logger.info("No voice reference configured - using model's default voice. "
                       "To enable voice cloning, set omni_ref_audio in character config.")

        logger.info("MiniCPM-o 4.5 loaded successfully")
        return self._model

    def _load_ref_audio(self, path: str):
        """Load a reference audio file for voice-cloned TTS."""
        import librosa

        audio, _ = librosa.load(path, sr=16000, mono=True)
        self._ref_audio = audio
        logger.info(f"Reference audio loaded: {path} ({len(audio)} samples)")

    # ------------------------------------------------------------------
    # ASR
    # ------------------------------------------------------------------

    def transcribe(
        self,
        audio_input: "str | Path | np.ndarray",
        language: Optional[str] = None,
    ) -> str:
        """
        Transcribe audio to text.

        Args:
            audio_input: Path to a WAV file or a float32 numpy array (16 kHz mono).
            language: Optional language hint (currently unused; the model auto-detects).

        Returns:
            Transcribed text string.
        """
        import librosa

        model = self._get_model()

        if isinstance(audio_input, np.ndarray):
            audio = audio_input.astype(np.float32)
            # Ensure 16 kHz
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)
        else:
            audio, _ = librosa.load(str(audio_input), sr=16000, mono=True)

        task_prompt = (
            "Please listen to the audio snippet carefully "
            "and transcribe the content.\n"
        )
        msgs = [{"role": "user", "content": [task_prompt, audio]}]

        result = model.chat(
            msgs=msgs,
            do_sample=True,
            max_new_tokens=512,
            use_tts_template=False,
            generate_audio=False,
            temperature=0.3,
        )
        return result.strip() if isinstance(result, str) else str(result).strip()

    # ------------------------------------------------------------------
    # LLM chat
    # ------------------------------------------------------------------

    def chat(
        self,
        messages: list[dict],
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        do_sample: bool = True,
    ) -> str:
        """
        Generate a text response given a conversation history.

        Args:
            messages: List of dicts with "role" and "content" keys.
                      Content can be a string or a list mixing strings,
                      numpy audio arrays, and PIL images.
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            do_sample: Whether to sample or greedy decode.

        Returns:
            The assistant's text response.
        """
        model = self._get_model()

        logger.info(f"LLM chat START - {len(messages)} messages, max_tokens={max_new_tokens}")

        result = model.chat(
            msgs=messages,
            do_sample=do_sample,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            use_tts_template=False,
            generate_audio=False,
        )

        logger.info(f"LLM chat END - response: {str(result)[:100]}...")
        return result.strip() if isinstance(result, str) else str(result).strip()

    def chat_stream(
        self,
        messages: list[dict],
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        do_sample: bool = True,
    ):
        """
        Stream text tokens from the model.

        NOTE: MiniCPM-o doesn't support streaming via model.chat().
        Real streaming requires streaming_prefill() + streaming_generate().
        This method falls back to non-streaming for compatibility.

        Yields:
            Text chunks as they are generated (single chunk in non-streaming mode).
        """
        # MiniCPM-o streaming requires streaming_prefill() + streaming_generate()
        # For simplicity, fall back to non-streaming chat
        result = self.chat(messages, max_new_tokens, temperature, do_sample)
        yield result

    # ------------------------------------------------------------------
    # TTS
    # ------------------------------------------------------------------

    def synthesize(
        self,
        text: str,
        output_path: "str | Path | None" = None,
    ) -> "str | Path":
        """
        Synthesize speech from text.

        If a reference audio was provided at init, voice cloning is used.

        Args:
            text: The text to speak.
            output_path: Where to write the WAV file. If None a temp file is used.

        Returns:
            Path to the generated WAV file.
        """
        model = self._get_model()

        logger.info(f"TTS synthesize START - text: {text[:50]}...")

        if output_path is None:
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            output_path = Path(tmp.name)
            tmp.close()
        else:
            output_path = Path(output_path)

        msgs = []

        # Voice cloning OR standard TTS - system message always required
        if self._ref_audio is not None:
            # Voice cloning mode with reference audio
            msgs.append({
                "role": "system",
                "content": [
                    "Clone the voice in the provided audio prompt.",
                    self._ref_audio,
                    "Please assist users while maintaining this voice style.",
                ],
            })
        else:
            # Standard TTS mode without voice cloning - still needs system message
            msgs.append({
                "role": "system",
                "content": ["You are a helpful assistant with a natural speaking voice."],
            })

        msgs.append({
            "role": "user",
            "content": [f"Please read: {text}"],
        })

        model.chat(
            msgs=msgs,
            do_sample=True,
            max_new_tokens=512,
            use_tts_template=True,
            generate_audio=True,
            temperature=0.1,
            output_audio_path=str(output_path),
        )

        logger.info(f"TTS synthesize END - output: {output_path}")
        return output_path

    # ------------------------------------------------------------------
    # Omni (audio-in → text + audio-out)
    # ------------------------------------------------------------------

    def chat_omni(
        self,
        messages: list[dict],
        output_audio_path: "str | Path | None" = None,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
    ) -> tuple[str, "Path | None"]:
        """
        Full omni turn: accept multimodal input, return text + audio.

        Args:
            messages: Conversation messages (may contain audio/image content).
            output_audio_path: Where to write the response audio. None to skip.
            max_new_tokens: Max tokens for text generation.
            temperature: Sampling temperature.

        Returns:
            Tuple of (response_text, audio_path_or_None).
        """
        model = self._get_model()

        generate_audio = output_audio_path is not None
        if output_audio_path is not None:
            output_audio_path = Path(output_audio_path)

        # Build message list - keep original system prompt, optionally add voice clone
        full_msgs = []

        # Find and extract the original system prompt
        original_system = None
        for m in messages:
            if m.get("role") == "system":
                content = m.get("content", [])
                if isinstance(content, list) and content:
                    original_system = content[0] if isinstance(content[0], str) else None
                elif isinstance(content, str):
                    original_system = content
                break

        # Build new system message with voice cloning if available
        if self._ref_audio is not None:
            system_content = [
                "Clone the voice in the provided audio prompt.",
                self._ref_audio,
            ]
            if original_system:
                system_content.append(original_system)
            else:
                system_content.append("Please assist users while maintaining this voice style.")
            full_msgs.append({"role": "system", "content": system_content})
            logger.debug("Using voice cloning with reference audio")
        elif original_system:
            # No voice cloning, but keep the character system prompt
            full_msgs.append({"role": "system", "content": [original_system]})
            logger.debug("No voice cloning - using default voice with character prompt")
        else:
            # Fallback: always include a minimal system prompt for stability
            full_msgs.append({"role": "system", "content": ["You are a helpful AI assistant."]})
            logger.debug("No voice cloning - using fallback system prompt")

        # Add non-system messages
        for m in messages:
            if m.get("role") != "system":
                full_msgs.append(m)

        logger.info(f"chat_omni: {len(full_msgs)} messages, generate_audio={generate_audio}")

        result = model.chat(
            msgs=full_msgs,
            do_sample=True,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            use_tts_template=generate_audio,
            omni_mode=generate_audio,  # Enable omni mode per official docs
            generate_audio=generate_audio,
            output_audio_path=str(output_audio_path) if output_audio_path else None,
        )

        text = result.strip() if isinstance(result, str) else str(result).strip()
        audio_out = output_audio_path if generate_audio else None
        return text, audio_out
