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
import tempfile
import wave
from pathlib import Path
from typing import AsyncGenerator, Optional

import numpy as np

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
    """

    def __init__(
        self,
        model_name: str = "openbmb/MiniCPM-o-4_5",
        device: str = "cuda",
        dtype: str = "bfloat16",
        quantization: Optional[str] = None,
        attn_implementation: str = "sdpa",
        ref_audio_path: Optional[str] = None,
    ):
        self.model_name = model_name
        self.device = device
        self.dtype = dtype
        self.quantization = quantization
        self.attn_implementation = attn_implementation
        self.ref_audio_path = ref_audio_path

        self._model = None
        self._ref_audio: Optional[np.ndarray] = None

    def _get_model(self):
        """Lazy-load the MiniCPM-o model."""
        if self._model is not None:
            return self._model

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
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch_dtype,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
                logger.info(f"Loading MiniCPM-o 4.5 (int4, ~11GB) on {self.device}...")
            else:
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                )
                logger.info(f"Loading MiniCPM-o 4.5 (int8, ~15GB) on {self.device}...")
        else:
            logger.info(f"Loading MiniCPM-o 4.5 ({self.dtype}) on {self.device}...")

        load_kwargs = dict(
            trust_remote_code=True,
            attn_implementation=self.attn_implementation,
            torch_dtype=torch_dtype,
            init_vision=True,
            init_audio=True,
            init_tts=True,
        )

        if quantization_config is not None:
            load_kwargs["quantization_config"] = quantization_config
            load_kwargs["device_map"] = "auto"

        self._model = AutoModel.from_pretrained(
            self.model_name,
            **load_kwargs,
        )
        self._model.eval()

        # Only manually move to device if not using quantization (device_map handles it)
        if quantization_config is None and self.device == "cuda":
            self._model = self._model.cuda()

        # Initialise TTS internals
        self._model.init_tts(streaming=False)

        # Pre-load reference audio for voice cloning if provided
        if self.ref_audio_path:
            self._load_ref_audio(self.ref_audio_path)

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

        result = model.chat(
            msgs=messages,
            do_sample=do_sample,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            use_tts_template=False,
            generate_audio=False,
        )
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

        Yields:
            Text chunks as they are generated.
        """
        model = self._get_model()

        gen = model.chat(
            msgs=messages,
            do_sample=do_sample,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            use_tts_template=False,
            generate_audio=False,
            stream=True,
        )

        if hasattr(gen, "__iter__"):
            for chunk in gen:
                yield chunk
        else:
            yield str(gen)

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

        if output_path is None:
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            output_path = Path(tmp.name)
            tmp.close()
        else:
            output_path = Path(output_path)

        msgs = []

        # Voice cloning via system message with reference audio
        if self._ref_audio is not None:
            msgs.append({
                "role": "system",
                "content": [
                    "Clone the voice in the provided audio prompt.",
                    self._ref_audio,
                    "Please assist users while maintaining this voice style.",
                ],
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

        # Prepend voice-clone system message if reference audio is available
        full_msgs = list(messages)
        if self._ref_audio is not None:
            full_msgs.insert(0, {
                "role": "system",
                "content": [
                    "Clone the voice in the provided audio prompt.",
                    self._ref_audio,
                    "Please assist users while maintaining this voice style.",
                ],
            })

        result = model.chat(
            msgs=full_msgs,
            do_sample=True,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            use_tts_template=generate_audio,
            generate_audio=generate_audio,
            output_audio_path=str(output_audio_path) if output_audio_path else None,
        )

        text = result.strip() if isinstance(result, str) else str(result).strip()
        audio_out = output_audio_path if generate_audio else None
        return text, audio_out
