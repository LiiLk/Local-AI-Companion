"""
Gemma 4 E4B-it Provider — Unified ASR + LLM + Vision.

Wraps Gemma 4 E4B with TorchAO int4 quantization.
Handles audio, image, and text input natively via multimodal tokens.

Requirements:
- transformers >= 4.51.0
- torchao
- ~4.1 GB VRAM (int4)
"""

import asyncio
import logging
import os
import threading
import wave
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import AsyncGenerator, Optional

import numpy as np

logger = logging.getLogger(__name__)

# VRAM optimization — must be set before importing torch
os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "expandable_segments:True,garbage_collection_threshold:0.8",
)


class GemmaProvider:
    """
    Unified provider wrapping Gemma 4 E4B-it for ASR + LLM + Vision.

    The model is lazily loaded on first use with TorchAO int4 quantization.
    Thread-safe via a loading lock (same pattern as MiniCPMoProvider).

    Args:
        model_id: HuggingFace model ID.
        device: "cuda" or "cpu".
        quantization: "int4" or None.
        max_new_tokens: Max tokens to generate.
        temperature: Sampling temperature.
        top_p: Nucleus sampling threshold.
        context_max_turns: Max conversation turns to keep.
    """

    def __init__(
        self,
        model_id: str = "google/gemma-4-E4B-it",
        device: str = "cuda",
        quantization: str = "int4",
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
        context_max_turns: int = 10,
    ):
        self.model_id = model_id
        self.device = device
        self.quantization = quantization
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.context_max_turns = context_max_turns

        self._model = None
        self._processor = None
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gemma")

    def _load_model(self):
        """Load model with TorchAO int4 quantization. Thread-safe."""
        if self._model is not None:
            return

        with self._lock:
            if self._model is not None:
                return

            import torch
            from transformers import AutoProcessor

            logger.info(f"Loading Gemma E4B from {self.model_id}...")

            self._processor = AutoProcessor.from_pretrained(self.model_id)

            # NOTE: Gemma 4 E4B is a multimodal model (audio+image+text).
            # Use AutoModelForMultimodalLM if available in your transformers version.
            # If not available, fall back to AutoModelForImageTextToText and verify
            # audio support in the smoke test (Task 4).
            try:
                from transformers import AutoModelForMultimodalLM as ModelClass
                logger.info("Using AutoModelForMultimodalLM (native audio support)")
            except ImportError:
                from transformers import AutoModelForImageTextToText as ModelClass
                logger.warning("AutoModelForMultimodalLM not available, using AutoModelForImageTextToText")

            if self.quantization == "int4":
                from torchao.quantization import int4_weight_only, quantize_

                logger.info("Applying TorchAO int4 quantization (group_size=128)...")
                model = ModelClass.from_pretrained(
                    self.model_id,
                    torch_dtype=torch.bfloat16,
                    device_map=self.device,
                    attn_implementation="sdpa",
                )
                quantize_(model, int4_weight_only(group_size=128))
            else:
                model = ModelClass.from_pretrained(
                    self.model_id,
                    torch_dtype=torch.bfloat16,
                    device_map=self.device,
                    attn_implementation="sdpa",
                )

            self._model = model
            logger.info("Gemma E4B loaded successfully")

            # Log VRAM
            if torch.cuda.is_available():
                alloc = torch.cuda.memory_allocated() / 1024 / 1024
                logger.info(f"VRAM after Gemma load: {alloc:.0f}MB")

    def preload(self):
        """Pre-load model (call at startup to avoid first-request latency)."""
        self._load_model()

    def _build_messages(
        self,
        text: str,
        history: list[dict] | None = None,
        audio: bytes | None = None,
        images: list | None = None,
    ) -> list[dict]:
        """
        Build chat messages in Gemma's expected format.

        Audio and images are placed as special tokens before text.
        """
        messages = []

        # Add history (trimmed to max turns)
        if history:
            max_msgs = self.context_max_turns * 2  # user + assistant pairs
            messages.extend(history[-max_msgs:])

        # Build user content parts
        content_parts = []

        if audio is not None:
            content_parts.append({"type": "audio", "audio": audio})

        if images:
            for img in images:
                content_parts.append({"type": "image", "image": img})

        if text:
            content_parts.append({"type": "text", "text": text})

        messages.append({"role": "user", "content": content_parts})
        return messages

    def _generate(
        self,
        messages: list[dict],
        stream: bool = False,
    ):
        """Synchronous generation (runs in executor thread)."""
        import torch

        self._load_model()

        inputs = self._processor.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
        ).to(self._model.device)

        input_len = inputs["input_ids"].shape[-1]

        with torch.no_grad():
            if stream:
                from transformers import TextIteratorStreamer

                streamer = TextIteratorStreamer(
                    self._processor.tokenizer,
                    skip_prompt=True,
                    skip_special_tokens=True,
                )

                gen_kwargs = {
                    **inputs,
                    "max_new_tokens": self.max_new_tokens,
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "do_sample": self.temperature > 0,
                    "streamer": streamer,
                }

                # Run generation in a thread so streamer can yield
                thread = threading.Thread(
                    target=self._model.generate, kwargs=gen_kwargs
                )
                thread.start()
                return streamer, thread
            else:
                output = self._model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    do_sample=self.temperature > 0,
                )
                return self._processor.decode(
                    output[0][input_len:], skip_special_tokens=True
                )

    async def chat(
        self,
        text: str = "",
        history: list[dict] | None = None,
        audio: bytes | None = None,
        images: list | None = None,
    ) -> str:
        """
        Single-turn inference. Returns full response text.

        Args:
            text: User text input.
            history: Prior conversation messages.
            audio: Raw audio bytes (16-bit PCM, 16kHz mono) or file path.
            images: List of PIL Images or file paths.

        Returns:
            Generated response text.
        """
        messages = self._build_messages(text, history, audio, images)
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            self._executor, self._generate, messages, False
        )
        return result

    async def chat_stream(
        self,
        text: str = "",
        history: list[dict] | None = None,
        audio: bytes | None = None,
        images: list | None = None,
    ) -> AsyncGenerator[str, None]:
        """
        Streaming token generation. Yields tokens as they're generated.

        Args:
            text: User text input.
            history: Prior conversation messages.
            audio: Raw audio bytes or file path.
            images: List of PIL Images or file paths.

        Yields:
            Token strings as they're generated.
        """
        messages = self._build_messages(text, history, audio, images)
        loop = asyncio.get_running_loop()
        streamer, thread = await loop.run_in_executor(
            self._executor, self._generate, messages, True
        )

        try:
            for token in streamer:
                if token:
                    yield token
        finally:
            thread.join(timeout=5)

    async def transcribe(self, audio_bytes: bytes) -> str:
        """
        ASR-only mode: transcribe audio to text.

        Args:
            audio_bytes: Raw PCM 16-bit 16kHz mono audio.

        Returns:
            Transcribed text.
        """
        return await self.chat(
            text="Transcribe this audio exactly as spoken.",
            audio=audio_bytes,
        )

    def cleanup(self):
        """Unload model and free VRAM."""
        import gc

        if self._model is not None:
            del self._model
            self._model = None
        if self._processor is not None:
            del self._processor
            self._processor = None
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        logger.info("GemmaProvider cleaned up")
