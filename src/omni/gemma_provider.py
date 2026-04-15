"""
Gemma 4 Provider — Unified ASR + LLM + Vision.

Wraps Gemma 4 E2B/E4B with BitsAndBytes NF4 quantization.
Handles audio, image, and text input natively via multimodal tokens.

Requirements:
- transformers >= 5.0.0
- bitsandbytes >= 0.45.0
- ~3-4 GB VRAM (E2B int4), ~8.9 GB (E4B int4)
"""

import asyncio
import logging
import os
import threading
import time
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

REALTIME_AUDIO_REPLY_PROMPT = (
    "Listen to the following speech segment and reply briefly and naturally in the user's language. "
    "Do not transcribe unless the user explicitly asks for a transcription. "
    "Keep the answer short, conversational, and suitable for real-time voice chat."
)


class GemmaProvider:
    """
    Unified provider wrapping Gemma 4 E2B/E4B-it for ASR + LLM + Vision.

    The model is lazily loaded on first use with BitsAndBytes NF4 quantization.
    Thread-safe via a loading lock (same pattern as MiniCPMoProvider).
    """

    def __init__(
        self,
        model_id: str = "google/gemma-4-E2B-it",
        device: str = "cuda",
        quantization: str = "int4",
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
        context_max_turns: int = 10,
        cpu_offload: bool = True,
        offload_dir: str | None = None,
    ):
        self.model_id = model_id
        self.device = device
        self.quantization = quantization
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.context_max_turns = context_max_turns
        self.cpu_offload = cpu_offload
        self.offload_dir = Path(offload_dir) if offload_dir else Path(".cache/hf-offload/gemma")

        self._model = None
        self._processor = None
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gemma")

    def _build_quantized_load_kwargs(self, torch, bitsandbytes_config_cls=None):
        if bitsandbytes_config_cls is None:
            from transformers import BitsAndBytesConfig as bitsandbytes_config_cls

        logger.info("Applying BitsAndBytes NF4 quantization...")
        quantization_config = bitsandbytes_config_cls(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            llm_int8_skip_modules=["lm_head", "audio_tower", "embed_audio"],
            # Some multimodal Gemma modules remain in higher precision.
            # Allow explicit CPU offload instead of hard-failing when auto device_map
            # cannot keep every skipped module on the GPU.
            llm_int8_enable_fp32_cpu_offload=self.cpu_offload,
        )

        load_kwargs = {
            "quantization_config": quantization_config,
            "device_map": "auto",
            "torch_dtype": torch.bfloat16,
            "low_cpu_mem_usage": True,
            "attn_implementation": "sdpa",
        }

        if self.cpu_offload:
            self.offload_dir.mkdir(parents=True, exist_ok=True)
            load_kwargs["offload_folder"] = str(self.offload_dir)

        return load_kwargs

    def _load_model(self):
        """Load model with TorchAO int4 quantization. Thread-safe."""
        if self._model is not None:
            return

        with self._lock:
            if self._model is not None:
                return

            import torch
            from transformers import AutoProcessor

            logger.info(f"Loading Gemma from {self.model_id}...")

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
                load_kwargs = self._build_quantized_load_kwargs(torch)
                logger.info(
                    "Gemma int4 load strategy: device_map=auto, cpu_offload=%s, offload_dir=%s",
                    self.cpu_offload,
                    str(self.offload_dir),
                )
                model = ModelClass.from_pretrained(
                    self.model_id,
                    **load_kwargs,
                )
            else:
                model = ModelClass.from_pretrained(
                    self.model_id,
                    torch_dtype=torch.bfloat16,
                    device_map=self.device,
                    attn_implementation="sdpa",
                )

            self._model = model
            logger.info("Gemma loaded successfully")

            # Fix: BitsAndBytes quantizes audio tower layers to uint8, which breaks
            # torch.finfo() calls in Gemma4AudioFeedForward.forward (line 392).
            # Monkey-patch the forward method to use input dtype instead of weight dtype.
            if self.quantization == "int4":
                self._patch_audio_feed_forward()

            # Log VRAM
            if torch.cuda.is_available():
                alloc = torch.cuda.memory_allocated() / 1024 / 1024
                logger.info(f"VRAM after Gemma load: {alloc:.0f}MB")

    @staticmethod
    def _patch_audio_feed_forward():
        """Monkey-patch Gemma4 audio classes to handle BitsAndBytes quantized weight dtypes.

        BitsAndBytes NF4 quantizes audio tower Linear layers to uint8.
        Three forward methods call torch.finfo(weight.dtype) which fails on uint8.
        This patch uses hidden_states.dtype (always float) as fallback.
        """
        import torch
        from transformers.models.gemma4.modeling_gemma4 import (
            Gemma4AudioFeedForward,
            Gemma4AudioLightConv1d,
            Gemma4AudioLayer,
        )

        def _safe_finfo_max(dtype_source, fallback_dtype):
            """Get finfo.max, falling back to fallback_dtype if source is not float."""
            dtype = dtype_source.dtype if hasattr(dtype_source, 'dtype') else dtype_source
            if not dtype.is_floating_point:
                dtype = fallback_dtype
            return torch.finfo(dtype).max

        # Patch 1: Gemma4AudioFeedForward.forward (line 392)
        def _ff_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
            gradient_clipping = min(
                self.gradient_clipping,
                _safe_finfo_max(self.ffw_layer_1.linear.weight, hidden_states.dtype),
            )
            residual = hidden_states
            hidden_states = torch.clamp(hidden_states, -gradient_clipping, gradient_clipping)
            hidden_states = self.pre_layer_norm(hidden_states)
            hidden_states = self.ffw_layer_1(hidden_states)
            hidden_states = self.act_fn(hidden_states)
            hidden_states = self.ffw_layer_2(hidden_states)
            hidden_states = torch.clamp(hidden_states, -gradient_clipping, gradient_clipping)
            hidden_states = self.post_layer_norm(hidden_states)
            hidden_states *= self.post_layer_scale
            hidden_states += residual
            return hidden_states

        # Patch 2: Gemma4AudioLightConv1d.forward (line 475)
        def _lconv_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
            import torch.nn as nn
            gradient_clipping = min(
                self.gradient_clipping,
                _safe_finfo_max(self.linear_start.linear.weight, hidden_states.dtype),
            )
            residual = hidden_states
            hidden_states = self.pre_layer_norm(hidden_states)
            hidden_states = self.linear_start(hidden_states)
            hidden_states = nn.functional.glu(hidden_states, dim=-1)
            hidden_states = self.depthwise_conv1d(hidden_states.transpose(1, 2)).transpose(1, 2)
            hidden_states = torch.clamp(hidden_states, -gradient_clipping, gradient_clipping)
            hidden_states = self.conv_norm(hidden_states)
            hidden_states = self.act_fn(hidden_states)
            hidden_states = self.linear_end(hidden_states)
            hidden_states += residual
            return hidden_states

        # Patch 3: Gemma4AudioLayer.forward (line 509)
        def _layer_forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask=None,
            position_embeddings=None,
            **kwargs,
        ) -> torch.Tensor:
            gradient_clipping = min(
                self.gradient_clipping,
                _safe_finfo_max(self.norm_pre_attn.weight, hidden_states.dtype),
            )
            hidden_states = self.feed_forward1(hidden_states)
            residual = hidden_states
            hidden_states = torch.clamp(hidden_states, -gradient_clipping, gradient_clipping)
            hidden_states = self.norm_pre_attn(hidden_states)
            hidden_states, _ = self.self_attn(
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
            )
            hidden_states = torch.clamp(hidden_states, -gradient_clipping, gradient_clipping)
            hidden_states = self.norm_post_attn(hidden_states)
            hidden_states += residual
            hidden_states = self.lconv1d(hidden_states)
            hidden_states = self.feed_forward2(hidden_states)
            hidden_states = torch.clamp(hidden_states, -gradient_clipping, gradient_clipping)
            hidden_states = self.norm_out(hidden_states)
            return hidden_states

        Gemma4AudioFeedForward.forward = _ff_forward
        Gemma4AudioLightConv1d.forward = _lconv_forward
        Gemma4AudioLayer.forward = _layer_forward
        logger.info("Patched 3 Gemma4 audio classes for BitsAndBytes compatibility")

    def preload(self):
        """Pre-load model (call at startup to avoid first-request latency)."""
        self._load_model()

    def _normalize_audio(
        self,
        audio: bytes | np.ndarray | None,
    ) -> Optional[np.ndarray]:
        """Convert raw PCM16 audio bytes into the waveform Gemma's processor expects."""
        if audio is None:
            return None

        if isinstance(audio, np.ndarray):
            waveform = audio.astype(np.float32, copy=False)
        else:
            pcm = np.frombuffer(bytes(audio), dtype=np.int16)
            waveform = pcm.astype(np.float32) / 32768.0

        if waveform.ndim > 1:
            waveform = waveform.reshape(-1)
        return waveform

    def _build_messages(
        self,
        text: str,
        history: list[dict] | None = None,
        audio: bytes | None = None,
        images: list | None = None,
    ) -> tuple[list[dict], Optional[np.ndarray], Optional[list]]:
        """
        Build chat messages in Gemma's expected format.

        Audio and images are added as placeholders in the prompt while the raw
        waveform / image objects are passed separately to the processor.
        """
        messages = []

        # Add history (trimmed to max turns)
        if history:
            max_msgs = self.context_max_turns * 2  # user + assistant pairs
            messages.extend(history[-max_msgs:])

        # Build user content parts
        content_parts = []
        audio_waveform = self._normalize_audio(audio)
        image_inputs = list(images) if images else None

        if audio_waveform is not None:
            content_parts.append({"type": "audio"})

        if image_inputs:
            for _ in image_inputs:
                content_parts.append({"type": "image"})

        if audio_waveform is not None and not text:
            text = REALTIME_AUDIO_REPLY_PROMPT

        if text:
            content_parts.append({"type": "text", "text": text})

        messages.append({"role": "user", "content": content_parts})
        return messages, audio_waveform, image_inputs

    def _generate(
        self,
        messages: list[dict],
        audio_inputs: Optional[np.ndarray] = None,
        image_inputs: Optional[list] = None,
        stream: bool = False,
    ):
        """Synchronous generation (runs in executor thread)."""
        import torch

        self._load_model()

        prompt = self._processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self._processor(
            text=prompt,
            audio=audio_inputs,
            images=image_inputs,
            return_tensors="pt",
        ).to(self._model.device)

        input_len = inputs["input_ids"].shape[-1]
        logger.info(
            "Gemma generate start (stream=%s, max_new_tokens=%s, input_tokens=%s, has_audio=%s, has_images=%s)",
            stream,
            self.max_new_tokens,
            input_len,
            audio_inputs is not None,
            bool(image_inputs),
        )
        started_at = time.perf_counter()

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
                    target=self._model.generate, kwargs=gen_kwargs, daemon=True
                )
                thread.start()
                logger.info("Gemma stream generation thread started")
                return streamer, thread
            else:
                output = self._model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    do_sample=self.temperature > 0,
                )
                decoded = self._processor.decode(
                    output[0][input_len:], skip_special_tokens=True
                )
                logger.info("Gemma generate finished in %.1fs", time.perf_counter() - started_at)
                return decoded

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
        messages, audio_inputs, image_inputs = self._build_messages(text, history, audio, images)
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            self._executor, self._generate, messages, audio_inputs, image_inputs, False
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
        messages, audio_inputs, image_inputs = self._build_messages(text, history, audio, images)
        loop = asyncio.get_running_loop()
        streamer, thread = await loop.run_in_executor(
            self._executor, self._generate, messages, audio_inputs, image_inputs, True
        )
        stream_queue: asyncio.Queue[object] = asyncio.Queue()
        stream_done = object()

        def forward_stream() -> None:
            try:
                for token in streamer:
                    if token:
                        loop.call_soon_threadsafe(stream_queue.put_nowait, token)
            except Exception as exc:
                loop.call_soon_threadsafe(stream_queue.put_nowait, exc)
            finally:
                loop.call_soon_threadsafe(stream_queue.put_nowait, stream_done)

        forwarder = threading.Thread(target=forward_stream, daemon=True, name="GemmaStreamerForwarder")
        forwarder.start()

        try:
            while True:
                item = await stream_queue.get()
                if item is stream_done:
                    break
                if isinstance(item, Exception):
                    raise item
                yield item
        finally:
            forwarder.join(timeout=5)
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
        self._executor.shutdown(wait=False)
        logger.info("GemmaProvider cleaned up")
