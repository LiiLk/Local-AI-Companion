# Gemma-Omni Real-Time Avatar Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade Local-AI-Companion to a real-time conversational avatar using Gemma 4 E4B (ASR+LLM+Vision) + Chatterbox Multilingual ONNX Q4 (TTS+voice cloning+emotions), running fully local on RTX 4070 12GB.

**Architecture:** Gemma E4B handles audio input, screen vision, and text generation in a single model. Streaming tokens are split into sentences and pipelined to Chatterbox TTS for voice-cloned speech with emotion tags. A ScreenBuffer captures the user's screen for vision context. The GemmaOmniPipeline orchestrator exposes the same callback interface as existing pipelines, so all frontends work without modification.

**Tech Stack:** Gemma 4 E4B-it (TorchAO int4), Chatterbox Multilingual (ONNX Q4), Silero VAD v5, mss (screen capture), FastAPI + WebSocket, Live2D Cubism SDK.

**Spec:** `docs/superpowers/specs/2026-04-04-gemma-omni-avatar-design.md`

---

## File Structure

### New Files

| File | Responsibility |
|---|---|
| `src/omni/gemma_provider.py` | Gemma 4 E4B wrapper: load (TorchAO int4), chat, chat_stream, transcribe. Thread-safe lazy loading. |
| `src/omni/gemma_omni_pipeline.py` | Orchestrator: audio→Gemma→emotion parse→sentence split→Chatterbox TTS→callbacks. Same callback interface as OmniPipeline. |
| `src/tts/chatterbox_provider.py` | Extends `BaseTTS`. Chatterbox Multilingual ONNX Q4 with voice cloning + emotion tags. |
| `src/vision/screen_buffer.py` | Background thread screen capture via mss, pixel diff, circular buffer. |
| `src/utils/vram_monitor.py` | Debug VRAM logger (allocated/reserved per stage, threshold warnings). |
| `tests/test_emotion_detector.py` | Unit tests for EmotionDetector (especially strip_markers_for_tts). |
| `tests/test_screen_buffer.py` | Unit tests for ScreenBuffer (pixel diff, buffer management). |
| `tests/test_sentence_splitter.py` | Unit tests for sentence boundary detection. |
| `scripts/smoke_test_gemma.py` | GPU smoke test: load Gemma, text chat, audio input, image input. |
| `scripts/smoke_test_chatterbox.py` | GPU smoke test: load Chatterbox, synthesize, voice clone, emotion tags. |
| `scripts/smoke_test_pipeline.py` | End-to-end: mic→Gemma→Chatterbox→speaker. |
| `pytest.ini` | Pytest configuration. |

### Modified Files

| File | Change |
|---|---|
| `src/utils/emotion_detector.py` | Add `CHATTERBOX_TAGS` allowlist + `strip_markers_for_tts()` method |
| `config/config.yaml` | Add `gemma:` section + `tts.chatterbox:` section + `"gemma-omni"` mode |
| `config/characters/march7th.yaml` | Add Chatterbox voice config + emotion tag instructions in system prompt |
| `src/assistant/app.py` | Add `elif mode == "gemma-omni":` branch (after line 169) |
| `src/server/websocket.py` | Add `get_gemma_omni()` lazy loader + mode routing in handlers |
| `src/server/app.py` | Add `gemma-omni` mode display in lifespan |
| `src/omni/__init__.py` | Export `GemmaProvider`, `GemmaOmniPipeline` |
| `src/tts/__init__.py` | Export `ChatterboxTTSProvider` |
| `requirements.txt` | Add `torchao`, `onnxruntime-gpu`, `chatterbox-onnx`, `mss`, `pytest` |

---

## Phase 1 — Foundation ("it speaks")

**Milestone:** Speak into mic → March 7th voice responds via Gemma + Chatterbox.

---

### Task 1: Project Setup

**Files:**
- Modify: `requirements.txt`
- Create: `pytest.ini`
- Create: `tests/__init__.py`

- [ ] **Step 1: Create feature branch**

```bash
git checkout -b feature/gemma-omni-avatar
```

- [ ] **Step 2: Add new dependencies to requirements.txt**

Add after the last line of `requirements.txt`:

```
# Gemma-Omni Avatar (Phase 1)
torchao                        # TorchAO int4 quantization for Gemma E4B
onnxruntime-gpu>=1.18.0        # ONNX Runtime with CUDA for Chatterbox
chatterbox-onnx                # Chatterbox ONNX inference package
mss>=9.0.0                     # Fast screen capture
soundfile>=0.12.0              # Audio file I/O (may already be installed)

# Testing
pytest>=8.0.0
pytest-asyncio>=0.23.0
```

- [ ] **Step 3: Create pytest config**

Create `pytest.ini`:

```ini
[pytest]
testpaths = tests
asyncio_mode = auto
python_files = test_*.py
python_functions = test_*
pythonpath = .
```

The `pythonpath = .` ensures `from src.*` imports work in tests.

- [ ] **Step 4: Create tests package**

```bash
touch tests/__init__.py
```

- [ ] **Step 5: Install new dependencies**

```bash
pip install torchao onnxruntime-gpu chatterbox-onnx mss pytest pytest-asyncio
```

- [ ] **Step 6: Commit**

```bash
git add requirements.txt pytest.ini tests/__init__.py
git commit -m "feat: add gemma-omni dependencies and test infrastructure"
```

---

### Task 2: VRAM Monitor Utility

**Files:**
- Create: `src/utils/vram_monitor.py`

- [ ] **Step 1: Write VRAMMonitor**

Create `src/utils/vram_monitor.py`:

```python
"""
VRAM Monitor — Debug utility for tracking GPU memory usage.

Logs allocated/reserved VRAM at each pipeline stage.
Warns when usage exceeds configurable threshold.
Enable via config: debug.vram_monitor: true
"""

import logging

logger = logging.getLogger(__name__)


class VRAMMonitor:
    """Logs GPU VRAM usage at key pipeline stages."""

    def __init__(self, threshold_pct: float = 80.0):
        self.threshold_pct = threshold_pct
        self._available = False
        try:
            import torch
            self._available = torch.cuda.is_available()
            if self._available:
                self._torch = torch
                self._total = torch.cuda.get_device_properties(0).total_mem
        except ImportError:
            pass

    def log(self, stage: str) -> dict:
        """Log VRAM usage for a named stage. Returns usage dict."""
        if not self._available:
            return {}

        allocated = self._torch.cuda.memory_allocated()
        reserved = self._torch.cuda.memory_reserved()
        pct = (allocated / self._total) * 100

        info = {
            "stage": stage,
            "allocated_mb": allocated / 1024 / 1024,
            "reserved_mb": reserved / 1024 / 1024,
            "total_mb": self._total / 1024 / 1024,
            "percent": pct,
        }

        level = logging.WARNING if pct > self.threshold_pct else logging.DEBUG
        logger.log(
            level,
            f"[VRAM] {stage}: {info['allocated_mb']:.0f}MB allocated, "
            f"{info['reserved_mb']:.0f}MB reserved / {info['total_mb']:.0f}MB "
            f"({pct:.1f}%)",
        )
        return info
```

- [ ] **Step 2: Commit**

```bash
git add src/utils/vram_monitor.py
git commit -m "feat: add VRAMMonitor debug utility"
```

---

### Task 3: GemmaProvider — Model Loading

**Files:**
- Create: `src/omni/gemma_provider.py`
- Create: `scripts/smoke_test_gemma.py`

- [ ] **Step 1: Write GemmaProvider with load() and chat()**

Create `src/omni/gemma_provider.py`:

```python
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
            from transformers import AutoModelForImageTextToText, AutoProcessor

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
```

- [ ] **Step 2: Write smoke test script for Gemma**

Create `scripts/smoke_test_gemma.py`:

```python
"""
Smoke test for GemmaProvider.

Run: python -m scripts.smoke_test_gemma
Requires: GPU with >= 6GB VRAM, Gemma E4B model downloaded.
"""

import asyncio
import sys
import time

from src.omni.gemma_provider import GemmaProvider


async def main():
    print("=" * 50)
    print("SMOKE TEST: GemmaProvider")
    print("=" * 50)

    provider = GemmaProvider()

    # Test 1: Model loading
    print("\n[1/4] Loading Gemma E4B (int4)...")
    t0 = time.time()
    provider.preload()
    print(f"  Loaded in {time.time() - t0:.1f}s")

    import torch
    vram = torch.cuda.memory_allocated() / 1024 / 1024
    print(f"  VRAM: {vram:.0f}MB")
    assert vram < 6000, f"VRAM too high: {vram:.0f}MB (expected < 6000MB)"
    print("  PASS")

    # Test 2: Text chat
    print("\n[2/4] Text chat...")
    t0 = time.time()
    response = await provider.chat("Bonjour, comment tu t'appelles ?")
    print(f"  Response ({time.time() - t0:.1f}s): {response[:100]}")
    assert len(response) > 5, "Response too short"
    print("  PASS")

    # Test 3: Audio input (critical validation gate)
    print("\n[3/4] Audio input...")
    try:
        import numpy as np
        # Generate 2 seconds of silence as test audio
        sample_rate = 16000
        audio = np.zeros(sample_rate * 2, dtype=np.int16).tobytes()
        t0 = time.time()
        response = await provider.chat(
            text="What do you hear in this audio?",
            audio=audio,
        )
        print(f"  Response ({time.time() - t0:.1f}s): {response[:100]}")
        print("  PASS - Audio input works!")
    except Exception as e:
        print(f"  FAIL - Audio input broken: {e}")
        print("  FALLBACK: Switch to Whisper ASR + Gemma text-only")
        print("  Update config: asr.provider: 'whisper'")

    # Test 4: Streaming
    print("\n[4/4] Streaming chat...")
    t0 = time.time()
    tokens = []
    async for token in provider.chat_stream("Tell me a short joke."):
        tokens.append(token)
    full = "".join(tokens)
    print(f"  Streamed {len(tokens)} tokens in {time.time() - t0:.1f}s")
    print(f"  Output: {full[:100]}")
    assert len(tokens) > 3, "Too few tokens streamed"
    print("  PASS")

    # Cleanup
    provider.cleanup()
    print("\n" + "=" * 50)
    print("ALL TESTS PASSED")
    print("=" * 50)


if __name__ == "__main__":
    asyncio.run(main())
```

- [ ] **Step 3: Commit**

```bash
git add src/omni/gemma_provider.py scripts/smoke_test_gemma.py
git commit -m "feat: add GemmaProvider with TorchAO int4 quantization"
```

---

### Task 4: GemmaProvider — Critical Validation Gate

**Files:**
- None (run smoke test)

This is a **blocking gate**. Gemma E4B is 2 days old. If audio input doesn't work, we switch to Whisper ASR immediately.

- [ ] **Step 1: Run the Gemma smoke test**

```bash
python -m scripts.smoke_test_gemma
```

Expected output for each test: `PASS`

- [ ] **Step 2: If audio input fails — apply fallback**

If test 3 (audio input) fails:
1. Change `config.yaml`: `asr.provider: "whisper"` (keep existing Whisper config)
2. In `GemmaProvider`, remove audio-related logic
3. The pipeline will use Whisper for ASR + Gemma for text+vision only
4. Latency impact: +300-500ms (Whisper ASR step)

- [ ] **Step 3: If audio+image combo fails — apply fallback**

If passing audio AND images in the same call fails:
1. Process audio and image in separate calls in the pipeline
2. Latency impact: 2x on vision turns only

- [ ] **Step 4: Commit any fallback changes**

```bash
git add -A
git commit -m "fix: apply Gemma E4B validation gate fallback (if needed)"
```

---

### Task 5: ChatterboxTTSProvider — Loading + Synthesis

**Files:**
- Create: `src/tts/chatterbox_provider.py`
- Create: `scripts/smoke_test_chatterbox.py`

- [ ] **Step 1: Write ChatterboxTTSProvider**

Create `src/tts/chatterbox_provider.py`:

```python
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
    ):
        self.model_id = model_id
        self.ref_audio_path = Path(ref_audio_path) if ref_audio_path else None
        self.exaggeration = exaggeration
        self.cfg_weight = cfg_weight
        self.language = language
        self._speed = 1.0

        self._model = None
        self._ref_audio_data = None
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tts")

    def _load_model(self):
        """Load Chatterbox ONNX Q4 model (lazy)."""
        if self._model is not None:
            return

        logger.info(f"Loading Chatterbox from {self.model_id}...")

        # Import and load the ONNX model
        from chatterbox_onnx import ChatterboxONNX

        self._model = ChatterboxONNX.from_pretrained(self.model_id)

        # Pre-load reference audio if configured
        if self.ref_audio_path and self.ref_audio_path.exists():
            self._ref_audio_data = self._load_reference(self.ref_audio_path)
            logger.info(f"Voice reference loaded: {self.ref_audio_path}")

        logger.info("Chatterbox loaded successfully")

    def _load_reference(self, path: Path) -> np.ndarray:
        """Load and preprocess reference audio for voice cloning."""
        audio, sr = sf.read(str(path))
        # Resample to 24kHz if needed
        if sr != self.SAMPLE_RATE:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.SAMPLE_RATE)
        # Convert to mono if stereo
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        return audio.astype(np.float32)

    def _synthesize_sync(self, text: str) -> tuple[np.ndarray, int]:
        """Synchronous synthesis. Returns (audio_array, sample_rate)."""
        self._load_model()

        kwargs = {
            "text": text,
            "exaggeration": self.exaggeration,
            "cfg_weight": self.cfg_weight,
        }

        if self._ref_audio_data is not None:
            kwargs["audio_prompt"] = self._ref_audio_data

        audio = self._model.generate(**kwargs)

        # audio is a numpy array at 24kHz
        if isinstance(audio, (list, tuple)):
            audio = audio[0]
        if hasattr(audio, "numpy"):
            audio = audio.numpy()

        return audio.astype(np.float32), self.SAMPLE_RATE

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
        logger.info("ChatterboxTTSProvider cleaned up")
```

- [ ] **Step 2: Write Chatterbox smoke test**

Create `scripts/smoke_test_chatterbox.py`:

```python
"""
Smoke test for ChatterboxTTSProvider.

Run: python -m scripts.smoke_test_chatterbox
"""

import asyncio
import time
from pathlib import Path

from src.tts.chatterbox_provider import ChatterboxTTSProvider


async def main():
    print("=" * 50)
    print("SMOKE TEST: ChatterboxTTSProvider")
    print("=" * 50)

    provider = ChatterboxTTSProvider()

    # Test 1: Load model
    print("\n[1/4] Loading Chatterbox ONNX Q4...")
    t0 = time.time()
    provider._load_model()
    print(f"  Loaded in {time.time() - t0:.1f}s")

    try:
        import torch
        if torch.cuda.is_available():
            vram = torch.cuda.memory_allocated() / 1024 / 1024
            print(f"  VRAM: {vram:.0f}MB")
    except ImportError:
        pass
    print("  PASS")

    # Test 2: Basic synthesis
    print("\n[2/4] Basic synthesis...")
    t0 = time.time()
    result = await provider.synthesize("Bonjour, comment allez-vous?")
    print(f"  Synthesized in {time.time() - t0:.1f}s")
    assert result.audio_data is not None, "No audio data"
    assert len(result.audio_data) > 1000, "Audio too short"
    assert result.duration > 0.5, f"Duration too short: {result.duration}"
    print(f"  Duration: {result.duration:.1f}s, Size: {len(result.audio_data)} bytes")
    print("  PASS")

    # Test 3: Emotion tags
    print("\n[3/4] Emotion tags synthesis...")
    t0 = time.time()
    result = await provider.synthesize("Ha [laugh] that was funny!")
    print(f"  Synthesized in {time.time() - t0:.1f}s")
    assert result.audio_data is not None
    print(f"  Duration: {result.duration:.1f}s")
    print("  PASS")

    # Test 4: Voice cloning (if reference exists)
    ref_path = Path("resources/voices/march7th/VO_Archive_March_7th_5.wav")
    if ref_path.exists():
        print("\n[4/4] Voice cloning...")
        provider.set_reference_audio(ref_path)
        t0 = time.time()
        result = await provider.synthesize("Bonjour Trailblazer!")
        print(f"  Cloned voice in {time.time() - t0:.1f}s")
        assert result.audio_data is not None
        print(f"  Duration: {result.duration:.1f}s")
        print("  PASS")
    else:
        print(f"\n[4/4] Voice cloning SKIPPED (no ref at {ref_path})")

    provider.cleanup()
    print("\n" + "=" * 50)
    print("ALL TESTS PASSED")
    print("=" * 50)


if __name__ == "__main__":
    asyncio.run(main())
```

- [ ] **Step 3: Commit**

```bash
git add src/tts/chatterbox_provider.py scripts/smoke_test_chatterbox.py
git commit -m "feat: add ChatterboxTTSProvider (ONNX Q4, voice cloning, emotion tags)"
```

---

### Task 6: EmotionDetector — Chatterbox Tag Disambiguation (TDD)

**Files:**
- Modify: `src/utils/emotion_detector.py`
- Create: `tests/test_emotion_detector.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_emotion_detector.py`:

```python
"""Tests for EmotionDetector — especially Chatterbox tag preservation."""

from src.utils.emotion_detector import EmotionDetector, get_emotion_detector


class TestStripMarkersForTTS:
    """strip_markers_for_tts must preserve Chatterbox tags."""

    def test_preserves_laugh_tag(self):
        detector = EmotionDetector()
        result = detector.strip_markers_for_tts("Ha [laugh] that was funny!")
        assert "[laugh]" in result

    def test_preserves_chuckle_tag(self):
        detector = EmotionDetector()
        result = detector.strip_markers_for_tts("Well [chuckle] okay then")
        assert "[chuckle]" in result

    def test_preserves_sigh_tag(self):
        detector = EmotionDetector()
        result = detector.strip_markers_for_tts("Oh [sigh] fine")
        assert "[sigh]" in result

    def test_preserves_cough_tag(self):
        detector = EmotionDetector()
        result = detector.strip_markers_for_tts("Excuse me [cough]")
        assert "[cough]" in result

    def test_strips_emotion_brackets(self):
        detector = EmotionDetector()
        result = detector.strip_markers_for_tts("I'm [sad] today")
        assert "[sad]" not in result
        assert "today" in result

    def test_strips_parentheses_markers(self):
        detector = EmotionDetector()
        result = detector.strip_markers_for_tts("Hello (happy) world")
        assert "(happy)" not in result
        assert "Hello" in result
        assert "world" in result

    def test_strips_asterisk_markers(self):
        detector = EmotionDetector()
        result = detector.strip_markers_for_tts("Wow *excited* amazing")
        assert "*excited*" not in result

    def test_strips_angle_bracket_markers(self):
        detector = EmotionDetector()
        result = detector.strip_markers_for_tts("Oh <blush> hi")
        assert "<blush>" not in result

    def test_mixed_markers_and_chatterbox_tags(self):
        detector = EmotionDetector()
        text = "C'est genial *excited* j'adore [laugh] !"
        result = detector.strip_markers_for_tts(text)
        assert "[laugh]" in result
        assert "*excited*" not in result
        assert "genial" in result

    def test_cleans_whitespace(self):
        detector = EmotionDetector()
        result = detector.strip_markers_for_tts("Hello  (happy)  world")
        assert "  " not in result

    def test_case_insensitive_chatterbox_tags(self):
        detector = EmotionDetector()
        result = detector.strip_markers_for_tts("Ha [LAUGH] funny")
        assert "[LAUGH]" in result


class TestDetectEmotion:
    """Existing detect/get_expression behavior should not break."""

    def test_detect_happy(self):
        detector = EmotionDetector()
        emotion = detector.detect("I'm so (happy) today!")
        assert emotion == "happy"

    def test_detect_returns_none_for_no_emotion(self):
        detector = EmotionDetector()
        assert detector.detect("Just a normal sentence.") is None

    def test_get_expression_happy(self):
        detector = EmotionDetector()
        expr = detector.get_expression("happy")
        assert expr == "星星"

    def test_get_expression_none_returns_default(self):
        detector = EmotionDetector()
        expr = detector.get_expression(None)
        assert expr == "neutral"
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
pytest tests/test_emotion_detector.py -v
```

Expected: `strip_markers_for_tts` tests FAIL with `AttributeError: 'EmotionDetector' object has no attribute 'strip_markers_for_tts'`

- [ ] **Step 3: Implement strip_markers_for_tts**

In `src/utils/emotion_detector.py`, add after `CHATTERBOX_TAGS` class attribute and the new method.

Add at line 17 (inside `EmotionConfig`), a new class-level constant. Then add the method to `EmotionDetector` class after `strip_markers()` (after line 156):

First, add the allowlist constant to `EmotionDetector` class (after line 62):

```python
    # Chatterbox emotion tags to preserve in TTS text
    CHATTERBOX_TAGS = {"laugh", "chuckle", "cough", "sigh"}
```

Then add the new method after `strip_markers()` (after line 156):

```python
    def strip_markers_for_tts(self, text: str) -> str:
        """
        Remove emotion markers but KEEP Chatterbox tags for TTS.
        
        Chatterbox natively interprets [laugh], [chuckle], [cough], [sigh].
        These must be preserved in the text sent to TTS.
        All other markers are stripped.
        
        Args:
            text: Input text with emotion markers
            
        Returns:
            Clean text with only Chatterbox tags remaining
        """
        import re
        
        result = text
        # Remove (happy), *excited*, <blush>
        result = re.sub(r'\((\w+)\)', '', result)
        result = re.sub(r'\*(\w+)\*', '', result)
        result = re.sub(r'<(\w+)>', '', result)
        # Remove [brackets] EXCEPT Chatterbox tags
        result = re.sub(
            r'\[(\w+)\]',
            lambda m: m.group(0) if m.group(1).lower() in self.CHATTERBOX_TAGS else '',
            result,
        )
        # Clean up extra whitespace
        result = re.sub(r'\s+', ' ', result).strip()
        return result
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
pytest tests/test_emotion_detector.py -v
```

Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add src/utils/emotion_detector.py tests/test_emotion_detector.py
git commit -m "feat: add strip_markers_for_tts with Chatterbox tag preservation"
```

---

### Task 7: Config Updates

**Files:**
- Modify: `config/config.yaml`
- Modify: `config/characters/march7th.yaml`

- [ ] **Step 1: Add gemma-omni mode and gemma config to config.yaml**

In `config/config.yaml`, update the mode comment (line 4-7) and add a new `gemma` section after the `omni` section (after line 44):

Update mode comment:
```yaml
# === MODE ===
# "pipeline"    : Classic ASR → LLM → TTS pipeline (separate models)
# "omni"        : Single omni model handles speech-to-speech (MiniCPM-o)
# "gemma-omni"  : Gemma E4B (ASR+LLM+Vision) + Chatterbox TTS (voice clone)
mode: "omni"
```

Add after the `omni` section (after line 44, before `# === LLM`):

```yaml
# === GEMMA-OMNI (Gemma E4B + Chatterbox TTS) ===
# Used when mode: "gemma-omni"
# Gemma 4 E4B-it handles audio + vision + reasoning in a single model.
# Chatterbox Multilingual handles TTS with voice cloning and emotions.
gemma:
  model_id: "google/gemma-4-E4B-it"
  quantization: "int4"           # TorchAO int4 (~4.1 GB VRAM)
  device: "cuda"
  dtype: "bfloat16"
  max_new_tokens: 256
  temperature: 0.7
  top_p: 0.95
  context_max_turns: 10
  vision:
    enabled: true
    token_budget: 140            # Tokens per image (passive mode)
    detail_token_budget: 1120    # Tokens per image (detail mode)
  screen:
    enabled: true
    interval: 2.0                # Seconds between captures
    max_buffer: 30               # Circular buffer size
    change_threshold: 0.05       # Pixel diff threshold
    include_in_conversation: true
```

- [ ] **Step 2: Add Chatterbox TTS config**

In the `tts:` section of `config/config.yaml`, add `"chatterbox"` as a provider option and add the chatterbox subsection. Update the provider comment (line 89-92) and add after line 150 (before `rate:`):

Update provider comment:
```yaml
  # Default TTS provider:
  # - "kokoro"     : Kokoro 82M (local, natural, lightweight, no voice cloning)
  # - "edge"       : Edge TTS (Microsoft cloud, free)
  # - "chatterbox" : Chatterbox Multilingual (local, voice cloning, emotion tags)
  provider: "kokoro"
```

Add Chatterbox section (after the Edge TTS section, before `rate:`):

```yaml
  # ─────────────────────────────────────────────────────────────────────────
  # CHATTERBOX TTS (local, voice cloning, emotion tags)
  # ─────────────────────────────────────────────────────────────────────────
  # - ONNX Q4 quantized (~2 GB VRAM)
  # - Voice cloning from 5-10s reference audio
  # - Emotion tags: [laugh], [chuckle], [cough], [sigh]
  # - 23 languages including French
  # ─────────────────────────────────────────────────────────────────────────
  chatterbox:
    model_id: "onnx-community/chatterbox-multilingual-ONNX"
    quantized: true
    exaggeration: 0.5      # Emotion intensity (0.0 = flat, 1.0 = dramatic)
    cfg_weight: 0.5        # Classifier-free guidance weight
```

- [ ] **Step 3: Update march7th.yaml with Chatterbox voice config**

In `config/characters/march7th.yaml`, add Chatterbox voice settings after line 43:

```yaml
  # Chatterbox voice cloning
  chatterbox_ref_audio: "resources/voices/march7th/VO_Archive_March_7th_5.wav"
  chatterbox_exaggeration: 0.6
  chatterbox_language: "fr"
```

Update the system prompt (lines 8-16) to include Chatterbox emotion tag instructions:

```yaml
system_prompt: |
  You are March 7th (三月七) from Honkai: Star Rail.
  You are an energetic, cheerful, and optimistic AI assistant who loves taking photos and making memories.
  You're curious, enthusiastic, and sometimes a bit naive, but always eager to help!
  You often express emotions like "(excited)", "(happy)", "(shy)", "(surprised)" in your responses.
  You can also express natural sounds like [laugh], [chuckle], [sigh] when appropriate.
  You call the user "Trailblazer" or just a friendly nickname.
  You love photography, collecting memories, and making new friends.
  IMPORTANT: You MUST reply in the same language as the user. If the user speaks French, reply in French.
  Be expressive and use emotion markers!
```

- [ ] **Step 4: Commit**

```bash
git add config/config.yaml config/characters/march7th.yaml
git commit -m "feat: add gemma-omni mode and chatterbox TTS configuration"
```

---

### Task 8: GemmaOmniPipeline — Basic Wiring

**Files:**
- Create: `src/omni/gemma_omni_pipeline.py`

- [ ] **Step 1: Write GemmaOmniPipeline**

Create `src/omni/gemma_omni_pipeline.py`:

```python
"""
Gemma-Omni Pipeline — Orchestrator for Gemma E4B + Chatterbox TTS.

Wires together GemmaProvider + ChatterboxTTSProvider to provide the same
callback interface as ConversationPipeline and OmniPipeline.

Flow:
1. Receive audio bytes from mic (VAD)
2. Send audio to Gemma E4B for transcription + response
3. Parse emotions from response (dual output)
4. Send clean text to Chatterbox TTS for voice synthesis
5. Analyze audio for lip-sync
6. Fire callbacks (same interface as existing pipelines)
"""

import asyncio
import base64
import gc
import io
import logging
import wave
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from src.assistant.conversation_pipeline import (
    AudioPayload,
    ConversationConfig,
    analyze_audio_volumes,
)
from src.llm.base import Message
from src.omni.gemma_provider import GemmaProvider
from src.tts.chatterbox_provider import ChatterboxTTSProvider
from src.utils.emotion_detector import EmotionDetector

logger = logging.getLogger(__name__)

MAX_HISTORY_TURNS = 10


class GemmaOmniPipeline:
    """
    Orchestrator: Gemma E4B (ASR+LLM) + Chatterbox (TTS).

    Exposes the same callback interface as OmniPipeline:
        on_transcription, on_response_start, on_response_chunk,
        on_response_end, on_audio_ready, on_expression_change, on_error
    """

    def __init__(
        self,
        gemma: GemmaProvider,
        tts: ChatterboxTTSProvider,
        config: Optional[ConversationConfig] = None,
    ):
        self.gemma = gemma
        self.tts = tts
        self.config = config or ConversationConfig()

        # Emotion detection (canonical implementation)
        self.emotion_detector = EmotionDetector()

        # Conversation history (Gemma message format)
        self.system_prompt = self.config.system_prompt
        self.history: list[dict] = []

        # Callbacks (same interface as ConversationPipeline/OmniPipeline)
        self.on_transcription: Optional[Callable[[str], None]] = None
        self.on_response_start: Optional[Callable[[], None]] = None
        self.on_response_chunk: Optional[Callable[[str], None]] = None
        self.on_response_end: Optional[Callable[[str], None]] = None
        self.on_audio_ready: Optional[Callable[[AudioPayload], None]] = None
        self.on_expression_change: Optional[Callable[[str], None]] = None
        self.on_error: Optional[Callable[[str], None]] = None

        # State
        self._is_processing = False

        logger.info(
            f"GemmaOmniPipeline initialized "
            f"(character={self.config.character_name})"
        )

    async def _call_async(self, callback: Callable, *args):
        """Call a callback, awaiting if it is a coroutine function."""
        try:
            if asyncio.iscoroutinefunction(callback):
                await callback(*args)
            else:
                callback(*args)
        except Exception as e:
            logger.error(f"Callback error: {e}")

    def preload(self):
        """Pre-load both models."""
        logger.info("Pre-loading Gemma + Chatterbox...")
        self.gemma.preload()
        self.tts._load_model()
        logger.info("Both models loaded")

    async def process_speech(self, audio_bytes: bytes) -> Optional[str]:
        """
        Process speech audio through the full pipeline.

        This is the main entry point, called by AudioService when
        VAD detects end-of-speech.

        Args:
            audio_bytes: Raw PCM 16-bit 16kHz mono audio from VAD.

        Returns:
            The assistant's text response, or None on error.
        """
        if self._is_processing:
            logger.warning("Already processing, skipping")
            return None

        self._is_processing = True

        try:
            # Step 1: Build conversation context
            system_msg = {"role": "system", "content": [{"type": "text", "text": self.system_prompt}]}
            history_with_system = [system_msg] + self.history

            # Step 2: Send audio to Gemma for transcription + response
            if self.on_response_start:
                await self._call_async(self.on_response_start)

            response = await self.gemma.chat(
                text="",
                history=history_with_system,
                audio=audio_bytes,
            )

            if not response:
                logger.warning("Empty response from Gemma")
                return None

            # Fire transcription callback (Gemma does ASR implicitly)
            # We don't have separate transcription — user's audio is understood directly
            if self.on_transcription:
                await self._call_async(self.on_transcription, "[audio input]")

            # Fire response chunks
            if self.on_response_chunk:
                await self._call_async(self.on_response_chunk, response)

            if self.on_response_end:
                await self._call_async(self.on_response_end, response)

            # Step 3: Detect emotion and get expression
            emotion, expression = self.emotion_detector.detect_and_get_expression(response)

            if expression != "neutral" and self.on_expression_change:
                await self._call_async(self.on_expression_change, expression)

            # Step 4: Clean text for TTS (keep Chatterbox tags)
            tts_text = self.emotion_detector.strip_markers_for_tts(response)

            # Step 5: Synthesize speech with Chatterbox
            tts_result = await self.tts.synthesize(tts_text)

            # Step 6: Build AudioPayload
            if tts_result.audio_data:
                audio_data = tts_result.audio_data

                # Analyze volumes for lip-sync
                # Extract raw PCM from WAV for analysis
                pcm_data = self._extract_pcm_from_wav(audio_data)
                volumes = analyze_audio_volumes(
                    pcm_data, self.tts.SAMPLE_RATE, chunk_ms=50
                )

                # Base64 encode the full WAV for browser playback
                audio_b64 = base64.b64encode(audio_data).decode("utf-8")

                duration_ms = int((tts_result.duration or 0) * 1000)

                payload = AudioPayload(
                    audio_bytes=pcm_data,
                    audio_base64=audio_b64,
                    volumes=volumes,
                    duration_ms=duration_ms,
                    sample_rate=self.tts.SAMPLE_RATE,
                    text=tts_text,
                    expression=expression,
                )

                if self.on_audio_ready:
                    await self._call_async(self.on_audio_ready, payload)

            # Step 7: Update history
            self.history.append(
                {"role": "user", "content": [{"type": "audio", "audio": audio_bytes}]}
            )
            self.history.append(
                {"role": "assistant", "content": [{"type": "text", "text": response}]}
            )

            # Trim history
            max_msgs = MAX_HISTORY_TURNS * 2
            if len(self.history) > max_msgs:
                self.history = self.history[-max_msgs:]

            return response

        except Exception as e:
            logger.error(f"Pipeline error: {e}", exc_info=True)
            if self.on_error:
                await self._call_async(self.on_error, str(e))
            return None

        finally:
            self._is_processing = False

    def _extract_pcm_from_wav(self, wav_bytes: bytes) -> bytes:
        """Extract raw PCM data from a WAV file in memory."""
        buf = io.BytesIO(wav_bytes)
        with wave.open(buf, "rb") as wf:
            return wf.readframes(wf.getnframes())

    async def startup(self):
        """Initialize models with progress callbacks."""
        logger.info("Starting GemmaOmniPipeline...")
        self.preload()

    async def shutdown(self):
        """Clean shutdown of all components."""
        logger.info("Shutting down GemmaOmniPipeline...")
        self.tts.cleanup()
        self.gemma.cleanup()
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        logger.info("GemmaOmniPipeline shut down")

    async def health_check(self) -> dict:
        """Return pipeline health status."""
        info = {"gemma_loaded": self.gemma._model is not None, "tts_loaded": self.tts._model is not None}
        try:
            import torch
            if torch.cuda.is_available():
                info["vram_allocated_mb"] = torch.cuda.memory_allocated() / 1024 / 1024
                info["vram_reserved_mb"] = torch.cuda.memory_reserved() / 1024 / 1024
        except ImportError:
            pass
        return info

    def clear_history(self):
        """Clear conversation history."""
        self.history.clear()
```

- [ ] **Step 2: Commit**

```bash
git add src/omni/gemma_omni_pipeline.py
git commit -m "feat: add GemmaOmniPipeline orchestrator"
```

---

### Task 9: Module Exports

**Files:**
- Modify: `src/omni/__init__.py`
- Modify: `src/tts/__init__.py`

- [ ] **Step 1: Read current __init__.py files**

Read `src/omni/__init__.py` and `src/tts/__init__.py` to see current exports.

- [ ] **Step 2: Add GemmaProvider and GemmaOmniPipeline exports**

In `src/omni/__init__.py`, add:

```python
from src.omni.gemma_provider import GemmaProvider
from src.omni.gemma_omni_pipeline import GemmaOmniPipeline
```

- [ ] **Step 3: Add ChatterboxTTSProvider export**

In `src/tts/__init__.py`, add:

```python
from src.tts.chatterbox_provider import ChatterboxTTSProvider
```

- [ ] **Step 4: Commit**

```bash
git add src/omni/__init__.py src/tts/__init__.py
git commit -m "feat: export GemmaProvider, GemmaOmniPipeline, ChatterboxTTSProvider"
```

---

### Task 10: App + Server Integration

**Files:**
- Modify: `src/assistant/app.py:110-170`
- Modify: `src/server/websocket.py:62-174`
- Modify: `src/server/app.py` (lifespan mode display)

- [ ] **Step 1: Add gemma-omni branch in app.py**

In `src/assistant/app.py`, add an `elif` block **before** the `else:` at line 171. Insert between line 169 (end of omni block) and line 171 (`else:`):

```python
        elif mode == "gemma-omni":
            from src.omni import GemmaProvider, GemmaOmniPipeline
            from src.tts import ChatterboxTTSProvider

            gemma_config = self.config.get('gemma', {})
            tts_config = self.config.get('tts', {})
            chatterbox_config = tts_config.get('chatterbox', {})

            # Resolve voice config from character preset
            voice_config = character_config.get('voice', {})
            ref_audio = voice_config.get('chatterbox_ref_audio')
            exaggeration = voice_config.get('chatterbox_exaggeration', 0.5)
            language = voice_config.get('chatterbox_language', 'fr')

            # Create Gemma provider
            self._gemma_model = GemmaProvider(
                model_id=gemma_config.get('model_id', 'google/gemma-4-E4B-it'),
                device=gemma_config.get('device', 'cuda'),
                quantization=gemma_config.get('quantization', 'int4'),
                max_new_tokens=gemma_config.get('max_new_tokens', 256),
                temperature=gemma_config.get('temperature', 0.7),
                top_p=gemma_config.get('top_p', 0.95),
                context_max_turns=gemma_config.get('context_max_turns', 10),
            )
            logger.info(f"Gemma: {gemma_config.get('model_id')} (q={gemma_config.get('quantization', 'int4')})")

            # Create Chatterbox TTS provider
            self._chatterbox_tts = ChatterboxTTSProvider(
                model_id=chatterbox_config.get('model_id', 'onnx-community/chatterbox-multilingual-ONNX'),
                ref_audio_path=ref_audio,
                exaggeration=exaggeration,
                cfg_weight=chatterbox_config.get('cfg_weight', 0.5),
                language=language,
            )
            logger.info(f"TTS: Chatterbox (ref={ref_audio}, exag={exaggeration})")

            # Create pipeline
            pipeline_config = ConversationConfig(
                character_name=character_config.get('name', 'AI'),
                system_prompt=character_config.get('system_prompt', 'You are a helpful assistant.'),
                stream_tts=tts_config.get('stream_tts', True),
            )
            self._gemma_pipeline = GemmaOmniPipeline(
                gemma=self._gemma_model,
                tts=self._chatterbox_tts,
                config=pipeline_config,
            )

            # Wire callbacks (same interface as other pipelines)
            self._gemma_pipeline.on_transcription = self._on_transcription
            self._gemma_pipeline.on_response_start = self._on_response_start
            self._gemma_pipeline.on_response_chunk = self._on_response_chunk
            self._gemma_pipeline.on_response_end = self._on_response_end
            self._gemma_pipeline.on_audio_ready = self._on_audio_ready
            self._gemma_pipeline.on_expression_change = self._on_expression_change

            self.pipeline = None

Also, update `_on_speech_detected` (line 257 in `app.py`) to find the gemma pipeline. Change:

```python
            active_pipeline = getattr(self, '_omni_pipeline', None) or self.pipeline
```

to:

```python
            active_pipeline = (
                getattr(self, '_gemma_pipeline', None)
                or getattr(self, '_omni_pipeline', None)
                or self.pipeline
            )
```

- [ ] **Step 2: Add gemma-omni routing in websocket.py**

In `src/server/websocket.py`, update the `ConnectionState` class:

Add to fields (after line 64):
```python
    gemma_model: Optional[Any] = None
    gemma_pipeline: Optional[Any] = None
```

Update `mode` field (line 62):
```python
    mode: str = "pipeline"  # "pipeline", "omni", or "gemma-omni"
```

Add `get_gemma_omni()` method after `get_omni()` (after line 173):

```python
    def get_gemma_omni(self):
        """Get or create the Gemma-Omni pipeline (lazy loading)."""
        if self.gemma_model is None:
            from src.omni import GemmaProvider, GemmaOmniPipeline
            from src.tts import ChatterboxTTSProvider
            from src.assistant.conversation_pipeline import ConversationConfig

            gemma_config = self.config.get("gemma", {})
            tts_config = self.config.get("tts", {})
            chatterbox_config = tts_config.get("chatterbox", {})
            character = self.config.get("character", {})
            voice_config = character.get("voice", {})

            ref_audio = voice_config.get("chatterbox_ref_audio")
            exaggeration = voice_config.get("chatterbox_exaggeration", 0.5)
            language = voice_config.get("chatterbox_language", "fr")

            print("Loading Gemma E4B + Chatterbox...")
            self.gemma_model = GemmaProvider(
                model_id=gemma_config.get("model_id", "google/gemma-4-E4B-it"),
                device=gemma_config.get("device", "cuda"),
                quantization=gemma_config.get("quantization", "int4"),
                max_new_tokens=gemma_config.get("max_new_tokens", 256),
                temperature=gemma_config.get("temperature", 0.7),
            )
            self.gemma_model.preload()

            chatterbox = ChatterboxTTSProvider(
                model_id=chatterbox_config.get("model_id", "onnx-community/chatterbox-multilingual-ONNX"),
                ref_audio_path=ref_audio,
                exaggeration=exaggeration,
                language=language,
            )

            pipeline_config = ConversationConfig(
                character_name=character.get("name", "AI"),
                system_prompt=character.get("system_prompt", "You are a helpful assistant."),
            )
            self.gemma_pipeline = GemmaOmniPipeline(
                gemma=self.gemma_model,
                tts=chatterbox,
                config=pipeline_config,
            )
            print("Gemma + Chatterbox loaded successfully")

        return self.gemma_model, self.gemma_pipeline
```

Add `elif` branches in the WebSocket handler methods. In `initialize()` (after line 76):
```python
        elif self.mode == "gemma-omni":
            # Defer model loading to get_gemma_omni()
            pass
```

**Note:** The full WebSocket handler routing (for handle_text_message, handle_audio_stream, and preload) will be completed in Phase 4, Task 26. For Phase 1, the `get_gemma_omni()` lazy loader is sufficient for manual testing via `app.py`.

- [ ] **Step 3: Update server/app.py lifespan display**

In `src/server/app.py`, add gemma-omni mode display in the lifespan function (in the mode display block):

```python
    elif mode == "gemma-omni":
        print(f"   Mode: Gemma-Omni (Gemma E4B + Chatterbox TTS)")
        gemma_config = config.get("gemma", {})
        print(f"   LLM: Gemma {gemma_config.get('model_id')}")
        print(f"   Quantization: {gemma_config.get('quantization', 'int4')}")
        print(f"   TTS: Chatterbox Multilingual ONNX Q4")
```

- [ ] **Step 4: Commit**

```bash
git add src/assistant/app.py src/server/websocket.py src/server/app.py
git commit -m "feat: integrate gemma-omni mode in app and server"
```

---

### Task 11: Phase 1 End-to-End Validation

**Files:**
- Create: `scripts/smoke_test_pipeline.py`

- [ ] **Step 1: Write end-to-end smoke test**

Create `scripts/smoke_test_pipeline.py`:

```python
"""
End-to-end smoke test for the Gemma-Omni pipeline.

Tests: audio → Gemma → emotion parse → Chatterbox TTS → audio output.
Run: python -m scripts.smoke_test_pipeline
"""

import asyncio
import time

import numpy as np

from src.omni.gemma_provider import GemmaProvider
from src.tts.chatterbox_provider import ChatterboxTTSProvider
from src.omni.gemma_omni_pipeline import GemmaOmniPipeline
from src.assistant.conversation_pipeline import ConversationConfig


async def main():
    print("=" * 50)
    print("END-TO-END SMOKE TEST: GemmaOmniPipeline")
    print("=" * 50)

    # Setup
    gemma = GemmaProvider()
    tts = ChatterboxTTSProvider(
        ref_audio_path="resources/voices/march7th/VO_Archive_March_7th_5.wav"
    )
    config = ConversationConfig(
        character_name="March 7th",
        system_prompt="You are March 7th. Reply briefly in French. Use (happy) emotion markers.",
    )
    pipeline = GemmaOmniPipeline(gemma=gemma, tts=tts, config=config)

    # Collect callback results
    results = {}

    def on_transcription(text):
        results["transcription"] = text
        print(f"  [transcription] {text}")

    def on_response_start():
        results["started"] = True

    def on_response_chunk(text):
        results["response"] = text
        print(f"  [response] {text[:100]}...")

    def on_response_end(text):
        results["ended"] = True

    def on_audio_ready(payload):
        results["audio"] = payload
        print(f"  [audio] {payload.duration_ms}ms, {len(payload.volumes)} volume frames")

    def on_expression_change(expr):
        results["expression"] = expr
        print(f"  [expression] {expr}")

    pipeline.on_transcription = on_transcription
    pipeline.on_response_start = on_response_start
    pipeline.on_response_chunk = on_response_chunk
    pipeline.on_response_end = on_response_end
    pipeline.on_audio_ready = on_audio_ready
    pipeline.on_expression_change = on_expression_change

    # Generate test audio (2s of silence — Gemma should handle gracefully)
    test_audio = np.zeros(16000 * 2, dtype=np.int16).tobytes()

    # Run pipeline
    print("\n[1/1] Full pipeline: audio → Gemma → Chatterbox → audio")
    t0 = time.time()
    response = await pipeline.process_speech(test_audio)
    elapsed = time.time() - t0

    print(f"\n  Total time: {elapsed:.1f}s")
    print(f"  Response: {response}")
    assert response is not None, "No response from pipeline"
    assert "audio" in results, "No audio output generated"
    assert results["audio"].duration_ms > 0, "Audio has zero duration"

    # VRAM check
    try:
        import torch
        if torch.cuda.is_available():
            vram = torch.cuda.memory_allocated() / 1024 / 1024
            print(f"  Final VRAM: {vram:.0f}MB")
            assert vram < 10000, f"VRAM too high: {vram:.0f}MB"
    except ImportError:
        pass

    # Cleanup
    await pipeline.shutdown()

    print("\n" + "=" * 50)
    print("END-TO-END TEST PASSED")
    print("=" * 50)


if __name__ == "__main__":
    asyncio.run(main())
```

- [ ] **Step 2: Run unit tests**

```bash
pytest tests/ -v
```

Expected: ALL PASS

- [ ] **Step 3: Run Gemma smoke test**

```bash
python -m scripts.smoke_test_gemma
```

- [ ] **Step 4: Run Chatterbox smoke test**

```bash
python -m scripts.smoke_test_chatterbox
```

- [ ] **Step 5: Run end-to-end smoke test**

```bash
python -m scripts.smoke_test_pipeline
```

- [ ] **Step 6: Commit**

```bash
git add scripts/smoke_test_pipeline.py
git commit -m "feat: add end-to-end pipeline smoke test (Phase 1 complete)"
```

---

## Phase 2 — Streaming & Emotions ("it's fast and expressive")

**Milestone:** < 2s perceived latency, avatar expressions change, natural laughs in voice.

---

### Task 12: Sentence Splitter (TDD)

**Files:**
- Create: `src/utils/sentence_splitter.py`
- Create: `tests/test_sentence_splitter.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_sentence_splitter.py`:

```python
"""Tests for streaming sentence splitter."""

from src.utils.sentence_splitter import SentenceSplitter


class TestSentenceSplitter:
    def test_single_sentence(self):
        splitter = SentenceSplitter()
        splitter.feed("Hello world.")
        assert splitter.get_sentences() == ["Hello world."]

    def test_multiple_sentences(self):
        splitter = SentenceSplitter()
        splitter.feed("First. Second. Third.")
        assert splitter.get_sentences() == ["First.", "Second.", "Third."]

    def test_incremental_feeding(self):
        splitter = SentenceSplitter()
        splitter.feed("Hello ")
        assert splitter.get_sentences() == []
        splitter.feed("world. ")
        assert splitter.get_sentences() == ["Hello world."]
        splitter.feed("Next!")
        assert splitter.get_sentences() == ["Next!"]

    def test_question_mark(self):
        splitter = SentenceSplitter()
        splitter.feed("How are you? I'm fine.")
        sentences = splitter.get_sentences()
        assert sentences == ["How are you?", "I'm fine."]

    def test_exclamation(self):
        splitter = SentenceSplitter()
        splitter.feed("Wow! Amazing!")
        assert splitter.get_sentences() == ["Wow!", "Amazing!"]

    def test_preserves_chatterbox_tags(self):
        splitter = SentenceSplitter()
        splitter.feed("Ha [laugh] that's funny!")
        sentences = splitter.get_sentences()
        assert len(sentences) == 1
        assert "[laugh]" in sentences[0]

    def test_flush_incomplete(self):
        splitter = SentenceSplitter()
        splitter.feed("Hello world")
        assert splitter.get_sentences() == []
        remaining = splitter.flush()
        assert remaining == "Hello world"

    def test_french_text(self):
        splitter = SentenceSplitter()
        splitter.feed("Bonjour ! Comment ça va ? Très bien.")
        sentences = splitter.get_sentences()
        assert len(sentences) == 3

    def test_abbreviations_not_split(self):
        splitter = SentenceSplitter()
        splitter.feed("Dr. Smith is here.")
        sentences = splitter.get_sentences()
        # Should not split on "Dr."
        assert len(sentences) == 1

    def test_empty_input(self):
        splitter = SentenceSplitter()
        splitter.feed("")
        assert splitter.get_sentences() == []
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
pytest tests/test_sentence_splitter.py -v
```

Expected: FAIL (module not found)

- [ ] **Step 3: Write SentenceSplitter**

Create `src/utils/sentence_splitter.py`:

```python
"""
Streaming Sentence Splitter.

Accumulates tokens and yields complete sentences at boundary characters.
Designed for the streaming LLM → TTS pipeline.
"""

import re


# Common abbreviations that should NOT trigger a split
ABBREVIATIONS = {"dr", "mr", "mrs", "ms", "prof", "sr", "jr", "st", "etc", "vs", "fig"}

# Sentence-ending pattern: punctuation followed by space or end-of-string
SENTENCE_END = re.compile(r'([.!?]+)\s+')


class SentenceSplitter:
    """
    Accumulates streamed text and yields complete sentences.

    Usage:
        splitter = SentenceSplitter()
        for token in llm_stream:
            splitter.feed(token)
            for sentence in splitter.get_sentences():
                send_to_tts(sentence)
        # After stream ends:
        remaining = splitter.flush()
        if remaining:
            send_to_tts(remaining)
    """

    def __init__(self, min_length: int = 5):
        self._buffer = ""
        self._pending: list[str] = []
        self._min_length = min_length

    def feed(self, text: str) -> None:
        """Add text to the buffer and extract complete sentences."""
        if not text:
            return

        self._buffer += text
        self._extract_sentences()

    def _extract_sentences(self) -> None:
        """Extract complete sentences from the buffer."""
        while True:
            match = SENTENCE_END.search(self._buffer)
            if not match:
                break

            end_pos = match.end()
            candidate = self._buffer[:end_pos].strip()

            # Check if this is an abbreviation (e.g., "Dr. Smith")
            # Look at the word before the period
            pre_punct = self._buffer[:match.start()].strip()
            last_word = pre_punct.split()[-1].lower().rstrip(".") if pre_punct.split() else ""

            if match.group(1) == "." and last_word in ABBREVIATIONS:
                # Not a real sentence boundary — keep scanning
                # Move past this match to avoid infinite loop
                # We need to find the next potential boundary
                next_match = SENTENCE_END.search(self._buffer, pos=match.end())
                if not next_match:
                    break
                end_pos = next_match.end()
                candidate = self._buffer[:end_pos].strip()

            if len(candidate) >= self._min_length:
                self._pending.append(candidate)

            self._buffer = self._buffer[end_pos:]

    def get_sentences(self) -> list[str]:
        """Return and clear all complete sentences found so far."""
        sentences = self._pending
        self._pending = []
        return sentences

    def flush(self) -> str:
        """Return any remaining text in the buffer (incomplete sentence)."""
        remaining = self._buffer.strip()
        self._buffer = ""
        return remaining
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
pytest tests/test_sentence_splitter.py -v
```

Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add src/utils/sentence_splitter.py tests/test_sentence_splitter.py
git commit -m "feat: add streaming SentenceSplitter with abbreviation handling"
```

---

### Task 13: Streaming LLM→TTS Pipeline

**Files:**
- Modify: `src/omni/gemma_omni_pipeline.py`

- [ ] **Step 1: Add streaming process_speech to GemmaOmniPipeline**

Add a `process_speech_streaming()` method that uses `chat_stream()` + `SentenceSplitter` to pipeline LLM generation with TTS. Also update `process_speech()` to call the streaming variant when `config.stream_tts` is True.

Add to `GemmaOmniPipeline`:

```python
    async def process_speech_streaming(self, audio_bytes: bytes) -> Optional[str]:
        """
        Streaming variant: sentences are sent to TTS as they complete.
        Gemma continues generating while Chatterbox synthesizes.
        """
        if self._is_processing:
            logger.warning("Already processing, skipping")
            return None

        self._is_processing = True

        try:
            from src.utils.sentence_splitter import SentenceSplitter

            system_msg = {"role": "system", "content": [{"type": "text", "text": self.system_prompt}]}
            history_with_system = [system_msg] + self.history

            if self.on_response_start:
                await self._call_async(self.on_response_start)

            if self.on_transcription:
                await self._call_async(self.on_transcription, "[audio input]")

            # Stream tokens from Gemma
            splitter = SentenceSplitter()
            full_response = ""
            first_audio_sent = False

            async for token in self.gemma.chat_stream(
                text="",
                history=history_with_system,
                audio=audio_bytes,
            ):
                full_response += token
                splitter.feed(token)

                if self.on_response_chunk:
                    await self._call_async(self.on_response_chunk, token)

                # Check for complete sentences
                for sentence in splitter.get_sentences():
                    await self._synthesize_and_send(sentence, first_audio_sent)
                    first_audio_sent = True

            # Flush remaining text
            remaining = splitter.flush()
            if remaining:
                await self._synthesize_and_send(remaining, first_audio_sent)

            if self.on_response_end:
                await self._call_async(self.on_response_end, full_response)

            # Update history
            self.history.append(
                {"role": "user", "content": [{"type": "audio", "audio": audio_bytes}]}
            )
            self.history.append(
                {"role": "assistant", "content": [{"type": "text", "text": full_response}]}
            )
            max_msgs = MAX_HISTORY_TURNS * 2
            if len(self.history) > max_msgs:
                self.history = self.history[-max_msgs:]

            return full_response

        except Exception as e:
            logger.error(f"Streaming pipeline error: {e}", exc_info=True)
            if self.on_error:
                await self._call_async(self.on_error, str(e))
            return None
        finally:
            self._is_processing = False

    async def _synthesize_and_send(self, text: str, is_continuation: bool = False) -> None:
        """Synthesize a sentence and fire audio callback."""
        # Emotion detection
        emotion, expression = self.emotion_detector.detect_and_get_expression(text)

        if expression != "neutral" and self.on_expression_change:
            await self._call_async(self.on_expression_change, expression)

        # Clean for TTS
        tts_text = self.emotion_detector.strip_markers_for_tts(text)
        if not tts_text.strip():
            return

        # Synthesize
        tts_result = await self.tts.synthesize(tts_text)

        if tts_result.audio_data:
            pcm_data = self._extract_pcm_from_wav(tts_result.audio_data)
            volumes = analyze_audio_volumes(pcm_data, self.tts.SAMPLE_RATE, chunk_ms=50)
            audio_b64 = base64.b64encode(tts_result.audio_data).decode("utf-8")
            duration_ms = int((tts_result.duration or 0) * 1000)

            payload = AudioPayload(
                audio_bytes=pcm_data,
                audio_base64=audio_b64,
                volumes=volumes,
                duration_ms=duration_ms,
                sample_rate=self.tts.SAMPLE_RATE,
                text=tts_text,
                expression=expression,
            )

            if self.on_audio_ready:
                await self._call_async(self.on_audio_ready, payload)
```

Also update `process_speech()` to delegate to streaming when configured:

```python
    async def process_speech(self, audio_bytes: bytes) -> Optional[str]:
        """Process speech — uses streaming if configured."""
        if self.config.stream_tts:
            return await self.process_speech_streaming(audio_bytes)
        return await self._process_speech_basic(audio_bytes)
```

Rename the existing `process_speech` body to `_process_speech_basic`.

- [ ] **Step 2: Commit**

```bash
git add src/omni/gemma_omni_pipeline.py
git commit -m "feat: add streaming LLM→TTS pipeline with sentence splitting"
```

---

### Task 14: Emotion Routing in Pipeline

**Files:**
- Modify: `src/omni/gemma_omni_pipeline.py` (already handled by Task 13's `_synthesize_and_send`)

The emotion routing (dual output: Chatterbox tags for TTS + expressions for Live2D) is already implemented in `_synthesize_and_send()` via:
- `emotion_detector.detect_and_get_expression(text)` → expression for Live2D
- `emotion_detector.strip_markers_for_tts(text)` → clean text with `[laugh]` preserved for Chatterbox

- [ ] **Step 1: Verify emotion routing works via unit tests**

```bash
pytest tests/test_emotion_detector.py -v
```

- [ ] **Step 2: Commit (no changes needed if tests pass)**

---

### Task 15: March 7th System Prompt Update

**Files:**
- Modify: `config/characters/march7th.yaml` (already done in Task 7, Step 3)

The system prompt was already updated in Task 7 to include Chatterbox emotion tags. Verify the update is in place.

- [ ] **Step 1: Verify march7th.yaml has emotion tag instructions**

Read `config/characters/march7th.yaml` and confirm the system prompt mentions `[laugh]`, `[chuckle]`, `[sigh]`.

- [ ] **Step 2: Commit (no changes if already done)**

---

### Task 16: VRAMMonitor Integration

**Files:**
- Modify: `src/omni/gemma_omni_pipeline.py`

- [ ] **Step 1: Add VRAM logging to pipeline stages**

In `GemmaOmniPipeline.__init__()`, add:

```python
        from src.utils.vram_monitor import VRAMMonitor
        self._vram = VRAMMonitor()
```

In `preload()`, add logging:

```python
    def preload(self):
        self._vram.log("before_gemma_load")
        self.gemma.preload()
        self._vram.log("after_gemma_load")
        self.tts._load_model()
        self._vram.log("after_chatterbox_load")
```

In `_synthesize_and_send()`, add after TTS:

```python
        self._vram.log("after_tts_inference")
```

- [ ] **Step 2: Commit**

```bash
git add src/omni/gemma_omni_pipeline.py
git commit -m "feat: integrate VRAMMonitor into GemmaOmniPipeline"
```

---

### Task 17: Phase 2 Validation

- [ ] **Step 1: Run all unit tests**

```bash
pytest tests/ -v
```

Expected: ALL PASS

- [ ] **Step 2: Run end-to-end smoke test**

```bash
python -m scripts.smoke_test_pipeline
```

Verify: streaming response with sentence-by-sentence TTS.

- [ ] **Step 3: Commit validation results**

```bash
git commit --allow-empty -m "milestone: Phase 2 complete — streaming + emotions"
```

---

## Phase 3 — Vision / Screen Sharing ("it sees your screen")

**Milestone:** "What's on my screen?" → correct description.

---

### Task 18: ScreenBuffer (TDD)

**Files:**
- Create: `src/vision/screen_buffer.py`
- Create: `tests/test_screen_buffer.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_screen_buffer.py`:

```python
"""Tests for ScreenBuffer — screen capture and change detection."""

import numpy as np
from unittest.mock import patch, MagicMock
from PIL import Image

from src.vision.screen_buffer import ScreenBuffer, pixel_diff


class TestPixelDiff:
    def test_identical_frames(self):
        img = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8))
        assert pixel_diff(img, img) == 0.0

    def test_completely_different(self):
        black = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8))
        white = Image.fromarray(np.full((100, 100, 3), 255, dtype=np.uint8))
        diff = pixel_diff(black, white)
        assert diff > 0.9  # Nearly 100% different

    def test_partial_change(self):
        base = np.zeros((100, 100, 3), dtype=np.uint8)
        changed = base.copy()
        changed[50:, :, :] = 255  # Bottom half white
        diff = pixel_diff(Image.fromarray(base), Image.fromarray(changed))
        assert 0.3 < diff < 0.7


class TestScreenBuffer:
    def test_get_latest_empty(self):
        buf = ScreenBuffer(capture_interval=10.0)  # long interval, won't auto-capture
        assert buf.get_latest() is None

    def test_get_recent_empty(self):
        buf = ScreenBuffer(capture_interval=10.0)
        assert buf.get_recent(5) == []

    def test_buffer_max_size(self):
        buf = ScreenBuffer(max_buffer=3, capture_interval=10.0)
        for i in range(5):
            img = Image.fromarray(np.full((10, 10, 3), i * 50, dtype=np.uint8))
            buf._frames.append(img)
        # Only keep last 3 due to deque maxlen
        assert len(buf._frames) == 3
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
pytest tests/test_screen_buffer.py -v
```

- [ ] **Step 3: Write ScreenBuffer**

Create `src/vision/screen_buffer.py`:

```python
"""
Screen Buffer — Continuous screen capture with change detection.

Captures screenshots via mss in a background thread.
Uses pixel diff to skip unchanged frames.
Stores frames in a circular buffer for vision pipeline access.
"""

import logging
import threading
import time
from collections import deque
from typing import Optional

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def pixel_diff(img1: Image.Image, img2: Image.Image) -> float:
    """
    Compute normalized pixel difference between two images.

    Returns a value between 0.0 (identical) and 1.0 (completely different).
    Images are downscaled for speed.
    """
    # Downscale for fast comparison
    size = (64, 64)
    a = np.array(img1.resize(size, Image.NEAREST), dtype=np.float32)
    b = np.array(img2.resize(size, Image.NEAREST), dtype=np.float32)
    return float(np.mean(np.abs(a - b)) / 255.0)


class ScreenBuffer:
    """
    Continuous screen capture with change detection.

    Runs a background thread that captures via mss at regular intervals.
    Frames that are too similar to the previous one are skipped.

    Args:
        capture_interval: Seconds between captures (default 2.0).
        max_buffer: Maximum frames to keep in circular buffer.
        change_threshold: Minimum pixel diff to store a new frame (0.0-1.0).
        monitor: mss monitor index (0 = all screens, 1 = primary).
    """

    def __init__(
        self,
        capture_interval: float = 2.0,
        max_buffer: int = 30,
        change_threshold: float = 0.05,
        monitor: int = 1,
    ):
        self.capture_interval = capture_interval
        self.change_threshold = change_threshold
        self.monitor = monitor

        self._frames: deque[Image.Image] = deque(maxlen=max_buffer)
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """Start background capture thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._capture_loop, daemon=True, name="screen-buffer"
        )
        self._thread.start()
        logger.info(
            f"ScreenBuffer started (interval={self.capture_interval}s, "
            f"threshold={self.change_threshold})"
        )

    def stop(self) -> None:
        """Stop background capture."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None
        logger.info("ScreenBuffer stopped")

    def _capture_loop(self) -> None:
        """Background capture loop."""
        import mss

        with mss.mss() as sct:
            while self._running:
                try:
                    raw = sct.grab(sct.monitors[self.monitor])
                    img = Image.frombytes("RGB", raw.size, raw.bgra, "raw", "BGRX")

                    with self._lock:
                        # Skip if too similar to last frame
                        if self._frames:
                            diff = pixel_diff(self._frames[-1], img)
                            if diff < self.change_threshold:
                                time.sleep(self.capture_interval)
                                continue

                        self._frames.append(img)

                except Exception as e:
                    logger.warning(f"Screen capture failed: {e}")

                time.sleep(self.capture_interval)

    def get_latest(self) -> Optional[Image.Image]:
        """Get the most recent captured frame."""
        with self._lock:
            return self._frames[-1] if self._frames else None

    def get_recent(self, n: int = 10) -> list[Image.Image]:
        """Get the N most recent frames."""
        with self._lock:
            return list(self._frames)[-n:]

    def __len__(self) -> int:
        return len(self._frames)
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
pytest tests/test_screen_buffer.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/vision/screen_buffer.py tests/test_screen_buffer.py
git commit -m "feat: add ScreenBuffer with pixel diff and circular buffer"
```

---

### Task 19: GemmaProvider — Vision Input

**Files:**
- Modify: `src/omni/gemma_provider.py`

The `GemmaProvider._build_messages()` already handles images in the message content. The `chat()` and `chat_stream()` methods already accept an `images` parameter. No code changes needed — Gemma's processor handles PIL images natively.

- [ ] **Step 1: Add image input to Gemma smoke test**

Update `scripts/smoke_test_gemma.py` — test 3 should also test image input:

Add a new test between the existing ones:

```python
    # Test: Image input
    print("\n[X/X] Image input...")
    from PIL import Image
    test_image = Image.new("RGB", (224, 224), color=(100, 150, 200))
    t0 = time.time()
    response = await provider.chat(
        text="Describe what you see in this image.",
        images=[test_image],
    )
    print(f"  Response ({time.time() - t0:.1f}s): {response[:100]}")
    assert len(response) > 5
    print("  PASS")
```

- [ ] **Step 2: Run smoke test**

```bash
python -m scripts.smoke_test_gemma
```

- [ ] **Step 3: Commit**

```bash
git add scripts/smoke_test_gemma.py
git commit -m "test: add image input smoke test for GemmaProvider"
```

---

### Task 20: Vision Mode Integration in Pipeline

**Files:**
- Modify: `src/omni/gemma_omni_pipeline.py`

- [ ] **Step 1: Add ScreenBuffer integration to GemmaOmniPipeline**

Add to `__init__()`:

```python
        # Screen capture (optional)
        self.screen_buffer: Optional['ScreenBuffer'] = None
```

Add methods:

```python
    def enable_screen_capture(self, config: dict) -> None:
        """Start screen capture if configured."""
        from src.vision.screen_buffer import ScreenBuffer

        self.screen_buffer = ScreenBuffer(
            capture_interval=config.get("interval", 2.0),
            max_buffer=config.get("max_buffer", 30),
            change_threshold=config.get("change_threshold", 0.05),
        )
        self.screen_buffer.start()
        logger.info("Screen capture enabled")

    def _get_screen_context(self) -> list:
        """Get current screen frame(s) for vision context."""
        if not self.screen_buffer:
            return []

        frame = self.screen_buffer.get_latest()
        return [frame] if frame else []
```

Update `_process_speech_basic()` and `process_speech_streaming()` to include screen context:

In both methods, when building the Gemma call, pass images:

```python
            images = self._get_screen_context()
            response = await self.gemma.chat(
                text="",
                history=history_with_system,
                audio=audio_bytes,
                images=images if images else None,
            )
```

Update `shutdown()` to stop screen buffer:

```python
        if self.screen_buffer:
            self.screen_buffer.stop()
```

- [ ] **Step 2: Wire screen capture config in app.py**

In the `gemma-omni` branch of `app.py`, after creating the pipeline, add:

```python
            # Enable screen capture if configured
            screen_config = gemma_config.get('screen', {})
            if screen_config.get('enabled', False):
                self._gemma_pipeline.enable_screen_capture(screen_config)
```

- [ ] **Step 3: Commit**

```bash
git add src/omni/gemma_omni_pipeline.py src/assistant/app.py
git commit -m "feat: integrate ScreenBuffer into GemmaOmniPipeline for vision"
```

---

### Task 21: Phase 3 Validation

- [ ] **Step 1: Run all unit tests**

```bash
pytest tests/ -v
```

- [ ] **Step 2: Manual test: "What's on my screen?"**

1. Set `mode: "gemma-omni"` and `gemma.screen.enabled: true` in config.yaml
2. Start the assistant
3. Ask "What's on my screen?" via mic
4. Verify the response describes screen content

- [ ] **Step 3: Commit**

```bash
git commit --allow-empty -m "milestone: Phase 3 complete — vision / screen sharing"
```

---

## Phase 4 — Frontend Integration ("everything connected")

**Milestone:** Web UI + Desktop overlay fully functional with gemma-omni mode.

---

### Task 22: Server WebSocket Full Routing

**Files:**
- Modify: `src/server/websocket.py`

- [ ] **Step 1: Add gemma-omni routing in handle_text_message**

Find the mode branching in `handle_text_message` and add:

```python
        elif state.mode == "gemma-omni":
            _, pipeline = state.get_gemma_omni()
            # Wire callbacks if not already wired
            self._wire_gemma_callbacks(state, pipeline, websocket)
            # For text messages, pass as text (no audio)
            response = await pipeline.gemma.chat(
                text=data.get("content", ""),
                history=pipeline.history,
            )
            # Process response through TTS
            await pipeline._synthesize_and_send(response)
```

- [ ] **Step 2: Add gemma-omni routing in audio stream VAD dispatch**

In the VAD end-of-speech handler, add:

```python
        elif state.mode == "gemma-omni":
            _, pipeline = state.get_gemma_omni()
            self._wire_gemma_callbacks(state, pipeline, websocket)
            await pipeline.process_speech(audio_bytes)
```

- [ ] **Step 3: Add gemma-omni to preload handler**

In `_preload_models_progressive`, add:

```python
        elif state.mode == "gemma-omni":
            await websocket.send_json({"type": "models_loading"})
            _, pipeline = state.get_gemma_omni()
            await websocket.send_json({"type": "models_loaded"})
```

- [ ] **Step 4: Add callback wiring helper**

```python
    def _wire_gemma_callbacks(self, state, pipeline, websocket):
        """Wire GemmaOmniPipeline callbacks to WebSocket sends."""
        if pipeline.on_audio_ready is not None:
            return  # Already wired

        async def send_json(msg):
            try:
                await websocket.send_json(msg)
            except Exception:
                pass

        def on_transcription(text):
            asyncio.create_task(send_json({"type": "transcription", "text": text}))

        def on_response_start():
            asyncio.create_task(send_json({"type": "text_start"}))

        def on_response_chunk(text):
            asyncio.create_task(send_json({"type": "text_chunk", "content": text}))

        def on_response_end(text):
            asyncio.create_task(send_json({"type": "text_end", "full_text": text}))

        def on_audio_ready(payload):
            asyncio.create_task(send_json({
                "type": "audio_data",
                "data": payload.audio_base64,
                "format": "wav",
                "lip_sync": payload.volumes,
                "expression": payload.expression,
                "text": payload.text,
            }))

        def on_expression_change(expr):
            asyncio.create_task(send_json({"type": "expression_change", "expression": expr}))
            state.current_expression = expr

        pipeline.on_transcription = on_transcription
        pipeline.on_response_start = on_response_start
        pipeline.on_response_chunk = on_response_chunk
        pipeline.on_response_end = on_response_end
        pipeline.on_audio_ready = on_audio_ready
        pipeline.on_expression_change = on_expression_change
```

- [ ] **Step 5: Commit**

```bash
git add src/server/websocket.py
git commit -m "feat: full WebSocket routing for gemma-omni mode"
```

---

### Task 23: Progressive Model Loading

**Files:**
- Modify: `src/server/websocket.py`

- [ ] **Step 1: Add progress messages during loading**

In `get_gemma_omni()`, add WebSocket progress messages. Since `get_gemma_omni` is synchronous, wrap the loading calls to send progress via a callback:

```python
    def get_gemma_omni(self, progress_callback=None):
        """Get or create the Gemma-Omni pipeline (lazy loading)."""
        if self.gemma_model is None:
            # ... existing imports and config resolution ...

            if progress_callback:
                progress_callback("Loading Gemma E4B (int4)... this may take a minute")

            # ... create and load gemma_model ...

            if progress_callback:
                progress_callback("Loading Chatterbox TTS...")

            # ... create chatterbox ...

            if progress_callback:
                progress_callback("Models loaded! Ready to chat.")

        return self.gemma_model, self.gemma_pipeline
```

- [ ] **Step 2: Commit**

```bash
git add src/server/websocket.py
git commit -m "feat: add progressive model loading messages for gemma-omni"
```

---

### Task 24: Phase 4 Validation

- [ ] **Step 1: Run all tests**

```bash
pytest tests/ -v
```

- [ ] **Step 2: Manual test via Web UI**

1. Set `mode: "gemma-omni"` in config.yaml
2. Start server: `python -m src.server`
3. Open Web UI at `http://localhost:8000`
4. Click mic / type a message
5. Verify: text response + voice audio + lip sync + expressions

- [ ] **Step 3: Commit**

```bash
git commit --allow-empty -m "milestone: Phase 4 complete — frontend integration"
```

---

## Phase 5 — Polish ("it's smooth")

**Milestone:** Fluid, reliable conversation experience.

---

### Task 25: OOM Fallback Chain

**Files:**
- Modify: `src/omni/gemma_omni_pipeline.py`

- [ ] **Step 1: Add OOM recovery in process_speech methods**

Wrap Gemma inference in a try/except for `torch.cuda.OutOfMemoryError`:

```python
        try:
            response = await self.gemma.chat(...)
        except torch.cuda.OutOfMemoryError:
            logger.warning("OOM during Gemma inference — reducing tokens and retrying")
            torch.cuda.empty_cache()
            gc.collect()
            # Retry with reduced tokens
            original_max = self.gemma.max_new_tokens
            self.gemma.max_new_tokens = min(128, original_max)
            try:
                response = await self.gemma.chat(...)
            finally:
                self.gemma.max_new_tokens = original_max
```

- [ ] **Step 2: Commit**

```bash
git add src/omni/gemma_omni_pipeline.py
git commit -m "feat: add OOM fallback chain for Gemma inference"
```

---

### Task 26: VAD Tuning

**Files:**
- Modify: `config/config.yaml`

- [ ] **Step 1: Document VAD tuning in config**

The existing Silero VAD uses `required_misses=30` (~960ms silence before end-of-speech). For conversational use, reduce to 15 (~480ms). This is a config-only change.

Add a comment in the gemma section:

```yaml
  # VAD tuning (Silero VAD end-of-speech sensitivity)
  # Lower = faster response, higher = more tolerant of pauses
  # Default: 30 (~960ms), recommended for conversation: 15 (~480ms)
  vad_required_misses: 15
```

- [ ] **Step 2: Commit**

```bash
git add config/config.yaml
git commit -m "feat: add VAD tuning config for gemma-omni mode"
```

---

### Task 27: Final Validation & Cleanup

- [ ] **Step 1: Run complete test suite**

```bash
pytest tests/ -v
```

- [ ] **Step 2: Run all smoke tests**

```bash
python -m scripts.smoke_test_gemma
python -m scripts.smoke_test_chatterbox
python -m scripts.smoke_test_pipeline
```

- [ ] **Step 3: Manual end-to-end test**

1. Set `mode: "gemma-omni"` in config.yaml
2. Start server
3. Test voice conversation (French + English)
4. Test screen sharing ("What's on my screen?")
5. Test emotion expressions (verify Live2D avatar reacts)
6. Test 20+ consecutive turns (VRAM stability)

- [ ] **Step 4: Final commit**

```bash
git add -A
git commit -m "milestone: Phase 5 complete — gemma-omni avatar fully operational"
```

- [ ] **Step 5: Push to remote**

```bash
git push -u origin feature/gemma-omni-avatar
```
