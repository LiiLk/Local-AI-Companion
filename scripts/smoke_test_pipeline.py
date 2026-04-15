"""
End-to-end smoke test for the Pipeline mode.

Tests: text → GemmaTextVisionLLM (text-only) → Chatterbox TTS → audio output.
Run: python -m scripts.smoke_test_pipeline

Expected: ~4min first load (conditional_decoder), then 3-7s per response.
"""

import asyncio
import subprocess
import time
from pathlib import Path

import torch

from src.llm.base import Message
from src.llm.gemma_text_vision_llm import GemmaTextVisionLLM
from src.omni.gemma_provider import GemmaProvider
from src.tts.chatterbox_provider import ChatterboxTTSProvider


async def main():
    print("=" * 50)
    print("PIPELINE SMOKE TEST: Gemma text + Chatterbox TTS")
    print("=" * 50)

    # --- Step 1: Load Gemma E2B ---
    print("\n[1/4] Loading Gemma E2B (NF4, text-only)...")
    t0 = time.time()
    gemma = GemmaProvider()
    llm = GemmaTextVisionLLM(gemma=gemma)
    llm.preload()
    print(f"  Loaded in {time.time() - t0:.1f}s")
    print(f"  VRAM: {torch.cuda.memory_allocated() / 1024**2:.0f}MB")

    # --- Step 2: Load Chatterbox TTS ---
    print("\n[2/4] Loading Chatterbox ONNX Q4 (full GPU, ~3min)...")
    t0 = time.time()
    tts = ChatterboxTTSProvider(
        ref_audio_path="resources/voices/march7th/VO_Archive_March_7th_5.wav",
        prefer_full_gpu=True,
    )
    tts._load_model()
    print(f"  Loaded in {time.time() - t0:.1f}s")
    r = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader"],
        capture_output=True, text=True,
    )
    print(f"  Total VRAM: {r.stdout.strip()}")

    # --- Step 3: Text → LLM streaming ---
    print("\n[3/4] Pipeline: text → Gemma streaming...")
    messages = [
        Message(role="system", content="Tu es March 7th. Reponds brievement en francais."),
        Message(role="user", content="Bonjour, comment vas-tu ?"),
    ]

    t0 = time.time()
    full_response = ""
    first_token_time = None
    async for chunk in llm.chat_stream(messages):
        if first_token_time is None:
            first_token_time = time.time() - t0
        full_response += chunk
    gemma_time = time.time() - t0

    print(f"  First token: {first_token_time * 1000:.0f}ms")
    print(f"  Total: {gemma_time:.1f}s")
    print(f"  Response: {full_response[:120]}")
    assert len(full_response) > 5, "Response too short"
    print("  PASS")

    # --- Step 4: LLM output → TTS ---
    print("\n[4/4] Pipeline: text → Chatterbox TTS...")
    output_path = Path("smoke_test_output.wav")

    t0 = time.time()
    result = await tts.synthesize(full_response[:200], output_path)
    tts_time = time.time() - t0

    print(f"  TTS: {tts_time:.1f}s, audio: {result.duration:.1f}s")
    assert result.duration > 0, "No audio generated"
    assert output_path.exists(), "Output file not created"
    print("  PASS")

    # --- Summary ---
    total_latency = gemma_time + tts_time
    print(f"\n{'=' * 50}")
    print("PIPELINE SMOKE TEST PASSED")
    print(f"  Gemma first token: {first_token_time * 1000:.0f}ms")
    print(f"  Gemma total:       {gemma_time:.1f}s")
    print(f"  Chatterbox TTS:    {tts_time:.1f}s")
    print(f"  Total latency:     {total_latency:.1f}s")
    print(f"  VRAM:              {r.stdout.strip()}")
    print(f"{'=' * 50}")

    # Cleanup
    llm.cleanup()
    output_path.unlink(missing_ok=True)


if __name__ == "__main__":
    asyncio.run(main())
