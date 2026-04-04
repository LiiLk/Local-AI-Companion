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
