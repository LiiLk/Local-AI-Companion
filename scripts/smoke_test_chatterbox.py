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
