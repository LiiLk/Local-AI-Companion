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
    print("\n[1/1] Full pipeline: audio -> Gemma -> Chatterbox -> audio")
    t0 = time.time()
    response = await pipeline.process_speech(test_audio)
    elapsed = time.time() - t0

    print(f"\n  Total time: {elapsed:.1f}s")
    print(f"  Response: {response.encode('ascii', 'replace').decode()}")
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
