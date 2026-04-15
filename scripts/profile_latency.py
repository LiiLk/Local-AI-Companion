"""Profile latency: audio input vs text-only for Gemma E2B."""
import os, time, torch, asyncio
torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
os.environ["PATH"] = torch_lib + os.pathsep + os.environ.get("PATH", "")

import numpy as np
from src.omni.gemma_provider import GemmaProvider

P = lambda msg: print(msg, flush=True)

async def main():
    gemma = GemmaProvider()
    gemma.preload()

    # 2s audio
    sr = 16000
    t = np.linspace(0, 2, sr * 2, dtype=np.float32)
    audio_bytes = (np.sin(2 * np.pi * 440 * t) * 16000).astype(np.int16).tobytes()

    P("--- AUDIO INPUT (what we use now) ---")
    # Cold
    t0 = time.perf_counter()
    resp = await gemma.chat("Reponds OK.", audio=audio_bytes)
    P(f"  Cold call:  {(time.perf_counter()-t0)*1000:.0f} ms")

    # Warm x3
    for i in range(3):
        t0 = time.perf_counter()
        resp = await gemma.chat("Reponds OK.", audio=audio_bytes)
        P(f"  Warm {i+1}:    {(time.perf_counter()-t0)*1000:.0f} ms")

    # Streaming first token
    t0 = time.perf_counter()
    first_token_time = None
    tokens = []
    async for tok in gemma.chat_stream("Reponds brievement.", audio=audio_bytes):
        if first_token_time is None:
            first_token_time = time.perf_counter() - t0
        tokens.append(tok)
    total_stream = time.perf_counter() - t0
    P(f"  Stream first token: {first_token_time*1000:.0f} ms, total: {total_stream*1000:.0f} ms, {len(tokens)} tokens")

    P("")
    P("--- TEXT ONLY (no audio processing) ---")
    for i in range(3):
        t0 = time.perf_counter()
        resp = await gemma.chat("Reponds OK en francais.")
        P(f"  Run {i+1}:    {(time.perf_counter()-t0)*1000:.0f} ms")

    # Streaming first token
    t0 = time.perf_counter()
    first_token_time = None
    tokens = []
    async for tok in gemma.chat_stream("Reponds brievement en francais."):
        if first_token_time is None:
            first_token_time = time.perf_counter() - t0
        tokens.append(tok)
    total_stream = time.perf_counter() - t0
    P(f"  Stream first token: {first_token_time*1000:.0f} ms, total: {total_stream*1000:.0f} ms, {len(tokens)} tokens")

    gemma.cleanup()

asyncio.run(main())
