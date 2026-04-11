import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from src.omni.gemma_provider import GemmaProvider


class FakeStreamer:
    def __iter__(self):
        for token in ("hello", " ", "world"):
            time.sleep(0.05)
            yield token


def _build_test_provider() -> GemmaProvider:
    provider = GemmaProvider.__new__(GemmaProvider)
    provider._executor = ThreadPoolExecutor(max_workers=1)
    provider._build_messages = lambda text, history, audio, images: ([], None, None)

    def fake_generate(messages, audio_inputs, image_inputs, stream):
        thread = threading.Thread(target=lambda: None, daemon=True)
        thread.start()
        return FakeStreamer(), thread

    provider._generate = fake_generate
    return provider


def test_gemma_chat_stream_does_not_block_event_loop():
    provider = _build_test_provider()

    async def scenario():
        started = time.perf_counter()
        tokens = []

        async def consume():
            async for token in provider.chat_stream(text="bonjour"):
                tokens.append(token)

        async def ticker():
            for _ in range(5):
                await asyncio.sleep(0.02)

        await asyncio.gather(consume(), ticker())
        return time.perf_counter() - started, "".join(tokens)

    elapsed, text = asyncio.run(scenario())
    provider._executor.shutdown(wait=False)

    assert text == "hello world"
    assert elapsed < 0.22
