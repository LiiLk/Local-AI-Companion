import json

import httpx
import pytest

from src.llm.base import Message
from src.llm.ollama_llm import OllamaLLM


async def _make_llm(handler, *, think=False, options=None):
    llm = OllamaLLM(
        model="qwen3.5:4b",
        base_url="http://test-ollama",
        think=think,
        options=options,
    )
    await llm._client.aclose()
    llm._client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url="http://test-ollama",
        timeout=60.0,
    )
    return llm


@pytest.mark.asyncio
async def test_ollama_chat_sends_think_and_options():
    seen_payloads = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_payloads.append(json.loads(request.content.decode("utf-8")))
        return httpx.Response(
            200,
            json={"model": "qwen3.5:4b", "message": {"content": "ok"}},
        )

    llm = await _make_llm(handler, think=False, options={"temperature": 0.6, "top_p": 0.9})
    try:
        response = await llm.chat([Message(role="user", content="Hello")])
    finally:
        await llm.close()

    assert response.content == "ok"
    assert seen_payloads == [
        {
            "model": "qwen3.5:4b",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": False,
            "think": False,
            "options": {"temperature": 0.6, "top_p": 0.9},
        }
    ]


@pytest.mark.asyncio
async def test_ollama_stream_ignores_thinking_chunks():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=(
                b'{"message":{"thinking":"internal"}}\n'
                b'{"message":{"content":"Hello"}}\n'
                b'{"message":{"content":" world"}}\n'
            ),
        )

    llm = await _make_llm(handler, think=False)
    try:
        chunks = [chunk async for chunk in llm.chat_stream([Message(role="user", content="Hi")])]
    finally:
        await llm.close()

    assert chunks == ["Hello", " world"]


@pytest.mark.asyncio
async def test_ollama_chat_retries_without_think_when_daemon_rejects_it():
    seen_payloads = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content.decode("utf-8"))
        seen_payloads.append(payload)
        if "think" in payload:
            return httpx.Response(400, text="unsupported field think")
        return httpx.Response(
            200,
            json={"model": "qwen3.5:4b", "message": {"content": "fallback-ok"}},
        )

    llm = await _make_llm(handler, think=False)
    try:
        response = await llm.chat([Message(role="user", content="Hello")])
    finally:
        await llm.close()

    assert response.content == "fallback-ok"
    assert len(seen_payloads) == 2
    assert "think" in seen_payloads[0]
    assert "think" not in seen_payloads[1]
    assert llm.degraded_reason == "Ollama daemon rejected the think parameter; retried without it."


def test_ollama_error_text_falls_back_to_exception_type():
    assert OllamaLLM._error_text(httpx.ReadTimeout("")) == "ReadTimeout"


@pytest.mark.asyncio
async def test_ollama_stream_retries_without_think_when_daemon_rejects_it():
    seen_payloads = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content.decode("utf-8"))
        seen_payloads.append(payload)
        if "think" in payload:
            return httpx.Response(400, text="unknown think option")
        return httpx.Response(
            200,
            content=b'{"message":{"content":"stream-ok"}}\n',
        )

    llm = await _make_llm(handler, think=False)
    try:
        chunks = [chunk async for chunk in llm.chat_stream([Message(role="user", content="Hello")])]
    finally:
        await llm.close()

    assert chunks == ["stream-ok"]
    assert len(seen_payloads) == 2
    assert "think" in seen_payloads[0]
    assert "think" not in seen_payloads[1]
    assert llm.degraded_reason == "Ollama daemon rejected the think parameter; retried without it."
