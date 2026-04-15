import json

import httpx
import pytest

from src.llm.base import Message
from src.llm.openrouter_llm import OpenRouterLLM


async def _make_llm(handler):
    llm = OpenRouterLLM(
        model="x-ai/grok-4.1-fast",
        api_key="test-key",
        base_url="https://openrouter.test/api/v1",
        app_url="http://localhost",
        app_title="Local-AI-Companion",
        options={"temperature": 0.6, "max_completion_tokens": 96},
    )
    await llm._client.aclose()
    llm._client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url="https://openrouter.test/api/v1",
        timeout=60.0,
        headers=llm._build_headers(),
    )
    return llm


@pytest.mark.asyncio
async def test_openrouter_chat_sends_openai_compatible_payload():
    seen_payloads = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_payloads.append(json.loads(request.content.decode("utf-8")))
        assert request.headers["Authorization"] == "Bearer test-key"
        assert request.headers["HTTP-Referer"] == "http://localhost"
        assert request.headers["X-Title"] == "Local-AI-Companion"
        return httpx.Response(
            200,
            json={
                "model": "x-ai/grok-4.1-fast",
                "choices": [{"message": {"content": "ok"}}],
            },
        )

    llm = await _make_llm(handler)
    try:
        response = await llm.chat([Message(role="user", content="Hello")])
    finally:
        await llm.close()

    assert response.content == "ok"
    assert seen_payloads == [
        {
            "model": "x-ai/grok-4.1-fast",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": False,
            "temperature": 0.6,
            "max_completion_tokens": 96,
        }
    ]


@pytest.mark.asyncio
async def test_openrouter_stream_ignores_comments_and_reasoning_chunks():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=(
                b": OPENROUTER PROCESSING\n\n"
                b'data: {"choices":[{"delta":{"reasoning":"internal"}}]}\n\n'
                b'data: {"choices":[{"delta":{"content":"Hello"}}]}\n\n'
                b'data: {"choices":[{"delta":{"content":" world"}}]}\n\n'
                b"data: [DONE]\n\n"
            ),
        )

    llm = await _make_llm(handler)
    try:
        chunks = [chunk async for chunk in llm.chat_stream([Message(role="user", content="Hi")])]
    finally:
        await llm.close()

    assert chunks == ["Hello", " world"]


def test_openrouter_error_text_falls_back_to_exception_type():
    assert OpenRouterLLM._error_text(httpx.ReadTimeout("")) == "ReadTimeout"


def test_openrouter_requires_api_key():
    with pytest.raises(RuntimeError, match="OpenRouter API key is missing"):
        OpenRouterLLM(model="x-ai/grok-4.1-fast", api_key=None, api_key_env="MISSING_TEST_KEY")


def test_openrouter_extracts_modalities_from_model_metadata():
    payload = {
        "data": [
            {"architecture": {"input_modalities": ["text", "image"]}},
            {"architecture": {"input_modalities": ["text"]}},
        ]
    }

    assert OpenRouterLLM._extract_input_modalities(payload) == {"text", "image"}


def test_openrouter_extracts_modalities_from_single_model_payload():
    payload = {
        "data": {
            "id": "x-ai/grok-4.1-fast",
            "architecture": {"input_modalities": ["text", "image", "file"]},
        }
    }

    assert OpenRouterLLM._extract_input_modalities(payload) == {"text", "image", "file"}


def test_openrouter_validate_required_modalities_accepts_vision_model():
    llm = OpenRouterLLM(
        model="x-ai/grok-4.1-fast",
        api_key="test-key",
        required_input_modalities=["image"],
    )
    llm._validate_required_modalities(
        {"data": [{"architecture": {"input_modalities": ["text", "image"]}}]}
    )


def test_openrouter_validate_required_modalities_rejects_text_only_model():
    llm = OpenRouterLLM(
        model="x-ai/grok-4.1-fast",
        api_key="test-key",
        required_input_modalities=["image"],
    )

    with pytest.raises(RuntimeError, match="missing required input modalities: image"):
        llm._validate_required_modalities(
            {"data": [{"architecture": {"input_modalities": ["text"]}}]}
        )
