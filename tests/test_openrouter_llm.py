import asyncio
import json
import logging
from types import SimpleNamespace

import httpx
import pytest

from src.llm.base import Message
from src.llm.openrouter_llm import OpenRouterLLM
from src.llm import openrouter_llm as openrouter_module


async def _make_llm(handler):
    llm = OpenRouterLLM(
        model="deepseek/deepseek-v4-flash",
        api_key="test-key",
        base_url="https://openrouter.test/api/v1",
        app_url="http://localhost",
        app_title="Local-AI-Companion",
        options={
            "temperature": 0.6,
            "max_completion_tokens": 512,
            "reasoning": {"effort": "high"},
        },
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
                "model": "deepseek/deepseek-v4-flash",
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
            "model": "deepseek/deepseek-v4-flash",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": False,
            "temperature": 0.6,
            "max_completion_tokens": 512,
            "reasoning": {"effort": "high"},
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


def _call_logs(caplog):
    return [
        record.getMessage()
        for record in caplog.records
        if record.name == "src.llm.openrouter_llm"
        and record.levelno == logging.INFO
        and record.getMessage().startswith("llm_call ")
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "usage, expected_tokens",
    [
        (
            {"prompt_tokens": 120, "completion_tokens": 8,
             "prompt_tokens_details": {"cached_tokens": 64}},
            "prompt_tokens=120 cached_tokens=64 completion_tokens=8",
        ),
        (
            {"prompt_tokens": 0, "completion_tokens": 0,
             "prompt_tokens_details": {"cached_tokens": 0}},
            "prompt_tokens=0 cached_tokens=0 completion_tokens=0",
        ),
        (
            {"prompt_tokens": 120, "completion_tokens": 8},
            "prompt_tokens=120 cached_tokens=na completion_tokens=8",
        ),
    ],
)
async def test_openrouter_call_trace_captures_metadata_and_first_content_time(
    caplog, monkeypatch, usage, expected_tokens,
):
    clock = SimpleNamespace(now=10.0)
    monkeypatch.setattr(
        openrouter_module, "time",
        SimpleNamespace(perf_counter=lambda: clock.now), raising=False,
    )
    caplog.set_level(logging.INFO, logger="src.llm.openrouter_llm")
    seen_payloads = []

    class TimedStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            frames = [
                (10.3, ": OPENROUTER PROCESSING"),
                (10.4, 'data: {"provider":"Alibaba","choices":[{"delta":{"role":"assistant"}}]}'),
                (10.5, 'data: {"choices":[{"delta":{"reasoning":"internal"}}]}'),
                (10.6, 'data: {"provider":null,"usage":null,"choices":[{"delta":{"content":""}}]}'),
                (10.785, 'data: {"choices":[{"delta":{"content":"Hello"}}]}'),
                (11.0, 'data: {"choices":[{"delta":{"content":" world"}}]}'),
                (11.1, "data: " + json.dumps({"choices": [], "usage": usage})),
                (11.25, "data: [DONE]"),
            ]
            for now, frame in frames:
                clock.now = now
                yield (frame + "\n\n").encode()

    def handler(request):
        seen_payloads.append(json.loads(request.content))
        clock.now = 10.25
        return httpx.Response(200, stream=TimedStream())

    llm = await _make_llm(handler)
    try:
        chunks = [chunk async for chunk in llm.chat_stream([Message(role="user", content="Hi")])]
    finally:
        await llm.close()

    assert chunks == ["Hello", " world"]
    assert seen_payloads == [{
        "model": "deepseek/deepseek-v4-flash",
        "messages": [{"role": "user", "content": "Hi"}],
        "stream": True,
        "temperature": 0.6,
        "max_completion_tokens": 512,
        "reasoning": {"effort": "high"},
    }]
    assert _call_logs(caplog) == [
        "llm_call provider=Alibaba model=deepseek/deepseek-v4-flash "
        f"ttft_ms=785.0 total_ms=1250.0 {expected_tokens} status=ok"
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("content", ["Hello", ""])
async def test_openrouter_call_trace_handles_missing_metadata(caplog, monkeypatch, content):
    monkeypatch.setattr(
        openrouter_module, "time", SimpleNamespace(perf_counter=lambda: 10.0), raising=False,
    )
    caplog.set_level(logging.INFO, logger="src.llm.openrouter_llm")

    def handler(request):
        event = {"choices": [{"delta": {"content": content}}]}
        return httpx.Response(200, content="data: " + json.dumps(event) + "\n\ndata: [DONE]\n\n")

    llm = await _make_llm(handler)
    try:
        chunks = [chunk async for chunk in llm.chat_stream([Message(role="user", content="Hi")])]
    finally:
        await llm.close()

    assert chunks == ([content] if content else [])
    ttft = "0.0" if content else "na"
    assert _call_logs(caplog) == [
        "llm_call provider=unknown model=deepseek/deepseek-v4-flash "
        f"ttft_ms={ttft} total_ms=0.0 "
        "prompt_tokens=na cached_tokens=na completion_tokens=na status=ok"
    ]


@pytest.mark.asyncio
async def test_openrouter_call_trace_logs_cancelled_when_consumer_closes(caplog):
    caplog.set_level(logging.INFO, logger="src.llm.openrouter_llm")

    def handler(request):
        return httpx.Response(200, content=(
            b'data: {"provider":"Relace","choices":[{"delta":{"content":"Hello"}}]}\n\n'
            b'data: {"usage":{"prompt_tokens":120,"completion_tokens":8},"choices":[]}\n\n'
            b'data: [DONE]\n\n'
        ))

    llm = await _make_llm(handler)
    stream = llm.chat_stream([Message(role="user", content="Hi")])
    try:
        assert await anext(stream) == "Hello"
        assert _call_logs(caplog) == []
        await stream.aclose()
        await stream.aclose()
    finally:
        await stream.aclose()
        await llm.close()

    logs = _call_logs(caplog)
    assert len(logs) == 1
    assert "provider=Relace " in logs[0]
    assert logs[0].endswith(
        "prompt_tokens=na cached_tokens=na completion_tokens=na status=cancelled"
    )


@pytest.mark.asyncio
async def test_openrouter_call_trace_logs_task_cancellation(caplog):
    caplog.set_level(logging.INFO, logger="src.llm.openrouter_llm")
    reading = asyncio.Event()

    class WaitingStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b'data: {"provider":"Relace","choices":[]}\n\n'
            reading.set()
            await asyncio.Event().wait()

    llm = await _make_llm(lambda request: httpx.Response(200, stream=WaitingStream()))
    stream = llm.chat_stream([Message(role="user", content="Hi")])
    task = asyncio.create_task(anext(stream))
    try:
        await asyncio.wait_for(reading.wait(), timeout=1.0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        task.cancel()
        await stream.aclose()
        await llm.close()

    logs = _call_logs(caplog)
    assert len(logs) == 1
    assert "provider=Relace " in logs[0]
    assert "ttft_ms=na " in logs[0]
    assert logs[0].endswith("status=cancelled")


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["http", "json", "timeout"])
async def test_openrouter_call_trace_preserves_errors(caplog, failure):
    caplog.set_level(logging.INFO, logger="src.llm.openrouter_llm")
    timeout = httpx.ReadTimeout("stream timed out")

    def handler(request):
        if failure == "timeout":
            raise timeout
        if failure == "http":
            return httpx.Response(503)
        return httpx.Response(200, content=b"data: invalid-json\n\n")

    expected_error = {
        "http": httpx.HTTPStatusError, "json": json.JSONDecodeError, "timeout": httpx.ReadTimeout,
    }[failure]
    llm = await _make_llm(handler)
    try:
        with pytest.raises(expected_error) as caught:
            _ = [chunk async for chunk in llm.chat_stream([Message(role="user", content="Hi")])]
    finally:
        await llm.close()

    if failure == "timeout":
        assert caught.value is timeout
    logs = _call_logs(caplog)
    assert len(logs) == 1
    assert "ttft_ms=na " in logs[0]
    assert logs[0].endswith("status=error")


@pytest.mark.asyncio
@pytest.mark.parametrize("first_content", ["", "Hello"])
async def test_openrouter_call_trace_marks_sse_errors_without_changing_chunks(caplog, first_content):
    caplog.set_level(logging.INFO, logger="src.llm.openrouter_llm")

    def handler(request):
        events = [
            {"provider": "Alibaba", "choices": [{"delta": {"content": first_content}}]},
            {"error": {"code": "server_error", "message": "Provider disconnected"},
             "choices": [{"delta": {"content": ""}, "finish_reason": "error"}]},
        ]
        content = "".join("data: " + json.dumps(event) + "\n\n" for event in events)
        return httpx.Response(200, content=content + "data: [DONE]\n\n")

    llm = await _make_llm(handler)
    try:
        chunks = [chunk async for chunk in llm.chat_stream([Message(role="user", content="Hi")])]
    finally:
        await llm.close()

    assert chunks == ([first_content] if first_content else [])
    logs = _call_logs(caplog)
    assert len(logs) == 1
    assert "provider=Alibaba " in logs[0]
    assert logs[0].endswith("status=error")


@pytest.mark.asyncio
async def test_openrouter_call_trace_logs_both_adaptive_reasoning_calls(caplog):
    from src.assistant.reasoning_router import (
        AdaptiveReasoningConfig,
        stream_llm_with_adaptive_reasoning,
    )

    caplog.set_level(logging.INFO, logger="src.llm.openrouter_llm")
    seen_payloads = []

    def handler(request):
        seen_payloads.append(json.loads(request.content))
        if len(seen_payloads) == 1:
            event = {"provider": "Relace", "choices": [{"delta": {"content": "<|THINK|>"}}]}
        else:
            event = {"provider": "Alibaba", "choices": [{"delta": {"content": "Answer."}}]}
        return httpx.Response(200, content="data: " + json.dumps(event) + "\n\ndata: [DONE]\n\n")

    llm = await _make_llm(handler)
    try:
        chunks = [chunk async for chunk in stream_llm_with_adaptive_reasoning(
            llm, [Message(role="user", content="Plan this.")],
            AdaptiveReasoningConfig(enabled=True),
        )]
    finally:
        await llm.close()

    assert chunks == ["Answer."]
    assert [payload["reasoning"]["effort"] for payload in seen_payloads] == ["none", "medium"]
    logs = _call_logs(caplog)
    assert len(logs) == 2
    assert "provider=Relace " in logs[0]
    assert logs[0].endswith("status=cancelled")
    assert "provider=Alibaba " in logs[1]
    assert logs[1].endswith("status=ok")


def test_openrouter_error_text_falls_back_to_exception_type():
    assert OpenRouterLLM._error_text(httpx.ReadTimeout("")) == "ReadTimeout"


def test_openrouter_deep_merge_preserves_key_order():
    base = {
        "model": "m",
        "messages": [],
        "stream": True,
        "temperature": 0.6,
        "provider": {"order": ["a", "b"]},
        "reasoning": {"effort": "high"},
    }
    override = {"reasoning": {"effort": "none"}, "max_completion_tokens": 2048}

    merged = OpenRouterLLM._deep_merge(base, override)

    assert list(merged.keys()) == [
        "model",
        "messages",
        "stream",
        "temperature",
        "provider",
        "reasoning",
        "max_completion_tokens",
    ]
    assert merged["provider"] == {"order": ["a", "b"]}
    assert merged["reasoning"] == {"effort": "none"}
    assert merged["temperature"] == 0.6


@pytest.mark.asyncio
async def test_openrouter_stream_applies_deep_merged_options_override():
    seen_payloads = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_payloads.append(json.loads(request.content.decode("utf-8")))
        return httpx.Response(
            200,
            content=(
                b'data: {"choices":[{"delta":{"content":"ok"}}]}\n\n'
                b"data: [DONE]\n\n"
            ),
        )

    llm = await _make_llm(handler)
    try:
        chunks = [
            chunk
            async for chunk in llm.chat_stream(
                [Message(role="user", content="Hi")],
                options_override={
                    "reasoning": {"effort": "none"},
                    "max_completion_tokens": 2048,
                },
            )
        ]
    finally:
        await llm.close()

    assert chunks == ["ok"]
    payload = seen_payloads[0]
    assert payload["reasoning"] == {"effort": "none"}
    assert payload["max_completion_tokens"] == 2048
    assert payload["temperature"] == 0.6


def test_openrouter_requires_api_key():
    with pytest.raises(RuntimeError, match="OpenRouter API key is missing"):
        OpenRouterLLM(
            model="deepseek/deepseek-v4-flash",
            api_key=None,
            api_key_env="MISSING_TEST_KEY",
        )


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
            "id": "deepseek/deepseek-v4-flash",
            "architecture": {"input_modalities": ["text", "image", "file"]},
        }
    }

    assert OpenRouterLLM._extract_input_modalities(payload) == {"text", "image", "file"}


def test_openrouter_validate_required_modalities_accepts_vision_model():
    llm = OpenRouterLLM(
        model="deepseek/deepseek-v4-flash",
        api_key="test-key",
        required_input_modalities=["image"],
    )
    llm._validate_required_modalities(
        {"data": [{"architecture": {"input_modalities": ["text", "image"]}}]}
    )


def test_openrouter_validate_required_modalities_rejects_text_only_model():
    llm = OpenRouterLLM(
        model="deepseek/deepseek-v4-flash",
        api_key="test-key",
        required_input_modalities=["image"],
    )

    with pytest.raises(RuntimeError, match="missing required input modalities: image"):
        llm._validate_required_modalities(
            {"data": [{"architecture": {"input_modalities": ["text"]}}]}
        )
