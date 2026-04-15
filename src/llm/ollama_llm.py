"""
LLM implementation using Ollama.
"""

from __future__ import annotations

import json
from typing import Any, AsyncGenerator

import httpx

from .base import BaseLLM, LLMResponse, Message


class OllamaLLM(BaseLLM):
    """
    Async client for Ollama's ``/api/chat`` endpoint.

    The desktop runtime uses this wrapper for the stable local text path.
    """

    def __init__(
        self,
        model: str = "llama3.2:3b",
        base_url: str = "http://localhost:11434",
        think: bool | None = None,
        options: dict[str, Any] | None = None,
        request_timeout_sec: float = 180.0,
        preload_timeout_sec: float = 120.0,
    ):
        self.model = model
        self.base_url = base_url
        self.think = think
        self.options = dict(options or {})
        self.request_timeout_sec = float(request_timeout_sec)
        self.preload_timeout_sec = float(preload_timeout_sec)
        self.degraded_reason: str | None = None
        self._timeout = httpx.Timeout(
            connect=10.0,
            read=self.request_timeout_sec,
            write=30.0,
            pool=10.0,
        )
        self._client = httpx.AsyncClient(base_url=base_url, timeout=self._timeout)

    def _format_messages(self, messages: list[Message]) -> list[dict[str, str]]:
        return [{"role": m.role, "content": m.content} for m in messages]

    def _build_payload(self, messages: list[Message], stream: bool) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": self._format_messages(messages),
            "stream": stream,
        }
        if self.think is not None:
            payload["think"] = self.think
        if self.options:
            payload["options"] = self.options
        return payload

    def _should_retry_without_think(self, error_text: str) -> bool:
        lowered = (error_text or "").lower()
        return self.think is not None and "think" in lowered

    @staticmethod
    def _error_text(exc: BaseException) -> str:
        message = str(exc).strip()
        if message:
            return message
        return exc.__class__.__name__

    def preload(self) -> None:
        payload = self._build_payload([Message(role="user", content="Say ready.")], stream=False)
        payload["options"] = {
            **self.options,
            "num_predict": 1,
        }
        try:
            with httpx.Client(
                base_url=self.base_url,
                timeout=httpx.Timeout(
                    connect=10.0,
                    read=self.preload_timeout_sec,
                    write=30.0,
                    pool=10.0,
                ),
            ) as client:
                response = client.post("/api/chat", json=payload)
                if response.is_error and self._should_retry_without_think(response.text):
                    retry_payload = {key: value for key, value in payload.items() if key != "think"}
                    self.degraded_reason = "Ollama daemon rejected the think parameter; retried without it."
                    response = client.post("/api/chat", json=retry_payload)
                response.raise_for_status()
        except Exception as exc:
            raise RuntimeError(
                f"Ollama preload failed for model {self.model}: {self._error_text(exc)}"
            ) from exc

    async def _post_chat(self, payload: dict[str, Any]) -> httpx.Response:
        response = await self._client.post("/api/chat", json=payload)
        if not response.is_error:
            return response

        error_text = response.text
        if self._should_retry_without_think(error_text):
            retry_payload = {key: value for key, value in payload.items() if key != "think"}
            self.degraded_reason = "Ollama daemon rejected the think parameter; retried without it."
            response = await self._client.post("/api/chat", json=retry_payload)
            if not response.is_error:
                return response

        response.raise_for_status()
        return response

    async def chat(self, messages: list[Message]) -> LLMResponse:
        payload = self._build_payload(messages, stream=False)
        response = await self._post_chat(payload)
        response.raise_for_status()

        data = response.json()
        return LLMResponse(content=data["message"]["content"], model=data["model"])

    async def chat_stream(self, messages: list[Message]) -> AsyncGenerator[str, None]:
        payload = self._build_payload(messages, stream=True)

        async def _stream_once(stream_payload: dict[str, Any]):
            async with self._client.stream("POST", "/api/chat", json=stream_payload) as response:
                if response.is_error:
                    error_text = (await response.aread()).decode("utf-8", errors="replace")
                    raise httpx.HTTPStatusError(
                        f"ollama stream failed: {error_text or response.status_code}",
                        request=response.request,
                        response=response,
                    ) from RuntimeError(error_text)

                async for line in response.aiter_lines():
                    if not line:
                        continue
                    data = json.loads(line)
                    message = data.get("message") or {}
                    chunk = message.get("content")
                    if chunk:
                        yield chunk

        try:
            async for chunk in _stream_once(payload):
                yield chunk
            return
        except httpx.HTTPStatusError as exc:
            error_text = ""
            if exc.__cause__:
                error_text = str(exc.__cause__)
            if not self._should_retry_without_think(error_text):
                raise

        retry_payload = {key: value for key, value in payload.items() if key != "think"}
        self.degraded_reason = "Ollama daemon rejected the think parameter; retried without it."
        async for chunk in _stream_once(retry_payload):
            yield chunk

    async def close(self):
        await self._client.aclose()
