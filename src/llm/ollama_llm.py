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
        keep_alive: str | int | None = None,
        request_timeout_sec: float = 180.0,
        preload_timeout_sec: float = 120.0,
    ):
        self.model = model
        self.base_url = base_url
        self.think = think
        self.options = dict(options or {})
        self.keep_alive = keep_alive
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

    def _resolve_think(self, options_override: dict[str, Any] | None) -> bool | None:
        """Map a per-request override onto Ollama's ``think`` flag.

        ``reasoning.effort == "none"`` disables thinking, any other effort
        enables it. An explicit ``think`` key wins when present. The token cap
        from ``max_completion_tokens`` is handled in ``_build_payload`` as
        Ollama's ``options.num_predict``.
        """
        think = self.think
        if not options_override:
            return think
        if "think" in options_override:
            return options_override["think"]
        effort = (options_override.get("reasoning") or {}).get("effort")
        if effort is not None:
            return str(effort).lower() != "none"
        return think

    def _build_payload(
        self,
        messages: list[Message],
        stream: bool,
        options_override: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": self._format_messages(messages),
            "stream": stream,
        }
        think = self._resolve_think(options_override)
        if think is not None:
            payload["think"] = think
        options = dict(self.options)
        max_tokens = (options_override or {}).get("max_completion_tokens")
        if max_tokens is not None:
            try:
                options["num_predict"] = int(max_tokens)
            except (TypeError, ValueError):
                pass
        if options:
            payload["options"] = options
        if self.keep_alive is not None:
            payload["keep_alive"] = self.keep_alive
        return payload

    def _should_retry_without_think(
        self,
        error_text: str,
        has_think: bool | None = None,
    ) -> bool:
        if has_think is None:
            has_think = self.think is not None
        lowered = (error_text or "").lower()
        return has_think and "think" in lowered

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
        if self._should_retry_without_think(error_text, payload.get("think") is not None):
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

    async def chat_stream(
        self,
        messages: list[Message],
        options_override: dict[str, Any] | None = None,
    ) -> AsyncGenerator[str, None]:
        payload = self._build_payload(messages, stream=True, options_override=options_override)
        has_think = payload.get("think") is not None

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
            if not self._should_retry_without_think(error_text, has_think):
                raise

        retry_payload = {key: value for key, value in payload.items() if key != "think"}
        self.degraded_reason = "Ollama daemon rejected the think parameter; retried without it."
        async for chunk in _stream_once(retry_payload):
            yield chunk

    async def close(self):
        await self._client.aclose()
