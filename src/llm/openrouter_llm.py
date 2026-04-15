"""
LLM implementation using OpenRouter's OpenAI-compatible chat API.
"""

from __future__ import annotations

import json
import os
from typing import Any, AsyncGenerator

import httpx

from .base import BaseLLM, LLMResponse, Message


class OpenRouterLLM(BaseLLM):
    """Async client for OpenRouter's ``/api/v1/chat/completions`` endpoint."""

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        api_key_env: str = "OPENROUTER_API_KEY",
        base_url: str = "https://openrouter.ai/api/v1",
        app_url: str | None = None,
        app_title: str | None = None,
        options: dict[str, Any] | None = None,
        required_input_modalities: list[str] | None = None,
        request_timeout_sec: float = 180.0,
        preload_timeout_sec: float = 60.0,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or os.getenv(api_key_env)
        self.api_key_env = api_key_env
        self.app_url = app_url
        self.app_title = app_title
        self.options = dict(options or {})
        self.required_input_modalities = [
            str(value).strip().lower()
            for value in (required_input_modalities or [])
            if str(value).strip()
        ]
        self.request_timeout_sec = float(request_timeout_sec)
        self.preload_timeout_sec = float(preload_timeout_sec)
        self.degraded_reason: str | None = None

        if not self.api_key:
            raise RuntimeError(
                f"OpenRouter API key is missing. Set {api_key_env} or provide api_key in config."
            )

        self._timeout = httpx.Timeout(
            connect=10.0,
            read=self.request_timeout_sec,
            write=30.0,
            pool=10.0,
        )
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=self._timeout,
            headers=self._build_headers(),
        )

    def _build_headers(self) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.app_url:
            headers["HTTP-Referer"] = self.app_url
        if self.app_title:
            headers["X-Title"] = self.app_title
        return headers

    def _format_messages(self, messages: list[Message]) -> list[dict[str, str]]:
        return [{"role": m.role, "content": m.content} for m in messages]

    def _build_payload(self, messages: list[Message], stream: bool) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": self._format_messages(messages),
            "stream": stream,
        }
        payload.update(self.options)
        return payload

    @staticmethod
    def _error_text(exc: BaseException) -> str:
        message = str(exc).strip()
        if message:
            return message
        return exc.__class__.__name__

    def _model_endpoints_path(self) -> str:
        return f"/models/{self.model}/endpoints"

    @staticmethod
    def _extract_input_modalities(payload: dict[str, Any]) -> set[str]:
        data = payload.get("data") or []
        if isinstance(data, dict):
            data = [data]
        modalities: set[str] = set()
        for endpoint in data:
            if not isinstance(endpoint, dict):
                continue
            architecture = endpoint.get("architecture") or {}
            for modality in architecture.get("input_modalities") or []:
                value = str(modality).strip().lower()
                if value:
                    modalities.add(value)
        return modalities

    def _validate_required_modalities(self, payload: dict[str, Any]) -> None:
        if not self.required_input_modalities:
            return

        available = self._extract_input_modalities(payload)
        missing = sorted(set(self.required_input_modalities) - available)
        if missing:
            raise RuntimeError(
                "OpenRouter model "
                f"{self.model} is missing required input modalities: {', '.join(missing)}. "
                f"Available: {', '.join(sorted(available)) or 'unknown'}."
            )

    def preload(self) -> None:
        payload = self._build_payload([Message(role="user", content="Say ready.")], stream=False)
        payload["max_completion_tokens"] = int(self.options.get("max_completion_tokens", 8) or 8)
        try:
            with httpx.Client(
                base_url=self.base_url,
                timeout=httpx.Timeout(
                    connect=10.0,
                    read=self.preload_timeout_sec,
                    write=30.0,
                    pool=10.0,
                ),
                headers=self._build_headers(),
            ) as client:
                if self.required_input_modalities:
                    metadata_response = client.get(self._model_endpoints_path())
                    metadata_response.raise_for_status()
                    self._validate_required_modalities(metadata_response.json())
                response = client.post("/chat/completions", json=payload)
                response.raise_for_status()
        except Exception as exc:
            raise RuntimeError(
                f"OpenRouter preload failed for model {self.model}: {self._error_text(exc)}"
            ) from exc

    async def chat(self, messages: list[Message]) -> LLMResponse:
        payload = self._build_payload(messages, stream=False)
        response = await self._client.post("/chat/completions", json=payload)
        response.raise_for_status()
        data = response.json()
        content = ""
        if data.get("choices"):
            content = data["choices"][0].get("message", {}).get("content", "") or ""
        return LLMResponse(content=content, model=data.get("model", self.model))

    async def chat_stream(self, messages: list[Message]) -> AsyncGenerator[str, None]:
        payload = self._build_payload(messages, stream=True)
        async with self._client.stream("POST", "/chat/completions", json=payload) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line:
                    continue
                if line.startswith(":"):
                    continue
                if not line.startswith("data:"):
                    continue

                data = line[5:].strip()
                if not data or data == "[DONE]":
                    continue

                event = json.loads(data)
                choices = event.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                chunk = delta.get("content")
                if chunk:
                    yield chunk

    async def close(self):
        await self._client.aclose()
