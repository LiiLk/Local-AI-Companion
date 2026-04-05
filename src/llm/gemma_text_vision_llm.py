"""
Gemma text+vision LLM wrapper for the classic pipeline.

This keeps the live architecture simple:
ASR -> Gemma text/vision -> TTS.
Gemma audio input stays available elsewhere as an experimental mode.
"""

from __future__ import annotations

import logging
from typing import AsyncGenerator, Optional

from src.llm.base import BaseLLM, LLMResponse, Message
from src.omni.gemma_provider import GemmaProvider

logger = logging.getLogger(__name__)


class GemmaTextVisionLLM(BaseLLM):
    """Adapt GemmaProvider to the BaseLLM interface used by ConversationPipeline."""

    def __init__(
        self,
        gemma: GemmaProvider,
        screen_config: Optional[dict] = None,
    ):
        self.gemma = gemma
        self.screen_buffer = None
        self._include_screen_in_conversation = False

        if screen_config and screen_config.get("enabled", False):
            self.enable_screen_capture(screen_config)

    def preload(self) -> None:
        self.gemma.preload()

    def cleanup(self) -> None:
        if self.screen_buffer:
            self.screen_buffer.stop()
            self.screen_buffer = None
        self.gemma.cleanup()

    async def close(self) -> None:
        self.cleanup()

    def enable_screen_capture(self, config: dict) -> None:
        from src.vision.screen_buffer import ScreenBuffer

        if self.screen_buffer:
            self.screen_buffer.stop()

        self._include_screen_in_conversation = config.get("include_in_conversation", False)
        self.screen_buffer = ScreenBuffer(
            capture_interval=config.get("interval", 2.0),
            max_buffer=config.get("max_buffer", 30),
            change_threshold=config.get("change_threshold", 0.05),
        )
        self.screen_buffer.start()
        logger.info(
            "GemmaTextVisionLLM screen capture %s",
            "enabled" if self._include_screen_in_conversation else "armed (passive)",
        )

    def _get_screen_context(self) -> list:
        if not self.screen_buffer or not self._include_screen_in_conversation:
            return []
        frame = self.screen_buffer.get_latest()
        return [frame] if frame else []

    def _split_messages(self, messages: list[Message]) -> tuple[str, list[dict]]:
        if not messages:
            return "", []

        latest_user_text = ""
        history: list[dict] = []

        for index, msg in enumerate(messages):
            is_last = index == len(messages) - 1
            if is_last and msg.role == "user":
                latest_user_text = msg.content
                continue

            history.append({
                "role": msg.role,
                "content": [{"type": "text", "text": msg.content}],
            })

        return latest_user_text, history

    async def chat(self, messages: list[Message]) -> LLMResponse:
        text, history = self._split_messages(messages)
        response = await self.gemma.chat(
            text=text,
            history=history,
            images=self._get_screen_context() or None,
        )
        return LLMResponse(content=response, model=self.gemma.model_id)

    async def chat_stream(self, messages: list[Message]) -> AsyncGenerator[str, None]:
        text, history = self._split_messages(messages)
        async for chunk in self.gemma.chat_stream(
            text=text,
            history=history,
            images=self._get_screen_context() or None,
        ):
            yield chunk
