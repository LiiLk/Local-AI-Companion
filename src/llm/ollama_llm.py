"""
LLM implementation using Ollama.

Ollama allows running LLMs locally on your machine.
This class implements the BaseLLM interface for Ollama.
"""

import httpx
from typing import AsyncGenerator

from .base import BaseLLM, Message, LLMResponse


class OllamaLLM(BaseLLM):
    """
    Client for Ollama.
    
    Ollama exposes an HTTP API on localhost:11434.
    We use httpx (like requests, but async) to communicate.
    
    Attributes:
        model: Model name (e.g., "llama3.2:3b")
        base_url: Ollama API URL
    """
    
    def __init__(
        self, 
        model: str = "llama3.2:3b",
        base_url: str = "http://localhost:11434"
    ):
        """
        Initialize the Ollama client.
        
        Args:
            model: Model to use (must be already downloaded with 'ollama pull')
            base_url: URL where Ollama is listening (default localhost:11434)
        """
        self.model = model
        self.base_url = base_url
        # httpx.AsyncClient is like requests.Session but async
        # We keep a session open to reuse connections
        self._client = httpx.AsyncClient(base_url=base_url, timeout=60.0)
    
    def _format_messages(self, messages: list[Message]) -> list[dict]:
        """
        Convert our Message objects to format expected by Ollama.
        
        Ollama expects: [{"role": "user", "content": "..."}]
        """
        return [{"role": m.role, "content": m.content} for m in messages]
    
    async def chat(self, messages: list[Message]) -> LLMResponse:
        """
        Send a chat request to Ollama.
        
        Endpoint: POST /api/chat
        
        Args:
            messages: Conversation history
            
        Returns:
            The complete LLM response
        """
        print(f"📤 Sending request to Ollama ({self.model})...")
        response = await self._client.post(
            "/api/chat",
            json={
                "model": self.model,
                "messages": self._format_messages(messages),
                "stream": False,  # We want the complete response at once
            }
        )
        response.raise_for_status()  # Raise error if status != 200
        
        data = response.json()
        return LLMResponse(
            content=data["message"]["content"],
            model=data["model"]
        )
    
    async def chat_stream(self, messages: list[Message]) -> AsyncGenerator[str, None]:
        """
        Send a chat request with streaming.
        
        Streaming allows displaying the response as it is generated,
        like in ChatGPT.
        
        Yields:
            Text chunks (tokens) one by one
        """
        print(f"📤 Streaming request to Ollama ({self.model})...")
        async with self._client.stream(
            "POST",
            "/api/chat",
            json={
                "model": self.model,
                "messages": self._format_messages(messages),
                "stream": True,  # Enable streaming
            }
        ) as response:
            response.raise_for_status()
            
            # Ollama sends JSON lines, one per token
            async for line in response.aiter_lines():
                if line:
                    import json
                    data = json.loads(line)
                    # Each line contains a piece of the message
                    if "message" in data and "content" in data["message"]:
                        yield data["message"]["content"]
    
    async def close(self):
        """Properly close the HTTP connection."""
        await self._client.aclose()
