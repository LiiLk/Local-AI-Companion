"""
LLM Provider using llama.cpp server (llama-server).

This provider supports:
- Standard text chat (like Ollama)
- Vision/multimodal models (with images)
- OpenAI-compatible API
- "Thinking" models that provide reasoning_content (like Jan-v2-VL)

Perfect for models like Jan-v2-VL that need separate mmproj files.

llama-server command example:
    llama-server --model Jan-v2-VL-high-Q4_K_M.gguf \
                 --mmproj mmproj-Jan-v2-VL-high.gguf \
                 --host 0.0.0.0 --port 8080 \
                 --ctx-size 8192 --n-gpu-layers 99 \
                 --jinja --no-context-shift
"""

import httpx
import base64
import json
import logging
from pathlib import Path
from typing import AsyncGenerator, Optional, Union

from .base import BaseLLM, Message, LLMResponse

logger = logging.getLogger(__name__)


class LlamaCppProvider(BaseLLM):
    """
    LLM provider using llama.cpp server with OpenAI-compatible API.
    
    This provider supports:
    - Vision models like Jan-v2-VL, LLaVA, Qwen-VL
    - "Thinking" models that use reasoning_content
    - Streaming and non-streaming responses
    
    The server must be started separately with the model loaded.
    
    Attributes:
        base_url: URL of the llama-server (default: http://localhost:8080)
        model_name: Name for logging (doesn't affect the loaded model)
        max_tokens: Maximum tokens to generate (thinking models need more)
        temperature: Sampling temperature (1.0 recommended for Jan-v2-VL)
        top_p: Top-p sampling parameter
        presence_penalty: Presence penalty for repetition control
    """
    
    def __init__(
        self,
        base_url: str = "http://localhost:8080",
        model_name: str = "jan-v2-vl-high",
        timeout: float = 180.0,
        max_tokens: int = 2048,
        temperature: float = 1.0,
        top_p: float = 0.95,
        top_k: int = 20,
        presence_penalty: float = 1.5
    ):
        """
        Initialize the llama.cpp server client.
        
        Args:
            base_url: URL where llama-server is running
            model_name: Display name for the model
            timeout: Request timeout in seconds (thinking models need longer)
            max_tokens: Maximum tokens to generate (Jan-v2-VL needs 1000+ for thinking)
            temperature: Sampling temperature (1.0 recommended for Jan-v2-VL)
            top_p: Top-p sampling
            top_k: Top-k sampling
            presence_penalty: Repetition penalty (1.5 recommended for Jan-v2-VL)
        """
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.presence_penalty = presence_penalty
        self._client = httpx.AsyncClient(timeout=timeout)
    
    def _format_messages(
        self, 
        messages: list[Message],
        images: Optional[list[Union[str, Path]]] = None
    ) -> list[dict]:
        """
        Format messages for OpenAI-compatible API.
        
        For vision models, images are embedded in the last user message.
        
        Args:
            messages: Conversation history
            images: Optional list of image paths or base64 strings
            
        Returns:
            Formatted messages for the API
        """
        formatted = []
        
        for i, msg in enumerate(messages):
            if msg.role == "user" and images and i == len(messages) - 1:
                # Last user message with images (multimodal)
                content = []
                
                # Add text first
                content.append({
                    "type": "text",
                    "text": msg.content
                })
                
                # Add images
                for img in images:
                    if isinstance(img, Path) or (isinstance(img, str) and not img.startswith("data:")):
                        # File path - read and encode
                        img_path = Path(img)
                        if img_path.exists():
                            with open(img_path, "rb") as f:
                                img_data = base64.b64encode(f.read()).decode("utf-8")
                            
                            # Determine MIME type
                            suffix = img_path.suffix.lower()
                            mime_types = {
                                ".jpg": "image/jpeg",
                                ".jpeg": "image/jpeg", 
                                ".png": "image/png",
                                ".gif": "image/gif",
                                ".webp": "image/webp"
                            }
                            mime_type = mime_types.get(suffix, "image/jpeg")
                            
                            content.append({
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{mime_type};base64,{img_data}"
                                }
                            })
                    else:
                        # Already base64 or data URL
                        content.append({
                            "type": "image_url",
                            "image_url": {
                                "url": img if img.startswith("data:") else f"data:image/jpeg;base64,{img}"
                            }
                        })
                
                formatted.append({
                    "role": msg.role,
                    "content": content
                })
            else:
                # Regular text message
                formatted.append({
                    "role": msg.role,
                    "content": msg.content
                })
        
        return formatted
    
    async def chat(
        self, 
        messages: list[Message],
        images: Optional[list[Union[str, Path]]] = None
    ) -> LLMResponse:
        """
        Send a chat request to llama-server.
        
        Uses OpenAI-compatible /v1/chat/completions endpoint.
        Handles "thinking" models that return reasoning_content.
        
        Args:
            messages: Conversation history
            images: Optional images for vision models
            
        Returns:
            LLMResponse with the generated text (content, not reasoning)
        """
        formatted_messages = self._format_messages(messages, images)
        
        response = await self._client.post(
            f"{self.base_url}/v1/chat/completions",
            json={
                "model": self.model_name,
                "messages": formatted_messages,
                "stream": False,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "top_k": self.top_k,
                "presence_penalty": self.presence_penalty,
            }
        )
        response.raise_for_status()
        
        data = response.json()
        message = data["choices"][0]["message"]
        
        # Handle "thinking" models (Jan-v2-VL, DeepSeek, etc.)
        # They put reasoning in reasoning_content and final answer in content
        content = message.get("content", "")
        reasoning = message.get("reasoning_content", "")
        
        # If content is empty but reasoning exists, the model might still be thinking
        # This shouldn't happen with enough max_tokens, but log a warning
        if not content and reasoning:
            logger.warning(
                f"Model returned reasoning but no content. "
                f"Consider increasing max_tokens (current: {self.max_tokens})"
            )
            # Fall back to using reasoning as content
            content = reasoning
        
        # Log reasoning for debugging (can be useful)
        if reasoning:
            logger.debug(f"Model reasoning: {reasoning[:200]}...")
        
        return LLMResponse(
            content=content,
            model=self.model_name
        )
    
    async def chat_stream(
        self, 
        messages: list[Message],
        images: Optional[list[Union[str, Path]]] = None
    ) -> AsyncGenerator[str, None]:
        """
        Send a streaming chat request to llama-server.
        
        Yields tokens as they are generated for real-time display.
        For "thinking" models, only yields the final content, not reasoning.
        
        Note: Streaming with thinking models will include both reasoning
        and content tokens mixed together. For clean separation, use
        non-streaming chat() method instead.
        
        Args:
            messages: Conversation history
            images: Optional images for vision models
            
        Yields:
            Text chunks as they are generated
        """
        formatted_messages = self._format_messages(messages, images)
        
        # Track if we're in thinking mode (between <think> and </think>)
        in_thinking = False
        buffer = ""
        
        async with self._client.stream(
            "POST",
            f"{self.base_url}/v1/chat/completions",
            json={
                "model": self.model_name,
                "messages": formatted_messages,
                "stream": True,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "top_k": self.top_k,
                "presence_penalty": self.presence_penalty,
            }
        ) as response:
            response.raise_for_status()
            
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data_str = line[6:]  # Remove "data: " prefix
                    
                    if data_str.strip() == "[DONE]":
                        break
                    
                    try:
                        data = json.loads(data_str)
                        if "choices" in data and len(data["choices"]) > 0:
                            delta = data["choices"][0].get("delta", {})
                            
                            # Check for reasoning_content (thinking models)
                            reasoning = delta.get("reasoning_content", "")
                            content = delta.get("content", "")
                            
                            # For thinking models, skip reasoning tokens
                            # Only yield content tokens
                            if content:
                                # Filter out <think> tags if they appear in content
                                buffer += content
                                
                                # Check for thinking tags
                                if "<think>" in buffer:
                                    in_thinking = True
                                    # Yield any content before <think>
                                    before_think = buffer.split("<think>")[0]
                                    if before_think:
                                        yield before_think
                                    buffer = buffer.split("<think>")[-1]
                                
                                if "</think>" in buffer:
                                    in_thinking = False
                                    # Skip everything before </think>
                                    buffer = buffer.split("</think>")[-1]
                                
                                # If not in thinking mode, yield the buffer
                                if not in_thinking and buffer:
                                    yield buffer
                                    buffer = ""
                                elif in_thinking:
                                    # In thinking mode, don't yield, just clear buffer
                                    buffer = ""
                            
                    except json.JSONDecodeError:
                        continue
        
        # Yield any remaining buffer
        if buffer and not in_thinking:
            yield buffer
    
    async def health_check(self) -> bool:
        """
        Check if the llama-server is running and healthy.
        
        Returns:
            True if server is responsive, False otherwise
        """
        try:
            response = await self._client.get(f"{self.base_url}/health")
            return response.status_code == 200
        except Exception:
            return False
    
    async def close(self):
        """Close the HTTP client connection."""
        await self._client.aclose()


def create_llamacpp_llm(
    base_url: str = "http://localhost:8080",
    model_name: str = "jan-v2-vl-high",
    max_tokens: int = 2048,
    temperature: float = 1.0,
    top_p: float = 0.95,
    top_k: int = 20,
    presence_penalty: float = 1.5
) -> LlamaCppProvider:
    """
    Factory function to create a LlamaCppProvider.
    
    Default parameters are optimized for Jan-v2-VL thinking model.
    
    Args:
        base_url: URL of the llama-server
        model_name: Display name for the model
        max_tokens: Max tokens (2048+ recommended for thinking models)
        temperature: Sampling temperature (1.0 for Jan-v2-VL)
        top_p: Top-p sampling (0.95 for Jan-v2-VL)
        top_k: Top-k sampling
        presence_penalty: Repetition penalty (1.5 for Jan-v2-VL)
        
    Returns:
        Configured LlamaCppProvider instance
    """
    return LlamaCppProvider(
        base_url=base_url,
        model_name=model_name,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        presence_penalty=presence_penalty
    )
