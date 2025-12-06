"""
Base module for LLM (Large Language Models).

This file defines the INTERFACE that all LLMs must implement.
It's like a "contract": any class that inherits from BaseLLM
MUST implement the methods defined here.

Why is this useful?
- You can switch LLMs (Ollama → OpenAI) without modifying the rest of the code
- The main code uses BaseLLM, not a specific implementation
- This is the "Dependency Inversion" pattern (SOLID principles)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import AsyncGenerator


@dataclass
class Message:
    """
    Represents a message in a conversation.
    
    Attributes:
        role: "user", "assistant", or "system"
        content: The message content
    """
    role: str  # "user", "assistant", "system"
    content: str


@dataclass 
class LLMResponse:
    """
    Represents the LLM response.
    
    Attributes:
        content: The response text
        model: The name of the model used
    """
    content: str
    model: str


class BaseLLM(ABC):
    """
    Abstract base class for all LLMs.
    
    ABC = Abstract Base Class
    Methods with @abstractmethod MUST be implemented
    by child classes.
    
    Example:
        class OllamaLLM(BaseLLM):
            def chat(self, messages):
                # Ollama-specific implementation
                ...
    """
    
    @abstractmethod
    async def chat(self, messages: list[Message]) -> LLMResponse:
        """
        Send messages to the LLM and get a response.
        
        Args:
            messages: List of messages (conversation history)
            
        Returns:
            LLMResponse with the response content
            
        Note:
            'async' means this function is asynchronous.
            Use 'await' to call it. This prevents blocking
            the program while the LLM is thinking.
        """
        pass
    
    @abstractmethod
    async def chat_stream(self, messages: list[Message]) -> AsyncGenerator[str, None]:
        """
        Like chat(), but sends the response word by word (streaming).
        
        More responsive: the user sees the response being typed
        instead of waiting for everything to be generated.
        
        Yields:
            Text chunks progressively
        """
        pass
