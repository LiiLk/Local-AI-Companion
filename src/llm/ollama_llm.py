"""
Implémentation du LLM utilisant Ollama.

Ollama permet de faire tourner des LLM en local sur ta machine.
Cette classe "implémente" l'interface BaseLLM pour Ollama.
"""

import httpx
from typing import AsyncGenerator

from .base import BaseLLM, Message, LLMResponse


class OllamaLLM(BaseLLM):
    """
    Client pour Ollama.
    
    Ollama expose une API HTTP sur localhost:11434.
    On utilise httpx (comme requests, mais async) pour communiquer.
    
    Attributes:
        model: Nom du modèle (ex: "llama3.2:3b")
        base_url: URL de l'API Ollama
    """
    
    def __init__(
        self, 
        model: str = "llama3.2:3b",
        base_url: str = "http://localhost:11434"
    ):
        """
        Initialise le client Ollama.
        
        Args:
            model: Le modèle à utiliser (doit être déjà téléchargé avec 'ollama pull')
            base_url: URL où Ollama écoute (par défaut localhost:11434)
        """
        self.model = model
        self.base_url = base_url
        # httpx.AsyncClient est comme requests.Session mais async
        # On garde une session ouverte pour réutiliser les connexions
        self._client = httpx.AsyncClient(base_url=base_url, timeout=60.0)
    
    def _format_messages(self, messages: list[Message]) -> list[dict]:
        """
        Convertit nos objets Message en format attendu par Ollama.
        
        Ollama attend: [{"role": "user", "content": "..."}]
        """
        return [{"role": m.role, "content": m.content} for m in messages]
    
    async def chat(self, messages: list[Message]) -> LLMResponse:
        """
        Envoie une requête de chat à Ollama.
        
        Endpoint: POST /api/chat
        
        Args:
            messages: Historique de la conversation
            
        Returns:
            La réponse complète du LLM
        """
        response = await self._client.post(
            "/api/chat",
            json={
                "model": self.model,
                "messages": self._format_messages(messages),
                "stream": False,  # On veut la réponse complète d'un coup
            }
        )
        response.raise_for_status()  # Lève une erreur si status != 200
        
        data = response.json()
        return LLMResponse(
            content=data["message"]["content"],
            model=data["model"]
        )
    
    async def chat_stream(self, messages: list[Message]) -> AsyncGenerator[str, None]:
        """
        Envoie une requête de chat avec streaming.
        
        Le streaming permet d'afficher la réponse au fur et à mesure
        qu'elle est générée, comme dans ChatGPT.
        
        Yields:
            Morceaux de texte (tokens) un par un
        """
        async with self._client.stream(
            "POST",
            "/api/chat",
            json={
                "model": self.model,
                "messages": self._format_messages(messages),
                "stream": True,  # Active le streaming
            }
        ) as response:
            response.raise_for_status()
            
            # Ollama envoie des lignes JSON, une par token
            async for line in response.aiter_lines():
                if line:
                    import json
                    data = json.loads(line)
                    # Chaque ligne contient un morceau du message
                    if "message" in data and "content" in data["message"]:
                        yield data["message"]["content"]
    
    async def close(self):
        """Ferme proprement la connexion HTTP."""
        await self._client.aclose()
