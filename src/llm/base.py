"""
Module de base pour les LLM (Large Language Models).

Ce fichier définit l'INTERFACE que tous les LLM doivent respecter.
C'est comme un "contrat" : toute classe qui hérite de BaseLLM
DOIT implémenter les méthodes définies ici.

Pourquoi c'est utile ?
- Tu peux changer de LLM (Ollama → OpenAI) sans modifier le reste du code
- Le code principal utilise BaseLLM, pas une implémentation spécifique
- C'est le pattern "Dependency Inversion" (SOLID principles)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import AsyncGenerator


@dataclass
class Message:
    """
    Représente un message dans une conversation.
    
    Attributes:
        role: "user", "assistant", ou "system"
        content: Le contenu du message
    """
    role: str  # "user", "assistant", "system"
    content: str


@dataclass 
class LLMResponse:
    """
    Représente la réponse du LLM.
    
    Attributes:
        content: Le texte de la réponse
        model: Le nom du modèle utilisé
    """
    content: str
    model: str


class BaseLLM(ABC):
    """
    Classe abstraite de base pour tous les LLM.
    
    ABC = Abstract Base Class
    Les méthodes avec @abstractmethod DOIVENT être implémentées
    par les classes enfants.
    
    Example:
        class OllamaLLM(BaseLLM):
            def chat(self, messages):
                # Implémentation spécifique à Ollama
                ...
    """
    
    @abstractmethod
    async def chat(self, messages: list[Message]) -> LLMResponse:
        """
        Envoie des messages au LLM et obtient une réponse.
        
        Args:
            messages: Liste de messages (historique de conversation)
            
        Returns:
            LLMResponse avec le contenu de la réponse
            
        Note:
            'async' signifie que cette fonction est asynchrone.
            On utilise 'await' pour l'appeler. Ça permet de ne pas
            bloquer le programme pendant que le LLM réfléchit.
        """
        pass
    
    @abstractmethod
    async def chat_stream(self, messages: list[Message]) -> AsyncGenerator[str, None]:
        """
        Comme chat(), mais envoie la réponse mot par mot (streaming).
        
        C'est plus réactif : l'utilisateur voit la réponse s'écrire
        au lieu d'attendre que tout soit généré.
        
        Yields:
            Morceaux de texte au fur et à mesure
        """
        pass
