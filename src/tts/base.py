"""
Module de base pour le TTS (Text-to-Speech).

Ce fichier définit l'INTERFACE que tous les TTS doivent respecter.
Même principe que pour le LLM : on peut changer de provider
(Edge TTS → Coqui → Piper) sans modifier le code principal.

Le TTS convertit du texte en audio. Il peut :
1. Générer un fichier audio complet
2. Streamer l'audio chunk par chunk (pour le temps réel)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import AsyncGenerator
from pathlib import Path


@dataclass
class TTSResult:
    """
    Résultat d'une synthèse vocale.
    
    Attributes:
        audio_path: Chemin vers le fichier audio généré (si sauvegardé)
        audio_data: Données audio brutes en bytes (si en mémoire)
        duration: Durée de l'audio en secondes (si connue)
    """
    audio_path: Path | None = None
    audio_data: bytes | None = None
    duration: float | None = None


@dataclass
class Voice:
    """
    Représente une voix disponible.
    
    Attributes:
        id: Identifiant unique de la voix (ex: "fr-FR-DeniseNeural")
        name: Nom lisible (ex: "Denise")
        language: Code langue (ex: "fr-FR")
        gender: "Male" ou "Female"
    """
    id: str
    name: str
    language: str
    gender: str


class BaseTTS(ABC):
    """
    Classe abstraite de base pour tous les TTS.
    
    Toute implémentation TTS doit hériter de cette classe
    et implémenter les méthodes abstraites.
    
    Example:
        class EdgeTTSProvider(BaseTTS):
            async def synthesize(self, text, voice):
                # Implémentation spécifique à Edge TTS
                ...
    """
    
    @abstractmethod
    async def synthesize(
        self, 
        text: str, 
        output_path: Path | None = None
    ) -> TTSResult:
        """
        Convertit du texte en audio.
        
        Args:
            text: Le texte à convertir en parole
            output_path: Chemin où sauvegarder l'audio (optionnel)
                        Si None, l'audio est retourné en mémoire
        
        Returns:
            TTSResult contenant le chemin ou les données audio
        
        Example:
            result = await tts.synthesize("Bonjour !", Path("output.mp3"))
            # result.audio_path contient le chemin du fichier
        """
        pass
    
    @abstractmethod
    async def synthesize_stream(
        self, 
        text: str
    ) -> AsyncGenerator[bytes, None]:
        """
        Convertit du texte en audio en streaming.
        
        Utile pour commencer à jouer l'audio avant que
        toute la synthèse soit terminée (latence réduite).
        
        Args:
            text: Le texte à convertir
            
        Yields:
            Chunks d'audio (bytes) au fur et à mesure
            
        Example:
            async for chunk in tts.synthesize_stream("Bonjour !"):
                audio_player.feed(chunk)  # Joue immédiatement
        """
        pass
    
    @abstractmethod
    async def list_voices(self, language: str | None = None) -> list[Voice]:
        """
        Liste les voix disponibles.
        
        Args:
            language: Filtre par langue (ex: "fr-FR", "en-US")
                     Si None, retourne toutes les voix
        
        Returns:
            Liste des voix disponibles
            
        Example:
            voices = await tts.list_voices("fr-FR")
            for v in voices:
                print(f"{v.name} ({v.gender})")
        """
        pass
    
    @abstractmethod
    def set_voice(self, voice_id: str) -> None:
        """
        Change la voix utilisée pour la synthèse.
        
        Args:
            voice_id: Identifiant de la voix (ex: "fr-FR-DeniseNeural")
        """
        pass
    
    @abstractmethod
    def set_rate(self, rate: str) -> None:
        """
        Change la vitesse de parole.
        
        Args:
            rate: Modification de vitesse (ex: "+20%", "-10%", "+0%")
        """
        pass
    
    @abstractmethod
    def set_pitch(self, pitch: str) -> None:
        """
        Change la hauteur de la voix.
        
        Args:
            pitch: Modification de pitch (ex: "+10Hz", "-5Hz")
        """
        pass
