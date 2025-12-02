"""
Implémentation TTS utilisant Microsoft Edge TTS.

Edge TTS utilise l'API de synthèse vocale de Microsoft Edge.
C'est gratuit, de bonne qualité, et supporte 50+ langues.

Avantages :
- Gratuit et illimité
- Voix très naturelles (neural voices)
- Pas besoin de GPU
- Support async natif

Inconvénients :
- Nécessite une connexion internet
- Pas de voice cloning
"""

import edge_tts
from pathlib import Path
from typing import AsyncGenerator

from .base import BaseTTS, TTSResult, Voice


# Voix recommandées par langue
RECOMMENDED_VOICES = {
    "fr-FR": "fr-FR-DeniseNeural",      # Française, naturelle
    "fr-CA": "fr-CA-SylvieNeural",      # Québécoise
    "en-US": "en-US-JennyNeural",       # Américaine
    "en-GB": "en-GB-SoniaNeural",       # Britannique
    "ja-JP": "ja-JP-NanamiNeural",      # Japonaise
    "es-ES": "es-ES-ElviraNeural",      # Espagnole
    "de-DE": "de-DE-KatjaNeural",       # Allemande
    "it-IT": "it-IT-ElsaNeural",        # Italienne
    "zh-CN": "zh-CN-XiaoxiaoNeural",    # Chinoise
    "ko-KR": "ko-KR-SunHiNeural",       # Coréenne
}


class EdgeTTSProvider(BaseTTS):
    """
    Provider TTS utilisant Microsoft Edge TTS.
    
    Attributes:
        voice: Identifiant de la voix actuelle
        rate: Vitesse de parole (ex: "+0%")
        pitch: Hauteur de voix (ex: "+0Hz")
    
    Example:
        tts = EdgeTTSProvider(voice="fr-FR-DeniseNeural")
        result = await tts.synthesize("Bonjour !", Path("hello.mp3"))
    """
    
    def __init__(
        self,
        voice: str = "fr-FR-DeniseNeural",
        rate: str = "+20%",
        pitch: str = "+0Hz"
    ):
        """
        Initialise le provider Edge TTS.
        
        Args:
            voice: Identifiant de la voix (voir RECOMMENDED_VOICES)
            rate: Vitesse de parole (ex: "+20%" pour plus rapide)
            pitch: Hauteur de voix (ex: "+10Hz" pour plus aigu)
        """
        self.voice = voice
        self.rate = rate
        self.pitch = pitch
    
    async def synthesize(
        self,
        text: str,
        output_path: Path | None = None
    ) -> TTSResult:
        """
        Convertit du texte en fichier audio MP3.
        
        Args:
            text: Texte à synthétiser
            output_path: Chemin de sortie (défaut: temp file)
            
        Returns:
            TTSResult avec le chemin du fichier audio
        """
        # Créer le communicator Edge TTS
        communicate = edge_tts.Communicate(
            text=text,
            voice=self.voice,
            rate=self.rate,
            pitch=self.pitch
        )
        
        # Définir le chemin de sortie
        if output_path is None:
            output_path = Path(f"/tmp/tts_output_{hash(text)}.mp3")
        
        # Générer et sauvegarder l'audio
        await communicate.save(str(output_path))
        
        return TTSResult(audio_path=output_path)
    
    async def synthesize_stream(
        self,
        text: str
    ) -> AsyncGenerator[bytes, None]:
        """
        Génère l'audio en streaming (chunk par chunk).
        
        Permet de commencer à jouer l'audio avant que
        toute la synthèse soit terminée.
        
        Args:
            text: Texte à synthétiser
            
        Yields:
            Chunks audio (bytes) au format MP3
        """
        communicate = edge_tts.Communicate(
            text=text,
            voice=self.voice,
            rate=self.rate,
            pitch=self.pitch
        )
        
        # Edge TTS envoie des chunks audio + metadata
        async for chunk in communicate.stream():
            # On ne garde que les chunks audio (pas les metadata)
            if chunk["type"] == "audio":
                yield chunk["data"]
    
    async def synthesize_to_bytes(self, text: str) -> bytes:
        """
        Méthode helper : synthétise et retourne les bytes directement.
        
        Utile quand on veut l'audio en mémoire sans sauvegarder.
        
        Args:
            text: Texte à synthétiser
            
        Returns:
            Données audio complètes en bytes (MP3)
        """
        audio_chunks = []
        async for chunk in self.synthesize_stream(text):
            audio_chunks.append(chunk)
        return b"".join(audio_chunks)
    
    async def list_voices(self, language: str | None = None) -> list[Voice]:
        """
        Liste toutes les voix disponibles.
        
        Args:
            language: Filtre par langue (ex: "fr-FR", "en")
            
        Returns:
            Liste des voix disponibles
        """
        # Récupérer toutes les voix de Edge TTS
        voices_data = await edge_tts.list_voices()
        
        voices = []
        for v in voices_data:
            # Filtrer par langue si spécifié
            if language:
                # Supporte "fr" ou "fr-FR"
                if not v["Locale"].startswith(language):
                    continue
            
            voices.append(Voice(
                id=v["ShortName"],
                name=v["DisplayName"],
                language=v["Locale"],
                gender=v["Gender"]
            ))
        
        return voices
    
    def set_voice(self, voice_id: str) -> None:
        """Change la voix utilisée."""
        self.voice = voice_id
    
    def set_rate(self, rate: str) -> None:
        """Change la vitesse de parole."""
        self.rate = rate
    
    def set_pitch(self, pitch: str) -> None:
        """Change la hauteur de voix."""
        self.pitch = pitch
    
    @staticmethod
    def get_recommended_voice(language: str) -> str:
        """
        Retourne une voix recommandée pour une langue.
        
        Args:
            language: Code langue (ex: "fr-FR", "en-US")
            
        Returns:
            Identifiant de la voix recommandée
        """
        return RECOMMENDED_VOICES.get(language, "en-US-JennyNeural")
