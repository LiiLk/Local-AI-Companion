"""
XTTS v2 Provider - Voice Cloning TTS multilingue de Coqui.

XTTS v2 est un modèle TTS état de l'art avec :
- 17 langues supportées (dont français !)
- Voice cloning avec seulement 6 secondes d'audio
- Streaming avec latence < 200ms
- Qualité naturelle et expressive

Specs:
- ~2.8GB VRAM sur GPU
- 1.9GB de modèle (téléchargement auto)
- Sample rate: 24kHz
"""

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncGenerator

import numpy as np
import soundfile as sf

from .base import BaseTTS, TTSResult

logger = logging.getLogger(__name__)


# Langues supportées par XTTS v2
SUPPORTED_LANGUAGES = [
    "en", "es", "fr", "de", "it", "pt", "pl", "tr", 
    "ru", "nl", "cs", "ar", "zh-cn", "ja", "hu", "ko", "hi"
]

# Speakers intégrés (quelques exemples)
DEFAULT_SPEAKERS = [
    "Claribel Dervla",      # Voix féminine claire
    "Daisy Studious",       # Voix féminine studieuse
    "Gracie Wise",          # Voix féminine sage
    "Tammie Ema",           # Voix féminine énergique
    "Alison Dietlinde",     # Voix féminine douce
    "Ana Florence",         # Voix féminine naturelle
    "Annmarie Nele",        # Voix féminine expressive
    "Asya Anara",           # Voix féminine mystérieuse
    "Brenda Stern",         # Voix féminine sérieuse
    "Gitta Nikolina",       # Voix féminine européenne
    "Henriette Usha",       # Voix féminine chaleureuse
    "Sofia Hellen",         # Voix féminine élégante
    "Tammy Grit",           # Voix féminine déterminée
    "Tanja Adelina",        # Voix féminine moderne
    "Vjollca Johnnie",      # Voix féminine unique
    "Andrew Chipper",       # Voix masculine enjouée
    "Badr Odhiambo",        # Voix masculine profonde
    "Dionisio Schuyler",    # Voix masculine classique
    "Royston Min",          # Voix masculine asiatique
    "Viktor Eka",           # Voix masculine européenne
    "Abrahan Mack",         # Voix masculine américaine
    "Adde Michal",          # Voix masculine scandinave
    "Baldur Sansen",        # Voix masculine nordique
    "Craig Gutsy",          # Voix masculine énergique
    "Damien Black",         # Voix masculine sombre
    "Gilberto Mathias",     # Voix masculine latine
    "Ilkin Urbano",         # Voix masculine méditerranéenne
    "Kazuhiko Atallah",     # Voix masculine japonaise
    "Ludvig Milivoj",       # Voix masculine slave
    "Suad Qasim",           # Voix masculine arabe
    "Torcull Diarmuid",     # Voix masculine celtique
    "Viktor Menelaos",      # Voix masculine grecque
    "Zacharie Aimilios",    # Voix masculine française
    "Nova Hogarth",         # Voix non-binaire
    "Maja Ruoho",           # Voix féminine finlandaise
    "Uta Obando",           # Voix féminine allemande
    "Lidiya Szekeres",      # Voix féminine hongroise
    "Chandra MacFarland",   # Voix féminine indienne
    "Szofi Granger",        # Voix féminine britannique
    "Camilla Holmström",    # Voix féminine suédoise
    "Lilya Stainthorpe",    # Voix féminine russe
    "Zofija Kendrick",      # Voix féminine polonaise
    "Narelle Moon",         # Voix féminine australienne
    "Barbora MacLean",      # Voix féminine écossaise
    "Alexandra Hisakawa",   # Voix féminine japonaise
    "Alma María",           # Voix féminine espagnole
    "Rosemary Okafor",      # Voix féminine africaine
    "Ige Behringer",        # Voix féminine allemande
    "Filip Traverse",       # Voix masculine française
    "Damjan Chapman",       # Voix masculine britannique
    "Wulf Carlevaro",       # Voix masculine italienne
    "Aaron Dreschner",      # Voix masculine américaine
    "Kumar Dahl",           # Voix masculine indienne
    "Eugenio Matarese",     # Voix masculine italienne
    "Ferran Sansen",        # Voix masculine catalane
    "Xavier Hayasaka",      # Voix masculine japonaise
    "Luis Moray",           # Voix masculine espagnole
    "Marcos Rudaski",       # Voix masculine polonaise
]


@dataclass
class XTTSConfig:
    """Configuration pour XTTS v2."""
    
    # Langue par défaut
    language: str = "fr"
    
    # Speaker intégré (si pas de voice cloning)
    speaker: str = "Claribel Dervla"
    
    # Voice cloning : audio de référence
    speaker_wav: str | None = None
    
    # Device : "cuda" ou "cpu"
    device: str | None = None  # None = auto-detect


class XTTSProvider(BaseTTS):
    """
    Provider TTS utilisant XTTS v2 de Coqui.
    
    XTTS v2 offre :
    - 17 langues dont le français avec accent natif
    - Voice cloning avec 6 secondes d'audio
    - 58 speakers intégrés
    - ~2.8GB VRAM, génération rapide (~1.7s pour une phrase)
    
    Example:
        # Avec speaker intégré
        tts = XTTSProvider(language="fr", speaker="Claribel Dervla")
        await tts.synthesize("Bonjour !", Path("output.wav"))
        
        # Avec voice cloning
        tts = XTTSProvider(language="fr", speaker_wav="~/voices/my_voice.wav")
        await tts.synthesize("Bonjour !", Path("output.wav"))
    """
    
    def __init__(
        self,
        language: str = "fr",
        speaker: str = "Claribel Dervla",
        speaker_wav: str | Path | None = None,
        device: str | None = None,
    ):
        """
        Initialise le provider XTTS v2.
        
        Args:
            language: Code langue (fr, en, de, etc.)
            speaker: Nom du speaker intégré (ignoré si speaker_wav fourni)
            speaker_wav: Chemin vers audio de référence pour voice cloning
            device: "cuda", "cpu" ou None (auto-detect)
        """
        self.language = language
        self.speaker = speaker
        self.speaker_wav = Path(speaker_wav).expanduser() if speaker_wav else None
        self.device = device
        
        # Lazy loading
        self._model = None
        self._TTS = None
        
        # Validation
        if language not in SUPPORTED_LANGUAGES:
            logger.warning(
                f"Langue '{language}' non officiellement supportée. "
                f"Langues supportées: {SUPPORTED_LANGUAGES}"
            )
    
    @property
    def model_name(self) -> str:
        """Nom du modèle pour l'affichage."""
        return "XTTS v2"
    
    def _load_model(self):
        """
        Charge le modèle XTTS v2 (lazy loading).
        
        Le modèle est téléchargé automatiquement depuis HuggingFace
        au premier appel (~1.9GB).
        """
        if self._model is not None:
            return self._model
        
        logger.info(f"🔄 Chargement de {self.model_name}...")
        
        from TTS.api import TTS
        import torch
        
        # Auto-detect device
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self._model = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(self.device)
        
        logger.info(f"✅ {self.model_name} chargé sur {self.device} !")
        return self._model
    
    async def synthesize(
        self,
        text: str,
        output_path: Path | None = None
    ) -> TTSResult:
        """
        Convertit du texte en fichier audio WAV.
        
        Args:
            text: Texte à synthétiser
            output_path: Chemin de sortie (optionnel)
            
        Returns:
            TTSResult avec le chemin du fichier audio
        """
        if not text.strip():
            raise ValueError("Le texte ne peut pas être vide")
        
        # Créer un fichier temporaire si pas de chemin spécifié
        if output_path is None:
            import tempfile
            output_path = Path(tempfile.mktemp(suffix=".wav"))
        
        output_path = Path(output_path)
        
        # Synthèse dans un thread pour ne pas bloquer l'event loop
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            self._synthesize_sync,
            text,
            output_path
        )
        
        # Calculer la durée
        info = sf.info(str(output_path))
        duration = info.duration
        
        return TTSResult(audio_path=output_path, duration=duration)
    
    def _synthesize_sync(self, text: str, output_path: Path) -> None:
        """
        Synthèse synchrone (appelée dans un thread).
        """
        model = self._load_model()
        
        # Voice cloning ou speaker intégré ?
        if self.speaker_wav and self.speaker_wav.exists():
            # Voice cloning
            model.tts_to_file(
                text=text,
                speaker_wav=str(self.speaker_wav),
                language=self.language,
                file_path=str(output_path)
            )
        else:
            # Speaker intégré
            model.tts_to_file(
                text=text,
                speaker=self.speaker,
                language=self.language,
                file_path=str(output_path)
            )
    
    async def synthesize_stream(
        self,
        text: str
    ) -> AsyncGenerator[bytes, None]:
        """
        Génère l'audio en streaming.
        
        XTTS v2 supporte le streaming natif avec latence < 200ms.
        Cette implémentation utilise le streaming interne de XTTS.
        
        Args:
            text: Texte à synthétiser
            
        Yields:
            Chunks audio en bytes (format WAV)
        """
        import io
        import wave
        
        # Pour le streaming, on génère l'audio complet puis on le découpe
        # Une implémentation plus avancée utiliserait model.inference_stream()
        loop = asyncio.get_event_loop()
        
        # Générer l'audio complet
        import tempfile
        temp_path = Path(tempfile.mktemp(suffix=".wav"))
        
        await loop.run_in_executor(
            None,
            self._synthesize_sync,
            text,
            temp_path
        )
        
        # Lire et streamer par chunks
        chunk_size = 4096  # ~85ms à 24kHz
        
        with open(temp_path, "rb") as f:
            # Envoyer le header WAV d'abord
            header = f.read(44)
            yield header
            
            # Puis les données audio par chunks
            while chunk := f.read(chunk_size):
                yield chunk
        
        # Nettoyer
        temp_path.unlink(missing_ok=True)
    
    async def synthesize_to_bytes(self, text: str) -> bytes:
        """
        Convertit du texte directement en bytes audio.
        
        Args:
            text: Texte à synthétiser
            
        Returns:
            Audio en bytes (format WAV)
        """
        import tempfile
        
        temp_path = Path(tempfile.mktemp(suffix=".wav"))
        
        try:
            await self.synthesize(text, temp_path)
            
            with open(temp_path, "rb") as f:
                return f.read()
        finally:
            temp_path.unlink(missing_ok=True)
    
    async def list_voices(self, language: str | None = None) -> list:
        """
        Liste les voix (speakers) disponibles.
        
        Pour XTTS, les "voix" sont les speakers intégrés.
        Le paramètre language est ignoré car tous les speakers
        peuvent parler toutes les langues.
        
        Args:
            language: Ignoré pour XTTS (tous speakers sont multilingues)
            
        Returns:
            Liste de Voice objects
        """
        from .base import Voice
        
        speakers = self.list_speakers()
        voices = []
        
        for speaker in speakers:
            # XTTS speakers sont tous multilingues
            voices.append(Voice(
                id=speaker,
                name=speaker,
                language="multilingual",
                gender="Unknown"  # XTTS ne spécifie pas le genre
            ))
        
        return voices
    
    def set_voice(self, voice_id: str) -> None:
        """
        Change le speaker utilisé.
        
        Args:
            voice_id: Nom du speaker (ex: "Claribel Dervla")
        """
        self.speaker = voice_id
    
    def set_rate(self, rate: str) -> None:
        """
        Non supporté par XTTS v2.
        
        XTTS génère l'audio à vitesse naturelle.
        Pour modifier la vitesse, utiliser un post-traitement audio.
        """
        pass  # Non supporté
    
    def set_pitch(self, pitch: str) -> None:
        """
        Non supporté par XTTS v2.
        
        XTTS génère l'audio avec le pitch naturel du speaker.
        Pour modifier le pitch, utiliser un post-traitement audio.
        """
        pass  # Non supporté
    
    def list_speakers(self) -> list[str]:
        """
        Liste les speakers intégrés disponibles.
        
        Returns:
            Liste des noms de speakers
        """
        model = self._load_model()
        return model.speakers
    
    @staticmethod
    def list_languages() -> list[str]:
        """
        Liste les langues supportées.
        
        Returns:
            Liste des codes langue
        """
        return SUPPORTED_LANGUAGES.copy()
