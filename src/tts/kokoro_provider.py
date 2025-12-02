"""
Implémentation TTS utilisant Kokoro - TTS local haute qualité.

Kokoro est un modèle TTS open-source de 82M paramètres qui offre
une qualité comparable à ElevenLabs tout en étant 100% local.

Avantages :
- 100% local (pas besoin d'internet)
- Qualité vocale exceptionnelle et naturelle
- Débit de parole réaliste
- Support multilingue (FR, EN, JA, ZH, etc.)
- Léger (82M paramètres, ~300MB)
- Fonctionne sur CPU (GPU optionnel pour plus de vitesse)

Inconvénients :
- Premier chargement plus lent (téléchargement du modèle)
- Consomme plus de RAM que Edge TTS
"""

import io
import tempfile
from pathlib import Path
from typing import AsyncGenerator
import asyncio

import soundfile as sf
import numpy as np

from .base import BaseTTS, TTSResult, Voice


# Mapping des codes langue Kokoro
# Kokoro utilise des codes à une lettre pour les langues
LANG_CODES = {
    "en-US": "a",  # American English
    "en-GB": "b",  # British English
    "es-ES": "e",  # Spanish
    "fr-FR": "f",  # French
    "hi-IN": "h",  # Hindi
    "it-IT": "i",  # Italian
    "ja-JP": "j",  # Japanese
    "pt-BR": "p",  # Brazilian Portuguese
    "zh-CN": "z",  # Mandarin Chinese
}

# Voix recommandées par langue
# Format: voice_id (utilisé par Kokoro)
RECOMMENDED_VOICES = {
    "fr-FR": "ff_siwis",      # Voix française féminine (SIWIS dataset)
    "en-US": "af_heart",      # Voix américaine féminine
    "en-GB": "bf_emma",       # Voix britannique féminine
    "ja-JP": "jf_alpha",      # Voix japonaise féminine
    "zh-CN": "zf_xiaobei",    # Voix chinoise féminine
    "es-ES": "ef_dora",       # Voix espagnole féminine
    "it-IT": "if_sara",       # Voix italienne féminine
}

# Liste complète des voix disponibles
AVAILABLE_VOICES = [
    # Français
    Voice(id="ff_siwis", name="Siwis (French Female)", language="fr-FR", gender="Female"),
    
    # American English
    Voice(id="af_heart", name="Heart (US Female)", language="en-US", gender="Female"),
    Voice(id="af_bella", name="Bella (US Female)", language="en-US", gender="Female"),
    Voice(id="af_nicole", name="Nicole (US Female)", language="en-US", gender="Female"),
    Voice(id="af_sarah", name="Sarah (US Female)", language="en-US", gender="Female"),
    Voice(id="af_sky", name="Sky (US Female)", language="en-US", gender="Female"),
    Voice(id="am_adam", name="Adam (US Male)", language="en-US", gender="Male"),
    Voice(id="am_michael", name="Michael (US Male)", language="en-US", gender="Male"),
    
    # British English
    Voice(id="bf_emma", name="Emma (UK Female)", language="en-GB", gender="Female"),
    Voice(id="bf_isabella", name="Isabella (UK Female)", language="en-GB", gender="Female"),
    Voice(id="bm_george", name="George (UK Male)", language="en-GB", gender="Male"),
    Voice(id="bm_lewis", name="Lewis (UK Male)", language="en-GB", gender="Male"),
    
    # Japanese
    Voice(id="jf_alpha", name="Alpha (JP Female)", language="ja-JP", gender="Female"),
    Voice(id="jf_gongitsune", name="Gongitsune (JP Female)", language="ja-JP", gender="Female"),
    Voice(id="jm_kumo", name="Kumo (JP Male)", language="ja-JP", gender="Male"),
    
    # Chinese
    Voice(id="zf_xiaobei", name="Xiaobei (CN Female)", language="zh-CN", gender="Female"),
    Voice(id="zf_xiaoni", name="Xiaoni (CN Female)", language="zh-CN", gender="Female"),
    Voice(id="zm_yunjian", name="Yunjian (CN Male)", language="zh-CN", gender="Male"),
]


class KokoroProvider(BaseTTS):
    """
    Provider TTS utilisant Kokoro - Modèle local haute qualité.
    
    Kokoro génère de l'audio à 24kHz avec un débit naturel.
    Le modèle est chargé au premier appel (lazy loading).
    
    Attributes:
        voice: Identifiant de la voix Kokoro
        lang_code: Code langue pour la phonétisation
        speed: Vitesse de parole (1.0 = normal)
        _pipeline: Pipeline Kokoro (chargé à la demande)
    
    Example:
        tts = KokoroProvider(voice="ff_siwis")
        result = await tts.synthesize("Bonjour le monde !")
    """
    
    # Sample rate de Kokoro (fixe)
    SAMPLE_RATE = 24000
    
    def __init__(
        self,
        voice: str = "ff_siwis",
        lang_code: str | None = None,
        speed: float = 1.0
    ):
        """
        Initialise le provider Kokoro.
        
        Args:
            voice: Identifiant de la voix (ex: "ff_siwis", "af_heart")
                   Le préfixe indique la langue (ff=français, af=anglais US, etc.)
            lang_code: Code langue explicite (auto-détecté depuis la voix si None)
            speed: Vitesse de parole (0.5 à 2.0, 1.0 = normal)
        """
        self.voice = voice
        self.speed = speed
        self._pipeline = None
        
        # Auto-détecter le code langue depuis le préfixe de la voix
        # ff_siwis -> f (français), af_heart -> a (anglais US)
        if lang_code:
            self.lang_code = lang_code
        else:
            voice_prefix = voice[:2] if len(voice) >= 2 else "a"
            # Mapping préfixe voix -> code langue Kokoro
            prefix_to_lang = {
                "ff": "f",  # French female
                "fm": "f",  # French male
                "af": "a",  # American female
                "am": "a",  # American male
                "bf": "b",  # British female
                "bm": "b",  # British male
                "jf": "j",  # Japanese female
                "jm": "j",  # Japanese male
                "zf": "z",  # Chinese female
                "zm": "z",  # Chinese male
                "ef": "e",  # Spanish female
                "em": "e",  # Spanish male
                "if": "i",  # Italian female
                "im": "i",  # Italian male
            }
            self.lang_code = prefix_to_lang.get(voice_prefix, "a")
    
    def _load_pipeline(self):
        """
        Charge le pipeline Kokoro (lazy loading).
        
        Le modèle est téléchargé automatiquement au premier appel
        depuis HuggingFace (~300MB).
        """
        if self._pipeline is None:
            from kokoro import KPipeline
            
            print(f"🔄 Chargement de Kokoro (lang={self.lang_code})...")
            self._pipeline = KPipeline(lang_code=self.lang_code)
            print("✅ Kokoro chargé !")
        
        return self._pipeline
    
    async def synthesize(
        self,
        text: str,
        output_path: Path | None = None
    ) -> TTSResult:
        """
        Convertit du texte en fichier audio WAV.
        
        Args:
            text: Texte à synthétiser
            output_path: Chemin de sortie (défaut: temp file)
            
        Returns:
            TTSResult avec le chemin du fichier audio
        """
        # Kokoro est synchrone, on l'exécute dans un thread
        loop = asyncio.get_event_loop()
        audio_data = await loop.run_in_executor(
            None, 
            self._synthesize_sync, 
            text
        )
        
        # Définir le chemin de sortie
        if output_path is None:
            # Créer un fichier temporaire WAV
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            output_path = Path(tmp.name)
            tmp.close()
        
        # Sauvegarder en WAV
        sf.write(str(output_path), audio_data, self.SAMPLE_RATE)
        
        return TTSResult(audio_path=output_path)
    
    def _synthesize_sync(self, text: str) -> np.ndarray:
        """
        Synthèse synchrone (appelée dans un thread).
        
        Returns:
            Array numpy contenant les données audio
        """
        pipeline = self._load_pipeline()
        
        # Générer l'audio
        # Le générateur retourne (graphemes, phonemes, audio) pour chaque segment
        audio_segments = []
        
        for _, _, audio in pipeline(text, voice=self.voice, speed=self.speed):
            audio_segments.append(audio)
        
        # Concaténer tous les segments
        if audio_segments:
            return np.concatenate(audio_segments)
        else:
            return np.array([], dtype=np.float32)
    
    async def synthesize_stream(
        self,
        text: str
    ) -> AsyncGenerator[bytes, None]:
        """
        Génère l'audio en streaming (segment par segment).
        
        Kokoro génère naturellement par segments (phrases),
        ce qui permet de commencer la lecture avant la fin.
        
        Args:
            text: Texte à synthétiser
            
        Yields:
            Chunks audio en bytes (format WAV PCM)
        """
        loop = asyncio.get_event_loop()
        
        # Générateur synchrone vers async
        def generate_segments():
            pipeline = self._load_pipeline()
            for _, _, audio in pipeline(text, voice=self.voice, speed=self.speed):
                yield audio
        
        # Convertir en async
        for audio in await loop.run_in_executor(None, list, generate_segments()):
            # Convertir numpy array en bytes WAV
            buffer = io.BytesIO()
            sf.write(buffer, audio, self.SAMPLE_RATE, format='WAV')
            buffer.seek(0)
            yield buffer.read()
    
    async def synthesize_to_bytes(self, text: str) -> bytes:
        """
        Synthétise et retourne les bytes audio directement.
        
        Args:
            text: Texte à synthétiser
            
        Returns:
            Données audio en bytes (format WAV)
        """
        loop = asyncio.get_event_loop()
        audio_data = await loop.run_in_executor(
            None,
            self._synthesize_sync,
            text
        )
        
        # Convertir en WAV bytes
        buffer = io.BytesIO()
        sf.write(buffer, audio_data, self.SAMPLE_RATE, format='WAV')
        buffer.seek(0)
        return buffer.read()
    
    async def list_voices(self, language: str | None = None) -> list[Voice]:
        """
        Liste les voix disponibles.
        
        Args:
            language: Filtre par langue (ex: "fr-FR", "en")
            
        Returns:
            Liste des voix disponibles
        """
        voices = AVAILABLE_VOICES.copy()
        
        if language:
            # Filtrer par langue
            voices = [v for v in voices if v.language.startswith(language)]
        
        return voices
    
    def set_voice(self, voice_id: str) -> None:
        """
        Change la voix utilisée.
        
        Note: Si la langue change, le pipeline sera rechargé.
        """
        old_lang = self.lang_code
        self.voice = voice_id
        
        # Recalculer le code langue
        voice_prefix = voice_id[:2] if len(voice_id) >= 2 else "a"
        prefix_to_lang = {
            "ff": "f", "fm": "f",
            "af": "a", "am": "a",
            "bf": "b", "bm": "b",
            "jf": "j", "jm": "j",
            "zf": "z", "zm": "z",
            "ef": "e", "em": "e",
            "if": "i", "im": "i",
        }
        self.lang_code = prefix_to_lang.get(voice_prefix, "a")
        
        # Si la langue a changé, forcer le rechargement du pipeline
        if old_lang != self.lang_code:
            self._pipeline = None
    
    def set_speed(self, speed: float) -> None:
        """Change la vitesse de parole (0.5 à 2.0)."""
        self.speed = max(0.5, min(2.0, speed))
    
    def set_rate(self, rate: str) -> None:
        """
        Change la vitesse de parole (compatibilité avec l'interface).
        
        Convertit le format Edge TTS ("+20%") en float pour Kokoro.
        
        Args:
            rate: Modification de vitesse (ex: "+20%", "-10%")
        """
        # Convertir "+20%" -> 1.2, "-10%" -> 0.9
        try:
            rate_clean = rate.replace("%", "").replace("+", "")
            rate_value = float(rate_clean) / 100
            self.speed = 1.0 + rate_value
            self.speed = max(0.5, min(2.0, self.speed))
        except ValueError:
            self.speed = 1.0
    
    def set_pitch(self, pitch: str) -> None:
        """
        Change la hauteur de voix (non supporté par Kokoro).
        
        Cette méthode existe pour la compatibilité avec l'interface,
        mais Kokoro ne supporte pas le changement de pitch.
        
        Args:
            pitch: Ignoré (Kokoro ne supporte pas le pitch)
        """
        # Kokoro ne supporte pas le pitch, on ignore silencieusement
        pass
    
    @staticmethod
    def get_recommended_voice(language: str) -> str:
        """
        Retourne une voix recommandée pour une langue.
        
        Args:
            language: Code langue (ex: "fr-FR", "en-US")
            
        Returns:
            Identifiant de la voix recommandée
        """
        return RECOMMENDED_VOICES.get(language, "af_heart")
