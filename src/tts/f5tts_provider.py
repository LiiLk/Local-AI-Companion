"""
Implémentation TTS utilisant F5-TTS - Voice Cloning léger et rapide.

F5-TTS est un modèle TTS basé sur Flow Matching avec ~300M paramètres.
Il offre le voice cloning avec seulement 10-30s d'audio de référence.

Avantages :
- Léger (~2-3GB VRAM) - peut coexister avec un LLM 7B
- Très rapide (RTF ~0.04x sur GPU = temps réel!)
- Voice cloning avec 10-30s d'audio de référence
- Support multilingue natif (FR, EN, ZH, JA, etc.)
- Téléchargement automatique des modèles depuis HuggingFace
- API Python simple et propre

Inconvénients :
- Qualité légèrement inférieure à OpenAudio S1-mini
- Licence CC-BY-NC (non commercial)

Usage :
    # Sans voice cloning (voix par défaut)
    tts = F5TTSProvider()
    result = await tts.synthesize("Bonjour le monde !")
    
    # Avec voice cloning
    tts = F5TTSProvider(
        ref_audio="reference.wav",
        ref_text="Transcription exacte de l'audio de référence."
    )
"""

import asyncio
import tempfile
from pathlib import Path
from typing import AsyncGenerator

import numpy as np
import soundfile as sf

from .base import BaseTTS, TTSResult, Voice


# Voix par défaut (sans voice cloning)
AVAILABLE_VOICES = [
    Voice(id="default", name="Default F5-TTS", language="multi", gender="Unknown"),
    Voice(id="cloned", name="Cloned Voice", language="multi", gender="Unknown"),
]


class F5TTSProvider(BaseTTS):
    """
    Provider TTS utilisant F5-TTS - Voice Cloning léger.
    
    F5-TTS utilise un système de voice cloning : vous fournissez
    un échantillon audio de référence et sa transcription, et le modèle
    génère de la parole dans cette voix.
    
    Sans référence, le modèle utilise une voix par défaut.
    
    Attributes:
        ref_audio: Chemin vers l'audio de référence pour voice cloning
        ref_text: Transcription de l'audio de référence
        _model: Instance F5TTS (chargée à la demande)
    
    Example:
        # Sans voice cloning
        tts = F5TTSProvider()
        result = await tts.synthesize("Bonjour le monde !")
        
        # Avec voice cloning
        tts = F5TTSProvider(
            ref_audio=Path("reference.wav"),
            ref_text="Bonjour, je suis la voix de référence."
        )
    """
    
    # Sample rate de F5-TTS (24kHz via Vocos vocoder)
    SAMPLE_RATE = 24000
    
    def __init__(
        self,
        ref_audio: str | Path | None = None,
        ref_text: str | None = None,
        model: str = "F5TTS_v1_Base",
        device: str | None = None,
        seed: int | None = None,
    ):
        """
        Initialise le provider F5-TTS.
        
        Args:
            ref_audio: Chemin vers l'audio de référence (10-30s recommandé)
            ref_text: Transcription exacte de l'audio de référence
                      Si vide, F5-TTS utilisera un ASR pour transcrire (+ VRAM)
            model: Modèle à utiliser ("F5TTS_v1_Base" ou "E2TTS_Base")
            device: Device pour l'inférence (None = auto-détection cuda/cpu)
            seed: Graine pour la reproductibilité (None = aléatoire)
        """
        # Configuration voice cloning
        self.ref_audio = Path(ref_audio) if ref_audio else None
        self.ref_text = ref_text or ""  # Vide = auto-transcription
        
        if self.ref_audio and not self.ref_audio.exists():
            raise FileNotFoundError(f"Reference audio not found: {self.ref_audio}")
        
        # Configuration modèle
        self.model_name = model
        self.device = device
        self.seed = seed
        
        # Modèle chargé à la demande (lazy loading)
        self._model = None
    
    def _load_model(self):
        """
        Charge le modèle F5-TTS (lazy loading).
        
        Le modèle est téléchargé automatiquement depuis HuggingFace
        au premier appel (~1.4GB).
        """
        if self._model is not None:
            return self._model
        
        print(f"🔄 Chargement de F5-TTS ({self.model_name})...")
        
        from f5_tts.api import F5TTS
        
        self._model = F5TTS(
            model=self.model_name,
            device=self.device,
        )
        
        print("✅ F5-TTS chargé !")
        return self._model
    
    def _get_default_ref(self) -> tuple[str, str]:
        """
        Retourne l'audio et texte de référence par défaut de F5-TTS.
        
        F5-TTS inclut un exemple de référence anglais par défaut.
        """
        from importlib.resources import files
        
        default_audio = str(files("f5_tts").joinpath("infer/examples/basic/basic_ref_en.wav"))
        default_text = "Some call me nature, others call me mother nature."
        
        return default_audio, default_text
    
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
        # L'inférence est synchrone, on l'exécute dans un thread
        loop = asyncio.get_event_loop()
        wav, sr = await loop.run_in_executor(
            None, 
            self._synthesize_sync, 
            text
        )
        
        # Définir le chemin de sortie
        if output_path is None:
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            output_path = Path(tmp.name)
            tmp.close()
        
        # Sauvegarder en WAV
        sf.write(str(output_path), wav, sr)
        
        # Calculer la durée
        duration = len(wav) / sr
        
        return TTSResult(audio_path=output_path, duration=duration)
    
    def _synthesize_sync(self, text: str) -> tuple[np.ndarray, int]:
        """
        Synthèse synchrone (appelée dans un thread).
        
        Returns:
            Tuple (wav_array, sample_rate)
        """
        model = self._load_model()
        
        # Déterminer la référence à utiliser
        if self.ref_audio:
            ref_file = str(self.ref_audio)
            ref_text = self.ref_text
        else:
            ref_file, ref_text = self._get_default_ref()
        
        # Générer l'audio
        wav, sr, _ = model.infer(
            ref_file=ref_file,
            ref_text=ref_text,
            gen_text=text,
            seed=self.seed,
        )
        
        return wav, sr
    
    async def synthesize_stream(
        self,
        text: str
    ) -> AsyncGenerator[bytes, None]:
        """
        Génère l'audio en streaming.
        
        Note: F5-TTS supporte le chunk inference en interne,
        mais l'API actuelle retourne l'audio complet.
        On simule le streaming en découpant l'audio.
        
        Args:
            text: Texte à synthétiser
            
        Yields:
            Chunks audio en bytes (format WAV)
        """
        import io
        
        loop = asyncio.get_event_loop()
        wav, sr = await loop.run_in_executor(
            None, 
            self._synthesize_sync, 
            text
        )
        
        # Découper en chunks de ~0.5s
        chunk_size = sr // 2  # 0.5 seconde
        
        for i in range(0, len(wav), chunk_size):
            chunk = wav[i:i + chunk_size]
            
            # Convertir en WAV bytes
            buffer = io.BytesIO()
            sf.write(buffer, chunk, sr, format='WAV')
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
        import io
        
        loop = asyncio.get_event_loop()
        wav, sr = await loop.run_in_executor(
            None,
            self._synthesize_sync,
            text
        )
        
        # Convertir en WAV bytes
        buffer = io.BytesIO()
        sf.write(buffer, wav, sr, format='WAV')
        buffer.seek(0)
        return buffer.read()
    
    async def list_voices(self, language: str | None = None) -> list[Voice]:
        """
        Liste les voix disponibles.
        
        F5-TTS utilise le voice cloning, donc les "voix" sont
        définies par l'audio de référence, pas par des presets.
        
        Args:
            language: Ignoré (F5-TTS est multilingue)
            
        Returns:
            Liste des voix disponibles
        """
        voices = AVAILABLE_VOICES.copy()
        
        # Ajouter une voix personnalisée si configurée
        if self.ref_audio:
            voices.append(Voice(
                id="custom",
                name=f"Custom ({self.ref_audio.stem})",
                language="multi",
                gender="Unknown"
            ))
        
        return voices
    
    def set_voice(self, voice_id: str) -> None:
        """
        F5-TTS n'a pas de voix préréglées.
        
        Pour changer de voix, utilisez set_reference() avec
        un nouvel audio de référence.
        """
        pass  # No-op car F5-TTS utilise voice cloning
    
    def set_reference(
        self,
        ref_audio: str | Path,
        ref_text: str = ""
    ) -> None:
        """
        Configure la voix de référence pour le voice cloning.
        
        Args:
            ref_audio: Chemin vers l'audio de référence (10-30s recommandé)
            ref_text: Transcription exacte de l'audio (vide = auto-transcription)
            
        Example:
            tts.set_reference(
                "reference.wav",
                "Bonjour, je suis une voix de référence claire et naturelle."
            )
        """
        ref_path = Path(ref_audio)
        if not ref_path.exists():
            raise FileNotFoundError(f"Reference audio not found: {ref_path}")
        
        self.ref_audio = ref_path
        self.ref_text = ref_text
    
    def set_rate(self, rate: str) -> None:
        """
        Change la vitesse de parole (non supporté directement).
        
        F5-TTS ne supporte pas le changement de vitesse.
        Cette méthode existe pour la compatibilité avec l'interface.
        
        Args:
            rate: Ignoré
        """
        pass  # F5-TTS ne supporte pas le rate
    
    def set_pitch(self, pitch: str) -> None:
        """
        Change la hauteur de voix (non supporté).
        
        F5-TTS ne supporte pas le changement de pitch.
        Cette méthode existe pour la compatibilité avec l'interface.
        
        Args:
            pitch: Ignoré
        """
        pass  # F5-TTS ne supporte pas le pitch
    
    def set_seed(self, seed: int | None) -> None:
        """
        Change la graine de génération.
        
        Une graine fixe permet des résultats reproductibles.
        
        Args:
            seed: Graine (None = aléatoire)
        """
        self.seed = seed
