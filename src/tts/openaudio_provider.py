"""
Implémentation TTS utilisant OpenAudio S1-mini (Fish Speech).

OpenAudio S1-mini est le modèle TTS #1 sur TTS-Arena2 (Mai 2025) avec 0.5B paramètres.
Il offre une qualité exceptionnelle, le voice cloning et le support multilingue natif.

Avantages :
- Qualité #1 sur TTS-Arena2 (meilleur que ElevenLabs)
- Voice cloning avec 10-30 secondes d'audio de référence
- Support multilingue natif (FR, EN, JA, ZH, etc.)
- Émotions via tags: (excited), (whispering), (sad), (laughing)
- Streaming audio pour latence réduite
- 100% local et gratuit

Inconvénients :
- Modèle plus lourd (0.5B params, ~3.5GB)
- Premier chargement lent (~30s sur CPU)
- Requiert ~4GB RAM sur CPU ou ~2GB VRAM sur GPU

Configuration requise :
- Checkpoints dans ~/models/openaudio-s1-mini/
  - model.pth (1.7GB) - Modèle text-to-semantic
  - codec.pth (1.8GB) - Décodeur audio DAC

Usage voice cloning :
    Préparez un fichier audio de référence (10-30s de parole claire)
    et sa transcription. Le modèle clonera cette voix.
"""

import io
import os
import sys
import queue
import tempfile
from pathlib import Path
from typing import AsyncGenerator
import asyncio
import threading

import numpy as np
import soundfile as sf

from .base import BaseTTS, TTSResult, Voice


# Voix disponibles (styles par défaut sans voice cloning)
AVAILABLE_VOICES = [
    Voice(id="default", name="Default (Neural)", language="multi", gender="Unknown"),
    Voice(id="cloned", name="Cloned Voice", language="multi", gender="Unknown"),
]


class OpenAudioProvider(BaseTTS):
    """
    Provider TTS utilisant OpenAudio S1-mini (Fish Speech).
    
    OpenAudio utilise un système de voice cloning : vous fournissez
    un échantillon audio de référence et sa transcription, et le modèle
    génère de la parole dans cette voix.
    
    Sans référence, le modèle utilise une voix neutre par défaut.
    
    Attributes:
        checkpoint_path: Chemin vers le dossier des checkpoints
        device: Device pour l'inférence ("cpu", "cuda", "mps")
        speaker_wav: Chemin vers l'audio de référence pour voice cloning
        speaker_text: Transcription de l'audio de référence
        _engine: Engine d'inférence TTS (chargé à la demande)
    
    Example:
        # Sans voice cloning
        tts = OpenAudioProvider()
        result = await tts.synthesize("Bonjour le monde !")
        
        # Avec voice cloning
        tts = OpenAudioProvider(
            speaker_wav=Path("reference.wav"),
            speaker_text="Bonjour, je suis la voix de référence."
        )
    """
    
    # Sample rate de OpenAudio (44.1kHz)
    SAMPLE_RATE = 44100
    
    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        device: str = "cpu",
        speaker_wav: str | Path | None = None,
        speaker_text: str | None = None,
        compile_model: bool = False,
        half_precision: bool = False,
    ):
        """
        Initialise le provider OpenAudio S1-mini.
        
        Args:
            checkpoint_path: Chemin vers les checkpoints OpenAudio S1-mini
                            (default: ~/models/openaudio-s1-mini)
            device: Device pour l'inférence ("cpu", "cuda", "mps")
                    Note: "cpu" est recommandé pour libérer le GPU pour le LLM
            speaker_wav: Chemin vers l'audio de référence pour voice cloning
            speaker_text: Transcription exacte de l'audio de référence
            compile_model: Compiler le modèle avec torch.compile (plus lent au démarrage)
            half_precision: Utiliser half precision (fp16/bf16) - recommandé sur GPU
        """
        # Configuration des chemins
        if checkpoint_path is None:
            checkpoint_path = Path.home() / "models" / "openaudio-s1-mini"
        self.checkpoint_path = Path(checkpoint_path)
        
        # Vérifier que les checkpoints existent
        self._validate_checkpoints()
        
        # Configuration device
        self.device = device
        self.compile_model = compile_model
        self.half_precision = half_precision
        
        # Configuration voice cloning
        self.speaker_wav = Path(speaker_wav) if speaker_wav else None
        self.speaker_text = speaker_text
        
        if self.speaker_wav and not self.speaker_wav.exists():
            raise FileNotFoundError(f"Speaker WAV not found: {self.speaker_wav}")
        
        # Engine chargé à la demande (lazy loading)
        self._engine = None
        self._lock = threading.Lock()
        
        # Paramètres de génération par défaut
        self.temperature = 0.8
        self.top_p = 0.8
        self.repetition_penalty = 1.1
        self.max_new_tokens = 1024
        
    def _validate_checkpoints(self) -> None:
        """Vérifie que les fichiers checkpoint sont présents."""
        required_files = ["model.pth", "codec.pth"]
        
        for filename in required_files:
            filepath = self.checkpoint_path / filename
            if not filepath.exists():
                raise FileNotFoundError(
                    f"Checkpoint manquant: {filepath}\n"
                    f"Téléchargez avec: huggingface-cli download fishaudio/openaudio-s1-mini "
                    f"--local-dir {self.checkpoint_path}"
                )
    
    def _load_engine(self):
        """
        Charge l'engine d'inférence OpenAudio (lazy loading).
        
        Cette méthode est thread-safe et ne charge le modèle qu'une fois.
        Le chargement peut prendre ~30s sur CPU.
        """
        if self._engine is not None:
            return self._engine
        
        with self._lock:
            # Double-check après avoir acquis le lock
            if self._engine is not None:
                return self._engine
            
            print(f"🔄 Chargement d'OpenAudio S1-mini (device={self.device})...")
            print("   ⏳ Cela peut prendre ~30 secondes sur CPU...")
            
            # Ajouter fish-speech au path si nécessaire
            fish_speech_path = Path.home() / "tools" / "fish-speech"
            if str(fish_speech_path) not in sys.path:
                sys.path.insert(0, str(fish_speech_path))
            
            # Forcer CPU si demandé (masquer CUDA)
            if self.device == "cpu":
                os.environ["CUDA_VISIBLE_DEVICES"] = ""
            
            import torch
            from fish_speech.inference_engine import TTSInferenceEngine
            from fish_speech.models.dac.inference import load_model as load_decoder_model
            from fish_speech.models.text2semantic.inference import launch_thread_safe_queue
            
            # Déterminer la précision
            if self.half_precision:
                precision = torch.float16
            else:
                precision = torch.bfloat16 if torch.cuda.is_available() else torch.float32
            
            # Charger le modèle LLM (text-to-semantic)
            llama_checkpoint = str(self.checkpoint_path)
            llama_queue = launch_thread_safe_queue(
                checkpoint_path=llama_checkpoint,
                device=self.device,
                precision=precision,
                compile=self.compile_model,
            )
            
            # Charger le décodeur audio (DAC)
            decoder_checkpoint = str(self.checkpoint_path / "codec.pth")
            decoder_model = load_decoder_model(
                config_name="modded_dac_vq",
                checkpoint_path=decoder_checkpoint,
                device=self.device,
            )
            
            # Créer l'engine d'inférence
            self._engine = TTSInferenceEngine(
                llama_queue=llama_queue,
                decoder_model=decoder_model,
                precision=precision,
                compile=self.compile_model,
            )
            
            # Récupérer le sample rate réel
            if hasattr(decoder_model, "spec_transform"):
                self.SAMPLE_RATE = decoder_model.spec_transform.sample_rate
            elif hasattr(decoder_model, "sample_rate"):
                self.SAMPLE_RATE = decoder_model.sample_rate
            
            print(f"✅ OpenAudio S1-mini chargé ! (sample_rate={self.SAMPLE_RATE}Hz)")
            
            return self._engine
    
    def _get_reference_audio(self) -> list:
        """
        Prépare l'audio de référence pour le voice cloning.
        
        Returns:
            Liste de ServeReferenceAudio ou liste vide si pas de référence
        """
        if self.speaker_wav is None or self.speaker_text is None:
            return []
        
        from fish_speech.utils.schema import ServeReferenceAudio
        
        with open(self.speaker_wav, "rb") as f:
            audio_bytes = f.read()
        
        return [ServeReferenceAudio(audio=audio_bytes, text=self.speaker_text)]
    
    async def synthesize(
        self,
        text: str,
        output_path: Path | None = None
    ) -> TTSResult:
        """
        Convertit du texte en fichier audio WAV.
        
        Supporte les tags d'émotion OpenAudio:
        - (excited) pour l'excitation
        - (whispering) pour chuchoter
        - (sad) pour la tristesse
        - (laughing) pour rire
        
        Args:
            text: Texte à synthétiser (peut contenir des tags d'émotion)
            output_path: Chemin de sortie (défaut: temp file)
            
        Returns:
            TTSResult avec le chemin du fichier audio
            
        Example:
            result = await tts.synthesize("Bonjour (excited) le monde !")
        """
        # L'inférence est synchrone, on l'exécute dans un thread
        loop = asyncio.get_event_loop()
        audio_data, sample_rate = await loop.run_in_executor(
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
        sf.write(str(output_path), audio_data, sample_rate)
        
        # Calculer la durée
        duration = len(audio_data) / sample_rate
        
        return TTSResult(audio_path=output_path, duration=duration)
    
    def _synthesize_sync(self, text: str) -> tuple[np.ndarray, int]:
        """
        Synthèse synchrone (appelée dans un thread).
        
        Returns:
            Tuple (audio_data, sample_rate)
        """
        from fish_speech.utils.schema import ServeTTSRequest
        
        engine = self._load_engine()
        
        # Préparer la requête
        request = ServeTTSRequest(
            text=text,
            references=self._get_reference_audio(),
            reference_id=None,
            max_new_tokens=self.max_new_tokens,
            chunk_length=200,
            top_p=self.top_p,
            repetition_penalty=self.repetition_penalty,
            temperature=self.temperature,
            streaming=False,
            format="wav",
        )
        
        # Exécuter l'inférence
        for result in engine.inference(request):
            if result.code == "final":
                sample_rate, audio = result.audio
                return audio, sample_rate
            elif result.code == "error":
                raise RuntimeError(f"Erreur OpenAudio: {result.error}")
        
        raise RuntimeError("Aucun audio généré")
    
    async def synthesize_stream(
        self,
        text: str
    ) -> AsyncGenerator[bytes, None]:
        """
        Génère l'audio en streaming (segment par segment).
        
        Permet de commencer la lecture avant que toute
        la synthèse soit terminée (latence réduite).
        
        Args:
            text: Texte à synthétiser
            
        Yields:
            Chunks audio en bytes (format WAV)
        """
        from fish_speech.utils.schema import ServeTTSRequest
        
        loop = asyncio.get_event_loop()
        
        def generate_segments():
            """Générateur synchrone de segments audio."""
            engine = self._load_engine()
            
            request = ServeTTSRequest(
                text=text,
                references=self._get_reference_audio(),
                reference_id=None,
                max_new_tokens=self.max_new_tokens,
                chunk_length=200,
                top_p=self.top_p,
                repetition_penalty=self.repetition_penalty,
                temperature=self.temperature,
                streaming=True,
                format="wav",
            )
            
            for result in engine.inference(request):
                if result.code == "segment":
                    sample_rate, audio = result.audio
                    yield audio, sample_rate
                elif result.code == "final":
                    sample_rate, audio = result.audio
                    yield audio, sample_rate
                elif result.code == "error":
                    raise RuntimeError(f"Erreur OpenAudio: {result.error}")
        
        # Collecter tous les segments (run_in_executor ne supporte pas les générateurs)
        segments = await loop.run_in_executor(None, list, generate_segments())
        
        # Yield chaque segment converti en WAV bytes
        for audio, sample_rate in segments:
            buffer = io.BytesIO()
            sf.write(buffer, audio, sample_rate, format='WAV')
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
        audio_data, sample_rate = await loop.run_in_executor(
            None,
            self._synthesize_sync,
            text
        )
        
        # Convertir en WAV bytes
        buffer = io.BytesIO()
        sf.write(buffer, audio_data, sample_rate, format='WAV')
        buffer.seek(0)
        return buffer.read()
    
    async def list_voices(self, language: str | None = None) -> list[Voice]:
        """
        Liste les voix disponibles.
        
        OpenAudio utilise le voice cloning, donc les "voix" sont
        définies par l'audio de référence, pas par des presets.
        
        Args:
            language: Ignoré (OpenAudio est multilingue)
            
        Returns:
            Liste des voix disponibles
        """
        voices = AVAILABLE_VOICES.copy()
        
        # Ajouter une voix personnalisée si configurée
        if self.speaker_wav:
            voices.append(Voice(
                id="custom",
                name=f"Custom ({self.speaker_wav.stem})",
                language="multi",
                gender="Unknown"
            ))
        
        return voices
    
    def set_voice(self, voice_id: str) -> None:
        """
        OpenAudio n'a pas de voix préréglées.
        
        Pour changer de voix, utilisez set_speaker() avec
        un nouvel audio de référence.
        """
        pass  # No-op car OpenAudio utilise voice cloning
    
    def set_speaker(
        self,
        speaker_wav: str | Path,
        speaker_text: str
    ) -> None:
        """
        Configure la voix de référence pour le voice cloning.
        
        Args:
            speaker_wav: Chemin vers l'audio de référence (10-30s recommandé)
            speaker_text: Transcription exacte de l'audio
            
        Example:
            tts.set_speaker(
                "reference.wav",
                "Bonjour, je suis une voix de référence claire et naturelle."
            )
        """
        speaker_path = Path(speaker_wav)
        if not speaker_path.exists():
            raise FileNotFoundError(f"Speaker WAV not found: {speaker_path}")
        
        self.speaker_wav = speaker_path
        self.speaker_text = speaker_text
    
    def set_temperature(self, temperature: float) -> None:
        """
        Change la température de génération.
        
        Plus haute = plus de variété/créativité
        Plus basse = plus de stabilité/cohérence
        
        Args:
            temperature: Valeur entre 0.1 et 1.0 (défaut: 0.8)
        """
        self.temperature = max(0.1, min(1.0, temperature))
    
    def set_rate(self, rate: str) -> None:
        """
        Change la vitesse de parole (non supporté directement).
        
        OpenAudio ne supporte pas le changement de vitesse.
        Cette méthode existe pour la compatibilité avec l'interface.
        
        Args:
            rate: Ignoré
        """
        # OpenAudio ne supporte pas le rate
        pass
    
    def set_pitch(self, pitch: str) -> None:
        """
        Change la hauteur de voix (non supporté).
        
        OpenAudio ne supporte pas le changement de pitch.
        Cette méthode existe pour la compatibilité avec l'interface.
        
        Args:
            pitch: Ignoré
        """
        # OpenAudio ne supporte pas le pitch
        pass


# Alias pour cohérence avec le reste du projet
OpenAudioS1Provider = OpenAudioProvider
