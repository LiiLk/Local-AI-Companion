import asyncio
import io
import re
from pathlib import Path
from typing import AsyncGenerator

import soundfile as sf
import numpy as np

# Try importing F5TTS locally or from venv
try:
    from f5_tts.api import F5TTS
except ImportError:
    print("Warning: F5TTS not found. Make sure f5-tts is installed.")
    F5TTS = None

from .base import BaseTTS, TTSResult, Voice

class F5TTSProvider(BaseTTS):
    """
    TTS provider using F5-TTS (Zero-shot Voice Cloning).
    Supports emotion control via Reference Audio Switching.
    """
    
    def __init__(self, device: str = None):
        self.device = device
        self._model = None
        self.sample_rate = 24000 # F5 default
        
        # Reference definitions
        self.ref_dir = Path("resources/voices/f5_refs")
        self.refs = {
            "neutral": {
                "file": self.ref_dir / "juri_neutral.wav",
                "text": "My kicks will make you feel real good."
            },
            "neutral_fr": {
                "file": self.ref_dir / "juri_neutral_fr.wav",
                "text": "Salut, je suis Juri. Prête pour le combat ? Ça va faire mal."
            },
            "happy": {
                "file": self.ref_dir / "juri_happy.wav",
                "text": ""
            },
            "sad": {
                "file": self.ref_dir / "juri_sad.wav",
                "text": ""
            },
            "angry": {
                "file": self.ref_dir / "juri_angry.wav",
                "text": ""
            }
        }
        
        # Default voice/emotion
        self.current_emotion = "neutral"
        self.current_language = "en"

    def _load_model(self):
        if self._model is None:
            print("🔄 Loading F5-TTS...")
            if F5TTS is None:
                raise RuntimeError("F5TTS package not installed.")
            
            # Initialize with default settings
            self._model = F5TTS(
                model="F5TTS_Base",
                device=self.device
            )
            print("✅ F5-TTS loaded!")
        return self._model

    def _detect_emotion(self, text: str) -> tuple[str, str]:
        """
        Detect emotion tag in text and return (emotion, cleaned_text).
        Tags: [happy], [sad], [angry], [neutral] (also French equivalents)
        """
        text_lower = text.lower()
        
        # Map common tags to keys
        tag_map = {
            "[happy]": "happy", "[joie]": "happy", "[joyeux]": "happy",
            "[sad]": "sad", "[triste]": "sad",
            "[angry]": "angry", "[colere]": "angry", "[fache]": "angry",
            "[neutral]": "neutral", "[neutre]": "neutral"
        }
        
        detected_emotion = self.current_emotion # Default to current set emotion
        
        for tag, emotion in tag_map.items():
            if tag in text_lower:
                detected_emotion = emotion
                # Remove tag (case insensitive replacement)
                text = re.sub(re.escape(tag), "", text, flags=re.IGNORECASE)
        
        return detected_emotion, text.strip()

    async def synthesize(self, text: str, output_path: Path | None = None) -> TTSResult:
        loop = asyncio.get_event_loop()
        
        # Detect emotion
        emotion, clean_text = self._detect_emotion(text)
        if not clean_text:
            return TTSResult(audio_data=b"")

        # Select reference based on Language AND Emotion
        ref_key = emotion
        
        # If we are in French mode, try to find a French version of the emotion
        # Fallback to "neutral_fr" if specific emotion not found, then to English neutral
        if self.current_language == "fr":
            if emotion == "neutral":
                ref_key = "neutral_fr"
            # You could add "happy_fr", "angry_fr" etc. later
        
        ref = self.refs.get(ref_key, self.refs.get("neutral_fr" if self.current_language == "fr" else "neutral"))
        
        # Check if ref file exists
        if not ref["file"].exists():
            print(f"⚠️ Reference file not found: {ref['file']}, using neutral")
            ref = self.refs["neutral"]

        print(f"🎤 F5-TTS Generating ({emotion}, lang={self.current_language}): '{clean_text}'")
        
        # Run inference in executor (it's sync)
        wav, sr, spec = await loop.run_in_executor(
            None,
            self._infer_sync,
            clean_text,
            str(ref["file"]),
            ref["text"]
        )
        
        # Return result
        buffer = io.BytesIO()
        sf.write(buffer, wav, sr, format='WAV')
        buffer.seek(0)
        
        if output_path:
            with open(output_path, "wb") as f:
                f.write(buffer.read())
            return TTSResult(audio_path=output_path, duration=len(wav)/sr)
        else:
            return TTSResult(audio_data=buffer.read(), duration=len(wav)/sr)

    def _infer_sync(self, text, ref_file, ref_text):
        model = self._load_model()
        wav, sr, spec = model.infer(
            ref_file=ref_file,
            ref_text=ref_text,
            gen_text=text,
            file_wave=None,
            file_spec=None,
            seed=-1 # Random seed
        )
        return wav, sr, spec

    async def synthesize_stream(self, text: str) -> AsyncGenerator[bytes, None]:
        # F5-TTS doesn't support real streaming yet, so we generate full and yield
        result = await self.synthesize(text)
        if result.audio_data:
            yield result.audio_data

    async def list_voices(self, language: str | None = None) -> list[Voice]:
        # We only have one "voice" (Aria) but with multiple emotions
        return [Voice(id="f5_aria", name="Aria (F5-TTS)", language="fr", gender="Female")]

    def set_voice(self, voice_id: str) -> None:
        """
        Switch language/voice based on ID.
        Called by the system when language changes.
        """
        if "fr" in voice_id.lower():
            self.current_language = "fr"
            print("🇫🇷 F5-TTS: Switched to French reference")
        else:
            self.current_language = "en"
            print("🇺🇸 F5-TTS: Switched to English reference")

    def set_rate(self, rate: str) -> None:
        pass # Not implemented

    def set_pitch(self, pitch: str) -> None:
        pass # Not implemented
