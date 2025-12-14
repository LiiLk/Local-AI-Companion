"""
GPT-SoVITS TTS Provider

High-quality voice cloning TTS using GPT-SoVITS API.

Features:
- Zero-shot voice cloning with ~5 seconds of reference audio
- Support for v1, v2, v3 (24kHz), v4 (48kHz), v2Pro, v2ProPlus models
- Multi-language: Chinese, English, Japanese, Korean, Cantonese, French
- Streaming support
- Very high quality voice reproduction

Requirements:
- GPT-SoVITS server running (python api.py or api_v2.py)
- Default port: 9880

Usage:
    provider = GPTSoVITSProvider(config)
    await provider.initialize()
    result = await provider.synthesize("Hello world!", Path("output.wav"))
"""

import logging
import httpx
from pathlib import Path
from typing import AsyncGenerator
from io import BytesIO

from .base import BaseTTS, TTSResult, Voice

logger = logging.getLogger(__name__)


class GPTSoVITSProvider(BaseTTS):
    """
    GPT-SoVITS TTS provider using the HTTP API.
    
    The provider connects to a running GPT-SoVITS server (api.py or api_v2.py)
    and sends synthesis requests with reference audio for voice cloning.
    
    Config options:
        api_url: Base URL for GPT-SoVITS API (default: http://127.0.0.1:9880)
        ref_audio_path: Path to reference audio for voice cloning
        ref_text: Text spoken in the reference audio (improves quality)
        ref_language: Language of reference audio (zh, en, ja, ko, yue, fr)
        language: Target synthesis language (zh, en, ja, ko, yue, fr)
        speed: Speech speed factor (default: 1.0)
        top_k: Top-K sampling (default: 15)
        top_p: Top-P sampling (default: 0.6)
        temperature: Sampling temperature (default: 0.6)
        sample_steps: VITS sampling steps for v3/v4 (default: 32)
    """
    
    # Language code mapping
    LANGUAGE_MAP = {
        "fr": "fr",    # French (supported in newer versions)
        "en": "en",    # English
        "zh": "zh",    # Chinese
        "ja": "ja",    # Japanese
        "ko": "ko",    # Korean
        "yue": "yue",  # Cantonese
        "auto": "auto"
    }
    
    def __init__(self, config: dict):
        """
        Initialize GPT-SoVITS provider.
        
        Args:
            config: TTS configuration from config.yaml
        """
        self.config = config
        sovits_config = config.get("gpt_sovits", {})
        
        # API configuration
        self.api_url = sovits_config.get("api_url", "http://127.0.0.1:9880")
        self.api_version = sovits_config.get("api_version", "v1")  # v1 or v2
        
        # Reference audio configuration (for voice cloning)
        self.ref_audio_path = sovits_config.get("ref_audio_path", "")
        self.ref_text = sovits_config.get("ref_text", "")
        self.ref_language = sovits_config.get("ref_language", "ja")
        
        # Synthesis parameters
        self.language = sovits_config.get("language", "fr")
        self.speed = sovits_config.get("speed", 1.0)
        self.top_k = sovits_config.get("top_k", 15)
        self.top_p = sovits_config.get("top_p", 0.6)
        self.temperature = sovits_config.get("temperature", 0.6)
        self.sample_steps = sovits_config.get("sample_steps", 32)
        self.cut_punc = sovits_config.get("cut_punc", "")
        
        # HTTP client
        self.client: httpx.AsyncClient | None = None
        
        logger.info(f"GPT-SoVITS provider initialized (API: {self.api_url})")
    
    async def initialize(self) -> None:
        """Initialize HTTP client and verify server connection."""
        self.client = httpx.AsyncClient(timeout=120.0)  # Long timeout for TTS
        
        # Test connection
        try:
            response = await self.client.get(f"{self.api_url}/")
            # GPT-SoVITS returns 400 if no parameters, which is expected
            if response.status_code in [200, 400]:
                logger.info("GPT-SoVITS server is reachable")
            else:
                logger.warning(f"GPT-SoVITS server returned unexpected status: {response.status_code}")
        except Exception as e:
            logger.warning(f"Could not connect to GPT-SoVITS server: {e}")
            logger.warning("Make sure GPT-SoVITS server is running: python api.py")
    
    async def close(self) -> None:
        """Close HTTP client."""
        if self.client:
            await self.client.aclose()
            self.client = None
    
    async def synthesize(
        self,
        text: str,
        output_path: Path | None = None
    ) -> TTSResult:
        """
        Synthesize text to speech using GPT-SoVITS.
        
        Args:
            text: Text to synthesize
            output_path: Optional path to save the audio file
            
        Returns:
            TTSResult with audio data or path
        """
        if not self.client:
            await self.initialize()
        
        if not text.strip():
            logger.warning("Empty text provided for synthesis")
            return TTSResult(audio_data=b"")
        
        try:
            # Build request based on API version
            if self.api_version == "v2":
                audio_data = await self._synthesize_v2(text)
            else:
                audio_data = await self._synthesize_v1(text)
            
            if output_path:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_bytes(audio_data)
                logger.info(f"Audio saved to {output_path}")
                return TTSResult(audio_path=output_path, audio_data=audio_data)
            
            return TTSResult(audio_data=audio_data)
            
        except httpx.ConnectError:
            logger.error("Cannot connect to GPT-SoVITS server. Is it running?")
            raise RuntimeError("GPT-SoVITS server not available")
        except Exception as e:
            logger.error(f"GPT-SoVITS synthesis failed: {e}")
            raise
    
    async def _synthesize_v1(self, text: str) -> bytes:
        """
        Synthesize using API v1 (api.py).
        
        Uses GET or POST to /
        """
        # Prepare parameters
        params = {
            "text": text,
            "text_language": self._map_language(self.language),
            "top_k": self.top_k,
            "top_p": self.top_p,
            "temperature": self.temperature,
            "speed": self.speed,
            "sample_steps": self.sample_steps,
        }
        
        # Add reference audio if configured
        if self.ref_audio_path:
            params["refer_wav_path"] = self.ref_audio_path
            params["prompt_text"] = self.ref_text
            params["prompt_language"] = self._map_language(self.ref_language)
        
        if self.cut_punc:
            params["cut_punc"] = self.cut_punc
        
        logger.debug(f"GPT-SoVITS v1 request: {params}")
        
        # Use POST for complex requests
        response = await self.client.post(
            f"{self.api_url}/",
            json=params
        )
        
        if response.status_code != 200:
            error_msg = response.text
            logger.error(f"GPT-SoVITS error: {error_msg}")
            raise RuntimeError(f"GPT-SoVITS synthesis failed: {error_msg}")
        
        return response.content
    
    async def _synthesize_v2(self, text: str) -> bytes:
        """
        Synthesize using API v2 (api_v2.py).
        
        Uses POST to /tts with more parameters.
        """
        # Prepare request body
        request_body = {
            "text": text,
            "text_lang": self._map_language(self.language),
            "ref_audio_path": self.ref_audio_path,
            "prompt_text": self.ref_text,
            "prompt_lang": self._map_language(self.ref_language),
            "top_k": self.top_k,
            "top_p": self.top_p,
            "temperature": self.temperature,
            "speed_factor": self.speed,
            "sample_steps": self.sample_steps,
            "media_type": "wav",
            "streaming_mode": False,
            "batch_size": 1,
            "parallel_infer": True,
        }
        
        logger.debug(f"GPT-SoVITS v2 request: text='{text[:50]}...'")
        
        response = await self.client.post(
            f"{self.api_url}/tts",
            json=request_body
        )
        
        if response.status_code != 200:
            error_msg = response.text
            logger.error(f"GPT-SoVITS v2 error: {error_msg}")
            raise RuntimeError(f"GPT-SoVITS synthesis failed: {error_msg}")
        
        return response.content
    
    async def synthesize_stream(
        self,
        text: str
    ) -> AsyncGenerator[bytes, None]:
        """
        Stream audio synthesis (if supported by server).
        
        Note: Streaming requires api_v2.py with streaming_mode enabled.
        Falls back to non-streaming if not available.
        """
        if not self.client:
            await self.initialize()
        
        if not text.strip():
            return
        
        if self.api_version == "v2":
            # Use streaming mode
            request_body = {
                "text": text,
                "text_lang": self._map_language(self.language),
                "ref_audio_path": self.ref_audio_path,
                "prompt_text": self.ref_text,
                "prompt_lang": self._map_language(self.ref_language),
                "top_k": self.top_k,
                "top_p": self.top_p,
                "temperature": self.temperature,
                "speed_factor": self.speed,
                "sample_steps": self.sample_steps,
                "media_type": "wav",
                "streaming_mode": 2,  # Medium quality streaming
                "batch_size": 1,
            }
            
            try:
                async with self.client.stream(
                    "POST",
                    f"{self.api_url}/tts",
                    json=request_body
                ) as response:
                    if response.status_code != 200:
                        error = await response.aread()
                        raise RuntimeError(f"Streaming failed: {error}")
                    
                    async for chunk in response.aiter_bytes(chunk_size=4096):
                        yield chunk
                        
            except Exception as e:
                logger.warning(f"Streaming failed, falling back to non-streaming: {e}")
                # Fallback to non-streaming
                audio_data = await self._synthesize_v2(text)
                yield audio_data
        else:
            # v1 doesn't support true streaming, return full audio
            audio_data = await self._synthesize_v1(text)
            yield audio_data
    
    async def list_voices(self, language: str | None = None) -> list[Voice]:
        """
        List available voices.
        
        GPT-SoVITS uses reference audio for voice cloning,
        so this returns information about the current configuration.
        """
        voices = []
        
        if self.ref_audio_path:
            # Voice cloning mode
            voices.append(Voice(
                id="clone",
                name=f"Voice Clone ({Path(self.ref_audio_path).stem})",
                language=self.ref_language,
                gender="Unknown"
            ))
        
        # Could query server for available preset voices if supported
        return voices
    
    def set_voice(self, voice_id: str) -> None:
        """
        Set the reference audio for voice cloning.
        
        Args:
            voice_id: Path to reference audio file
        """
        if Path(voice_id).exists():
            self.ref_audio_path = voice_id
            logger.info(f"Reference audio set to: {voice_id}")
        else:
            logger.warning(f"Reference audio not found: {voice_id}")
    
    def set_reference(
        self,
        audio_path: str,
        text: str = "",
        language: str = "ja"
    ) -> None:
        """
        Set reference audio with text and language.
        
        Args:
            audio_path: Path to reference audio
            text: Text spoken in the audio (improves quality)
            language: Language of the reference audio
        """
        self.ref_audio_path = audio_path
        self.ref_text = text
        self.ref_language = language
        logger.info(f"Reference set: {audio_path} ({language})")
    
    def set_language(self, language: str) -> None:
        """
        Set the target synthesis language.
        
        Args:
            language: Language code (fr, en, zh, ja, ko, yue)
        """
        if language.lower() in self.LANGUAGE_MAP:
            self.language = language.lower()
            logger.info(f"Synthesis language set to: {self.language}")
        else:
            logger.warning(f"Unknown language: {language}, keeping {self.language}")
    
    def set_rate(self, rate: str) -> None:
        """
        Set speech speed.
        
        Args:
            rate: Speed modifier ("+20%", "-10%", etc.) or float
        """
        try:
            if rate.endswith("%"):
                # Convert percentage to factor
                percentage = float(rate.rstrip("%"))
                self.speed = 1.0 + (percentage / 100.0)
            else:
                self.speed = float(rate)
            logger.info(f"Speed set to: {self.speed}")
        except ValueError:
            logger.warning(f"Invalid rate: {rate}")
    
    def set_pitch(self, pitch: str) -> None:
        """
        Set voice pitch.
        
        Note: GPT-SoVITS doesn't support pitch modification directly.
        This is a no-op for API compatibility.
        """
        logger.debug("GPT-SoVITS doesn't support pitch modification")
    
    def _map_language(self, lang: str) -> str:
        """Map language code to GPT-SoVITS format."""
        return self.LANGUAGE_MAP.get(lang.lower(), lang.lower())
    
    def __repr__(self) -> str:
        return (
            f"GPTSoVITSProvider(api={self.api_url}, "
            f"lang={self.language}, ref={self.ref_audio_path})"
        )


async def test_gpt_sovits():
    """Test GPT-SoVITS provider."""
    import asyncio
    
    config = {
        "gpt_sovits": {
            "api_url": "http://127.0.0.1:9880",
            "api_version": "v1",
            "ref_audio_path": "resources/voices/juri_neutral.wav",
            "ref_text": "",  # Leave empty if unknown
            "ref_language": "ja",
            "language": "fr",
            "speed": 1.0,
            "top_k": 15,
            "top_p": 0.6,
            "temperature": 0.6,
        }
    }
    
    provider = GPTSoVITSProvider(config)
    await provider.initialize()
    
    try:
        # Test synthesis
        result = await provider.synthesize(
            "Bonjour! Je suis ton assistante virtuelle.",
            Path("test_gpt_sovits.wav")
        )
        print(f"Audio saved to: {result.audio_path}")
        print(f"Audio size: {len(result.audio_data)} bytes")
    finally:
        await provider.close()


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_gpt_sovits())
