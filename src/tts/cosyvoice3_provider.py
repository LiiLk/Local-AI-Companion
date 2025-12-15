"""
CosyVoice3 TTS Provider - HTTP client for CosyVoice3 server.

CosyVoice3-0.5B is a state-of-the-art TTS system with:
- 9 languages including French (EN, ZH, JA, KO, DE, ES, FR, IT, RU)
- Zero-shot voice cloning with minimal reference audio
- Streaming support with ~150ms latency
- Only 0.5B parameters (~2GB VRAM)
- Apache 2.0 license (commercial use OK)

Architecture:
    The CosyVoice3 model requires Python 3.10 and specific dependencies that
    conflict with the main project. Therefore, it runs as a separate server
    (scripts/cosyvoice3_server.py) in its own conda environment.

    This provider communicates with the server via HTTP.

Setup:
    1. Start the CosyVoice3 server:
       ./scripts/start_cosyvoice3.sh start

    2. Configure in config.yaml:
       tts:
         provider: "cosyvoice3"
         cosyvoice3:
           api_url: "http://127.0.0.1:9881"
           ref_audio_path: "resources/voices/your_voice.wav"

Usage:
    provider = CosyVoice3Provider(config)
    await provider.initialize()
    result = await provider.synthesize("Bonjour!", Path("output.wav"))
"""

import logging
from pathlib import Path
from typing import AsyncGenerator

import httpx

from .base import BaseTTS, TTSResult, Voice

logger = logging.getLogger(__name__)


class CosyVoice3Provider(BaseTTS):
    """
    TTS Provider using CosyVoice3 via HTTP API.

    CosyVoice3 offers:
    - 9 languages with native support
    - Zero-shot voice cloning
    - Streaming with 150ms latency
    - SOTA quality for 0.5B parameters
    - Only ~2GB VRAM required

    The provider connects to a CosyVoice3 server running in a separate
    conda environment (Python 3.10).

    Example:
        # Using config dict (from config.yaml)
        config = {"cosyvoice3": {"api_url": "http://127.0.0.1:9881"}}
        tts = CosyVoice3Provider(config)
        await tts.initialize()
        await tts.synthesize("Bonjour!", Path("output.wav"))

        # Direct initialization
        tts = CosyVoice3Provider(
            api_url="http://127.0.0.1:9881",
            ref_audio_path="resources/voices/my_voice.wav",
            language="fr"
        )
    """

    # Supported languages
    SUPPORTED_LANGUAGES = ["en", "zh", "ja", "ko", "de", "es", "fr", "it", "ru"]

    def __init__(
        self,
        config: dict | None = None,
        api_url: str = "http://127.0.0.1:9881",
        ref_audio_path: str | Path | None = None,
        prompt_text: str = "",
        language: str = "fr",
        speed: float = 1.0,
    ):
        """
        Initialize CosyVoice3 provider.

        Args:
            config: Configuration dict (from config.yaml). If provided,
                   other args are used as fallbacks.
            api_url: URL of the CosyVoice3 server
            ref_audio_path: Path to reference audio for voice cloning
            prompt_text: Transcription of reference audio (for better cloning)
            language: Default language code (fr, en, ja, etc.)
            speed: Speech speed factor (default 1.0)
        """
        # Extract config if provided
        if config:
            cv3_config = config.get("cosyvoice3", {})
            api_url = cv3_config.get("api_url", api_url)
            ref_audio_path = cv3_config.get("ref_audio_path", ref_audio_path)
            prompt_text = cv3_config.get("prompt_text", prompt_text)
            language = cv3_config.get("language", language)
            speed = cv3_config.get("speed", speed)

        self.api_url = api_url.rstrip("/")
        self.ref_audio_path = Path(ref_audio_path).expanduser() if ref_audio_path else None
        self.prompt_text = prompt_text
        self.language = language
        self.speed = speed

        # HTTP client (initialized lazily)
        self.client: httpx.AsyncClient | None = None

        # Server info cache
        self._server_info: dict | None = None

        logger.info(
            f"CosyVoice3 provider initialized "
            f"(api={self.api_url}, lang={self.language}, "
            f"zero_shot={'yes' if prompt_text else 'no'})"
        )

    @property
    def model_name(self) -> str:
        """Model name for display."""
        return "CosyVoice3-0.5B"

    async def initialize(self) -> None:
        """Initialize HTTP client and verify server connection."""
        if self.client is None:
            self.client = httpx.AsyncClient(timeout=120.0)

        # Test connection
        try:
            response = await self.client.get(f"{self.api_url}/health")
            if response.status_code == 200:
                logger.info("CosyVoice3 server is healthy")
            else:
                logger.warning(f"CosyVoice3 server returned: {response.status_code}")
        except httpx.ConnectError:
            logger.warning(
                f"Cannot connect to CosyVoice3 server at {self.api_url}. "
                "Start it with: ./scripts/start_cosyvoice3.sh start"
            )
        except Exception as e:
            logger.warning(f"Error connecting to CosyVoice3 server: {e}")

    async def close(self) -> None:
        """Close HTTP client."""
        if self.client:
            await self.client.aclose()
            self.client = None

    def set_language(self, language: str) -> None:
        """
        Change the synthesis language.

        Args:
            language: Language code (fr, en, ja, ko, de, es, it, ru, zh)
        """
        if language not in self.SUPPORTED_LANGUAGES:
            logger.warning(
                f"Language '{language}' may not be fully supported. "
                f"Supported: {self.SUPPORTED_LANGUAGES}"
            )
        self.language = language
        logger.debug(f"CosyVoice3 language set to: {language}")

    def set_reference_audio(self, path: str | Path) -> None:
        """
        Set reference audio for voice cloning.

        Args:
            path: Path to reference audio file (WAV format recommended)
        """
        self.ref_audio_path = Path(path).expanduser()
        if not self.ref_audio_path.exists():
            logger.warning(f"Reference audio not found: {self.ref_audio_path}")
        else:
            logger.info(f"Reference audio set to: {self.ref_audio_path}")

    async def synthesize(
        self,
        text: str,
        output_path: Path | None = None
    ) -> TTSResult:
        """
        Convert text to WAV audio.

        Args:
            text: Text to synthesize
            output_path: Output path (optional, creates temp file if None)

        Returns:
            TTSResult with the audio file path and/or data
        """
        if not self.client:
            await self.initialize()

        if not text.strip():
            return TTSResult(audio_data=b"")

        try:
            # Prepare request
            form_data = {
                "text": text,
                "language": self.language,
                "speed": str(self.speed),
            }

            # Use reference audio path on server if available
            if self.ref_audio_path and self.ref_audio_path.exists():
                form_data["ref_audio_path"] = str(self.ref_audio_path)
                
                # Add prompt_text for better zero-shot cloning
                if self.prompt_text:
                    form_data["prompt_text"] = self.prompt_text
                
                endpoint = f"{self.api_url}/synthesize_with_ref"
            else:
                endpoint = f"{self.api_url}/synthesize"

            logger.debug(f"Synthesizing: '{text[:50]}...' via {endpoint}")

            # Make request
            response = await self.client.post(endpoint, data=form_data)

            if response.status_code != 200:
                error_msg = response.text
                logger.error(f"CosyVoice3 error: {error_msg}")
                raise RuntimeError(f"CosyVoice3 synthesis failed: {error_msg}")

            audio_data = response.content

            # Get duration from header if available
            duration = None
            if "X-Audio-Duration" in response.headers:
                duration = float(response.headers["X-Audio-Duration"])

            # Save to file if path provided
            if output_path:
                output_path = Path(output_path)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_bytes(audio_data)
                logger.debug(f"Audio saved to: {output_path}")
                return TTSResult(
                    audio_path=output_path,
                    audio_data=audio_data,
                    duration=duration
                )

            return TTSResult(audio_data=audio_data, duration=duration)

        except httpx.ConnectError:
            logger.error(
                f"Cannot connect to CosyVoice3 server. "
                "Start it with: ./scripts/start_cosyvoice3.sh start"
            )
            raise RuntimeError("CosyVoice3 server not available")
        except Exception as e:
            logger.error(f"CosyVoice3 synthesis failed: {e}")
            raise

    async def synthesize_stream(
        self,
        text: str
    ) -> AsyncGenerator[bytes, None]:
        """
        Generate audio in streaming mode.

        Note: Current implementation generates full audio then streams it.
        True streaming will be added when the server supports it.

        Args:
            text: Text to synthesize

        Yields:
            Audio chunks in bytes (WAV format)
        """
        if not self.client:
            await self.initialize()

        if not text.strip():
            return

        try:
            # For now, generate full audio and stream it
            # TODO: Implement true streaming when server supports it
            result = await self.synthesize(text)

            if result.audio_data:
                # Stream in chunks
                chunk_size = 4096
                data = result.audio_data

                for i in range(0, len(data), chunk_size):
                    yield data[i:i + chunk_size]

        except Exception as e:
            logger.error(f"CosyVoice3 streaming failed: {e}")
            raise

    async def synthesize_to_bytes(self, text: str) -> bytes:
        """
        Convert text directly to audio bytes.

        Args:
            text: Text to synthesize

        Returns:
            Audio in bytes (WAV format)
        """
        result = await self.synthesize(text)
        return result.audio_data or b""

    async def list_voices(self, language: str | None = None) -> list[Voice]:  # noqa: ARG002
        """
        List available voices.

        CosyVoice3 uses zero-shot cloning, so voices are defined by
        reference audio files, not built-in speaker names.

        Returns:
            List with current reference info (if any)
        """
        voices = []

        if self.ref_audio_path and self.ref_audio_path.exists():
            voices.append(Voice(
                id=str(self.ref_audio_path),
                name=f"Voice Clone ({self.ref_audio_path.stem})",
                language="multilingual",
                gender="Unknown"
            ))

        return voices

    def set_voice(self, voice_id: str) -> None:
        """
        Set reference audio for voice cloning.

        Args:
            voice_id: Path to reference audio file
        """
        self.set_reference_audio(voice_id)

    def set_rate(self, rate: str) -> None:
        """
        Set speech speed.

        Args:
            rate: Speed modifier ("+20%", "-10%", etc.) or float string
        """
        try:
            if rate.endswith("%"):
                percentage = float(rate.rstrip("%"))
                self.speed = 1.0 + (percentage / 100.0)
            else:
                self.speed = float(rate)
            logger.info(f"Speed set to: {self.speed}")
        except ValueError:
            logger.warning(f"Invalid rate: {rate}")

    def set_pitch(self, pitch: str) -> None:
        """
        Not supported by CosyVoice3.
        Pitch is determined by the reference audio.
        """
        logger.debug("CosyVoice3 doesn't support pitch modification")

    @staticmethod
    def list_languages() -> list[str]:
        """
        List supported languages.

        Returns:
            List of language codes
        """
        return CosyVoice3Provider.SUPPORTED_LANGUAGES.copy()

    async def get_server_info(self) -> dict | None:
        """
        Get information about the CosyVoice3 server.

        Returns:
            Server info dict or None if not available
        """
        if not self.client:
            await self.initialize()

        try:
            response = await self.client.get(f"{self.api_url}/")
            if response.status_code == 200:
                self._server_info = response.json()
                return self._server_info
        except Exception as e:
            logger.warning(f"Could not get server info: {e}")

        return None

    def __repr__(self) -> str:
        return (
            f"CosyVoice3Provider(api={self.api_url}, "
            f"lang={self.language}, ref={self.ref_audio_path})"
        )


# Convenience function for testing
async def test_cosyvoice3():
    """Test CosyVoice3 provider."""
    logging.basicConfig(level=logging.INFO)

    provider = CosyVoice3Provider(
        api_url="http://127.0.0.1:9881",
        ref_audio_path="resources/voices/juri_neutral.wav",
        language="fr"
    )

    await provider.initialize()

    try:
        # Check server info
        info = await provider.get_server_info()
        print(f"Server info: {info}")

        # Test synthesis
        result = await provider.synthesize(
            "Bonjour! Je suis ton assistante virtuelle.",
            Path("test_cosyvoice3_output.wav")
        )
        print(f"Audio saved to: {result.audio_path}")
        print(f"Duration: {result.duration:.2f}s" if result.duration else "Duration unknown")

    finally:
        await provider.close()


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_cosyvoice3())
