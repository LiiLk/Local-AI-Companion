#!/usr/bin/env python3
"""
Script to create a reference voice for TTS voice cloning.

TTS models like XTTS and F5-TTS use "voice cloning": they imitate the accent and timbre
of a reference audio. This script generates a reference audio with Kokoro TTS,
which can then be used by other TTS models.

Usage:
    python scripts/create_reference_voice.py
    
    # With a different Kokoro voice:
    python scripts/create_reference_voice.py --voice ff_siwis
    
    # To list available voices:
    python scripts/create_reference_voice.py --list-voices
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# Reference texts by language
REFERENCE_TEXTS = {
    "fr": """Bonjour, je suis une assistante intelligente et je suis ravie de vous aider.
J'adore discuter de sujets variés, que ce soit la technologie, la science, ou simplement la vie quotidienne.
N'hésitez pas à me poser toutes vos questions, je ferai de mon mieux pour y répondre avec précision et clarté.""",
    
    "en": """Hello, I'm an intelligent assistant and I'm delighted to help you.
I love discussing various topics, whether it's technology, science, or just everyday life.
Feel free to ask me any questions, I'll do my best to answer with precision and clarity.""",
    
    "ja": """こんにちは、私はインテリジェントアシスタントです。お手伝いできて嬉しいです。
テクノロジー、科学、日常生活など、さまざまなトピックについてお話しするのが大好きです。
何でもお気軽にご質問ください。正確かつ明確にお答えするよう努めます。"""
}

# Recommended Kokoro voices by language
RECOMMENDED_VOICES = {
    "fr": "ff_siwis",    # French female voice
    "en": "af_heart",    # American female voice
    "ja": "jf_alpha",    # Japanese female voice
}


async def create_reference_voice(
    voice: str = "ff_siwis",
    language: str = "fr",
    output_dir: Path = None,
    custom_text: str = None
) -> Path:
    """
    Create a reference audio file for TTS voice cloning.
    
    Args:
        voice: Kokoro voice to use
        language: Text language (fr, en, ja)
        output_dir: Output folder (default: ~/voices)
        custom_text: Custom text (optional)
        
    Returns:
        Path to the created audio file
    """
    from src.tts import KokoroProvider
    
    # Default output folder
    if output_dir is None:
        output_dir = Path.home() / "voices"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Reference text
    text = custom_text or REFERENCE_TEXTS.get(language, REFERENCE_TEXTS["en"])
    
    # File name
    output_path = output_dir / f"{language}_ref_{voice}.wav"
    
    print(f"🎤 Generating reference voice...")
    print(f"   Voice: {voice}")
    print(f"   Language: {language}")
    print(f"   Output: {output_path}")
    print()
    
    # Generate audio
    tts = KokoroProvider(voice=voice)
    await tts.synthesize(text.strip(), output_path)
    
    # File information
    size = output_path.stat().st_size
    print()
    print(f"✅ Reference created: {output_path}")
    print(f"   Size: {size / 1024:.1f} KB")
    print()
    print("📝 To use this voice with XTTS, add in config.yaml:")
    print()
    print("   xtts:")
    print(f'     speaker_wav: "{output_path}"')
    
    return output_path


def list_kokoro_voices():
    """List available Kokoro voices."""
    print("🎵 Available Kokoro voices:")
    print()
    print("  French:")
    print("    ff_siwis  - French female voice (recommended)")
    print()
    print("  English (US):")
    print("    af_heart  - American female voice")
    print("    af_bella  - American female voice")
    print("    am_adam   - American male voice")
    print()
    print("  English (UK):")
    print("    bf_emma   - British female voice")
    print("    bm_george - British male voice")
    print()
    print("  Japanese:")
    print("    jf_alpha  - Japanese female voice")
    print()
    print("Usage:")
    print("  python scripts/create_reference_voice.py --voice ff_siwis")


async def main():
    parser = argparse.ArgumentParser(
        description="Create a reference voice for TTS voice cloning"
    )
    parser.add_argument(
        "--voice", "-v",
        default="af_heart",
        help="Kokoro voice to use (default: af_heart)"
    )
    parser.add_argument(
        "--language", "-l",
        default="en",
        choices=["fr", "en", "ja"],
        help="Reference text language (default: en)"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        help="Output folder (default: ~/voices)"
    )
    parser.add_argument(
        "--text", "-t",
        help="Custom text to synthesize"
    )
    parser.add_argument(
        "--list-voices",
        action="store_true",
        help="List available Kokoro voices"
    )
    
    args = parser.parse_args()
    
    if args.list_voices:
        list_kokoro_voices()
        return
    
    await create_reference_voice(
        voice=args.voice,
        language=args.language,
        output_dir=args.output,
        custom_text=args.text
    )


if __name__ == "__main__":
    asyncio.run(main())
