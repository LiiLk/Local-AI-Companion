"""
Generate a voice reference file for MiniCPM-o voice cloning.

This script uses Edge TTS to create a reference audio that the omni model
can use for voice cloning. For best results, use actual character voice
samples from the game instead.

Usage:
    python scripts/generate_voice_reference.py --character march7th
    python scripts/generate_voice_reference.py --character march7th --voice en-US-AnaNeural
"""

import argparse
import asyncio
from pathlib import Path

# Sample texts for voice reference (varied to capture voice characteristics)
REFERENCE_TEXTS = {
    "march7th": """
        Hi there, Trailblazer! I'm so excited to meet you!
        Oh wow, this is amazing! I absolutely have to take a photo of this moment!
        Don't worry, I'll always be here to help you out. That's what friends are for, right?
        Sometimes I wonder about my past, but every new memory we make together means so much to me.
        Let's go on an adventure! I can't wait to see what's out there!
    """,
    "default": """
        Hello! I'm here to help you with anything you need.
        Let me think about that for a moment. That's a great question!
        I really enjoy our conversations. They're always so interesting!
        Please don't hesitate to ask if you need anything else.
        Together, we can figure this out. I believe in you!
    """,
}

# Recommended voices for characters (young, energetic female voices)
RECOMMENDED_VOICES = {
    "march7th": "en-US-AnaNeural",  # Young, cheerful female voice
    "default": "en-US-JennyNeural",
}


async def generate_reference(character: str, voice: str, output_path: Path):
    """Generate voice reference using Edge TTS."""
    import edge_tts

    text = REFERENCE_TEXTS.get(character, REFERENCE_TEXTS["default"])
    # Clean up text
    text = " ".join(text.split())

    print(f"Generating voice reference for '{character}'...")
    print(f"Voice: {voice}")
    print(f"Output: {output_path}")

    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(str(output_path))

    print(f"✅ Voice reference saved: {output_path}")
    print(f"   Duration: ~{len(text.split()) / 2.5:.1f}s (estimated)")
    print()
    print("To enable voice cloning, update your character config:")
    print(f'  omni_ref_audio: "{output_path.relative_to(Path.cwd())}"')


def main():
    parser = argparse.ArgumentParser(description="Generate voice reference for MiniCPM-o")
    parser.add_argument(
        "--character", "-c",
        default="march7th",
        help="Character name (default: march7th)"
    )
    parser.add_argument(
        "--voice", "-v",
        default=None,
        help="Edge TTS voice (default: auto-select based on character)"
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output path (default: resources/voices/<character>/reference.wav)"
    )

    args = parser.parse_args()

    # Determine voice
    voice = args.voice or RECOMMENDED_VOICES.get(args.character, RECOMMENDED_VOICES["default"])

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path("resources/voices") / args.character / "reference.wav"

    # Create directory
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Generate
    asyncio.run(generate_reference(args.character, voice, output_path))


if __name__ == "__main__":
    main()
