#!/usr/bin/env python3
"""
Script pour créer une voix de référence française pour F5-TTS.

F5-TTS utilise le "voice cloning" : il imite l'accent et le timbre
d'un audio de référence. Sans référence française, il parle avec
un accent anglais !

Ce script génère un audio de référence français avec Kokoro TTS,
que F5-TTS pourra ensuite imiter.

Usage:
    python scripts/create_french_voice.py
    
    # Ou avec une voix Kokoro différente:
    python scripts/create_french_voice.py --voice ff_siwis
    
    # Pour lister les voix disponibles:
    python scripts/create_french_voice.py --list-voices
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Ajouter le répertoire parent au path
sys.path.insert(0, str(Path(__file__).parent.parent))


# Textes de référence par langue
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

# Voix Kokoro recommandées par langue
RECOMMENDED_VOICES = {
    "fr": "ff_siwis",    # Voix française féminine
    "en": "af_heart",    # Voix américaine féminine
    "ja": "jf_alpha",    # Voix japonaise féminine
}


async def create_reference_voice(
    voice: str = "ff_siwis",
    language: str = "fr",
    output_dir: Path = None,
    custom_text: str = None
) -> Path:
    """
    Crée un fichier audio de référence pour F5-TTS.
    
    Args:
        voice: Voix Kokoro à utiliser
        language: Langue du texte (fr, en, ja)
        output_dir: Dossier de sortie (défaut: ~/voices)
        custom_text: Texte personnalisé (optionnel)
        
    Returns:
        Path vers le fichier audio créé
    """
    from src.tts import KokoroProvider
    
    # Dossier de sortie par défaut
    if output_dir is None:
        output_dir = Path.home() / "voices"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Texte de référence
    text = custom_text or REFERENCE_TEXTS.get(language, REFERENCE_TEXTS["fr"])
    
    # Nom du fichier
    output_path = output_dir / f"{language}_ref_{voice}.wav"
    
    print(f"🎤 Génération de la voix de référence...")
    print(f"   Voix: {voice}")
    print(f"   Langue: {language}")
    print(f"   Sortie: {output_path}")
    print()
    
    # Générer l'audio
    tts = KokoroProvider(voice=voice)
    await tts.synthesize(text.strip(), output_path)
    
    # Informations sur le fichier
    size = output_path.stat().st_size
    print()
    print(f"✅ Référence créée: {output_path}")
    print(f"   Taille: {size / 1024:.1f} KB")
    print()
    print("📝 Pour utiliser cette voix avec F5-TTS, ajoutez dans config.yaml:")
    print()
    print("   f5tts:")
    print(f'     ref_audio: "{output_path}"')
    print(f'     ref_text: |')
    for line in text.strip().split('\n'):
        print(f'       {line}')
    
    return output_path


def list_kokoro_voices():
    """Liste les voix Kokoro disponibles."""
    print("🎵 Voix Kokoro disponibles:")
    print()
    print("  Français:")
    print("    ff_siwis  - Voix féminine française (recommandé)")
    print()
    print("  Anglais (US):")
    print("    af_heart  - Voix féminine américaine")
    print("    af_bella  - Voix féminine américaine")
    print("    am_adam   - Voix masculine américaine")
    print()
    print("  Anglais (UK):")
    print("    bf_emma   - Voix féminine britannique")
    print("    bm_george - Voix masculine britannique")
    print()
    print("  Japonais:")
    print("    jf_alpha  - Voix féminine japonaise")
    print()
    print("Usage:")
    print("  python scripts/create_french_voice.py --voice ff_siwis")


async def main():
    parser = argparse.ArgumentParser(
        description="Créer une voix de référence pour F5-TTS"
    )
    parser.add_argument(
        "--voice", "-v",
        default="ff_siwis",
        help="Voix Kokoro à utiliser (défaut: ff_siwis)"
    )
    parser.add_argument(
        "--language", "-l",
        default="fr",
        choices=["fr", "en", "ja"],
        help="Langue du texte de référence (défaut: fr)"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        help="Dossier de sortie (défaut: ~/voices)"
    )
    parser.add_argument(
        "--text", "-t",
        help="Texte personnalisé à synthétiser"
    )
    parser.add_argument(
        "--list-voices",
        action="store_true",
        help="Lister les voix Kokoro disponibles"
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
