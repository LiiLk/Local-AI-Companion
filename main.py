"""
Point d'entrée principal de l'assistant IA avec voix.

L'IA répond en texte ET en voix simultanément !
Le TTS se déclenche phrase par phrase pour une latence minimale.

Usage:
    python main.py                    # Mode texte uniquement
    python main.py --voice            # Mode texte + voix (Kokoro par défaut)
    python main.py --voice --tts edge # Mode texte + voix Edge TTS
"""

import asyncio
import argparse
import subprocess
import tempfile
import re
import yaml
from pathlib import Path

from src.llm import OllamaLLM
from src.llm.base import Message
from src.tts import EdgeTTSProvider, KokoroProvider
from src.tts.base import BaseTTS


def load_config() -> dict:
    """Charge la configuration depuis config.yaml"""
    config_path = Path(__file__).parent / "config" / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def play_audio(audio_path: Path) -> subprocess.Popen:
    """
    Joue un fichier audio en arrière-plan.
    
    Supporte WAV (Kokoro) et MP3 (Edge TTS).
    Utilise mpv, ffplay ou aplay selon ce qui est disponible.
    Retourne le processus pour pouvoir l'arrêter si besoin.
    """
    players = [
        ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", str(audio_path)],
        ["mpv", "--no-terminal", "--no-video", str(audio_path)],
        ["aplay", str(audio_path)],  # WAV uniquement
    ]
    
    for player_cmd in players:
        try:
            # Lancer en arrière-plan, sans output
            process = subprocess.Popen(
                player_cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            return process
        except FileNotFoundError:
            continue
    
    print("\n⚠️  Aucun lecteur audio trouvé (ffplay, mpv, aplay)")
    return None


def create_tts(provider: str, tts_config: dict) -> BaseTTS:
    """
    Crée le provider TTS approprié.
    
    Args:
        provider: "kokoro" ou "edge"
        tts_config: Configuration TTS depuis config.yaml
        
    Returns:
        Instance du provider TTS
    """
    if provider == "kokoro":
        # Kokoro - TTS local haute qualité
        voice = tts_config.get("kokoro_voice", "ff_siwis")
        return KokoroProvider(voice=voice)
    else:
        # Edge TTS - Cloud Microsoft (fallback)
        voice = tts_config.get("voice", "fr-FR-DeniseNeural")
        rate = tts_config.get("rate", "+20%")
        pitch = tts_config.get("pitch", "+0Hz")
        return EdgeTTSProvider(voice=voice, rate=rate, pitch=pitch)


def split_into_sentences(text: str) -> list[str]:
    """
    Découpe le texte en phrases pour le TTS.
    
    On veut des phrases complètes pour un TTS naturel.
    """
    # Pattern pour détecter les fins de phrases
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]


async def speak_text(tts: BaseTTS, text: str, temp_dir: Path) -> subprocess.Popen | None:
    """
    Synthétise et joue le texte.
    
    Returns:
        Le processus audio pour pouvoir attendre qu'il finisse
    """
    if not text.strip():
        return None
    
    # Extension selon le type de TTS
    ext = ".wav" if isinstance(tts, KokoroProvider) else ".mp3"
    
    # Générer un fichier temporaire unique
    audio_file = temp_dir / f"speech_{hash(text) % 10000}{ext}"
    
    # Synthétiser
    await tts.synthesize(text, audio_file)
    
    # Jouer
    return play_audio(audio_file)


async def main():
    """
    Boucle principale du chatbot avec support vocal.
    """
    # Parser les arguments
    parser = argparse.ArgumentParser(description="Local AI Companion")
    parser.add_argument("--voice", "-v", action="store_true", 
                       help="Activer la synthèse vocale")
    parser.add_argument("--tts", type=str, default="kokoro",
                       choices=["kokoro", "edge"],
                       help="Provider TTS: kokoro (local) ou edge (cloud)")
    args = parser.parse_args()
    
    # Charger la configuration
    config = load_config()
    llm_config = config["llm"]["ollama"]
    character = config["character"]
    tts_config = config.get("tts", {})
    
    print("=" * 50)
    print(f"🤖 {character['name']} - Local AI Companion")
    print("=" * 50)
    
    if args.voice:
        tts_name = "Kokoro (local)" if args.tts == "kokoro" else "Edge TTS (cloud)"
        print(f"🔊 Mode vocal ACTIVÉ - {tts_name}")
    else:
        print("🔇 Mode texte (utilise --voice pour activer la voix)")
    
    print("\nCommandes: 'quit', 'clear', 'voice on', 'voice off'")
    print()
    
    # Créer le client LLM
    llm = OllamaLLM(
        model=llm_config["model"],
        base_url=llm_config["base_url"]
    )
    
    # Créer le TTS si mode vocal
    tts = None
    tts_provider = args.tts
    temp_dir = None
    
    if args.voice:
        tts = create_tts(tts_provider, tts_config)
        temp_dir = Path(tempfile.mkdtemp(prefix="ai_companion_"))
        
        voice_info = tts.voice if hasattr(tts, 'voice') else "default"
        print(f"🎤 Voix: {voice_info}\n")
    
    # Historique de la conversation
    messages: list[Message] = [
        Message(role="system", content=character["system_prompt"])
    ]
    
    audio_processes: list[subprocess.Popen] = []
    
    try:
        while True:
            # Attendre que les audios précédents finissent
            for proc in audio_processes:
                if proc:
                    proc.wait()
            audio_processes.clear()
            
            # 1. Lire l'entrée utilisateur
            try:
                user_input = input("\n👤 Toi: ").strip()
            except EOFError:
                break
                
            if not user_input:
                continue
            if user_input.lower() == "quit":
                print("\n👋 À bientôt !")
                break
            if user_input.lower() == "clear":
                messages = [Message(role="system", content=character["system_prompt"])]
                print("🗑️  Historique effacé !")
                continue
            if user_input.lower() == "voice on":
                if not tts:
                    tts = create_tts(tts_provider, tts_config)
                    temp_dir = Path(tempfile.mkdtemp(prefix="ai_companion_"))
                print("🔊 Mode vocal activé !")
                continue
            if user_input.lower() == "voice off":
                tts = None
                print("🔇 Mode vocal désactivé !")
                continue
            
            # 2. Ajouter le message utilisateur à l'historique
            messages.append(Message(role="user", content=user_input))
            
            # 3. Obtenir la réponse du LLM (avec streaming)
            print(f"\n🤖 {character['name']}: ", end="", flush=True)
            
            full_response = ""
            
            async for chunk in llm.chat_stream(messages):
                print(chunk, end="", flush=True)
                full_response += chunk
            
            print()  # Nouvelle ligne
            
            # 4. TTS sur la réponse complète
            if tts and full_response.strip():
                proc = await speak_text(tts, full_response, temp_dir)
                if proc:
                    audio_processes.append(proc)
            
            # 5. Ajouter la réponse à l'historique
            messages.append(Message(role="assistant", content=full_response))
            
    finally:
        for proc in audio_processes:
            if proc:
                proc.wait()
        
        if temp_dir and temp_dir.exists():
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
        
        await llm.close()


if __name__ == "__main__":
    asyncio.run(main())
