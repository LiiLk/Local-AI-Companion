"""
Point d'entrée principal de l'assistant IA avec voix.

L'IA répond en texte ET en voix simultanément !
Le TTS se déclenche phrase par phrase pour une latence minimale.

Usage:
    python main.py           # Mode texte uniquement
    python main.py --voice   # Mode texte + voix
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
from src.tts import EdgeTTSProvider


def load_config() -> dict:
    """Charge la configuration depuis config.yaml"""
    config_path = Path(__file__).parent / "config" / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def play_audio(audio_path: Path) -> subprocess.Popen:
    """
    Joue un fichier audio en arrière-plan.
    
    Utilise mpv, ffplay ou aplay selon ce qui est disponible.
    Retourne le processus pour pouvoir l'arrêter si besoin.
    """
    players = [
        ["mpv", "--no-terminal", "--no-video", str(audio_path)],
        ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", str(audio_path)],
        ["aplay", str(audio_path)],
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
    
    print("\n⚠️  Aucun lecteur audio trouvé (mpv, ffplay, aplay)")
    return None


def split_into_sentences(text: str) -> list[str]:
    """
    Découpe le texte en phrases pour le TTS.
    
    On veut des phrases complètes pour un TTS naturel.
    """
    # Pattern pour détecter les fins de phrases
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]


async def speak_text(tts: EdgeTTSProvider, text: str, temp_dir: Path) -> subprocess.Popen | None:
    """
    Synthétise et joue le texte.
    
    Returns:
        Le processus audio pour pouvoir attendre qu'il finisse
    """
    if not text.strip():
        return None
    
    # Générer un fichier temporaire unique
    audio_file = temp_dir / f"speech_{hash(text) % 10000}.mp3"
    
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
    parser.add_argument("--voice-id", type=str, default=None,
                       help="ID de la voix (ex: fr-FR-DeniseNeural)")
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
        print("🔊 Mode vocal ACTIVÉ")
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
    temp_dir = None
    if args.voice:
        voice_id = args.voice_id or tts_config.get("voice", "fr-FR-DeniseNeural")
        tts = EdgeTTSProvider(voice=voice_id)
        temp_dir = Path(tempfile.mkdtemp(prefix="ai_companion_"))
        print(f"🎤 Voix: {voice_id}\n")
    
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
                    voice_id = tts_config.get("voice", "fr-FR-DeniseNeural")
                    tts = EdgeTTSProvider(voice=voice_id)
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
