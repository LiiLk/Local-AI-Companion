"""
Point d'entrée principal de l'assistant IA avec voix.

L'IA répond en texte ET en voix simultanément !
Le TTS se déclenche phrase par phrase pour une latence minimale.

Usage:
    python main.py                    # Mode texte uniquement
    python main.py --voice            # Mode texte + voix (Kokoro par défaut)
    python main.py --voice --tts edge # Mode texte + voix Edge TTS
    python main.py --voice --listen   # Mode conversation vocale complète
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
from src.tts import EdgeTTSProvider, KokoroProvider, XTTSProvider
from src.tts.base import BaseTTS
from src.asr import RealtimeWhisperProvider
from src.asr.base import BaseASR


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
        provider: "f5tts", "kokoro" ou "edge"
        tts_config: Configuration TTS depuis config.yaml
        
    Returns:
        Instance du provider TTS
    """
    if provider == "xtts":
        # XTTS v2 - Voice cloning multilingue de qualité
        from pathlib import Path
        xtts_config = tts_config.get("xtts", {})
        speaker_wav = xtts_config.get("speaker_wav")
        
        # Expand speaker_wav path if provided
        if speaker_wav:
            speaker_wav = str(Path(speaker_wav).expanduser())
        
        return XTTSProvider(
            language=xtts_config.get("language", "fr"),
            speaker=xtts_config.get("speaker", "Claribel Dervla"),
            speaker_wav=speaker_wav,
            device=xtts_config.get("device"),  # None = auto-detect
        )
    elif provider == "kokoro":
        # Kokoro - TTS local haute qualité
        voice = tts_config.get("kokoro_voice", "ff_siwis")
        return KokoroProvider(voice=voice)
    else:
        # Edge TTS - Cloud Microsoft (fallback)
        voice = tts_config.get("voice", "fr-FR-DeniseNeural")
        rate = tts_config.get("rate", "+20%")
        pitch = tts_config.get("pitch", "+0Hz")
        return EdgeTTSProvider(voice=voice, rate=rate, pitch=pitch)


def create_asr(asr_config: dict) -> RealtimeWhisperProvider:
    """
    Crée le provider ASR (Speech-to-Text).
    
    Args:
        asr_config: Configuration ASR depuis config.yaml
        
    Returns:
        Instance du provider ASR
    """
    model_size = asr_config.get("model_size", "base")
    device = asr_config.get("device", "cpu")  # CPU par défaut (cuDNN issues)
    
    return RealtimeWhisperProvider(
        model_size=model_size,
        device=device
    )


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
    # XTTS et Kokoro génèrent du WAV, Edge TTS génère du MP3
    ext = ".wav" if isinstance(tts, (KokoroProvider, XTTSProvider)) else ".mp3"
    
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
                       choices=["xtts", "kokoro", "edge"],
                       help="Provider TTS: xtts (voice cloning), kokoro (local) ou edge (cloud)")
    parser.add_argument("--listen", "-l", action="store_true",
                       help="Activer l'écoute vocale (microphone)")
    parser.add_argument("--asr-model", type=str, default="base",
                       choices=["tiny", "base", "small", "medium", "large-v3"],
                       help="Taille du modèle Whisper pour l'ASR")
    args = parser.parse_args()
    
    # Si --listen est activé, activer aussi --voice automatiquement
    if args.listen:
        args.voice = True
    
    # Charger la configuration
    config = load_config()
    llm_config = config["llm"]["ollama"]
    character = config["character"]
    tts_config = config.get("tts", {})
    asr_config = config.get("asr", {})
    
    # Surcharger avec les arguments CLI
    asr_config["model_size"] = args.asr_model
    
    print("=" * 50)
    print(f"🤖 {character['name']} - Local AI Companion")
    print("=" * 50)
    
    if args.listen:
        print("🎤 Mode CONVERSATION VOCALE activé")
        print("   Parlez dans votre micro, l'IA vous répondra à voix haute !")
    elif args.voice:
        tts_names = {
            "openaudio": "OpenAudio S1-mini (#1 quality)",
            "kokoro": "Kokoro (local)",
            "edge": "Edge TTS (cloud)"
        }
        tts_name = tts_names.get(args.tts, args.tts)
        print(f"🔊 Mode vocal ACTIVÉ - {tts_name}")
    else:
        print("🔇 Mode texte (utilise --voice ou --listen)")
    
    print("\nCommandes: 'quit', 'clear', 'voice on', 'voice off', 'listen on', 'listen off'")
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
        print(f"🔊 Voix TTS: {voice_info}")
    
    # Créer l'ASR si mode écoute
    asr = None
    listen_mode = args.listen
    
    if listen_mode:
        asr = create_asr(asr_config)
        print(f"🎤 ASR: Whisper {args.asr_model}")
    
    print()
    
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
            
            # 1. Obtenir l'entrée utilisateur (texte ou voix)
            user_input = None
            
            if listen_mode and asr:
                # Mode écoute vocale
                print("\n🎤 [Parlez maintenant... ou tapez du texte]")
                
                # On utilise un système hybride: 
                # - Soit l'utilisateur parle (ASR)
                # - Soit il tape du texte (fallback)
                try:
                    # Essayer d'écouter pendant 10 secondes max
                    result = await asr.listen_once(timeout=10.0)
                    user_input = result.text.strip()
                    
                    if user_input:
                        print(f"👤 Toi (voix): {user_input}")
                    else:
                        print("   (Pas de parole détectée, tapez votre message)")
                        user_input = input("👤 Toi: ").strip()
                        
                except KeyboardInterrupt:
                    # L'utilisateur a appuyé sur Ctrl+C pendant l'écoute
                    print("\n   (Écoute annulée)")
                    user_input = input("👤 Toi: ").strip()
                except Exception as e:
                    print(f"\n   ⚠️ Erreur ASR: {e}")
                    user_input = input("👤 Toi: ").strip()
            else:
                # Mode texte classique
                try:
                    user_input = input("\n👤 Toi: ").strip()
                except EOFError:
                    break
            
            if not user_input:
                continue
            
            # Commandes spéciales
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
            if user_input.lower() == "listen on":
                if not asr:
                    asr = create_asr(asr_config)
                listen_mode = True
                print("🎤 Mode écoute activé !")
                continue
            if user_input.lower() == "listen off":
                listen_mode = False
                print("⌨️  Mode écoute désactivé (texte uniquement)")
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
