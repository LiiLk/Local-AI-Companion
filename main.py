"""
Main entry point for the AI assistant with voice support.

The AI responds with both text AND voice simultaneously!
TTS triggers sentence by sentence for minimal latency.

Usage:
    python main.py                    # Text-only mode
    python main.py --voice            # Text + voice (Kokoro by default)
    python main.py --voice --tts edge # Text + voice with Edge TTS
    python main.py --voice --listen   # Full voice conversation mode
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
    """Load configuration from config.yaml"""
    config_path = Path(__file__).parent / "config" / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def play_audio(audio_path: Path) -> subprocess.Popen:
    """
    Play an audio file in background.
    
    Supports WAV (Kokoro) and MP3 (Edge TTS).
    Uses mpv, ffplay or aplay depending on availability.
    Returns the process to be able to stop it if needed.
    """
    players = [
        ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", str(audio_path)],
        ["mpv", "--no-terminal", "--no-video", str(audio_path)],
        ["aplay", str(audio_path)],  # WAV only
    ]
    
    for player_cmd in players:
        try:
            # Launch in background, no output
            process = subprocess.Popen(
                player_cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            return process
        except FileNotFoundError:
            continue
    
    print("\n⚠️  No audio player found (ffplay, mpv, aplay)")
    return None


def create_tts(provider: str, tts_config: dict) -> BaseTTS:
    """
    Create the appropriate TTS provider.
    
    Args:
        provider: "xtts", "kokoro" or "edge"
        tts_config: TTS configuration from config.yaml
        
    Returns:
        TTS provider instance
    """
    if provider == "xtts":
        # XTTS v2 - High quality multilingual voice cloning
        from pathlib import Path
        xtts_config = tts_config.get("xtts", {})
        speaker_wav = xtts_config.get("speaker_wav")
        
        # Expand speaker_wav path if provided
        if speaker_wav:
            speaker_wav = str(Path(speaker_wav).expanduser())
        
        return XTTSProvider(
            language=xtts_config.get("language", "en"),
            speaker=xtts_config.get("speaker", "Claribel Dervla"),
            speaker_wav=speaker_wav,
            device=xtts_config.get("device"),  # None = auto-detect
        )
    elif provider == "kokoro":
        # Kokoro - High quality local TTS
        voice = tts_config.get("kokoro_voice", "af_heart")
        return KokoroProvider(voice=voice)
    else:
        # Edge TTS - Microsoft Cloud (fallback)
        voice = tts_config.get("voice", "en-US-JennyNeural")
        rate = tts_config.get("rate", "+0%")
        pitch = tts_config.get("pitch", "+0Hz")
        return EdgeTTSProvider(voice=voice, rate=rate, pitch=pitch)


def create_asr(asr_config: dict) -> RealtimeWhisperProvider:
    """
    Create the ASR provider (Speech-to-Text).
    
    Args:
        asr_config: ASR configuration from config.yaml
        
    Returns:
        ASR provider instance
    """
    model_size = asr_config.get("model_size", "base")
    device = asr_config.get("device", "cpu")  # CPU by default (cuDNN issues)
    
    return RealtimeWhisperProvider(
        model_size=model_size,
        device=device
    )


def split_into_sentences(text: str) -> list[str]:
    """
    Split text into sentences for TTS.
    
    We want complete sentences for natural TTS output.
    """
    # Pattern to detect sentence endings
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]


async def speak_text(tts: BaseTTS, text: str, temp_dir: Path) -> subprocess.Popen | None:
    """
    Synthesize and play the text.
    
    Returns:
        The audio process to wait for completion
    """
    if not text.strip():
        return None
    
    # Extension based on TTS type
    # XTTS and Kokoro generate WAV, Edge TTS generates MP3
    ext = ".wav" if isinstance(tts, (KokoroProvider, XTTSProvider)) else ".mp3"
    
    # Generate unique temp file
    audio_file = temp_dir / f"speech_{hash(text) % 10000}{ext}"
    
    # Synthesize
    await tts.synthesize(text, audio_file)
    
    # Play
    return play_audio(audio_file)


async def main():
    """
    Main chatbot loop with voice support.
    """
    # Parse arguments
    parser = argparse.ArgumentParser(description="Local AI Companion")
    parser.add_argument("--voice", "-v", action="store_true", 
                       help="Enable voice synthesis")
    parser.add_argument("--tts", type=str, default="kokoro",
                       choices=["xtts", "kokoro", "edge"],
                       help="TTS provider: xtts (voice cloning), kokoro (local) or edge (cloud)")
    parser.add_argument("--listen", "-l", action="store_true",
                       help="Enable voice listening (microphone)")
    parser.add_argument("--asr-model", type=str, default="base",
                       choices=["tiny", "base", "small", "medium", "large-v3"],
                       help="Whisper model size for ASR")
    args = parser.parse_args()
    
    # If --listen is enabled, also enable --voice automatically
    if args.listen:
        args.voice = True
    
    # Load configuration
    config = load_config()
    llm_config = config["llm"]["ollama"]
    character = config["character"]
    tts_config = config.get("tts", {})
    asr_config = config.get("asr", {})
    
    # Override with CLI arguments
    asr_config["model_size"] = args.asr_model
    
    print("=" * 50)
    print(f"🤖 {character['name']} - Local AI Companion")
    print("=" * 50)
    
    if args.listen:
        print("🎤 VOICE CONVERSATION mode enabled")
        print("   Speak into your mic, the AI will respond out loud!")
    elif args.voice:
        tts_names = {
            "xtts": "XTTS v2 (voice cloning)",
            "kokoro": "Kokoro (local)",
            "edge": "Edge TTS (cloud)"
        }
        tts_name = tts_names.get(args.tts, args.tts)
        print(f"🔊 Voice mode ENABLED - {tts_name}")
    else:
        print("🔇 Text mode (use --voice or --listen)")
    
    print("\nCommands: 'quit', 'clear', 'voice on', 'voice off', 'listen on', 'listen off'")
    print()
    
    # Create LLM client
    llm = OllamaLLM(
        model=llm_config["model"],
        base_url=llm_config["base_url"]
    )
    
    # Create TTS if voice mode
    tts = None
    tts_provider = args.tts
    temp_dir = None
    
    if args.voice:
        tts = create_tts(tts_provider, tts_config)
        temp_dir = Path(tempfile.mkdtemp(prefix="ai_companion_"))
        
        voice_info = tts.voice if hasattr(tts, 'voice') else "default"
        print(f"🔊 TTS Voice: {voice_info}")
    
    # Create ASR if listen mode
    asr = None
    listen_mode = args.listen
    
    if listen_mode:
        asr = create_asr(asr_config)
        print(f"🎤 ASR: Whisper {args.asr_model}")
    
    print()
    
    # Conversation history
    messages: list[Message] = [
        Message(role="system", content=character["system_prompt"])
    ]
    
    audio_processes: list[subprocess.Popen] = []
    
    try:
        while True:
            # Wait for previous audio to finish
            for proc in audio_processes:
                if proc:
                    proc.wait()
            audio_processes.clear()
            
            # 1. Get user input (text or voice)
            user_input = None
            
            if listen_mode and asr:
                # Voice listening mode
                print("\n🎤 [Speak now... or type text]")
                
                # Hybrid system:
                # - Either user speaks (ASR)
                # - Or types text (fallback)
                try:
                    # Try to listen for 10 seconds max
                    result = await asr.listen_once(timeout=10.0)
                    user_input = result.text.strip()
                    
                    if user_input:
                        print(f"👤 You (voice): {user_input}")
                    else:
                        print("   (No speech detected, type your message)")
                        user_input = input("👤 You: ").strip()
                        
                except KeyboardInterrupt:
                    # User pressed Ctrl+C during listening
                    print("\n   (Listening cancelled)")
                    user_input = input("👤 You: ").strip()
                except Exception as e:
                    print(f"\n   ⚠️ ASR Error: {e}")
                    user_input = input("👤 You: ").strip()
            else:
                # Classic text mode
                try:
                    user_input = input("\n👤 You: ").strip()
                except EOFError:
                    break
            
            if not user_input:
                continue
            
            # Special commands
            if user_input.lower() == "quit":
                print("\n👋 Goodbye!")
                break
            if user_input.lower() == "clear":
                messages = [Message(role="system", content=character["system_prompt"])]
                print("🗑️  History cleared!")
                continue
            if user_input.lower() == "voice on":
                if not tts:
                    tts = create_tts(tts_provider, tts_config)
                    temp_dir = Path(tempfile.mkdtemp(prefix="ai_companion_"))
                print("🔊 Voice mode enabled!")
                continue
            if user_input.lower() == "voice off":
                tts = None
                print("🔇 Voice mode disabled!")
                continue
            if user_input.lower() == "listen on":
                if not asr:
                    asr = create_asr(asr_config)
                listen_mode = True
                print("🎤 Listen mode enabled!")
                continue
            if user_input.lower() == "listen off":
                listen_mode = False
                print("⌨️  Listen mode disabled (text only)")
                continue
            
            # 2. Add user message to history
            messages.append(Message(role="user", content=user_input))
            
            # 3. Get LLM response (with streaming)
            print(f"\n🤖 {character['name']}: ", end="", flush=True)
            
            full_response = ""
            
            async for chunk in llm.chat_stream(messages):
                print(chunk, end="", flush=True)
                full_response += chunk
            
            print()  # New line
            
            # 4. TTS on complete response
            if tts and full_response.strip():
                proc = await speak_text(tts, full_response, temp_dir)
                if proc:
                    audio_processes.append(proc)
            
            # 5. Add response to history
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
