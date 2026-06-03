"""
Main entry point for the AI assistant with voice support.

The AI responds with both text AND voice simultaneously!
TTS triggers sentence by sentence for minimal latency.

Usage:
    python main.py                           # Text-only mode (pipeline)
    python main.py --voice                   # Text + voice (Kokoro by default)
    python main.py --voice --tts edge        # Text + voice with Edge TTS
    python main.py --voice --listen          # Full voice conversation mode
    python main.py --mode omni              # Omni mode (MiniCPM-o)
    python main.py --mode omni --listen     # Omni mode with mic input
"""

import asyncio
import argparse
import logging
import subprocess
import tempfile
import re
from pathlib import Path

from src.llm import OllamaLLM
from src.llm.base import Message
from src.tts import EdgeTTSProvider, KokoroProvider
from src.tts.base import BaseTTS
from src.asr import RealtimeWhisperProvider
from src.asr.base import BaseASR
from src.utils.config_loader import load_yaml_config
from src.assistant.pipeline_runtime import resolve_whisper_profile

# Setup logging
logger = logging.getLogger(__name__)


def load_config() -> dict:
    """Load configuration from config.yaml"""
    config_path = Path(__file__).parent / "config" / "config.yaml"
    return load_yaml_config(config_path)


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

    print("\n  No audio player found (ffplay, mpv, aplay)")
    return None


def create_tts(provider: str, tts_config: dict) -> BaseTTS:
    """
    Create the appropriate TTS provider.

    Args:
        provider: "kokoro", "qwen3", or "edge"
        tts_config: TTS configuration from config.yaml

    Returns:
        TTS provider instance
    """
    if provider == "kokoro":
        # Kokoro - High quality local TTS
        voice = tts_config.get("kokoro_voice", "af_heart")
        return KokoroProvider(voice=voice)
    elif provider == "qwen3":
        from src.tts import Qwen3TTSProvider

        qwen3_config = tts_config.get("qwen3", {})
        if not Qwen3TTSProvider.is_available(
            backend=qwen3_config.get("backend", "worker"),
            python_path=qwen3_config.get("python_path"),
            site_packages_dir=qwen3_config.get("site_packages_dir"),
            worker_script=qwen3_config.get("worker_script"),
        ):
            raise RuntimeError(
                "Qwen3-TTS runtime is not installed. Run scripts/install_qwen3_tts_windows.ps1 first."
            )
        return Qwen3TTSProvider(
            model_id=qwen3_config.get("model_id", "Qwen/Qwen3-TTS-12Hz-0.6B-Base"),
            mode=qwen3_config.get("mode", "voice_clone"),
            language=qwen3_config.get("language", "auto"),
            speaker=qwen3_config.get("speaker"),
            instruct=qwen3_config.get("instruct"),
            ref_audio_path=qwen3_config.get("ref_audio_path"),
            ref_text=qwen3_config.get("ref_text"),
            x_vector_only_mode=qwen3_config.get("x_vector_only_mode"),
            device=qwen3_config.get("device", "cuda:0"),
            dtype=qwen3_config.get("dtype", "bfloat16"),
            attn_implementation=qwen3_config.get("attn_implementation", "flash_attention_2"),
            backend=qwen3_config.get("backend", "worker"),
            python_path=qwen3_config.get("python_path"),
            site_packages_dir=qwen3_config.get("site_packages_dir"),
            worker_script=qwen3_config.get("worker_script"),
            request_timeout_sec=qwen3_config.get("request_timeout_sec", 20),
        )
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
    settings = resolve_whisper_profile(asr_config)
    return RealtimeWhisperProvider(
        model_size=settings["model_size"],
        device=settings["device"],
        compute_type=settings["compute_type"],
    )


def split_into_sentences(text: str) -> list[str]:
    """
    Split text into sentences for TTS.

    We want complete sentences for natural TTS output.
    """
    # Pattern to detect sentence endings
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]


class SentenceBuffer:
    """
    Buffer that accumulates text and yields complete sentences.

    Used for streaming TTS: as soon as a sentence is complete,
    we can start synthesizing it while the LLM continues.
    """

    # Sentence ending patterns
    SENTENCE_ENDINGS = re.compile(r'[.!?:]\s*$')
    # Minimum sentence length to trigger TTS (avoid very short phrases)
    MIN_SENTENCE_LENGTH = 10

    def __init__(self):
        self.buffer = ""
        self.sentences: list[str] = []

    def add(self, text: str) -> list[str]:
        """
        Add text to buffer and return any complete sentences.

        Args:
            text: New text chunk from LLM

        Returns:
            List of complete sentences (may be empty)
        """
        self.buffer += text
        complete = []

        # Check for sentence endings
        while True:
            # Look for sentence boundary
            match = re.search(r'([.!?:])\s+', self.buffer)
            if match:
                end_pos = match.end()
                sentence = self.buffer[:end_pos].strip()

                # Only yield if sentence is long enough
                if len(sentence) >= self.MIN_SENTENCE_LENGTH:
                    complete.append(sentence)
                    self.buffer = self.buffer[end_pos:]
                else:
                    # Too short, wait for more text
                    break
            else:
                break

        return complete

    def flush(self) -> str | None:
        """
        Get any remaining text in the buffer.

        Call this when the LLM stream ends.

        Returns:
            Remaining text or None if buffer is empty
        """
        if self.buffer.strip():
            remaining = self.buffer.strip()
            self.buffer = ""
            return remaining
        return None


async def speak_text(
    tts: BaseTTS,
    text: str,
    temp_dir: Path,
    tts_config: dict | None = None
) -> subprocess.Popen | None:
    """
    Synthesize and play the text.

    For Kokoro and Edge TTS, handles automatic language detection
    and voice switching if auto_detect_language is enabled in config.

    Args:
        tts: TTS provider instance
        text: Text to synthesize
        temp_dir: Temporary directory for audio files
        tts_config: TTS config for voice mapping (optional)

    Returns:
        The audio process to wait for completion
    """
    if not text.strip():
        return None

    # Handle auto-language detection for Kokoro and Edge TTS
    if tts_config and tts_config.get("auto_detect_language", False):
        voice_mapping = tts_config.get("voice_mapping", {})

        # Detect language
        from src.utils.language_detection import detect_language
        detected_lang = str(detect_language(text))

        if isinstance(tts, KokoroProvider):
            # Switch Kokoro voice based on language
            kokoro_voices = voice_mapping.get("kokoro", {})
            if detected_lang in kokoro_voices:
                new_voice = kokoro_voices[detected_lang]
                if tts.voice != new_voice:
                    tts.voice = new_voice
                    print(f"  [{detected_lang.upper()}] Kokoro voice: {new_voice}")

        elif isinstance(tts, EdgeTTSProvider):
            # Switch Edge TTS voice based on language
            edge_voices = voice_mapping.get("edge", {})
            if detected_lang in edge_voices:
                new_voice = edge_voices[detected_lang]
                if tts.voice != new_voice:
                    tts.set_voice(new_voice)
                    print(f"  [{detected_lang.upper()}] Edge voice: {new_voice}")

    # Extension based on TTS type
    # Kokoro generates WAV, Edge TTS generates MP3
    ext = ".wav" if isinstance(tts, KokoroProvider) else ".mp3"

    # Generate unique temp file
    audio_file = temp_dir / f"speech_{hash(text) % 10000}{ext}"

    # Synthesize
    await tts.synthesize(text, audio_file)

    # Play
    return play_audio(audio_file)


async def stream_tts_with_llm(
    llm,
    messages: list,
    tts: BaseTTS,
    temp_dir: Path,
    tts_config: dict | None,
    character_name: str
) -> str:
    """
    Stream LLM response while synthesizing TTS sentence by sentence.

    This provides much lower latency than waiting for the complete response:
    - As soon as a sentence is complete, TTS starts synthesizing
    - Audio plays as soon as each sentence is ready
    - TTS calls are serialized to avoid CUDA conflicts

    Args:
        llm: LLM client instance
        messages: Conversation history
        tts: TTS provider instance
        temp_dir: Temporary directory for audio files
        tts_config: TTS configuration
        character_name: Character name for display

    Returns:
        Full response text
    """
    print(f"\n{character_name}: ", end="", flush=True)

    sentence_buffer = SentenceBuffer()
    full_response = ""
    sentence_index = 0

    # Queues for producer-consumer pattern
    sentences_to_synthesize: asyncio.Queue[tuple[int, str] | None] = asyncio.Queue()
    audio_to_play: asyncio.Queue[tuple[int, Path | None] | None] = asyncio.Queue()

    async def tts_worker():
        """
        Worker that synthesizes sentences sequentially.
        This avoids CUDA conflicts by ensuring only one TTS call at a time.
        """
        while True:
            item = await sentences_to_synthesize.get()

            if item is None:  # Sentinel to stop
                await audio_to_play.put(None)
                break

            idx, sentence = item

            try:
                # Handle auto-language detection for Kokoro and Edge TTS
                if tts_config and tts_config.get("auto_detect_language", False):
                    voice_mapping = tts_config.get("voice_mapping", {})
                    from src.utils.language_detection import detect_language
                    detected_lang = str(detect_language(sentence))

                    if isinstance(tts, KokoroProvider):
                        kokoro_voices = voice_mapping.get("kokoro", {})
                        if detected_lang in kokoro_voices:
                            new_voice = kokoro_voices[detected_lang]
                            if tts.voice != new_voice:
                                tts.voice = new_voice

                    elif isinstance(tts, EdgeTTSProvider):
                        edge_voices = voice_mapping.get("edge", {})
                        if detected_lang in edge_voices:
                            new_voice = edge_voices[detected_lang]
                            if tts.voice != new_voice:
                                tts.set_voice(new_voice)

                # Extension based on TTS type
                ext = ".wav" if isinstance(tts, KokoroProvider) else ".mp3"

                # Generate unique temp file
                audio_file = temp_dir / f"sentence_{idx:03d}_{hash(sentence) % 10000}{ext}"

                # Synthesize (sequential, no CUDA conflicts)
                await tts.synthesize(sentence, audio_file)

                await audio_to_play.put((idx, audio_file))

            except Exception as e:
                logger.error(f"TTS error for sentence {idx}: {e}")
                await audio_to_play.put((idx, None))

    async def audio_player():
        """Play audio files in order as they become ready."""
        next_to_play = 0
        pending: dict[int, Path] = {}

        while True:
            item = await audio_to_play.get()

            if item is None:  # Sentinel to stop
                # Play any remaining audio in order
                while next_to_play in pending:
                    proc = play_audio(pending[next_to_play])
                    if proc:
                        await asyncio.get_event_loop().run_in_executor(None, proc.wait)
                    del pending[next_to_play]
                    next_to_play += 1
                break

            idx, path = item

            if path:
                pending[idx] = path

            # Play in order
            while next_to_play in pending:
                proc = play_audio(pending[next_to_play])
                if proc:
                    # Wait for audio to finish before playing next
                    await asyncio.get_event_loop().run_in_executor(None, proc.wait)
                del pending[next_to_play]
                next_to_play += 1

    # Start worker tasks
    tts_task = asyncio.create_task(tts_worker())
    player_task = asyncio.create_task(audio_player())

    # Stream LLM response
    async for chunk in llm.chat_stream(messages):
        print(chunk, end="", flush=True)
        full_response += chunk

        # Check for complete sentences
        complete_sentences = sentence_buffer.add(chunk)

        for sentence in complete_sentences:
            # Queue sentence for TTS (will be processed sequentially)
            await sentences_to_synthesize.put((sentence_index, sentence))
            sentence_index += 1

    print()  # New line after response

    # Handle remaining text in buffer
    remaining = sentence_buffer.flush()
    if remaining:
        await sentences_to_synthesize.put((sentence_index, remaining))

    # Signal end of sentences
    await sentences_to_synthesize.put(None)

    # Wait for TTS and audio to finish
    await tts_task
    await player_task

    return full_response


async def run_omni_mode(config: dict, args):
    """
    Run in omni mode using MiniCPM-o for unified speech-to-speech.

    The omni model handles ASR, LLM, and TTS in a single model.
    """
    from src.omni import MiniCPMoProvider

    character = config["character"]
    omni_config = config.get("omni", {}).get("minicpmo", {})

    print("=" * 50)
    print(f"{character['name']} - Local AI Companion (Omni Mode)")
    print("=" * 50)
    quantization = omni_config.get("quantization")
    ref_audio = omni_config.get("ref_audio_path")

    # Check character preset for voice cloning reference
    voice_config = character.get("voice", {})
    if voice_config.get("omni_ref_audio"):
        ref_audio = voice_config["omni_ref_audio"]

    print(f"Model: MiniCPM-o ({omni_config.get('model_id', 'openbmb/MiniCPM-o-4_5')})")
    print(f"Device: {omni_config.get('device', 'cuda')}, Quantization: {quantization or 'none'}")

    if args.listen:
        print("Mic input enabled - speak and the model responds with audio")
    else:
        print("Text input mode - type and the model responds")

    print("\nCommands: 'quit', 'clear'")
    print()

    # Create omni provider
    omni = MiniCPMoProvider(
        model_name=omni_config.get("model_id", "openbmb/MiniCPM-o-4_5"),
        device=omni_config.get("device", "cuda"),
        quantization=quantization,
        ref_audio_path=ref_audio,
    )

    print("Loading MiniCPM-o model...")
    # Trigger lazy load
    omni._get_model()
    print("Model loaded.\n")

    # Conversation history
    messages = [
        {"role": "system", "content": [character.get("system_prompt", "You are a helpful assistant.")]}
    ]

    temp_dir = Path(tempfile.mkdtemp(prefix="ai_omni_"))

    # Optional: ASR for listen mode (the omni model can also do ASR, but
    # we use the mic capture + omni transcribe flow)
    asr = None
    listen_mode = args.listen
    if listen_mode:
        asr_config = config.get("asr", {})
        asr = create_asr(asr_config)
        print(f"ASR: Whisper {resolve_whisper_profile(asr_config)['model_size']} (for mic capture)")

    try:
        while True:
            # Get user input
            user_input = None

            if listen_mode and asr:
                print("\n[Speak now... or type text]")
                try:
                    result = await asr.listen_once(timeout=10.0)
                    user_input = result.text.strip()
                    if user_input:
                        print(f"You (voice): {user_input}")
                    else:
                        print("   (No speech detected, type your message)")
                        user_input = input("You: ").strip()
                except KeyboardInterrupt:
                    print("\n   (Listening cancelled)")
                    user_input = input("You: ").strip()
                except Exception as e:
                    print(f"\n   ASR Error: {e}")
                    user_input = input("You: ").strip()
            else:
                try:
                    user_input = input("\nYou: ").strip()
                except EOFError:
                    break

            if not user_input:
                continue

            if user_input.lower() == "quit":
                print("\nGoodbye!")
                break
            if user_input.lower() == "clear":
                messages = [messages[0]]
                print("History cleared!")
                continue

            # Add user message
            messages.append({"role": "user", "content": [user_input]})

            # Generate response with audio
            audio_path = temp_dir / f"response_{len(messages)}.wav"
            print(f"\n{character['name']}: ", end="", flush=True)

            text, audio_out = omni.chat_omni(
                messages=messages,
                output_audio_path=audio_path if args.voice or args.listen else None,
                max_new_tokens=omni_config.get("max_tokens", 4096),
                temperature=omni_config.get("temperature", 0.7),
            )

            print(text)

            # Add assistant response to history
            messages.append({"role": "assistant", "content": [text]})

            # Play audio if generated
            if audio_out and audio_out.exists():
                proc = play_audio(audio_out)
                if proc:
                    await asyncio.get_event_loop().run_in_executor(None, proc.wait)

    finally:
        if temp_dir and temp_dir.exists():
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)


async def main():
    """
    Main chatbot loop with voice support.
    """
    # Parse arguments
    parser = argparse.ArgumentParser(description="Local AI Companion")
    parser.add_argument("--voice", "-v", action="store_true",
                       help="Enable voice synthesis")
    parser.add_argument("--tts", type=str, default="kokoro",
                       choices=["kokoro", "qwen3", "edge"],
                       help="TTS provider: kokoro (local), qwen3 (local voice clone), or edge (cloud)")
    parser.add_argument("--listen", "-l", action="store_true",
                       help="Enable voice listening (microphone)")
    parser.add_argument("--asr-model", type=str, default=None,
                       choices=["tiny", "base", "small", "medium", "large-v3", "large-v3-turbo"],
                       help="Override Whisper model for ASR (default: use the config asr.profile)")
    parser.add_argument("--mode", type=str, default=None,
                       choices=["omni", "pipeline"],
                       help="Override mode from config")
    args = parser.parse_args()

    # If --listen is enabled, also enable --voice automatically
    if args.listen:
        args.voice = True

    # Load configuration
    config = load_config()

    # Resolve character preset
    from src.utils.character_loader import resolve_character_config
    config = resolve_character_config(config)

    # Determine mode: CLI override > config > default
    mode = args.mode or config.get("mode", "pipeline")

    # Omni mode: use MiniCPM-o for everything
    if mode == "omni":
        await run_omni_mode(config, args)
        return

    # Pipeline mode: Ollama + Kokoro/Edge + Whisper
    llm_config = config["llm"]["ollama"]
    character = config["character"]
    tts_config = config.get("tts", {})
    asr_config = config.get("asr", {})

    # Override the config profile only when --asr-model is explicitly passed
    if args.asr_model:
        asr_config["model_size"] = args.asr_model

    print("=" * 50)
    print(f"{character['name']} - Local AI Companion")
    print("=" * 50)

    if args.listen:
        print("VOICE CONVERSATION mode enabled")
        print("   Speak into your mic, the AI will respond out loud!")
    elif args.voice:
        tts_names = {
            "kokoro": "Kokoro (local)",
            "qwen3": "Qwen3-TTS (local)",
            "edge": "Edge TTS (cloud)"
        }
        tts_name = tts_names.get(args.tts, args.tts)
        print(f"Voice mode ENABLED - {tts_name}")
    else:
        print("Text mode (use --voice or --listen)")

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
        print(f"TTS Voice: {voice_info}")

        # Show auto-detect status
        if tts_config.get("auto_detect_language", False):
            print("Auto language detection: ENABLED (FR/EN)")

        # Show streaming TTS status
        if tts_config.get("stream_tts", True):
            print("Streaming TTS: ENABLED (sentence-by-sentence)")
        else:
            print("Streaming TTS: DISABLED (full response)")

    # Create ASR if listen mode
    asr = None
    listen_mode = args.listen

    if listen_mode:
        asr = create_asr(asr_config)
        print(f"ASR: Whisper {resolve_whisper_profile(asr_config)['model_size']}")

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
                print("\n[Speak now... or type text]")

                try:
                    # Try to listen for 10 seconds max
                    result = await asr.listen_once(timeout=10.0)
                    user_input = result.text.strip()

                    if user_input:
                        print(f"You (voice): {user_input}")
                    else:
                        print("   (No speech detected, type your message)")
                        user_input = input("You: ").strip()

                except KeyboardInterrupt:
                    print("\n   (Listening cancelled)")
                    user_input = input("You: ").strip()
                except Exception as e:
                    print(f"\n   ASR Error: {e}")
                    user_input = input("You: ").strip()
            else:
                # Classic text mode
                try:
                    user_input = input("\nYou: ").strip()
                except EOFError:
                    break

            if not user_input:
                continue

            # Special commands
            if user_input.lower() == "quit":
                print("\nGoodbye!")
                break
            if user_input.lower() == "clear":
                messages = [Message(role="system", content=character["system_prompt"])]
                print("History cleared!")
                continue
            if user_input.lower() == "voice on":
                if not tts:
                    tts = create_tts(tts_provider, tts_config)
                    temp_dir = Path(tempfile.mkdtemp(prefix="ai_companion_"))
                print("Voice mode enabled!")
                continue
            if user_input.lower() == "voice off":
                tts = None
                print("Voice mode disabled!")
                continue
            if user_input.lower() == "listen on":
                if not asr:
                    asr = create_asr(asr_config)
                listen_mode = True
                print("Listen mode enabled!")
                continue
            if user_input.lower() == "listen off":
                listen_mode = False
                print("Listen mode disabled (text only)")
                continue

            # 2. Add user message to history
            messages.append(Message(role="user", content=user_input))

            # 3. Get LLM response with optional streaming TTS
            stream_tts_enabled = tts_config.get("stream_tts", True) if tts_config else True

            if tts and stream_tts_enabled:
                # Use streaming TTS (sentence by sentence, parallel to LLM)
                full_response = await stream_tts_with_llm(
                    llm=llm,
                    messages=messages,
                    tts=tts,
                    temp_dir=temp_dir,
                    tts_config=tts_config,
                    character_name=character['name']
                )
            elif tts:
                # Non-streaming TTS (wait for full response)
                print(f"\n{character['name']}: ", end="", flush=True)

                full_response = ""
                async for chunk in llm.chat_stream(messages):
                    print(chunk, end="", flush=True)
                    full_response += chunk

                print()  # New line

                # TTS on complete response
                if full_response.strip():
                    proc = await speak_text(tts, full_response, temp_dir, tts_config)
                    if proc:
                        audio_processes.append(proc)
            else:
                # Text-only mode (no TTS)
                print(f"\n{character['name']}: ", end="", flush=True)

                full_response = ""
                async for chunk in llm.chat_stream(messages):
                    print(chunk, end="", flush=True)
                    full_response += chunk

                print()  # New line

            # 4. Add response to history
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
