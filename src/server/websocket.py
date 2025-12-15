"""
WebSocket Handler - Real-time bidirectional communication.

Handles:
- Text messages (chat)
- Audio streaming with VAD (Voice Activity Detection)
- Configuration updates
"""

import json
import asyncio
import base64
import emoji
import tempfile
import numpy as np
from langdetect import detect, LangDetectException
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass, field

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import yaml

from src.llm import OllamaLLM, LlamaCppProvider
from src.llm.base import Message
from src.tts import KokoroProvider, EdgeTTSProvider, XTTSProvider, F5TTSProvider, GPTSoVITSProvider, CosyVoice3Provider
from src.asr import WhisperProvider, CanaryProvider, ParakeetProvider
from src.vad import SileroVAD
from src.utils.audio_analysis import analyze_audio_volumes, read_wav_pcm, calculate_audio_duration_ms
from src.utils.emotion_detector import EmotionDetector, strip_emotion_markers

websocket_router = APIRouter()


def load_config() -> dict:
    """Load configuration from config.yaml"""
    config_path = Path(__file__).parent.parent.parent / "config" / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@dataclass
class ConversationState:
    """
    State for a single WebSocket conversation.
    
    Each connected client has their own state.
    """
    messages: list = field(default_factory=list)
    llm: Optional[OllamaLLM] = None
    tts: Optional[Any] = None
    asr: Optional[WhisperProvider] = None
    vad: Optional[SileroVAD] = None
    config: dict = field(default_factory=dict)
    is_recording: bool = False
    audio_buffer: list = field(default_factory=list)  # Buffer for streaming audio
    current_language: str = "fr"  # Track current language context
    emotion_detector: Optional[EmotionDetector] = None  # For Live2D expressions
    current_expression: str = "neutral"  # Current Live2D expression
    
    def __post_init__(self):
        self.config = load_config()
        self.emotion_detector = EmotionDetector()
        
    async def initialize(self):
        """Initialize models (lazy loading)."""
        if self.llm is None:
            llm_config = self.config.get("llm", {})
            provider = llm_config.get("provider", "ollama")
            
            if provider == "llamacpp":
                # llama.cpp server (for Qwen3-VL and other vision models)
                llamacpp_config = llm_config.get("llamacpp", {})
                self.llm = LlamaCppProvider(
                    base_url=llamacpp_config.get("base_url", "http://localhost:8080"),
                    model_name=llamacpp_config.get("model_name", "qwen3-vl-8b-instruct"),
                    max_tokens=llamacpp_config.get("max_tokens", 2048),
                    temperature=llamacpp_config.get("temperature", 1.0),
                    top_p=llamacpp_config.get("top_p", 0.95),
                    top_k=llamacpp_config.get("top_k", 20),
                    presence_penalty=llamacpp_config.get("presence_penalty", 1.5)
                )
            else:
                # Ollama (default)
                ollama_config = llm_config.get("ollama", {})
                self.llm = OllamaLLM(
                    model=ollama_config.get("model", "llama3.2:3b"),
                    base_url=ollama_config.get("base_url", "http://localhost:11434")
                )
            
        # Initialize conversation with system prompt
        if not self.messages:
            character = self.config.get("character", {})
            system_prompt = character.get("system_prompt", "You are a helpful assistant.")
            self.messages.append(Message(role="system", content=system_prompt))
    
    def get_tts(self):
        """Get or create TTS provider (lazy loading)."""
        if self.tts is None:
            tts_config = self.config.get("tts", {})
            provider = tts_config.get("provider", "kokoro")
            
            # Check if auto-language detection is enabled
            auto_detect = tts_config.get("auto_detect_language", False)
            
            print(f"🔊 Loading TTS provider: {provider} (auto_detect={auto_detect})")
            
            if provider == "cosyvoice3":
                # CosyVoice3 - State-of-the-art zero-shot voice cloning (~2GB VRAM)
                cv3_config = tts_config.get("cosyvoice3", {})
                print(f"   CosyVoice3 config: api={cv3_config.get('api_url', 'http://127.0.0.1:9881')}")
                self.tts = CosyVoice3Provider(config=self.config)
            elif provider == "gpt_sovits":
                # GPT-SoVITS - Best voice cloning quality (requires server)
                sovits_config = tts_config.get("gpt_sovits", {})
                print(f"   GPT-SoVITS config: api={sovits_config.get('api_url', 'http://127.0.0.1:9880')}")
                self.tts = GPTSoVITSProvider(tts_config)
            elif provider == "xtts":
                # XTTS v2 - Multilingual voice cloning (~2.8GB VRAM)
                xtts_config = tts_config.get("xtts", {})
                speaker_wav = xtts_config.get("speaker_wav")
                
                if speaker_wav:
                    speaker_wav = str(Path(speaker_wav).expanduser())
                
                device = xtts_config.get("device")
                print(f"   XTTS config: language={xtts_config.get('language', 'en')}, device={device}")
                
                self.tts = XTTSProvider(
                    language=xtts_config.get("language", "en"),
                    speaker=xtts_config.get("speaker", "Claribel Dervla"),
                    speaker_wav=speaker_wav,
                    device=device,
                    auto_detect_language=auto_detect,
                )
            elif provider == "kokoro":
                voice = tts_config.get("kokoro_voice", "ff_siwis")
                print(f"   Kokoro config: voice={voice}")
                self.tts = KokoroProvider(voice=voice)
            elif provider == "f5tts":
                print(f"   F5-TTS config: standard preset")
                self.tts = F5TTSProvider(device="cuda")
            else:
                voice = tts_config.get("voice", "fr-FR-DeniseNeural")
                print(f"   Edge TTS config: voice={voice}")
                self.tts = EdgeTTSProvider(voice=voice)
                
        return self.tts
    
    def get_asr(self):
        """Get or create ASR provider (lazy loading)."""
        if self.asr is None:
            asr_config = self.config.get("asr", {})
            provider = asr_config.get("provider", "whisper")
            device = asr_config.get("device", "cpu")
            
            if provider == "canary":
                # NVIDIA Canary 1B v2 - state-of-the-art ASR (heavy)
                self.asr = CanaryProvider(device=device)
            elif provider == "parakeet":
                # NVIDIA Parakeet TDT 0.6B v3 - fast and accurate
                self.asr = ParakeetProvider(device=device)
            else:
                # Whisper (default) - best for conversational audio
                model_size = asr_config.get("model_size", "base")
                initial_prompt = asr_config.get("prompt", None)
                self.asr = WhisperProvider(
                    model_size=model_size, 
                    device=device,
                    initial_prompt=initial_prompt
                )
            
            # Store language preference (empty or "auto" = auto-detection)
            self.asr_language = asr_config.get("language", "")
            
        return self.asr
    
    def get_vad(self):
        """Get or create VAD engine (lazy loading)."""
        if self.vad is None:
            self.vad = SileroVAD()
        return self.vad
    
    async def cleanup(self):
        """Cleanup resources."""
        if self.llm:
            await self.llm.close()


class WebSocketManager:
    """
    Manages WebSocket connections and message handling.
    """
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.states: Dict[str, ConversationState] = {}
        self._preloading: Dict[str, bool] = {}  # Track preloading state per client
    
    async def connect(self, websocket: WebSocket, client_id: str):
        """Accept a new WebSocket connection."""
        await websocket.accept()
        self.active_connections[client_id] = websocket
        self.states[client_id] = ConversationState()
        await self.states[client_id].initialize()
        print(f"✅ Client connected: {client_id}")
        
        # Start preloading models in background (don't await - non-blocking)
        asyncio.create_task(self._preload_models_progressive(client_id))
    
    async def disconnect(self, client_id: str):
        """Handle client disconnection."""
        if client_id in self.states:
            await self.states[client_id].cleanup()
            del self.states[client_id]
        if client_id in self.active_connections:
            del self.active_connections[client_id]
        print(f"👋 Client disconnected: {client_id}")
    
    async def send_json(self, client_id: str, data: dict):
        """Send JSON message to a client."""
        if client_id in self.active_connections:
            try:
                await self.active_connections[client_id].send_json(data)
            except Exception as e:
                print(f"⚠️ Failed to send to {client_id}: {e}")
    
    def _get_state(self, client_id: str) -> Optional[ConversationState]:
        """Safely get client state, returns None if client disconnected."""
        return self.states.get(client_id)
    
    def _clean_text_for_tts(self, text: str) -> str:
        """
        Clean text before TTS:
        1. Remove emojis
        2. Remove markdown symbols (*, #, _, etc.) that TTS reads out loud
        """
        # Remove emojis
        text = emoji.replace_emoji(text, replace="")
        
        import re
        # Remove actions between asterisks like *rolls eyes* or *sighs*
        text = re.sub(r'\*[^*]+\*', '', text)
        
        # Remove markdown chars: * # _ ` ~ >
        # We replace them with empty string to avoid "asterix" recitation
        text = re.sub(r'[\*\#\_\`\~\>]+', '', text)
        
        # Remove multiple spaces
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    async def _update_voice_for_language(self, state: ConversationState, text: str):
        """
        Detect language and update TTS voice if needed.
        """
        if not text.strip() or len(text) < 5:  # Too short to detect reliability
            return

        try:
            # Detect language
            lang = detect(text)
            
            # Simple mapping for common detections
            if lang in ['fr', 'fr-fr']:
                lang_code = 'fr'
            elif lang in ['en', 'en-us', 'en-gb']:
                lang_code = 'en'
            else:
                return # Ignore other languages for now
            
            # Update state if changed
            if state.current_language != lang_code:
                print(f"🌐 Language switch detected: {state.current_language} -> {lang_code}")
                state.current_language = lang_code
                
                # Update TTS voice
                tts_config = state.config.get("tts", {})
                provider_name = tts_config.get("provider", "kokoro")
                voice_mapping = tts_config.get("voice_mapping", {}).get(provider_name, {})
                
                new_voice = voice_mapping.get(lang_code)
                if new_voice and state.tts:
                    if hasattr(state.tts, 'set_voice'):
                        state.tts.set_voice(new_voice)
                        print(f"   🗣️ Switched {provider_name} voice to: {new_voice}")

        except LangDetectException:
            pass
        except Exception as e:
            print(f"⚠️ Voice switch error: {e}")

    async def _process_tts_chunk(self, client_id: str, text: str):
        """Generate and send audio for a text chunk with lip-sync data."""
        if not text.strip():
            return
        
        state = self._get_state(client_id)
        if not state:
            return  # Client disconnected
        
        try:
            # 1. Detect emotion BEFORE cleaning text
            emotion = None
            expression = None
            if state.emotion_detector:
                emotion = state.emotion_detector.detect(text)
                if emotion:
                    expression = state.emotion_detector.get_expression(emotion)
                    if expression != state.current_expression:
                        state.current_expression = expression
                        print(f"😊 Expression change: {expression}")
                        # Send expression change event
                        await self.send_json(client_id, {
                            "type": "expression_change",
                            "expression": expression,
                            "emotion": emotion
                        })
            
            # 2. Update voice based on language
            await self._update_voice_for_language(state, text)
            
            # 3. Clean text for TTS (remove emojis, emotion markers, etc.)
            if state.emotion_detector:
                text = state.emotion_detector.strip_markers(text)
            text = self._clean_text_for_tts(text)
            
            if not text.strip():
                return
            
            # Lazy load TTS provider
            tts = state.get_tts()
        except Exception as e:
            print(f"❌ TTS init error: {e}")
            await self.send_json(client_id, {
                "type": "error",
                "message": f"TTS unavailable: {e}"
            })
            return
        
        try:
            # Generate audio to temp file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                temp_path = Path(f.name)
            
            # Run TTS (in executor if blocking)
            if asyncio.iscoroutinefunction(tts.synthesize):
                await tts.synthesize(text, temp_path)
            else:
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, tts.synthesize, text, temp_path)
            
            # Read full WAV file for browser playback
            with open(temp_path, "rb") as f:
                audio_data = f.read()
            
            # 4. Extract volume data for lip-sync
            volumes = []
            duration_ms = 0
            try:
                pcm_data, sample_rate = read_wav_pcm(temp_path)
                volumes = analyze_audio_volumes(
                    pcm_data, 
                    sample_rate=sample_rate,
                    chunk_ms=50  # 50ms chunks = 20 values per second
                )
                duration_ms = calculate_audio_duration_ms(pcm_data, sample_rate)
            except Exception as e:
                print(f"⚠️ Volume analysis error: {e}")
            
            # 5. Send audio with lip-sync data
            audio_base64 = base64.b64encode(audio_data).decode("utf-8")
            await self.send_json(client_id, {
                "type": "audio_data",
                "data": audio_base64,
                "format": "wav",
                # Live2D lip-sync data
                "lip_sync": {
                    "volumes": volumes,
                    "duration_ms": duration_ms,
                    "chunk_ms": 50
                },
                "expression": expression,
                "text": text
            })
            
            # Cleanup
            temp_path.unlink(missing_ok=True)
            
        except Exception as e:
            print(f"❌ TTS chunk error: {e}")

    async def handle_text_message(self, client_id: str, content: str, language: str | None = None):
        """
        Handle a text message from the client.
        
        1. Add user message to history
        2. Stream LLM response
        3. Generate and stream TTS audio (sentence by sentence)
        
        Args:
            client_id: Client identifier
            content: User message text
            language: Detected language code (e.g. "fr", "en")
        """
        state = self._get_state(client_id)
        if not state:
            return  # Client disconnected
        
        # Add user message
        state.messages.append(Message(role="user", content=content))
        
        # Notify start of response
        await self.send_json(client_id, {"type": "text_start"})
        await self.send_json(client_id, {"type": "audio_start"})
        
        # Stream LLM response
        full_response = ""
        current_sentence = ""
        
        # Prepare messages for LLM
        llm_messages = list(state.messages)
        
        # If language is detected, enforce it via a temporary system instruction
        # This helps the LLM switch languages even if history is in another language
        if language:
            lang_map = {"fr": "French", "en": "English", "es": "Spanish", "de": "German", "it": "Italian", "ja": "Japanese"}
            lang_name = lang_map.get(language, language)
            
            # Prepend instruction to the last user message
            # This is safer than appending a system message at the end
            if llm_messages and llm_messages[-1].role == "user":
                last_msg = llm_messages[-1]
                new_content = f"(System: The user is speaking {lang_name}. Reply in {lang_name}.)\n\n{last_msg.content}"
                llm_messages[-1] = Message(role="user", content=new_content)
                print(f"🧠 Enforcing language: {lang_name}")
        
        async for chunk in state.llm.chat_stream(llm_messages):
            full_response += chunk
            current_sentence += chunk
            
            await self.send_json(client_id, {
                "type": "text_chunk",
                "content": chunk
            })
            
            # Check for sentence delimiters
            if any(punct in chunk for punct in ".!?\n"):
                # Simple split by delimiters
                import re
                # Split keeping delimiters
                parts = re.split(r'([.!?\n]+)', current_sentence)
                
                if len(parts) > 1:
                    # Process all complete sentences
                    for i in range(0, len(parts) - 1, 2):
                        sentence = parts[i] + parts[i+1]
                        if sentence.strip():
                            await self._process_tts_chunk(client_id, sentence)
                    
                    # Keep the remainder
                    current_sentence = parts[-1]
        
        # Process remaining text
        if current_sentence.strip():
            await self._process_tts_chunk(client_id, current_sentence)
        
        # Add assistant message to history
        state.messages.append(Message(role="assistant", content=full_response))
        
        # Notify end of text
        await self.send_json(client_id, {
            "type": "text_end",
            "full_text": full_response
        })
        
        # Notify audio end
        await self.send_json(client_id, {"type": "audio_end"})
    
    async def generate_and_send_audio(self, client_id: str, text: str):
        """Generate TTS audio and send to client."""
        text = self._clean_text_for_tts(text)
        
        if not text.strip():
            return
        
        state = self._get_state(client_id)
        if not state:
            return  # Client disconnected
        
        tts = state.get_tts()
        
        # Notify audio start
        await self.send_json(client_id, {"type": "audio_start"})
        
        try:
            # Generate audio to temp file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                temp_path = Path(f.name)
            
            await tts.synthesize(text, temp_path)
            
            # Read and encode audio
            with open(temp_path, "rb") as f:
                audio_data = f.read()
            
            # Send as base64
            audio_base64 = base64.b64encode(audio_data).decode("utf-8")
            await self.send_json(client_id, {
                "type": "audio_data",
                "data": audio_base64,
                "format": "wav"
            })
            
            # Cleanup
            temp_path.unlink(missing_ok=True)
            
        except Exception as e:
            await self.send_json(client_id, {
                "type": "error",
                "message": f"TTS error: {str(e)}"
            })
        
        # Notify audio end
        await self.send_json(client_id, {"type": "audio_end"})
    
    async def handle_audio_message(self, client_id: str, audio_data: str):
        """
        Handle audio data from the client (WebM blob).
        
        1. Decode base64 audio
        2. Convert WebM to WAV if needed
        3. Transcribe with ASR
        4. Process as text message
        """
        state = self._get_state(client_id)
        if not state:
            return  # Client disconnected
        
        asr = state.get_asr()
        
        try:
            # Decode base64 audio
            audio_bytes = base64.b64decode(audio_data)
            
            # Save to temp file (browser sends WebM)
            with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as f:
                f.write(audio_bytes)
                webm_path = Path(f.name)
            
            # Convert WebM to WAV using ffmpeg
            wav_path = webm_path.with_suffix(".wav")
            
            import subprocess
            try:
                result = subprocess.run(
                    [
                        "ffmpeg", "-y", "-i", str(webm_path),
                        "-ar", "16000",  # 16kHz for Whisper
                        "-ac", "1",      # mono
                        "-f", "wav",
                        str(wav_path)
                    ],
                    capture_output=True,
                    timeout=30
                )
                if result.returncode != 0:
                    raise Exception(f"ffmpeg error: {result.stderr.decode()}")
            except FileNotFoundError:
                raise Exception("ffmpeg not found. Please install ffmpeg.")
            
            # Cleanup webm
            webm_path.unlink(missing_ok=True)
            
            # Transcribe (with forced language)
            await self.send_json(client_id, {"type": "transcribing"})
            language = getattr(state, 'asr_language', 'fr')
            result = asr.transcribe(wav_path, language=language)
            
            # Cleanup wav
            wav_path.unlink(missing_ok=True)
            
            if result.text.strip():
                # Send transcription
                await self.send_json(client_id, {
                    "type": "transcription",
                    "text": result.text,
                    "language": result.language
                })
                
                # Process as text message
                await self.handle_text_message(client_id, result.text, language=result.language)
            else:
                await self.send_json(client_id, {
                    "type": "transcription",
                    "text": "",
                    "message": "No speech detected"
                })
                
        except Exception as e:
            print(f"❌ ASR error: {e}")
            await self.send_json(client_id, {
                "type": "error",
                "message": f"ASR error: {str(e)}"
            })
    
    async def handle_audio_stream(self, client_id: str, audio_samples: list):
        """
        Handle streaming audio data with VAD.
        
        The frontend sends raw PCM samples (float32, 16kHz).
        VAD detects speech end and triggers transcription.
        """
        state = self._get_state(client_id)
        if not state:
            return  # Client disconnected
        
        vad = state.get_vad()
        
        try:
            # Process through VAD
            for event in vad.process_audio(audio_samples):
                if event == b"<|START|>":
                    # Speech started
                    state.is_recording = True
                    await self.send_json(client_id, {"type": "vad_start"})
                    
                elif event == b"<|END|>":
                    # Speech ended - will receive audio bytes next
                    state.is_recording = False
                    await self.send_json(client_id, {"type": "vad_end"})
                    
                elif len(event) > 100:
                    # This is audio data - transcribe it!
                    # Run in background to avoid blocking VAD loop (and delaying vad_end)
                    asyncio.create_task(self._transcribe_and_respond(client_id, event))
                    
        except Exception as e:
            print(f"❌ VAD error: {e}")
            await self.send_json(client_id, {
                "type": "error",
                "message": f"VAD error: {str(e)}"
            })
    
    async def _transcribe_and_respond(self, client_id: str, audio_bytes: bytes):
        """Transcribe audio bytes and generate response."""
        state = self._get_state(client_id)
        if not state:
            return  # Client disconnected
        
        asr = state.get_asr()
        
        try:
            # Convert int16 bytes to float32 for Whisper
            audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
            audio_float = audio_int16.astype(np.float32) / 32767.0
            
            # Calculate audio duration
            duration_sec = len(audio_int16) / 16000
            print(f"🎤 Audio received: {len(audio_bytes)} bytes, {duration_sec:.2f}s, {len(audio_int16)} samples")
            
            # Check if audio is too short (< 0.5s often causes hallucinations)
            if duration_sec < 0.5:
                print("⚠️ Audio too short (< 0.5s), ignored")
                await self.send_json(client_id, {
                    "type": "transcription",
                    "text": "",
                    "message": "Audio too short"
                })
                return
            
            # Transcribe (pass numpy array directly to avoid disk I/O)
            await self.send_json(client_id, {"type": "transcribing"})
            language = getattr(state, 'asr_language', 'fr')
            
            # Run in thread pool to avoid blocking event loop
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, 
                lambda: asr.transcribe(audio_float, language=language)
            )
            
            if result.text.strip():
                await self.send_json(client_id, {
                    "type": "transcription",
                    "text": result.text,
                    "language": result.language
                })
                
                # Process as text message
                await self.handle_text_message(client_id, result.text, language=result.language)
            else:
                await self.send_json(client_id, {
                    "type": "transcription",
                    "text": "",
                    "message": "No speech detected"
                })
                
        except Exception as e:
            print(f"❌ Transcription error: {e}")
            await self.send_json(client_id, {
                "type": "error", 
                "message": f"Transcription error: {str(e)}"
            })
    
    async def handle_clear(self, client_id: str):
        """Clear conversation history."""
        state = self._get_state(client_id)
        if not state:
            return  # Client disconnected
        
        character = state.config.get("character", {})
        system_prompt = character.get("system_prompt", "You are a helpful assistant.")
        state.messages = [Message(role="system", content=system_prompt)]
        
        await self.send_json(client_id, {
            "type": "cleared",
            "message": "Conversation history cleared"
        })
    
    async def _preload_models_progressive(self, client_id: str):
        """
        Preload models progressively (one by one) on connection.
        
        This runs in background and loads models sequentially to:
        1. Avoid memory spikes from parallel loading
        2. Provide feedback to user as each model loads
        3. Prioritize TTS (most needed for quick response)
        
        Inspired by Open-LLM-VTuber's approach to model management.
        """
        state = self.states.get(client_id)
        if not state:
            return
        
        # Avoid duplicate preloading
        if self._preloading.get(client_id, False):
            return
        self._preloading[client_id] = True
        
        loop = asyncio.get_event_loop()
        
        def is_connected() -> bool:
            """Check if client is still connected."""
            return client_id in self.active_connections
        
        async def safe_send(data: dict):
            """Send message only if client still connected."""
            if is_connected():
                await self.send_json(client_id, data)
        
        try:
            # Small delay to ensure WebSocket is fully established
            await asyncio.sleep(0.2)
            
            if not is_connected():
                return
            
            # Notify start
            await safe_send({
                "type": "models_loading",
                "message": "Preparing voice models...",
                "progress": 0
            })
            
            # 1. Load VAD first (lightest, ~5MB, needed for speech detection)
            if not state.vad and is_connected():
                await safe_send({
                    "type": "model_loading",
                    "model": "vad",
                    "message": "Loading VAD (voice detection)...",
                    "progress": 5
                })
                try:
                    await loop.run_in_executor(None, state.get_vad)
                    print(f"   ✅ VAD loaded for {client_id}")
                    await safe_send({
                        "type": "model_loaded",
                        "model": "vad",
                        "message": "VAD ready!",
                        "progress": 15
                    })
                except Exception as e:
                    print(f"   ⚠️ VAD load warning: {e}")
            
            # 2. Load TTS (needed for responses, ~2.8GB for XTTS)
            if not state.tts and is_connected():
                await safe_send({
                    "type": "model_loading",
                    "model": "tts",
                    "message": "Loading TTS (voice synthesis)...",
                    "progress": 20
                })
                try:
                    await loop.run_in_executor(None, state.get_tts)
                    print(f"   ✅ TTS loaded for {client_id}")
                    await safe_send({
                        "type": "model_loaded",
                        "model": "tts",
                        "message": "TTS ready!",
                        "progress": 60
                    })
                except Exception as e:
                    print(f"   ❌ TTS load error: {e}")
                    await safe_send({
                        "type": "models_error",
                        "message": f"TTS failed: {str(e)}"
                    })
            
            # 3. Load ASR last (can be heavy, ~600MB for Parakeet)
            if not state.asr and is_connected():
                await safe_send({
                    "type": "model_loading",
                    "model": "asr",
                    "message": "Loading ASR (speech recognition)...",
                    "progress": 65
                })
                try:
                    await loop.run_in_executor(None, state.get_asr)
                    print(f"   ✅ ASR loaded for {client_id}")
                    await safe_send({
                        "type": "model_loaded",
                        "model": "asr",
                        "message": "ASR ready!",
                        "progress": 100
                    })
                except Exception as e:
                    print(f"   ❌ ASR load error: {e}")
                    await safe_send({
                        "type": "models_error",
                        "message": f"ASR failed: {str(e)}"
                    })
            
            # All done!
            if is_connected():
                await safe_send({
                    "type": "models_ready",
                    "message": "All voice models loaded!",
                    "progress": 100
                })
                print(f"✅ All models preloaded for {client_id}")
            
        except asyncio.CancelledError:
            print(f"⚠️ Preloading cancelled for {client_id} (client disconnected)")
        except Exception as e:
            print(f"❌ Preloading error for {client_id}: {e}")
            import traceback
            traceback.print_exc()
            if is_connected():
                await safe_send({
                    "type": "models_error",
                    "message": f"Failed to preload models: {str(e)}"
                })
        finally:
            self._preloading[client_id] = False
    
    async def preload_models(self, client_id: str):
        """
        Preload all models in parallel when user clicks mic.
        This ensures smooth conversation without loading delays.
        """
        # Avoid duplicate preloading
        if self._preloading.get(client_id, False):
            return
        
        state = self.states.get(client_id)
        if not state:
            return
        
        # Check if already loaded
        if state.vad and state.asr and state.tts:
            await self.send_json(client_id, {
                "type": "models_ready",
                "message": "Models already loaded"
            })
            return
        
        self._preloading[client_id] = True
        
        await self.send_json(client_id, {
            "type": "models_loading",
            "message": "Loading voice models..."
        })
        
        try:
            # Load models in parallel using asyncio
            loop = asyncio.get_event_loop()
            
            # These are blocking calls, run in thread pool
            async def load_vad():
                if not state.vad:
                    await loop.run_in_executor(None, state.get_vad)
                    
            async def load_asr():
                if not state.asr:
                    await loop.run_in_executor(None, state.get_asr)
                    
            async def load_tts():
                if not state.tts:
                    await loop.run_in_executor(None, state.get_tts)
            
            # Run all in parallel
            await asyncio.gather(load_vad(), load_asr(), load_tts())
            
            await self.send_json(client_id, {
                "type": "models_ready",
                "message": "Voice models loaded!"
            })
            
        except Exception as e:
            print(f"❌ Model preloading error: {e}")
            await self.send_json(client_id, {
                "type": "error",
                "message": f"Failed to load models: {str(e)}"
            })
        finally:
            self._preloading[client_id] = False


# Global manager instance
manager = WebSocketManager()


@websocket_router.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    """
    WebSocket endpoint for real-time conversation.
    
    Protocol:
    - Client sends: {"type": "text|audio|clear", "content|data": "..."}
    - Server sends: {"type": "text_start|text_chunk|text_end|audio_start|audio_data|audio_end|error", ...}
    """
    await manager.connect(websocket, client_id)
    
    try:
        while True:
            # Receive message
            data = await websocket.receive_json()
            msg_type = data.get("type", "text")
            
            if msg_type == "text":
                content = data.get("content", "")
                if content.strip():
                    # Detect language for text input
                    lang = None
                    try:
                        from langdetect import detect
                        detected = detect(content)
                        if detected.startswith('fr'): lang = 'fr'
                        elif detected.startswith('en'): lang = 'en'
                    except:
                        pass
                        
                    await manager.handle_text_message(client_id, content, language=lang)
                    
            elif msg_type == "audio":
                # WebM blob audio (fallback mode)
                audio_data = data.get("data", "")
                if audio_data:
                    await manager.handle_audio_message(client_id, audio_data)
                    
            elif msg_type == "audio_stream":
                # Streaming raw PCM with VAD
                audio_samples = data.get("samples", [])
                if audio_samples:
                    await manager.handle_audio_stream(client_id, audio_samples)
                    
            elif msg_type == "mic_stop":
                # User manually stopped mic - force end VAD
                state = manager.states.get(client_id)
                if state and state.vad:
                    audio_bytes = state.vad.force_end()
                    if audio_bytes:
                        await manager._transcribe_and_respond(client_id, audio_bytes)
                    
            elif msg_type == "clear":
                await manager.handle_clear(client_id)
                
            elif msg_type == "ping":
                await manager.send_json(client_id, {"type": "pong"})
                
            elif msg_type == "preload_models":
                # Preload models in background when user clicks mic
                await manager.preload_models(client_id)
                
    except WebSocketDisconnect:
        await manager.disconnect(client_id)
    except Exception as e:
        import traceback
        print(f"❌ WebSocket error for {client_id}: {type(e).__name__}: {e}")
        traceback.print_exc()
        await manager.disconnect(client_id)
