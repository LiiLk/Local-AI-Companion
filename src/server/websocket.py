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
import tempfile
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass, field

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import yaml

from src.llm import OllamaLLM
from src.llm.base import Message
from src.tts import KokoroProvider, EdgeTTSProvider
from src.asr import WhisperProvider
from src.vad import SileroVAD

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
    
    def __post_init__(self):
        self.config = load_config()
        
    async def initialize(self):
        """Initialize models (lazy loading)."""
        if self.llm is None:
            llm_config = self.config["llm"]["ollama"]
            self.llm = OllamaLLM(
                model=llm_config["model"],
                base_url=llm_config["base_url"]
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
            
            if provider == "kokoro":
                voice = tts_config.get("kokoro_voice", "ff_siwis")
                self.tts = KokoroProvider(voice=voice)
            else:
                voice = tts_config.get("voice", "fr-FR-DeniseNeural")
                self.tts = EdgeTTSProvider(voice=voice)
                
        return self.tts
    
    def get_asr(self):
        """Get or create ASR provider (lazy loading)."""
        if self.asr is None:
            asr_config = self.config.get("asr", {})
            model_size = asr_config.get("model_size", "base")
            device = asr_config.get("device", "cpu")
            # Prompt helps Whisper detect the correct language without forcing it
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
    
    async def connect(self, websocket: WebSocket, client_id: str):
        """Accept a new WebSocket connection."""
        await websocket.accept()
        self.active_connections[client_id] = websocket
        self.states[client_id] = ConversationState()
        await self.states[client_id].initialize()
        print(f"✅ Client connected: {client_id}")
    
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
            await self.active_connections[client_id].send_json(data)
    
    async def handle_text_message(self, client_id: str, content: str):
        """
        Handle a text message from the client.
        
        1. Add user message to history
        2. Stream LLM response
        3. Generate and stream TTS audio
        """
        state = self.states[client_id]
        
        # Add user message
        state.messages.append(Message(role="user", content=content))
        
        # Notify start of response
        await self.send_json(client_id, {"type": "text_start"})
        
        # Stream LLM response
        full_response = ""
        async for chunk in state.llm.chat_stream(state.messages):
            full_response += chunk
            await self.send_json(client_id, {
                "type": "text_chunk",
                "content": chunk
            })
        
        # Add assistant message to history
        state.messages.append(Message(role="assistant", content=full_response))
        
        # Notify end of text
        await self.send_json(client_id, {
            "type": "text_end",
            "full_text": full_response
        })
        
        # Generate TTS audio
        await self.generate_and_send_audio(client_id, full_response)
    
    async def generate_and_send_audio(self, client_id: str, text: str):
        """Generate TTS audio and send to client."""
        if not text.strip():
            return
            
        state = self.states[client_id]
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
        state = self.states[client_id]
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
                await self.handle_text_message(client_id, result.text)
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
        state = self.states[client_id]
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
                    await self._transcribe_and_respond(client_id, event)
                    
        except Exception as e:
            print(f"❌ VAD error: {e}")
            await self.send_json(client_id, {
                "type": "error",
                "message": f"VAD error: {str(e)}"
            })
    
    async def _transcribe_and_respond(self, client_id: str, audio_bytes: bytes):
        """Transcribe audio bytes and generate response."""
        state = self.states[client_id]
        asr = state.get_asr()
        
        try:
            # Convert int16 bytes to float32 for Whisper
            audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
            audio_float = audio_int16.astype(np.float32) / 32767.0
            
            # Save to temp WAV file
            import wave
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                wav_path = Path(f.name)
            
            with wave.open(str(wav_path), 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(16000)
                wf.writeframes(audio_bytes)
            
            # Transcribe (with forced language)
            await self.send_json(client_id, {"type": "transcribing"})
            language = getattr(state, 'asr_language', 'fr')
            result = asr.transcribe(wav_path, language=language)
            
            # Cleanup
            wav_path.unlink(missing_ok=True)
            
            if result.text.strip():
                await self.send_json(client_id, {
                    "type": "transcription",
                    "text": result.text,
                    "language": result.language
                })
                
                # Process as text message
                await self.handle_text_message(client_id, result.text)
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
        state = self.states[client_id]
        character = state.config.get("character", {})
        system_prompt = character.get("system_prompt", "You are a helpful assistant.")
        state.messages = [Message(role="system", content=system_prompt)]
        
        await self.send_json(client_id, {
            "type": "cleared",
            "message": "Conversation history cleared"
        })


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
                    await manager.handle_text_message(client_id, content)
                    
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
                
    except WebSocketDisconnect:
        await manager.disconnect(client_id)
    except Exception as e:
        print(f"❌ WebSocket error: {e}")
        await manager.disconnect(client_id)
