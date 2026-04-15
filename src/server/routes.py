"""
HTTP Routes - REST API endpoints.

Provides configuration and health check endpoints.
"""

from fastapi import APIRouter, Request
from pydantic import BaseModel
from typing import Optional

router = APIRouter(tags=["API"])


class HealthResponse(BaseModel):
    status: str
    character: str
    version: str


class ConfigResponse(BaseModel):
    mode: str
    character_name: str
    llm_model: str
    tts_provider: str
    tts_voice: str
    asr_model: str


@router.get("/health", response_model=HealthResponse)
async def health_check(request: Request):
    """
    Health check endpoint.
    
    Returns server status and basic info.
    """
    character = getattr(request.app.state, 'character', {})
    return HealthResponse(
        status="ok",
        character=character.get("name", "AI"),
        version="0.2.0"
    )


@router.get("/config", response_model=ConfigResponse)
async def get_config(request: Request):
    """
    Get current server configuration.
    
    Returns character, LLM, TTS, and ASR settings.
    """
    config = getattr(request.app.state, 'config', {})
    mode = config.get("mode", "pipeline")
    character = config.get("character", {})
    tts = config.get("tts", {})
    asr = config.get("asr", {})

    if mode == "omni":
        omni_config = config.get("omni", {}).get("minicpmo", {})
        llm_model = omni_config.get("model_id", "openbmb/MiniCPM-o-4_5")
        tts_provider = "minicpmo"
        tts_voice = "minicpmo"
        asr_model = "minicpmo"
    else:
        llm_config = config.get("llm", {})
        llm_provider = llm_config.get("provider", "ollama")
        if llm_provider == "gemma":
            llm_model = config.get("gemma", {}).get("model_id", "google/gemma-4-E4B-it")
        else:
            llm = llm_config.get("ollama", {})
            llm_model = llm.get("model", "unknown")
        tts_provider = tts.get("provider", "kokoro")
        if tts_provider == "chatterbox":
            tts_voice = config.get("character", {}).get("voice", {}).get("chatterbox_ref_audio", "chatterbox")
        elif tts_provider == "qwen3":
            tts_voice = (
                config.get("character", {}).get("voice", {}).get("qwen_ref_audio")
                or tts.get("qwen3", {}).get("ref_audio_path", "qwen3")
            )
        else:
            tts_voice = tts.get("kokoro_voice", tts.get("voice", "ff_siwis"))
        asr_model = asr.get("model_size", "base")

    return ConfigResponse(
        mode=mode,
        character_name=character.get("name", "AI"),
        llm_model=llm_model,
        tts_provider=tts_provider,
        tts_voice=tts_voice,
        asr_model=asr_model
    )


@router.get("/voices")
async def list_voices():
    """
    List available TTS voices.
    """
    # Kokoro voices
    kokoro_voices = [
        {"id": "af_heart", "name": "Heart (American Female)", "language": "en-US"},
        {"id": "af_alloy", "name": "Alloy (American Female)", "language": "en-US"},
        {"id": "af_aoede", "name": "Aoede (American Female)", "language": "en-US"},
        {"id": "am_adam", "name": "Adam (American Male)", "language": "en-US"},
        {"id": "am_echo", "name": "Echo (American Male)", "language": "en-US"},
        {"id": "bf_emma", "name": "Emma (British Female)", "language": "en-GB"},
        {"id": "bm_george", "name": "George (British Male)", "language": "en-GB"},
        {"id": "ff_siwis", "name": "Siwis (French Female)", "language": "fr-FR"},
        {"id": "jf_alpha", "name": "Alpha (Japanese Female)", "language": "ja-JP"},
        {"id": "jf_gongitsune", "name": "Gongitsune (Japanese Female)", "language": "ja-JP"},
        {"id": "jm_kumo", "name": "Kumo (Japanese Male)", "language": "ja-JP"},
    ]
    
    return {
        "kokoro": kokoro_voices,
        "default": "ff_siwis"
    }
