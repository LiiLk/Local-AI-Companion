"""
HTTP Routes - REST API endpoints.

Provides configuration, health checks, and editable Settings v1 endpoints.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from src.llm.ollama_llm import OllamaLLM
from src.llm.openrouter_llm import OpenRouterLLM
from src.utils.character_loader import resolve_character_config
from src.utils.config_loader import (
    load_local_yaml_config,
    load_yaml_config,
    write_local_yaml_config,
)

router = APIRouter(tags=["API"])


class HealthResponse(BaseModel):
    status: str
    character: str
    version: str


class ConfigResponse(BaseModel):
    mode: str
    character_name: str
    llm_provider: str
    llm_model: str
    tts_provider: str
    tts_voice: str
    asr_model: str


class OllamaSettingsPayload(BaseModel):
    base_url: str
    model: str


class OpenRouterSettingsPayload(BaseModel):
    base_url: str
    model: str
    api_key: str | None = None


class OpenRouterSettingsResponse(BaseModel):
    base_url: str
    model: str
    api_key_configured: bool
    api_key_source: str


class LLMSettingsPayload(BaseModel):
    provider: Literal["ollama", "openrouter"]
    ollama: OllamaSettingsPayload | None = None
    openrouter: OpenRouterSettingsPayload | None = None


class LLMSettingsResponse(BaseModel):
    provider: Literal["ollama", "openrouter"]
    ollama: OllamaSettingsPayload
    openrouter: OpenRouterSettingsResponse
    apply_strategy: str = "reload_page"


class LLMSettingsSaveResponse(LLMSettingsResponse):
    message: str


class LLMSettingsTestResponse(BaseModel):
    ok: bool
    message: str


def _project_config_path() -> Path:
    return Path(__file__).parent.parent.parent / "config" / "config.yaml"


def _strip_required(value: str | None, field_name: str) -> str:
    normalized = (value or "").strip()
    if not normalized:
        raise HTTPException(status_code=422, detail=f"{field_name} is required.")
    return normalized


def _refresh_app_state(request: Request, config_path: Path) -> None:
    resolved_config = resolve_character_config(load_yaml_config(config_path))
    request.app.state.config = resolved_config
    request.app.state.character = resolved_config.get("character", {})


def _get_openrouter_key_source(config: dict[str, Any]) -> tuple[bool, str]:
    llm_config = config.get("llm", {})
    openrouter_config = llm_config.get("openrouter", {})
    saved_key = str(openrouter_config.get("api_key") or "").strip()
    if saved_key:
        return True, "saved"

    env_name = str(openrouter_config.get("api_key_env", "OPENROUTER_API_KEY") or "OPENROUTER_API_KEY")
    env_value = os.getenv(env_name)
    if env_value:
        return True, f"env:{env_name}"

    return False, "missing"


def _build_llm_settings_response(config: dict[str, Any]) -> LLMSettingsResponse:
    llm_config = config.get("llm", {})
    ollama_config = llm_config.get("ollama", {})
    openrouter_config = llm_config.get("openrouter", {})
    api_key_configured, api_key_source = _get_openrouter_key_source(config)

    return LLMSettingsResponse(
        provider=llm_config.get("provider", "ollama"),
        ollama=OllamaSettingsPayload(
            base_url=ollama_config.get("base_url", "http://localhost:11434"),
            model=ollama_config.get("model", "llama3.2:3b"),
        ),
        openrouter=OpenRouterSettingsResponse(
            base_url=openrouter_config.get("base_url", "https://openrouter.ai/api/v1"),
            model=openrouter_config.get("model", "openai/gpt-4.1-mini"),
            api_key_configured=api_key_configured,
            api_key_source=api_key_source,
        ),
    )


def _normalize_settings_payload(
    payload: LLMSettingsPayload,
    existing_config: dict[str, Any],
) -> dict[str, Any]:
    normalized: dict[str, Any] = {"provider": payload.provider}

    if payload.provider == "ollama":
        if payload.ollama is None:
            raise HTTPException(status_code=422, detail="Ollama settings are required.")
        normalized["ollama"] = {
            "base_url": _strip_required(payload.ollama.base_url, "ollama.base_url"),
            "model": _strip_required(payload.ollama.model, "ollama.model"),
        }
        return normalized

    if payload.openrouter is None:
        raise HTTPException(status_code=422, detail="OpenRouter settings are required.")

    api_key = (payload.openrouter.api_key or "").strip() or None
    api_key_configured, _ = _get_openrouter_key_source(existing_config)
    if api_key is None and not api_key_configured:
        raise HTTPException(
            status_code=422,
            detail="An OpenRouter API key is required the first time you enable the API provider.",
        )

    normalized["openrouter"] = {
        "base_url": _strip_required(payload.openrouter.base_url, "openrouter.base_url"),
        "model": _strip_required(payload.openrouter.model, "openrouter.model"),
    }
    if api_key is not None:
        normalized["openrouter"]["api_key"] = api_key
    return normalized


def _merge_llm_settings_into_local_override(
    local_override: dict[str, Any],
    normalized_settings: dict[str, Any],
) -> dict[str, Any]:
    updated = dict(local_override)
    llm_override = dict(updated.get("llm") or {})

    llm_override["provider"] = normalized_settings["provider"]

    if normalized_settings["provider"] == "ollama":
        ollama_override = dict(llm_override.get("ollama") or {})
        ollama_override.update(normalized_settings["ollama"])
        llm_override["ollama"] = ollama_override
    else:
        openrouter_override = dict(llm_override.get("openrouter") or {})
        openrouter_override.update(normalized_settings["openrouter"])
        llm_override["openrouter"] = openrouter_override

    updated["llm"] = llm_override
    return updated


async def _close_maybe_async(value: Any) -> None:
    close = getattr(value, "close", None)
    if not callable(close):
        return
    result = close()
    if asyncio.iscoroutine(result):
        await result


async def _test_llm_settings(
    normalized_settings: dict[str, Any],
    existing_config: dict[str, Any],
) -> tuple[bool, str]:
    if normalized_settings["provider"] == "ollama":
        ollama_settings = normalized_settings["ollama"]
        llm = OllamaLLM(
            model=ollama_settings["model"],
            base_url=ollama_settings["base_url"],
            request_timeout_sec=15,
            preload_timeout_sec=15,
        )
        label = f"Ollama ({ollama_settings['model']})"
    else:
        openrouter_settings = normalized_settings["openrouter"]
        llm = OpenRouterLLM(
            model=openrouter_settings["model"],
            base_url=openrouter_settings["base_url"],
            api_key=openrouter_settings.get("api_key"),
            api_key_env=existing_config.get("llm", {})
            .get("openrouter", {})
            .get("api_key_env", "OPENROUTER_API_KEY"),
            app_url=existing_config.get("llm", {}).get("openrouter", {}).get("app_url"),
            app_title=existing_config.get("llm", {}).get("openrouter", {}).get("app_title", "Local-AI-Companion"),
            options=existing_config.get("llm", {}).get("openrouter", {}).get("options"),
            required_input_modalities=[],
            request_timeout_sec=20,
            preload_timeout_sec=20,
        )
        label = f"OpenRouter ({openrouter_settings['model']})"

    try:
        await asyncio.to_thread(llm.preload)
    except Exception as exc:
        return False, f"{label} test failed: {exc}"
    finally:
        await _close_maybe_async(llm)

    return True, f"{label} connection succeeded."


@router.get("/health", response_model=HealthResponse)
async def health_check(request: Request):
    """
    Health check endpoint.

    Returns server status and basic info.
    """
    character = getattr(request.app.state, "character", {})
    return HealthResponse(
        status="ok",
        character=character.get("name", "AI"),
        version="0.2.0",
    )


@router.get("/config", response_model=ConfigResponse)
async def get_config(request: Request):
    """
    Get current server configuration.

    Returns character, LLM, TTS, and ASR settings.
    """
    config = getattr(request.app.state, "config", {})
    mode = config.get("mode", "pipeline")
    character = config.get("character", {})
    tts = config.get("tts", {})
    asr = config.get("asr", {})

    if mode == "omni":
        omni_config = config.get("omni", {}).get("minicpmo", {})
        llm_provider = "minicpmo"
        llm_model = omni_config.get("model_id", "openbmb/MiniCPM-o-4_5")
        tts_provider = "minicpmo"
        tts_voice = "minicpmo"
        asr_model = "minicpmo"
    else:
        llm_config = config.get("llm", {})
        llm_provider = llm_config.get("provider", "ollama")
        if llm_provider == "gemma":
            llm_model = config.get("gemma", {}).get("model_id", "google/gemma-4-E4B-it")
        elif llm_provider == "openrouter":
            llm_model = llm_config.get("openrouter", {}).get("model", "unknown")
        else:
            llm_model = llm_config.get("ollama", {}).get("model", "unknown")
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
        llm_provider=llm_provider,
        llm_model=llm_model,
        tts_provider=tts_provider,
        tts_voice=tts_voice,
        asr_model=asr_model,
    )


@router.get("/settings/llm", response_model=LLMSettingsResponse)
async def get_llm_settings():
    config = load_yaml_config(_project_config_path())
    return _build_llm_settings_response(config)


@router.put("/settings/llm", response_model=LLMSettingsSaveResponse)
async def save_llm_settings(payload: LLMSettingsPayload, request: Request):
    config_path = _project_config_path()
    merged_config = load_yaml_config(config_path)
    local_override = load_local_yaml_config(config_path)
    normalized_settings = _normalize_settings_payload(payload, merged_config)
    updated_local_override = _merge_llm_settings_into_local_override(local_override, normalized_settings)
    write_local_yaml_config(config_path, updated_local_override)
    _refresh_app_state(request, config_path)

    updated_config = load_yaml_config(config_path)
    settings_response = _build_llm_settings_response(updated_config)
    return LLMSettingsSaveResponse(
        **settings_response.model_dump(),
        message="Settings saved to config.local.yaml. Reloading the page applies them to a new session.",
    )


@router.post("/settings/llm/test", response_model=LLMSettingsTestResponse)
async def test_llm_settings(payload: LLMSettingsPayload):
    merged_config = load_yaml_config(_project_config_path())
    normalized_settings = _normalize_settings_payload(payload, merged_config)
    ok, message = await _test_llm_settings(normalized_settings, merged_config)
    return LLMSettingsTestResponse(ok=ok, message=message)


@router.get("/voices")
async def list_voices():
    """
    List available TTS voices.
    """
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
        "default": "ff_siwis",
    }
