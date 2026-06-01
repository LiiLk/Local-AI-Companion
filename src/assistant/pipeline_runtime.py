"""Shared construction helpers for the pipeline-mode backend stack."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import inspect
import logging
from typing import Any

from src.assistant.conversation_memory import (
    ConversationMemoryStore,
    create_conversation_memory,
)
from src.assistant.conversation_pipeline import ConversationConfig
from src.utils.rvc_config import build_rvc_runtime_config

logger = logging.getLogger(__name__)


@dataclass
class PipelineRuntimeComponents:
    llm: Any
    tts: Any
    asr: Any
    rvc: Any | None
    memory: ConversationMemoryStore | None
    conversation_config: ConversationConfig
    llm_summary: str
    tts_summary: str
    asr_summary: str
    rvc_summary: str | None


@dataclass
class PipelineBackendStatus:
    state: str
    degraded_reason: str | None
    runtime_error: str | None


class PipelineRuntime:
    """Own the lifecycle and readiness state of pipeline-mode services."""

    def __init__(
        self,
        config: dict,
        *,
        initial_tts_language: str | None = None,
    ):
        self.config = config
        self._tts_language = resolve_initial_tts_language(config, initial_tts_language)
        self.llm: Any | None = None
        self.tts: Any | None = None
        self.asr: Any | None = None
        self.rvc: Any | None = None
        self.memory: ConversationMemoryStore | None = None
        self.llm_summary: str | None = None
        self.tts_summary: str | None = None
        self.asr_summary: str | None = None
        self.rvc_summary: str | None = None

    def build_conversation_config(self) -> ConversationConfig:
        return build_pipeline_conversation_config(self.config)

    def ensure_llm(self) -> Any:
        if self.llm is None:
            self.llm, self.llm_summary = create_pipeline_llm(self.config)
        return self.llm

    def ensure_tts(self, language: str | None = None) -> Any:
        if language:
            self._tts_language = resolve_initial_tts_language(self.config, language)
        if self.tts is None:
            self.tts, self.tts_summary = create_pipeline_tts(
                self.config,
                initial_language=self._tts_language,
            )
        elif language:
            self.set_tts_language(language)
        return self.tts

    def ensure_asr(self) -> Any:
        if self.asr is None:
            self.asr, self.asr_summary = create_pipeline_asr(self.config)
        return self.asr

    def ensure_rvc(self) -> Any | None:
        if self.rvc is None:
            self.rvc, self.rvc_summary = create_pipeline_rvc(self.config)
        return self.rvc

    def ensure_memory(self) -> ConversationMemoryStore | None:
        if self.memory is None:
            self.memory = create_conversation_memory(self.config)
        return self.memory

    def build_components(self) -> PipelineRuntimeComponents:
        return PipelineRuntimeComponents(
            llm=self.ensure_llm(),
            tts=self.ensure_tts(),
            asr=self.ensure_asr(),
            rvc=self.ensure_rvc(),
            memory=self.ensure_memory(),
            conversation_config=self.build_conversation_config(),
            llm_summary=self.llm_summary or "unknown",
            tts_summary=self.tts_summary or "unknown",
            asr_summary=self.asr_summary or "unknown",
            rvc_summary=self.rvc_summary,
        )

    def set_tts_language(self, language: str | None) -> None:
        resolved = resolve_initial_tts_language(self.config, language)
        if resolved:
            self._tts_language = resolved
        if self.tts is not None:
            _maybe_set_tts_language(self.tts, self._tts_language)

    def preload_llm(self) -> Any:
        self.llm = preload_pipeline_llm(self.ensure_llm())
        return self.llm

    def preload_asr(self) -> Any:
        self.asr = preload_pipeline_asr(self.ensure_asr())
        return self.asr

    def preload_tts(
        self,
        *,
        on_load_error: Callable[[Any, Exception], Any] | None = None,
    ) -> Any:
        self.tts = preload_pipeline_tts(
            self.ensure_tts(),
            warmup=bool(self.config.get("tts", {}).get("warmup_on_start", False)),
            on_load_error=on_load_error,
        )
        return self.tts

    def preload_rvc(self) -> Any | None:
        self.rvc = preload_pipeline_rvc(
            self.ensure_rvc(),
            warmup=bool(self.config.get("tts", {}).get("rvc", {}).get("enabled", False)),
        )
        return self.rvc

    def preload_all(
        self,
        *,
        tts_on_load_error: Callable[[Any, Exception], Any] | None = None,
    ) -> tuple[Any, Any | None]:
        self.preload_llm()
        self.preload_asr()
        self.preload_tts(on_load_error=tts_on_load_error)
        self.preload_rvc()
        return self.tts, self.rvc

    async def close(self) -> None:
        await close_pipeline_runtime_services(
            llm=self.llm,
            tts=self.tts,
            asr=self.asr,
            rvc=self.rvc,
        )

    def collect_degraded_reason(self, extra_reason: str | None = None) -> str | None:
        parts: list[str] = []
        llm_reason = getattr(self.llm, "degraded_reason", None)
        if llm_reason:
            parts.append(str(llm_reason))

        tts_reason = getattr(self.tts, "degraded_reason", None)
        if tts_reason:
            parts.append(str(tts_reason))

        if extra_reason:
            parts.append(str(extra_reason))

        unique_parts: list[str] = []
        for part in parts:
            if part and part not in unique_parts:
                unique_parts.append(part)
        return " | ".join(unique_parts) if unique_parts else None

    def resolve_backend_status(
        self,
        *,
        requested_state: str = "ready",
        runtime_error: str | None = None,
        extra_degraded_reason: str | None = None,
    ) -> PipelineBackendStatus:
        degraded_reason = self.collect_degraded_reason(extra_reason=extra_degraded_reason)
        state = requested_state
        if runtime_error:
            state = "error"
        elif state != "warming_up" and degraded_reason:
            state = "degraded"
        return PipelineBackendStatus(
            state=state,
            degraded_reason=degraded_reason,
            runtime_error=runtime_error,
        )

    def is_ready(self) -> bool:
        return (
            self._llm_is_ready()
            and self.asr is not None
            and self.tts is not None
            and self._rvc_is_ready()
        )

    def _llm_is_ready(self) -> bool:
        if self.llm is None:
            return False
        if self.config.get("llm", {}).get("provider", "ollama") == "gemma":
            return bool(getattr(getattr(self.llm, "gemma", None), "_model", None))
        return True

    def _rvc_is_ready(self) -> bool:
        rvc_enabled = bool(self.config.get("tts", {}).get("rvc", {}).get("enabled", False))
        return self.rvc is not None or not rvc_enabled


def resolve_initial_tts_language(
    config: dict,
    fallback_language: str | None = None,
) -> str | None:
    return config.get("pipeline", {}).get("reply_language") or fallback_language


def build_pipeline_conversation_config(config: dict) -> ConversationConfig:
    character_config = config.get("character", {})
    tts_config = config.get("tts", {})
    asr_config = config.get("asr", {})
    return ConversationConfig(
        character_name=character_config.get("name", "AI"),
        system_prompt=character_config.get("system_prompt", "You are a helpful assistant."),
        stream_tts=tts_config.get("stream_tts", True),
        tts_max_queue_size=max(0, int(tts_config.get("max_queue_size", 8))),
        auto_detect_language=tts_config.get("auto_detect_language", True),
        asr_language=asr_config.get("language", "auto"),
        reply_language=config.get("pipeline", {}).get("reply_language"),
    )


def create_pipeline_llm(config: dict) -> tuple[Any, str]:
    from src.llm import GemmaTextVisionLLM, OllamaLLM, OpenRouterLLM

    llm_config = config.get("llm", {})
    llm_provider = llm_config.get("provider", "ollama")

    if llm_provider == "gemma":
        from src.omni import GemmaProvider

        gemma_config = config.get("gemma", {})
        gemma_model = GemmaProvider(
            model_id=gemma_config.get("model_id", "google/gemma-4-E2B-it"),
            device=gemma_config.get("device", "cuda"),
            quantization=gemma_config.get("quantization", "int4"),
            max_new_tokens=gemma_config.get("max_new_tokens", 96),
            temperature=gemma_config.get("temperature", 0.7),
            top_p=gemma_config.get("top_p", 0.95),
            context_max_turns=gemma_config.get("context_max_turns", 10),
            cpu_offload=gemma_config.get("cpu_offload", True),
            offload_dir=gemma_config.get("offload_dir"),
        )
        llm = GemmaTextVisionLLM(
            gemma=gemma_model,
            screen_config=gemma_config.get("screen", {}),
        )
        return llm, f"Gemma text+vision ({gemma_config.get('model_id', 'google/gemma-4-E2B-it')})"

    if llm_provider == "openrouter":
        openrouter_config = llm_config.get("openrouter", {})
        llm = OpenRouterLLM(
            model=openrouter_config.get("model", "openai/gpt-4.1-mini"),
            api_key=openrouter_config.get("api_key"),
            api_key_env=openrouter_config.get("api_key_env", "OPENROUTER_API_KEY"),
            base_url=openrouter_config.get("base_url", "https://openrouter.ai/api/v1"),
            app_url=openrouter_config.get("app_url"),
            app_title=openrouter_config.get("app_title", "Local-AI-Companion"),
            options=openrouter_config.get("options"),
            required_input_modalities=openrouter_config.get("required_input_modalities"),
            request_timeout_sec=openrouter_config.get("request_timeout_sec", 180),
            preload_timeout_sec=openrouter_config.get("preload_timeout_sec", 60),
        )
        return llm, f"OpenRouter ({openrouter_config.get('model', 'openai/gpt-4.1-mini')})"

    ollama_config = llm_config.get("ollama", {})
    llm = OllamaLLM(
        model=ollama_config.get("model", "llama3.2:3b"),
        base_url=ollama_config.get("base_url", "http://localhost:11434"),
        think=ollama_config.get("think"),
        options=ollama_config.get("options"),
        request_timeout_sec=ollama_config.get("request_timeout_sec", 180),
        preload_timeout_sec=ollama_config.get("preload_timeout_sec", 120),
    )
    return llm, f"Ollama ({ollama_config.get('model', 'llama3.2:3b')})"


def create_pipeline_tts(
    config: dict,
    *,
    initial_language: str | None = None,
) -> tuple[Any, str]:
    from src.tts import (
        ChatterboxTTSProvider,
        EdgeTTSProvider,
        KokoroProvider,
        Qwen3TTSProvider,
        RoutedTTSProvider,
    )

    voice_config = config.get("character", {}).get("voice", {})
    tts_config = config.get("tts", {})
    provider = tts_config.get("provider", "kokoro")

    if provider == "kokoro":
        voice = voice_config.get("kokoro_voice") or tts_config.get("kokoro_voice", "ff_siwis")
        tts = KokoroProvider(voice=voice)
        _maybe_set_tts_language(tts, initial_language)
        return tts, f"Kokoro ({voice})"

    if provider == "qwen3":
        qwen3_config = tts_config.get("qwen3", {})
        ref_audio = (
            voice_config.get("qwen_ref_audio")
            or voice_config.get("chatterbox_ref_audio")
            or voice_config.get("omni_ref_audio")
            or qwen3_config.get("ref_audio_path")
        )
        ref_text = voice_config.get("qwen_ref_text") or qwen3_config.get("ref_text")
        kokoro_voice = voice_config.get("kokoro_voice") or tts_config.get("kokoro_voice", "ff_siwis")
        kokoro = KokoroProvider(voice=kokoro_voice)
        chatterbox_config = tts_config.get("chatterbox", {})
        chatterbox = ChatterboxTTSProvider(
            model_id=chatterbox_config.get("model_id", "onnx-community/chatterbox-multilingual-ONNX"),
            ref_audio_path=voice_config.get("chatterbox_ref_audio"),
            exaggeration=voice_config.get("chatterbox_exaggeration", 0.5),
            cfg_weight=chatterbox_config.get("cfg_weight", 0.5),
            language=voice_config.get("chatterbox_language", "en"),
            prefer_full_gpu=chatterbox_config.get("prefer_full_gpu", True),
            model_revision=chatterbox_config.get("model_revision"),
        )
        qwen3_provider = None
        degraded_reason = None
        resolved_qwen_mode, qwen_mode_reason = Qwen3TTSProvider.resolve_mode_for_model(
            qwen3_config.get("model_id", "Qwen/Qwen3-TTS-12Hz-0.6B-Base"),
            qwen3_config.get("mode", "voice_clone"),
        )
        if qwen_mode_reason:
            degraded_reason = qwen_mode_reason
            logger.warning(qwen_mode_reason)
        if Qwen3TTSProvider.is_available(
            backend=qwen3_config.get("backend", "worker"),
            python_path=qwen3_config.get("python_path"),
            site_packages_dir=qwen3_config.get("site_packages_dir"),
            worker_script=qwen3_config.get("worker_script"),
        ):
            qwen3_provider = Qwen3TTSProvider(
                model_id=qwen3_config.get("model_id", "Qwen/Qwen3-TTS-12Hz-0.6B-Base"),
                mode=resolved_qwen_mode,
                language=qwen3_config.get("language", "auto"),
                speaker=qwen3_config.get("speaker"),
                instruct=qwen3_config.get("instruct"),
                ref_audio_path=ref_audio,
                ref_text=ref_text,
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
            runtime_reason = "Qwen3-TTS runtime is not installed. Falling back to Chatterbox/Kokoro."
            degraded_reason = f"{degraded_reason} {runtime_reason}".strip() if degraded_reason else runtime_reason

        tts = RoutedTTSProvider(
            qwen3=qwen3_provider,
            chatterbox=chatterbox,
            kokoro=kokoro,
            default_language=qwen3_config.get("language", "auto"),
            provider_order_by_language=tts_config.get("routed", {}).get("provider_order_by_language"),
            preload_fallbacks=tts_config.get("routed", {}).get("preload_fallbacks", False),
            warmup_fallbacks=tts_config.get("routed", {}).get("warmup_fallbacks", False),
        )
        if degraded_reason:
            tts.degraded_reason = degraded_reason
        _maybe_set_tts_language(tts, initial_language)
        summary = "routed (qwen3=%s, chatterbox=%s, kokoro=%s)" % (
            "ready" if qwen3_provider else "fallback-only",
            chatterbox.__class__.__name__,
            kokoro.voice,
        )
        return tts, summary

    if provider == "chatterbox":
        chatterbox_config = tts_config.get("chatterbox", {})
        ref_audio = voice_config.get("chatterbox_ref_audio")
        exaggeration = voice_config.get("chatterbox_exaggeration", 0.5)
        language = voice_config.get("chatterbox_language", "en")
        tts = ChatterboxTTSProvider(
            model_id=chatterbox_config.get("model_id", "onnx-community/chatterbox-multilingual-ONNX"),
            ref_audio_path=ref_audio,
            exaggeration=exaggeration,
            cfg_weight=chatterbox_config.get("cfg_weight", 0.5),
            language=language,
            prefer_full_gpu=chatterbox_config.get("prefer_full_gpu", True),
            model_revision=chatterbox_config.get("model_revision"),
        )
        _maybe_set_tts_language(tts, initial_language)
        return tts, f"Chatterbox (ref={ref_audio}, exag={exaggeration})"

    voice = tts_config.get("voice", "en-US-JennyNeural")
    tts = EdgeTTSProvider(voice=voice)
    _maybe_set_tts_language(tts, initial_language)
    return tts, f"Edge ({voice})"


def create_pipeline_asr(config: dict) -> tuple[Any, str]:
    asr_config = config.get("asr", {})
    asr_provider = asr_config.get("provider", "whisper")

    if asr_provider == "qwen3":
        from src.asr.qwen3_asr_provider import Qwen3ASRProvider

        qwen3_config = asr_config.get("qwen3", {})
        asr = Qwen3ASRProvider(
            model_id=qwen3_config.get("model_id", "Qwen/Qwen3-ASR-0.6B"),
            device=qwen3_config.get("device", "cuda:0"),
            dtype=qwen3_config.get("dtype", "bfloat16"),
            max_new_tokens=qwen3_config.get("max_new_tokens", 256),
            backend=qwen3_config.get("backend", "worker"),
            python_path=qwen3_config.get("python_path"),
            site_packages_dir=qwen3_config.get("site_packages_dir"),
            worker_script=qwen3_config.get("worker_script"),
        )
        return asr, f"Qwen3-ASR ({qwen3_config.get('model_id', 'Qwen/Qwen3-ASR-0.6B')})"

    from src.asr import WhisperProvider

    device = asr_config.get("device", "cpu")
    model_size = asr_config.get("model_size", "base")
    compute_type = asr_config.get("compute_type", "float16")
    prompt = asr_config.get("prompt")
    asr = WhisperProvider(
        model_size=model_size,
        device=device,
        compute_type=compute_type,
        initial_prompt=prompt,
        beam_size=asr_config.get("beam_size", 1),
    )
    return asr, f"Whisper {model_size} on {device}"


def create_pipeline_rvc(config: dict) -> tuple[Any | None, str | None]:
    from src.tts.rvc_provider import RVCConverter

    rvc_config = build_rvc_runtime_config(config)
    if not rvc_config:
        return None, None

    if not RVCConverter.is_available(
        backend=rvc_config.get("backend", "auto"),
        python_path=rvc_config.get("python_path"),
        site_packages_dir=rvc_config.get("site_packages_dir"),
        worker_script=rvc_config.get("worker_script"),
    ):
        logger.warning("RVC requested but backend is not usable in this environment")
        return None, None

    try:
        rvc = RVCConverter(
            model_path=rvc_config.get("model_path"),
            index_path=rvc_config.get("index_path"),
            device=rvc_config.get("device", "cuda:0"),
            f0_method=rvc_config.get("f0_method", "rmvpe"),
            index_rate=rvc_config.get("index_rate", 0.75),
            protect=rvc_config.get("protect", 0.33),
            backend=rvc_config.get("backend", "auto"),
            python_path=rvc_config.get("python_path"),
            site_packages_dir=rvc_config.get("site_packages_dir"),
            worker_script=rvc_config.get("worker_script"),
            f0_up_key=rvc_config.get("f0_up_key", 0.0),
            output_freq=rvc_config.get("output_freq"),
            request_timeout_sec=rvc_config.get("request_timeout_sec", 15.0),
            model_sha256=rvc_config.get("model_sha256"),
            index_sha256=rvc_config.get("index_sha256"),
        )
        return rvc, str(rvc_config.get("model_path"))
    except Exception as exc:
        logger.warning("RVC init failed, voice conversion disabled: %s", exc)
        return None, None


def create_pipeline_runtime_components(
    config: dict,
    *,
    initial_tts_language: str | None = None,
) -> PipelineRuntimeComponents:
    return create_pipeline_runtime(
        config,
        initial_tts_language=initial_tts_language,
    ).build_components()


def create_pipeline_runtime(
    config: dict,
    *,
    initial_tts_language: str | None = None,
) -> PipelineRuntime:
    return PipelineRuntime(config, initial_tts_language=initial_tts_language)


def preload_pipeline_llm(llm: Any) -> Any:
    """Preload the configured LLM when supported."""
    if llm is None:
        return None

    preload = getattr(llm, "preload", None)
    if callable(preload):
        preload()
    return llm


def preload_pipeline_asr(asr: Any) -> Any:
    """Preload the configured ASR backend when supported."""
    if asr is None:
        return None

    preload = getattr(asr, "preload", None)
    if callable(preload):
        preload()
    elif hasattr(asr, "_get_model"):
        asr._get_model()
    return asr


def preload_pipeline_tts(
    tts: Any,
    *,
    warmup: bool = False,
    on_load_error: Callable[[Any, Exception], Any] | None = None,
) -> Any:
    """Preload and optionally warm up the configured TTS backend."""
    if tts is None:
        return None

    try:
        _load_tts_model(tts)
    except Exception as exc:
        if on_load_error is None:
            raise
        replacement = on_load_error(tts, exc)
        if replacement is None:
            raise
        tts = replacement
        _load_tts_model(tts)

    if warmup and hasattr(tts, "warmup"):
        tts.warmup()
    return tts


def preload_pipeline_rvc(
    rvc: Any | None,
    *,
    warmup: bool = True,
) -> Any | None:
    """Preload and optionally warm up RVC when enabled."""
    if rvc is None:
        return None

    try:
        preload = getattr(rvc, "preload", None)
        if callable(preload):
            preload()
        if warmup and hasattr(rvc, "warmup"):
            rvc.warmup()
        return rvc
    except Exception as exc:
        logger.warning("RVC preload/warmup failed, disabling RVC: %s", exc)
        close = getattr(rvc, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass
        return None


def preload_pipeline_runtime_services(
    *,
    llm: Any = None,
    asr: Any = None,
    tts: Any = None,
    rvc: Any | None = None,
    tts_warmup_on_start: bool = False,
    tts_on_load_error: Callable[[Any, Exception], Any] | None = None,
    rvc_warmup_on_start: bool = True,
) -> tuple[Any, Any | None]:
    """Preload pipeline services in the same order across entry points."""
    preload_pipeline_llm(llm)
    preload_pipeline_asr(asr)
    tts = preload_pipeline_tts(
        tts,
        warmup=tts_warmup_on_start,
        on_load_error=tts_on_load_error,
    )
    rvc = preload_pipeline_rvc(rvc, warmup=rvc_warmup_on_start)
    return tts, rvc


async def close_pipeline_runtime_services(
    *,
    llm: Any = None,
    tts: Any = None,
    asr: Any = None,
    rvc: Any | None = None,
) -> None:
    """Close pipeline services consistently across entry points."""
    if tts and hasattr(tts, "cleanup"):
        tts.cleanup()
    if asr and hasattr(asr, "cleanup"):
        asr.cleanup()

    if llm is not None:
        close = getattr(llm, "close", None)
        if callable(close):
            result = close()
            if inspect.isawaitable(result):
                await result
        elif hasattr(llm, "cleanup"):
            llm.cleanup()

    if rvc and hasattr(rvc, "close"):
        rvc.close()


def _maybe_set_tts_language(tts: Any, language: str | None) -> None:
    if language and hasattr(tts, "set_language"):
        try:
            tts.set_language(language)
        except Exception as exc:
            logger.debug("Initial TTS language setup failed: %s", exc)


def _load_tts_model(tts: Any) -> None:
    preload = getattr(tts, "preload", None)
    if callable(preload):
        preload()
    elif hasattr(tts, "_load_model"):
        tts._load_model()
