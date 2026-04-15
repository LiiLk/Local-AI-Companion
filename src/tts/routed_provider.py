"""
Language-aware TTS router with provider fallbacks.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import suppress
from typing import Any, AsyncGenerator

from src.utils.language_detection import normalize_language_code

from .base import BaseTTS, TTSResult, Voice, prefers_full_response_tts

logger = logging.getLogger(__name__)

QWEN3_LANGUAGE_CODES = {"de", "en", "es", "fr", "it", "ja", "ko", "pt", "ru", "zh"}
CHATTERBOX_LANGUAGE_CODES = {
    "ar",
    "bg",
    "cs",
    "da",
    "de",
    "el",
    "en",
    "es",
    "fi",
    "fr",
    "he",
    "hi",
    "hu",
    "it",
    "ja",
    "ko",
    "ms",
    "nl",
    "no",
    "pl",
    "pt",
    "ro",
    "ru",
    "sk",
    "sv",
    "ta",
    "tr",
    "vi",
    "zh",
}
KOKORO_LANGUAGE_CODES = {"en", "es", "fr", "hi", "it", "ja", "pt", "zh"}


class RoutedTTSProvider(BaseTTS):
    """Select the best available provider for the active language."""

    def __init__(
        self,
        qwen3: Any | None = None,
        chatterbox: Any | None = None,
        kokoro: Any | None = None,
        default_language: str | None = None,
        provider_order_by_language: dict[str, list[str]] | None = None,
        preload_fallbacks: bool = False,
        warmup_fallbacks: bool = False,
    ):
        self._providers = {
            "qwen3": qwen3,
            "chatterbox": chatterbox,
            "kokoro": kokoro,
        }
        self._language_hint = normalize_language_code(default_language) or "en"
        self._provider_order_by_language = {
            normalize_language_code(language) or str(language).strip().lower(): list(order)
            for language, order in (provider_order_by_language or {}).items()
        }
        self._preload_fallbacks = bool(preload_fallbacks)
        self._warmup_fallbacks = bool(warmup_fallbacks)
        self._disabled_providers: dict[str, str] = {}
        self.active_provider_name = self._initial_provider_name()
        self.degraded_reason: str | None = None

    @staticmethod
    def _describe_exception(exc: Exception) -> str:
        message = str(exc).strip()
        if message:
            return f"{type(exc).__name__}: {message}"
        return type(exc).__name__

    def _initial_provider_name(self) -> str | None:
        for name in ("qwen3", "chatterbox", "kokoro"):
            if self._providers.get(name) is not None and name not in self._disabled_providers:
                return name
        return None

    def _disable_provider(self, provider_name: str, reason: str) -> None:
        provider = self._providers.get(provider_name)
        if provider is None:
            return

        if provider_name not in self._disabled_providers:
            logger.warning(
                "Disabling TTS provider %s for current session: %s",
                provider_name,
                reason,
            )
        self._disabled_providers[provider_name] = reason

        cancel_fn = getattr(provider, "cancel_inflight", None)
        if callable(cancel_fn):
            with suppress(Exception):
                cancel_fn()

        cleanup_fn = getattr(provider, "cleanup", None)
        if callable(cleanup_fn):
            with suppress(Exception):
                cleanup_fn()

    def _qwen3_disabled_reason(self) -> str | None:
        return self._disabled_providers.get("qwen3")

    def _fallback_reason(
        self,
        provider_name: str,
        language_code: str | None,
        *,
        qwen3_error: Exception | None = None,
        qwen3_failure_prefix: str = "failed",
    ) -> str | None:
        if provider_name == "qwen3":
            return None

        if qwen3_error is not None:
            return (
                f"Qwen3-TTS {qwen3_failure_prefix} ({self._describe_exception(qwen3_error)}); "
                f"using {provider_name} for language {language_code or 'auto'}."
            )

        qwen3_disabled_reason = self._qwen3_disabled_reason()
        if qwen3_disabled_reason:
            return (
                f"Qwen3-TTS unavailable ({qwen3_disabled_reason}); "
                f"using {provider_name} for language {language_code or 'auto'}."
            )

        return None

    def _supports_language(self, provider_name: str, language_code: str | None) -> bool:
        if not language_code:
            return True
        if provider_name == "qwen3":
            return language_code in QWEN3_LANGUAGE_CODES
        if provider_name == "chatterbox":
            return language_code in CHATTERBOX_LANGUAGE_CODES
        if provider_name == "kokoro":
            return language_code in KOKORO_LANGUAGE_CODES
        return False

    def _candidate_order(self, language_code: str | None) -> list[str]:
        override = self._provider_order_override(language_code)
        if override:
            return [
                name
                for name in override
                if self._providers.get(name) is not None and name not in self._disabled_providers
            ]

        if self._supports_language("qwen3", language_code):
            order = ["qwen3", "chatterbox", "kokoro"]
        elif self._supports_language("chatterbox", language_code):
            order = ["chatterbox", "kokoro", "qwen3"]
        elif self._supports_language("kokoro", language_code):
            order = ["kokoro", "chatterbox", "qwen3"]
        else:
            order = ["chatterbox", "kokoro", "qwen3"]

        return [
            name
            for name in order
            if self._providers.get(name) is not None and name not in self._disabled_providers
        ]

    def _provider_order_override(self, language_code: str | None) -> list[str] | None:
        if not language_code:
            return None
        normalized = normalize_language_code(language_code) or language_code
        if normalized in self._provider_order_by_language:
            return self._provider_order_by_language[normalized]
        if "-" in normalized:
            prefix = normalized.split("-", 1)[0]
            return self._provider_order_by_language.get(prefix)
        return None

    def _provider_for_language(self, language_code: str | None) -> tuple[str, Any] | tuple[None, None]:
        for name in self._candidate_order(language_code):
            provider = self._providers.get(name)
            if provider is not None:
                return name, provider
        return None, None

    @staticmethod
    def _provider_timeout_sec(provider_name: str, provider: Any) -> float | None:
        timeout = getattr(provider, "request_timeout_sec", None)
        if timeout is not None:
            return float(timeout)
        if provider_name == "qwen3":
            return 20.0
        return None

    def _mark_provider_selected(
        self,
        provider_name: str,
        language_code: str | None,
        degraded: bool,
        degraded_reason: str | None = None,
    ) -> None:
        self.active_provider_name = provider_name
        if degraded:
            self.degraded_reason = (
                degraded_reason
                or self.degraded_reason
                or f"Using {provider_name} for language {language_code or 'auto'} instead of qwen3."
            )
        else:
            self.degraded_reason = None

    @property
    def prefer_full_response_tts(self) -> bool:
        provider_name, provider = self._provider_for_language(self._language_hint)
        self.active_provider_name = provider_name
        return prefers_full_response_tts(provider)

    async def synthesize(self, text: str, output_path=None) -> TTSResult:
        language_code = normalize_language_code(self._language_hint)
        last_error: Exception | None = None
        qwen3_error: Exception | None = None

        for provider_name in self._candidate_order(language_code):
            provider = self._providers.get(provider_name)
            if provider is None:
                continue

            with suppress(Exception):
                if hasattr(provider, "set_language"):
                    provider.set_language(language_code)

            timeout_sec = self._provider_timeout_sec(provider_name, provider)
            try:
                if timeout_sec is not None:
                    result = await asyncio.wait_for(
                        provider.synthesize(text, output_path),
                        timeout=timeout_sec,
                    )
                else:
                    result = await provider.synthesize(text, output_path)
            except asyncio.TimeoutError:
                timeout_exc = RuntimeError(
                    f"{provider_name} timed out after {timeout_sec:.1f}s"
                )
                last_error = timeout_exc
                self._disable_provider(provider_name, self._describe_exception(timeout_exc))
                if provider_name == "qwen3":
                    qwen3_error = timeout_exc
                logger.warning(
                    "TTS provider %s timed out for language %s after %.1fs",
                    provider_name,
                    language_code or "auto",
                    timeout_sec,
                )
                continue
            except Exception as exc:
                last_error = exc
                self._disable_provider(provider_name, self._describe_exception(exc))
                if provider_name == "qwen3":
                    qwen3_error = exc
                logger.warning(
                    "TTS provider %s failed for language %s: %s",
                    provider_name,
                    language_code or "auto",
                    self._describe_exception(exc),
                )
                continue

            degraded_reason = self._fallback_reason(
                provider_name,
                language_code,
                qwen3_error=qwen3_error,
            )
            self._mark_provider_selected(
                provider_name,
                language_code,
                degraded=(provider_name != "qwen3"),
                degraded_reason=degraded_reason,
            )

            metadata = dict(result.metadata or {})
            metadata["provider_name"] = provider_name
            metadata["language_code"] = language_code
            if self.degraded_reason:
                metadata["degraded_reason"] = self.degraded_reason
            result.metadata = metadata
            return result

        if last_error:
            raise last_error
        raise RuntimeError("No TTS providers are available")

    async def synthesize_stream(self, text: str) -> AsyncGenerator[bytes, None]:
        provider_name, provider = self._provider_for_language(self._language_hint)
        if provider is None:
            raise RuntimeError("No TTS providers are available")
        self.active_provider_name = provider_name
        async for chunk in provider.synthesize_stream(text):
            yield chunk

    async def list_voices(self, language: str | None = None) -> list[Voice]:
        provider_name, provider = self._provider_for_language(normalize_language_code(language) or self._language_hint)
        if provider is None:
            return []
        self.active_provider_name = provider_name
        return await provider.list_voices(language)

    def set_voice(self, voice_id: str) -> None:
        for provider in self._providers.values():
            if provider is None:
                continue
            with suppress(Exception):
                provider.set_voice(voice_id)

    def set_rate(self, rate: str) -> None:
        for provider in self._providers.values():
            if provider is None:
                continue
            with suppress(Exception):
                provider.set_rate(rate)

    def set_pitch(self, pitch: str) -> None:
        for provider in self._providers.values():
            if provider is None:
                continue
            with suppress(Exception):
                provider.set_pitch(pitch)

    def set_language(self, language: str | None) -> None:
        self._language_hint = normalize_language_code(language) or self._language_hint
        for provider in self._providers.values():
            if provider is None or not hasattr(provider, "set_language"):
                continue
            with suppress(Exception):
                provider.set_language(self._language_hint)

    def preload(self) -> None:
        language_code = self._language_hint
        last_error: Exception | None = None
        qwen3_error: Exception | None = None
        selected_provider_name: str | None = None
        for provider_name in self._candidate_order(language_code):
            provider = self._providers.get(provider_name)
            if provider is None or not hasattr(provider, "preload"):
                continue
            with suppress(Exception):
                if hasattr(provider, "set_language"):
                    provider.set_language(language_code)
            try:
                provider.preload()
                if selected_provider_name is None:
                    selected_provider_name = provider_name
                    degraded_reason = self._fallback_reason(
                        provider_name,
                        language_code,
                        qwen3_error=qwen3_error,
                        qwen3_failure_prefix="preload failed",
                    )
                    self._mark_provider_selected(
                        provider_name,
                        language_code,
                        degraded=(provider_name != "qwen3"),
                        degraded_reason=degraded_reason,
                    )
                    if not self._preload_fallbacks:
                        return
                else:
                    logger.info(
                        "Preloaded fallback TTS provider %s for language %s",
                        provider_name,
                        language_code or "auto",
                    )
            except Exception as exc:
                last_error = exc
                self._disable_provider(provider_name, self._describe_exception(exc))
                if provider_name == "qwen3":
                    qwen3_error = exc
                logger.warning(
                    "TTS provider %s preload failed for language %s: %s",
                    provider_name,
                    language_code or "auto",
                    self._describe_exception(exc),
                )
        if selected_provider_name is not None:
            return
        if last_error:
            raise last_error

    def warmup(self) -> None:
        language_code = self._language_hint
        last_error: Exception | None = None
        qwen3_error: Exception | None = None
        selected_provider_name: str | None = None
        for provider_name in self._candidate_order(language_code):
            provider = self._providers.get(provider_name)
            if provider is None or not hasattr(provider, "warmup"):
                continue
            with suppress(Exception):
                if hasattr(provider, "set_language"):
                    provider.set_language(language_code)
            try:
                provider.warmup()
                if selected_provider_name is None:
                    selected_provider_name = provider_name
                    degraded_reason = self._fallback_reason(
                        provider_name,
                        language_code,
                        qwen3_error=qwen3_error,
                        qwen3_failure_prefix="warmup failed",
                    )
                    self._mark_provider_selected(
                        provider_name,
                        language_code,
                        degraded=(provider_name != "qwen3"),
                        degraded_reason=degraded_reason,
                    )
                    if not self._warmup_fallbacks:
                        return
                else:
                    logger.info(
                        "Warmed fallback TTS provider %s for language %s",
                        provider_name,
                        language_code or "auto",
                    )
            except Exception as exc:
                last_error = exc
                self._disable_provider(provider_name, self._describe_exception(exc))
                if provider_name == "qwen3":
                    qwen3_error = exc
                logger.warning(
                    "TTS provider %s warmup failed for language %s: %s",
                    provider_name,
                    language_code or "auto",
                    self._describe_exception(exc),
                )
        if selected_provider_name is not None:
            return
        if last_error:
            raise last_error

    def cleanup(self) -> None:
        for provider in self._providers.values():
            if provider is None or not hasattr(provider, "cleanup"):
                continue
            with suppress(Exception):
                provider.cleanup()

    def cancel_inflight(self) -> None:
        for provider in self._providers.values():
            if provider is None:
                continue
            cancel_fn = getattr(provider, "cancel_inflight", None)
            if not callable(cancel_fn):
                continue
            with suppress(Exception):
                cancel_fn()
