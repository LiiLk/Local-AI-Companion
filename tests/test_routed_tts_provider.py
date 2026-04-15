import asyncio

import pytest

from src.tts.base import TTSResult
from src.tts.routed_provider import RoutedTTSProvider


class FakeProvider:
    def __init__(self, name: str, *, fail: bool = False, delay: float = 0.0, request_timeout_sec=None):
        self.name = name
        self.fail = fail
        self.delay = delay
        self.request_timeout_sec = request_timeout_sec
        self.languages: list[str | None] = []
        self.cancelled = False
        self.cleaned_up = False
        self.preloaded = False
        self.warmed = False

    async def synthesize(self, text, output_path=None):
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.fail:
            raise RuntimeError(f"{self.name} failed")
        return TTSResult(audio_data=b"fake", metadata={"provider": self.name})

    def preload(self):
        if self.fail:
            raise RuntimeError(f"{self.name} failed")
        self.preloaded = True

    def warmup(self):
        if self.fail:
            raise RuntimeError(f"{self.name} failed")
        self.warmed = True

    def set_language(self, language):
        self.languages.append(language)

    def cancel_inflight(self):
        self.cancelled = True

    def cleanup(self):
        self.cleaned_up = True


@pytest.mark.asyncio
async def test_routed_tts_prefers_qwen3_for_supported_language():
    qwen3 = FakeProvider("qwen3")
    chatterbox = FakeProvider("chatterbox")
    kokoro = FakeProvider("kokoro")
    provider = RoutedTTSProvider(
        qwen3=qwen3,
        chatterbox=chatterbox,
        kokoro=kokoro,
        default_language="fr",
    )

    result = await provider.synthesize("Bonjour")

    assert provider.active_provider_name == "qwen3"
    assert provider.degraded_reason is None
    assert result.metadata["provider_name"] == "qwen3"
    assert qwen3.languages[-1] == "fr"


@pytest.mark.asyncio
async def test_routed_tts_falls_back_to_chatterbox_for_unsupported_qwen3_language():
    qwen3 = FakeProvider("qwen3")
    chatterbox = FakeProvider("chatterbox")
    kokoro = FakeProvider("kokoro")
    provider = RoutedTTSProvider(
        qwen3=qwen3,
        chatterbox=chatterbox,
        kokoro=kokoro,
        default_language="tr",
    )

    result = await provider.synthesize("Merhaba")

    assert provider.active_provider_name == "chatterbox"
    assert "Using chatterbox for language tr instead of qwen3." == provider.degraded_reason
    assert result.metadata["provider_name"] == "chatterbox"
    assert result.metadata["degraded_reason"] == provider.degraded_reason


@pytest.mark.asyncio
async def test_routed_tts_uses_kokoro_when_other_providers_fail():
    chatterbox = FakeProvider("chatterbox", fail=True)
    kokoro = FakeProvider("kokoro")
    provider = RoutedTTSProvider(
        qwen3=None,
        chatterbox=chatterbox,
        kokoro=kokoro,
        default_language="es",
    )

    result = await provider.synthesize("Hola")

    assert provider.active_provider_name == "kokoro"
    assert provider.degraded_reason == "Using kokoro for language es instead of qwen3."
    assert result.metadata["provider_name"] == "kokoro"


@pytest.mark.asyncio
async def test_routed_tts_surfaces_qwen3_failure_reason_when_falling_back():
    qwen3 = FakeProvider("qwen3", fail=True)
    chatterbox = FakeProvider("chatterbox")
    provider = RoutedTTSProvider(
        qwen3=qwen3,
        chatterbox=chatterbox,
        kokoro=None,
        default_language="fr",
    )

    result = await provider.synthesize("Bonjour")

    assert provider.active_provider_name == "chatterbox"
    assert provider.degraded_reason == (
        "Qwen3-TTS failed (RuntimeError: qwen3 failed); using chatterbox for language fr."
    )
    assert result.metadata["provider_name"] == "chatterbox"
    assert result.metadata["degraded_reason"] == provider.degraded_reason


@pytest.mark.asyncio
async def test_routed_tts_disables_failed_qwen3_for_later_requests():
    qwen3 = FakeProvider("qwen3", fail=True)
    chatterbox = FakeProvider("chatterbox")
    provider = RoutedTTSProvider(
        qwen3=qwen3,
        chatterbox=chatterbox,
        kokoro=None,
        default_language="fr",
    )

    first_result = await provider.synthesize("Bonjour")
    second_result = await provider.synthesize("Salut encore")

    assert first_result.metadata["provider_name"] == "chatterbox"
    assert second_result.metadata["provider_name"] == "chatterbox"
    assert provider.degraded_reason == (
        "Qwen3-TTS unavailable (RuntimeError: qwen3 failed); using chatterbox for language fr."
    )
    assert qwen3.cleaned_up is True


def test_routed_tts_cancel_inflight_fans_out_to_providers():
    qwen3 = FakeProvider("qwen3")
    chatterbox = FakeProvider("chatterbox")
    provider = RoutedTTSProvider(qwen3=qwen3, chatterbox=chatterbox, kokoro=None, default_language="fr")

    provider.cancel_inflight()

    assert qwen3.cancelled is True
    assert chatterbox.cancelled is True


@pytest.mark.asyncio
async def test_routed_tts_disables_qwen3_after_timeout():
    qwen3 = FakeProvider("qwen3", delay=0.05, request_timeout_sec=0.01)
    chatterbox = FakeProvider("chatterbox")
    provider = RoutedTTSProvider(
        qwen3=qwen3,
        chatterbox=chatterbox,
        kokoro=None,
        default_language="fr",
    )

    result = await provider.synthesize("Bonjour")

    assert result.metadata["provider_name"] == "chatterbox"
    assert provider.degraded_reason == (
        "Qwen3-TTS failed (RuntimeError: qwen3 timed out after 0.0s); using chatterbox for language fr."
    ) or provider.degraded_reason == (
        "Qwen3-TTS unavailable (RuntimeError: qwen3 timed out after 0.0s); using chatterbox for language fr."
    )
    assert qwen3.cancelled is True
    assert qwen3.cleaned_up is True


@pytest.mark.asyncio
async def test_routed_tts_honors_provider_order_override_by_language():
    qwen3 = FakeProvider("qwen3")
    chatterbox = FakeProvider("chatterbox")
    provider = RoutedTTSProvider(
        qwen3=qwen3,
        chatterbox=chatterbox,
        kokoro=None,
        default_language="fr",
        provider_order_by_language={"fr": ["chatterbox", "qwen3"]},
    )

    result = await provider.synthesize("Bonjour")

    assert result.metadata["provider_name"] == "chatterbox"
    assert provider.active_provider_name == "chatterbox"


def test_routed_tts_preloads_fallbacks_when_enabled():
    qwen3 = FakeProvider("qwen3")
    chatterbox = FakeProvider("chatterbox")
    provider = RoutedTTSProvider(
        qwen3=qwen3,
        chatterbox=chatterbox,
        kokoro=None,
        default_language="en",
        preload_fallbacks=True,
    )

    provider.preload()

    assert qwen3.preloaded is True
    assert chatterbox.preloaded is True
    assert provider.active_provider_name == "qwen3"
