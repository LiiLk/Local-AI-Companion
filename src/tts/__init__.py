# TTS Module - Text-to-Speech implementations
#
# Providers:
# - KokoroProvider  : Lightweight local 82M params, good quality, no voice cloning
# - EdgeTTSProvider : Microsoft cloud (free fallback)

from .base import BaseTTS, TTSResult, Voice

__all__ = [
    "BaseTTS",
    "TTSResult",
    "Voice",
    "EdgeTTSProvider",
    "KokoroProvider",
    "ChatterboxTTSProvider",
    "Qwen3TTSProvider",
    "RoutedTTSProvider",
]


_PROVIDER_MODULES = {
    "EdgeTTSProvider": ".edge_tts_provider",
    "KokoroProvider": ".kokoro_provider",
    "ChatterboxTTSProvider": ".chatterbox_provider",
    "Qwen3TTSProvider": ".qwen3_tts_provider",
    "RoutedTTSProvider": ".routed_provider",
}


def __getattr__(name):
    module_name = _PROVIDER_MODULES.get(name)
    if module_name:
        from importlib import import_module

        value = getattr(import_module(module_name, package=__name__), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(__all__))
