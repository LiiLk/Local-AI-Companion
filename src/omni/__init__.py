"""
Omni Module - Unified multimodal model providers.

An omni model handles ASR, LLM, and TTS in a single model,
replacing the traditional pipeline of separate providers.

Imports are lazy to avoid module-level side effects
(e.g., MiniCPM-o sets HF_HUB_OFFLINE which would block Gemma downloads).
"""


def __getattr__(name):
    if name == "MiniCPMoProvider":
        from .minicpmo_provider import MiniCPMoProvider
        return MiniCPMoProvider
    if name == "OmniPipeline":
        from .omni_pipeline import OmniPipeline
        return OmniPipeline
    if name == "GemmaProvider":
        from .gemma_provider import GemmaProvider
        return GemmaProvider
    if name == "GemmaOmniPipeline":
        from .gemma_omni_pipeline import GemmaOmniPipeline
        return GemmaOmniPipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "MiniCPMoProvider",
    "OmniPipeline",
    "GemmaProvider",
    "GemmaOmniPipeline",
]
