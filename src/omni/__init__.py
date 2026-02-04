"""
Omni Module - Unified multimodal model providers.

An omni model handles ASR, LLM, and TTS in a single model,
replacing the traditional pipeline of separate providers.
"""

from .minicpmo_provider import MiniCPMoProvider
from .omni_pipeline import OmniPipeline

__all__ = [
    "MiniCPMoProvider",
    "OmniPipeline",
]
