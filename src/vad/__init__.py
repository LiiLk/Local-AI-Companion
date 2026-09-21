"""
Voice Activity Detection (VAD) Module.

Provides Silero VAD for detecting speech in audio streams and Smart Turn v3.2
for semantic end-of-turn detection.
"""

from .silero_vad import SileroVAD
from .smart_turn import (
    SmartTurnConfig,
    SmartTurnDetector,
    resolve_commit_delay_for_turn,
    resolve_turn_commit_delay_ms,
)

__all__ = [
    "SileroVAD",
    "SmartTurnConfig",
    "SmartTurnDetector",
    "resolve_commit_delay_for_turn",
    "resolve_turn_commit_delay_ms",
]