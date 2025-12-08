"""
Live2D Avatar module for Local AI Companion.

This module provides a transparent overlay window displaying a Live2D avatar
that can be controlled for expressions and lip-sync with the AI assistant.

Usage:
    from src.live2d import Live2DOverlay, ExpressionMapper

    # Start overlay
    overlay = Live2DOverlay()
    overlay.start(blocking=False)
    
    # Control avatar
    overlay.set_expression('happy')
    overlay.set_lip_sync(0.5)
    
    # With AI pipeline integration
    from src.live2d import create_live2d_integration
    
    integration = create_live2d_integration(overlay)
    cleaned_text = integration.process_llm_output("Hello! (happy)")
    integration.start_lip_sync_from_bytes(tts_audio)
"""

from .overlay import (
    Live2DOverlay,
    OverlayConfig,
    ExpressionMapper,
    AudioLipSyncController
)

from .integration import (
    Live2DPipelineIntegration,
    EmotionConfig,
    EmotionDetector,
    AudioAnalyzer,
    WebSocketLive2DHandler,
    create_live2d_integration
)

__all__ = [
    # Overlay
    'Live2DOverlay',
    'OverlayConfig',
    'ExpressionMapper',
    'AudioLipSyncController',
    # Integration
    'Live2DPipelineIntegration',
    'EmotionConfig',
    'EmotionDetector',
    'AudioAnalyzer',
    'WebSocketLive2DHandler',
    'create_live2d_integration'
]
