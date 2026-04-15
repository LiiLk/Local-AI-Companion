"""
VRAM Monitor — Debug utility for tracking GPU memory usage.

Logs allocated/reserved VRAM at each pipeline stage.
Warns when usage exceeds configurable threshold.
Enable via config: debug.vram_monitor: true
"""

import logging

logger = logging.getLogger(__name__)


class VRAMMonitor:
    """Logs GPU VRAM usage at key pipeline stages."""

    def __init__(self, threshold_pct: float = 80.0):
        self.threshold_pct = threshold_pct
        self._available = False
        try:
            import torch
            self._available = torch.cuda.is_available()
            if self._available:
                self._torch = torch
                self._total = torch.cuda.get_device_properties(0).total_memory
        except ImportError:
            pass

    def log(self, stage: str) -> dict:
        """Log VRAM usage for a named stage. Returns usage dict."""
        if not self._available:
            return {}

        allocated = self._torch.cuda.memory_allocated()
        reserved = self._torch.cuda.memory_reserved()
        pct = (allocated / self._total) * 100

        info = {
            "stage": stage,
            "allocated_mb": allocated / 1024 / 1024,
            "reserved_mb": reserved / 1024 / 1024,
            "total_mb": self._total / 1024 / 1024,
            "percent": pct,
        }

        level = logging.WARNING if pct > self.threshold_pct else logging.DEBUG
        logger.log(
            level,
            f"[VRAM] {stage}: {info['allocated_mb']:.0f}MB allocated, "
            f"{info['reserved_mb']:.0f}MB reserved / {info['total_mb']:.0f}MB "
            f"({pct:.1f}%)",
        )
        return info
