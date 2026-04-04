"""
Screen Buffer — Continuous screen capture with change detection.

Captures screenshots via mss in a background thread.
Uses pixel diff to skip unchanged frames.
Stores frames in a circular buffer for vision pipeline access.
"""

import logging
import threading
import time
from collections import deque
from typing import Optional

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def pixel_diff(img1: Image.Image, img2: Image.Image) -> float:
    """
    Compute normalized pixel difference between two images.

    Returns a value between 0.0 (identical) and 1.0 (completely different).
    Images are downscaled for speed.
    """
    # Downscale for fast comparison
    size = (64, 64)
    a = np.array(img1.resize(size, Image.NEAREST), dtype=np.float32)
    b = np.array(img2.resize(size, Image.NEAREST), dtype=np.float32)
    return float(np.mean(np.abs(a - b)) / 255.0)


class ScreenBuffer:
    """
    Continuous screen capture with change detection.

    Runs a background thread that captures via mss at regular intervals.
    Frames that are too similar to the previous one are skipped.

    Args:
        capture_interval: Seconds between captures (default 2.0).
        max_buffer: Maximum frames to keep in circular buffer.
        change_threshold: Minimum pixel diff to store a new frame (0.0-1.0).
        monitor: mss monitor index (0 = all screens, 1 = primary).
    """

    def __init__(
        self,
        capture_interval: float = 2.0,
        max_buffer: int = 30,
        change_threshold: float = 0.05,
        monitor: int = 1,
    ):
        self.capture_interval = capture_interval
        self.change_threshold = change_threshold
        self.monitor = monitor

        self._frames: deque[Image.Image] = deque(maxlen=max_buffer)
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """Start background capture thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._capture_loop, daemon=True, name="screen-buffer"
        )
        self._thread.start()
        logger.info(
            f"ScreenBuffer started (interval={self.capture_interval}s, "
            f"threshold={self.change_threshold})"
        )

    def stop(self) -> None:
        """Stop background capture."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None
        logger.info("ScreenBuffer stopped")

    def _capture_loop(self) -> None:
        """Background capture loop."""
        import mss

        with mss.mss() as sct:
            while self._running:
                try:
                    raw = sct.grab(sct.monitors[self.monitor])
                    img = Image.frombytes("RGB", raw.size, raw.bgra, "raw", "BGRX")

                    with self._lock:
                        # Skip if too similar to last frame
                        if self._frames:
                            diff = pixel_diff(self._frames[-1], img)
                            if diff < self.change_threshold:
                                time.sleep(self.capture_interval)
                                continue

                        self._frames.append(img)

                except Exception as e:
                    logger.warning(f"Screen capture failed: {e}")

                time.sleep(self.capture_interval)

    def get_latest(self) -> Optional[Image.Image]:
        """Get the most recent captured frame."""
        with self._lock:
            return self._frames[-1] if self._frames else None

    def get_recent(self, n: int = 10) -> list[Image.Image]:
        """Get the N most recent frames."""
        with self._lock:
            return list(self._frames)[-n:]

    def __len__(self) -> int:
        return len(self._frames)
