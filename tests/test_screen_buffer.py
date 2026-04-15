"""Tests for ScreenBuffer — screen capture and change detection."""

import numpy as np
from unittest.mock import patch, MagicMock
from PIL import Image

from src.vision.screen_buffer import ScreenBuffer, pixel_diff


class TestPixelDiff:
    def test_identical_frames(self):
        img = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8))
        assert pixel_diff(img, img) == 0.0

    def test_completely_different(self):
        black = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8))
        white = Image.fromarray(np.full((100, 100, 3), 255, dtype=np.uint8))
        diff = pixel_diff(black, white)
        assert diff > 0.9  # Nearly 100% different

    def test_partial_change(self):
        base = np.zeros((100, 100, 3), dtype=np.uint8)
        changed = base.copy()
        changed[50:, :, :] = 255  # Bottom half white
        diff = pixel_diff(Image.fromarray(base), Image.fromarray(changed))
        assert 0.3 < diff < 0.7


class TestScreenBuffer:
    def test_get_latest_empty(self):
        buf = ScreenBuffer(capture_interval=10.0)  # long interval, won't auto-capture
        assert buf.get_latest() is None

    def test_get_recent_empty(self):
        buf = ScreenBuffer(capture_interval=10.0)
        assert buf.get_recent(5) == []

    def test_buffer_max_size(self):
        buf = ScreenBuffer(max_buffer=3, capture_interval=10.0)
        for i in range(5):
            img = Image.fromarray(np.full((10, 10, 3), i * 50, dtype=np.uint8))
            buf._frames.append(img)
        # Only keep last 3 due to deque maxlen
        assert len(buf._frames) == 3
