from collections import deque

import numpy as np
import torch

from src.vad.silero_vad import SileroVAD, State, VADConfig


class _DummyModel:
    def __init__(self, prob: float = 0.0):
        self.prob = prob

    def __call__(self, _tensor, _sample_rate):
        return torch.tensor(self.prob, dtype=torch.float32)

    def reset_states(self):
        pass


def _make_vad(config: VADConfig) -> SileroVAD:
    vad = SileroVAD.__new__(SileroVAD)
    vad.config = config
    vad.model = _DummyModel()
    vad.state = State.IDLE
    vad.hit_count = 0
    vad.miss_count = 0
    vad.voiced_chunk_count = 0
    vad.audio_buffer = bytearray()
    vad.pre_buffer = deque(maxlen=20)
    vad.prob_window = deque(maxlen=config.smoothing_window)
    vad.db_window = deque(maxlen=config.smoothing_window)
    vad.chunk_size = 512 if config.sample_rate == 16000 else 256
    return vad


def test_vad_drops_segment_when_voiced_duration_too_short():
    vad = _make_vad(
        VADConfig(
            sample_rate=16000,
            required_hits=3,
            required_misses=2,
            min_speech_ms=200,
            min_voiced_ms=180,
        )
    )
    chunk = b"\x01\x02" * vad.chunk_size
    vad.state = State.ACTIVE
    vad.audio_buffer = bytearray(chunk * 8)
    vad.voiced_chunk_count = 3  # ~96ms at 32ms/chunk
    vad.miss_count = 1

    events = list(vad.process_chunk(np.zeros(vad.chunk_size, dtype=np.float32)))

    assert events == [b"<|END|>"]
    assert vad.state == State.IDLE


def test_vad_emits_segment_when_voiced_duration_is_sufficient():
    vad = _make_vad(
        VADConfig(
            sample_rate=16000,
            required_hits=3,
            required_misses=2,
            min_speech_ms=200,
            min_voiced_ms=180,
        )
    )
    chunk = b"\x01\x02" * vad.chunk_size
    vad.state = State.ACTIVE
    vad.audio_buffer = bytearray(chunk * 10)
    vad.voiced_chunk_count = 6  # ~192ms at 32ms/chunk
    vad.miss_count = 1

    events = list(vad.process_chunk(np.zeros(vad.chunk_size, dtype=np.float32)))

    assert len(events) == 2
    assert isinstance(events[0], bytes)
    assert len(events[0]) > 0
    assert events[1] == b"<|END|>"
    assert vad.state == State.IDLE
