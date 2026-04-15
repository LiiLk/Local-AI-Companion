from src.vad.silero_vad import SileroVAD, State, VADConfig


class _DummyModel:
    def reset_states(self):
        pass


def test_force_end_appends_tail_silence():
    vad = SileroVAD.__new__(SileroVAD)
    vad.config = VADConfig(sample_rate=16000, force_end_silence_ms=200)
    vad.state = State.ACTIVE
    vad.hit_count = 0
    vad.miss_count = 0
    vad.audio_buffer = bytearray(b"\x01\x02" * 1000)
    vad.pre_buffer = []
    vad.prob_window = []
    vad.db_window = []
    vad.model = _DummyModel()

    audio = vad.force_end()

    assert audio is not None
    assert len(audio) == 2000 + (16000 * 200 // 1000) * 2
    assert vad.state == State.IDLE


def test_force_end_returns_none_when_idle():
    vad = SileroVAD.__new__(SileroVAD)
    vad.config = VADConfig(sample_rate=16000, force_end_silence_ms=200)
    vad.state = State.IDLE
    vad.hit_count = 0
    vad.miss_count = 0
    vad.audio_buffer = bytearray()
    vad.pre_buffer = []
    vad.prob_window = []
    vad.db_window = []
    vad.model = _DummyModel()

    assert vad.force_end() is None
