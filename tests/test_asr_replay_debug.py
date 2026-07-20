from src.asr.base import ASRResult
from scripts import asr_replay_debug


def test_replay_preloads_whisper_before_timed_transcription(tmp_path, monkeypatch):
    events = []

    class FakeWhisperProvider:
        def __init__(self, model_size, **_kwargs):
            self.model_size = model_size
            self.beam_size = 0

        def _get_model(self):
            events.append(("load", self.model_size))

        def transcribe(self, _wav, language=None):
            events.append(("transcribe", self.model_size, language))
            return ASRResult(text="ok")

    wav_path = tmp_path / "sample.wav"
    wav_path.touch()
    monkeypatch.setattr(asr_replay_debug, "WhisperProvider", FakeWhisperProvider)
    monkeypatch.setattr(asr_replay_debug, "_cleanup", lambda: None)
    monkeypatch.setattr(
        "sys.argv",
        ["asr_replay_debug.py", "--language", "fr", str(wav_path)],
    )

    asr_replay_debug.main()

    for model_name, _beams in asr_replay_debug.WHISPER_PLAN:
        load_index = events.index(("load", model_name))
        first_transcribe = next(
            index
            for index, event in enumerate(events)
            if event[:2] == ("transcribe", model_name)
        )
        assert load_index < first_transcribe
