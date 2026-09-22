import logging
import re
import sys
import wave
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import src.vad.smart_turn as smart_turn
from src.vad.smart_turn import (
    SMART_TURN_THRESHOLD,
    SmartTurnConfig,
    SmartTurnDetector,
    prepare_audio,
    resolve_commit_delay_for_turn,
    resolve_turn_commit_delay_ms,
    resolve_turn_tier,
    resolve_vad_required_misses,
    save_debug_wav,
)

SR = 16000
MAX_SAMPLES = 8 * SR


class FakeFeatureExtractor:
    def __init__(self):
        self.calls = []

    def __call__(self, audio, **kwargs):
        self.calls.append((np.asarray(audio), kwargs))
        return SimpleNamespace(input_features=np.zeros((1, 80, 3000), dtype=np.float32))


class FakeSession:
    def __init__(self, probability: float):
        self.probability = probability
        self.calls = []

    def run(self, output_names, inputs):
        self.calls.append(inputs)
        return [np.array([[self.probability]], dtype=np.float32)]


def _loaded_detector(probability: float, **config_kwargs) -> SmartTurnDetector:
    detector = SmartTurnDetector(SmartTurnConfig(**config_kwargs))
    detector._session = FakeSession(probability)
    detector._feature_extractor = FakeFeatureExtractor()
    return detector


# --------------------------------------------------------------------------
# Delay policy (pure function)
# --------------------------------------------------------------------------

def test_policy_complete_uses_short_delay():
    delay = resolve_turn_commit_delay_ms(
        True, enabled=True, complete_delay_ms=250, incomplete_delay_ms=2500, fallback_delay_ms=700
    )
    assert delay == 250


def test_policy_incomplete_uses_long_delay():
    delay = resolve_turn_commit_delay_ms(
        False, enabled=True, complete_delay_ms=250, incomplete_delay_ms=2500, fallback_delay_ms=700
    )
    assert delay == 2500


def test_policy_unavailable_falls_back_to_fixed_delay():
    delay = resolve_turn_commit_delay_ms(
        None, enabled=True, complete_delay_ms=250, incomplete_delay_ms=2500, fallback_delay_ms=700
    )
    assert delay == 700


def test_policy_disabled_falls_back_to_fixed_delay():
    delay = resolve_turn_commit_delay_ms(
        True, enabled=False, complete_delay_ms=250, incomplete_delay_ms=2500, fallback_delay_ms=700
    )
    assert delay == 700


def test_turn_delay_unavailable_returns_fallback_and_clamps_elapsed():
    config = SmartTurnConfig(
        enabled=True,
        complete_delay_ms=250,
        incomplete_delay_ms=2500,
        fallback_delay_ms=700,
    )

    assert resolve_commit_delay_for_turn(config, None) == 700
    # Already spent more than the fallback: clamp to 0.
    assert resolve_commit_delay_for_turn(config, None, fallback_delay_ms=max(0, 700 - 900)) == 0
    assert resolve_commit_delay_for_turn(config, None, elapsed_ms=900.0, fallback_delay_ms=700) == 700


def test_turn_delay_complete_deducts_elapsed_time():
    config = SmartTurnConfig(enabled=True, complete_delay_ms=250, incomplete_delay_ms=2500)

    assert resolve_commit_delay_for_turn(config, True) == 250
    assert resolve_commit_delay_for_turn(config, True, elapsed_ms=40.0) == 210
    assert resolve_commit_delay_for_turn(config, True, elapsed_ms=400.0) == 0


def test_turn_delay_incomplete_keeps_long_delay():
    config = SmartTurnConfig(enabled=True, complete_delay_ms=250, incomplete_delay_ms=2500)

    assert resolve_commit_delay_for_turn(config, False) == 2500
    assert resolve_commit_delay_for_turn(config, False, elapsed_ms=40.0) == 2500


def test_turn_delay_disabled_returns_fallback_even_with_verdict():
    config = SmartTurnConfig(enabled=False, complete_delay_ms=250, incomplete_delay_ms=2500)

    assert resolve_commit_delay_for_turn(config, True) == config.fallback_delay_ms
    assert resolve_commit_delay_for_turn(config, None) == config.fallback_delay_ms


# --------------------------------------------------------------------------
# Three-tier delay policy (P1)
# --------------------------------------------------------------------------


def test_resolve_turn_tier_classifies_probability():
    config = SmartTurnConfig(threshold=0.5, uncertain_threshold=0.15)

    assert resolve_turn_tier(config, 0.99) == "complete"
    assert resolve_turn_tier(config, 0.5) == "complete"
    assert resolve_turn_tier(config, 0.30) == "uncertain"
    assert resolve_turn_tier(config, 0.15) == "uncertain"
    assert resolve_turn_tier(config, 0.01) == "incomplete"


def test_turn_delay_uncertain_uses_uncertain_delay():
    config = SmartTurnConfig(
        enabled=True,
        complete_delay_ms=250,
        uncertain_delay_ms=900,
        incomplete_delay_ms=2500,
        uncertain_threshold=0.15,
        threshold=0.5,
    )

    assert resolve_commit_delay_for_turn(config, 0.99) == 250
    assert resolve_commit_delay_for_turn(config, 0.50) == 250
    assert resolve_commit_delay_for_turn(config, 0.30) == 900
    assert resolve_commit_delay_for_turn(config, 0.15) == 900
    assert resolve_commit_delay_for_turn(config, 0.01) == 2500


def test_turn_delay_uncertain_does_not_deduct_elapsed_time():
    config = SmartTurnConfig(enabled=True, uncertain_delay_ms=900, incomplete_delay_ms=2500)

    assert resolve_commit_delay_for_turn(config, 0.30, elapsed_ms=400.0) == 900


def test_turn_delay_boolean_verdicts_still_map_to_complete_and_incomplete():
    config = SmartTurnConfig(
        enabled=True,
        complete_delay_ms=250,
        uncertain_delay_ms=900,
        incomplete_delay_ms=2500,
    )

    assert resolve_commit_delay_for_turn(config, True) == 250
    assert resolve_commit_delay_for_turn(config, False) == 2500


# --------------------------------------------------------------------------
# VAD silence threshold override (P2)
# --------------------------------------------------------------------------


def test_resolve_vad_required_misses_uses_short_threshold_only_when_active():
    config = SmartTurnConfig(enabled=True, vad_required_misses=8)

    assert resolve_vad_required_misses(config, detector_available=True, default_misses=20) == 8
    assert resolve_vad_required_misses(config, detector_available=False, default_misses=20) == 20

    disabled = SmartTurnConfig(enabled=False, vad_required_misses=8)
    assert resolve_vad_required_misses(disabled, detector_available=True, default_misses=20) == 20


# --------------------------------------------------------------------------
# Preprocessing
# --------------------------------------------------------------------------

def test_prepare_audio_converts_int16_bytes_to_float32():
    audio = prepare_audio(b"\x00\x80", SR)

    assert audio is not None
    assert audio.dtype == np.float32
    assert audio.shape[0] == MAX_SAMPLES
    assert np.all(audio[:-1] == 0.0)
    assert audio[-1] == pytest.approx(-1.0)


def test_prepare_audio_converts_int16_ndarray():
    samples = np.array([0, 32767, -32768], dtype=np.int16)

    audio = prepare_audio(samples, SR)

    assert audio[-1] == pytest.approx(-1.0)
    assert audio[-2] == pytest.approx(32767 / 32768.0)
    assert np.all(audio[:-3] == 0.0)


def test_prepare_audio_truncates_to_last_eight_seconds_keeping_the_end():
    ten_seconds = np.arange(10 * SR, dtype=np.float32) / (10 * SR)

    audio = prepare_audio(ten_seconds, SR)

    assert audio.shape[0] == MAX_SAMPLES
    assert audio[0] == pytest.approx(ten_seconds[-MAX_SAMPLES])
    assert audio[-1] == pytest.approx(ten_seconds[-1])


def test_prepare_audio_pads_at_the_beginning():
    one_second = np.ones(SR, dtype=np.float32)

    audio = prepare_audio(one_second, SR)

    assert audio.shape[0] == MAX_SAMPLES
    assert np.all(audio[:-SR] == 0.0)
    assert np.all(audio[-SR:] == 1.0)


def test_prepare_audio_refuses_other_sample_rates():
    assert prepare_audio(np.zeros(SR, dtype=np.float32), 8000) is None


def test_prepare_audio_rejects_empty_input():
    assert prepare_audio(np.array([], dtype=np.float32), SR) is None


# --------------------------------------------------------------------------
# Predict
# --------------------------------------------------------------------------

def test_predict_returns_complete_and_probability():
    detector = _loaded_detector(0.93)

    verdict = detector.predict(b"\x00\x00" * SR, SR)

    assert verdict == (True, pytest.approx(0.93))
    extractor = detector._feature_extractor
    _, kwargs = extractor.calls[0]
    assert kwargs["sampling_rate"] == SR
    assert kwargs["return_tensors"] == "np"
    assert kwargs["do_normalize"] is True
    assert kwargs["max_length"] == MAX_SAMPLES
    # ONNX input name matches the official inference.py.
    assert "input_features" in detector._session.calls[0]


def test_predict_returns_incomplete_below_threshold():
    detector = _loaded_detector(SMART_TURN_THRESHOLD - 0.1)

    assert detector.predict(b"\x00\x00" * SR, SR)[0] is False


def test_predict_returns_none_for_wrong_sample_rate():
    detector = _loaded_detector(0.9)

    assert detector.predict(np.zeros(SR, dtype=np.float32), 8000) is None


def test_predict_disables_on_inference_error(caplog):
    detector = _loaded_detector(0.9)

    def boom(_inputs):
        raise RuntimeError("onnx blew up")

    detector._session.run = boom

    with caplog.at_level(logging.WARNING):
        assert detector.predict(b"\x00\x00" * SR, SR) is None

    assert detector.available is False
    assert sum("disabled" in r.message for r in caplog.records) == 1


# --------------------------------------------------------------------------
# Lazy loading / fallback
# --------------------------------------------------------------------------

def test_detector_disables_once_when_download_fails(monkeypatch, caplog):
    downloads = {"count": 0}

    def fail_download(**kwargs):
        downloads["count"] += 1
        raise RuntimeError("offline")

    monkeypatch.setattr(smart_turn, "onnxruntime", SimpleNamespace())
    monkeypatch.setattr(smart_turn, "hf_hub_download", fail_download)
    monkeypatch.setattr(smart_turn, "WhisperFeatureExtractor", lambda chunk_length=8: None)

    detector = SmartTurnDetector(SmartTurnConfig(enabled=True))

    with caplog.at_level(logging.WARNING):
        assert detector.predict(b"\x00\x00" * SR, SR) is None
        assert detector.predict(b"\x00\x00" * SR, SR) is None

    assert downloads["count"] == 1
    assert detector.available is False
    assert sum("disabled" in r.message for r in caplog.records) == 1


def test_missing_optional_dependencies_logs_single_install_hint(monkeypatch, caplog):
    monkeypatch.setattr(smart_turn, "_missing_deps_warned", False, raising=False)
    monkeypatch.setattr(smart_turn, "onnxruntime", None)
    monkeypatch.setattr(smart_turn, "WhisperFeatureExtractor", None)
    monkeypatch.setattr(smart_turn, "hf_hub_download", lambda **kwargs: "model.onnx")
    monkeypatch.setitem(sys.modules, "onnxruntime", None)
    monkeypatch.setitem(sys.modules, "transformers", None)

    detector_one = SmartTurnDetector(SmartTurnConfig(enabled=True))
    detector_two = SmartTurnDetector(SmartTurnConfig(enabled=True))

    with caplog.at_level(logging.WARNING):
        assert detector_one.warmup() is False
        assert detector_two.warmup() is False

    hints = [
        record.getMessage()
        for record in caplog.records
        if record.levelno == logging.WARNING and "pip install" in record.getMessage()
    ]
    assert len(hints) == 1
    assert "onnxruntime" in hints[0]
    assert "transformers" in hints[0]


def test_requirements_declare_smart_turn_dependencies():
    requirements = (
        Path(__file__).resolve().parents[1] / "requirements.txt"
    ).read_text(encoding="utf-8")
    packages = [
        line.strip().lower()
        for line in requirements.splitlines()
        if line.strip() and not line.strip().startswith(("#", "-"))
    ]

    def declares(package: str) -> bool:
        return any(re.match(rf"^{package}\s*>=", line) for line in packages)

    assert declares("transformers"), "transformers must be a base runtime dependency"
    assert declares("onnxruntime"), "onnxruntime must be a base runtime dependency"


def test_detector_loads_via_huggingface_hub_and_uses_cpu_provider(monkeypatch):
    created = {}

    class StubSessionOptions:
        def __init__(self):
            self.execution_mode = None
            self.inter_op_num_threads = None
            self.intra_op_num_threads = None
            self.graph_optimization_level = None

    class StubInferenceSession:
        def __init__(self, path, sess_options=None, providers=None):
            created["path"] = path
            created["providers"] = providers
            created["sess_options"] = sess_options

    stub_ort = SimpleNamespace(
        SessionOptions=StubSessionOptions,
        InferenceSession=StubInferenceSession,
        ExecutionMode=SimpleNamespace(ORT_SEQUENTIAL="seq"),
        GraphOptimizationLevel=SimpleNamespace(ORT_ENABLE_ALL="all"),
    )

    def fake_download(**kwargs):
        created["download"] = kwargs
        return "model.onnx"

    monkeypatch.setattr(smart_turn, "onnxruntime", stub_ort)
    monkeypatch.setattr(smart_turn, "hf_hub_download", fake_download)
    monkeypatch.setattr(
        smart_turn,
        "WhisperFeatureExtractor",
        lambda chunk_length=8: FakeFeatureExtractor(),
    )

    detector = SmartTurnDetector(SmartTurnConfig(enabled=True))
    assert detector.warmup() is True

    assert created["providers"] == ["CPUExecutionProvider"]
    assert created["path"] == "model.onnx"
    assert created["download"]["filename"] == "smart-turn-v3.2-cpu.onnx"
    assert created["download"]["revision"] == smart_turn.SMART_TURN_REVISION
    assert created["sess_options"].intra_op_num_threads == 1


# --------------------------------------------------------------------------
# Config parsing
# --------------------------------------------------------------------------

def test_config_from_audio_section():
    config = SmartTurnConfig.from_config(
        {
            "speech_commit_delay_ms": 650,
            "turn_detection": {
                "enabled": False,
                "model": "smart-turn-v3.2-cpu",
                "threshold": 0.6,
                "complete_delay_ms": 200,
                "incomplete_delay_ms": 3000,
            },
        },
        fallback_delay_ms=650,
    )

    assert config.enabled is False
    assert config.threshold == pytest.approx(0.6)
    assert config.complete_delay_ms == 200
    assert config.incomplete_delay_ms == 3000
    assert config.fallback_delay_ms == 650
    assert config.filename == "smart-turn-v3.2-cpu.onnx"


def test_config_defaults_when_section_missing():
    config = SmartTurnConfig.from_config({"speech_commit_delay_ms": 700}, fallback_delay_ms=700)

    assert config.enabled is True
    assert config.threshold == pytest.approx(SMART_TURN_THRESHOLD)
    assert config.complete_delay_ms == 250
    assert config.incomplete_delay_ms == 2500
    assert config.fallback_delay_ms == 700


def test_config_parses_uncertain_vad_misses_and_debug_dir():
    config = SmartTurnConfig.from_config(
        {
            "turn_detection": {
                "uncertain_threshold": 0.2,
                "uncertain_delay_ms": 800,
                "vad_required_misses": 6,
                "debug_save_dir": "data/turn_debug",
            }
        }
    )

    assert config.uncertain_threshold == pytest.approx(0.2)
    assert config.uncertain_delay_ms == 800
    assert config.vad_required_misses == 6
    assert config.debug_save_dir == "data/turn_debug"


def test_config_new_keys_have_safe_defaults():
    config = SmartTurnConfig.from_config({})

    assert config.uncertain_threshold == pytest.approx(0.15)
    assert config.uncertain_delay_ms == 900
    assert config.vad_required_misses == 8
    assert config.debug_save_dir is None


# --------------------------------------------------------------------------
# Debug WAV capture (P4)
# --------------------------------------------------------------------------


def test_save_debug_wav_writes_mono_16k_pcm(tmp_path):
    samples = np.zeros(16000, dtype=np.float32)

    path = save_debug_wav(str(tmp_path), samples, 16000, "uncertain", 0.3)

    assert path is not None
    name = Path(path).name
    assert re.fullmatch(r"\d{8}-\d{6}_uncertain_p0\.30\.wav", name)
    with wave.open(path, "rb") as handle:
        assert handle.getnchannels() == 1
        assert handle.getsampwidth() == 2
        assert handle.getframerate() == 16000
        assert handle.getnframes() == 16000


def test_save_debug_wav_is_disabled_without_directory():
    assert save_debug_wav(None, np.zeros(10, dtype=np.float32), 16000, "complete", 0.9) is None
    assert save_debug_wav("", np.zeros(10, dtype=np.float32), 16000, "complete", 0.9) is None


def test_save_debug_wav_prunes_to_max_files_keeping_newest(tmp_path):
    for index in range(205):
        (tmp_path / f"20260101-{index:06d}_complete_p0.90.wav").write_bytes(b"x")

    save_debug_wav(str(tmp_path), np.zeros(160, dtype=np.float32), 16000, "complete", 0.9)

    remaining = sorted(p.name for p in tmp_path.glob("*.wav"))
    assert len(remaining) == 200
    assert remaining[-1].endswith("_complete_p0.90.wav")
    assert "20260101-000000_complete_p0.90.wav" not in remaining


def test_save_debug_wav_prunes_only_detector_generated_files(tmp_path):
    foreign = []
    for name in ("000_custom.wav", "00_custom.wav", "0_custom.wav"):
        path = tmp_path / name
        path.write_bytes(b"x")
        foreign.append(path)
    for index in range(205):
        (tmp_path / f"20260101-{index:06d}_complete_p0.90.wav").write_bytes(b"x")

    save_debug_wav(str(tmp_path), np.zeros(160, dtype=np.float32), 16000, "complete", 0.9)

    detector_pattern = re.compile(
        r"\d{8}-\d{6}_(?:complete|uncertain|incomplete)_p\d\.\d{2}(?:_\d{3})?\.wav"
    )
    remaining_detector = sorted(
        p.name for p in tmp_path.glob("*.wav") if detector_pattern.fullmatch(p.name)
    )
    assert len(remaining_detector) == 200
    for path in foreign:
        assert path.exists(), f"{path.name} should not be pruned"


def test_predict_saves_debug_audio_when_configured(tmp_path):
    detector = _loaded_detector(0.30, debug_save_dir=str(tmp_path))

    detector.predict(b"\x00\x00" * SR, SR)

    saved = list(tmp_path.glob("*_uncertain_p0.30.wav"))
    assert len(saved) == 1


def test_predict_saves_debug_audio_as_complete_above_threshold(tmp_path):
    detector = _loaded_detector(0.93, debug_save_dir=str(tmp_path))

    detector.predict(b"\x00\x00" * SR, SR)

    assert len(list(tmp_path.glob("*_complete_p0.93.wav"))) == 1


def test_predict_does_not_save_debug_audio_by_default(monkeypatch):
    calls = []
    monkeypatch.setattr(smart_turn, "save_debug_wav", lambda *args, **kwargs: calls.append(args))
    detector = _loaded_detector(0.30)

    detector.predict(b"\x00\x00" * SR, SR)

    assert calls == []