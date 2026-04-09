from pathlib import Path

from src.tts import rvc_provider
from src.tts.rvc_provider import resolve_rvc_paths


def test_resolve_rvc_paths_from_model_name(tmp_path, monkeypatch):
    monkeypatch.setattr(rvc_provider, "DEFAULT_RVC_MODELS_DIR", tmp_path / "resources" / "rvc")
    monkeypatch.setattr(rvc_provider, "DEFAULT_VOICE_MODELS_DIR", tmp_path / "resources" / "voices")

    model_dir = rvc_provider.DEFAULT_RVC_MODELS_DIR / "march7th"
    model_dir.mkdir(parents=True)
    (model_dir / "March-7th.pth").write_bytes(b"model")
    (model_dir / "March-7th.index").write_bytes(b"index")

    model_path, index_path = resolve_rvc_paths({"enabled": True, "model_name": "march7th"}, {})

    assert model_path.name == "March-7th.pth"
    assert index_path is not None
    assert index_path.name == "March-7th.index"


def test_resolve_rvc_paths_from_character_preset(tmp_path, monkeypatch):
    monkeypatch.setattr(rvc_provider, "DEFAULT_RVC_MODELS_DIR", tmp_path / "resources" / "rvc")
    monkeypatch.setattr(rvc_provider, "DEFAULT_VOICE_MODELS_DIR", tmp_path / "resources" / "voices")

    model_dir = rvc_provider.DEFAULT_VOICE_MODELS_DIR / "march7th"
    model_dir.mkdir(parents=True)
    (model_dir / "voice.pth").write_bytes(b"model")

    model_path, index_path = resolve_rvc_paths({"enabled": True}, {"preset": "march7th"})

    assert model_path.name == "voice.pth"
    assert index_path is None


def test_resolve_rvc_paths_prefers_explicit_model_dir(tmp_path):
    model_dir = tmp_path / "my-rvc-model"
    model_dir.mkdir()
    (model_dir / "custom.pth").write_bytes(b"model")
    (model_dir / "custom.index").write_bytes(b"index")

    model_path, index_path = resolve_rvc_paths({"enabled": True, "model_dir": str(model_dir)}, {})

    assert model_path == (model_dir / "custom.pth").resolve()
    assert index_path == (model_dir / "custom.index").resolve()
