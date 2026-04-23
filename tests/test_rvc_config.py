from pathlib import Path

from src.utils import rvc_config
from src.utils.rvc_config import build_rvc_runtime_config


def test_build_rvc_runtime_config_applies_character_voice_overrides(monkeypatch):
    model_path = Path("C:/tmp/march7th/March-7th.pth")
    index_path = Path("C:/tmp/march7th/March-7th.index")

    def fake_resolve_rvc_paths(runtime_config: dict, character_config: dict):
        assert runtime_config["index_rate"] == 0.2
        assert runtime_config["protect"] == 0.2
        assert runtime_config["backend"] == "worker"
        assert character_config["preset"] == "march7th"
        return model_path, index_path

    monkeypatch.setattr(rvc_config, "resolve_rvc_paths", fake_resolve_rvc_paths)

    config = {
        "tts": {
            "rvc": {
                "enabled": True,
                "backend": "worker",
                "index_rate": 0.0,
                "protect": 0.33,
            }
        },
        "character": {
            "preset": "march7th",
            "voice": {
                "rvc": {
                    "index_rate": 0.2,
                    "protect": 0.2,
                }
            },
        },
    }

    runtime_config = build_rvc_runtime_config(config)

    assert runtime_config is not None
    assert runtime_config["index_rate"] == 0.2
    assert runtime_config["protect"] == 0.2
    assert runtime_config["model_path"] == str(model_path)
    assert runtime_config["index_path"] == str(index_path)


def test_build_rvc_runtime_config_keeps_base_settings_without_character_overrides(monkeypatch):
    model_path = Path("C:/tmp/base/base.pth")

    def fake_resolve_rvc_paths(runtime_config: dict, _character_config: dict):
        assert runtime_config["index_rate"] == 0.0
        assert runtime_config["protect"] == 0.33
        return model_path, None

    monkeypatch.setattr(rvc_config, "resolve_rvc_paths", fake_resolve_rvc_paths)

    config = {
        "tts": {
            "rvc": {
                "enabled": True,
                "index_rate": 0.0,
                "protect": 0.33,
            }
        },
        "character": {
            "preset": "march7th",
            "voice": {},
        },
    }

    runtime_config = build_rvc_runtime_config(config)

    assert runtime_config is not None
    assert runtime_config["index_rate"] == 0.0
    assert runtime_config["protect"] == 0.33
    assert runtime_config["model_path"] == str(model_path)
    assert runtime_config["index_path"] is None
