from pathlib import Path

from src.utils.config_loader import load_yaml_config
from src.utils.character_loader import (
    resolve_live2d_desktop_model,
    resolve_live2d_web_model,
)


def test_load_yaml_config_merges_local_override(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    local_path = tmp_path / "config.local.yaml"

    config_path.write_text(
        "llm:\n  provider: ollama\n  openrouter:\n    api_key: null\n",
        encoding="utf-8",
    )
    local_path.write_text(
        "llm:\n  provider: openrouter\n  openrouter:\n    api_key: secret\n",
        encoding="utf-8",
    )

    loaded = load_yaml_config(config_path)

    assert loaded["llm"]["provider"] == "openrouter"
    assert loaded["llm"]["openrouter"]["api_key"] == "secret"


def test_live2d_model_paths_are_resolved_for_each_frontend():
    config = {
        "live2d": {
            "model": {
                "path": "assets/models/starling/",
                "settings_file": "starling.model3.json",
            }
        }
    }

    assert resolve_live2d_web_model(config) == (
        "/assets/models/starling/",
        "starling.model3.json",
    )
    assert resolve_live2d_desktop_model(config) == (
        "../../assets/models/starling/",
        "starling.model3.json",
    )


def test_live2d_model_paths_reject_absolute_and_parent_paths():
    absolute = {
        "live2d": {"model": {"path": "/tmp/model", "settings_file": "a.json"}}
    }
    parent = {
        "live2d": {"model": {"path": "../model", "settings_file": "a.json"}}
    }
    unserved = {
        "live2d": {"model": {"path": "models/local", "settings_file": "a.json"}}
    }
    absolute_settings = {
        "live2d": {"model": {"path": "assets/models/local", "settings_file": "/tmp/a.json"}}
    }
    remote_settings = {
        "live2d": {
            "model": {
                "path": "assets/models/local",
                "settings_file": "https://example.test/a.json",
            }
        }
    }

    assert resolve_live2d_web_model(absolute) == (None, None)
    assert resolve_live2d_desktop_model(parent) == (None, None)
    assert resolve_live2d_web_model(unserved) == (None, None)
    assert resolve_live2d_web_model(absolute_settings) == (None, None)
    assert resolve_live2d_desktop_model(remote_settings) == (None, None)
