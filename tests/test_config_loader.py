from pathlib import Path

from src.utils.config_loader import (
    get_local_override_path,
    load_local_yaml_config,
    load_yaml_config,
    write_local_yaml_config,
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


def test_load_local_yaml_config_returns_empty_dict_when_missing(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("llm:\n  provider: ollama\n", encoding="utf-8")

    assert load_local_yaml_config(config_path) == {}


def test_write_local_yaml_config_writes_atomic_override_file(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("llm:\n  provider: ollama\n", encoding="utf-8")

    override = {
        "llm": {
            "provider": "openrouter",
            "openrouter": {
                "api_key": "secret",
                "model": "x-ai/grok-4.1-fast",
            },
        },
    }

    write_local_yaml_config(config_path, override)

    local_path = get_local_override_path(config_path)
    assert local_path.exists()
    assert load_local_yaml_config(config_path) == override
