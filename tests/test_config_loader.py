from pathlib import Path

from src.utils.config_loader import load_yaml_config


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
