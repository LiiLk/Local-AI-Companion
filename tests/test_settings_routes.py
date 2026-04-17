from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.server.routes import router
from src.utils.character_loader import resolve_character_config
from src.utils.config_loader import load_local_yaml_config, load_yaml_config


def _write_config(config_path: Path, *, provider: str = "ollama", api_key: str | None = None) -> None:
    api_key_value = "null" if api_key is None else f'"{api_key}"'
    config_path.write_text(
        "\n".join(
            [
                'mode: "pipeline"',
                "character:",
                '  name: "March 7th"',
                "llm:",
                f'  provider: "{provider}"',
                "  ollama:",
                '    model: "qwen3.5:4b"',
                '    base_url: "http://localhost:11434"',
                "  openrouter:",
                '    model: "x-ai/grok-4.1-fast"',
                f"    api_key: {api_key_value}",
                '    base_url: "https://openrouter.ai/api/v1"',
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _build_client(config_path: Path, monkeypatch) -> TestClient:
    monkeypatch.setattr("src.server.routes._project_config_path", lambda: config_path)

    app = FastAPI()
    app.include_router(router, prefix="/api")
    app.state.config = resolve_character_config(load_yaml_config(config_path))
    app.state.character = app.state.config.get("character", {})
    return TestClient(app)


def test_get_llm_settings_reads_merged_config(tmp_path: Path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    _write_config(config_path, provider="openrouter", api_key="secret")
    client = _build_client(config_path, monkeypatch)

    response = client.get("/api/settings/llm")

    assert response.status_code == 200
    body = response.json()
    assert body["provider"] == "openrouter"
    assert body["ollama"]["model"] == "qwen3.5:4b"
    assert body["openrouter"]["api_key_configured"] is True
    assert body["openrouter"]["api_key_source"] == "saved"


def test_save_llm_settings_writes_local_override_and_refreshes_app_state(tmp_path: Path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    _write_config(config_path)
    client = _build_client(config_path, monkeypatch)

    response = client.put(
        "/api/settings/llm",
        json={
            "provider": "openrouter",
            "ollama": {
                "base_url": "http://localhost:11434",
                "model": "qwen3.5:4b",
            },
            "openrouter": {
                "base_url": "https://openrouter.ai/api/v1",
                "model": "openai/gpt-4.1-mini",
                "api_key": "new-secret",
            },
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["provider"] == "openrouter"
    assert body["openrouter"]["model"] == "openai/gpt-4.1-mini"

    local_override = load_local_yaml_config(config_path)
    assert local_override["llm"]["provider"] == "openrouter"
    assert local_override["llm"]["openrouter"]["api_key"] == "new-secret"
    assert client.app.state.config["llm"]["provider"] == "openrouter"


def test_test_llm_settings_uses_ollama_preload(tmp_path: Path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    _write_config(config_path)
    client = _build_client(config_path, monkeypatch)
    seen: dict[str, str] = {}

    class FakeOllamaLLM:
        def __init__(self, *, model: str, base_url: str, **kwargs):
            seen["model"] = model
            seen["base_url"] = base_url

        def preload(self) -> None:
            seen["preloaded"] = "yes"

        async def close(self) -> None:
            seen["closed"] = "yes"

    monkeypatch.setattr("src.server.routes.OllamaLLM", FakeOllamaLLM)

    response = client.post(
        "/api/settings/llm/test",
        json={
            "provider": "ollama",
            "ollama": {
                "base_url": "http://localhost:11434",
                "model": "llama3.2:3b",
            },
            "openrouter": {
                "base_url": "https://openrouter.ai/api/v1",
                "model": "x-ai/grok-4.1-fast",
            },
        },
    )

    assert response.status_code == 200
    assert response.json()["ok"] is True
    assert seen == {
        "model": "llama3.2:3b",
        "base_url": "http://localhost:11434",
        "preloaded": "yes",
        "closed": "yes",
    }


def test_get_config_reports_openrouter_model(tmp_path: Path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    _write_config(config_path, provider="openrouter", api_key="secret")
    client = _build_client(config_path, monkeypatch)

    response = client.get("/api/config")

    assert response.status_code == 200
    body = response.json()
    assert body["llm_provider"] == "openrouter"
    assert body["llm_model"] == "x-ai/grok-4.1-fast"
