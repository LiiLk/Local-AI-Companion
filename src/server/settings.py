"""Server runtime settings helpers."""

from __future__ import annotations

from typing import Any

DEFAULT_SERVER_HOST = "127.0.0.1"
DEFAULT_SERVER_PORT = 8000


def get_server_config(config: dict[str, Any]) -> dict[str, Any]:
    server_config = config.get("server", {}) or {}
    return server_config if isinstance(server_config, dict) else {}


def resolve_server_host(config: dict[str, Any]) -> str:
    host = get_server_config(config).get("host", DEFAULT_SERVER_HOST)
    return str(host or DEFAULT_SERVER_HOST)


def resolve_server_port(config: dict[str, Any]) -> int:
    port = get_server_config(config).get("port", DEFAULT_SERVER_PORT)
    try:
        return int(port)
    except (TypeError, ValueError):
        return DEFAULT_SERVER_PORT


def default_cors_origins(port: int = DEFAULT_SERVER_PORT) -> list[str]:
    return [
        f"http://127.0.0.1:{port}",
        f"http://localhost:{port}",
    ]


def resolve_cors_settings(config: dict[str, Any]) -> dict[str, Any]:
    server_config = get_server_config(config)
    cors_config = server_config.get("cors", {}) or {}
    if not isinstance(cors_config, dict):
        cors_config = {}

    configured_origins = cors_config.get("allow_origins")
    origins = _normalize_origins(configured_origins)
    if origins is None:
        origins = default_cors_origins(resolve_server_port(config))

    allow_credentials = bool(cors_config.get("allow_credentials", False))
    if "*" in origins:
        allow_credentials = False

    return {
        "allow_origins": origins,
        "allow_credentials": allow_credentials,
        "allow_methods": _normalize_list(cors_config.get("allow_methods")) or ["*"],
        "allow_headers": _normalize_list(cors_config.get("allow_headers")) or ["*"],
    }


def _normalize_origins(value: Any) -> list[str] | None:
    if value is None:
        return None
    return _normalize_list(value)


def _normalize_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    if isinstance(value, (list, tuple, set)):
        return [str(part).strip() for part in value if str(part).strip()]
    return [str(value).strip()] if str(value).strip() else []
