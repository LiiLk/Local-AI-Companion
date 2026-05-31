"""Server runtime settings helpers."""

from __future__ import annotations

from typing import Any
from urllib.parse import urlsplit

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


def resolve_websocket_allowed_origins(config: dict[str, Any]) -> list[str]:
    """Return the explicit WebSocket Origin allowlist.

    WebSocket handshakes are not protected by CORS middleware, so the backend
    applies the same local-first origin policy before accepting browser clients.
    Non-browser local clients usually omit the Origin header and are handled by
    ``is_websocket_origin_allowed``.
    """
    server_config = get_server_config(config)
    websocket_config = server_config.get("websocket", {}) or {}
    if not isinstance(websocket_config, dict):
        websocket_config = {}

    configured_origins = websocket_config.get("allow_origins")
    origins = _normalize_origins(configured_origins)
    if origins is None:
        origins = resolve_cors_settings(config)["allow_origins"]
    return origins


def is_websocket_origin_allowed(config: dict[str, Any], origin: str | None) -> bool:
    """Check whether a browser WebSocket Origin may connect."""
    normalized_origin = _normalize_origin(origin)
    if normalized_origin is None:
        return True

    allowed_origins = resolve_websocket_allowed_origins(config)
    if "*" in allowed_origins:
        return True

    normalized_allowed = {
        normalized
        for value in allowed_origins
        if (normalized := _normalize_origin(value)) is not None
    }
    return normalized_origin in normalized_allowed


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


def _normalize_origin(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().rstrip("/")
    if not text:
        return None
    if text == "null":
        return text

    try:
        parsed = urlsplit(text)
    except ValueError:
        return text

    if not parsed.scheme or not parsed.netloc or not parsed.hostname:
        return text

    host = parsed.hostname.lower()
    netloc = f"{host}:{parsed.port}" if parsed.port else host
    return f"{parsed.scheme.lower()}://{netloc}"
