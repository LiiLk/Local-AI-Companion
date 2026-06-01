from src.server.settings import (
    DEFAULT_SERVER_HOST,
    DEFAULT_SERVER_PORT,
    default_cors_origins,
    is_websocket_origin_allowed,
    resolve_cors_settings,
    resolve_server_host,
    resolve_server_port,
    resolve_websocket_allowed_origins,
)


def test_server_defaults_are_loopback_only():
    config = {}

    assert resolve_server_host(config) == DEFAULT_SERVER_HOST
    assert resolve_server_port(config) == DEFAULT_SERVER_PORT
    assert resolve_cors_settings(config)["allow_origins"] == default_cors_origins(
        DEFAULT_SERVER_PORT
    )


def test_server_config_overrides_host_port_and_derives_cors_port():
    config = {"server": {"host": "localhost", "port": "9000", "cors": {}}}

    assert resolve_server_host(config) == "localhost"
    assert resolve_server_port(config) == 9000
    assert resolve_cors_settings(config)["allow_origins"] == default_cors_origins(9000)


def test_cors_explicit_origins_are_normalized():
    config = {
        "server": {
            "cors": {
                "allow_origins": "http://localhost:3000, http://127.0.0.1:5173",
                "allow_credentials": True,
            }
        }
    }

    assert resolve_cors_settings(config) == {
        "allow_origins": ["http://localhost:3000", "http://127.0.0.1:5173"],
        "allow_credentials": True,
        "allow_methods": ["*"],
        "allow_headers": ["*"],
    }


def test_cors_wildcard_disables_credentials():
    config = {
        "server": {
            "cors": {
                "allow_origins": ["*"],
                "allow_credentials": True,
            }
        }
    }

    settings = resolve_cors_settings(config)

    assert settings["allow_origins"] == ["*"]
    assert settings["allow_credentials"] is False


def test_websocket_origins_default_to_local_cors_policy():
    config = {"server": {"port": 8765}}

    assert resolve_websocket_allowed_origins(config) == default_cors_origins(8765)
    assert is_websocket_origin_allowed(config, "http://127.0.0.1:8765")
    assert is_websocket_origin_allowed(config, "http://localhost:8765/")
    assert not is_websocket_origin_allowed(config, "https://evil.example")


def test_websocket_origins_allow_non_browser_clients_without_origin():
    config = {}

    assert is_websocket_origin_allowed(config, None)
    assert is_websocket_origin_allowed(config, "")


def test_websocket_origins_can_be_configured_separately_from_cors():
    config = {
        "server": {
            "port": 8000,
            "cors": {"allow_origins": ["http://127.0.0.1:8000"]},
            "websocket": {"allow_origins": ["http://localhost:3000"]},
        }
    }

    assert resolve_websocket_allowed_origins(config) == ["http://localhost:3000"]
    assert is_websocket_origin_allowed(config, "http://localhost:3000")
    assert not is_websocket_origin_allowed(config, "http://127.0.0.1:8000")


def test_websocket_origins_reject_malformed_origin_headers():
    config = {"server": {"port": 8000}}

    assert not is_websocket_origin_allowed(config, "localhost:8000")
    assert not is_websocket_origin_allowed(config, "http://127.0.0.1:8000:bad")
    assert not is_websocket_origin_allowed(config, "http://127.0.0.1:8000@evil.example")


def test_websocket_origins_ignore_malformed_allowlist_entries():
    config = {
        "server": {
            "websocket": {
                "allow_origins": ["localhost:8000", "http://127.0.0.1:8000"],
            }
        }
    }

    assert is_websocket_origin_allowed(config, "http://127.0.0.1:8000")
    assert not is_websocket_origin_allowed(config, "localhost:8000")
