from src.server.settings import (
    DEFAULT_SERVER_HOST,
    DEFAULT_SERVER_PORT,
    default_cors_origins,
    resolve_cors_settings,
    resolve_server_host,
    resolve_server_port,
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
