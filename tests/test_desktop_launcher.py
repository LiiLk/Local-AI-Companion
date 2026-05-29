import importlib


def test_desktop_launcher_normalizes_localhost_to_loopback():
    launcher = importlib.import_module("src.desktop.__main__")

    assert launcher.normalize_backend_host("localhost") == "127.0.0.1"
    assert launcher.normalize_backend_host("  localhost  ") == "127.0.0.1"
    assert launcher.normalize_backend_host(None) == "127.0.0.1"


def test_desktop_launcher_keeps_explicit_wildcard_host():
    launcher = importlib.import_module("src.desktop.__main__")

    assert launcher.normalize_backend_host("0.0.0.0") == "0.0.0.0"
