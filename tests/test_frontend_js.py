from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


def test_frontend_node_suite():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is not installed")

    test_files = sorted((Path(__file__).parent / "js").glob("*.test.js"))
    result = subprocess.run(
        [node, "--test", *(str(path) for path in test_files)],
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr


def test_web_app_live2d_uses_server_configured_model():
    app_js = Path("frontend/web/js/app.js").read_text(encoding="utf-8")

    assert "serverConfig?.live2d_model_path" in app_js
    assert "serverConfig?.live2d_model_name" in app_js
    assert "/assets/models/default/" not in app_js
    assert "march7th" not in app_js.lower()


def test_desktop_fetches_runtime_before_initializing_avatar():
    desktop_html = Path("frontend/live2d/index.html").read_text(encoding="utf-8")
    startup = desktop_html.index("document.addEventListener('DOMContentLoaded'")
    refresh = desktop_html.index("await refreshRuntime();", startup)
    layout = desktop_html.index("await setLayout('compact');", startup)

    assert refresh < layout
    assert "models/default" not in desktop_html
    assert "window.onExpressionChange" in desktop_html
    assert "window.Live2DAPI.setExpression(expression)" in desktop_html


def test_bridge_proxy_uses_bounded_websockets_receive_limit():
    bridge_proxy = Path("src/desktop/bridge_proxy.py").read_text(encoding="utf-8")

    assert "BRIDGE_MAX_MESSAGE_SIZE = 8 * 1024 * 1024" in bridge_proxy
    assert "max_size=BRIDGE_MAX_MESSAGE_SIZE" in bridge_proxy
    assert "max_size=None" not in bridge_proxy
