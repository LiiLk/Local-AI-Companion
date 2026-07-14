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


def test_web_app_live2d_retries_legacy_model_path():
    app_js = Path("frontend/web/js/app.js").read_text(encoding="utf-8")

    assert "modelPath: '/live2d/runtime-assets/models/march7th_tauri/'" in app_js
    assert "modelPath: '/assets/models/march7th/'" in app_js
    assert "}) || await this.live2d.init({" in app_js


def test_bridge_proxy_uses_unbounded_websockets_receive_limit():
    bridge_proxy = Path("src/desktop/bridge_proxy.py").read_text(encoding="utf-8")

    assert "max_size=None" in bridge_proxy
