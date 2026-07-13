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
