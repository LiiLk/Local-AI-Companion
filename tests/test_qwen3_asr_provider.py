import tempfile
from pathlib import Path

import pytest

from src.asr.qwen3_asr_provider import Qwen3ASRProvider


def test_qwen3_asr_worker_import_check_rejects_missing_worker():
    assert Qwen3ASRProvider._worker_import_check(
        Path(__file__),
        Path(__file__).with_name("missing_worker.py"),
    ) is False


def test_qwen3_asr_worker_validation_rejects_non_python_script():
    with tempfile.NamedTemporaryFile(suffix=".cmd", delete=False) as handle:
        worker_script = Path(handle.name)

    try:
        with pytest.raises(RuntimeError, match="must be a Python file"):
            Qwen3ASRProvider._validate_worker_process_inputs(Path(__file__), worker_script)
    finally:
        worker_script.unlink(missing_ok=True)
