import json
import logging
from pathlib import Path

from src.utils.logging_setup import (
    configure_root_logging,
    get_conversation_logger,
    log_conversation_event,
)


def test_configure_root_logging_creates_runtime_log(tmp_path):
    runtime_log_path = configure_root_logging(tmp_path)

    logging.getLogger("test.runtime").info("runtime log smoke test")

    assert runtime_log_path.exists()
    assert "runtime log smoke test" in runtime_log_path.read_text(encoding="utf-8")


def test_real_project_log_root_is_redirected_during_tests():
    real_project_root = Path(__file__).resolve().parents[1]

    runtime_log_path = configure_root_logging(real_project_root)

    assert (real_project_root / "logs").resolve() not in runtime_log_path.resolve().parents


def test_conversation_logger_writes_jsonl(tmp_path):
    conversation_logger, conversation_log_path = get_conversation_logger(tmp_path)

    log_conversation_event(
        conversation_logger,
        "user_transcription",
        turn_id=7,
        text="hello there",
    )

    assert conversation_log_path.exists()
    lines = conversation_log_path.read_text(encoding="utf-8").strip().splitlines()
    payload = json.loads(lines[-1])
    assert payload["event"] == "user_transcription"
    assert payload["turn_id"] == 7
    assert payload["text"] == "hello there"
