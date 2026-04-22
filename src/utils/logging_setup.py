"""Shared logging setup for runtime and conversation logs."""

from __future__ import annotations

import json
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any


LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
RUNTIME_LOG_NAME = "assistant.log"
CONVERSATION_LOG_NAME = "conversation.jsonl"
MAX_LOG_BYTES = 5 * 1024 * 1024
BACKUP_COUNT = 5


def _ensure_logs_dir(project_root: Path) -> Path:
    logs_dir = project_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    return logs_dir


def configure_root_logging(project_root: Path, *, level: int = logging.INFO) -> Path:
    """Configure root logging once with console + rotating file handlers."""
    logs_dir = _ensure_logs_dir(project_root)
    runtime_log_path = logs_dir / RUNTIME_LOG_NAME
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    if not any(getattr(handler, "_codex_console_handler", False) for handler in root_logger.handlers):
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(logging.Formatter(LOG_FORMAT))
        console_handler._codex_console_handler = True  # type: ignore[attr-defined]
        root_logger.addHandler(console_handler)

    existing_runtime_handler = next(
        (handler for handler in root_logger.handlers if getattr(handler, "_codex_runtime_log", False)),
        None,
    )
    if existing_runtime_handler is not None:
        existing_path = Path(getattr(existing_runtime_handler, "baseFilename", ""))
        if existing_path != runtime_log_path:
            root_logger.removeHandler(existing_runtime_handler)
            existing_runtime_handler.close()
            existing_runtime_handler = None

    if existing_runtime_handler is None:
        file_handler = RotatingFileHandler(
            runtime_log_path,
            maxBytes=MAX_LOG_BYTES,
            backupCount=BACKUP_COUNT,
            encoding="utf-8",
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
        file_handler._codex_runtime_log = True  # type: ignore[attr-defined]
        root_logger.addHandler(file_handler)

    return runtime_log_path


def get_conversation_logger(project_root: Path) -> tuple[logging.Logger, Path]:
    """Return a file-only logger for structured conversation events."""
    logs_dir = _ensure_logs_dir(project_root)
    conversation_log_path = logs_dir / CONVERSATION_LOG_NAME
    logger = logging.getLogger("conversation")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    existing_handler = next(
        (handler for handler in logger.handlers if getattr(handler, "_codex_conversation_log", False)),
        None,
    )
    if existing_handler is not None:
        existing_path = Path(getattr(existing_handler, "baseFilename", ""))
        if existing_path != conversation_log_path:
            logger.removeHandler(existing_handler)
            existing_handler.close()
            existing_handler = None

    if existing_handler is None:
        handler = RotatingFileHandler(
            conversation_log_path,
            maxBytes=MAX_LOG_BYTES,
            backupCount=BACKUP_COUNT,
            encoding="utf-8",
        )
        handler.setLevel(logging.INFO)
        handler.setFormatter(logging.Formatter("%(message)s"))
        handler._codex_conversation_log = True  # type: ignore[attr-defined]
        logger.addHandler(handler)

    return logger, conversation_log_path


def log_conversation_event(logger: logging.Logger, event_type: str, **payload: Any) -> None:
    """Write a single structured conversation event as JSONL."""
    event = {"event": event_type, **payload}
    logger.info(json.dumps(event, ensure_ascii=False))
