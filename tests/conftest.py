"""Shared pytest fixtures.

The test suite must not write into the real ``logs/`` directory. Several
modules (``src.assistant.app``, ``src.desktop.__main__``) call
``configure_root_logging`` / ``get_conversation_logger`` at import time, which
happens during test collection. The module-level patch below redirects those
import-time calls to a throwaway temp directory before any test module is
imported. Explicit per-test calls that pass their own root (e.g. ``tmp_path``)
are left untouched.
"""

import logging
import tempfile
from pathlib import Path

import src.utils.logging_setup as _logging_setup

_REAL_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_TEST_LOG_ROOT = Path(tempfile.mkdtemp(prefix="lac-test-logs-"))

_original_configure_root_logging = _logging_setup.configure_root_logging
_original_get_conversation_logger = _logging_setup.get_conversation_logger


def _resolves_to_real_project_root(project_root) -> bool:
    try:
        return Path(project_root).resolve() == _REAL_PROJECT_ROOT
    except TypeError:
        return False


def _redirected_configure_root_logging(project_root, *, level=logging.INFO):
    if _resolves_to_real_project_root(project_root):
        project_root = _TEST_LOG_ROOT
    return _original_configure_root_logging(project_root, level=level)


def _redirected_get_conversation_logger(project_root):
    if _resolves_to_real_project_root(project_root):
        project_root = _TEST_LOG_ROOT
    return _original_get_conversation_logger(project_root)


_logging_setup.configure_root_logging = _redirected_configure_root_logging
_logging_setup.get_conversation_logger = _redirected_get_conversation_logger