from __future__ import annotations

import json

import pytest

QtAvatarShell = pytest.importorskip("desktop.qt_avatar_shell").QtAvatarShell


class FakeQtShell:
    def __init__(self):
        self.scripts: list[str] = []
        self._logged_user_turn_ids: set[int] = set()
        self._logged_assistant_turn_ids: set[int] = set()

    def _run_javascript(self, code: str) -> None:
        self.scripts.append(code)


class FakeShutdownAssistant:
    def __init__(self, calls, error=None):
        self.calls = calls
        self.error = error

    def request_shutdown(self, source: str) -> None:
        self.calls.append(("shutdown", source))
        if self.error:
            raise self.error


class FakeQuitShell:
    def __init__(self):
        self._quit_in_progress = False
        self.calls = []
        self._assistant = FakeShutdownAssistant(self.calls)

    def _close_internal(self) -> None:
        self.calls.append("close")


def test_remote_audio_event_is_forwarded_to_webview():
    shell = FakeQtShell()
    payload = {"audio": "ZmFrZQ==", "turn_id": 7, "duration": 250}

    QtAvatarShell._handle_frontend_event(
        shell,
        "onAudioReady",
        json.dumps([payload]),
    )

    assert shell.scripts == [
        'window["onAudioReady"]?.(...[{"audio": "ZmFrZQ==", "turn_id": 7, "duration": 250}])'
    ]


def test_quit_requests_backend_cleanup_before_closing_shell():
    shell = FakeQuitShell()

    QtAvatarShell._request_quit(shell)
    QtAvatarShell._request_quit(shell)

    assert shell.calls == [("shutdown", "qt_hud"), "close"]


def test_quit_closes_shell_when_backend_request_fails():
    shell = FakeQuitShell()
    shell._assistant = FakeShutdownAssistant(shell.calls, ConnectionError("disconnected"))

    QtAvatarShell._request_quit(shell)

    assert shell.calls == [("shutdown", "qt_hud"), "close"]
