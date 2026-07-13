from __future__ import annotations

import json

import pytest

pytest.importorskip("PyQt6")

from desktop.qt_avatar_shell import QtAvatarShell


class FakeQtShell:
    def __init__(self):
        self.scripts: list[str] = []
        self._logged_user_turn_ids: set[int] = set()
        self._logged_assistant_turn_ids: set[int] = set()

    def _run_javascript(self, code: str) -> None:
        self.scripts.append(code)


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
