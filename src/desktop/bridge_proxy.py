"""
Bridge client for a Windows-only pet shell talking to a WSL (or remote) backend.

Protocol: same JSON messages as DesktopBridgeServer / desktop-bridge.js (Tauri path).
"""

from __future__ import annotations

import contextlib
import json
import logging
import threading
import time
import uuid
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)
BRIDGE_MAX_MESSAGE_SIZE = 8 * 1024 * 1024

try:
    import websocket  # websocket-client (optional)
except ImportError:  # pragma: no cover
    websocket = None  # type: ignore

try:
    import websockets
    from websockets.sync.client import connect as ws_sync_connect
except ImportError:  # pragma: no cover
    websockets = None
    ws_sync_connect = None


class BridgeProxyAssistant:
    """
    Duck-types the subset of Live2DAssistant used by QtAvatarShell / QtDesktopBridge.

    Runs on Windows; backend (ASR/LLM/TTS/mic) stays on the bridge server (often WSL).
    """

    def __init__(self, bridge_url: str = "ws://127.0.0.1:8765", *, connect_timeout: float = 15.0):
        self.bridge_url = bridge_url
        self._connect_timeout = connect_timeout
        self._ws = None
        self._lock = threading.Lock()
        self._pending: dict[str, dict[str, Any]] = {}
        self._runtime: dict[str, Any] = {
            "mode": "pipeline",
            "mic_state": "loading",
            "backend_state": "warming_up",
            "character_name": "Assistant",
            "backend": "assistant-bridge",
        }
        self._event_handler: Optional[Callable[[str, tuple], None]] = None
        self._closed = threading.Event()
        self._reader: Optional[threading.Thread] = None
        self._connect()

    def set_event_handler(self, handler: Callable[[str, tuple], None]) -> None:
        self._event_handler = handler

    def _connect(self) -> None:
        deadline = time.time() + self._connect_timeout
        last_err: Exception | None = None
        while time.time() < deadline:
            try:
                self._ws = self._open_socket()
                self._reader = threading.Thread(
                    target=self._read_loop,
                    name="bridge-proxy-reader",
                    daemon=True,
                )
                self._reader.start()
                logger.info("BridgeProxy connected to %s", self.bridge_url)
                return
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                time.sleep(0.35)
        raise ConnectionError(
            f"Could not connect to assistant bridge at {self.bridge_url}: {last_err}"
        )

    def _open_socket(self):
        if ws_sync_connect is not None:
            return ws_sync_connect(
                self.bridge_url, open_timeout=3, close_timeout=2, max_size=BRIDGE_MAX_MESSAGE_SIZE
            )
        if websocket is not None:
            return websocket.create_connection(self.bridge_url, timeout=3)
        raise RuntimeError(
            "No WebSocket client available. Install 'websockets' "
            "(preferred) or 'websocket-client' in the Windows Python env."
        )

    def _read_loop(self) -> None:
        assert self._ws is not None
        try:
            while not self._closed.is_set():
                try:
                    raw = self._ws.recv()
                except Exception:
                    if self._closed.is_set():
                        break
                    raise
                if raw is None:
                    break
                if len(raw) > BRIDGE_MAX_MESSAGE_SIZE:
                    raise ValueError("Bridge message exceeds the configured receive limit")
                try:
                    message = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                self._handle_message(message)
        except Exception as exc:  # noqa: BLE001
            if not self._closed.is_set():
                logger.warning("BridgeProxy reader stopped: %s", exc)
        finally:
            # Fail any waiters
            with self._lock:
                for waiter in self._pending.values():
                    waiter["error"] = "bridge disconnected"
                    waiter["event"].set()

    def _handle_message(self, message: dict[str, Any]) -> None:
        mtype = message.get("type")
        if mtype == "backend_ready":
            runtime = message.get("runtime") or {}
            if isinstance(runtime, dict):
                self._runtime.update(runtime)
            if self._event_handler:
                self._event_handler("onBackendReady", (dict(self._runtime),))
            return

        if mtype == "frontend_event":
            name = str(message.get("name") or "")
            args = message.get("args") or []
            runtime = message.get("runtime")
            if isinstance(runtime, dict):
                self._runtime.update(runtime)
            if self._event_handler and name:
                self._event_handler(name, tuple(args))
            return

        if mtype == "command_result":
            request_id = str(message.get("request_id") or "")
            with self._lock:
                waiter = self._pending.get(request_id)
            if not waiter:
                return
            if message.get("ok"):
                result = message.get("result") or {}
                if isinstance(result, dict):
                    self._runtime.update(result)
                waiter["result"] = result
            else:
                waiter["error"] = message.get("error") or "command failed"
            waiter["event"].set()
            return

    def _rpc(self, name: str, timeout: float = 30.0, **payload: Any) -> dict[str, Any]:
        if self._ws is None:
            raise RuntimeError("BridgeProxy is not connected")
        request_id = uuid.uuid4().hex
        event = threading.Event()
        with self._lock:
            self._pending[request_id] = {"event": event, "result": None, "error": None}
        message = {
            "type": "command",
            "name": name,
            "request_id": request_id,
            **payload,
        }
        try:
            with self._lock:
                self._ws.send(json.dumps(message, ensure_ascii=False))
        except Exception:
            with self._lock:
                self._pending.pop(request_id, None)
            raise
        if not event.wait(timeout):
            with self._lock:
                self._pending.pop(request_id, None)
            raise TimeoutError(f"Bridge command timed out: {name}")
        with self._lock:
            waiter = self._pending.pop(request_id, None) or {}
        if waiter.get("error"):
            raise RuntimeError(str(waiter["error"]))
        result = waiter.get("result")
        return result if isinstance(result, dict) else {}

    # --- Live2DAssistant surface used by Qt shell ---

    def submit_text(self, text: str) -> dict[str, Any]:
        return self._rpc("send_text", text=str(text or ""))

    def request_interrupt(self, source: str = "windows-shell") -> dict[str, Any]:
        return self._rpc("interrupt")

    def toggle_mute(self) -> dict[str, Any]:
        return self._rpc("toggle_mute")

    def get_runtime_state(self) -> dict[str, Any]:
        try:
            return self._rpc("get_runtime_state", timeout=5.0)
        except Exception:
            return dict(self._runtime)

    def toggle_debug(self) -> dict[str, Any]:
        return self._rpc("toggle_debug")

    def close(self) -> None:
        self._closed.set()
        with contextlib.suppress(Exception):
            if self._ws is not None:
                self._ws.close()
        self._ws = None
        reader = self._reader
        if reader is not None and reader is not threading.current_thread():
            reader.join(timeout=2.0)
            if not reader.is_alive():
                self._reader = None
