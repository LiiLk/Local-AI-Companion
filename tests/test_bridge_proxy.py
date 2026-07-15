import threading

import pytest

from src.desktop.bridge_proxy import BridgeProxyAssistant


def make_proxy() -> BridgeProxyAssistant:
    proxy = BridgeProxyAssistant.__new__(BridgeProxyAssistant)
    proxy._lock = threading.Lock()
    proxy._pending = {}
    proxy._closed = threading.Event()
    proxy._reader = None
    return proxy


def test_rpc_removes_waiter_when_send_fails():
    class FailingSocket:
        def send(self, payload):
            raise OSError("socket closed")

    proxy = make_proxy()
    proxy._ws = FailingSocket()

    with pytest.raises(OSError, match="socket closed"):
        proxy._rpc("get_runtime_state")

    assert proxy._pending == {}


def test_reader_preserves_disconnect_error_for_pending_rpc():
    class DisconnectedSocket:
        def recv(self):
            return None

    proxy = make_proxy()
    proxy._ws = DisconnectedSocket()
    event = threading.Event()
    proxy._pending["request"] = {"event": event, "result": None, "error": None}

    proxy._read_loop()

    assert event.is_set()
    assert proxy._pending["request"]["error"] == "bridge disconnected"


def test_close_closes_socket_and_joins_reader():
    class Socket:
        closed = False

        def close(self):
            self.closed = True

    class Reader:
        joined = False

        def join(self, timeout):
            self.joined = timeout

        def is_alive(self):
            return False

    proxy = make_proxy()
    proxy._ws = Socket()
    proxy._reader = Reader()

    proxy.close()

    assert proxy._ws is None
    assert proxy._reader is None
