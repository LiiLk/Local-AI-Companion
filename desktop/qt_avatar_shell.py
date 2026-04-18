from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Callable, Optional

from PyQt6.QtCore import QObject, QPoint, QRect, Qt, QTimer, QUrl, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QColor, QCursor
from PyQt6.QtWebChannel import QWebChannel
from PyQt6.QtWebEngineCore import QWebEnginePage, QWebEngineSettings
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout

if sys.platform == "win32":
    import ctypes
    from ctypes import wintypes

    _USER32 = ctypes.windll.user32
    _GWL_EXSTYLE = -20
    _WS_EX_LAYERED = 0x00080000
    _WS_EX_TRANSPARENT = 0x00000020
    _SWP_NOSIZE = 0x0001
    _SWP_NOMOVE = 0x0002
    _SWP_NOZORDER = 0x0004
    _SWP_NOACTIVATE = 0x0010
    _SWP_FRAMECHANGED = 0x0020

    _USER32.GetWindowLongW.argtypes = [wintypes.HWND, ctypes.c_int]
    _USER32.GetWindowLongW.restype = ctypes.c_long
    _USER32.SetWindowLongW.argtypes = [wintypes.HWND, ctypes.c_int, ctypes.c_long]
    _USER32.SetWindowLongW.restype = ctypes.c_long
    _USER32.SetWindowPos.argtypes = [
        wintypes.HWND,
        wintypes.HWND,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_uint,
    ]
    _USER32.SetWindowPos.restype = wintypes.BOOL


LAYOUT_SIZES: dict[str, tuple[int, int]] = {
    "compact": (860, 760),
    "expanded": (1040, 980),
}


class QtDesktopBridge(QObject):
    def __init__(self, assistant, shell: "QtAvatarShell"):
        super().__init__()
        self._assistant = assistant
        self._shell = shell

    def _result(self, payload: dict) -> str:
        return json.dumps(payload, ensure_ascii=False)

    def _invoke(self, fn: Callable[[], dict]) -> str:
        try:
            return self._result(fn())
        except Exception as exc:
            return self._result({"status": "error", "message": str(exc)})

    @pyqtSlot(str, result=str)
    def sendText(self, text: str) -> str:
        return self._invoke(lambda: self._assistant.submit_text(text))

    @pyqtSlot(result=str)
    def interrupt(self) -> str:
        return self._invoke(lambda: self._assistant.request_interrupt("qt"))

    @pyqtSlot(result=str)
    def toggleMute(self) -> str:
        return self._invoke(self._assistant.toggle_mute)

    @pyqtSlot(result=str)
    def getRuntimeState(self) -> str:
        return self._invoke(self._assistant.get_runtime_state)

    @pyqtSlot(result=str)
    def toggleDebug(self) -> str:
        return self._invoke(self._assistant.toggle_debug)

    @pyqtSlot(str, result=str)
    def setLayoutMode(self, layout: str) -> str:
        self._shell.set_layout_mode(layout)
        return self._result({"status": "ok", "layout": layout, **self._assistant.get_runtime_state()})

    @pyqtSlot(int, int, result=str)
    def startDrag(self, screen_x: int, screen_y: int) -> str:
        self._shell.start_drag(screen_x, screen_y)
        return self._result({"status": "ok"})

    @pyqtSlot(int, int, result=str)
    def dragMove(self, screen_x: int, screen_y: int) -> str:
        self._shell.drag_move(screen_x, screen_y)
        return self._result({"status": "ok"})

    @pyqtSlot(int, int, int, int, result=str)
    def setHudInteractiveRect(self, screen_x: int, screen_y: int, width: int, height: int) -> str:
        self._shell.set_hud_interactive_rect(screen_x, screen_y, width, height)
        return self._result({"status": "ok"})

    @pyqtSlot(result=str)
    def endDrag(self) -> str:
        self._shell.end_drag()
        return self._result({"status": "ok"})


class TransparentWebView(QWebEngineView):
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, False)
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, False)
        self.setAutoFillBackground(False)
        self.setStyleSheet("background: transparent; border: 0;")
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.NoContextMenu)
        self.page().setBackgroundColor(QColor(0, 0, 0, 0))
        settings = self.settings()
        settings.setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessFileUrls, True)
        settings.setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessRemoteUrls, True)
        settings.setAttribute(QWebEngineSettings.WebAttribute.JavascriptEnabled, True)


class QtAvatarShell(QWidget):
    evaluate_js_requested = pyqtSignal(str)
    close_requested = pyqtSignal()
    layout_mode_requested = pyqtSignal(str)

    def __init__(
        self,
        assistant,
        html_path: Path,
        *,
        width: int = 860,
        height: int = 760,
        x: int | None = None,
        y: int | None = None,
        always_on_top: bool = True,
    ):
        self._app = QApplication.instance() or QApplication(sys.argv)
        super().__init__(None)
        self._assistant = assistant
        self._html_path = html_path
        self._loaded_callback: Optional[Callable[[], None]] = None
        self._drag_origin: Optional[QPoint] = None
        self._window_origin: Optional[QPoint] = None
        self._layout_mode = "compact"
        self._hud_interactive_rect: Optional[QRect] = None
        self._click_through_enabled = False
        self._last_top_left: Optional[QPoint] = None

        flags = (
            Qt.WindowType.Window
            | Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.NoDropShadowWindowHint
        )
        if always_on_top:
            flags |= Qt.WindowType.WindowStaysOnTopHint
        self.setWindowFlags(flags)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, False)
        self.setAutoFillBackground(False)
        self.setStyleSheet("background: transparent;")
        self.resize(width, height)
        if x is not None and y is not None:
            self.move(x, y)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._view = TransparentWebView(self)
        layout.addWidget(self._view)

        self._bridge = QtDesktopBridge(assistant, self)
        channel = QWebChannel(self._view.page())
        channel.registerObject("desktopBridge", self._bridge)
        self._view.page().setWebChannel(channel)
        self._view.loadFinished.connect(self._on_load_finished)

        self.evaluate_js_requested.connect(self._run_javascript)
        self.close_requested.connect(self._close_internal)
        self.layout_mode_requested.connect(self._apply_layout_mode)

        self._cursor_poll_timer = QTimer(self)
        self._cursor_poll_timer.setInterval(40)
        self._cursor_poll_timer.timeout.connect(self._update_click_through_state)

        self._view.setUrl(QUrl.fromLocalFile(str(self._html_path.resolve())))

    def run(self, on_loaded: Callable[[], None]) -> int:
        self._loaded_callback = on_loaded
        self.show()
        self._last_top_left = self.frameGeometry().topLeft()
        self.raise_()
        self.activateWindow()
        self._cursor_poll_timer.start()
        self._set_click_through_enabled(True)
        return self._app.exec()

    def evaluate_js(self, code: str) -> None:
        self.evaluate_js_requested.emit(code)

    def close(self) -> None:  # type: ignore[override]
        self.close_requested.emit()

    def start_drag(self, screen_x: int, screen_y: int) -> None:
        self._drag_origin = QPoint(screen_x, screen_y)
        self._window_origin = self.frameGeometry().topLeft()

    def drag_move(self, screen_x: int, screen_y: int) -> None:
        if self._drag_origin is None or self._window_origin is None:
            return
        delta = QPoint(screen_x, screen_y) - self._drag_origin
        self.move(self._window_origin + delta)

    def end_drag(self) -> None:
        self._drag_origin = None
        self._window_origin = None

    def set_layout_mode(self, layout: str) -> None:
        self.layout_mode_requested.emit(layout)

    def set_hud_interactive_rect(self, screen_x: int, screen_y: int, width: int, height: int) -> None:
        safe_w = max(0, int(width))
        safe_h = max(0, int(height))
        if safe_w == 0 or safe_h == 0:
            self._hud_interactive_rect = None
        else:
            self._hud_interactive_rect = QRect(int(screen_x), int(screen_y), safe_w, safe_h)
        self._update_click_through_state()

    def closeEvent(self, event) -> None:  # noqa: N802
        self._cursor_poll_timer.stop()
        super().closeEvent(event)
        self._app.quit()

    def moveEvent(self, event) -> None:  # noqa: N802
        new_top_left = self.frameGeometry().topLeft()
        if self._last_top_left is not None and self._hud_interactive_rect is not None:
            delta = new_top_left - self._last_top_left
            if not delta.isNull():
                self._hud_interactive_rect.translate(delta)
        self._last_top_left = new_top_left
        super().moveEvent(event)

    @pyqtSlot(str)
    def _run_javascript(self, code: str) -> None:
        self._view.page().runJavaScript(code)

    @pyqtSlot()
    def _close_internal(self) -> None:
        if self.isVisible():
            super().close()
        else:
            self._app.quit()

    def _set_click_through_enabled(self, enabled: bool) -> None:
        enabled = bool(enabled)
        if self._click_through_enabled == enabled:
            return

        if sys.platform == "win32":
            hwnd = int(self.winId())
            ex_style = _USER32.GetWindowLongW(hwnd, _GWL_EXSTYLE)
            target_style = ex_style | _WS_EX_LAYERED
            if enabled:
                target_style |= _WS_EX_TRANSPARENT
            else:
                target_style &= ~_WS_EX_TRANSPARENT

            if target_style != ex_style:
                ctypes.set_last_error(0)
                _USER32.SetWindowLongW(hwnd, _GWL_EXSTYLE, target_style)
                _USER32.SetWindowPos(
                    hwnd,
                    0,
                    0,
                    0,
                    0,
                    0,
                    _SWP_NOMOVE | _SWP_NOSIZE | _SWP_NOZORDER | _SWP_NOACTIVATE | _SWP_FRAMECHANGED,
                )
        else:
            self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, enabled)

        self._click_through_enabled = enabled

    def _is_cursor_in_hud_rect(self) -> bool:
        if self._hud_interactive_rect is None:
            return False
        cursor_pos = QCursor.pos()
        return self._hud_interactive_rect.adjusted(-8, -8, 8, 8).contains(cursor_pos)

    @pyqtSlot()
    def _update_click_through_state(self) -> None:
        self._set_click_through_enabled(not self._is_cursor_in_hud_rect())

    @pyqtSlot(str)
    def _apply_layout_mode(self, layout: str) -> None:
        normalized = "expanded" if layout == "expanded" else "compact"
        width, height = LAYOUT_SIZES[normalized]
        current_geometry = self.geometry()
        bottom_right = current_geometry.bottomRight()
        self.setGeometry(bottom_right.x() - width + 1, bottom_right.y() - height + 1, width, height)
        self._layout_mode = normalized
        mode_handler = (
            "window.__desktopBridgeSetLayoutMode?.('expanded') || window.setExpandedMode?.();"
            if normalized == "expanded"
            else "window.__desktopBridgeSetLayoutMode?.('compact') || window.setCompactMode?.();"
        )
        # Wait one frame so QWebEngine has applied the new viewport before matrix recompute.
        QTimer.singleShot(16, lambda: self.evaluate_js_requested.emit(mode_handler))

    @pyqtSlot(bool)
    def _on_load_finished(self, ok: bool) -> None:
        if ok and self._loaded_callback:
            QTimer.singleShot(0, self._loaded_callback)
