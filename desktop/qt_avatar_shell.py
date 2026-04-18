from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Callable, Optional

from PyQt6.QtCore import QObject, QPoint, Qt, QTimer, QUrl, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QColor
from PyQt6.QtWebChannel import QWebChannel
from PyQt6.QtWebEngineCore import QWebEnginePage, QWebEngineSettings
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout


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

    @pyqtSlot(result=str)
    def endDrag(self) -> str:
        self._shell.end_drag()
        return self._result({"status": "ok"})


class TransparentWebView(QWebEngineView):
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, True)
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

        flags = Qt.WindowType.FramelessWindowHint | Qt.WindowType.Tool
        if always_on_top:
            flags |= Qt.WindowType.WindowStaysOnTopHint
        self.setWindowFlags(flags)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, True)
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

        self._view.setUrl(QUrl.fromLocalFile(str(self._html_path.resolve())))

    def run(self, on_loaded: Callable[[], None]) -> int:
        self._loaded_callback = on_loaded
        self.show()
        self.raise_()
        self.activateWindow()
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

    def closeEvent(self, event) -> None:  # noqa: N802
        super().closeEvent(event)
        self._app.quit()

    @pyqtSlot(str)
    def _run_javascript(self, code: str) -> None:
        self._view.page().runJavaScript(code)

    @pyqtSlot()
    def _close_internal(self) -> None:
        if self.isVisible():
            super().close()
        else:
            self._app.quit()

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
