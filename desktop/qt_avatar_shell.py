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
from PyQt6.QtWidgets import QApplication, QHBoxLayout, QPushButton, QVBoxLayout, QWidget


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

    # Kept for compatibility with older frontend bundles.
    @pyqtSlot(int, int, int, int, result=str)
    def setHudInteractiveRect(self, screen_x: int, screen_y: int, width: int, height: int) -> str:
        return self._result({"status": "ok", "ignored": True})

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
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setAutoFillBackground(False)
        self.setStyleSheet("background: transparent; border: 0;")
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.NoContextMenu)
        self.page().setBackgroundColor(QColor(0, 0, 0, 0))
        settings = self.settings()
        settings.setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessFileUrls, True)
        settings.setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessRemoteUrls, True)
        settings.setAttribute(QWebEngineSettings.WebAttribute.JavascriptEnabled, True)


class HudOverlay(QWidget):
    toggle_mute_requested = pyqtSignal()
    interrupt_requested = pyqtSignal()
    toggle_chat_requested = pyqtSignal()
    toggle_settings_requested = pyqtSignal()
    toggle_layout_requested = pyqtSignal()

    def __init__(self, shell: "QtAvatarShell", *, always_on_top: bool = True):
        super().__init__(None)
        self._shell = shell
        self._drag_origin: Optional[QPoint] = None
        self._shell_origin: Optional[QPoint] = None

        flags = (
            Qt.WindowType.Tool
            | Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.NoDropShadowWindowHint
        )
        if always_on_top:
            flags |= Qt.WindowType.WindowStaysOnTopHint
        self.setWindowFlags(flags)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, False)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        self.setAutoFillBackground(False)
        self.setStyleSheet(
            """
            QWidget#hudRoot {
                background: rgba(0, 0, 0, 0.42);
                border: 1px solid rgba(255, 255, 255, 0.16);
                border-radius: 16px;
            }
            QPushButton {
                min-width: 44px;
                height: 34px;
                border: 1px solid rgba(255, 255, 255, 0.26);
                border-radius: 10px;
                background: rgba(255, 255, 255, 0.02);
                color: #f4f8ff;
                font-size: 11px;
                font-weight: 700;
                padding: 0 9px;
            }
            QPushButton:hover {
                border-color: rgba(255, 255, 255, 0.42);
                background: rgba(255, 255, 255, 0.08);
            }
            QPushButton#muteButton[active="true"] {
                border-color: rgba(255, 122, 122, 0.72);
                color: #ffd1d1;
            }
            """
        )

        root = QWidget(self)
        root.setObjectName("hudRoot")
        root_layout = QHBoxLayout(root)
        root_layout.setContentsMargins(10, 8, 10, 8)
        root_layout.setSpacing(6)

        container = QVBoxLayout(self)
        container.setContentsMargins(0, 0, 0, 0)
        container.addWidget(root)

        self._mute_button = QPushButton("MIC", root)
        self._mute_button.setObjectName("muteButton")
        self._stop_button = QPushButton("STOP", root)
        self._chat_button = QPushButton("CHAT", root)
        self._settings_button = QPushButton("SET", root)
        self._layout_button = QPushButton("EXPAND", root)

        root_layout.addWidget(self._mute_button)
        root_layout.addWidget(self._stop_button)
        root_layout.addWidget(self._chat_button)
        root_layout.addWidget(self._settings_button)
        root_layout.addWidget(self._layout_button)

        self._mute_button.clicked.connect(self.toggle_mute_requested)
        self._stop_button.clicked.connect(self.interrupt_requested)
        self._chat_button.clicked.connect(self.toggle_chat_requested)
        self._settings_button.clicked.connect(self.toggle_settings_requested)
        self._layout_button.clicked.connect(self.toggle_layout_requested)

        self.adjustSize()

    def set_layout_mode(self, layout_mode: str) -> None:
        self._layout_button.setText("REDUCE" if layout_mode == "expanded" else "EXPAND")

    def set_mute_active(self, active: bool) -> None:
        self._mute_button.setProperty("active", "true" if active else "false")
        self._mute_button.style().unpolish(self._mute_button)
        self._mute_button.style().polish(self._mute_button)

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            target = self.childAt(event.position().toPoint())
            if not isinstance(target, QPushButton):
                self._drag_origin = event.globalPosition().toPoint()
                self._shell_origin = self._shell.frameGeometry().topLeft()
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self._drag_origin is None or self._shell_origin is None:
            super().mouseMoveEvent(event)
            return
        delta = event.globalPosition().toPoint() - self._drag_origin
        self._shell.move(self._shell_origin + delta)
        event.accept()

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        self._drag_origin = None
        self._shell_origin = None
        super().mouseReleaseEvent(event)


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

        flags = (
            Qt.WindowType.Window
            | Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.NoDropShadowWindowHint
            | Qt.WindowType.WindowTransparentForInput
        )
        if always_on_top:
            flags |= Qt.WindowType.WindowStaysOnTopHint
        self.setWindowFlags(flags)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, False)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
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

        self._hud = HudOverlay(self, always_on_top=always_on_top)
        self._hud.toggle_mute_requested.connect(self._toggle_mute)
        self._hud.interrupt_requested.connect(self._interrupt_turn)
        self._hud.toggle_chat_requested.connect(self._toggle_chat)
        self._hud.toggle_settings_requested.connect(self._toggle_settings)
        self._hud.toggle_layout_requested.connect(self._toggle_layout)
        self._hud.set_layout_mode(self._layout_mode)

        self._hud_state_timer = QTimer(self)
        self._hud_state_timer.setInterval(850)
        self._hud_state_timer.timeout.connect(self._refresh_hud_runtime_state)

        self._view.setUrl(QUrl.fromLocalFile(str(self._html_path.resolve())))

    def run(self, on_loaded: Callable[[], None]) -> int:
        self._loaded_callback = on_loaded
        self.show()
        self._sync_hud_geometry()
        self._hud.show()
        self.raise_()
        self._hud.raise_()
        self._hud_state_timer.start()
        self._refresh_hud_runtime_state()
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
        self._hud_state_timer.stop()
        if self._hud.isVisible():
            self._hud.close()
        super().closeEvent(event)
        self._app.quit()

    def moveEvent(self, event) -> None:  # noqa: N802
        self._sync_hud_geometry()
        super().moveEvent(event)

    def resizeEvent(self, event) -> None:  # noqa: N802
        self._sync_hud_geometry()
        super().resizeEvent(event)

    @pyqtSlot(str)
    def _run_javascript(self, code: str) -> None:
        self._view.page().runJavaScript(code)

    @pyqtSlot()
    def _close_internal(self) -> None:
        if self._hud.isVisible():
            self._hud.close()
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
        self._hud.set_layout_mode(normalized)
        self._sync_hud_geometry()
        mode_handler = (
            "window.__desktopBridgeSetLayoutMode?.('expanded') || window.setExpandedMode?.();"
            if normalized == "expanded"
            else "window.__desktopBridgeSetLayoutMode?.('compact') || window.setCompactMode?.();"
        )
        # Wait one frame so QWebEngine has applied the new viewport before matrix recompute.
        QTimer.singleShot(16, lambda: self.evaluate_js_requested.emit(mode_handler))

    @pyqtSlot(bool)
    def _on_load_finished(self, ok: bool) -> None:
        if ok:
            self.evaluate_js_requested.emit("document.body.classList.add('qt-external-hud');")
        if ok and self._loaded_callback:
            QTimer.singleShot(0, self._loaded_callback)

    def _sync_hud_geometry(self) -> None:
        self._hud.adjustSize()
        width = max(10, self._hud.width())
        height = max(10, self._hud.height())
        frame = self.frameGeometry()
        x = frame.x() + frame.width() - width - 22
        y = frame.y() + frame.height() - height - 18
        self._hud.setGeometry(x, y, width, height)

    def _refresh_hud_runtime_state(self) -> None:
        try:
            runtime = self._assistant.get_runtime_state() or {}
        except Exception:
            return
        muted = runtime.get("mic_state") == "muted"
        self._hud.set_mute_active(bool(muted))

    def _toggle_mute(self) -> None:
        try:
            runtime = self._assistant.toggle_mute() or {}
        except Exception:
            return
        self._hud.set_mute_active(runtime.get("mic_state") == "muted")

    def _interrupt_turn(self) -> None:
        try:
            self._assistant.request_interrupt("qt_hud")
        except Exception:
            return

    def _toggle_chat(self) -> None:
        self.evaluate_js_requested.emit("document.getElementById('chat-toggle-button')?.click();")

    def _toggle_settings(self) -> None:
        self.evaluate_js_requested.emit("document.getElementById('settings-button')?.click();")

    def _toggle_layout(self) -> None:
        next_layout = "expanded" if self._layout_mode == "compact" else "compact"
        self.set_layout_mode(next_layout)
