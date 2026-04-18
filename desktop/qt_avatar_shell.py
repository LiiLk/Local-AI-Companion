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
from PyQt6.QtWidgets import QApplication, QFrame, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

if sys.platform == "win32":
    import ctypes
    from ctypes import wintypes

    _USER32 = ctypes.windll.user32
    _DWMAPI = ctypes.windll.dwmapi

    _GWL_STYLE = -16
    _GWL_EXSTYLE = -20

    _WS_CAPTION = 0x00C00000
    _WS_THICKFRAME = 0x00040000
    _WS_MINIMIZEBOX = 0x00020000
    _WS_MAXIMIZEBOX = 0x00010000
    _WS_SYSMENU = 0x00080000
    _WS_BORDER = 0x00800000
    _WS_DLGFRAME = 0x00400000
    _WS_OVERLAPPEDWINDOW = 0x00CF0000

    _WS_EX_TOOLWINDOW = 0x00000080
    _WS_EX_APPWINDOW = 0x00040000
    _WS_EX_LAYERED = 0x00080000
    _WS_EX_TRANSPARENT = 0x00000020
    _WS_EX_NOACTIVATE = 0x08000000

    _SWP_NOSIZE = 0x0001
    _SWP_NOMOVE = 0x0002
    _SWP_NOZORDER = 0x0004
    _SWP_NOACTIVATE = 0x0010
    _SWP_FRAMECHANGED = 0x0020

    _DWMWA_NCRENDERING_POLICY = 2
    _DWMNCRP_DISABLED = 1
    _DWMWA_TRANSITIONS_FORCEDISABLED = 3
    _DWMWA_WINDOW_CORNER_PREFERENCE = 33
    _DWMWCP_DONOTROUND = 1
    _DWMWA_BORDER_COLOR = 34
    _DWMWA_COLOR_NONE = 0xFFFFFFFE

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

    _DWMAPI.DwmSetWindowAttribute.argtypes = [
        wintypes.HWND,
        ctypes.c_uint,
        ctypes.c_void_p,
        ctypes.c_uint,
    ]
    _DWMAPI.DwmSetWindowAttribute.restype = ctypes.c_long


LAYOUT_SIZES: dict[str, tuple[int, int]] = {
    "compact": (860, 760),
    "expanded": (1040, 980),
}


def _apply_windows_borderless_style(widget: QWidget, *, click_through: bool, no_activate: bool) -> None:
    if sys.platform != "win32":
        return

    hwnd = int(widget.winId())
    style = _USER32.GetWindowLongW(hwnd, _GWL_STYLE)
    style &= ~(
        _WS_OVERLAPPEDWINDOW
        | _WS_CAPTION
        | _WS_THICKFRAME
        | _WS_MINIMIZEBOX
        | _WS_MAXIMIZEBOX
        | _WS_SYSMENU
        | _WS_BORDER
        | _WS_DLGFRAME
    )
    _USER32.SetWindowLongW(hwnd, _GWL_STYLE, style)

    ex_style = _USER32.GetWindowLongW(hwnd, _GWL_EXSTYLE)
    ex_style |= _WS_EX_LAYERED | _WS_EX_TOOLWINDOW
    ex_style &= ~_WS_EX_APPWINDOW
    if click_through:
        ex_style |= _WS_EX_TRANSPARENT
    else:
        ex_style &= ~_WS_EX_TRANSPARENT
    if no_activate:
        ex_style |= _WS_EX_NOACTIVATE
    else:
        ex_style &= ~_WS_EX_NOACTIVATE
    _USER32.SetWindowLongW(hwnd, _GWL_EXSTYLE, ex_style)

    _USER32.SetWindowPos(
        hwnd,
        0,
        0,
        0,
        0,
        0,
        _SWP_NOMOVE | _SWP_NOSIZE | _SWP_NOZORDER | _SWP_NOACTIVATE | _SWP_FRAMECHANGED,
    )

    def _set_dwm_int(attribute: int, value: int) -> None:
        raw = ctypes.c_int(value)
        _DWMAPI.DwmSetWindowAttribute(
            hwnd,
            attribute,
            ctypes.byref(raw),
            ctypes.sizeof(raw),
        )

    # Remove any non-client rendering artifacts (border/shadow/corners).
    _set_dwm_int(_DWMWA_NCRENDERING_POLICY, _DWMNCRP_DISABLED)
    _set_dwm_int(_DWMWA_TRANSITIONS_FORCEDISABLED, 1)
    _set_dwm_int(_DWMWA_WINDOW_CORNER_PREFERENCE, _DWMWCP_DONOTROUND)
    _set_dwm_int(_DWMWA_BORDER_COLOR, _DWMWA_COLOR_NONE)


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
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        self.setAutoFillBackground(False)
        self.setStyleSheet("background: transparent; border: 0; outline: 0;")
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.NoContextMenu)
        self.page().setBackgroundColor(QColor(0, 0, 0, 0))
        settings = self.settings()
        settings.setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessFileUrls, True)
        settings.setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessRemoteUrls, True)
        settings.setAttribute(QWebEngineSettings.WebAttribute.JavascriptEnabled, True)


class DragHandle(QWidget):
    drag_started = pyqtSignal(QPoint)
    drag_moved = pyqtSignal(QPoint)
    drag_ended = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setCursor(Qt.CursorShape.OpenHandCursor)
        self.setFixedHeight(20)

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            self.drag_started.emit(event.globalPosition().toPoint())
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if event.buttons() & Qt.MouseButton.LeftButton:
            self.drag_moved.emit(event.globalPosition().toPoint())
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        self.setCursor(Qt.CursorShape.OpenHandCursor)
        self.drag_ended.emit()
        super().mouseReleaseEvent(event)


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
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:1,
                    stop:0 rgba(16, 19, 29, 236),
                    stop:1 rgba(10, 12, 20, 224)
                );
                border: 1px solid rgba(255, 255, 255, 0.09);
                border-radius: 14px;
            }
            QWidget#dragHandle {
                background: rgba(255, 255, 255, 0.05);
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 7px;
            }
            QLabel#statusLabel {
                color: rgba(233, 243, 255, 0.96);
                font-size: 10px;
                font-weight: 700;
                letter-spacing: 0.1em;
            }
            QFrame#statusDot {
                background: rgba(141, 255, 178, 0.95);
                border: 1px solid rgba(226, 244, 255, 0.28);
                border-radius: 4px;
            }
            QLabel#metaLabel {
                color: rgba(198, 215, 236, 0.78);
                font-size: 9px;
            }
            QPushButton {
                min-width: 44px;
                height: 33px;
                border: 1px solid rgba(255, 255, 255, 0.18);
                border-radius: 9px;
                background: rgba(255, 255, 255, 0.03);
                color: #e7f2ff;
                font-size: 10px;
                font-weight: 700;
                padding: 0 9px;
            }
            QPushButton:hover {
                border-color: rgba(145, 196, 255, 0.54);
                background: rgba(124, 176, 255, 0.16);
            }
            QPushButton#muteButton[active="true"] {
                border-color: rgba(255, 128, 128, 0.7);
                background: rgba(255, 122, 122, 0.18);
                color: #ffd6d6;
            }
            """
        )

        root = QWidget(self)
        root.setObjectName("hudRoot")
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(10, 8, 10, 8)
        root_layout.setSpacing(5)

        container = QVBoxLayout(self)
        container.setContentsMargins(0, 0, 0, 0)
        container.addWidget(root)

        self._drag_handle = DragHandle(root)
        self._drag_handle.setObjectName("dragHandle")
        drag_layout = QHBoxLayout(self._drag_handle)
        drag_layout.setContentsMargins(8, 0, 8, 0)
        drag_layout.setSpacing(0)
        drag_hint = QLabel("  • • •  DRAG  • • •", self._drag_handle)
        drag_hint.setStyleSheet("color: rgba(240, 248, 255, 0.72); font-size: 9px; font-weight: 700;")
        drag_layout.addWidget(drag_hint)
        drag_layout.addStretch(1)

        self._status_dot = QFrame(root)
        self._status_dot.setObjectName("statusDot")
        self._status_dot.setFixedSize(8, 8)
        self._status_label = QLabel("LISTENING", root)
        self._status_label.setObjectName("statusLabel")
        self._meta_label = QLabel("Desktop mascot overlay", root)
        self._meta_label.setObjectName("metaLabel")

        self._mute_button = QPushButton("MIC", root)
        self._mute_button.setObjectName("muteButton")
        self._stop_button = QPushButton("STOP", root)
        self._chat_button = QPushButton("CHAT", root)
        self._settings_button = QPushButton("SET", root)
        self._layout_button = QPushButton("EXPAND", root)

        buttons_layout = QHBoxLayout()
        buttons_layout.setContentsMargins(0, 0, 0, 0)
        buttons_layout.setSpacing(6)
        buttons_layout.addWidget(self._mute_button)
        buttons_layout.addWidget(self._stop_button)
        buttons_layout.addWidget(self._chat_button)
        buttons_layout.addWidget(self._settings_button)
        buttons_layout.addWidget(self._layout_button)

        status_row = QHBoxLayout()
        status_row.setContentsMargins(0, 0, 0, 0)
        status_row.setSpacing(6)
        status_row.addWidget(self._status_dot, 0, Qt.AlignmentFlag.AlignVCenter)
        status_row.addWidget(self._status_label, 0, Qt.AlignmentFlag.AlignVCenter)
        status_row.addStretch(1)

        root_layout.addWidget(self._drag_handle)
        root_layout.addLayout(status_row)
        root_layout.addWidget(self._meta_label)
        root_layout.addLayout(buttons_layout)

        self._mute_button.clicked.connect(self.toggle_mute_requested)
        self._stop_button.clicked.connect(self.interrupt_requested)
        self._chat_button.clicked.connect(self.toggle_chat_requested)
        self._settings_button.clicked.connect(self.toggle_settings_requested)
        self._layout_button.clicked.connect(self.toggle_layout_requested)
        self._drag_handle.drag_started.connect(self._on_drag_started)
        self._drag_handle.drag_moved.connect(self._on_drag_moved)
        self._drag_handle.drag_ended.connect(self._on_drag_ended)

        self.adjustSize()

    def set_layout_mode(self, layout_mode: str) -> None:
        self._layout_button.setText("REDUCE" if layout_mode == "expanded" else "EXPAND")

    def set_mute_active(self, active: bool) -> None:
        self._mute_button.setProperty("active", "true" if active else "false")
        self._mute_button.style().unpolish(self._mute_button)
        self._mute_button.style().polish(self._mute_button)

    def set_status(self, text: str, meta: str = "") -> None:
        normalized = str(text or "listening").strip().lower()
        self._status_label.setText(normalized.upper())
        color_map = {
            "listening": "rgba(141, 255, 178, 0.95)",
            "speaking": "rgba(123, 210, 255, 0.96)",
            "warming up": "rgba(255, 211, 107, 0.96)",
            "muted": "rgba(255, 156, 108, 0.96)",
            "degraded": "rgba(255, 196, 111, 0.96)",
            "error": "rgba(255, 122, 122, 0.98)",
        }
        dot_color = color_map.get(normalized, "rgba(141, 255, 178, 0.95)")
        self._status_dot.setStyleSheet(
            "background: {color}; border: 1px solid rgba(226, 244, 255, 0.28); border-radius: 4px;".format(
                color=dot_color
            )
        )
        self._meta_label.setText(meta or "Desktop mascot overlay")

    def _on_drag_started(self, screen_pos: QPoint) -> None:
        self._drag_origin = screen_pos
        self._shell_origin = self._shell.frameGeometry().topLeft()

    def _on_drag_moved(self, screen_pos: QPoint) -> None:
        if self._drag_origin is None or self._shell_origin is None:
            return
        delta = screen_pos - self._drag_origin
        self._shell.move(self._shell_origin + delta)

    def _on_drag_ended(self) -> None:
        self._drag_origin = None
        self._shell_origin = None

    def apply_native_style(self) -> None:
        _apply_windows_borderless_style(self, click_through=False, no_activate=False)


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
            Qt.WindowType.Tool
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
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        self.setAutoFillBackground(False)
        self.setStyleSheet("background: transparent;")
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
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
        _apply_windows_borderless_style(self, click_through=True, no_activate=True)
        self._sync_hud_geometry()
        self._hud.show()
        self._hud.apply_native_style()
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
        backend_state = runtime.get("backend_state") or "warming_up"
        muted = runtime.get("mic_state") == "muted"
        self._hud.set_mute_active(bool(muted))
        status = "listening"
        if backend_state == "error":
            status = "error"
        elif backend_state == "warming_up":
            status = "warming up"
        elif muted:
            status = "muted"
        elif runtime.get("response_active") or runtime.get("playback_active"):
            status = "speaking"
        elif backend_state == "degraded":
            status = "degraded"
        model_name = runtime.get("active_llm_model") or runtime.get("character_name") or "desktop mascot"
        self._hud.set_status(status, str(model_name))

    def _toggle_mute(self) -> None:
        try:
            runtime = self._assistant.toggle_mute() or {}
        except Exception:
            return
        self._hud.set_mute_active(runtime.get("mic_state") == "muted")
        self._refresh_hud_runtime_state()

    def _interrupt_turn(self) -> None:
        try:
            self._assistant.request_interrupt("qt_hud")
        except Exception:
            return
        self._refresh_hud_runtime_state()

    def _toggle_chat(self) -> None:
        self.evaluate_js_requested.emit("window.__qtToggleChat?.();")

    def _toggle_settings(self) -> None:
        self.evaluate_js_requested.emit("window.__qtToggleSettings?.();")

    def _toggle_layout(self) -> None:
        next_layout = "expanded" if self._layout_mode == "compact" else "compact"
        self.set_layout_mode(next_layout)
