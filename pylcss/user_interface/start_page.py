# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.
"""Choose an installed PyLCSS workspace before loading engineering libraries."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

from PySide6 import QtCore, QtGui, QtWidgets

from pylcss.user_interface.common.theme_manager import is_dark_theme
from pylcss.workspaces import create_window

import threading

logger = logging.getLogger(__name__)


class WorkspaceImportWorker(QtCore.QObject):
    """Asynchronously imports heavy scientific dependencies without blocking the Qt event loop."""

    finished = QtCore.Signal(str)
    failed = QtCore.Signal(str, str)

    def __init__(self, parent: QtCore.QObject | None = None) -> None:
        super().__init__(parent)
        self._lock = threading.Lock()
        self._loaded: set[str] = set()
        self._target_requested: str | None = None

    def warmup(self, available: Sequence[str]) -> None:
        def _run() -> None:
            for name in ("mdao", "cae"):
                if name in available:
                    with self._lock:
                        if name in self._loaded:
                            continue
                    try:
                        self._import_one(name)
                        with self._lock:
                            self._loaded.add(name)
                            target = self._target_requested
                        if target == name:
                            self.finished.emit(name)
                    except Exception as exc:
                        with self._lock:
                            target = self._target_requested
                        if target == name:
                            self.failed.emit(name, str(exc))
        threading.Thread(target=_run, daemon=True).start()

    def request_load(self, name: str) -> None:
        with self._lock:
            if name in self._loaded:
                self.finished.emit(name)
                return
            self._target_requested = name

        def _run() -> None:
            try:
                self._import_one(name)
                with self._lock:
                    self._loaded.add(name)
                self.finished.emit(name)
            except Exception as exc:
                self.failed.emit(name, str(exc))
        threading.Thread(target=_run, daemon=True).start()

    def _import_one(self, name: str) -> None:
        if name == "mdao":
            import pylcss.mdao.user_interface.main_application_window  # noqa: F401
        elif name == "cae":
            import pylcss.cae.user_interface.main_window  # noqa: F401


class WorkspaceButton(QtWidgets.QPushButton):
    """Clean, two-line workspace launch button."""

    def __init__(
        self,
        title: str,
        description: str,
        enabled: bool = True,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(16, 8, 16, 8)
        layout.setSpacing(2)

        self.title_label = QtWidgets.QLabel(title)
        self.title_label.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents, True)
        self.title_label.setAlignment(QtCore.Qt.AlignCenter)

        desc_text = (
            description
            if enabled
            else f"{description} • Not installed"
        )
        self.desc_label = QtWidgets.QLabel(desc_text)
        self.desc_label.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents, True)
        self.desc_label.setAlignment(QtCore.Qt.AlignCenter)

        layout.addWidget(self.title_label)
        layout.addWidget(self.desc_label)
        self.setFixedHeight(64)
        self.setEnabled(enabled)

    def setEnabled(self, enabled: bool) -> None:
        super().setEnabled(enabled)
        if hasattr(self, "title_label") and hasattr(self, "desc_label"):
            if enabled:
                self.title_label.setStyleSheet(
                    "font-size: 13px; font-weight: bold; color: #ffffff; background: transparent;"
                )
                self.desc_label.setStyleSheet(
                    "font-size: 10.5px; color: rgba(255, 255, 255, 0.85); background: transparent;"
                )
                self.setCursor(QtCore.Qt.PointingHandCursor)
            else:
                self.title_label.setStyleSheet(
                    "font-size: 13px; font-weight: bold; color: #888888; background: transparent;"
                )
                self.desc_label.setStyleSheet(
                    "font-size: 10.5px; color: #888888; background: transparent;"
                )
                self.setCursor(QtCore.Qt.ForbiddenCursor)


class StartPage(QtWidgets.QMainWindow):
    """Compact workspace chooser dialog with canonical branding and background warmup."""

    def __init__(self, available: Sequence[str], preload: bool = True) -> None:
        super().__init__()
        QtWidgets.QApplication.instance().setQuitOnLastWindowClosed(False)
        self.setWindowTitle("PyLCSS — Choose a workspace")

        icon_path = Path(__file__).with_name("icon.png")
        if icon_path.is_file():
            self.setWindowIcon(QtGui.QIcon(str(icon_path)))

        self.setFixedSize(460, 390)
        self.setWindowFlags(
            QtCore.Qt.Window
            | QtCore.Qt.WindowTitleHint
            | QtCore.Qt.WindowSystemMenuHint
            | QtCore.Qt.WindowMinimizeButtonHint
            | QtCore.Qt.WindowCloseButtonHint
        )
        self._available = set(available)
        self._windows: dict[str, QtWidgets.QMainWindow] = {}
        self._buttons: dict[str, WorkspaceButton] = {}
        self._centered = False

        container = QtWidgets.QWidget()
        self.setCentralWidget(container)
        layout = QtWidgets.QVBoxLayout(container)
        layout.setContentsMargins(36, 20, 36, 20)
        layout.setSpacing(7)

        # PyLCSS Brand Logo
        if icon_path.is_file():
            logo_label = QtWidgets.QLabel()
            pixmap = QtGui.QPixmap(str(icon_path))
            scaled_pixmap = pixmap.scaled(
                68, 68, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation
            )
            logo_label.setPixmap(scaled_pixmap)
            logo_label.setAlignment(QtCore.Qt.AlignCenter)
            layout.addWidget(logo_label)

        # App Title
        title = QtWidgets.QLabel("PyLCSS")
        title.setAlignment(QtCore.Qt.AlignCenter)
        title.setStyleSheet("font-size: 22px; font-weight: 700;")
        layout.addWidget(title)

        # Subtitle
        self.subtitle = QtWidgets.QLabel("Choose a workspace to open")
        self.subtitle.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(self.subtitle)

        # Indeterminate Progress Bar for Loading
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setFixedHeight(4)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.hide()
        layout.addWidget(self.progress_bar)

        # Workspace Buttons
        for name, title_text, description in (
            (
                "mdao",
                "PyLCSS MDAO",
                "System modeling, optimization & surrogate models",
            ),
            (
                "cae",
                "PyLCSS CAE",
                "CAD, meshing, simulation & topology optimization",
            ),
        ):
            is_avail = name in self._available
            button = WorkspaceButton(title_text, description, enabled=is_avail)
            button.clicked.connect(
                lambda checked=False, selected=name: self.open_workspace(selected)
            )
            layout.addWidget(button)
            self._buttons[name] = button

        self._apply_theme()

        # Non-blocking background loader and warmup worker
        self._worker = WorkspaceImportWorker(self)
        self._worker.finished.connect(self._on_import_finished)
        self._worker.failed.connect(self._on_import_failed)
        if preload:
            self._worker.warmup(available)

    def center_on_screen(self) -> None:
        """Center the compact window on the active primary screen."""
        screen = QtGui.QGuiApplication.primaryScreen()
        if screen:
            geo = screen.availableGeometry()
            x = geo.x() + (geo.width() - self.width()) // 2
            y = geo.y() + (geo.height() - self.height()) // 2
            self.move(x, y)

    def showEvent(self, event: QtGui.QShowEvent) -> None:
        super().showEvent(event)
        if not self._centered:
            self.center_on_screen()
            self._centered = True

    def _apply_theme(self, theme_name: str | None = None) -> None:
        dark = is_dark_theme() if theme_name is None else (theme_name == "dark")
        dim_color = "#b5bac1" if dark else "#667085"
        self.subtitle.setStyleSheet(
            f"font-size: 11px; color: {dim_color}; margin-bottom: 2px;"
        )
        bg_col = "#282a2e" if dark else "#d5dce5"
        chunk_col = "#d29922" if dark else "#4f6f91"
        self.progress_bar.setStyleSheet(
            f"QProgressBar {{ border: none; background-color: {bg_col}; height: 4px; border-radius: 2px; }}"
            f"QProgressBar::chunk {{ background-color: {chunk_col}; border-radius: 2px; }}"
        )

    def apply_theme(self, theme_name: str | None = None) -> None:
        """Called automatically when application theme changes."""
        self._apply_theme(theme_name)

    def open_workspace(self, name: str) -> None:
        if name in self._windows:
            self._windows[name].showMaximized()
            self._windows[name].raise_()
            self.hide()
            return

        # Immediate visual feedback so the user sees work is active
        self.subtitle.setText(f"Loading PyLCSS {name.upper()}...")
        self.progress_bar.show()
        for btn in self._buttons.values():
            btn.setEnabled(False)
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)

        # Delegate library import to background worker so main Qt event loop remains 100% interactive
        self._worker.request_load(name)

    def _on_import_finished(self, name: str) -> None:
        self.subtitle.setText(f"Initializing {name.upper()}...")
        QtWidgets.QApplication.processEvents()
        try:
            if name not in self._windows:
                window = create_window(name)
                window.setAttribute(QtCore.Qt.WA_DeleteOnClose)
                window.destroyed.connect(lambda: self.workspace_closed(name))
                self._windows[name] = window
            self._windows[name].showMaximized()
            self._windows[name].raise_()
            self.hide()
        except Exception as exc:
            logger.exception("Workspace startup failed")
            QtWidgets.QMessageBox.critical(
                self, "Could not open workspace", str(exc)
            )
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
            self._reset_loading_state()

    def _on_import_failed(self, name: str, error: str) -> None:
        QtWidgets.QApplication.restoreOverrideCursor()
        self._reset_loading_state()
        logger.error(f"Failed to load workspace {name}: {error}")
        QtWidgets.QMessageBox.critical(self, "Could not open workspace", error)

    def _reset_loading_state(self) -> None:
        self.progress_bar.hide()
        self.subtitle.setText("Choose a workspace to open")
        self._apply_theme()
        for k, btn in self._buttons.items():
            btn.setEnabled(k in self._available)

    def workspace_closed(self, name: str) -> None:
        self._windows.pop(name, None)
        self.show()
        self.raise_()
        self.activateWindow()

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        if any(not window.close() for window in list(self._windows.values())):
            event.ignore()
            return
        QtWidgets.QApplication.instance().quit()
        event.accept()

