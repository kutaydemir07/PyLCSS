# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE.
"""Centered hint shown over a widget while it has nothing to display."""

from __future__ import annotations

from collections.abc import Callable

from PySide6 import QtCore, QtWidgets

__all__ = ["EmptyStateOverlay"]


class EmptyStateOverlay(QtCore.QObject):
    """Overlay label over ``target`` that is visible while ``is_empty()`` is true."""

    def __init__(
        self,
        target: QtWidgets.QWidget,
        text: str,
        is_empty: Callable[[], bool],
    ) -> None:
        super().__init__(target)
        self._target = target
        self._is_empty = is_empty
        self._label = QtWidgets.QLabel(text, target)
        self._label.setAlignment(QtCore.Qt.AlignCenter)
        self._label.setWordWrap(True)
        self._label.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents)
        self._label.setStyleSheet(
            "color: gray; font-size: 11pt; background: transparent;"
        )
        target.installEventFilter(self)
        # Contents change through many code paths, so poll instead of hooking each one.
        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(400)
        self._timer.timeout.connect(self.refresh)
        self._timer.start()
        self.refresh()

    def eventFilter(self, obj, event) -> bool:
        if event.type() == QtCore.QEvent.Resize:
            self._label.setGeometry(self._target.rect())
        return False

    def set_text(self, text: str) -> None:
        self._label.setText(text)

    def refresh(self) -> None:
        empty = bool(self._is_empty())
        self._label.setGeometry(self._target.rect())
        self._label.setVisible(empty)
        if empty:
            self._label.raise_()
