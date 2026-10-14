# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE.
"""Toolbar that keeps actions accessible when the window becomes narrow."""

from PySide6 import QtCore, QtWidgets


class ResponsiveToolBar(QtWidgets.QToolBar):
    """Show labels when they fit, otherwise use icons and native overflow."""

    def __init__(self, title: str, parent: QtWidgets.QWidget | None = None):
        super().__init__(title, parent)
        self._expanded_width = None
        self._resize_timer = QtCore.QTimer(self)
        self._resize_timer.setSingleShot(True)
        self._resize_timer.timeout.connect(self._update_button_style)
        self.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        self.iconSizeChanged.connect(self._invalidate_expanded_width)

    def _invalidate_expanded_width(self):
        self._expanded_width = None
        self._resize_timer.start(0)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._resize_timer.start(0)

    def showEvent(self, event):
        super().showEvent(event)
        self._invalidate_expanded_width()

    def actionEvent(self, event):
        super().actionEvent(event)
        # Qt can deliver events while the base constructor is still running.
        if hasattr(self, "_resize_timer"):
            self._invalidate_expanded_width()

    def changeEvent(self, event):
        super().changeEvent(event)
        if hasattr(self, "_resize_timer") and event.type() in (
            QtCore.QEvent.FontChange,
            QtCore.QEvent.StyleChange,
            QtCore.QEvent.LanguageChange,
        ):
            self._invalidate_expanded_width()

    def _update_button_style(self):
        if self._expanded_width is None:
            self.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
            # Cache the labelled size: the compact size hint would otherwise
            # make successive resize events alternate between both styles.
            self._expanded_width = self.sizeHint().width()

        button_style = (
            QtCore.Qt.ToolButtonIconOnly
            if self.width() < self._expanded_width
            else QtCore.Qt.ToolButtonTextBesideIcon
        )
        if self.toolButtonStyle() != button_style:
            self.setToolButtonStyle(button_style)
