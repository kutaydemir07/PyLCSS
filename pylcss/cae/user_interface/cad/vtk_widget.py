# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE.
"""Keep VTK's graphics context within the lifetime of its Qt native window."""

from __future__ import annotations

from PySide6 import QtCore, QtGui, QtWidgets
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor


class VTKRenderWidget(QVTKRenderWindowInteractor):
    """Render on Qt's GUI thread, after placement, and clean up before HWND loss.

    The upstream widget captures winId() in its constructor. Qt can replace
    that handle when a viewer is inserted into a splitter or another window.
    Initializing VTK before that move leaves its WGL context on the old HWND.
    """

    _render_requested = QtCore.Signal()

    def __init__(self, parent=None):
        # Qt sends native-window events during the base constructor.
        self._closed = False
        self._context_active = False
        self._bound_window_id = None
        super().__init__(parent)
        self._render_requested.connect(
            self._schedule_render, QtCore.Qt.ConnectionType.QueuedConnection
        )

    def request_render(self):
        """Queue a repaint; safe for callers outside the GUI thread."""
        if not self._closed:
            self._render_requested.emit()

    @QtCore.Slot()
    def _schedule_render(self):
        if not self._closed and self.isVisible():
            # QWidget.update coalesces requests and cannot re-enter VTK from
            # a render observer. Hidden widgets repaint when shown again.
            self.update()

    def Render(self):
        self.request_render()

    def Initialize(self):
        self.request_render()

    def Start(self):
        # Qt owns the event loop; VTK must not start a second one.
        self.request_render()

    def paintEvent(self, event):
        if (
            self._closed or not self.isVisible()
            or self.width() <= 0 or self.height() <= 0
        ):
            return
        if not self._context_active:
            # Bind only once Qt has finished parenting and showing the widget.
            self._bound_window_id = int(self.winId())
            self._RenderWindow.SetWindowInfo(str(self._bound_window_id))
            self._context_active = True
            self._Iren.Initialize()
            self._Iren.Enable()
        self._Iren.Render()

    def release_context(self):
        """Release GL resources while the old native surface still exists."""
        if not self._context_active:
            return
        self._Timer.stop()
        self._Iren.Disable()
        self._RenderWindow.Finalize()
        self._context_active = False
        self._bound_window_id = None

    def event(self, event):
        if "_RenderWindow" in self.__dict__:
            if event.type() == QtCore.QEvent.Type.ParentAboutToChange:
                self.release_context()
            elif event.type() == QtCore.QEvent.Type.PlatformSurface:
                if event.surfaceEventType() == (
                    QtGui.QPlatformSurfaceEvent.SurfaceEventType.SurfaceAboutToBeDestroyed
                ):
                    self.release_context()
        return super().event(event)

    def showEvent(self, event):
        super().showEvent(event)
        if not self._closed:
            if self._context_active:
                self._Iren.Enable()
            self.request_render()

    def hideEvent(self, event):
        self._Timer.stop()
        self._Iren.Disable()
        super().hideEvent(event)

    def resizeEvent(self, event):
        if not self._closed:
            super().resizeEvent(event)

    def CreateTimer(self, obj, event):
        if not self._closed and self._context_active and self.isVisible():
            super().CreateTimer(obj, event)

    def TimerEvent(self):
        if not self._closed and self._context_active and self.isVisible():
            super().TimerEvent()

    def ShowCursor(self):
        if not self._closed:
            super().ShowCursor()

    def Finalize(self):
        """Permanently shut down; parent destruction may call this again."""
        if self._closed:
            return
        self._closed = True
        self._Timer.stop()
        self._Iren.Disable()
        self.release_context()
        self.setUpdatesEnabled(False)

    def closeEvent(self, event):
        self.Finalize()
        QtWidgets.QWidget.closeEvent(self, event)
