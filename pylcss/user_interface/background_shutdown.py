"""Close immediately, then give background workers a bounded time to stop."""

from __future__ import annotations

import logging
import os
import subprocess
import time

from PySide6 import QtCore, QtWidgets

from pylcss.process_utils import headless_subprocess_kwargs


logger = logging.getLogger(__name__)
_closing_windows = []
_stopping_threads = []
_detached_widget_threads = []
_widget_cleanup_timer = None


def _running_threads(window):
    """Include parentless workers stored on widgets as well as Qt children."""
    threads = {}
    for obj in (window, *window.findChildren(QtCore.QObject)):
        for value in (obj, *vars(obj).values()):
            if isinstance(value, QtCore.QThread):
                try:
                    if value.isRunning():
                        threads[id(value)] = value
                except RuntimeError:
                    # A finished worker's Python attribute may outlive Qt's
                    # deleteLater() of the underlying object.
                    continue
    return list(threads.values())


def _retain_and_stop(window, threads):
    # The workspace chooser normally deletes a closed window. Retain its graph
    # and workers until the event loop has returned and cleanup is finished.
    window.setAttribute(QtCore.Qt.WA_DeleteOnClose, False)
    if window not in _closing_windows:
        _closing_windows.append(window)
    _stop_threads(threads)


def _stop_threads(threads):
    for thread in threads:
        if thread in _stopping_threads:
            continue
        _stopping_threads.append(thread)
        # Completion/error dialogs must not interrupt an explicit exit.
        thread.blockSignals(True)
        thread.requestInterruption()
        thread.quit()
        try:
            if hasattr(thread, "stop_flag"):
                thread.stop_flag = True
            for name in ("cancel", "stop"):
                stop = getattr(thread, name, None)
                if callable(stop):
                    stop()
                    break
        except Exception:
            logger.debug("Background worker cancellation failed", exc_info=True)


def stop_widget_workers(widget):
    """Cancel workers before a replaceable widget destroys its Qt children.

    Keep detached QThreads alive until they finish. Application shutdown's
    bounded cleanup handles any native process still stopping at exit.
    """
    threads = _running_threads(widget)
    for thread in threads:
        thread.setParent(None)
        if thread not in _detached_widget_threads:
            _detached_widget_threads.append(thread)
    _stop_threads(threads)
    if threads:
        global _widget_cleanup_timer
        if _widget_cleanup_timer is None:
            _widget_cleanup_timer = QtCore.QTimer(QtCore.QCoreApplication.instance())
            _widget_cleanup_timer.setInterval(100)
            _widget_cleanup_timer.timeout.connect(_release_stopped_widget_workers)
        _widget_cleanup_timer.start()


def _release_stopped_widget_workers():
    # Signals are blocked during close, including QThread.finished. Reap only
    # after the native thread actually ends, without waiting on the GUI thread.
    for thread in list(_detached_widget_threads):
        if thread.isRunning():
            continue
        _detached_widget_threads.remove(thread)
        if thread in _stopping_threads:
            _stopping_threads.remove(thread)
        thread.deleteLater()
    if not _detached_widget_threads and _widget_cleanup_timer is not None:
        _widget_cleanup_timer.stop()


def prepare_application_quit():
    """Retain all active workspaces on any application exit path."""
    for window in QtWidgets.QApplication.topLevelWidgets():
        threads = _running_threads(window)
        if threads:
            _retain_and_stop(window, threads)


def prepare_window_close(window):
    """Accept closing during work and exit without returning to the chooser."""
    threads = _running_threads(window)
    if threads:
        _retain_and_stop(window, threads)
        QtWidgets.QApplication.instance().quit()


def finish_background_shutdown(exit_code=0, timeout_seconds=1.0):
    """Run after app.exec(), with no visible window or unbounded Qt waits."""
    deadline = time.monotonic() + timeout_seconds
    for thread in _stopping_threads:
        remaining_ms = max(0, int((deadline - time.monotonic()) * 1000))
        thread.wait(remaining_ms)
    if not any(thread.isRunning() for thread in _stopping_threads):
        return

    # Native meshing/solver calls may not reach a cancellation checkpoint. Do
    # not destroy their live QThreads or let them hold the application open.
    # On Windows also end solver/CAD descendants, including venv launchers.
    if os.name == "nt":
        try:
            subprocess.run(
                ["taskkill", "/PID", str(os.getpid()), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=2.0,
                check=False,
                **headless_subprocess_kwargs(),
            )
        except (OSError, subprocess.SubprocessError):
            pass
    os._exit(exit_code)
