# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""
Main entry point for the PyLCSS application.

This module provides the main function that initializes the Qt application,
applies necessary patches, and launches the main application window.
"""

import os
import sys
import shutil
import logging
import faulthandler
import numpy as np

# Install a fault handler as early as possible.  A native (C/C++) crash -- e.g.
# a VTK / OpenCASCADE / solver access violation -- cannot be caught as a Python
# exception and otherwise just shows the Windows "Fehler in Anwendung" dialog
# with no diagnosable trace.  faulthandler dumps the Python stack that led into
# the native call to pylcss_crash.log (and stderr) when the process faults.
try:
    _crash_log_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "pylcss_crash.log"
    )
    _crash_log_file = open(_crash_log_path, "a", encoding="utf-8", buffering=1)
    faulthandler.enable(file=_crash_log_file, all_threads=True)
except Exception:
    # Never let crash-logging setup itself prevent startup.
    try:
        faulthandler.enable()
    except Exception:
        pass

# Suppress Qt DPI awareness warning on Windows
# Must be done before any Qt imports
if sys.platform == 'win32':
    # Suppress Qt's DPI warning messages
    os.environ['QT_LOGGING_RULES'] = 'qt.qpa.window=false'
    try:
        import ctypes
        # Set DPI awareness using the newer Windows 8.1+ API
        ctypes.windll.shcore.SetProcessDpiAwareness(2)  # PROCESS_PER_MONITOR_DPI_AWARE
    except Exception:
        try:
            # Fallback to older API
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass  # Ignore if it fails

# Patch for NumPy 2.0 compatibility
if not hasattr(np, 'float_'):
    np.float_ = np.float64

from typing import NoReturn

# Set environment variables for PySide6
os.environ['QT_API'] = 'pyside6'
# Suppress Qt DPI awareness warning on Windows
os.environ['QT_AUTO_SCREEN_SCALE_FACTOR'] = '1'

from PySide6 import QtWidgets

# Initialize logging FIRST
from pylcss.config import setup_logging, TEMP_MODELS_DIR
setup_logging(level=logging.INFO)
logger = logging.getLogger(__name__)

# Apply patches and import main window
from pylcss.user_interface.common import qt_patches
from pylcss.user_interface.main_application_window import MainWindow


def cleanup_temp_models():
    """
    Removes the temp_models directory and its contents.
    """
    temp_dir = TEMP_MODELS_DIR
    
    if os.path.exists(temp_dir):
        try:
            shutil.rmtree(temp_dir)
            logger.info(f"Cleaned up temporary models in {temp_dir}")
        except Exception as e:
            logger.warning(f"Failed to clean up temporary models: {e}")


def main() -> NoReturn:
    """
    Main entry point function for PyLCSS application.

    Initializes the Qt application, creates and shows the main window,
    and starts the event loop. This function never returns as it enters
    the Qt event loop.

    Returns:
        NoReturn: This function never returns due to Qt event loop
    """
    app: QtWidgets.QApplication = QtWidgets.QApplication(sys.argv)
    window: MainWindow = MainWindow()
    window.showMaximized()
    
    exit_code = app.exec()
    cleanup_temp_models()
    sys.exit(exit_code)




