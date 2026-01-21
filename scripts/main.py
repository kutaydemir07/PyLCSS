# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""
# Main entry point for the PyLCSS application.

This script serves as a compatibility wrapper that calls the main function
from the pylcss package. For new installations, use 'pylcss' command directly.
"""

import os
import sys

# Suppress Qt DPI awareness warning on Windows
# Must be set before any Qt imports
os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
os.environ["QT_ENABLE_HIGHDPI_SCALING"] = "1"
os.environ["QT_SCALE_FACTOR_ROUNDING_POLICY"] = "PassThrough"

# Prevent Windows DPI virtualization from interfering
if sys.platform == "win32":
    try:
        import ctypes
        # Set DPI awareness before Qt initializes
        # 2 = PROCESS_PER_MONITOR_DPI_AWARE_V2 (best for multi-monitor)
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except (AttributeError, OSError):
        try:
            # Fallback for older Windows
            ctypes.windll.user32.SetProcessDPIAware()
        except (AttributeError, OSError):
            pass

# Add the parent directory to Python path so we can import pylcss
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from pylcss.main import main

if __name__ == "__main__":
    main()