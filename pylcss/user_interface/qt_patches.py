# Copyright (c) 2025 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""
Qt and NodeGraphQt patches for PyLCSS.

This module contains patches to fix compatibility issues and bugs in
third-party libraries, particularly NodeGraphQt pipe drawing crashes.
"""

import os
os.environ['QT_API'] = 'pyside6'

# --- NEW: Monkey Patch to prevent 'NoneType' crash in Pipe Drawing ---
# This fixes the crash when a port is deleted/renamed while dragging or connecting.
try:
    from NodeGraphQt.qgraphics.pipe import PipeItem
    
    # Store original method
    _original_draw_horizontal = PipeItem._draw_path_horizontal

    def _safe_draw_horizontal(self, start_port, pos1, pos2, path):
        """
        Safe version of _draw_path_horizontal that checks for port validity.

        Prevents crashes when ports are deleted during drawing operations.
        """
        # Check if start_port and its node still exist
        if not start_port or not start_port.node:
            return
        # Proceed with original logic
        _original_draw_horizontal(self, start_port, pos1, pos2, path)

    # Apply patch
    PipeItem._draw_path_horizontal = _safe_draw_horizontal
except (ImportError, AttributeError):
    pass # NodeGraphQt might not be installed yet
# ----------------------------------------------------------------------




from PySide6.QtWidgets import QTableWidgetItem

class NumericTableWidgetItem(QTableWidgetItem):
    """
    A TableWidgetItem that sorts numerically instead of lexicographically.
    Handles floats, negatives, and scientific notation correctly.
    """
    def __lt__(self, other):
        try:
            # Try converting to float for comparison
            # This respects negative values (-10.0 < -5.0)
            return float(self.text()) < float(other.text())
        except ValueError:
            # Fallback to default string sorting if not a number
            return super().__lt__(other)






