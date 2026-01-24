# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""
CadQuery Visual Editor - Node-based parametric CAD with full GUI control.

Package Structure:
    cad/
    ├── core/           # Base classes (CadQueryNode, registry)
    ├── nodes/          # Consolidated node exports (NEW)
    ├── nodes_impl/     # Node implementations by category
    ├── gui/            # GUI components (NEW)
    ├── viewer.py       # VTK 3D viewer
    ├── engine.py       # Graph execution engine
    └── professional_gui.py  # Main application window
"""

__version__ = "1.0.0"
__author__ = "Kutay Demir"

# Convenience imports for common usage
from pylcss.cad.node_library import NODE_CLASS_MAPPING, NODE_NAME_MAPPING
