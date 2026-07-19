# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Design Studio: node-based CAD and engineering simulation components.

Package structure:
    core/                   shared node contracts
    nodes/                  CAD, values, assembly, analysis, and import/export
    fem/                    meshing, materials, loads, and CalculiX studies
    crash/                  impact setup and OpenRadioss studies
    topology_optimization/  voxel SIMP optimization and shape recovery
    freecad_bridge/         FreeCAD document synchronization
    engine.py               dependency-aware graph execution
    runtime.py              headless/API project execution
"""

__version__ = "1.0.0"
__author__ = "Kutay Demir"

from pylcss.design_studio.node_library import NODE_CLASS_MAPPING, NODE_NAME_MAPPING

__all__ = ["NODE_CLASS_MAPPING", "NODE_NAME_MAPPING"]
