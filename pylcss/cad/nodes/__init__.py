# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""
CAD Nodes — code-first authoring + the FEA / crash / optimisation surface.

PyLCSS used to ship a long menu of hand-placed primitive / sketch / 3-D-op /
transform / pattern / hole-wizard nodes (Box, Cylinder, Sketch, Extrude,
Fillet, Boolean, Translate, Linear Pattern, …).  Those are retired in favour
of a single :class:`CadQueryCodeNode` that hosts one readable parametric
CadQuery script.  The source files for the retired nodes still live under
``pylcss/cad/nodes/`` so any in-tree code that historically imported them
keeps working, but they are no longer re-exported here, no longer in
``pylcss.cad.node_library.NODE_CLASS_MAPPING``, and no longer in the toolbar.

Structure that is still part of the active surface:
    nodes/
    ├── code_part.py    # CadQueryCodeNode — primary authoring node
    ├── analysis.py     # MassPropertiesNode, BoundingBoxNode
    ├── assembly.py     # AssemblyNode (combine multiple shapes)
    ├── values.py       # NumberNode, VariableNode (with exposed_name)
    ├── io.py           # ExportStepNode, ExportStlNode
    ├── advanced.py     # ImportStep/Stl + MathExpression/MeasureDistance/SurfaceArea
    ├── modeling.py     # SelectFaceNode + InteractiveSelectFaceNode
    ├── fem/            # FEA simulation sub-package
    └── crash/          # Crash simulation sub-package
"""

# Core base classes
from pylcss.cad.core.base_node import (
    CadQueryNode, is_numeric, is_shape,
    resolve_numeric_input, resolve_shape_input,
)
from pylcss.cad.core.registry import NODE_REGISTRY, register_node

# Geometry — code-first.
from pylcss.cad.nodes.code_part import CadQueryCodeNode

# Face / surface selection — needed for boundary-condition wiring.
from pylcss.cad.nodes.modeling import (
    SelectFaceNode, InteractiveSelectFaceNode,
)

# Assembly aggregator.
from pylcss.cad.nodes.assembly import AssemblyNode

# Analysis utilities.
from pylcss.cad.nodes.analysis import MassPropertiesNode, BoundingBoxNode

# FEM / simulation.
from pylcss.cad.nodes.fem import (
    MaterialNode, MeshNode, ConstraintNode, LoadNode, PressureLoadNode,
    SolverNode, TopologyOptimizationNode,
    RemeshNode, SizeOptimizationNode, ShapeOptimizationNode,
)

# Crash / impact.
from pylcss.cad.nodes.crash import (
    CrashMaterialNode, ImpactConditionNode, CrashSolverNode, RunRadiossDeckNode,
)

# IO + parameter scalars + advanced (import / math / measurement).
from pylcss.cad.nodes.io import ExportStepNode, ExportStlNode
from pylcss.cad.nodes.values import NumberNode, VariableNode
from pylcss.cad.nodes.advanced import (
    ImportStepNode, ImportStlNode,
    MathExpressionNode, MeasureDistanceNode, SurfaceAreaNode,
)

__all__ = [
    # Core
    "CadQueryNode", "NODE_REGISTRY", "register_node",
    "is_numeric", "is_shape", "resolve_numeric_input", "resolve_shape_input",

    # Code-first geometry
    "CadQueryCodeNode",

    # Selection + assembly
    "SelectFaceNode", "InteractiveSelectFaceNode",
    "AssemblyNode",

    # Analysis
    "MassPropertiesNode", "BoundingBoxNode",
    "MathExpressionNode", "MeasureDistanceNode", "SurfaceAreaNode",

    # FEM
    "MaterialNode", "MeshNode",
    "ConstraintNode", "LoadNode", "PressureLoadNode",
    "SolverNode", "TopologyOptimizationNode",
    "RemeshNode", "SizeOptimizationNode", "ShapeOptimizationNode",

    # Crash / impact
    "CrashMaterialNode", "ImpactConditionNode", "CrashSolverNode",
    "RunRadiossDeckNode",

    # IO + parameter scalars + geometry import
    "ImportStepNode", "ImportStlNode",
    "ExportStepNode", "ExportStlNode",
    "NumberNode", "VariableNode",
]
