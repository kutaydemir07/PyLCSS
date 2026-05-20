# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""
CAD Nodes — code-first authoring + the FEA / crash / optimisation surface.

PyLCSS is code-first: parametric geometry is authored in a
:class:`CadQueryCodeNode` (one readable CadQuery script per part / assembly),
or imported via STEP / STL.  The hand-placed primitive / sketch / 3-D-op /
transform / pattern nodes have been removed.

Active node modules:
    nodes/
    ├── code_part.py    # CadQueryCodeNode — primary authoring node
    ├── analysis.py     # MassPropertiesNode, BoundingBoxNode
    ├── assembly.py     # AssemblyNode (combine multiple shapes)
    ├── values.py       # NumberNode, VariableNode
    ├── io.py           # ExportStepNode, ExportStlNode
    ├── advanced.py     # ImportStep/Stl + MathExpression/MeasureDistance/SurfaceArea
    ├── modeling.py     # SelectFaceNode + InteractiveSelectFaceNode

Simulation packages live beside nodes:
    design_studio/fem/                   # FEA simulation package
    design_studio/crash/                 # Crash simulation package
    design_studio/topology_optimization/ # Topology optimization package
"""

# Core base classes
from pylcss.design_studio.core.base_node import (
    CadQueryNode, is_numeric, is_shape,
    resolve_numeric_input, resolve_shape_input,
)
from pylcss.design_studio.core.registry import NODE_REGISTRY, register_node

# Geometry — code-first.
from pylcss.design_studio.nodes.code_part import CadQueryCodeNode

# Geometry — interactive (FreeCAD GUI subprocess + BREP round-trip).
from pylcss.design_studio.nodes.freecad_part import FreeCadPartNode

# Face / surface selection — needed for boundary-condition wiring.
from pylcss.design_studio.nodes.modeling import (
    SelectFaceNode, InteractiveSelectFaceNode,
)

# Assembly aggregator.
from pylcss.design_studio.nodes.assembly import AssemblyNode

# Analysis utilities.
from pylcss.design_studio.nodes.analysis import MassPropertiesNode, BoundingBoxNode

# FEM / simulation.
from pylcss.design_studio.fem import (
    MaterialNode, MeshNode, ConstraintNode, LoadNode, PressureLoadNode,
    SolverNode, TopologyOptVoxelNode,
    RemeshNode,
)

# Crash / impact.
from pylcss.design_studio.crash import (
    CrashMaterialNode, ImpactConditionNode, CrashSolverNode, RunRadiossDeckNode,
)

# IO + parameter scalars + advanced (import / math / measurement).
from pylcss.design_studio.nodes.io import ExportStepNode, ExportStlNode
from pylcss.design_studio.nodes.values import NumberNode, VariableNode
from pylcss.design_studio.nodes.advanced import (
    ImportStepNode, ImportStlNode,
    MathExpressionNode, MeasureDistanceNode, SurfaceAreaNode,
)

__all__ = [
    # Core
    "CadQueryNode", "NODE_REGISTRY", "register_node",
    "is_numeric", "is_shape", "resolve_numeric_input", "resolve_shape_input",

    # Code-first geometry
    "CadQueryCodeNode",

    # Interactive geometry (FreeCAD)
    "FreeCadPartNode",

    # Selection + assembly
    "SelectFaceNode", "InteractiveSelectFaceNode",
    "AssemblyNode",

    # Analysis
    "MassPropertiesNode", "BoundingBoxNode",
    "MathExpressionNode", "MeasureDistanceNode", "SurfaceAreaNode",

    # FEM
    "MaterialNode", "MeshNode",
    "ConstraintNode", "LoadNode", "PressureLoadNode",
    "SolverNode", "TopologyOptVoxelNode",
    "RemeshNode",

    # Crash / impact
    "CrashMaterialNode", "ImpactConditionNode", "CrashSolverNode",
    "RunRadiossDeckNode",

    # IO + parameter scalars + geometry import
    "ImportStepNode", "ImportStlNode",
    "ExportStepNode", "ExportStlNode",
    "NumberNode", "VariableNode",
]
