# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""
CAD Nodes — interactive parametric authoring and engineering analysis.

GUI-native primitive, feature, transform, and pattern nodes are the default
authoring path. A CadQuery code node remains available for exceptional custom
geometry, but ordinary examples and engineering studies require no scripts.

Active node modules:
    nodes/
    ├── parametric.py   # native primitives, features, transforms, patterns
    ├── code_part.py    # optional expert CadQuery node
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

# GUI-native parametric geometry.
from pylcss.design_studio.nodes.parametric import (
    BooleanNode,
    BoxNode,
    CylindricalShellNode,
    CylinderNode,
    FilletNode,
    LinearPatternNode,
    ThroughHoleNode,
    TransformNode,
    TubeNode,
)

# Optional expert geometry.
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
    "CadQueryNode",
    "is_numeric", "is_shape", "resolve_numeric_input", "resolve_shape_input",

    # GUI-native geometry
    "BoxNode", "CylinderNode", "TubeNode", "CylindricalShellNode", "BooleanNode",
    "ThroughHoleNode", "FilletNode", "TransformNode", "LinearPatternNode",

    # Optional expert geometry
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
