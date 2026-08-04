# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""CAD authoring, selection, analysis, I/O, and engineering nodes.

This module is a compatibility facade. Implementations stay grouped by
responsibility in the sibling modules and are imported only when requested.
"""

from __future__ import annotations

from pylcss.design_studio._lazy_imports import load_attribute, public_names

__all__ = [
    "AssemblyNode",
    "BooleanNode",
    "BoundingBoxNode",
    "BoxNode",
    "CadQueryCodeNode",
    "CadQueryNode",
    "ConstraintNode",
    "CrashSolverNode",
    "CylinderNode",
    "CylindricalShellNode",
    "ExportStepNode",
    "ExportStlNode",
    "FEAComponentNode",
    "FilletNode",
    "FreeCadPartNode",
    "ImpactConditionNode",
    "ImportStepNode",
    "ImportStlNode",
    "InteractiveSelectFaceNode",
    "LinearPatternNode",
    "LoadNode",
    "MassPropertiesNode",
    "MaterialNode",
    "MathExpressionNode",
    "MeasureDistanceNode",
    "MeshNode",
    "NumberNode",
    "PressureLoadNode",
    "SelectFaceNode",
    "SolverNode",
    "SurfaceAreaNode",
    "ThroughHoleNode",
    "TopologyOptVoxelNode",
    "TransformNode",
    "TubeNode",
    "VariableNode",
    "is_numeric",
    "is_shape",
    "resolve_numeric_input",
    "resolve_shape_input",
]

_BASE = "pylcss.design_studio.core.base_node"
_PARAMETRIC = "pylcss.design_studio.nodes.parametric"
_ADVANCED = "pylcss.design_studio.nodes.advanced"
_FEM = "pylcss.design_studio.fem"
_CRASH = "pylcss.design_studio.crash"

_LAZY_EXPORTS = {
    "AssemblyNode": ("pylcss.design_studio.nodes.assembly", "AssemblyNode"),
    "BooleanNode": (_PARAMETRIC, "BooleanNode"),
    "BoundingBoxNode": (
        "pylcss.design_studio.nodes.analysis",
        "BoundingBoxNode",
    ),
    "BoxNode": (_PARAMETRIC, "BoxNode"),
    "CadQueryCodeNode": (
        "pylcss.design_studio.nodes.code_part",
        "CadQueryCodeNode",
    ),
    "CadQueryNode": (_BASE, "CadQueryNode"),
    "ConstraintNode": (_FEM, "ConstraintNode"),
    "CrashSolverNode": (_CRASH, "CrashSolverNode"),
    "CylinderNode": (_PARAMETRIC, "CylinderNode"),
    "CylindricalShellNode": (_PARAMETRIC, "CylindricalShellNode"),
    "ExportStepNode": ("pylcss.design_studio.nodes.io", "ExportStepNode"),
    "ExportStlNode": ("pylcss.design_studio.nodes.io", "ExportStlNode"),
    "FEAComponentNode": (_FEM, "FEAComponentNode"),
    "FilletNode": (_PARAMETRIC, "FilletNode"),
    "FreeCadPartNode": (
        "pylcss.design_studio.nodes.freecad_part",
        "FreeCadPartNode",
    ),
    "ImpactConditionNode": (_CRASH, "ImpactConditionNode"),
    "ImportStepNode": (_ADVANCED, "ImportStepNode"),
    "ImportStlNode": (_ADVANCED, "ImportStlNode"),
    "InteractiveSelectFaceNode": (
        "pylcss.design_studio.nodes.selection",
        "InteractiveSelectFaceNode",
    ),
    "LinearPatternNode": (_PARAMETRIC, "LinearPatternNode"),
    "LoadNode": (_FEM, "LoadNode"),
    "MassPropertiesNode": (
        "pylcss.design_studio.nodes.analysis",
        "MassPropertiesNode",
    ),
    "MaterialNode": (_FEM, "MaterialNode"),
    "MathExpressionNode": (_ADVANCED, "MathExpressionNode"),
    "MeasureDistanceNode": (_ADVANCED, "MeasureDistanceNode"),
    "MeshNode": (_FEM, "MeshNode"),
    "NumberNode": ("pylcss.design_studio.nodes.values", "NumberNode"),
    "PressureLoadNode": (_FEM, "PressureLoadNode"),
    "SelectFaceNode": (
        "pylcss.design_studio.nodes.selection",
        "SelectFaceNode",
    ),
    "SolverNode": (_FEM, "SolverNode"),
    "SurfaceAreaNode": (_ADVANCED, "SurfaceAreaNode"),
    "ThroughHoleNode": (_PARAMETRIC, "ThroughHoleNode"),
    "TopologyOptVoxelNode": (_FEM, "TopologyOptVoxelNode"),
    "TransformNode": (_PARAMETRIC, "TransformNode"),
    "TubeNode": (_PARAMETRIC, "TubeNode"),
    "VariableNode": ("pylcss.design_studio.nodes.values", "VariableNode"),
    "is_numeric": (_BASE, "is_numeric"),
    "is_shape": (_BASE, "is_shape"),
    "resolve_numeric_input": (_BASE, "resolve_numeric_input"),
    "resolve_shape_input": (_BASE, "resolve_shape_input"),
}


def __getattr__(name: str) -> object:
    return load_attribute(name, _LAZY_EXPORTS, globals())


def __dir__() -> list[str]:
    return public_names(_LAZY_EXPORTS, globals())
