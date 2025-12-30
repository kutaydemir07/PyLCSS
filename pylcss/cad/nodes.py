# Copyright (c) 2025 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""
CadQuery Node definitions for the visual CAD editor.
Refactored into modular components.
"""

from pylcss.cad.core.base_node import CadQueryNode, is_numeric, is_shape, resolve_numeric_input, resolve_shape_input
from pylcss.cad.core.registry import NODE_REGISTRY, register_node

# Aliases for backward compatibility
_resolve_numeric_input = resolve_numeric_input
_resolve_shape_input = resolve_shape_input

from pylcss.cad.nodes_impl.primitives import BoxNode, CylinderNode, SphereNode
from pylcss.cad.nodes_impl.operations import (
    ExtrudeNode, PocketNode, FilletNode, SelectFaceNode, 
    CutExtrudeNode, BooleanNode, RevolveNode, CylinderCutNode,
    ChamferNode, ShellNode
)
from pylcss.cad.nodes_impl.io import ExportStepNode, ExportStlNode
from pylcss.cad.nodes_impl.values import NumberNode, VariableNode
from pylcss.cad.nodes_impl.simulation import (
    MaterialNode, MeshNode, ConstraintNode, LoadNode, SolverNode, TopologyOptimizationNode
)

# Re-export for backward compatibility
__all__ = [
    "NODE_REGISTRY", "register_node",
    "CadQueryNode",
    "BoxNode", "CylinderNode", "SphereNode",
    "ExtrudeNode", "PocketNode", "FilletNode",
    "SelectFaceNode", "CutExtrudeNode", "BooleanNode", "RevolveNode", "CylinderCutNode",
    "ChamferNode", "ShellNode",
    "ExportStepNode", "ExportStlNode",
    "NumberNode", "VariableNode",
    "MaterialNode", "MeshNode", "ConstraintNode", "LoadNode", "SolverNode", "TopologyOptimizationNode"
]

