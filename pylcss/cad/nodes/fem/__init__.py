# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.
"""pylcss.cad.nodes.fem — FEM simulation nodes package.

Re-exports every public symbol so that all existing imports of the form
    ``from pylcss.cad.nodes.fem import SolverNode``
continue to work unchanged after the monolithic fem.py was split into
this package.
"""

# Helpers & constants (also used by crash package)
from pylcss.cad.nodes.fem._helpers import (
    lam_lame,
    MATERIAL_DATABASE,
    tr,
    suppress_output,
    OCCGeometry,
    build_filter_matrix,
    sensitivity_filter,
    density_filter_3d,
    density_filter_chainrule,
    heaviside_projection,
    mma_update,
    shape_recovery,
    _find_matching_boundary_facets,
    _assemble_traction_force,
    _assemble_pressure_force,
)

# Node classes
from pylcss.cad.nodes.fem.materials          import MaterialNode
from pylcss.cad.nodes.fem.mesh               import MeshNode
from pylcss.cad.nodes.fem.boundary_conditions import (
    ConstraintNode, LoadNode, PressureLoadNode,
)
from pylcss.cad.nodes.fem.solver              import SolverNode
from pylcss.cad.nodes.fem.topology_optimization import TopologyOptimizationNode
from pylcss.cad.nodes.fem.remesh              import RemeshNode
from pylcss.cad.nodes.fem.size_optimization   import SizeOptimizationNode
from pylcss.cad.nodes.fem.shape_optimization  import ShapeOptimizationNode

__all__ = [
    # helpers / constants
    'lam_lame', 'MATERIAL_DATABASE', 'tr', 'suppress_output', 'OCCGeometry',
    'build_filter_matrix', 'sensitivity_filter', 'density_filter_3d',
    'density_filter_chainrule', 'heaviside_projection', 'mma_update',
    'shape_recovery', '_find_matching_boundary_facets',
    '_assemble_traction_force', '_assemble_pressure_force',
    # nodes
    'MaterialNode', 'MeshNode',
    'ConstraintNode', 'LoadNode', 'PressureLoadNode',
    'SolverNode', 'TopologyOptimizationNode',
    'RemeshNode', 'SizeOptimizationNode', 'ShapeOptimizationNode',
]
