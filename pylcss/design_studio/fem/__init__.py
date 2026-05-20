# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.
"""pylcss.design_studio.fem — FEM simulation nodes package.

Re-exports every public symbol so that all existing imports of the form
    ``from pylcss.design_studio.fem import SolverNode``
continue to work unchanged after the monolithic fem.py was split into
this package.
"""

# Helpers & constants (also used by crash package)
from pylcss.design_studio.fem._helpers import (
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
from pylcss.design_studio.fem.materials          import MaterialNode
from pylcss.design_studio.fem.mesh               import MeshNode
from pylcss.design_studio.fem.boundary_conditions import (
    ConstraintNode, LoadNode, PressureLoadNode,
)
from pylcss.design_studio.fem.solver              import SolverNode
from pylcss.design_studio.topology_optimization import (
    TopologyOptVoxelNode,
    TopologyOptVoxelSolver,
    TopologyOptVoxelProblem,
    VoxelBC,
)
from pylcss.design_studio.fem.remesh              import RemeshNode

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
    'SolverNode', 'TopologyOptVoxelNode',
    'RemeshNode',
]
