# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.
"""pylcss.design_studio.fem — FEM simulation nodes package."""

from pylcss.design_studio.fem._helpers import (
    MATERIAL_DATABASE,
    suppress_output,
    OCCGeometry,
)

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
    'MATERIAL_DATABASE', 'suppress_output', 'OCCGeometry',
    'MaterialNode', 'MeshNode',
    'ConstraintNode', 'LoadNode', 'PressureLoadNode',
    'SolverNode',
    'TopologyOptVoxelNode', 'TopologyOptVoxelSolver',
    'TopologyOptVoxelProblem', 'VoxelBC',
    'RemeshNode',
]
