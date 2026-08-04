# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Finite-element meshing, boundary-condition, and solver nodes.

Topology names remain available here for backward compatibility. New code
should import them from :mod:`pylcss.design_studio.topology_optimization`.
"""

from __future__ import annotations

from pylcss.design_studio._lazy_imports import load_attribute, public_names

__all__ = [
    "ConstraintNode",
    "FEAComponentNode",
    "LoadNode",
    "MATERIAL_DATABASE",
    "MaterialNode",
    "MeshNode",
    "OCCGeometry",
    "PressureLoadNode",
    "SolverNode",
    "TopologyOptVoxelNode",
    "TopologyOptVoxelProblem",
    "TopologyOptVoxelSolver",
    "VoxelBC",
    "suppress_output",
]

_LAZY_EXPORTS = {
    "ConstraintNode": (
        "pylcss.design_studio.fem.boundary_conditions",
        "ConstraintNode",
    ),
    "FEAComponentNode": (
        "pylcss.design_studio.fem.components",
        "FEAComponentNode",
    ),
    "LoadNode": (
        "pylcss.design_studio.fem.boundary_conditions",
        "LoadNode",
    ),
    "MATERIAL_DATABASE": (
        "pylcss.design_studio.fem._helpers",
        "MATERIAL_DATABASE",
    ),
    "MaterialNode": ("pylcss.design_studio.fem.materials", "MaterialNode"),
    "MeshNode": ("pylcss.design_studio.fem.mesh", "MeshNode"),
    "OCCGeometry": ("pylcss.design_studio.fem._helpers", "OCCGeometry"),
    "PressureLoadNode": (
        "pylcss.design_studio.fem.boundary_conditions",
        "PressureLoadNode",
    ),
    "SolverNode": ("pylcss.design_studio.fem.solver", "SolverNode"),
    "TopologyOptVoxelNode": (
        "pylcss.design_studio.topology_optimization",
        "TopologyOptVoxelNode",
    ),
    "TopologyOptVoxelProblem": (
        "pylcss.design_studio.topology_optimization",
        "TopologyOptVoxelProblem",
    ),
    "TopologyOptVoxelSolver": (
        "pylcss.design_studio.topology_optimization",
        "TopologyOptVoxelSolver",
    ),
    "VoxelBC": (
        "pylcss.design_studio.topology_optimization",
        "VoxelBC",
    ),
    "suppress_output": (
        "pylcss.design_studio.fem._helpers",
        "suppress_output",
    ),
}


def __getattr__(name: str) -> object:
    return load_attribute(name, _LAZY_EXPORTS, globals())


def __dir__() -> list[str]:
    return public_names(_LAZY_EXPORTS, globals())
