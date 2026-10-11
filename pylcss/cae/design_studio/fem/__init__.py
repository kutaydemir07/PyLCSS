# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Finite-element meshing, boundary-condition, and solver nodes.

Topology names remain available here for backward compatibility. New code
should import them from :mod:`pylcss.cae.design_studio.topology_optimization`.
"""

from __future__ import annotations

from pylcss.cae.design_studio._lazy_imports import load_attribute, public_names

__all__ = [
    "ConstraintNode",
    "FEAComponentNode",
    "LoadNode",
    "MATERIAL_DATABASE",
    "MaterialNode",
    "MeshNode",
    "ModalSolverNode",
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
        "pylcss.cae.design_studio.fem.boundary_conditions",
        "ConstraintNode",
    ),
    "FEAComponentNode": (
        "pylcss.cae.design_studio.fem.components",
        "FEAComponentNode",
    ),
    "LoadNode": (
        "pylcss.cae.design_studio.fem.boundary_conditions",
        "LoadNode",
    ),
    "MATERIAL_DATABASE": (
        "pylcss.cae.design_studio.fem._helpers",
        "MATERIAL_DATABASE",
    ),
    "MaterialNode": ("pylcss.cae.design_studio.fem.materials", "MaterialNode"),
    "MeshNode": ("pylcss.cae.design_studio.fem.mesh", "MeshNode"),
    "ModalSolverNode": (
        "pylcss.cae.design_studio.fem.modal_solver",
        "ModalSolverNode",
    ),
    "OCCGeometry": ("pylcss.cae.design_studio.fem._helpers", "OCCGeometry"),
    "PressureLoadNode": (
        "pylcss.cae.design_studio.fem.boundary_conditions",
        "PressureLoadNode",
    ),
    "SolverNode": ("pylcss.cae.design_studio.fem.solver", "SolverNode"),
    "TopologyOptVoxelNode": (
        "pylcss.cae.design_studio.topology_optimization",
        "TopologyOptVoxelNode",
    ),
    "TopologyOptVoxelProblem": (
        "pylcss.cae.design_studio.topology_optimization",
        "TopologyOptVoxelProblem",
    ),
    "TopologyOptVoxelSolver": (
        "pylcss.cae.design_studio.topology_optimization",
        "TopologyOptVoxelSolver",
    ),
    "VoxelBC": (
        "pylcss.cae.design_studio.topology_optimization",
        "VoxelBC",
    ),
    "suppress_output": (
        "pylcss.cae.design_studio.fem._helpers",
        "suppress_output",
    ),
}


def __getattr__(name: str) -> object:
    return load_attribute(name, _LAZY_EXPORTS, globals())


def __dir__() -> list[str]:
    return public_names(_LAZY_EXPORTS, globals())
