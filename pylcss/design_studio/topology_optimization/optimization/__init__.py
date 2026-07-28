# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.
"""Numerical optimization orchestration and pyMOTO integration."""

from __future__ import annotations

from pylcss.design_studio._lazy_imports import load_attribute, public_names

__all__ = [
    "TopologyOptVoxelProblem",
    "TopologyOptVoxelResult",
    "TopologyOptVoxelSolver",
    "optimality_criteria_update",
    "projected_gradient_update",
    "volume_budget_from_masks",
    "level_set_heaviside",
    "restore_level_set_volume",
    "reaction_diffusion_level_set_update",
]

_LAZY_EXPORTS = {
    "TopologyOptVoxelProblem": (
        "pylcss.design_studio.topology_optimization.optimization.problem",
        "TopologyOptVoxelProblem",
    ),
    "TopologyOptVoxelResult": (
        "pylcss.design_studio.topology_optimization.optimization.results",
        "TopologyOptVoxelResult",
    ),
    "TopologyOptVoxelSolver": (
        "pylcss.design_studio.topology_optimization.optimization.voxel_solver",
        "TopologyOptVoxelSolver",
    ),
    "optimality_criteria_update": (
        "pylcss.design_studio.topology_optimization.optimization.update_algorithms",
        "optimality_criteria_update",
    ),
    "projected_gradient_update": (
        "pylcss.design_studio.topology_optimization.optimization.update_algorithms",
        "projected_gradient_update",
    ),
    "volume_budget_from_masks": (
        "pylcss.design_studio.topology_optimization.optimization.update_algorithms",
        "volume_budget_from_masks",
    ),
    "level_set_heaviside": (
        "pylcss.design_studio.topology_optimization.optimization.level_set",
        "level_set_heaviside",
    ),
    "restore_level_set_volume": (
        "pylcss.design_studio.topology_optimization.optimization.level_set",
        "restore_level_set_volume",
    ),
    "reaction_diffusion_level_set_update": (
        "pylcss.design_studio.topology_optimization.optimization.level_set",
        "reaction_diffusion_level_set_update",
    ),
}


def __getattr__(name: str) -> object:
    """Load public optimization components only when requested."""
    return load_attribute(name, _LAZY_EXPORTS, globals())


def __dir__() -> list[str]:
    """Expose lazy public names to interactive and documentation tools."""
    return public_names(_LAZY_EXPORTS, globals())
