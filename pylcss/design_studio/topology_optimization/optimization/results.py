# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Topology solver result and internal load-case state models."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .pymoto_runtime import PyMotoDomain


@dataclass
class TopologyOptVoxelResult:
    density: np.ndarray
    design_density: np.ndarray | None = None
    compliance_history: list[float] = field(default_factory=list)
    objective_history: list[float] = field(default_factory=list)
    change_history: list[float] = field(default_factory=list)
    thermal_compliance_history: list[float] = field(default_factory=list)
    stress_history: list[float] = field(default_factory=list)  # sigma_pn per iteration
    case_compliances: dict[str, float] = field(default_factory=dict)
    case_max_displacements: dict[str, float] = field(default_factory=dict)
    thermal_case_compliances: dict[str, float] = field(default_factory=dict)
    thermal_case_max_temperature_rises: dict[str, float] = field(
        default_factory=dict
    )
    solve_time_s: float = 0.0
    n_iter: int = 0
    converged: bool = False
    message: str = ""
    active_target_volfrac: float = 0.0
    min_source_volfrac: float = 0.0
    passive_source_volfrac: float = 0.0
    optimizer_used: str = ""
    formulation_used: str = "Density (SIMP)"
    # Surface recovery reads these, not `density`.
    #
    # `density` is the physical field that drove the FE solve. With the
    # three-field projection completed it is essentially binary (measured on a
    # 48x24x12 cantilever: 96% of voxels outside the 0.2-0.8 band), so its
    # 0.5-isosurface carries no sub-voxel information and marching cubes
    # reproduces the voxel terraces as visible stair steps and ripples.
    #
    # `recovery_density` is the *filtered* density before projection. The
    # physical cone filter spreads every interface over its prescribed radius,
    # what lets an isosurface land at sub-voxel accuracy and come out smooth.
    # `recovery_cutoff` is the level on that field describing the *same
    # boundary* as the projected design. The projection is a strictly
    # increasing map, so that level is the exact inverse image of the density
    # cutoff, not a volume-matched search: see
    # `voxel_solver._projection_matched_level`.
    #
    # `projection_beta`/`projection_eta` are the final projection parameters, or
    # None when the network had no projection. They are recorded because the
    # density cutoff is a post-processing control the user can change without
    # re-solving, and the matching filtered level has to move with it.
    recovery_density: np.ndarray | None = None
    recovery_cutoff: float = 0.5
    projection_beta: float | None = None
    projection_eta: float | None = None
    beta_history: list[float] = field(default_factory=list)
    projection_eta_history: list[float] = field(default_factory=list)
    # SIMP exponent per iteration. A run that continues p from 1 solves a
    # different (convex, at p=1) problem early on, so the compliance history is
    # only comparable within a segment of constant p.
    penal_history: list[float] = field(default_factory=list)
    topology_convergence: dict[str, object] | None = None
    progressive_resolution: dict[str, object] | None = None


@dataclass
class _StructuralCase:
    name: str
    weight: float
    force: np.ndarray
    boundary_dofs: np.ndarray
    joint_stiffness: object | None = None


@dataclass
class _ThermalCase:
    name: str
    weight: float
    heat: np.ndarray


def _density_grid_from_state(
    x: np.ndarray,
    domain: PyMotoDomain,
) -> np.ndarray:
    """Map pyMOTO's flat element numbering to density[ix, iy, iz]."""
    return np.asarray(x, dtype=float)[domain.elements]
