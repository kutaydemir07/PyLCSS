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
    stress_history: list[float] = field(default_factory=list)  # sigma_pn per iteration
    case_compliances: dict[str, float] = field(default_factory=dict)
    case_max_displacements: dict[str, float] = field(default_factory=dict)
    solve_time_s: float = 0.0
    n_iter: int = 0
    converged: bool = False
    message: str = ""
    active_target_volfrac: float = 0.0
    min_source_volfrac: float = 0.0
    passive_source_volfrac: float = 0.0
    optimizer_used: str = ""
    formulation_used: str = "Density (SIMP)"
    # `density` is the delivered blueprint. Robust studies analyze its eroded
    # counterpart; ordinary studies analyze the blueprint itself.
    # Recover surfaces on the smoother filtered field at the inverse-projected
    # cutoff. Retain projection parameters so a later cutoff change can use
    # the corresponding filtered level without another solve.
    recovery_density: np.ndarray | None = None
    recovery_cutoff: float = 0.5
    projection_beta: float | None = None
    projection_eta: float | None = None
    beta_history: list[float] = field(default_factory=list)
    projection_eta_history: list[float] = field(default_factory=list)
    # Compare compliance within segments of constant SIMP exponent.
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




def _density_grid_from_state(
    x: np.ndarray,
    domain: PyMotoDomain,
) -> np.ndarray:
    """Map pyMOTO's flat element numbering to density[ix, iy, iz]."""
    return np.asarray(x, dtype=float)[domain.elements]
