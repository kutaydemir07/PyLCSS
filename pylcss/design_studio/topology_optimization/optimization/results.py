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
    stress_history: list[float] = field(default_factory=list)  # σ_pn per iteration
    n_iter: int = 0
    converged: bool = False
    message: str = ""
    active_target_volfrac: float = 0.0
    min_source_volfrac: float = 0.0
    passive_source_volfrac: float = 0.0
    optimizer_used: str = ""
    formulation_used: str = "Density (SIMP)"
    level_set_field: np.ndarray | None = None


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
