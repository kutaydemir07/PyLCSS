# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Validated inputs for structured voxel topology studies."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..models.study import ManufacturingConstraints, ThermalBC, VoxelBC

INDUSTRIAL_STRESS_RELAXATION_Q = 0.5
INDUSTRIAL_STRESS_PNORM_P = 8.0
INDUSTRIAL_HEAVISIDE_ENABLED = True
INDUSTRIAL_HEAVISIDE_BETA_INIT = 1.0
INDUSTRIAL_HEAVISIDE_BETA_MAX = 16.0
INDUSTRIAL_HEAVISIDE_BETA_STEP_ITERS = 30
INDUSTRIAL_HEAVISIDE_ETA = 0.5
MMA_FAMILY = {"MMA", "GCMMA"}


@dataclass
class TopologyOptVoxelProblem:
    """All parameters needed to solve a 3-D voxel topology optimisation."""

    nelx: int = 30
    nely: int = 20
    nelz: int = 10
    E0: float = 1.0
    Emin: float = 1e-9
    nu: float = 0.3
    penal: float = 3.0
    volfrac: float = 0.5
    rmin: float = 1.5
    unitx: float = 1.0
    unity: float = 1.0
    unitz: float = 1.0
    optimizer: str = "AUTO"  # 'AUTO' | 'OC' | 'MMA' | 'GCMMA' | 'PGD'
    formulation: str = "density"  # density | level_set
    max_iter: int = 80
    tol: float = 0.01
    # Early-stop is judged on the objective: stop once the relative compliance
    # change stays below `tol` for `patience` consecutive iterations.
    patience: int = 5
    bc: VoxelBC = field(default_factory=VoxelBC)
    mc: ManufacturingConstraints = field(default_factory=ManufacturingConstraints)
    # Optional source CAD/mesh volume mask, shape (nelx, nely, nelz).
    # True means designable material exists there; False is clamped to void.
    design_domain: Optional[np.ndarray] = None
    # Explicit user-authored non-design volumes on the same structured grid.
    # Solid wins over the source-domain mask; void is applied last.
    passive_solid_mask: Optional[np.ndarray] = None
    passive_void_mask: Optional[np.ndarray] = None
    # Compliance minimization at a material budget, or true mass/volume
    # minimization subject to the stress constraint.
    objective_mode: str = "compliance"  # compliance | minimum_mass
    # Structural, steady-state thermal, or a normalized combination of both.
    physics_mode: str = "structural"
    thermal_bc: ThermalBC = field(default_factory=ThermalBC)
    thermal_conductivity: float = 1.0
    thermal_conductivity_min: float = 1e-6
    thermal_penal: float = 3.0
    structural_weight: float = 1.0
    thermal_weight: float = 1.0
    # Load envelopes can minimize a weighted sum or a smooth worst case.
    load_aggregation: str = "weighted_sum"
    load_pnorm_p: float = 8.0
    # Projected-gradient step size.
    projected_gradient_step: float = 0.15
    # Phase 3 — stress constraint (P-norm aggregated von Mises ≤ yield).
    # When enabled the optimiser is forced to GCMMA unless an MMA-family
    # method is already selected (OC cannot handle a second constraint beyond
    # the volume budget). All load cases are aggregated into a single PNorm.
    stress_constraint_enabled: bool = False
    yield_stress: float = 1.0
    # Internal qp stress relaxation and P-norm aggregation policy. The user
    # enters the allowable stress; the solver owns these numerical defaults.
    stress_penalty: float = INDUSTRIAL_STRESS_RELAXATION_Q
    stress_pnorm_p: float = INDUSTRIAL_STRESS_PNORM_P

    # Three-field SIMP projection is an internal solver default. It drives
    # intermediate densities toward 0/1 without exposing continuation knobs.
    heaviside_enabled: bool = INDUSTRIAL_HEAVISIDE_ENABLED
    heaviside_beta_init: float = INDUSTRIAL_HEAVISIDE_BETA_INIT
    heaviside_beta_max: float = INDUSTRIAL_HEAVISIDE_BETA_MAX
    heaviside_beta_step_iters: int = INDUSTRIAL_HEAVISIDE_BETA_STEP_ITERS
    heaviside_eta: float = INDUSTRIAL_HEAVISIDE_ETA

    def __post_init__(self) -> None:
        """Reject physically invalid or memory-dangerous studies up front."""
        self.nelx, self.nely, self.nelz = int(self.nelx), int(self.nely), int(self.nelz)
        if min(self.nelx, self.nely, self.nelz) < 1:
            raise ValueError("Topology grid dimensions must each be at least 1 voxel.")
        n_voxels = self.nelx * self.nely * self.nelz
        if n_voxels > 500_000:
            raise ValueError(
                f"Topology grid has {n_voxels:,} voxels; the supported limit is "
                "500,000. Reduce expert grid dimensions or use Guided mode."
            )

        finite_fields = {
            "E0": self.E0,
            "Emin": self.Emin,
            "nu": self.nu,
            "penal": self.penal,
            "volfrac": self.volfrac,
            "rmin": self.rmin,
            "unitx": self.unitx,
            "unity": self.unity,
            "unitz": self.unitz,
            "tol": self.tol,
            "yield_stress": self.yield_stress,
            "stress_penalty": self.stress_penalty,
            "stress_pnorm_p": self.stress_pnorm_p,
            "heaviside_beta_init": self.heaviside_beta_init,
            "heaviside_beta_max": self.heaviside_beta_max,
            "heaviside_eta": self.heaviside_eta,
            "thermal_conductivity": self.thermal_conductivity,
            "thermal_conductivity_min": self.thermal_conductivity_min,
            "thermal_penal": self.thermal_penal,
            "structural_weight": self.structural_weight,
            "thermal_weight": self.thermal_weight,
            "load_pnorm_p": self.load_pnorm_p,
            "projected_gradient_step": self.projected_gradient_step,
        }
        for name, raw in finite_fields.items():
            value = float(raw)
            if not np.isfinite(value):
                raise ValueError(f"Topology parameter {name} must be finite.")
            setattr(self, name, value)
        if self.E0 <= 0.0 or self.Emin <= 0.0 or self.Emin > self.E0:
            raise ValueError("Topology stiffnesses require 0 < Emin <= E0.")
        if not (-1.0 < self.nu < 0.5):
            raise ValueError(
                "Topology Poisson ratio must be between -1 and 0.5 (exclusive)."
            )
        if self.penal < 1.0:
            raise ValueError("SIMP penalization must be at least 1.")
        if not (0.0 < self.volfrac <= 1.0):
            raise ValueError("Topology volume fraction must be in (0, 1].")
        if self.rmin <= 0.0 or min(self.unitx, self.unity, self.unitz) <= 0.0:
            raise ValueError(
                "Filter radius and voxel dimensions must be greater than zero."
            )
        self.max_iter = int(self.max_iter)
        self.patience = int(self.patience)
        if self.max_iter < 1 or self.patience < 1 or self.tol <= 0.0:
            raise ValueError(
                "Iterations, convergence patience, and tolerance must be positive."
            )
        self.optimizer = str(self.optimizer).upper()
        if self.optimizer not in {"AUTO", "OC", "MMA", "GCMMA", "PGD"}:
            raise ValueError("Topology optimizer must be Auto, OC, MMA, GCMMA, or PGD.")
        self.formulation = (
            str(self.formulation).strip().lower().replace("-", "_").replace(" ", "_")
        )
        if self.formulation in {"simp", "density_simp", "density_(simp)"}:
            self.formulation = "density"
        if self.formulation in {
            "levelset",
            "reaction_diffusion_level_set",
            "level_set_(reaction_diffusion)",
        }:
            self.formulation = "level_set"
        if self.formulation not in {"density", "level_set"}:
            raise ValueError("Topology formulation must be Density or Level Set.")
        if self.formulation == "level_set":
            if min(self.nelx, self.nely, self.nelz) < 8:
                raise ValueError(
                    "Level-set topology optimization needs at least eight voxels "
                    "through every grid direction."
                )
            if (
                str(self.objective_mode).strip().lower() == "minimum_mass"
                or self.stress_constraint_enabled
            ):
                raise ValueError(
                    "Level-set currently supports compliance with a volume "
                    "constraint; use Density (SIMP) with GCMMA for stress or "
                    "minimum-mass studies."
                )
            has_level_set_incompatible_projection = (
                str(self.mc.symmetry or "none").lower() != "none"
                or str(self.mc.extrusion or "none").lower() != "none"
                or str(self.mc.overhang_build_axis or "none").lower() != "none"
                or float(self.mc.max_member_size_voxels or 0.0) > 0.0
                or int(self.mc.pattern_repeat or 1) > 1
            )
            if has_level_set_incompatible_projection:
                raise ValueError(
                    "Level-set does not support density-field manufacturing "
                    "projections. Use Density (SIMP) for this study."
                )
        self.objective_mode = str(self.objective_mode).strip().lower()
        if self.objective_mode not in {"compliance", "minimum_mass"}:
            raise ValueError("Topology objective must be compliance or minimum_mass.")
        self.physics_mode = str(self.physics_mode).strip().lower().replace("-", "_")
        if self.physics_mode not in {"structural", "thermal", "thermo_mechanical"}:
            raise ValueError(
                "Topology physics must be structural, thermal, or thermo_mechanical."
            )
        self.load_aggregation = (
            str(self.load_aggregation)
            .strip()
            .lower()
            .replace(" ", "_")
            .replace("-", "_")
        )
        if self.load_aggregation not in {"weighted_sum", "worst_case"}:
            raise ValueError("Load aggregation must be weighted_sum or worst_case.")
        if self.objective_mode == "minimum_mass" and not self.stress_constraint_enabled:
            raise ValueError(
                "Minimum-mass topology optimization requires the stress constraint."
            )
        if self.stress_constraint_enabled and self.physics_mode == "thermal":
            raise ValueError(
                "A structural stress constraint cannot be used in thermal-only mode."
            )
        if (
            self.thermal_conductivity <= 0.0
            or self.thermal_conductivity_min <= 0.0
            or self.thermal_conductivity_min > self.thermal_conductivity
            or self.thermal_penal < 1.0
        ):
            raise ValueError(
                "Thermal interpolation requires 0 < k_min <= k and penalization >= 1."
            )
        if self.structural_weight < 0.0 or self.thermal_weight < 0.0:
            raise ValueError("Physics weights must be non-negative.")
        if self.physics_mode == "thermo_mechanical" and (
            self.structural_weight + self.thermal_weight <= 0.0
        ):
            raise ValueError(
                "Thermo-mechanical optimization needs a positive physics weight."
            )
        if self.load_pnorm_p <= 1.0:
            raise ValueError("Worst-case load P-norm exponent must be greater than 1.")
        if not 0.0 < self.projected_gradient_step <= 1.0:
            raise ValueError("Projected-gradient step must be in (0, 1].")
        if self.stress_constraint_enabled and self.yield_stress <= 0.0:
            raise ValueError(
                "Stress-constrained TopOpt needs a positive allowable stress."
            )
        if not (0.0 <= self.stress_penalty <= 1.0):
            raise ValueError("Topology stress relaxation must be between 0 and 1.")
        if self.stress_pnorm_p <= 2.0:
            raise ValueError("Topology stress P-norm exponent must be greater than 2.")
        self.heaviside_beta_step_iters = int(self.heaviside_beta_step_iters)
        if (
            self.heaviside_beta_init <= 0.0
            or self.heaviside_beta_max < self.heaviside_beta_init
            or self.heaviside_beta_step_iters < 1
            or not 0.0 < self.heaviside_eta < 1.0
        ):
            raise ValueError(
                "Heaviside projection requires positive beta, beta_max >= beta_init, "
                "a positive continuation interval, and eta in (0, 1)."
            )
        if self.design_domain is not None:
            domain = np.asarray(self.design_domain, dtype=bool)
            expected = (self.nelx, self.nely, self.nelz)
            if domain.shape != expected or not np.any(domain):
                raise ValueError(
                    f"Design-domain mask must be non-empty with shape {expected}."
                )
            self.design_domain = domain
