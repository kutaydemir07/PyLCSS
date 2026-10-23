# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Validated inputs for structured voxel topology studies."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..configuration.length_scale import (
    MAX_THRESHOLD_OFFSET,
    coarsest_active_edge,
)
from ..models.study import VoxelBC
from ..configuration.voxel_meshing import DEFAULT_MAX_VOXEL_ELEMENTS

INDUSTRIAL_STRESS_RELAXATION_Q = 0.5
INDUSTRIAL_STRESS_PNORM_P = 8.0
INDUSTRIAL_STRESS_SCALING_DAMPING = 0.5
INDUSTRIAL_HEAVISIDE_ENABLED = True
INDUSTRIAL_HEAVISIDE_BETA_INIT = 1.0
INDUSTRIAL_HEAVISIDE_BETA_MAX = 16.0
INDUSTRIAL_HEAVISIDE_BETA_STEP_ITERS = 30
INDUSTRIAL_HEAVISIDE_ETA = 0.5
# Increase the SIMP exponent gradually so the load path can develop before
# intermediate densities receive the full stiffness penalty.
INDUSTRIAL_PENAL_CONTINUATION = True
INDUSTRIAL_PENAL_INIT = 1.0
INDUSTRIAL_PENAL_STEP = 0.5
# Fraction of the iteration budget the p-continuation is scheduled to finish
# within, leaving the remainder to the projection continuation and the final
# convergence at full penalization.
INDUSTRIAL_PENAL_BUDGET_FRACTION = 0.4
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
    volfrac: float = 0.5  # Upper material budget relative to the source design domain.
    rmin: float = 1.5
    # New studies store the cone-filter radius in physical model units so a
    # mesh refinement does not change the manufacturing requirement. False
    # preserves the meaning of legacy/programmatic problem definitions.
    filter_radius_is_physical: bool = False
    minimum_solid_size: float = 0.0
    minimum_void_size: float = 0.0
    unitx: float = 1.0
    unity: float = 1.0
    unitz: float = 1.0
    optimizer: str = "AUTO"  # 'AUTO' | 'OC' | 'MMA' | 'GCMMA'
    formulation: str = "density"  # density
    max_iter: int = 80
    tol: float = 0.01
    # Early-stop is judged on the objective: stop once the relative compliance
    # change stays below `tol` for `patience` consecutive iterations.
    patience: int = 5
    bc: VoxelBC = field(default_factory=VoxelBC)
    # Optional raw design-density field used to warm-start a refined grid.
    # This is an internal continuation state, not a user-facing parameter.
    initial_density: Optional[np.ndarray] = None
    # Optional source CAD/mesh volume mask, shape (nelx, nely, nelz).
    # True means designable material exists there; False is clamped to void.
    design_domain: Optional[np.ndarray] = None
    # Explicit user-authored non-design volumes on the same structured grid.
    # Solid wins over the source-domain mask; void is applied last.
    passive_solid_mask: Optional[np.ndarray] = None
    passive_void_mask: Optional[np.ndarray] = None
    # Infill recovery also uses this grid description, without a structural
    # solve. Solver-only resource and filter checks do not apply in that case.
    solve_enabled: bool = True
    # Compliance minimization at a material budget, or true mass/volume
    # minimization subject to both the stress constraint and material budget.
    objective_mode: str = "compliance"  # compliance | minimum_mass
    # The backend accepts structural studies only.
    physics_mode: str = "structural"
    # Load envelopes can minimize a weighted sum or a smooth worst case.
    load_aggregation: str = "weighted_sum"
    load_pnorm_p: float = 8.0
    # Stress constraint (P-norm aggregated von Mises <= yield).
    # When enabled the optimiser is forced to GCMMA unless an MMA-family
    # method is already selected (OC cannot handle a second constraint beyond
    # the volume budget). All load cases are aggregated into a single PNorm.
    stress_constraint_enabled: bool = False
    yield_stress: float = 1.0
    # Internal qp stress relaxation and P-norm aggregation policy. The user
    # enters the allowable stress; the solver owns these numerical defaults.
    stress_penalty: float = INDUSTRIAL_STRESS_RELAXATION_Q
    stress_pnorm_p: float = INDUSTRIAL_STRESS_PNORM_P
    # Adaptive constraint scaling corrects the finite-P overestimate toward
    # the current true maximum while retaining a differentiable P-norm
    # sensitivity.  A damping of 0 freezes no correction; 1 freezes the first
    # correction forever.  The mid-range default follows the established
    # adaptive-constraint-scaling practice without exposing a fragile UI knob.
    stress_scaling_damping: float = INDUSTRIAL_STRESS_SCALING_DAMPING

    # Robust (eroded/blueprint/dilated) formulation. This is what makes
    # `minimum_solid_size` and `minimum_void_size` bind: the objective is
    # evaluated on the eroded projection, so a member thinner than the
    # requested size disappears from the analysed design and costs stiffness,
    # while compliance studies hold an adapted budget on the dilated field.
    # Minimum-mass studies constrain the blueprint volume and minimize the
    # dilated mass. Off leaves a single projection at `heaviside_eta`, which
    # regularizes but constrains nothing. `eta_eroded`/`eta_dilated` come from
    # the length-scale policy,
    # which derives them from the requested sizes and the filter radius.
    robust_length_scale: bool = False
    eta_eroded: float = 0.5 + MAX_THRESHOLD_OFFSET
    eta_dilated: float = 0.5 - MAX_THRESHOLD_OFFSET
    # Iterations between rescalings of the dilated volume budget so the
    # blueprint lands on the requested volume fraction.
    dilated_volume_update_iters: int = 10

    # Three-field SIMP projection is an internal solver default. It drives
    # intermediate densities toward 0/1 without exposing continuation knobs.
    heaviside_enabled: bool = INDUSTRIAL_HEAVISIDE_ENABLED
    heaviside_beta_init: float = INDUSTRIAL_HEAVISIDE_BETA_INIT
    heaviside_beta_max: float = INDUSTRIAL_HEAVISIDE_BETA_MAX
    heaviside_beta_step_iters: int = INDUSTRIAL_HEAVISIDE_BETA_STEP_ITERS
    heaviside_eta: float = INDUSTRIAL_HEAVISIDE_ETA

    # SIMP penalization continuation, also an internal solver default. The user
    # states the final penalization; how the solver reaches it is not a knob.
    penal_continuation_enabled: bool = INDUSTRIAL_PENAL_CONTINUATION
    penal_init: float = INDUSTRIAL_PENAL_INIT
    penal_step: float = INDUSTRIAL_PENAL_STEP
    # Share the explicit resource safeguard with the mesh planner. Zero
    # disables the cap; the solver must not impose a second hidden limit.
    max_voxel_elements: int = DEFAULT_MAX_VOXEL_ELEMENTS
    # Library override for comparisons. Desktop studies select automatically.
    linear_solver: str = 'auto'

    def __post_init__(self) -> None:
        """Reject physically invalid or memory-dangerous studies up front."""
        self.nelx, self.nely, self.nelz = int(self.nelx), int(self.nely), int(self.nelz)
        if self.linear_solver not in ('auto', 'assembled', 'matrix_free'):
            raise ValueError('Topology linear solver must be auto, assembled, or matrix_free.')
        if min(self.nelx, self.nely, self.nelz) < 1:
            raise ValueError("Topology grid dimensions must each be at least 1 voxel.")
        n_voxels = self.nelx * self.nely * self.nelz
        limit = float(self.max_voxel_elements)
        if not np.isfinite(limit) or limit < 0 or not limit.is_integer():
            raise ValueError("Maximum voxel elements must be a non-negative integer.")
        self.max_voxel_elements = int(limit)
        if (
            self.solve_enabled
            and self.max_voxel_elements
            and n_voxels > self.max_voxel_elements
        ):
            raise ValueError(
                f"Topology grid has {n_voxels:,} voxels; the configured maximum is "
                f"{self.max_voxel_elements:,}. Increase Maximum Elements or choose a coarser mesh."
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
            "minimum_solid_size": self.minimum_solid_size,
            "minimum_void_size": self.minimum_void_size,
        }
        if self.stress_constraint_enabled:
            finite_fields.update(
                yield_stress=self.yield_stress,
                stress_penalty=self.stress_penalty,
                stress_pnorm_p=self.stress_pnorm_p,
                stress_scaling_damping=self.stress_scaling_damping,
            )
        if self.heaviside_enabled:
            finite_fields.update(
                heaviside_beta_init=self.heaviside_beta_init,
                heaviside_beta_max=self.heaviside_beta_max,
                heaviside_eta=self.heaviside_eta,
            )
        if self.penal_continuation_enabled:
            finite_fields.update(penal_init=self.penal_init, penal_step=self.penal_step)
        if self.robust_length_scale:
            finite_fields.update(
                eta_eroded=self.eta_eroded, eta_dilated=self.eta_dilated
            )
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
        self.filter_radius_is_physical = bool(self.filter_radius_is_physical)
        # A cone narrower than the grid spacing cannot regularize that axis.
        coarsest_edge = coarsest_active_edge((self.unitx, self.unity, self.unitz))
        radius_in_elements = (
            self.rmin / coarsest_edge if self.filter_radius_is_physical else self.rmin
        )
        if self.solve_enabled and radius_in_elements < 1.0:
            raise ValueError(
                f"Filter radius spans only {radius_in_elements:.3g} elements on "
                "the coarsest voxel axis; at least 1 is required for the "
                "density filter to regularize the design. Increase the filter "
                "radius or the minimum member size, or refine the grid."
            )
        self.max_iter = int(self.max_iter)
        self.patience = int(self.patience)
        if self.max_iter < 1 or self.patience < 1 or self.tol <= 0.0:
            raise ValueError(
                "Iterations, convergence patience, and tolerance must be positive."
            )
        self.optimizer = str(self.optimizer).strip().upper()
        if self.optimizer not in {"AUTO", "OC", "MMA", "GCMMA"}:
            raise ValueError("Topology optimizer must be Auto, OC, MMA, or GCMMA.")
        self.formulation = (
            str(self.formulation).strip().lower().replace("-", "_").replace(" ", "_")
        )
        if self.formulation in {"simp", "density_simp", "density_(simp)"}:
            self.formulation = "density"
        if self.formulation != "density":
            raise ValueError("Topology formulation must be Density.")
        self.objective_mode = str(self.objective_mode).strip().lower()
        if self.objective_mode not in {"compliance", "minimum_mass"}:
            raise ValueError("Topology objective must be compliance or minimum_mass.")
        self.physics_mode = str(self.physics_mode).strip().lower().replace("-", "_")
        if self.physics_mode != "structural":
            raise ValueError("Topology optimization supports structural physics only.")
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
        if self.load_aggregation == "worst_case":
            self.load_pnorm_p = float(self.load_pnorm_p)
            if not np.isfinite(self.load_pnorm_p) or self.load_pnorm_p <= 1.0:
                raise ValueError(
                    "Worst-case load P-norm exponent must be finite and greater than 1."
                )
        if self.stress_constraint_enabled:
            if self.yield_stress <= 0.0:
                raise ValueError(
                    "Stress-constrained TopOpt needs a positive allowable stress."
                )
            if not 0.0 <= self.stress_penalty <= 1.0:
                raise ValueError("Topology stress relaxation must be between 0 and 1.")
            if self.stress_pnorm_p <= 2.0:
                raise ValueError(
                    "Topology stress P-norm exponent must be greater than 2."
                )
            if not 0.0 <= self.stress_scaling_damping < 1.0:
                raise ValueError(
                    "Topology stress aggregation damping must be in [0, 1)."
                )
        if self.heaviside_enabled:
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
        self.penal_continuation_enabled = bool(self.penal_continuation_enabled)
        if self.penal_continuation_enabled:
            if self.penal_init < 1.0 or self.penal_init > self.penal:
                raise ValueError(
                    "SIMP continuation must start at an exponent between 1 and the "
                    "study penalization."
                )
            if self.penal_step <= 0.0:
                raise ValueError("SIMP continuation step must be positive.")
        if self.minimum_solid_size < 0.0 or self.minimum_void_size < 0.0:
            raise ValueError("Minimum solid and void sizes cannot be negative.")
        self.robust_length_scale = bool(self.robust_length_scale)
        if self.robust_length_scale:
            if not self.heaviside_enabled:
                raise ValueError(
                    "The robust length-scale formulation requires the projection; "
                    "it is defined by the eroded and dilated projected fields."
                )
            if not 0.0 < self.eta_dilated < self.heaviside_eta < self.eta_eroded < 1.0:
                raise ValueError(
                    "Robust projection thresholds must satisfy "
                    "0 < eta_dilated < eta_blueprint < eta_eroded < 1."
                )
            self.dilated_volume_update_iters = int(self.dilated_volume_update_iters)
            if self.dilated_volume_update_iters < 1:
                raise ValueError("The dilated volume update interval must be positive.")
        if self.design_domain is not None:
            domain = np.asarray(self.design_domain, dtype=bool)
            expected = (self.nelx, self.nely, self.nelz)
            if domain.shape != expected or not np.any(domain):
                raise ValueError(
                    f"Design-domain mask must be non-empty with shape {expected}."
                )
            self.design_domain = domain
        if self.initial_density is not None:
            initial = np.asarray(self.initial_density, dtype=float)
            expected = (self.nelx, self.nely, self.nelz)
            if initial.shape != expected or not np.all(np.isfinite(initial)):
                raise ValueError(
                    f"Initial topology density must be finite with shape {expected}."
                )
            self.initial_density = np.clip(initial, 1e-3, 1.0)
