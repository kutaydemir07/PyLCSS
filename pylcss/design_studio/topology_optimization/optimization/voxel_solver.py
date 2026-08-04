# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Voxel topology solver orchestration and iteration loop."""

from __future__ import annotations

import logging
import io
from contextlib import redirect_stdout
from typing import Any, Callable, Optional

import numpy as np

from .assembly import _SolverAssemblyMixin
from .lattice_material import (
    cubic_basis_element_matrices,
    make_cell_material_module,
)
from .problem import (
    INDUSTRIAL_PENAL_BUDGET_FRACTION,
    MMA_FAMILY,
    TopologyOptVoxelProblem,
)
from .pymoto_modules import (
    _make_concat_module,
    _make_heaviside_module,
    _make_linear_grid_projection_module,
    _make_matrix_sum_module,
    _make_max_member_module,
    _make_passive_clamp_module,
    _make_pattern_repeat_module,
    _make_pull_out_module,
    _make_simp_module,
    _make_sparse_to_csc_module,
    _make_volume_constraint_module,
    _make_vm_module,
)
from .pymoto_runtime import import_pymoto
from .results import TopologyOptVoxelResult, _density_grid_from_state
from .update_algorithms import (
    optimality_criteria_update,
    restore_active_volume,
    volume_budget_from_masks,
)

logger = logging.getLogger(__name__)

# Private aliases preserve the established extension and test API.
_oc_update = optimality_criteria_update
_restore_active_volume = restore_active_volume
_volume_budget_from_masks = volume_budget_from_masks


# Sparse LU fill-in becomes expensive surprisingly early for a 3-D elasticity
# grid. Benchmarks on the bundled pyMOTO backend show geometric-multigrid CG is
# already 3.2x faster at 16x10x4 and 10.8x faster at 32x20x8, with objective
# agreement better than 4e-9 relative. Keep direct LU only for tiny grids where
# solver startup dominates, and use the scalable path for normal studies.
DIRECT_SOLVE_VOXEL_LIMIT = 256

# Relative residual for the iterative path. Compliance sensitivities are
# formed from the solution vector, so this has to stay well below the
# optimizer's own convergence tolerance to avoid polluting the gradients.
#
# 1e-6 is three orders below the tightest topology tolerance a study can ask
# for and leaves the answer unchanged. Measured on a 48x24x24 cantilever,
# 12 MMA iterations, mean CG iterations per solve and final compliance:
#
#     tol 1e-8   42 CG its   4.074 s/iteration   C = 4.726294e+00
#     tol 1e-6   30 CG its   3.034 s/iteration   C = 4.726294e+00
#     tol 1e-5   24 CG its   2.892 s/iteration   C = 4.726297e+00
#
# So 1e-8 buys no digits of compliance and costs 34% more time per iteration.
# 1e-5 starts to show in the seventh significant figure, which is close enough
# to the gradient noise floor that it is not worth the remaining 5%.
ITERATIVE_SOLVE_TOLERANCE = 1e-6
ITERATIVE_SOLVE_MAX_ITER = 5_000
JOINT_SOLVE_TOLERANCE = 1e-6
JOINT_SOLVE_MAX_ITER = 2_000

# Depth of the geometric-multigrid hierarchy and the damped-Jacobi smoothing
# steps per level. Recursing the V-cycle shrinks the coarse factorization, but
# it also replaces an exact coarse solve with an increasingly poor one, and
# pyMOTO's damped-Jacobi smoother plus trilinear interpolation does not hold up
# against the 1e9 stiffness contrast SIMP produces. Every extra level roughly
# doubles the CG count, and past two levels the cheaper coarse solve stops
# paying for it.
#
# Mean CG iterations per solve and s per design iteration, with the final
# compliance agreeing to 7 significant figures across every variant:
#
#                    112x36x8 (MBB)     48x24x24        64x32x32
#     levels     CG its   s/it     CG its  s/it     CG its   s/it
#       1           7.7   1.515       5.7  1.163       5.3   2.686
#       2          19.9   1.517      14.8  1.108      13.2   2.522
#       3          19.9   1.544      29.0  1.126      26.5   2.767
#       5          19.9   1.567      29.0  1.126      56.7   2.906
#
# So two levels is the optimum on every shape measured, and the previous cap of
# 5 cost 15% on the 64x32x32 grid. One level converges in a third of the CG
# iterations but pays for it in the larger coarse factorization.
#
# At that cap, two smoothing steps beat three -- the extra sweep buys fewer CG
# iterations than its matvecs cost (s/it, three smooth vs two):
#
#     112x36x8   1.585 -> 1.449     48x24x24  1.127 -> 1.102
#     64x32x32   2.545 -> 2.378
#
# Rebuilding the Galerkin coarse operator less often was also tried and is a
# clear loss: refreshing it every 2nd design iteration doubled the CG count on
# the MBB grid (20.1 -> 41.4) and every 5th quintupled it (94.4). The SIMP
# operator changes too much per step for a stale coarse correction.
MULTIGRID_MAX_LEVELS = 2
MULTIGRID_SMOOTH_STEPS = 2
# Coarsening stops while a level still has this many elements per axis, so the
# interpolation stencil stays meaningful.
MULTIGRID_MIN_COARSE_ELEMENTS = 2

# Test/benchmark override: a callable taking the VoxelDomain and returning a
# pyMOTO LinearSolver (or None for pyMOTO's automatic choice).
_LINEAR_SOLVER_FACTORY: Optional[Callable[[Any], Any]] = None

_OVERHANG_DIRECTIONS = {
    "+x": (1.0, 0.0, 0.0),
    "-x": (-1.0, 0.0, 0.0),
    "+y": (0.0, 1.0, 0.0),
    "-y": (0.0, -1.0, 0.0),
    "+z": (0.0, 0.0, 1.0),
    "-z": (0.0, 0.0, -1.0),
}


def _smooth_heaviside_values(
    values: np.ndarray,
    *,
    beta: float,
    eta: float,
) -> np.ndarray:
    """Evaluate the density projection without executing the FE network."""
    rho = np.asarray(values, dtype=float)
    beta_value = float(beta)
    eta_value = float(eta)
    if beta_value < 1e-6:
        return rho.copy()
    tanh_be = np.tanh(beta_value * eta_value)
    tanh_b1e = np.tanh(beta_value * (1.0 - eta_value))
    denominator = tanh_be + tanh_b1e
    return (
        tanh_be + np.tanh(beta_value * (rho - eta_value))
    ) / max(float(denominator), 1e-12)


def _projection_matched_level(
    cutoff: float,
    *,
    beta: float,
    eta: float,
) -> float:
    """Invert the density projection: the filtered level matching ``cutoff``.

    Surface recovery runs on the filtered field, not the projected one, because
    only the filtered field varies smoothly enough across an interface to place
    an isosurface at sub-voxel accuracy. That choice is sound, but it needs the
    *corresponding* level, and the correspondence is exact rather than
    something to be searched for: the projection is a strictly increasing map
    ``rho_bar = H(rho_tilde; beta, eta)``, so the ``rho_bar = cutoff`` boundary
    is precisely the ``rho_tilde = H^-1(cutoff)`` boundary. This is the closed
    form of that inverse.

    It replaces a bisection that matched ``count(filtered >= level)`` against
    ``sum(projected)`` -- a voxel count against a mass integral, two different
    quantities that agree only for a perfectly binary design. That search also
    ignored ``beta`` and ``eta`` entirely, so it returned essentially 0.5
    whatever the projection was doing, and was therefore worst exactly when the
    volume-preserving threshold had moved furthest from 0.5. Measured on a
    filtered field at volfrac 0.4, recovered against analyzed material:

        beta   1 (eta 0.833)   +83.1%      beta   8 (eta 0.510)   +8.1%
        beta   2 (eta 0.593)   +57.6%      beta  16 (eta 0.503)   +1.7%
        beta   4 (eta 0.531)   +26.2%

    This inverse reproduces the projected design cell for cell -- zero voxels
    differ at every beta tested -- because it is the same level set, not an
    approximation of it.

    With no projection in the network (``beta`` at zero) the filtered field is
    the physical field and the cutoff passes through unchanged.
    """
    level = float(cutoff)
    beta_value = float(beta)
    if not np.isfinite(beta_value) or beta_value < 1e-6:
        return float(np.clip(level, 0.02, 0.98))
    eta_value = float(eta)
    tanh_be = np.tanh(beta_value * eta_value)
    tanh_b1e = np.tanh(beta_value * (1.0 - eta_value))
    inner = float(np.clip(level * (tanh_be + tanh_b1e) - tanh_be, -1.0 + 1e-15, 1.0 - 1e-15))
    # Stay clear of the exact 0/1 endpoints: a level there has no gradient to
    # interpolate against and marching cubes degenerates back to voxel facets.
    return float(np.clip(eta_value + np.arctanh(inner) / beta_value, 0.02, 0.98))


def _volume_preserving_projection_eta(
    filtered_density: np.ndarray,
    *,
    beta: float,
    active_mask: np.ndarray,
    passive_density: np.ndarray,
    source_mask: np.ndarray,
    target_source_sum: float,
) -> float:
    """Choose the projection threshold that preserves the material budget.

    A fixed ``eta=0.5`` loses volume whenever beta is sharpened on an
    asymmetric filtered field. Compliance and conduction minimization should
    consume their full upper material budget, so solve for the threshold that
    keeps the projected *source-domain* sum unchanged. Passive interfaces stay
    exact and are excluded from the threshold solve.
    """
    filtered = np.asarray(filtered_density, dtype=float).reshape(-1)
    active = np.asarray(active_mask, dtype=bool).reshape(-1)
    passive = np.asarray(passive_density, dtype=float).reshape(-1)
    source = np.asarray(source_mask, dtype=bool).reshape(-1)
    active_source = active & source
    passive_source = (~active) & source
    if not np.any(active_source):
        return 0.5

    target_active_sum = float(target_source_sum) - float(
        np.sum(passive[passive_source])
    )
    target_active_sum = float(
        np.clip(target_active_sum, 0.0, float(np.sum(active_source)))
    )
    active_values = filtered[active_source]

    def projected_sum(eta: float) -> float:
        return float(
            np.sum(
                _smooth_heaviside_values(
                    active_values,
                    beta=beta,
                    eta=eta,
                )
            )
        )

    lower = 1e-6
    upper = 1.0 - 1e-6
    if target_active_sum >= projected_sum(lower):
        return lower
    if target_active_sum <= projected_sum(upper):
        return upper
    for _ in range(40):
        midpoint = 0.5 * (lower + upper)
        if projected_sum(midpoint) > target_active_sum:
            lower = midpoint
        else:
            upper = midpoint
    return 0.5 * (lower + upper)


def _multigrid_hierarchy(pym: Any, domain: Any, level: int = 1) -> Any:
    """Build a recursive geometric-multigrid V-cycle rooted at ``domain``.

    Each level halves every axis, and pyMOTO asserts that any domain a
    ``GeometricMultigrid`` is built on is itself divisible by two. Recursion
    therefore continues only while the *next* grid is still even and above the
    minimum coarse size -- testing the current grid instead builds a level on
    an odd coarse domain, which pyMOTO rejects, and the caller's ``except``
    then silently drops the whole study back to the direct solver. The
    innermost level keeps pyMOTO's automatic direct solver, which is what a
    proper V-cycle wants at its coarsest point.
    """
    sub_dims = (
        int(domain.nelx) // 2,
        int(domain.nely) // 2,
        int(domain.nelz) // 2,
    )
    inner_level = None
    if (
        level < MULTIGRID_MAX_LEVELS
        and all(n % 2 == 0 for n in sub_dims)
        and min(sub_dims) >= MULTIGRID_MIN_COARSE_ELEMENTS
    ):
        sub_domain = pym.VoxelDomain(
            *sub_dims,
            domain.unitx * 2,
            domain.unity * 2,
            domain.unitz * 2,
        )
        inner_level = _multigrid_hierarchy(pym, sub_domain, level + 1)
    return pym.solvers.GeometricMultigrid(
        domain,
        inner_level=inner_level,
        smooth_steps=MULTIGRID_SMOOTH_STEPS,
    )


def _linear_solver_for_domain(pym: Any, domain: Any) -> Any:
    """Choose the linear solver for one SIMP system.

    Returning ``None`` leaves pyMOTO's automatic direct solver in place. Above
    the tiny-grid cutoff where sparse-factorization fill-in starts to dominate, a
    geometric-multigrid preconditioned CG is used instead: its memory is
    linear in the number of voxels, which is what allows a grid fine enough to
    resolve a lattice cell or a recovered surface.

    Multigrid requires even grid dimensions. When they are odd the direct
    solver is kept rather than degrading to a Jacobi preconditioner —
    measured on a 48x24x24 cantilever, Jacobi-CG was 2.1x *slower* than the
    direct solve because the 1e9 SIMP stiffness contrast leaves the system
    badly conditioned.
    """
    if _LINEAR_SOLVER_FACTORY is not None:
        return _LINEAR_SOLVER_FACTORY(domain)

    n_voxels = int(domain.nelx) * int(domain.nely) * int(domain.nelz)
    if n_voxels <= DIRECT_SOLVE_VOXEL_LIMIT:
        return None

    dims = (int(domain.nelx), int(domain.nely), int(domain.nelz))
    if any(n % 2 for n in dims) or min(dims) < 2:
        logger.info(
            "Topology grid %s cannot use multigrid; keeping the direct solver.",
            dims,
        )
        return None
    try:
        preconditioner = _multigrid_hierarchy(pym, domain)
    except Exception:
        logger.debug(
            "Geometric multigrid unavailable for this domain; "
            "keeping the direct solver.",
            exc_info=True,
        )
        return None
    return pym.solvers.CG(
        preconditioner=preconditioner,
        tol=ITERATIVE_SOLVE_TOLERANCE,
        maxit=ITERATIVE_SOLVE_MAX_ITER,
    )


def _linear_solver_for_case(
    pym: Any,
    domain: Any,
    joint_stiffness: Any,
) -> Any:
    """Relax the iterative target appropriately for joint-coupled cases.

    The geometric multigrid hierarchy represents only the regular voxel
    operator. Long-range pin/connector springs are present in the assembled
    matrix but absent from that preconditioner. Chasing the plain-voxel
    residual target can therefore exhaust thousands of iterations, while
    sparse direct factorization has excessive fill-in for the 3-D system.
    A 1e-6 relative residual remains far below the topology convergence
    tolerance and is the practical engineering target for this connector
    surrogate.
    """
    solver = _linear_solver_for_domain(pym, domain)
    if joint_stiffness is not None:
        try:
            if int(getattr(joint_stiffness, "nnz", 0)) > 0:
                if solver is not None:
                    solver.tol = JOINT_SOLVE_TOLERANCE
                    solver.maxit = JOINT_SOLVE_MAX_ITER
                return solver
        except (TypeError, ValueError):
            return solver
    return solver


class TopologyOptVoxelSolver(_SolverAssemblyMixin):
    """3-D density/SIMP and reaction-diffusion level-set optimizer."""

    def __init__(self, problem: TopologyOptVoxelProblem) -> None:
        self.problem = problem
        self.stop_requested = False

    def stop(self) -> None:
        self.stop_requested = True

    def solve(
        self,
        callback: Optional[Callable[[int, float, float, np.ndarray], None]] = None,
    ) -> TopologyOptVoxelResult:
        """Compatibility wrapper matching the rest of PyLCSS' solver API."""
        return self.run(callback=callback)

    def run(
        self,
        callback: Optional[Callable[[int, float, float, np.ndarray], None]] = None,
    ) -> TopologyOptVoxelResult:
        """
        Run the optimisation loop.

        callback(iteration, compliance, change, density_3d) is called after
        every iteration so the UI can update live.  density_3d has shape
        (nelx, nely, nelz).
        """
        pym = import_pymoto()

        p = self.problem
        uses_structural = p.physics_mode in {"structural", "thermo_mechanical"}
        uses_thermal = p.physics_mode in {"thermal", "thermo_mechanical"}

        # ── two-scale cell law ────────────────────────────────────────────
        # When the study builds a lattice from a family whose periodic cell has
        # been homogenized, the macro operator uses that cell's anisotropic
        # tensor instead of an isotropic power law. Resolved before the network
        # is built because it decides which assembly path the network takes.
        cell_law = p.homogenized_cell_law()
        if cell_law is not None:
            logger.info(
                "Two-scale lattice study: %s cell law, %d homogenized samples, "
                "Zener %.2f at rho=0.3.",
                cell_law.cell_type,
                cell_law.relative_density.size,
                float(cell_law.zener_ratio(np.asarray(0.3))),
            )
        elif p.lattice_cell_type and p.stress_constraint_enabled:
            logger.warning(
                "Lattice study %r runs on the isotropic continuum surrogate "
                "because a stress constraint is active; the homogenized cell "
                "law cannot be paired with isotropic stress recovery.",
                p.lattice_cell_type,
            )
        self.cell_law = cell_law

        # ── domain ────────────────────────────────────────────────────────
        domain = pym.VoxelDomain(
            p.nelx,
            p.nely,
            p.nelz,
            unitx=p.unitx,
            unity=p.unity,
            unitz=p.unitz,
        )
        ndof = domain.nnodes * domain.dim  # 3 DOFs per node

        # ── supports, loads, passive regions ──────────────────────────────
        boundary_dofs = self._assemble_supports(domain, p.bc)
        load_cases = (
            self._assemble_load_cases(
                domain, p.bc, ndof, base_boundary_dofs=boundary_dofs
            )
            if uses_structural
            else []
        )
        thermal_boundary_nodes = (
            self._assemble_thermal_supports(domain, p.thermal_bc)
            if uses_thermal
            else np.array([], dtype=int)
        )
        thermal_cases = (
            self._assemble_thermal_load_cases(
                domain, p.thermal_bc, thermal_boundary_nodes
            )
            if uses_thermal
            else []
        )
        if uses_structural and not load_cases:
            raise ValueError(
                "Structural topology optimization needs a non-zero load case."
            )
        if uses_structural and any(case.boundary_dofs.size == 0 for case in load_cases):
            raise ValueError(
                "Every structural load/pose case needs at least one support."
            )
        if uses_thermal and thermal_boundary_nodes.size == 0:
            raise ValueError(
                "Thermal topology optimization needs a temperature-sink boundary."
            )
        if uses_thermal and not thermal_cases:
            raise ValueError(
                "Thermal topology optimization needs a non-zero heat-input case."
            )
        active_mask, passive_density = self._assemble_passive_masks(domain, p.bc)
        if not np.any(active_mask):
            raise ValueError(
                "Topology study has no designable voxels after source, passive, "
                "and joint-attachment regions are applied."
            )

        source_mask_flat = np.ones(domain.nel, dtype=bool)
        if p.design_domain is not None:
            try:
                source_grid = np.asarray(p.design_domain, dtype=bool)
                if source_grid.shape == (p.nelx, p.nely, p.nelz):
                    source_mask_flat[:] = False
                    source_mask_flat[domain.elements] = source_grid
            except Exception:
                source_mask_flat[:] = True

        volume_budget = _volume_budget_from_masks(
            p.volfrac,
            active_mask,
            passive_density,
            source_mask_flat,
        )
        active_volfrac = float(volume_budget["active_volfrac"])
        total_vol_target = float(volume_budget["flat_total_target"]) / float(domain.nel)
        if volume_budget["active_budget_exhausted"] or volume_budget["target_was_clamped"]:
            logger.warning(
                "Topology volume target %.3f is infeasible with passive regions; "
                "minimum source-domain volume is %.3f. Proceeding with active density set to minimum.",
                float(p.volfrac),
                float(volume_budget["min_source_volfrac"]),
            )

        # ── initial state — design = volfrac, passive = clamp value ───────
        x0 = np.ones(domain.nel) * active_volfrac
        if p.initial_density is not None:
            initial_grid = np.asarray(p.initial_density, dtype=float)
            expected_shape = (p.nelx, p.nely, p.nelz)
            if initial_grid.shape != expected_shape:
                raise ValueError(
                    "Initial topology density must have shape "
                    f"{expected_shape}, got {initial_grid.shape}."
                )
            x0[domain.elements] = np.clip(initial_grid, 1e-3, 1.0)
        x0[~active_mask] = passive_density[~active_mask]
        if p.initial_density is not None:
            # Interpolation across resolution levels is not exactly volume
            # preserving. Restore the active-material budget before the first
            # response so the refined stage starts feasible.
            x0 = _restore_active_volume(
                x0,
                active_mask,
                active_volfrac,
                passive_density,
            )

        if not load_cases and not thermal_cases:
            logger.warning(
                "TopologyOptVoxelNode: no load cases produced a non-zero force "
                "vector — returning a trivial (uniform) result."
            )
            density_3d = _density_grid_from_state(x0, domain)
            result = TopologyOptVoxelResult(
                density=density_3d.copy(),
                design_density=density_3d.copy(),
                compliance_history=[0.0],
                change_history=[0.0],
                n_iter=0,
                converged=True,
                active_target_volfrac=active_volfrac,
                min_source_volfrac=float(volume_budget["min_source_volfrac"]),
                passive_source_volfrac=(
                    float(volume_budget["passive_source_sum"])
                    / max(float(volume_budget["source_count"]), 1.0)
                ),
                message=(
                    "No non-zero load cases; returned the initial uniform "
                    "density field."
                ),
                formulation_used="Density (SIMP)",
            )
            if callback is not None:
                callback(0, 0.0, 0.0, density_3d.copy())
            return result

        sx = pym.Signal("x", state=x0)
        mc = p.mc
        max_member_radius_voxels = float(mc.max_member_size_voxels or 0.0)
        if float(mc.max_member_size_physical or 0.0) > 0.0:
            max_member_radius_voxels = (
                0.5
                * float(mc.max_member_size_physical)
                / max(float(p.unitx), float(p.unity), float(p.unitz))
            )
        has_manufacturing_mappings = (
            (mc.symmetry or "none").lower() != "none"
            or (mc.extrusion or "none").lower() != "none"
            or (mc.overhang_build_axis or "none").lower() != "none"
            or (mc.pull_out_axis or "none").lower() != "none"
            or max_member_radius_voxels > 0.0
            or int(mc.pattern_repeat or 1) > 1
        )
        minimum_mass_objective = str(p.objective_mode).lower() == "minimum_mass"
        optimizer_choice = p.optimizer.upper()
        if optimizer_choice == "AUTO":
            if minimum_mass_objective or p.stress_constraint_enabled:
                optimizer_choice = "GCMMA"
            elif p.heaviside_enabled or has_manufacturing_mappings:
                optimizer_choice = "MMA"
            else:
                optimizer_choice = "OC"
        if (
            minimum_mass_objective
            or p.stress_constraint_enabled
            or has_manufacturing_mappings
        ) and optimizer_choice not in MMA_FAMILY:
            logger.info(
                "The selected objective/constraints require full mapped-volume "
                "sensitivities; forcing an MMA-family optimizer (was '%s').",
                optimizer_choice,
            )
            optimizer_choice = (
                "GCMMA"
                if minimum_mass_objective or p.stress_constraint_enabled
                else "MMA"
            )
        use_heaviside = bool(p.heaviside_enabled and optimizer_choice in MMA_FAMILY)
        if p.heaviside_enabled and not use_heaviside:
            logger.info(
                "Smooth Heaviside projection is disabled for %s because OC "
                "does not solve its physical-volume constraint.",
                optimizer_choice,
            )
        use_robust = bool(p.robust_length_scale and use_heaviside)
        if p.robust_length_scale and not use_robust:
            logger.info(
                "The robust length-scale formulation needs the projection, "
                "which is unavailable here; the requested minimum member size "
                "is not enforced in this run."
            )
        elif use_robust:
            logger.info(
                "Robust length scale: eta_dilated %.3f / blueprint %.3f / "
                "eta_eroded %.3f at filter radius %.4g -> minimum member "
                "%.4g, minimum void %.4g model units.",
                float(p.eta_dilated),
                float(p.heaviside_eta),
                float(p.eta_eroded),
                float(p.rmin),
                float(p.minimum_solid_size),
                float(p.minimum_void_size),
            )

        # ── SIMP penalization continuation ────────────────────────────────
        # The exponent enters at `penal_init` — 1 by default, where compliance
        # minimization is convex and the solution is unique — and is raised to
        # the study penalization in `penal_step` increments. Starting at the
        # target exponent instead lets the first few descent directions commit
        # the topology to whichever local minimum they happen to fall into,
        # which is precisely what continuation exists to avoid.
        #
        # Disabled for a two-scale study: a homogenized cell law interpolates a
        # measured stiffness tensor rather than a power of the density, so
        # there is no exponent to continue.
        penal_init = float(p.penal_init)
        penal_target = float(p.penal)
        thermal_penal_target = float(p.thermal_penal)
        widest_penal_range = max(
            (penal_target - penal_init) if uses_structural else 0.0,
            (thermal_penal_target - penal_init) if uses_thermal else 0.0,
        )
        use_penal_continuation = bool(
            p.penal_continuation_enabled
            and cell_law is None
            and widest_penal_range > 1e-9
        )
        penal_steps = (
            max(1, int(np.ceil(widest_penal_range / max(float(p.penal_step), 1e-9))))
            if use_penal_continuation
            else 0
        )
        # Fraction of the way from `penal_init` to each target. One shared
        # parameter, so structural and thermal reach full penalization on the
        # same iteration even when their exponents differ.
        penal_progress = 0.0 if use_penal_continuation else 1.0

        def _continued_penal(target: float) -> float:
            return penal_init + penal_progress * (float(target) - penal_init)

        def _physical_density_grid(signal_state: np.ndarray) -> np.ndarray:
            physical = np.asarray(signal_state, dtype=float).copy()
            physical[~active_mask] = passive_density[~active_mask]
            return _density_grid_from_state(physical, domain)

        # ── pyMOTO network ────────────────────────────────────────────────
        # Density formulation: raw density -> exact-adjoint manufacturing maps
        # -> physical cone filter -> passive clamp -> optional projection -> SIMP.
        heaviside_module_ref: Any = None
        # Every SIMP interpolation in the network paired with the exponent it
        # is continuing toward — the structural penalization, and the thermal
        # one when the study is coupled. Advancing them together keeps the two
        # physics describing the same material model at every iteration.
        simp_modules: list[tuple[Any, float]] = []
        # Every projection module in the network, nominal first. beta continuation
        # has to advance all of them together or the branches stop describing the
        # same design at different thresholds.
        projection_modules: list[Any] = []
        volume_constraint_module_ref: Any = None
        with pym.Network() as net:
            PassiveClamp = _make_passive_clamp_module()
            smapped = sx
            if (
                (mc.symmetry or "none").lower() != "none"
                or (mc.extrusion or "none").lower() != "none"
            ):
                LinearMap = _make_linear_grid_projection_module(
                    domain,
                    symmetry=str(mc.symmetry or "none"),
                    extrusion=str(mc.extrusion or "none"),
                )
                smapped = LinearMap()(smapped)
            if int(mc.pattern_repeat or 1) > 1:
                PatternMap = _make_pattern_repeat_module(
                    domain,
                    int(mc.pattern_repeat),
                    str(mc.pattern_axis or "y"),
                )
                smapped = PatternMap()(smapped)

            # pyMOTO measures an element-count radius with unit cells on every
            # axis, so a non-cubic voxel makes the cone anisotropic in model
            # units: on a 1x4x1 mm grid a two-element radius regularizes over
            # 1 mm in x and z and 4 mm in y. Convert such a study to the
            # physical radius its coarsest axis implies, so the regularization
            # length is the same in every direction. A near-cubic grid keeps
            # the element-count kernel and its results bit for bit.
            filter_radius = float(p.rmin)
            radius_is_physical = bool(p.filter_radius_is_physical)
            voxel_edges = (float(p.unitx), float(p.unity), float(p.unitz))
            if not radius_is_physical and max(voxel_edges) > 1.05 * min(voxel_edges):
                filter_radius = float(p.rmin) * max(voxel_edges)
                radius_is_physical = True
                logger.info(
                    "Voxels are anisotropic (%.4g x %.4g x %.4g); the %.3g-element "
                    "filter radius is applied as %.4g model units so the cone "
                    "stays isotropic.",
                    *voxel_edges,
                    float(p.rmin),
                    filter_radius,
                )
            # Boundary treatment is deliberately pyMOTO's default zero-flux
            # (mirror) padding on all six faces, for both the physical and the
            # element-count radius.
            #
            # Padding with void instead treats everything outside the design box
            # as empty, and because the cone kernel is normalized to sum 1 that
            # caps the physical density a boundary voxel can ever reach: with
            # the program-controlled rmin of 1.5 voxels a fully solid domain
            # filters down to 0.85 on a face and 0.59 at a corner, i.e. E/E0 of
            # 0.61 and 0.20 under SIMP p=3. The optimizer cannot buy that back,
            # because the raw design variable is already at its upper bound, so
            # the study is biased against putting material on the outer surface
            # of the part -- exactly where a bending load path wants it -- and
            # the corners drop below the recovery cutoff and get chamfered off.
            # Mirror padding is the zero-flux condition: a design that is
            # uniform across a domain face stays uniform through the filter.
            sxfilt_raw = pym.FilterConv(
                domain=domain,
                radius=filter_radius,
                relative_units=not radius_is_physical,
            )(smapped)
            if (mc.extrusion or "none").lower() != "none":
                # Zero-flux padding already preserves an exactly invariant
                # field along the extrusion axis, so this is a guard rather
                # than a correction: it pins the density used by FEA,
                # projection, recovery and CAD to the requested constant
                # cross-section whatever the upstream mappings did. Void
                # padding *was* a correction here, and that it was needed at
                # all is what exposed the boundary erosion described above.
                # This module carries its exact transpose in the sensitivity
                # path, so it is free of gradient error either way.
                PhysicalExtrusionMap = _make_linear_grid_projection_module(
                    domain,
                    symmetry="none",
                    extrusion=str(mc.extrusion),
                )
                sxfilt_raw = PhysicalExtrusionMap()(sxfilt_raw)
            sxfilt = PassiveClamp(active_mask, passive_density)(sxfilt_raw)
            build_key = str(mc.overhang_build_axis or "none").lower()
            if build_key in _OVERHANG_DIRECTIONS:
                # pyMOTO 2.0.1 prints its sampled cone offsets during module
                # construction. Suppress that dependency debug output while
                # retaining the differentiable Langelaar filter itself.
                with redirect_stdout(io.StringIO()):
                    overhang_module = pym.OverhangFilter(
                        domain=domain,
                        direction=_OVERHANG_DIRECTIONS[build_key],
                        overhang_angle=float(mc.overhang_angle_deg),
                    )
                sxfilt = overhang_module(sxfilt)
                sxfilt = PassiveClamp(active_mask, passive_density)(sxfilt)
            if str(mc.pull_out_axis or "none").lower() != "none":
                PullOutMap = _make_pull_out_module(domain, mc.pull_out_axis)
                sxfilt = PullOutMap()(sxfilt)
                sxfilt = PassiveClamp(active_mask, passive_density)(sxfilt)
            if max_member_radius_voxels > 0.0:
                MaxMemberMap = _make_max_member_module(
                    domain,
                    max_member_radius_voxels,
                    float(mc.max_member_threshold),
                )
                sxfilt = MaxMemberMap()(sxfilt)
                sxfilt = PassiveClamp(active_mask, passive_density)(sxfilt)
            if use_heaviside:
                HeavisideCls = _make_heaviside_module()
                heaviside_module_ref = HeavisideCls(
                    beta=float(p.heaviside_beta_init),
                    eta=float(p.heaviside_eta),
                )
                projection_modules.append(heaviside_module_ref)
                sx_blueprint = heaviside_module_ref(sxfilt)
                sx_blueprint.tag = "physical_density"
                if use_robust:
                    # Robust formulation: one filtered field, three thresholds.
                    #
                    # The eroded field is what the physics is evaluated on. That
                    # is the whole mechanism by which the requested minimum
                    # member size binds -- a member thinner than the length
                    # scale survives in the blueprint but is gone from the
                    # eroded field, so the analysed structure loses that load
                    # path and the optimizer is charged for it. For compliance
                    # the eroded design is also the worst of the three, so
                    # optimizing it alone is the worst case and costs no
                    # additional linear solves: the other two branches are
                    # element-wise projections with no FE attached.
                    eroded_module = HeavisideCls(
                        beta=float(p.heaviside_beta_init),
                        eta=float(p.eta_eroded),
                    )
                    dilated_module = HeavisideCls(
                        beta=float(p.heaviside_beta_init),
                        eta=float(p.eta_dilated),
                    )
                    projection_modules.extend([eroded_module, dilated_module])
                    sxphys = eroded_module(sxfilt)
                    sxphys.tag = "eroded_density"
                    sx_volume = dilated_module(sxfilt)
                    sx_volume.tag = "dilated_density"
                else:
                    sxphys = sx_blueprint
                    sx_volume = sx_blueprint
            else:
                sxphys = sxfilt
                sx_blueprint = sxfilt
                sx_volume = sxfilt
            SparseToCSC = _make_sparse_to_csc_module()
            SimpCls = _make_simp_module()
            CellMaterialCls = (
                make_cell_material_module() if cell_law is not None else None
            )
            # Integrated once and shared by every load case; the geometry of an
            # element does not change during a run.
            cubic_basis = (
                cubic_basis_element_matrices(domain)
                if cell_law is not None
                else ()
            )

            def _compliance_branch(
                density_signal: Any,
                label: str,
            ) -> tuple[Any, list[Any], list[Any]]:
                """Assemble, solve and aggregate compliance for one density field.

                Returns the load-aggregated compliance signal, the per-case
                displacement signals and the unweighted per-case compliances.
                Costs one linear solve per load case.
                """
                if cell_law is not None:
                    # Two-scale path: density selects a homogenized cell, whose
                    # three cubic constants scale three constant basis element
                    # matrices. The assembled operator is genuinely anisotropic
                    # — an isotropic power law cannot express the octet cell's
                    # Zener ratio of ~1.5, i.e. half again the shear stiffness
                    # its bulk response would imply.
                    material = CellMaterialCls(
                        cell_law,
                        young=p.E0,
                        minimum_young=p.Emin,
                        poisson=p.nu,
                    )(density_signal)
                else:
                    simp = SimpCls(
                        minimum=float(p.Emin),
                        full=float(p.E0),
                        penal=_continued_penal(penal_target),
                    )
                    simp_modules.append((simp, penal_target))
                    material = (simp(density_signal),)

                # Per-load-case compliance, then weighted sum → objective.
                # pym.Scaling normalises (NOT multiplies) so we use MathExpression
                # to apply the true scalar weight to each compliance term.
                branch_us: list[Any] = []
                branch_weighted: list[Any] = []
                branch_raw: list[Any] = []
                for case in load_cases:
                    if cell_law is not None:
                        sK_raw = pym.AssembleGeneral(
                            domain,
                            list(cubic_basis),
                            bc=case.boundary_dofs,
                            add_constant=case.joint_stiffness,
                        )(*material)
                    else:
                        sK_raw = pym.AssembleStiffness(
                            domain=domain,
                            bc=case.boundary_dofs,
                            poisson_ratio=p.nu,
                            add_constant=case.joint_stiffness,
                        )(*material)
                    case_solver = _linear_solver_for_case(
                        pym,
                        domain,
                        case.joint_stiffness,
                    )
                    # Only the direct path factorizes this operator, and SciPy's
                    # splu wants CSC. Multigrid-CG never factorizes the fine
                    # grid -- it only forms matrix-vector products, which are
                    # faster in the CSR layout the assembler already emits.
                    # Converting there costs an O(nnz) rebuild per assembly and
                    # then slows every smoothing matvec in every V-cycle.
                    sK = SparseToCSC()(sK_raw) if case_solver is None else sK_raw
                    su_i = pym.LinSolve(
                        symmetric=True,
                        positive_definite=True,
                        solver=case_solver,
                    )(sK, case.force)
                    branch_us.append(su_i)
                    sc_i = pym.EinSum(expression="i,i->")(su_i, case.force)
                    sc_i.tag = f"compliance:{case.name}{label}"
                    branch_raw.append(sc_i)
                    if abs(case.weight - 1.0) > 1e-12:
                        sc_i = pym.MathExpression(
                            expression=f"{float(case.weight)}*inp0"
                        )(sc_i)
                    branch_weighted.append(sc_i)

                if len(branch_weighted) == 1:
                    aggregate = branch_weighted[0]
                elif len(branch_weighted) > 1:
                    if p.load_aggregation == "worst_case":
                        ConcatCls = _make_concat_module()
                        combined = ConcatCls()(*branch_weighted)
                        aggregate = pym.PNorm(p=float(p.load_pnorm_p))(combined)
                    else:
                        expr = " + ".join(
                            f"inp{i}" for i in range(len(branch_weighted))
                        )
                        aggregate = pym.MathExpression(expression=expr)(
                            *branch_weighted
                        )
                else:
                    # No loads — fabricate a zero compliance signal so the graph
                    # builds.
                    aggregate = pym.MathExpression(expression="0*inp0")(
                        density_signal
                    )
                return aggregate, branch_us, branch_raw

            sg0, sus, raw_scomps = _compliance_branch(sxphys, "")
            sg0.tag = "compliance"

            structural_objective = sg0

            sg0_scaled = (
                pym.Scaling(scaling=100.0)(structural_objective)
                if uses_structural
                else None
            )

            # Held on the dilated field under the robust formulation. Bounding
            # the eroded volume would leave the delivered blueprint free to
            # overshoot the budget, and bounding the blueprint directly is what
            # the literature reports as unstable; the dilated bound is
            # rescaled during the solve so the blueprint lands on the request.
            volume_density = sx_volume
            svol = pym.EinSum(expression="i->")(volume_density)
            svol.tag = "volume"
            VolumeConstraint = _make_volume_constraint_module()
            volume_constraint_module_ref = VolumeConstraint(total_vol_target)
            sg1 = volume_constraint_module_ref(volume_density)
            sg1.tag = "volume constraint"

            thermal_compliances: list[Any] = []
            raw_thermal_compliances: list[Any] = []
            thermal_states: list[Any] = []
            if uses_thermal:
                thermal_simp = SimpCls(
                    minimum=float(p.thermal_conductivity_min),
                    full=float(p.thermal_conductivity),
                    penal=_continued_penal(thermal_penal_target),
                )
                simp_modules.append((thermal_simp, thermal_penal_target))
                sk = thermal_simp(sxphys)
                sP_raw = pym.AssemblePoisson(
                    domain=domain,
                    bc=thermal_boundary_nodes,
                    material_property=1.0,
                )(sk)
                # Reduced-order distributed heat rejection. This is a
                # volumetric sink proportional to local material; it does not
                # resolve wetted surface area or fluid flow. Use conjugate CFD
                # and independent thermal validation for a physical heat sink.
                convection = float(
                    getattr(p.thermal_bc, "convection_coefficient", 0.0) or 0.0
                )
                if convection > 0.0:
                    MatrixSum = _make_matrix_sum_module()
                    sH = pym.AssembleMass(
                        domain=domain,
                        bc=thermal_boundary_nodes,
                        material_property=convection,
                        ndof=1,
                    )(sxphys)
                    sP_raw = MatrixSum()(sP_raw, sH)
                # Every case shares this operator but needs its own solver
                # instance, because pyMOTO's linear-dependency-aware wrapper
                # keeps a per-module solution database. See the structural
                # branch for why the CSC conversion is direct-solve only.
                thermal_solvers = [
                    _linear_solver_for_domain(pym, domain) for _ in thermal_cases
                ]
                sP = (
                    SparseToCSC()(sP_raw)
                    if any(solver is None for solver in thermal_solvers)
                    else sP_raw
                )
                for thermal_case, thermal_solver in zip(thermal_cases, thermal_solvers):
                    st_i = pym.LinSolve(
                        symmetric=True,
                        positive_definite=True,
                        solver=thermal_solver,
                    )(sP, thermal_case.heat)
                    thermal_states.append(st_i)
                    sth_i = pym.EinSum(expression="i,i->")(st_i, thermal_case.heat)
                    sth_i.tag = f"thermal_compliance:{thermal_case.name}"
                    raw_thermal_compliances.append(sth_i)
                    if abs(thermal_case.weight - 1.0) > 1e-12:
                        sth_i = pym.MathExpression(
                            expression=f"{float(thermal_case.weight)}*inp0"
                        )(sth_i)
                    thermal_compliances.append(sth_i)
                if len(thermal_compliances) == 1:
                    sg_thermal = thermal_compliances[0]
                elif p.load_aggregation == "worst_case":
                    ConcatCls = _make_concat_module()
                    combined_thermal = ConcatCls()(*thermal_compliances)
                    sg_thermal = pym.PNorm(p=float(p.load_pnorm_p))(combined_thermal)
                else:
                    expression = " + ".join(
                        f"inp{i}" for i in range(len(thermal_compliances))
                    )
                    sg_thermal = pym.MathExpression(expression=expression)(
                        *thermal_compliances
                    )
                sg_thermal.tag = "thermal_compliance"
                sg_thermal_scaled = pym.Scaling(scaling=100.0)(sg_thermal)
            else:
                sg_thermal = None
                sg_thermal_scaled = None

            if minimum_mass_objective:
                sobjective_raw = pym.MathExpression(expression=f"inp0/{domain.nel}")(
                    svol
                )
                sobjective_raw.tag = "material fraction objective"
                sobjective_scaled = pym.Scaling(scaling=100.0)(sobjective_raw)
            elif p.physics_mode == "thermal":
                sobjective_raw, sobjective_scaled = sg_thermal, sg_thermal_scaled
            elif p.physics_mode == "thermo_mechanical":
                sobjective_raw = pym.MathExpression(
                    expression=(
                        f"{float(p.structural_weight)}*inp0 + "
                        f"{float(p.thermal_weight)}*inp1"
                    )
                )(sg0_scaled, sg_thermal_scaled)
                sobjective_raw.tag = "thermo-mechanical objective"
                sobjective_scaled = sobjective_raw
            else:
                sobjective_raw, sobjective_scaled = structural_objective, sg0_scaled
            sobjective_scaled.tag = "objective"

            # ── Phase 3: stress constraint (P-norm aggregated von Mises) ───
            # Aggregate vm² over ALL load cases into a single PNorm so a
            # hot-spot under any LC is penalised. Previously only LC[0] was
            # used — blind to peak stresses under secondary loadings.
            sg_stress = None
            s_pn_stress = None
            if p.stress_constraint_enabled and sus:
                yield_sq = float(p.yield_stress) ** 2
                if yield_sq <= 0.0:
                    yield_sq = 1.0

                VonMisesCls = _make_vm_module()
                vm_sq_signals: list[Any] = []
                for lc_idx, su_i in enumerate(sus):
                    s_voigt_i = pym.Stress(
                        domain=domain,
                        e_modulus=float(p.E0),
                        poisson_ratio=float(p.nu),
                    )(su_i)
                    s_voigt_i.tag = f"stress_voigt:LC{lc_idx}"
                    vm_sq_i = VonMisesCls(stress_penalty=float(p.stress_penalty))(
                        s_voigt_i, sxphys
                    )
                    vm_sq_i.tag = f"vm_squared:LC{lc_idx}"
                    vm_sq_signals.append(vm_sq_i)

                if len(vm_sq_signals) == 1:
                    vm_sq_all = vm_sq_signals[0]
                else:
                    # PNorm of the concatenation = (Σ_lc Σ_e (vm²_{e,lc})^p)^(1/p),
                    # the tightest single envelope over all elements and LCs.
                    ConcatCls = _make_concat_module()
                    vm_sq_all = ConcatCls()(*vm_sq_signals)
                    vm_sq_all.tag = "vm_squared:all_LCs"

                # vm_sq is aggregated with p/2 so sqrt(PNorm(vm_sq, p/2))
                # equals the requested P-norm of von Mises stress.
                # An unscaled finite P-norm is always above the true maximum
                # and its conservatism grows with the number of similarly
                # stressed elements.  Adaptive constraint scaling tracks the
                # actual maximum between iterations while the P-norm supplies
                # a smooth sensitivity for GCMMA.
                stress_scaling = pym.AggScaling(
                    "max",
                    damping=float(p.stress_scaling_damping),
                )
                s_pn_stress = pym.PNorm(
                    p=0.5 * float(p.stress_pnorm_p),
                    scaling=stress_scaling,
                )(vm_sq_all)
                s_pn_stress.tag = "stress_pnorm_sq"

                sg_stress = pym.MathExpression(expression=f"inp0/{yield_sq} - 1.0")(
                    s_pn_stress
                )
                sg_stress.tag = "stress constraint"

        # Sub-network that produces the filtered field, i.e. everything up to
        # but not including the projection. It contains no linear solve, so the
        # volume-preserving threshold can be re-solved against the design that
        # is about to be analyzed instead of the previous one, without paying
        # for a second FE evaluation.
        filter_stage: Any = None
        if heaviside_module_ref is not None:
            for index, module in enumerate(net.mods):
                if any(signal is sxfilt for signal in (module.sig_out or ())):
                    filter_stage = net[: index + 1]
                    break

        net.response()

        # ── iteration loop ────────────────────────────────────────────────
        # `sx_blueprint` is the design that gets delivered: the nominal
        # projection at eta = 0.5, which is the shape recovery, the report and
        # the CAD all describe. Without the robust formulation it is the same
        # signal the FEA runs on; with it the FEA runs on the eroded field
        # (`sxphys`) and the blueprint is the nominal member of that family.
        # Reporting the eroded field instead would hand back a part thinner
        # than the one that was designed.
        result = TopologyOptVoxelResult(
            density=_physical_density_grid(np.asarray(sx_blueprint.state, dtype=float)),
            design_density=_density_grid_from_state(sx.state, domain),
            active_target_volfrac=active_volfrac,
            min_source_volfrac=float(volume_budget["min_source_volfrac"]),
            passive_source_volfrac=(
                float(volume_budget["passive_source_sum"])
                / max(float(volume_budget["source_count"]), 1.0)
            ),
            optimizer_used=optimizer_choice,
            formulation_used="Density (SIMP)",
        )
        comp_hist: list[float] = []
        change_hist: list[float] = []

        objective_hist: list[float] = []
        thermal_hist: list[float] = []
        beta_hist: list[float] = []
        eta_hist: list[float] = []
        penal_hist: list[float] = []
        # Optimizer compatibility was resolved before the network was built.
        if minimum_mass_objective and optimizer_choice not in MMA_FAMILY:
            logger.info(
                "Minimum-mass objective requires the MMA family; forcing "
                "GCMMA (was '%s').",
                optimizer_choice,
            )
            optimizer_choice = "GCMMA"

        if sg_stress is not None and optimizer_choice not in MMA_FAMILY:
            logger.info(
                "Stress constraint enabled — forcing optimizer to GCMMA (was '%s').",
                optimizer_choice,
            )
            optimizer_choice = "GCMMA"

        if optimizer_choice in MMA_FAMILY:
            # Passive variables are clamped by a differentiable network module
            # and re-applied after every step. Giving MMA nearly collapsed
            # bounds for those variables makes its primal-dual subproblem badly
            # conditioned; ordinary bounds plus zero sensitivities are both
            # exact for the physical problem and much faster numerically.
            xmin = np.full(domain.nel, 1e-3)
            # A lattice study's variable is the cell relative density, so its
            # ceiling is a bound on the design space, not a display preference.
            # See `TopologyOptVoxelProblem.lattice_maximum_density`.
            xmax = np.full(domain.nel, float(p.lattice_maximum_density))
            # For minimum mass, ``volfrac`` is the initial design only. Adding
            # the ordinary volume-fraction response here would turn it into an
            # unintended upper bound and can make an otherwise feasible stress
            # problem impossible whenever the optimizer must first add
            # material. The mass objective itself drives volume downward.
            mma_responses = [sobjective_scaled]
            if not minimum_mass_objective:
                mma_responses.append(sg1)
            if sg_stress is not None:
                mma_responses.append(sg_stress)
            mma = pym.MMA(
                sx,
                mma_responses,
                net,
                xmin=xmin,
                xmax=xmax,
                move=0.2,
                verbosity=0,
                mmaversion=("GCMMA" if optimizer_choice == "GCMMA" else "MMA2007"),
                # pyMOTO otherwise solves each barrier subproblem to 1e-10,
                # far beyond the density/FE discretization accuracy.
                epsimin=1e-7,
                gcmma_maxit=10,
            )

        stress_hist: list[float] = []

        # Early-stop is judged on the OBJECTIVE (relative compliance change),
        # sustained for `patience` iterations, with a loose robust density-change
        # gate.  Watching max|Δρ| alone never trips: a single voxel oscillating
        # by the OC move limit near ρ≈0.5 pins it above tol while the design has
        # long since settled, so the run always burns through to max_iter.
        obj_tol = float(p.tol)
        density_gate = max(2.0 * obj_tol, 0.015)
        patience = max(1, int(getattr(p, "patience", 5) or 5))
        min_iter = min(int(p.max_iter), max(10, 2 * patience))
        stall = 0
        prev_objective = None
        latest_obj_change: float | None = None

        # β-continuation schedule for the Heaviside projection. β doubles
        # until it reaches β_max, which is what drives the density field to
        # 0/1 and yields a crisp, manufacturable topology.
        #
        # The interval is derived from the iteration budget rather than taken
        # as a fixed count. With the fixed 30-iteration step, every shipped
        # study (35-60 iterations) stopped at β = 2 of 16 — one single step —
        # so the projection never engaged and the design stayed grey, which
        # in turn produced blobby recovered surfaces. Scheduling all the
        # doublings inside ~80% of the budget lets the continuation finish
        # whatever budget the user chose, while leaving a tail of iterations
        # to converge at full β.
        # p-continuation schedule. Scheduled ahead of the projection rather
        # than alongside it: raising the exponent decides *where* the load path
        # goes, sharpening the projection decides how crisp its edges are, and
        # committing the second before the first has settled sharpens a
        # topology that is still moving. All p steps therefore land inside the
        # first `INDUSTRIAL_PENAL_BUDGET_FRACTION` of the budget, which for the
        # 120-iteration guided budget is a step every 12 iterations for the
        # default 1 -> 3.
        penal_step_interval = 0
        if use_penal_continuation:
            penal_step_interval = max(
                1,
                int(
                    INDUSTRIAL_PENAL_BUDGET_FRACTION
                    * float(p.max_iter)
                    / float(penal_steps)
                ),
            )
            logger.info(
                "SIMP continuation: p %.3g -> %.3g in %d steps every %d "
                "iterations (budget %d).",
                penal_init,
                penal_target,
                penal_steps,
                penal_step_interval,
                int(p.max_iter),
            )
        penal_steps_taken = 0
        last_penal_step = 0

        hv_beta = (
            float(p.heaviside_beta_init) if heaviside_module_ref is not None else 0.0
        )
        hv_beta_max = float(p.heaviside_beta_max)
        hv_step = max(1, int(p.heaviside_beta_step_iters))
        last_beta_step = 0
        if heaviside_module_ref is not None and hv_beta > 0.0:
            doublings = max(
                1, int(np.ceil(np.log2(max(hv_beta_max / hv_beta, 1.0 + 1e-12))))
            )
            hv_step = max(
                1, min(hv_step, int(0.8 * float(p.max_iter) / float(doublings)))
            )
            logger.info(
                "Heaviside continuation: beta %.3g -> %.3g in %d steps every "
                "%d iterations (budget %d).",
                hv_beta,
                hv_beta_max,
                doublings,
                hv_step,
                int(p.max_iter),
            )

        it, change = 0, 1.0
        while it < p.max_iter:
            if self.stop_requested:
                result.message = "Stopped by user"
                break

            it += 1
            x_old = sx.state.copy()

            # ── design update ─────────────────────────────────────────────
            # The network already holds the response for the current design at
            # the current continuation parameters, evaluated at the end of the
            # previous iteration. Stepping first, and advancing p/beta/eta only
            # afterwards, keeps every subproblem self-consistent while paying
            # for exactly one FE evaluation per iteration. Advancing them
            # first instead invalidates that response and forces MMA to re-run
            # every linear solve before it can even form the subproblem.
            if optimizer_choice == "OC":
                net.reset()
                sobjective_scaled.sensitivity = 1.0
                net.sensitivity()
                dc = sx.sensitivity.copy()
                sx.state = _oc_update(
                    sx.state,
                    dc,
                    active_volfrac,
                    active_mask=active_mask,
                    passive_density=passive_density,
                    maximum_density=p.lattice_maximum_density,
                )
            else:
                x_new, _, _ = mma.step(x=sx.state)
                x_new = np.asarray(x_new, dtype=float)
                # MMA already respects xmin=xmax for passive, but re-clamp to
                # guarantee bit-exact passive density and avoid drift.
                x_new[~active_mask] = passive_density[~active_mask]
                sx.state = x_new
                mma.iter += 1

            # ── continuation, applied to the design just accepted ─────────
            # A continuation level gets a minimum dwell before either a plateau
            # or the fixed interval may advance it. This avoids repeated beta
            # jumps caused by a stale objective-change value.
            if (
                use_penal_continuation
                and it > 1
                and penal_steps_taken < penal_steps
                and it - last_penal_step >= penal_step_interval
            ):
                penal_steps_taken += 1
                penal_progress = float(penal_steps_taken) / float(penal_steps)
                for simp_module, simp_target in simp_modules:
                    simp_module.penal = _continued_penal(simp_target)
                # Raising the exponent changes the material model, so an
                # objective plateau accumulated under the softer one is not
                # convergence evidence for the penalized one.
                stall = 0
                prev_objective = None
                latest_obj_change = None
                last_penal_step = it

            should_step_beta = False
            if heaviside_module_ref is not None and it > 1 and hv_beta < hv_beta_max:
                level_age = it - last_beta_step
                minimum_dwell = max(3, min(hv_step, max(3, hv_step // 3)))
                if level_age >= hv_step:
                    should_step_beta = True
                elif (
                    level_age >= minimum_dwell
                    and latest_obj_change is not None
                    and latest_obj_change < 0.008
                ):
                    should_step_beta = True
            if should_step_beta:
                hv_beta = min(hv_beta * 2.0, hv_beta_max)
                # All projections share beta; the thresholds are what differ.
                for module in projection_modules:
                    module.beta = hv_beta
                # A continuation step changes the optimization problem. Any
                # objective plateau accumulated at the previous beta is no
                # longer convergence evidence for the sharpened design.
                stall = 0
                prev_objective = None
                latest_obj_change = None
                last_beta_step = it

            if use_robust and not minimum_mass_objective:
                # The three thresholds are the length scale, so none of them
                # may drift: moving the blueprint off 0.5 would change the
                # member size the eroded field is enforcing, silently, in the
                # middle of the solve. Volume is held instead by rescaling the
                # bound on the dilated field until the blueprint -- the design
                # actually delivered -- sits on the requested fraction. This is
                # the standard robust volume update, and it is why the bound
                # can be placed on a field that is not the one being reported.
                if it % max(1, int(p.dilated_volume_update_iters)) == 0:
                    # Re-run the filter chain so the blueprint volume is the
                    # one about to be analysed rather than the previous
                    # iteration's; no linear solve sits in that stage.
                    if filter_stage is not None:
                        filter_stage.response()
                    blueprint_state = _smooth_heaviside_values(
                        np.asarray(sxfilt.state, dtype=float),
                        beta=hv_beta,
                        eta=float(p.heaviside_eta),
                    )
                    blueprint_mean = float(np.mean(blueprint_state))
                    if blueprint_mean > 1e-9:
                        volume_constraint_module_ref.target_fraction = float(
                            np.clip(
                                volume_constraint_module_ref.target_fraction
                                * total_vol_target
                                / blueprint_mean,
                                1e-6,
                                1.0,
                            )
                        )
            elif (
                heaviside_module_ref is not None
                and not minimum_mass_objective
            ):
                # Keep beta continuation from silently deleting material.
                # Re-run the filter chain first so the threshold is solved
                # against the design that is about to be analyzed; those
                # modules carry no linear solve, so this is nearly free.
                if filter_stage is not None:
                    filter_stage.response()
                updated_eta = _volume_preserving_projection_eta(
                    np.asarray(sxfilt.state, dtype=float),
                    beta=hv_beta,
                    active_mask=active_mask,
                    passive_density=passive_density,
                    source_mask=source_mask_flat,
                    target_source_sum=float(
                        volume_budget["source_total_target"]
                    ),
                )
                heaviside_module_ref.eta = updated_eta

            # The single FE evaluation of the iteration. It reports the new
            # design and doubles as the response MMA differentiates next time,
            # which is why nothing above may invalidate it.
            net.response()

            comp_val = float(sg0.state)
            thermal_val = float(sg_thermal.state) if sg_thermal is not None else 0.0
            objective_val = float(sobjective_raw.state)
            objective_hist.append(objective_val)
            beta_hist.append(float(hv_beta))
            penal_hist.append(
                _continued_penal(
                    penal_target if uses_structural else thermal_penal_target
                )
            )
            eta_hist.append(
                float(heaviside_module_ref.eta)
                if heaviside_module_ref is not None
                else 0.5
            )
            # Robust density change: the MEAN over active voxels, not the single
            # worst one, so a handful of oscillating boundary voxels can't veto
            # the stop.  Reported to the callback / change_history.
            delta = np.abs(sx.state - x_old)
            change = (
                float(np.mean(delta[active_mask]))
                if np.any(active_mask)
                else float(np.mean(delta))
            )
            if uses_structural:
                comp_hist.append(comp_val)
            if uses_thermal:
                thermal_hist.append(thermal_val)
            change_hist.append(change)
            if s_pn_stress is not None:
                # s_pn_stress.state is the P-norm of vm² → sqrt gives σ_pn
                pn_val = float(np.asarray(s_pn_stress.state).flatten()[0])
                stress_hist.append(float(np.sqrt(max(pn_val, 0.0))))
            result.n_iter = it

            if callback is not None:
                density_3d = _physical_density_grid(
                    np.asarray(sx_blueprint.state, dtype=float)
                )
                callback(
                    it,
                    comp_val if uses_structural else thermal_val,
                    change,
                    density_3d,
                )

            # Objective-based early stop: relative compliance change below tol
            # for `patience` consecutive iterations, with the robust density
            # change also settled (a guard against stopping on a transient
            # compliance plateau while the topology is still reorganising).
            if prev_objective is not None:
                latest_obj_change = abs(objective_val - prev_objective) / max(
                    abs(objective_val), 1e-12
                )
                stress_feasible = not p.stress_constraint_enabled or (
                    bool(stress_hist)
                    and stress_hist[-1]
                    <= float(p.yield_stress) * (1.0 + max(obj_tol, 0.005))
                )
                # The projection is still being sharpened until beta reaches
                # its maximum, and the objective plateaus between beta steps.
                # Declaring convergence during that plateau ends the run on a
                # half-projected, grey design: measured on the shipped
                # cantilever, stopping mid-continuation left 11.6% grey where
                # a completed continuation reached 6.4%.
                # The same argument applies to the SIMP exponent, and more
                # sharply: p=1 is a convex problem whose optimum is a fully
                # grey compliant design. Stopping on its plateau would deliver
                # a study that never met the penalization the user asked for.
                continuation_complete = (
                    heaviside_module_ref is None
                    or hv_beta >= hv_beta_max - 1e-9
                ) and (
                    not use_penal_continuation
                    or penal_steps_taken >= penal_steps
                )
                continuation_tail_complete = continuation_complete and (
                    it >= max(last_beta_step, last_penal_step) + max(3, patience)
                )
                # Compliance minimization is monotone in material stiffness:
                # a settled solution should use essentially all of its upper
                # material budget. A large shortfall after beta continuation
                # is normally a transient MMA/projection plateau, not an
                # optimum. Minimum-mass studies intentionally do not satisfy
                # this check.
                # Judge the same density signal and target the active volume
                # constraint uses, rather than assuming they are the nominal
                # field and the nominal target.
                constrained_density = np.asarray(
                    volume_density.state,
                    dtype=float,
                )
                current_constrained_fraction = float(
                    np.mean(constrained_density)
                )
                current_volume_target = float(
                    volume_constraint_module_ref.target_fraction
                )
                material_budget_settled = minimum_mass_objective or (
                    current_constrained_fraction
                    >= current_volume_target
                    - max(0.01, 0.03 * current_volume_target)
                )
                if (
                    latest_obj_change < obj_tol
                    and change < density_gate
                    and stress_feasible
                    and continuation_complete
                    and continuation_tail_complete
                    and material_budget_settled
                ):
                    stall += 1
                else:
                    stall = 0
                if it >= min_iter and stall >= patience:
                    result.converged = True
                    result.message = (
                        f"Converged in {it} iterations "
                        f"(relative objective change < {obj_tol:.1e} for "
                        f"{patience} iters; mean |d_rho| = {change:.2e})"
                    )
                    break
            prev_objective = objective_val

        if not result.message:
            result.message = f"Maximum iterations ({p.max_iter}) reached"

        net.response()
        result.design_density = _density_grid_from_state(sx.state, domain)
        result.density = _physical_density_grid(
            np.asarray(sx_blueprint.state, dtype=float)
        )
        # Surface recovery needs the filtered field, not the projected one; see
        # TopologyOptVoxelResult.recovery_density.
        result.recovery_density = _physical_density_grid(
            np.asarray(sxfilt.state, dtype=float)
        )
        # The projection is what separates the filtered field from the physical
        # one, so inverting it is what converts a cutoff on the physical field
        # into the level describing the same boundary on the filtered field.
        # Keyed off the module rather than the beta history: with no projection
        # in the network the history still carries a beta, but no projection was
        # applied and the cutoff must pass through unchanged.
        result.projection_beta = (
            float(hv_beta) if heaviside_module_ref is not None else None
        )
        result.projection_eta = (
            float(heaviside_module_ref.eta)
            if heaviside_module_ref is not None
            else None
        )
        result.recovery_cutoff = _projection_matched_level(
            0.5,
            beta=result.projection_beta or 0.0,
            eta=result.projection_eta if result.projection_eta is not None else 0.5,
        )
        result.compliance_history = comp_hist
        result.change_history = change_hist
        result.stress_history = stress_hist
        result.thermal_compliance_history = thermal_hist
        result.objective_history = objective_hist
        result.beta_history = beta_hist
        result.projection_eta_history = eta_hist
        result.penal_history = penal_hist
        result.case_compliances = {
            case.name: float(np.asarray(signal.state).reshape(-1)[0])
            for case, signal in zip(load_cases, raw_scomps)
        }
        result.case_max_displacements = {
            case.name: float(
                np.max(
                    np.linalg.norm(
                        np.asarray(signal.state, dtype=float).reshape((-1, 3)),
                        axis=1,
                    )
                )
            )
            for case, signal in zip(load_cases, sus)
        }
        result.thermal_case_compliances = {
            case.name: float(np.asarray(signal.state).reshape(-1)[0])
            for case, signal in zip(thermal_cases, raw_thermal_compliances)
        }
        result.thermal_case_max_temperature_rises = {
            case.name: float(np.max(np.asarray(signal.state, dtype=float)))
            for case, signal in zip(thermal_cases, thermal_states)
        }
        return result
