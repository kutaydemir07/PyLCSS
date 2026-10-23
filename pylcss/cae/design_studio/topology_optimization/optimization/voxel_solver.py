# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Voxel topology solver orchestration and iteration loop."""

from __future__ import annotations

import logging
from time import perf_counter
from typing import Any, Callable, Optional

import numpy as np

from .assembly import _SolverAssemblyMixin
from .problem import (
    INDUSTRIAL_PENAL_BUDGET_FRACTION,
    MMA_FAMILY,
    TopologyOptVoxelProblem,
)
from .pymoto_modules import (
    _make_concat_module,
    _make_heaviside_module,
    _make_passive_clamp_module,
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

# Sparse direct factorization is economical for tiny systems. Larger compatible
# grids use geometric-multigrid CG to reduce factorization memory and runtime.
DIRECT_SOLVE_VOXEL_LIMIT = 256
# Keep linear residuals below topology convergence tolerances so solution error
# does not dominate compliance sensitivities.
ITERATIVE_SOLVE_TOLERANCE = 1e-6
ITERATIVE_SOLVE_MAX_ITER = 5_000
JOINT_SOLVE_TOLERANCE = 1e-6
JOINT_SOLVE_MAX_ITER = 2_000
# Limit depth because high SIMP stiffness contrast weakens coarse corrections.
MULTIGRID_MAX_LEVELS = 2
MULTIGRID_SMOOTH_STEPS = 2
MULTIGRID_MIN_COARSE_ELEMENTS = 2


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
    """Invert the projection to match a physical cutoff on the filtered field.

    Surface recovery uses the smoother filtered field. Its matching level is
    eta + atanh(cutoff * denominator - tanh(beta * eta)) / beta.
    With no projection, the cutoff passes through. Endpoints are clipped to
    avoid degenerate isosurfaces."""
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
    """Build a geometric-multigrid V-cycle rooted at ``domain``.

    Recurse only when the next grid is even and has enough elements. The
    coarsest level uses pyMOTO's direct solver."""
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
    """Choose direct factorization for tiny or incompatible grids, otherwise CG.

    Geometric multigrid requires even dimensions and avoids the fine-grid
    factorization cost of larger 3-D elasticity systems."""
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
    """Set iterative limits for cases containing joint couplings.

    Long-range connector springs are absent from the regular-grid multigrid
    preconditioner, so cap their iterative work separately."""
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
    """Optimize a 3-D solid density field with SIMP and optional robust projection."""

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

        started = perf_counter()
        performance = {'network_response_seconds': 0., 'network_response_calls': 0,
                       'design_update_seconds': 0., 'linear_solvers': []}
        matrix_free_modules = []
        p = self.problem

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
        load_cases = self._assemble_load_cases(
            domain, p.bc, ndof, base_boundary_dofs=boundary_dofs
        )
        if not load_cases:
            raise ValueError(
                "Structural topology optimization needs a non-zero load case."
            )
        if any(case.boundary_dofs.size == 0 for case in load_cases):
            raise ValueError(
                "Every structural load/pose case needs at least one support."
            )
        active_mask, passive_density = self._assemble_passive_masks(domain, p.bc)
        if not np.any(active_mask):
            raise ValueError(
                "Topology study has no designable voxels after source-domain "
                "and passive regions are applied."
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

        volume_budget = volume_budget_from_masks(
            p.volfrac,
            active_mask,
            passive_density,
            source_mask_flat,
        )
        active_volfrac = float(volume_budget["active_volfrac"])
        total_vol_target = float(volume_budget["flat_total_target"]) / float(domain.nel)
        if float(volume_budget["min_source_volfrac"]) > float(p.volfrac) + 1e-9:
            raise ValueError(
                f"Target Volume {float(p.volfrac):.1%} is below the minimum "
                f"{float(volume_budget['min_source_volfrac']):.1%} required by "
                "fixed non-design material and minimum active density. "
                "Increase Target Volume or reduce the non-design regions."
            )
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
            x0 = restore_active_volume(
                x0,
                active_mask,
                active_volfrac,
                passive_density,
            )

        sx = pym.Signal("x", state=x0)
        minimum_mass_objective = str(p.objective_mode).lower() == "minimum_mass"
        optimizer_choice = p.optimizer.upper()
        if optimizer_choice == "AUTO":
            if minimum_mass_objective or p.stress_constraint_enabled:
                optimizer_choice = "GCMMA"
            elif p.heaviside_enabled:
                optimizer_choice = "MMA"
            else:
                optimizer_choice = "OC"
        if (
            minimum_mass_objective
            or p.stress_constraint_enabled
            or p.robust_length_scale
        ) and optimizer_choice not in MMA_FAMILY:
            logger.info(
                "The selected objective/constraints require an MMA-family "
                "optimizer (was '%s').",
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
        if use_robust:
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

        # Start softly, then raise the SIMP exponent toward the study's target.
        penal_init = (
            float(p.penal_init) if p.penal_continuation_enabled else float(p.penal)
        )
        penal_target = float(p.penal)
        widest_penal_range = penal_target - penal_init
        use_penal_continuation = bool(
            p.penal_continuation_enabled
            and widest_penal_range > 1e-9
        )
        penal_steps = (
            max(1, int(np.ceil(widest_penal_range / max(float(p.penal_step), 1e-9))))
            if use_penal_continuation
            else 0
        )
        # Fraction of the way from the initial to the full SIMP exponent.
        penal_progress = 0.0 if use_penal_continuation else 1.0

        def _continued_penal(target: float) -> float:
            return penal_init + penal_progress * (float(target) - penal_init)

        def _physical_density_grid(signal_state: np.ndarray) -> np.ndarray:
            physical = np.asarray(signal_state, dtype=float).copy()
            physical[~active_mask] = passive_density[~active_mask]
            return _density_grid_from_state(physical, domain)

        # ── pyMOTO network ────────────────────────────────────────────────
        # Raw density -> physical cone filter -> passive clamp -> projection -> SIMP.
        heaviside_module_ref: Any = None
        # SIMP interpolations paired with their target exponents.
        simp_modules: list[tuple[Any, float]] = []
        # Every projection module in the network, nominal first. beta continuation
        # has to advance all of them together or the branches stop describing the
        # same design at different thresholds.
        projection_modules: list[Any] = []
        volume_constraint_module_ref: Any = None
        with pym.Network() as net:
            PassiveClamp = _make_passive_clamp_module()
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
            # Mirror padding preserves solid density on domain faces and corners.
            sxfilt_raw = pym.FilterConv(
                domain=domain,
                radius=filter_radius,
                relative_units=not radius_is_physical,
            )(sx)
            sxfilt = PassiveClamp(active_mask, passive_density)(sxfilt_raw)
            if use_heaviside:
                HeavisideCls = _make_heaviside_module()
                heaviside_module_ref = HeavisideCls(
                    beta=float(p.heaviside_beta_init),
                    eta=float(p.heaviside_eta),
                )
                projection_modules.append(heaviside_module_ref)
                # Projection changes even a prescribed void's small positive
                # density. Restore passive values on every branch so FEA,
                # volume accounting, and the delivered field use one material
                # definition throughout beta continuation.
                sx_blueprint = PassiveClamp(active_mask, passive_density)(
                    heaviside_module_ref(sxfilt)
                )
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
                    sxphys = PassiveClamp(active_mask, passive_density)(
                        eroded_module(sxfilt)
                    )
                    sxphys.tag = "eroded_density"
                    sx_volume = PassiveClamp(active_mask, passive_density)(
                        dilated_module(sxfilt)
                    )
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
            def _compliance_branch(
                density_signal: Any,
                label: str,
            ) -> tuple[Any, list[Any], list[Any]]:
                """Assemble, solve and aggregate compliance for one density field.

                Returns the load-aggregated compliance signal, the per-case
                displacement signals and the unweighted per-case compliances.
                Costs one linear solve per load case.
                """
                simp = SimpCls(
                    minimum=float(p.Emin),
                    full=float(p.E0),
                    penal=_continued_penal(penal_target),
                )
                simp_modules.append((simp, penal_target))
                material = simp(density_signal)

                # Per-load-case compliance, then weighted sum → objective.
                # pym.Scaling normalises (NOT multiplies) so we use MathExpression
                # to apply the true scalar weight to each compliance term.
                branch_us: list[Any] = []
                branch_weighted: list[Any] = []
                branch_raw: list[Any] = []
                shared_operators = {}
                for case in load_cases:
                    matrix_free = p.linear_solver == 'matrix_free' or (p.linear_solver == 'auto' and domain.nel >= 4096)
                    if matrix_free:
                        from .matrix_free import make_elastic_solve_module
                        operator_key = (np.asarray(case.boundary_dofs, dtype=np.int64).tobytes(),
                                        id(case.joint_stiffness) if getattr(case.joint_stiffness, 'nnz', 0) else None)
                        module = make_elastic_solve_module(pym)(domain, case.force, case.boundary_dofs,
                                                              p.nu, case.joint_stiffness,
                                                              cancel_callback=lambda: self.stop_requested,
                                                              elasticity=shared_operators.get(operator_key))
                        shared_operators[operator_key] = module.elasticity
                        matrix_free_modules.append(module)
                        performance['linear_solvers'].append(dict(case=case.name,
                            solver='Compiled matrix-free CG', preconditioner='Geometric multigrid; non-dyadic grids supported'))
                        su_i = module(material)
                    else:
                        sK_raw = pym.AssembleStiffness(
                            domain=domain,
                            bc=case.boundary_dofs,
                            poisson_ratio=p.nu,
                            add_constant=case.joint_stiffness,
                        )(material)
                        case_solver = _linear_solver_for_case(
                            pym, domain, case.joint_stiffness,
                        )
                        performance['linear_solvers'].append({
                            'case': case.name,
                            'solver': type(case_solver).__name__ if case_solver is not None else 'Sparse direct',
                            'preconditioner': 'Geometric multigrid' if case_solver is not None else None,
                        })
                        sK = SparseToCSC()(sK_raw) if case_solver is None else sK_raw
                        su_i = pym.LinSolve(
                            symmetric=True, positive_definite=True, solver=case_solver,
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
                else:
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
                return aggregate, branch_us, branch_raw

            sg0, sus, raw_scomps = _compliance_branch(sxphys, "")
            sg0.tag = "compliance"

            structural_objective = sg0

            sg0_scaled = pym.Scaling(scaling=100.0)(structural_objective)

            # Compliance studies adapt the dilated bound until the delivered
            # blueprint reaches the requested budget. Minimum-mass studies
            # may use less material, so constrain their blueprint directly
            # without rescaling the bound toward a full-budget design. Keep
            # the robust mass objective on the dilated field in both cases.
            volume_density = sx_blueprint if minimum_mass_objective else sx_volume
            svol = pym.EinSum(expression="i->")(sx_volume)
            svol.tag = "volume"
            VolumeConstraint = _make_volume_constraint_module()
            volume_constraint_module_ref = VolumeConstraint(total_vol_target)
            sg1 = volume_constraint_module_ref(volume_density)
            sg1.tag = "volume constraint"

            if minimum_mass_objective:
                sobjective_raw = pym.MathExpression(expression=f"inp0/{domain.nel}")(
                    svol
                )
                sobjective_raw.tag = "material fraction objective"
                sobjective_scaled = pym.Scaling(scaling=100.0)(sobjective_raw)
            else:
                sobjective_raw, sobjective_scaled = structural_objective, sg0_scaled
            sobjective_scaled.tag = "objective"

            # Aggregate relaxed stress across every element and load case.
            sg_stress = None
            s_pn_stress = None
            if p.stress_constraint_enabled and sus:
                yield_sq = float(p.yield_stress) ** 2
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

        performance['setup_seconds'] = perf_counter()-started
        def timed_response():
            response_started = perf_counter()
            net.response()
            performance['network_response_seconds'] += perf_counter()-response_started
            performance['network_response_calls'] += 1

        timed_response()

        # ── iteration loop ────────────────────────────────────────────────
        # `sx_blueprint` is the design that gets delivered: the nominal
        # projection at the blueprint threshold, which is the shape recovery, the report and
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
        beta_hist: list[float] = []
        eta_hist: list[float] = []
        penal_hist: list[float] = []
        if optimizer_choice in MMA_FAMILY:
            # Passive variables are clamped by a differentiable network module
            # and re-applied after every step. Giving MMA nearly collapsed
            # bounds for those variables makes its primal-dual subproblem badly
            # conditioned; ordinary bounds plus zero sensitivities are both
            # exact for the physical problem and much faster numerically.
            xmin = np.full(domain.nel, 1e-3)
            xmax = np.ones(domain.nel)
            # Target Volume is an upper material budget for both objectives.
            # A stress-limited study must satisfy it together with the stress
            # bound; an incompatible pair is an infeasible design request.
            mma_responses = [sobjective_scaled, sg1]
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

        # Schedule exponent increases within the first 40% of the budget and
        # projection sharpening within 80%, leaving iterations to settle.
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
        hv_beta_max = (
            float(p.heaviside_beta_max) if heaviside_module_ref is not None else 0.0
        )
        hv_step = (
            max(1, int(p.heaviside_beta_step_iters))
            if heaviside_module_ref is not None else 1
        )
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
            update_started = perf_counter()

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
                sx.state = optimality_criteria_update(
                    sx.state,
                    dc,
                    active_volfrac,
                    active_mask=active_mask,
                    passive_density=passive_density,
                )
            else:
                x_new, _, _ = mma.step(x=sx.state)
                x_new = np.asarray(x_new, dtype=float)
                # Passive variables have ordinary MMA bounds and zero physical
                # sensitivities. Restore their prescribed values after the step.
                x_new[~active_mask] = passive_density[~active_mask]
                sx.state = x_new
                mma.iter += 1
            performance['design_update_seconds'] += perf_counter()-update_started

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
                    blueprint_state[~active_mask] = passive_density[~active_mask]
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
            timed_response()

            comp_val = float(sg0.state)
            objective_val = float(sobjective_raw.state)
            objective_hist.append(objective_val)
            beta_hist.append(float(hv_beta))
            penal_hist.append(
                _continued_penal(
                    penal_target
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
            comp_hist.append(comp_val)
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
                    comp_val,
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
                # A plateau at an intermediate beta or SIMP exponent does not
                # establish convergence for the final optimization problem.
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
                # optimum. Minimum-mass studies may finish below the budget,
                # but must still satisfy its upper bound.
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
                # A plateau above the upper bound is infeasible, even when
                # the design has stopped changing. Also check the delivered
                # blueprint: the robust dilated bound can be satisfied before
                # its periodic rescaling has reached the requested budget.
                volume_tolerance = max(1e-4, 0.01 * current_volume_target)
                material_budget_settled = (
                    current_constrained_fraction
                    <= current_volume_target + volume_tolerance
                )
                if not minimum_mass_objective:
                    material_budget_settled = material_budget_settled and (
                        current_constrained_fraction
                        >= current_volume_target
                        - max(0.01, 0.03 * current_volume_target)
                    )
                if use_robust and not minimum_mass_objective:
                    blueprint_fraction = float(np.mean(sx_blueprint.state))
                    material_budget_settled = material_budget_settled and (
                        total_vol_target - max(0.01, 0.03 * total_vol_target)
                        <= blueprint_fraction
                        <= total_vol_target + max(1e-4, 0.01 * total_vol_target)
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

        # The last accepted design already has a current network response.
        # Re-evaluating here repeats assembly and every load-case solve.
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
        performance.update(total_seconds=perf_counter()-started,
                           grid_shape=[int(domain.nelx), int(domain.nely), int(domain.nelz)],
                           voxel_count=int(domain.nel), load_cases=len(load_cases),
                           stress_constraint_enabled=bool(p.stress_constraint_enabled),
                           robust_length_scale=bool(use_robust))
        if matrix_free_modules:
            performance['matrix_free'] = [module.elasticity.statistics for module in matrix_free_modules]
        result.performance = performance
        return result
