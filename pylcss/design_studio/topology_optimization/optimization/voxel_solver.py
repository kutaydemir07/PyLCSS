# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Voxel topology solver orchestration and iteration loop."""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional

import numpy as np

from ..manufacturing.constraints import (
    _apply_am_overhang,
    _apply_extrusion,
    _apply_max_member_size,
    _apply_pattern_repeat,
    _apply_symmetry,
)
from .assembly import _SolverAssemblyMixin
from .level_set import (
    LEVEL_SET_BETA,
    initialize_level_set,
    level_set_heaviside,
    reaction_diffusion_level_set_update,
)
from .problem import MMA_FAMILY, TopologyOptVoxelProblem
from .pymoto_modules import (
    _density_3d_to_flat,
    _make_concat_module,
    _make_heaviside_module,
    _make_level_set_heaviside_module,
    _make_passive_clamp_module,
    _make_sparse_to_csc_module,
    _make_vm_module,
)
from .pymoto_runtime import import_pymoto
from .results import TopologyOptVoxelResult, _density_grid_from_state
from .update_algorithms import (
    optimality_criteria_update,
    projected_gradient_update,
    restore_active_volume,
    volume_budget_from_masks,
)

logger = logging.getLogger(__name__)

# Private aliases preserve the established extension and test API.
_oc_update = optimality_criteria_update
_restore_active_volume = restore_active_volume
_projected_gradient_update = projected_gradient_update
_volume_budget_from_masks = volume_budget_from_masks


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
        if volume_budget["target_was_clamped"]:
            logger.warning(
                "Topology volume target %.3f is infeasible with passive regions; "
                "minimum source-domain volume is %.3f.",
                float(p.volfrac),
                float(volume_budget["min_source_volfrac"]),
            )

        # ── initial state — design = volfrac, passive = clamp value ───────
        x0 = np.ones(domain.nel) * active_volfrac
        x0[~active_mask] = passive_density[~active_mask]
        is_level_set = p.formulation == "level_set"
        passive_level_set = np.where(passive_density >= 0.5, 1.0, -1.0)
        if is_level_set:
            active_grid = _density_grid_from_state(active_mask, domain).astype(bool)
            passive_phi_grid = _density_grid_from_state(passive_level_set, domain)
            initial_phi_grid = initialize_level_set(
                (p.nelx, p.nely, p.nelz),
                active_grid,
                active_volfrac,
                passive_phi=passive_phi_grid,
            )
            x0 = _density_3d_to_flat(initial_phi_grid, domain)
            x0[~active_mask] = passive_level_set[~active_mask]

        if not load_cases and not thermal_cases:
            logger.warning(
                "TopologyOptVoxelNode: no load cases produced a non-zero force "
                "vector — returning a trivial (uniform) result."
            )
            density_3d = (
                level_set_heaviside(_density_grid_from_state(x0, domain))
                if is_level_set
                else _density_grid_from_state(x0, domain)
            )
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
                formulation_used=(
                    "Level Set (Reaction-Diffusion)"
                    if is_level_set
                    else "Density (SIMP)"
                ),
                level_set_field=(
                    _density_grid_from_state(x0, domain) if is_level_set else None
                ),
            )
            if callback is not None:
                callback(0, 0.0, 0.0, density_3d.copy())
            return result

        sx = pym.Signal("x", state=x0)
        minimum_mass_objective = str(p.objective_mode).lower() == "minimum_mass"
        optimizer_choice = p.optimizer.upper()
        if optimizer_choice == "AUTO":
            optimizer_choice = (
                "GCMMA"
                if minimum_mass_objective or p.stress_constraint_enabled
                else "OC"
            )
        if (
            minimum_mass_objective or p.stress_constraint_enabled
        ) and optimizer_choice not in MMA_FAMILY:
            logger.info(
                "The selected objective/constraints require the MMA family; "
                "forcing GCMMA "
                "(was '%s').",
                optimizer_choice,
            )
            optimizer_choice = "GCMMA"
        if is_level_set:
            optimizer_choice = "REACTION_DIFFUSION"
        use_heaviside = bool(
            not is_level_set and p.heaviside_enabled and optimizer_choice in MMA_FAMILY
        )
        if p.heaviside_enabled and not use_heaviside and not is_level_set:
            logger.info(
                "Smooth Heaviside projection is disabled for %s because OC/PGD "
                "do not solve its physical-volume constraint.",
                optimizer_choice,
            )

        def _physical_density_grid(signal_state: np.ndarray) -> np.ndarray:
            physical = np.asarray(signal_state, dtype=float).copy()
            physical[~active_mask] = passive_density[~active_mask]
            return _density_grid_from_state(physical, domain)

        # ── pyMOTO network ────────────────────────────────────────────────
        # Density formulation: raw density -> Helmholtz filter -> passive clamp
        # -> optional projection -> SIMP. Level-set formulation: signed phi ->
        # smooth ersatz-material phase -> SIMP.
        heaviside_module_ref: Any = None
        with pym.Network() as net:
            PassiveClamp = _make_passive_clamp_module()
            if is_level_set:
                sxfilt = PassiveClamp(active_mask, passive_level_set)(sx)
                LevelSetHeaviside = _make_level_set_heaviside_module()
                sxphys = LevelSetHeaviside(beta=LEVEL_SET_BETA)(sxfilt)
                sxphys.tag = "physical_density"
                sxphys = PassiveClamp(active_mask, passive_density)(sxphys)
            else:
                sxfilt_raw = pym.DensityFilter(domain=domain, radius=p.rmin)(sx)
                sxfilt = PassiveClamp(active_mask, passive_density)(sxfilt_raw)
            if use_heaviside:
                HeavisideCls = _make_heaviside_module()
                heaviside_module_ref = HeavisideCls(
                    beta=float(p.heaviside_beta_init),
                    eta=float(p.heaviside_eta),
                )
                sxphys = heaviside_module_ref(sxfilt)
                sxphys.tag = "physical_density"
                # Re-clamp passive voxels after the projection — the smooth
                # Heaviside is not exactly identity on the clamped endpoints.
                sxphys = PassiveClamp(active_mask, passive_density)(sxphys)
            elif not is_level_set:
                sxphys = sxfilt
            sSIMP = pym.MathExpression(
                expression=f"{p.Emin} + {p.E0 - p.Emin}*inp0^{p.penal}"
            )(sxphys)
            SparseToCSC = _make_sparse_to_csc_module()

            # Per-load-case compliance, then weighted sum → objective.
            # pym.Scaling normalises (NOT multiplies) so we use MathExpression
            # to apply the true scalar weight to each compliance term.
            sus: list[Any] = []
            scomps: list[Any] = []
            for case in load_cases:
                sK_raw = pym.AssembleStiffness(
                    domain=domain,
                    bc=case.boundary_dofs,
                    poisson_ratio=p.nu,
                    add_constant=case.joint_stiffness,
                )(sSIMP)
                sK = SparseToCSC()(sK_raw)
                su_i = pym.LinSolve(symmetric=True, positive_definite=True)(
                    sK, case.force
                )
                sus.append(su_i)
                sc_i = pym.EinSum(expression="i,i->")(su_i, case.force)
                sc_i.tag = f"compliance:{case.name}"
                if abs(case.weight - 1.0) > 1e-12:
                    sc_i = pym.MathExpression(expression=f"{float(case.weight)}*inp0")(
                        sc_i
                    )
                scomps.append(sc_i)

            if len(scomps) == 1:
                sg0 = scomps[0]
            elif len(scomps) > 1:
                if p.load_aggregation == "worst_case":
                    ConcatCls = _make_concat_module()
                    combined = ConcatCls()(*scomps)
                    sg0 = pym.PNorm(p=float(p.load_pnorm_p))(combined)
                else:
                    expr = " + ".join(f"inp{i}" for i in range(len(scomps)))
                    sg0 = pym.MathExpression(expression=expr)(*scomps)
            else:
                # No loads — fabricate a zero compliance signal so the graph builds.
                sg0 = pym.MathExpression(expression="0*inp0")(sxphys)
            sg0.tag = "compliance"

            sg0_scaled = pym.Scaling(scaling=100.0)(sg0) if uses_structural else None

            svol = pym.EinSum(expression="i->")(sxphys)
            svol.tag = "volume"
            sg1 = pym.MathExpression(
                expression=f"10*(inp0/{domain.nel} - {total_vol_target})"
            )(svol)
            sg1.tag = "volume constraint"

            thermal_compliances: list[Any] = []
            if uses_thermal:
                sk = pym.MathExpression(
                    expression=(
                        f"{p.thermal_conductivity_min} + "
                        f"{p.thermal_conductivity - p.thermal_conductivity_min}"
                        f"*inp0^{p.thermal_penal}"
                    )
                )(sxphys)
                sP_raw = pym.AssemblePoisson(
                    domain=domain,
                    bc=thermal_boundary_nodes,
                    material_property=1.0,
                )(sk)
                sP = SparseToCSC()(sP_raw)
                for thermal_case in thermal_cases:
                    st_i = pym.LinSolve(symmetric=True, positive_definite=True)(
                        sP, thermal_case.heat
                    )
                    sth_i = pym.EinSum(expression="i,i->")(st_i, thermal_case.heat)
                    sth_i.tag = f"thermal_compliance:{thermal_case.name}"
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
                sobjective_raw, sobjective_scaled = sg0, sg0_scaled
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
                s_pn_stress = pym.PNorm(p=0.5 * float(p.stress_pnorm_p))(vm_sq_all)
                s_pn_stress.tag = "stress_pnorm_sq"

                sg_stress = pym.MathExpression(expression=f"inp0/{yield_sq} - 1.0")(
                    s_pn_stress
                )
                sg_stress.tag = "stress constraint"

        net.response()

        # ── iteration loop ────────────────────────────────────────────────
        # `sxphys` is the physical-density signal that drives the FEA: it is
        # the Heaviside-projected field when projection is on, otherwise
        # identical to `sxfilt`. The recovery + final report must use it too.
        result = TopologyOptVoxelResult(
            density=_physical_density_grid(np.asarray(sxphys.state, dtype=float)),
            design_density=(
                _physical_density_grid(np.asarray(sxphys.state, dtype=float))
                if is_level_set
                else _density_grid_from_state(sx.state, domain)
            ),
            active_target_volfrac=active_volfrac,
            min_source_volfrac=float(volume_budget["min_source_volfrac"]),
            passive_source_volfrac=(
                float(volume_budget["passive_source_sum"])
                / max(float(volume_budget["source_count"]), 1.0)
            ),
            optimizer_used=optimizer_choice,
            formulation_used=(
                "Level Set (Reaction-Diffusion)" if is_level_set else "Density (SIMP)"
            ),
            level_set_field=(
                _density_grid_from_state(sx.state, domain) if is_level_set else None
            ),
        )
        comp_hist: list[float] = []
        change_hist: list[float] = []

        objective_hist: list[float] = []
        thermal_hist: list[float] = []
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
            xmax = np.ones(domain.nel)
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

        mc = p.mc
        has_projections = (
            (mc.symmetry or "none").lower() != "none"
            or (mc.extrusion or "none").lower() != "none"
            or (mc.overhang_build_axis or "none").lower() != "none"
            or float(mc.max_member_size_voxels or 0.0) > 0.0
            or int(mc.pattern_repeat or 1) > 1
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
        min_iter = min(int(p.max_iter), max(20, 4 * patience))
        stall = 0
        prev_objective = None

        # β-continuation schedule for the Heaviside projection. β doubles
        # every `heaviside_beta_step_iters` iterations, capped at β_max.
        hv_beta = (
            float(p.heaviside_beta_init) if heaviside_module_ref is not None else 0.0
        )
        hv_beta_max = float(p.heaviside_beta_max)
        hv_step = max(1, int(p.heaviside_beta_step_iters))

        it, change = 0, 1.0
        while it < p.max_iter:
            if self.stop_requested:
                result.message = "Stopped by user"
                break

            it += 1
            x_old = sx.state.copy()
            physical_old = np.asarray(sxphys.state, dtype=float).copy()

            # Step β BEFORE this iteration's response so the optimiser sees
            # the same β it just took a step under, and the sensitivities are
            # consistent with the projected field used in the FEA.
            if heaviside_module_ref is not None and it > 1 and (it - 1) % hv_step == 0:
                hv_beta = min(hv_beta * 2.0, hv_beta_max)
                heaviside_module_ref.beta = hv_beta

            if is_level_set:
                net.reset()
                sobjective_scaled.sensitivity = 1.0
                net.sensitivity()
                field = _density_grid_from_state(sx.state, domain)
                gradient = _density_grid_from_state(
                    np.asarray(sx.sensitivity, dtype=float),
                    domain,
                )
                active_grid = _density_grid_from_state(
                    active_mask,
                    domain,
                ).astype(bool)
                passive_phi_grid = _density_grid_from_state(
                    passive_level_set,
                    domain,
                )
                updated_field = reaction_diffusion_level_set_update(
                    field,
                    gradient,
                    active_grid,
                    active_volfrac,
                    passive_phi=passive_phi_grid,
                    regularization=min(
                        0.30,
                        max(0.08, 0.08 * float(p.rmin)),
                    ),
                )
                sx.state = _density_3d_to_flat(updated_field, domain)
                sx.state[~active_mask] = passive_level_set[~active_mask]
            elif optimizer_choice == "OC":
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
                )
            elif optimizer_choice == "PGD":
                net.reset()
                sobjective_scaled.sensitivity = 1.0
                net.sensitivity()
                sx.state = _projected_gradient_update(
                    sx.state,
                    sx.sensitivity.copy(),
                    active_volfrac,
                    active_mask=active_mask,
                    passive_density=passive_density,
                    step=float(p.projected_gradient_step),
                )
            else:
                x_new, _, _ = mma.step(x=sx.state)
                x_new = np.asarray(x_new, dtype=float)
                # MMA already respects xmin=xmax for passive, but re-clamp to
                # guarantee bit-exact passive density and avoid drift.
                x_new[~active_mask] = passive_density[~active_mask]
                sx.state = x_new
                mma.iter += 1

            # ── Manufacturing projections (after density update, before FE) ─
            if has_projections:
                x3 = _density_grid_from_state(sx.state, domain)
                x3 = _apply_symmetry(x3, mc.symmetry)
                x3 = _apply_extrusion(x3, mc.extrusion)
                x3 = _apply_am_overhang(x3, mc.overhang_build_axis)
                x3 = _apply_max_member_size(
                    x3,
                    float(mc.max_member_size_voxels or 0.0),
                    float(mc.max_member_threshold or 0.6),
                )
                x3 = _apply_pattern_repeat(
                    x3,
                    int(mc.pattern_repeat or 1),
                    str(mc.pattern_axis or "y"),
                )
                proj = _density_3d_to_flat(x3, domain)
                # Re-clamp passive voxels — projections may have nudged them.
                proj[~active_mask] = passive_density[~active_mask]
                if minimum_mass_objective:
                    sx.state = proj
                else:
                    sx.state = _restore_active_volume(
                        proj,
                        active_mask,
                        active_volfrac,
                        passive_density=passive_density,
                    )

            net.response()

            comp_val = float(sg0.state)
            thermal_val = float(sg_thermal.state) if sg_thermal is not None else 0.0
            objective_val = float(sobjective_raw.state)
            objective_hist.append(objective_val)
            # Robust density change: the MEAN over active voxels, not the single
            # worst one, so a handful of oscillating boundary voxels can't veto
            # the stop.  Reported to the callback / change_history.
            delta = (
                np.abs(np.asarray(sxphys.state, dtype=float) - physical_old)
                if is_level_set
                else np.abs(sx.state - x_old)
            )
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
                    np.asarray(sxphys.state, dtype=float)
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
                obj_change = abs(objective_val - prev_objective) / max(
                    abs(objective_val), 1e-12
                )
                stress_feasible = not p.stress_constraint_enabled or (
                    bool(stress_hist)
                    and stress_hist[-1]
                    <= float(p.yield_stress) * (1.0 + max(obj_tol, 0.005))
                )
                if obj_change < obj_tol and change < density_gate and stress_feasible:
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
        result.design_density = (
            _physical_density_grid(np.asarray(sxphys.state, dtype=float))
            if is_level_set
            else _density_grid_from_state(sx.state, domain)
        )
        result.level_set_field = (
            _density_grid_from_state(sx.state, domain) if is_level_set else None
        )
        result.density = _physical_density_grid(np.asarray(sxphys.state, dtype=float))
        result.compliance_history = comp_hist
        result.change_history = change_hist
        result.stress_history = stress_hist
        result.thermal_compliance_history = thermal_hist
        result.objective_history = objective_hist
        return result
