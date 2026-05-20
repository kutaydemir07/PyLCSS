# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.
"""SIMP voxel optimiser: problem/result types, the OC update, the von-Mises stress
module, and the pyMOTO-backed solver loop."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Tuple

import numpy as np

from .boundary_conditions import VoxelBC, LoadCase, ManufacturingConstraints
from .projections import (
    _apply_symmetry, _apply_extrusion, _apply_am_overhang,
    _apply_max_member_size, _apply_pattern_repeat,
)

logger = logging.getLogger(__name__)

@dataclass
class TopologyOptVoxelProblem:
    """All parameters needed to solve a 3-D voxel topology optimisation."""
    nelx:     int   = 30
    nely:     int   = 20
    nelz:     int   = 10
    E0:       float = 1.0
    Emin:     float = 1e-9
    nu:       float = 0.3
    penal:    float = 3.0
    volfrac:  float = 0.5
    rmin:     float = 1.5
    unitx:    float = 1.0
    unity:    float = 1.0
    unitz:    float = 1.0
    optimizer: str  = 'OC'   # 'OC' | 'MMA'
    max_iter: int   = 80
    tol:      float = 0.01
    # Early-stop is judged on the objective: stop once the relative compliance
    # change stays below `tol` for `patience` consecutive iterations.
    patience: int   = 5
    bc:       VoxelBC = field(default_factory=VoxelBC)
    mc:       ManufacturingConstraints = field(default_factory=ManufacturingConstraints)
    # Phase 3 — stress constraint (P-norm aggregated von Mises ≤ yield).
    # When enabled the optimiser is forced to MMA (OC cannot handle a second
    # constraint beyond the volume budget).  Only the FIRST load case feeds
    # the stress constraint; multi-LC aggregation is a future enhancement.
    stress_constraint_enabled: bool  = False
    yield_stress:              float = 1.0
    stress_penalty:            float = 1.0   # q in σ_relaxed = ρ^q · σ_linear
    stress_pnorm_p:            float = 8.0   # exponent for PNorm aggregation


# ---------------------------------------------------------------------------
# OC update (Sigmund 2001 / pyMOTO 69-line)
# ---------------------------------------------------------------------------

def _oc_update(
    x: np.ndarray,
    dc: np.ndarray,
    volfrac: float,
    active_mask: Optional[np.ndarray] = None,
    passive_density: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Optimality-criteria density update.

    When `active_mask` is given, the OC bisection runs over the design
    region only; `passive_density[~active_mask]` is re-applied at the end so
    must-solid / must-void voxels cannot be moved by the update.
    """
    if active_mask is None:
        active_mask = np.ones_like(x, dtype=bool)

    x_act  = x[active_mask]
    dc_act = dc[active_mask]
    n_act  = x_act.size
    if n_act == 0:
        xnew = x.copy()
        if passive_density is not None:
            xnew[~active_mask] = passive_density[~active_mask]
        return xnew

    maxvol = volfrac * n_act
    move   = 0.2
    l1, l2 = 0.0, 1e5
    x_act_new = x_act
    while l2 - l1 > 1e-4:
        lmid = 0.5 * (l1 + l2)
        be   = np.maximum(-dc_act / lmid, 0.0)
        x_act_new = np.clip(
            x_act * np.sqrt(be),
            np.maximum(1e-3, x_act - move),
            np.minimum(1.0,  x_act + move),
        )
        if np.sum(x_act_new) > maxvol:
            l1 = lmid
        else:
            l2 = lmid

    xnew = x.copy()
    xnew[active_mask] = x_act_new
    if passive_density is not None:
        xnew[~active_mask] = passive_density[~active_mask]
    return xnew



def _density_3d_to_flat(x_3d: np.ndarray, domain: Any) -> np.ndarray:
    """Inverse of `_density_grid_from_state` — write a (nelx,nely,nelz) grid back
    into a flat element vector using pyMOTO's `domain.elements` mapping."""
    flat = np.empty(domain.nel, dtype=float)
    flat[domain.elements] = np.asarray(x_3d, dtype=float)
    return flat


# ---------------------------------------------------------------------------
# Stress: SIMP-relaxed von Mises² as a pyMOTO Module
# ---------------------------------------------------------------------------

# Quadratic-form matrix for von Mises in 3-D Voigt notation
# s = (σ_xx, σ_yy, σ_zz, σ_yz, σ_xz, σ_xy) and σ_vm² = sᵀ A s.
_VM_A = np.array([
    [ 1.0, -0.5, -0.5, 0.0, 0.0, 0.0],
    [-0.5,  1.0, -0.5, 0.0, 0.0, 0.0],
    [-0.5, -0.5,  1.0, 0.0, 0.0, 0.0],
    [ 0.0,  0.0,  0.0, 3.0, 0.0, 0.0],
    [ 0.0,  0.0,  0.0, 0.0, 3.0, 0.0],
    [ 0.0,  0.0,  0.0, 0.0, 0.0, 3.0],
])


def _make_vm_module():
    """Build the relaxed-von-Mises pyMOTO Module class on first use.

    Defined as a factory so this file imports fine when pyMOTO is absent
    (the rest of the module stays usable for headless data manipulation).
    """
    import pymoto as pym

    class _VonMisesSquaredRelaxed(pym.Module):
        """SIMP-relaxed von Mises² per element.

        Inputs:
            s   — Voigt stress, shape (6, nel)
            rho — element density, shape (nel,)
        Output:
            vm_sq — σ_vm² per element, shape (nel,)
                    with σ_relaxed = ρ^q · σ_linear and vm_sq = σ_relaxedᵀ A σ_relaxed

        Sensitivities:
            ∂vm_sq/∂s_e = 2 · ρ_e^(2q) · A · s_e
            ∂vm_sq/∂ρ_e = 2q · ρ_e^(2q-1) · s_eᵀ A s_e
        """

        def __init__(self, stress_penalty: float = 1.0):
            super().__init__()
            self.q = float(stress_penalty)

        def __call__(self, s, rho):
            s_arr   = np.asarray(s,   dtype=float)
            rho_arr = np.asarray(rho, dtype=float)
            self._s   = s_arr
            self._rho = rho_arr
            rho_q     = rho_arr ** self.q                       # (nel,)
            s_relaxed = s_arr * rho_q[np.newaxis, :]            # (6, nel)
            vm_sq     = np.einsum('ij,ie,je->e', _VM_A, s_relaxed, s_relaxed)
            return vm_sq

        def _sensitivity(self, dvm_sq):
            dvm = np.asarray(dvm_sq, dtype=float)               # (nel,)
            s, rho = self._s, self._rho

            # ∂vm²/∂s = 2 · ρ^(2q) · A · s_lin
            A_s    = _VM_A @ s                                  # (6, nel)
            rho_2q = rho ** (2.0 * self.q)
            ds = 2.0 * (rho_2q[np.newaxis, :] * A_s) * dvm[np.newaxis, :]

            # ∂vm²/∂ρ = 2q · ρ^(2q-1) · s_linᵀ A s_lin
            vm_sq_lin = np.einsum('ij,ie,je->e', _VM_A, s, s)
            with np.errstate(divide='ignore', invalid='ignore'):
                drho_raw = np.where(
                    rho > 1e-12,
                    2.0 * self.q * (rho ** (2.0 * self.q - 1.0)) * vm_sq_lin,
                    0.0,
                )
            drho = drho_raw * dvm
            return [ds, drho]

    return _VonMisesSquaredRelaxed


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass
class TopologyOptVoxelResult:
    density:            np.ndarray
    design_density:     Optional[np.ndarray] = None
    compliance_history: List[float] = field(default_factory=list)
    change_history:     List[float] = field(default_factory=list)
    stress_history:     List[float] = field(default_factory=list)  # σ_pn per iteration
    n_iter:             int   = 0
    converged:          bool  = False
    message:            str   = ""


def _density_grid_from_state(x: np.ndarray, domain: Any) -> np.ndarray:
    """Map pyMOTO's flat element numbering to density[ix, iy, iz]."""
    return np.asarray(x, dtype=float)[domain.elements]



class TopologyOptVoxelSolver:
    """3-D SIMP topology optimiser backed by pyMOTO."""

    def __init__(self, problem: TopologyOptVoxelProblem):
        self.problem        = problem
        self.stop_requested = False

    def stop(self):
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
        import pymoto as pym

        p = self.problem

        # ── domain ────────────────────────────────────────────────────────
        domain = pym.VoxelDomain(
            p.nelx, p.nely, p.nelz,
            unitx=p.unitx,
            unity=p.unity,
            unitz=p.unitz,
        )
        ndof   = domain.nnodes * domain.dim  # 3 DOFs per node

        # ── supports, loads, passive regions ──────────────────────────────
        boundary_dofs = self._assemble_supports(domain, p.bc)
        load_cases    = self._assemble_load_cases(domain, p.bc, ndof)
        active_mask, passive_density = self._assemble_passive_masks(domain, p.bc)
        n_active      = int(np.sum(active_mask))

        if not load_cases:
            logger.warning(
                "TopologyOptVoxelNode: no load cases produced a non-zero force "
                "vector — optimisation will return a trivial (uniform) result."
            )

        # ── initial state — design = volfrac, passive = clamp value ───────
        x0 = np.ones(domain.nel) * p.volfrac
        x0[~active_mask] = passive_density[~active_mask]
        sx = pym.Signal("x", state=x0)

        # Volume budget incorporates passive solid mass so `volfrac` keeps its
        # familiar meaning of "fraction of the DESIGN region kept as material".
        passive_solid_vol = float(np.sum(passive_density[~active_mask] >= 0.5))
        total_vol_target  = (
            (p.volfrac * float(n_active) + passive_solid_vol) / float(domain.nel)
        )

        # ── pyMOTO network ────────────────────────────────────────────────
        with pym.Network() as net:
            sxfilt = pym.DensityFilter(domain=domain, radius=p.rmin)(sx)
            sSIMP  = pym.MathExpression(
                expression=f"{p.Emin} + {p.E0 - p.Emin}*inp0^{p.penal}"
            )(sxfilt)
            sK  = pym.AssembleStiffness(
                domain=domain, bc=boundary_dofs, poisson_ratio=p.nu
            )(sSIMP)

            # Per-load-case compliance, then weighted sum → objective.
            # pym.Scaling normalises (NOT multiplies) so we use MathExpression
            # to apply the true scalar weight to each compliance term.
            sus:    List[Any] = []
            scomps: List[Any] = []
            for (lc_name, weight, f_vec) in load_cases:
                su_i = pym.LinSolve(symmetric=True, positive_definite=True)(sK, f_vec)
                sus.append(su_i)
                sc_i = pym.EinSum(expression="i,i->")(su_i, f_vec)
                sc_i.tag = f"compliance:{lc_name}"
                if abs(weight - 1.0) > 1e-12:
                    sc_i = pym.MathExpression(
                        expression=f"{float(weight)}*inp0"
                    )(sc_i)
                scomps.append(sc_i)

            if len(scomps) == 1:
                sg0 = scomps[0]
            elif len(scomps) > 1:
                expr = " + ".join(f"inp{i}" for i in range(len(scomps)))
                sg0 = pym.MathExpression(expression=expr)(*scomps)
            else:
                # No loads — fabricate a zero compliance signal so the graph builds.
                sg0 = pym.MathExpression(expression="0*inp0")(sxfilt)
            sg0.tag = "compliance"

            sg0_scaled = pym.Scaling(scaling=100.0)(sg0)
            sg0_scaled.tag = "objective"

            svol = pym.EinSum(expression="i->")(sxfilt)
            svol.tag = "volume"
            sg1 = pym.MathExpression(
                expression=f"10*(inp0/{domain.nel} - {total_vol_target})"
            )(svol)
            sg1.tag = "volume constraint"

            # ── Phase 3: stress constraint (P-norm aggregated von Mises) ───
            sg_stress  = None
            s_pn_stress = None
            if p.stress_constraint_enabled and sus:
                if len(sus) > 1:
                    logger.warning(
                        "Stress constraint: using only LC '%s' for stress "
                        "(multi-LC stress aggregation is a future enhancement).",
                        load_cases[0][0],
                    )
                yield_sq = float(p.yield_stress) ** 2
                if yield_sq <= 0.0:
                    yield_sq = 1.0

                s_voigt = pym.Stress(
                    domain=domain,
                    e_modulus=float(p.E0),
                    poisson_ratio=float(p.nu),
                )(sus[0])
                s_voigt.tag = "stress_voigt"

                VonMisesCls = _make_vm_module()
                vm_sq = VonMisesCls(stress_penalty=float(p.stress_penalty))(s_voigt, sxfilt)
                vm_sq.tag = "vm_squared"

                s_pn_stress = pym.PNorm(p=float(p.stress_pnorm_p))(vm_sq)
                s_pn_stress.tag = "stress_pnorm_sq"

                sg_stress = pym.MathExpression(
                    expression=f"inp0/{yield_sq} - 1.0"
                )(s_pn_stress)
                sg_stress.tag = "stress constraint"

        net.response()

        # ── iteration loop ────────────────────────────────────────────────
        result = TopologyOptVoxelResult(
            density=_density_grid_from_state(
                np.asarray(sxfilt.state, dtype=float), domain
            ),
            design_density=_density_grid_from_state(sx.state, domain),
        )
        comp_hist:   List[float] = []
        change_hist: List[float] = []

        # Stress constraint requires multi-constraint MMA (OC can't handle it).
        optimizer_choice = p.optimizer.upper()
        if sg_stress is not None and optimizer_choice != 'MMA':
            logger.info(
                "Stress constraint enabled — forcing optimizer to MMA "
                "(was '%s').", optimizer_choice,
            )
            optimizer_choice = 'MMA'

        if optimizer_choice == 'MMA':
            # Passive DOFs pinned via xmin == xmax == passive value.
            xmin = np.full(domain.nel, 1e-3)
            xmax = np.ones(domain.nel)
            xmin[~active_mask] = passive_density[~active_mask]
            xmax[~active_mask] = passive_density[~active_mask]
            mma_responses = [sg0_scaled, sg1]
            if sg_stress is not None:
                mma_responses.append(sg_stress)
            mma = pym.MMA(
                sx, mma_responses, net,
                xmin=xmin,
                xmax=xmax,
                move=0.2,
                verbosity=0,
            )

        mc = p.mc
        has_projections = (
            (mc.symmetry            or 'none').lower() != 'none'
            or (mc.extrusion        or 'none').lower() != 'none'
            or (mc.overhang_build_axis or 'none').lower() != 'none'
            or float(mc.max_member_size_voxels or 0.0) > 0.0
            or int(mc.pattern_repeat or 1) > 1
        )

        stress_hist: List[float] = []

        # Early-stop is judged on the OBJECTIVE (relative compliance change),
        # sustained for `patience` iterations, with a loose robust density-change
        # gate.  Watching max|Δρ| alone never trips: a single voxel oscillating
        # by the OC move limit near ρ≈0.5 pins it above tol while the design has
        # long since settled, so the run always burns through to max_iter.
        obj_tol      = float(p.tol)
        density_gate = max(10.0 * obj_tol, 0.05)
        patience     = max(1, int(getattr(p, 'patience', 5) or 5))
        stall        = 0
        prev_comp    = None

        it, change = 0, 1.0
        while it < p.max_iter:
            if self.stop_requested:
                result.message = "Stopped by user"
                break

            it     += 1
            x_old   = sx.state.copy()

            if optimizer_choice == 'OC':
                net.reset()
                sg0.sensitivity = 1.0
                net.sensitivity()
                dc = sx.sensitivity.copy()
                sx.state = _oc_update(
                    sx.state, dc, p.volfrac,
                    active_mask=active_mask,
                    passive_density=passive_density,
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
                    str(mc.pattern_axis or 'y'),
                )
                proj = _density_3d_to_flat(x3, domain)
                # Re-clamp passive voxels — projections may have nudged them.
                proj[~active_mask] = passive_density[~active_mask]
                sx.state = proj

            net.response()

            comp_val = float(sg0.state)
            # Robust density change: the MEAN over active voxels, not the single
            # worst one, so a handful of oscillating boundary voxels can't veto
            # the stop.  Reported to the callback / change_history.
            delta = np.abs(sx.state - x_old)
            change = (float(np.mean(delta[active_mask])) if np.any(active_mask)
                      else float(np.mean(delta)))
            comp_hist.append(comp_val)
            change_hist.append(change)
            if s_pn_stress is not None:
                # s_pn_stress.state is the P-norm of vm² → sqrt gives σ_pn
                pn_val = float(np.asarray(s_pn_stress.state).flatten()[0])
                stress_hist.append(float(np.sqrt(max(pn_val, 0.0))))
            result.n_iter = it

            density_3d = _density_grid_from_state(
                np.asarray(sxfilt.state, dtype=float), domain
            )
            if callback is not None:
                callback(it, comp_val, change, density_3d.copy())

            # Objective-based early stop: relative compliance change below tol
            # for `patience` consecutive iterations, with the robust density
            # change also settled (a guard against stopping on a transient
            # compliance plateau while the topology is still reorganising).
            if prev_comp is not None:
                obj_change = abs(comp_val - prev_comp) / max(abs(comp_val), 1e-12)
                if obj_change < obj_tol and change < density_gate:
                    stall += 1
                else:
                    stall = 0
                if stall >= patience:
                    result.converged = True
                    result.message = (
                        f"Converged in {it} iterations "
                        f"(rel. compliance change < {obj_tol:.1e} for "
                        f"{patience} iters; mean |d_rho| = {change:.2e})"
                    )
                    break
            prev_comp = comp_val

        if not result.message:
            result.message = f"Maximum iterations ({p.max_iter}) reached"

        net.response()
        result.design_density   = _density_grid_from_state(sx.state, domain)
        result.density          = _density_grid_from_state(
            np.asarray(sxfilt.state, dtype=float), domain
        )
        result.compliance_history = comp_hist
        result.change_history   = change_hist
        result.stress_history   = stress_hist
        return result

    # ── BC assembly ───────────────────────────────────────────────────────

    def _assemble_supports(self, domain: Any, bc: VoxelBC) -> np.ndarray:
        """Build the global fixed-DOF array from VoxelBC face/box supports.

        VoxelDomain node grid (3-D):
            domain.nodes[ix, iy, iz]  →  scalar node number
            ix = 0 … nelx  (left  → right,  X)
            iy = 0 … nely  (bottom → top,   Y)
            iz = 0 … nelz  (front → back,   Z)
        DOF layout:  x-DOF = 3*n,  y-DOF = 3*n+1,  z-DOF = 3*n+2
        """
        p   = self.problem
        dim = domain.dim
        fixed: List[int] = []

        def _add_face_dofs(nodes_2d: np.ndarray, dofs: List[int]) -> None:
            fixed.extend(
                domain.get_dofnumber(nodes_2d.flatten(), dofs, ndof=dim)
                .flatten().astype(int).tolist()
            )

        def _node_slice(lo: float, hi: float, nmax: int) -> slice:
            lo_i = int(round(float(lo) * nmax))
            hi_i = int(round(float(hi) * nmax))
            lo_i, hi_i = sorted((lo_i, hi_i))
            lo_i = max(0, min(nmax, lo_i))
            hi_i = max(0, min(nmax, hi_i))
            return slice(lo_i, hi_i + 1)

        if bc.fixed_left_face_dofs:
            _add_face_dofs(domain.nodes[0,      :,      :], bc.fixed_left_face_dofs)
        if bc.fixed_right_face_dofs:
            _add_face_dofs(domain.nodes[p.nelx, :,      :], bc.fixed_right_face_dofs)
        if bc.fixed_top_face_dofs:
            _add_face_dofs(domain.nodes[:,      p.nely, :], bc.fixed_top_face_dofs)
        if bc.fixed_bottom_face_dofs:
            _add_face_dofs(domain.nodes[:,      0,      :], bc.fixed_bottom_face_dofs)
        if bc.fixed_front_face_dofs:
            _add_face_dofs(domain.nodes[:,      :,      0], bc.fixed_front_face_dofs)
        if bc.fixed_back_face_dofs:
            _add_face_dofs(domain.nodes[:,      :,      p.nelz], bc.fixed_back_face_dofs)

        for xmin, xmax, ymin, ymax, zmin, zmax, dofs in bc.fixed_boxes:
            if not dofs:
                continue
            nodes = domain.nodes[
                _node_slice(xmin, xmax, p.nelx),
                _node_slice(ymin, ymax, p.nely),
                _node_slice(zmin, zmax, p.nelz),
            ]
            _add_face_dofs(nodes, dofs)

        return np.unique(fixed).astype(int)

    def _apply_load_case_to_force(
        self,
        domain: Any,
        lc: LoadCase,
        f: np.ndarray,
    ) -> None:
        """Populate the force vector with point, patch, and face loads."""
        p   = self.problem
        dim = domain.dim

        def _node_slice(lo: float, hi: float, nmax: int) -> slice:
            lo_i = int(round(float(lo) * nmax))
            hi_i = int(round(float(hi) * nmax))
            lo_i, hi_i = sorted((lo_i, hi_i))
            lo_i = max(0, min(nmax, lo_i))
            hi_i = max(0, min(nmax, hi_i))
            if hi_i <= lo_i:
                hi_i = min(nmax, lo_i + 1)
            return slice(lo_i, hi_i + 1)

        for (ix_frac, iy_frac, iz_frac, fx, fy, fz) in lc.point_forces:
            ix = int(round(ix_frac * p.nelx))
            iy = int(round(iy_frac * p.nely))
            iz = int(round(iz_frac * p.nelz))
            n  = int(domain.nodes[ix, iy, iz])
            f[domain.get_dofnumber(n, 0, ndof=dim)] += fx
            f[domain.get_dofnumber(n, 1, ndof=dim)] += fy
            f[domain.get_dofnumber(n, 2, ndof=dim)] += fz

        for (x0, x1, y0, y1, z0, z1, fx, fy, fz) in lc.box_forces:
            nodes = np.asarray(domain.nodes[
                _node_slice(x0, x1, p.nelx),
                _node_slice(y0, y1, p.nely),
                _node_slice(z0, z1, p.nelz),
            ], dtype=int).ravel()
            if nodes.size == 0:
                continue
            nodes = np.unique(nodes)
            share = 1.0 / float(nodes.size)
            for n in nodes:
                f[domain.get_dofnumber(int(n), 0, ndof=dim)] += fx * share
                f[domain.get_dofnumber(int(n), 1, ndof=dim)] += fy * share
                f[domain.get_dofnumber(int(n), 2, ndof=dim)] += fz * share

        _face_nodes = {
            'left':   domain.nodes[0,      :,      :].flatten(),
            'right':  domain.nodes[p.nelx, :,      :].flatten(),
            'top':    domain.nodes[:,      p.nely, :].flatten(),
            'bottom': domain.nodes[:,      0,      :].flatten(),
            'front':  domain.nodes[:,      :,      0].flatten(),
            'back':   domain.nodes[:,      :,      p.nelz].flatten(),
        }
        for (face, fx_per, fy_per, fz_per) in lc.distributed_forces:
            nodes   = _face_nodes.get(face, np.array([], dtype=int))
            n_nodes = len(nodes)
            if n_nodes == 0:
                continue
            for n in nodes:
                f[domain.get_dofnumber(int(n), 0, ndof=dim)] += fx_per / n_nodes
                f[domain.get_dofnumber(int(n), 1, ndof=dim)] += fy_per / n_nodes
                f[domain.get_dofnumber(int(n), 2, ndof=dim)] += fz_per / n_nodes

    def _assemble_load_cases(
        self,
        domain: Any,
        bc: VoxelBC,
        ndof: int,
    ) -> List[Tuple[str, float, np.ndarray]]:
        """Return list of (name, weight, f).  Legacy bc.point_forces / bc.distributed_forces
        become a single synthesised 'LC1' when bc.load_cases is empty."""
        cases: List[Tuple[str, float, np.ndarray]] = []

        for lc in bc.load_cases:
            f = np.zeros(ndof)
            self._apply_load_case_to_force(domain, lc, f)
            if np.any(f):
                cases.append((str(lc.name or "LC"), float(lc.weight or 1.0), f))

        if not cases:
            legacy_lc = LoadCase(
                name="LC1",
                weight=1.0,
                point_forces=list(bc.point_forces),
                box_forces=list(bc.box_forces),
                distributed_forces=list(bc.distributed_forces),
            )
            f = np.zeros(ndof)
            self._apply_load_case_to_force(domain, legacy_lc, f)
            if np.any(f):
                cases.append(("LC1", 1.0, f))

        return cases

    def _assemble_passive_masks(
        self,
        domain: Any,
        bc: VoxelBC,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return (active_mask, passive_density) over the flat element vector.

        Voxels in `bc.solid_boxes` are clamped to ρ=1 (must-keep material).
        Voxels in `bc.void_boxes`  are clamped to ρ=1e-3 (must-be-empty).
        Where boxes overlap, void wins (final pass).
        """
        p = self.problem
        n_total = int(domain.nel)
        active_mask     = np.ones(n_total, dtype=bool)
        passive_density = np.zeros(n_total, dtype=float)

        def _voxel_slice(lo: float, hi: float, nmax: int) -> slice:
            lo_i = int(round(float(lo) * nmax))
            hi_i = int(round(float(hi) * nmax))
            lo_i, hi_i = sorted((lo_i, hi_i))
            lo_i = max(0, min(nmax, lo_i))
            hi_i = max(0, min(nmax, hi_i))
            return slice(lo_i, hi_i)

        def _flat_indices(box: Tuple[float, float, float, float, float, float]) -> np.ndarray:
            x0, x1, y0, y1, z0, z1 = box
            sub = domain.elements[
                _voxel_slice(x0, x1, p.nelx),
                _voxel_slice(y0, y1, p.nely),
                _voxel_slice(z0, z1, p.nelz),
            ]
            return np.asarray(sub, dtype=int).ravel()

        cylinder_grid = None

        def _cylinder_indices(
            cylinder: Tuple[str, float, float, float, float, float],
        ) -> np.ndarray:
            nonlocal cylinder_grid
            axis, c0, c1, lo, hi, radius = cylinder
            axis = str(axis or 'z').lower()
            lo, hi = sorted((float(lo), float(hi)))
            r2 = float(radius) ** 2
            if r2 <= 0.0:
                return np.array([], dtype=int)

            if cylinder_grid is None:
                xs = (np.arange(p.nelx, dtype=float) + 0.5) / max(p.nelx, 1)
                ys = (np.arange(p.nely, dtype=float) + 0.5) / max(p.nely, 1)
                zs = (np.arange(p.nelz, dtype=float) + 0.5) / max(p.nelz, 1)
                cylinder_grid = np.meshgrid(xs, ys, zs, indexing='ij')
            xx, yy, zz = cylinder_grid

            if axis == 'x':
                axial = xx
                dist2 = (yy - float(c0)) ** 2 + (zz - float(c1)) ** 2
            elif axis == 'y':
                axial = yy
                dist2 = (xx - float(c0)) ** 2 + (zz - float(c1)) ** 2
            else:
                axial = zz
                dist2 = (xx - float(c0)) ** 2 + (yy - float(c1)) ** 2

            mask = (axial >= lo) & (axial <= hi) & (dist2 <= r2)
            return np.asarray(domain.elements[mask], dtype=int).ravel()

        for box in bc.solid_boxes:
            idx = _flat_indices(box)
            if idx.size == 0:
                continue
            active_mask[idx] = False
            passive_density[idx] = 1.0

        for cylinder in getattr(bc, 'solid_cylinders', []):
            idx = _cylinder_indices(cylinder)
            if idx.size == 0:
                continue
            active_mask[idx] = False
            passive_density[idx] = 1.0

        for box in bc.void_boxes:
            idx = _flat_indices(box)
            if idx.size == 0:
                continue
            active_mask[idx] = False
            passive_density[idx] = 1e-3

        for cylinder in getattr(bc, 'void_cylinders', []):
            idx = _cylinder_indices(cylinder)
            if idx.size == 0:
                continue
            active_mask[idx] = False
            passive_density[idx] = 1e-3

        return active_mask, passive_density


