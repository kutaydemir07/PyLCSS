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

INDUSTRIAL_STRESS_RELAXATION_Q = 0.5
INDUSTRIAL_STRESS_PNORM_P = 8.0
INDUSTRIAL_HEAVISIDE_ENABLED = True
INDUSTRIAL_HEAVISIDE_BETA_INIT = 1.0
INDUSTRIAL_HEAVISIDE_BETA_MAX = 16.0
INDUSTRIAL_HEAVISIDE_BETA_STEP_ITERS = 30
INDUSTRIAL_HEAVISIDE_ETA = 0.5


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
    # Optional source CAD/mesh volume mask, shape (nelx, nely, nelz).
    # True means designable material exists there; False is clamped to void.
    design_domain: Optional[np.ndarray] = None
    # Compliance minimization at a material budget, or true mass/volume
    # minimization subject to the stress constraint.
    objective_mode: str = 'compliance'  # 'compliance' | 'minimum_mass'
    # Phase 3 — stress constraint (P-norm aggregated von Mises ≤ yield).
    # When enabled the optimiser is forced to MMA (OC cannot handle a second
    # constraint beyond the volume budget). All load cases are aggregated
    # into a single PNorm so a hot-spot under any LC is penalised.
    stress_constraint_enabled: bool  = False
    yield_stress:              float = 1.0
    # Internal qp stress relaxation and P-norm aggregation policy. The user
    # enters the allowable stress; the solver owns these numerical defaults.
    stress_penalty:            float = INDUSTRIAL_STRESS_RELAXATION_Q
    stress_pnorm_p:            float = INDUSTRIAL_STRESS_PNORM_P

    # Three-field SIMP projection is an internal solver default. It drives
    # intermediate densities toward 0/1 without exposing continuation knobs.
    heaviside_enabled:         bool  = INDUSTRIAL_HEAVISIDE_ENABLED
    heaviside_beta_init:       float = INDUSTRIAL_HEAVISIDE_BETA_INIT
    heaviside_beta_max:        float = INDUSTRIAL_HEAVISIDE_BETA_MAX
    heaviside_beta_step_iters: int   = INDUSTRIAL_HEAVISIDE_BETA_STEP_ITERS
    heaviside_eta:             float = INDUSTRIAL_HEAVISIDE_ETA


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
    lower  = np.maximum(1e-3, x_act - move)
    upper  = np.minimum(1.0,  x_act + move)

    def _candidate(lmid: float) -> np.ndarray:
        be = np.maximum(-dc_act / lmid, 0.0)
        return np.clip(x_act * np.sqrt(be), lower, upper)

    # Bracket the Lagrange multiplier λ. Volume V(λ) is non-increasing in λ, and
    # |λ*| tracks the sensitivity scale (∝ material stiffness), so the old fixed
    # [0, 1e5] bracket silently fails whenever run() is fed real-unit E. Grow a
    # geometric bracket until V(l1) >= maxvol >= V(l2), then bisect with a
    # scale-invariant relative test (Sigmund 88-line) instead of an absolute one.
    l1 = l2 = 1e-9
    for _ in range(200):
        if float(np.sum(_candidate(l2))) <= maxvol:
            break
        l1, l2 = l2, l2 * 2.0
    for _ in range(200):
        if float(np.sum(_candidate(l1))) >= maxvol:
            break
        l1, l2 = l1 * 0.5, l1

    x_act_new = _candidate(l2)
    while (l2 - l1) > 1e-3 * (l1 + l2):
        lmid = 0.5 * (l1 + l2)
        x_act_new = _candidate(lmid)
        if float(np.sum(x_act_new)) > maxvol:
            l1 = lmid
        else:
            l2 = lmid

    xnew = x.copy()
    xnew[active_mask] = x_act_new
    if passive_density is not None:
        xnew[~active_mask] = passive_density[~active_mask]
    return xnew


def _restore_active_volume(
    x: np.ndarray,
    active_mask: np.ndarray,
    volfrac: float,
    passive_density: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Restore the active-design volume after non-volume-preserving projections."""
    xnew = np.asarray(x, dtype=float).copy()
    active = np.asarray(active_mask, dtype=bool)
    if passive_density is not None:
        xnew[~active] = np.asarray(passive_density, dtype=float)[~active]
    if not np.any(active):
        return xnew

    target = float(np.clip(volfrac, 1e-3, 1.0)) * float(np.sum(active))
    lo, hi = -1.0, 1.0
    x_act = xnew[active]
    tol = 1e-9 * max(target, 1.0)
    mid = 0.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        shifted = np.clip(x_act + mid, 1e-3, 1.0)
        cur = float(np.sum(shifted))
        if abs(cur - target) <= tol:
            break
        if cur < target:
            lo = mid
        else:
            hi = mid
    xnew[active] = np.clip(x_act + mid, 1e-3, 1.0)
    if passive_density is not None:
        xnew[~active] = np.asarray(passive_density, dtype=float)[~active]
    return xnew


def _volume_budget_from_masks(
    volfrac: float,
    active_mask: np.ndarray,
    passive_density: np.ndarray,
    source_mask: Optional[np.ndarray] = None,
    *,
    min_density: float = 1e-3,
) -> dict:
    """Translate a total source-domain volume target into an active budget.

    `volfrac` is user-facing, so it should mean material fraction of the source
    design domain including passive solid/void regions. Passive solid material
    is subtracted from the budget; the remainder is what active voxels may use.
    """
    active = np.asarray(active_mask, dtype=bool).reshape(-1)
    passive = np.asarray(passive_density, dtype=float).reshape(-1)
    if passive.size != active.size:
        passive = np.resize(passive, active.size)
    if source_mask is None:
        source = np.ones(active.size, dtype=bool)
    else:
        source = np.asarray(source_mask, dtype=bool).reshape(-1)
        if source.size != active.size:
            source = np.ones(active.size, dtype=bool)

    source_count = max(1, int(np.sum(source)))
    active_source = active & source
    passive_source = (~active) & source
    passive_outside = (~active) & (~source)

    n_active = int(np.sum(active_source))
    target_source_sum = float(np.clip(volfrac, min_density, 1.0)) * float(source_count)
    passive_source_sum = float(np.sum(passive[passive_source]))
    passive_outside_sum = float(np.sum(passive[passive_outside]))

    min_active_sum = float(min_density) * float(n_active)
    raw_active_sum = target_source_sum - passive_source_sum
    feasible_active_sum = float(np.clip(raw_active_sum, min_active_sum, float(n_active)))
    active_volfrac = (
        feasible_active_sum / float(n_active)
        if n_active > 0 else float(min_density)
    )
    source_total_sum = passive_source_sum + feasible_active_sum
    flat_total_target = source_total_sum + passive_outside_sum
    min_source_sum = passive_source_sum + min_active_sum

    return {
        "active_volfrac": float(np.clip(active_volfrac, min_density, 1.0)),
        "flat_total_target": float(flat_total_target),
        "source_total_target": float(source_total_sum),
        "source_count": float(source_count),
        "active_count": float(n_active),
        "passive_source_sum": float(passive_source_sum),
        "min_source_volfrac": float(min_source_sum / float(source_count)),
        "target_was_clamped": bool(abs(feasible_active_sum - raw_active_sum) > 1e-9),
    }



def _density_3d_to_flat(x_3d: np.ndarray, domain: Any) -> np.ndarray:
    """Inverse of `_density_grid_from_state` — write a (nelx,nely,nelz) grid back
    into a flat element vector using pyMOTO's `domain.elements` mapping."""
    flat = np.empty(domain.nel, dtype=float)
    flat[domain.elements] = np.asarray(x_3d, dtype=float)
    return flat


def _make_passive_clamp_module():
    import pymoto as pym

    class _PassiveClamp(pym.Module):
        def __init__(self, active_mask, passive_density):
            super().__init__()
            self.active_mask = np.asarray(active_mask, dtype=bool)
            self.passive_density = np.asarray(passive_density, dtype=float)

        def __call__(self, x):
            y = np.asarray(x, dtype=float).copy()
            y[~self.active_mask] = self.passive_density[~self.active_mask]
            return y

        def _sensitivity(self, dy):
            dx = np.asarray(dy, dtype=float).copy()
            dx[~self.active_mask] = 0.0
            return [dx]

    return _PassiveClamp


def _make_concat_module():
    """Concatenate N vectors of length nel into one length-N·nel vector.

    Used to aggregate per-load-case vm² fields into a single PNorm.
    Concatenation + PNorm is mathematically equivalent to a single PNorm
    over the union of all elemental stresses.
    """
    import pymoto as pym

    class _Concat(pym.Module):
        def __call__(self, *inputs):
            self._sizes = [int(np.asarray(x).size) for x in inputs]
            return np.concatenate([np.asarray(x, dtype=float).ravel() for x in inputs])

        def _sensitivity(self, dy):
            dy_flat = np.asarray(dy, dtype=float).ravel()
            out = []
            offset = 0
            for sz in self._sizes:
                out.append(dy_flat[offset:offset + sz].copy())
                offset += sz
            return out

    return _Concat


def _make_heaviside_module():
    """Build the smooth-Heaviside projection pyMOTO Module class.

    Three-field SIMP (Sigmund/Wang/Lazarov 2011): physical density =
    H_β(filtered density), with β stepped from ~1 → ~32 by the iteration
    loop. β is a *mutable attribute* so the loop can update it between
    `net.response()` calls without rebuilding the network.
    """
    import pymoto as pym

    class _HeavisideProjection(pym.Module):
        def __init__(self, beta: float = 1.0, eta: float = 0.5):
            super().__init__()
            self.beta = float(beta)
            self.eta = float(eta)

        def __call__(self, x):
            x_arr = np.asarray(x, dtype=float)
            self._x = x_arr
            beta = float(self.beta)
            eta = float(self.eta)
            if beta < 1e-6:
                return x_arr.copy()
            tanh_be = np.tanh(beta * eta)
            tanh_b1e = np.tanh(beta * (1.0 - eta))
            denom = tanh_be + tanh_b1e
            return (tanh_be + np.tanh(beta * (x_arr - eta))) / denom

        def _sensitivity(self, dy):
            dy_arr = np.asarray(dy, dtype=float)
            beta = float(self.beta)
            eta = float(self.eta)
            if beta < 1e-6:
                return [dy_arr.copy()]
            x = self._x
            tanh_be = np.tanh(beta * eta)
            tanh_b1e = np.tanh(beta * (1.0 - eta))
            denom = tanh_be + tanh_b1e
            dproj = beta * (1.0 - np.tanh(beta * (x - eta)) ** 2) / denom
            return [dy_arr * dproj]

    return _HeavisideProjection


def _make_sparse_to_csc_module():
    import pymoto as pym
    from scipy.sparse import csc_matrix, isspmatrix_csc

    class _SparseToCSC(pym.Module):
        """Convert sparse matrices to CSC before SciPy splu.

        This is mathematically an identity operation; it only changes sparse
        storage format so pyMOTO/SciPy does not warn during factorization.
        """

        def __call__(self, A):
            self._input_format = getattr(A, "format", None)

            if isspmatrix_csc(A):
                return A

            if hasattr(A, "tocsc"):
                return A.tocsc(copy=False)

            return csc_matrix(A)

        def _sensitivity(self, dA):
            # Format conversion is identity with respect to matrix entries.
            try:
                fmt = getattr(self, "_input_format", None)
                if fmt and hasattr(dA, "asformat"):
                    return [dA.asformat(fmt)]
            except Exception:
                pass
            return [dA]

    return _SparseToCSC


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
    objective_history:  List[float] = field(default_factory=list)
    change_history:     List[float] = field(default_factory=list)
    stress_history:     List[float] = field(default_factory=list)  # σ_pn per iteration
    n_iter:             int   = 0
    converged:          bool  = False
    message:            str   = ""
    active_target_volfrac: float = 0.0
    min_source_volfrac:    float = 0.0
    passive_source_volfrac: float = 0.0


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
        load_cases = [
            (name, weight, np.asarray(f_vec, dtype=float).reshape(-1))
            for name, weight, f_vec in load_cases
        ]
        active_mask, passive_density = self._assemble_passive_masks(domain, p.bc)
        n_active      = int(np.sum(active_mask))

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

        if not load_cases:
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
            )
            if callback is not None:
                callback(0, 0.0, 0.0, density_3d.copy())
            return result

        sx = pym.Signal("x", state=x0)
        minimum_mass_objective = str(p.objective_mode).lower() == 'minimum_mass'

        def _physical_density_grid(signal_state: np.ndarray) -> np.ndarray:
            physical = np.asarray(signal_state, dtype=float).copy()
            if minimum_mass_objective:
                physical[~active_mask] = passive_density[~active_mask]
            else:
                physical = _restore_active_volume(
                    physical,
                    active_mask,
                    active_volfrac,
                    passive_density=passive_density,
                )
            return _density_grid_from_state(physical, domain)

        # ── pyMOTO network ────────────────────────────────────────────────
        # Three-field SIMP when heaviside_enabled: raw sx → filter → passive
        # clamp → Heaviside(β) → SIMP. The Heaviside module exposes a mutable
        # `beta` attribute so the iteration loop can step it (continuation).
        heaviside_module_ref: Any = None
        with pym.Network() as net:
            sxfilt_raw = pym.DensityFilter(domain=domain, radius=p.rmin)(sx)
            PassiveClamp = _make_passive_clamp_module()
            sxfilt = PassiveClamp(active_mask, passive_density)(sxfilt_raw)
            if p.heaviside_enabled:
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
            else:
                sxphys = sxfilt
            sSIMP  = pym.MathExpression(
                expression=f"{p.Emin} + {p.E0 - p.Emin}*inp0^{p.penal}"
            )(sxphys)
            sK_raw = pym.AssembleStiffness(
                domain=domain, bc=boundary_dofs, poisson_ratio=p.nu
            )(sSIMP)
            SparseToCSC = _make_sparse_to_csc_module()
            sK = SparseToCSC()(sK_raw)

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
                sg0 = pym.MathExpression(expression="0*inp0")(sxphys)
            sg0.tag = "compliance"

            sg0_scaled = pym.Scaling(scaling=100.0)(sg0)
            sg0_scaled.tag = "objective"

            svol = pym.EinSum(expression="i->")(sxphys)
            svol.tag = "volume"
            sg1 = pym.MathExpression(
                expression=f"10*(inp0/{domain.nel} - {total_vol_target})"
            )(svol)
            sg1.tag = "volume constraint"
            if minimum_mass_objective:
                sobjective_raw = pym.MathExpression(
                    expression=f"inp0/{domain.nel}"
                )(svol)
                sobjective_raw.tag = "material fraction objective"
                sobjective_scaled = pym.Scaling(scaling=100.0)(sobjective_raw)
            else:
                sobjective_raw, sobjective_scaled = sg0, sg0_scaled

            # ── Phase 3: stress constraint (P-norm aggregated von Mises) ───
            # Aggregate vm² over ALL load cases into a single PNorm so a
            # hot-spot under any LC is penalised. Previously only LC[0] was
            # used — blind to peak stresses under secondary loadings.
            sg_stress  = None
            s_pn_stress = None
            if p.stress_constraint_enabled and sus:
                yield_sq = float(p.yield_stress) ** 2
                if yield_sq <= 0.0:
                    yield_sq = 1.0

                VonMisesCls = _make_vm_module()
                vm_sq_signals: List[Any] = []
                for lc_idx, su_i in enumerate(sus):
                    s_voigt_i = pym.Stress(
                        domain=domain,
                        e_modulus=float(p.E0),
                        poisson_ratio=float(p.nu),
                    )(su_i)
                    s_voigt_i.tag = f"stress_voigt:LC{lc_idx}"
                    vm_sq_i = VonMisesCls(
                        stress_penalty=float(p.stress_penalty)
                    )(s_voigt_i, sxphys)
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

                s_pn_stress = pym.PNorm(p=float(p.stress_pnorm_p))(vm_sq_all)
                s_pn_stress.tag = "stress_pnorm_sq"

                sg_stress = pym.MathExpression(
                    expression=f"inp0/{yield_sq} - 1.0"
                )(s_pn_stress)
                sg_stress.tag = "stress constraint"

        net.response()

        # ── iteration loop ────────────────────────────────────────────────
        # `sxphys` is the physical-density signal that drives the FEA: it is
        # the Heaviside-projected field when projection is on, otherwise
        # identical to `sxfilt`. The recovery + final report must use it too.
        result = TopologyOptVoxelResult(
            density=_physical_density_grid(np.asarray(sxphys.state, dtype=float)),
            design_density=_density_grid_from_state(sx.state, domain),
            active_target_volfrac=active_volfrac,
            min_source_volfrac=float(volume_budget["min_source_volfrac"]),
            passive_source_volfrac=(
                float(volume_budget["passive_source_sum"])
                / max(float(volume_budget["source_count"]), 1.0)
            ),
        )
        comp_hist:   List[float] = []
        change_hist: List[float] = []

        objective_hist: List[float] = []
        # Stress constraint requires multi-constraint MMA (OC can't handle it).
        optimizer_choice = p.optimizer.upper()
        if minimum_mass_objective and optimizer_choice != 'MMA':
            logger.info(
                "Minimum-mass objective requires MMA; forcing MMA (was '%s').",
                optimizer_choice,
            )
            optimizer_choice = 'MMA'

        if sg_stress is not None and optimizer_choice != 'MMA':
            logger.info(
                "Stress constraint enabled — forcing optimizer to MMA "
                "(was '%s').", optimizer_choice,
            )
            optimizer_choice = 'MMA'

        if optimizer_choice == 'MMA':
            # Keep passive DOFs in a tiny non-zero interval. pyMOTO's MMA
            # computes asymptotes from xmax-xmin, so exact equal bounds produce
            # divide-by-zero warnings before we get a chance to re-clamp.
            xmin = np.full(domain.nel, 1e-3)
            xmax = np.ones(domain.nel)
            passive = ~active_mask
            if np.any(passive):
                passive_values = passive_density[passive]
                eps_bound = 1e-6
                xmin[passive] = np.clip(passive_values - eps_bound, 1e-3, 1.0)
                xmax[passive] = np.clip(passive_values + eps_bound, 1e-3, 1.0)
                collapsed = xmax[passive] <= xmin[passive]
                if np.any(collapsed):
                    xlo = xmin[passive]
                    xhi = xmax[passive]
                    xhi[collapsed] = np.minimum(1.0, xlo[collapsed] + eps_bound)
                    xlo[collapsed & (xhi >= 1.0)] = np.maximum(
                        1e-3, xhi[collapsed & (xhi >= 1.0)] - eps_bound
                    )
                    xmin[passive] = xlo
                    xmax[passive] = xhi
            mma_responses = [sobjective_scaled, sg1]
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
        density_gate = max(2.0 * obj_tol, 0.015)
        patience     = max(1, int(getattr(p, 'patience', 5) or 5))
        min_iter     = min(int(p.max_iter), max(20, 4 * patience))
        stall        = 0
        prev_objective = None

        # β-continuation schedule for the Heaviside projection. β doubles
        # every `heaviside_beta_step_iters` iterations, capped at β_max.
        hv_beta = float(p.heaviside_beta_init) if heaviside_module_ref is not None else 0.0
        hv_beta_max = float(p.heaviside_beta_max)
        hv_step = max(1, int(p.heaviside_beta_step_iters))

        it, change = 0, 1.0
        while it < p.max_iter:
            if self.stop_requested:
                result.message = "Stopped by user"
                break

            it     += 1
            x_old   = sx.state.copy()

            # Step β BEFORE this iteration's response so the optimiser sees
            # the same β it just took a step under, and the sensitivities are
            # consistent with the projected field used in the FEA.
            if heaviside_module_ref is not None and it > 1 and (it - 1) % hv_step == 0:
                hv_beta = min(hv_beta * 2.0, hv_beta_max)
                heaviside_module_ref.beta = hv_beta

            if optimizer_choice == 'OC':
                net.reset()
                sg0.sensitivity = 1.0
                net.sensitivity()
                dc = sx.sensitivity.copy()
                sx.state = _oc_update(
                    sx.state, dc, active_volfrac,
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
            objective_val = float(sobjective_raw.state)
            objective_hist.append(objective_val)
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

            if callback is not None:
                density_3d = _physical_density_grid(np.asarray(sxphys.state, dtype=float))
                callback(it, comp_val, change, density_3d)

            # Objective-based early stop: relative compliance change below tol
            # for `patience` consecutive iterations, with the robust density
            # change also settled (a guard against stopping on a transient
            # compliance plateau while the topology is still reorganising).
            if prev_objective is not None:
                obj_change = abs(objective_val - prev_objective) / max(abs(objective_val), 1e-12)
                if obj_change < obj_tol and change < density_gate:
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
        result.design_density   = _density_grid_from_state(sx.state, domain)
        result.density          = _physical_density_grid(np.asarray(sxphys.state, dtype=float))
        result.compliance_history = comp_hist
        result.change_history   = change_hist
        result.stress_history   = stress_hist
        result.objective_history = objective_hist
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
        Explicit passive solids are not clipped by source-mask voxelisation;
        this preserves selected rings/sleeves even when the coarse source grid
        misses a thin wall. Void regions still run last and win in overlaps.
        """
        p = self.problem
        n_total = int(domain.nel)
        active_mask     = np.ones(n_total, dtype=bool)
        passive_density = np.zeros(n_total, dtype=float)

        source_active = None
        if p.design_domain is not None:
            try:
                source_grid = np.asarray(p.design_domain, dtype=bool)
                if source_grid.shape == (p.nelx, p.nely, p.nelz):
                    source_active = np.zeros(n_total, dtype=bool)
                    source_active[domain.elements] = source_grid
                    active_mask[~source_active] = False
                    passive_density[~source_active] = 1e-3
                else:
                    logger.warning(
                        "TopologyOptVoxelNode: ignoring design-domain mask "
                        "with shape %s; expected (%d, %d, %d).",
                        source_grid.shape, p.nelx, p.nely, p.nelz,
                    )
            except Exception:
                logger.warning(
                    "TopologyOptVoxelNode: failed to apply design-domain mask.",
                    exc_info=True,
                )

        def _voxel_slice(lo: float, hi: float, nmax: int) -> slice:
            lo_f, hi_f = sorted((float(lo), float(hi)))
            lo_f = max(0.0, min(1.0, lo_f))
            hi_f = max(0.0, min(1.0, hi_f))
            if abs(hi_f - lo_f) < 1e-12:
                if lo_f <= 0.0:
                    return slice(0, min(1, nmax))
                if hi_f >= 1.0:
                    return slice(max(0, nmax - 1), nmax)
                idx = max(0, min(nmax - 1, int(np.floor(lo_f * nmax))))
                return slice(idx, idx + 1)
            lo_i = int(np.floor(lo_f * nmax))
            hi_i = int(np.ceil(hi_f * nmax))
            lo_i = max(0, min(nmax, lo_i))
            hi_i = max(0, min(nmax, hi_i))
            if hi_i <= lo_i:
                hi_i = min(nmax, lo_i + 1)
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
            cylinder: Tuple[Any, ...],
        ) -> np.ndarray:
            nonlocal cylinder_grid
            if len(cylinder) < 6:
                return np.array([], dtype=int)
            axis, c0, c1, lo, hi, radius_a = cylinder[:6]
            radius_b = cylinder[6] if len(cylinder) > 6 else radius_a
            axis = str(axis or 'z').lower()
            lo, hi = sorted((float(lo), float(hi)))
            radius_a = float(radius_a)
            radius_b = float(radius_b)
            if radius_a <= 0.0 or radius_b <= 0.0:
                return np.array([], dtype=int)

            if cylinder_grid is None:
                xs = (np.arange(p.nelx, dtype=float) + 0.5) / max(p.nelx, 1)
                ys = (np.arange(p.nely, dtype=float) + 0.5) / max(p.nely, 1)
                zs = (np.arange(p.nelz, dtype=float) + 0.5) / max(p.nelz, 1)
                cylinder_grid = np.meshgrid(xs, ys, zs, indexing='ij')
            xx, yy, zz = cylinder_grid

            if axis == 'x':
                axial = xx
                dist2 = ((yy - float(c0)) / radius_a) ** 2 + ((zz - float(c1)) / radius_b) ** 2
            elif axis == 'y':
                axial = yy
                dist2 = ((xx - float(c0)) / radius_a) ** 2 + ((zz - float(c1)) / radius_b) ** 2
            else:
                axial = zz
                dist2 = ((xx - float(c0)) / radius_a) ** 2 + ((yy - float(c1)) / radius_b) ** 2

            mask = (axial >= lo) & (axial <= hi) & (dist2 <= 1.0)
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

