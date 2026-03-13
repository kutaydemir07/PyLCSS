# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.
"""Low-level explicit FEM / crash-mechanics helpers.

All pure-NumPy + SciPy helper functions used by the crash solver nodes:
strain-displacement matrices, plasticity return mapping, contact mechanics,
CFL stability estimates, and element pre-computation routines.
"""
"""
Crash / Impact Simulation Nodes for PyLCSS CAD.

Implements explicit transient FEA using the Central Difference (Leapfrog)
time integration scheme with J2 von Mises plasticity (isotropic hardening,
radial return algorithm) and element deletion at fracture.

Workflow:
    CrashMaterial ──┐
    Mesh ───────────┼──► CrashSolver ──► crash_results (deformed mesh + fields)
    Constraint(s) ──┤
    ImpactCondition ┘
"""

import numpy as np
import logging

from pylcss.cad.core.base_node import CadQueryNode
from scipy.spatial import cKDTree
from pylcss.cad.nodes.fem import MATERIAL_DATABASE, lam_lame

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Low-level solid mechanics helpers
# ─────────────────────────────────────────────────────────────────────────────

def compute_tet_B(nodes):
    """
    Compute the constant strain-displacement B matrix for a linear
    tetrahedron (T4) and its volume.

    Reference: Bathe, "Finite Element Procedures", Section 5.4.

    Args:
        nodes: (4, 3) float array – coordinates of the four tet nodes.

    Returns:
        B   : (6, 12) strain-displacement matrix in Voigt notation
              [ ε_xx, ε_yy, ε_zz, γ_xy, γ_yz, γ_xz ]
        vol : scalar element volume (mm³ in the mm/tonne/N unit system).
    """
    T = np.ones((4, 4))
    T[:, 1:] = nodes           # each row: [1, x_i, y_i, z_i]
    try:
        T_inv = np.linalg.inv(T)
        vol = abs(np.linalg.det(T)) / 6.0
    except np.linalg.LinAlgError:
        return np.zeros((6, 12)), 0.0

    dNdx = T_inv[1, :]         # shape (4,) – ∂N_i / ∂x
    dNdy = T_inv[2, :]
    dNdz = T_inv[3, :]

    B = np.zeros((6, 12))
    for i in range(4):
        j = 3 * i
        B[0, j]     = dNdx[i]                   # ε_xx
        B[1, j + 1] = dNdy[i]                   # ε_yy
        B[2, j + 2] = dNdz[i]                   # ε_zz
        B[3, j]     = dNdy[i]; B[3, j + 1] = dNdx[i]   # γ_xy
        B[4, j + 1] = dNdz[i]; B[4, j + 2] = dNdy[i]   # γ_yz
        B[5, j]     = dNdz[i]; B[5, j + 2] = dNdx[i]   # γ_xz

    return B, vol


def build_D_matrix(E, nu):
    """
    Isotropic linear elastic constitutive matrix D (6×6) in Voigt notation.
    Units: consistent with E in MPa → stress in MPa.
    """
    c = E / ((1.0 + nu) * (1.0 - 2.0 * nu))
    D = np.zeros((6, 6))
    D[0, 0] = D[1, 1] = D[2, 2] = c * (1.0 - nu)
    D[0, 1] = D[0, 2] = D[1, 0] = D[1, 2] = D[2, 0] = D[2, 1] = c * nu
    D[3, 3] = D[4, 4] = D[5, 5] = c * (1.0 - 2.0 * nu) / 2.0
    return D


def radial_return(sigma_trial, eps_p_eq, sigma_y0, H, mu):
    """
    Radial return algorithm for J2 von Mises plasticity with
    linear isotropic hardening.

    Args:
        sigma_trial : (6,) trial stress vector [σ_xx, σ_yy, σ_zz, σ_xy, σ_yz, σ_xz]
        eps_p_eq    : scalar equivalent plastic strain (accumulated)
        sigma_y0    : initial yield stress  [MPa]
        H           : isotropic hardening modulus  [MPa]
        mu          : shear modulus = E / (2*(1+ν))  [MPa]

    Returns:
        sigma       : (6,) corrected Cauchy stress
        eps_p_eq_new: updated equivalent plastic strain
        delta_lam   : plastic consistency parameter increment
        is_plastic  : bool
    """
    # Hydrostatic pressure
    p = (sigma_trial[0] + sigma_trial[1] + sigma_trial[2]) / 3.0

    # Deviatoric stress
    s = sigma_trial.copy()
    s[0] -= p; s[1] -= p; s[2] -= p

    # s : s (double contraction, Voigt convention – off-diagonal factor of 2)
    s_sq = (s[0]**2 + s[1]**2 + s[2]**2
            + 2.0 * (s[3]**2 + s[4]**2 + s[5]**2))
    sigma_eq = np.sqrt(max(1.5 * s_sq, 0.0))    # √(3/2 s:s)

    # Current yield stress from isotropic hardening
    sigma_y = sigma_y0 + H * eps_p_eq

    # Yield function
    f = sigma_eq - sigma_y
    if f <= 0.0 or sigma_eq < 1e-12:
        return sigma_trial, eps_p_eq, 0.0, False

    # Plastic consistency parameter: Δλ = f / (3μ + H)
    delta_lam = f / (3.0 * mu + H)

    # Scale-back factor
    scale = 1.0 - (3.0 * mu * delta_lam) / sigma_eq

    # Corrected stress
    sigma = s * scale
    sigma[0] += p; sigma[1] += p; sigma[2] += p

    # Updated equivalent plastic strain
    eps_p_eq_new = eps_p_eq + delta_lam

    return sigma, eps_p_eq_new, delta_lam, True


def assemble_lumped_mass(mesh, rho):
    """
    Assemble a diagonal (lumped) mass matrix by row-sum technique (vectorized).

    For each tet: element mass = ρ·V is distributed equally to its 4 nodes.

    Args:
        mesh : skfem MeshTet
        rho  : density  [tonne/mm³]

    Returns:
        M_diag : (3·N_nodes,) array – one entry per DOF
    """
    N    = mesh.p.shape[1]
    pts  = mesh.p.T               # (N_nodes, 3)
    conn = mesh.t.T               # (N_elem, 4)

    # Build T matrices for all elements at once
    T = np.ones((conn.shape[0], 4, 4))
    T[:, :, 1:] = pts[conn]       # (N_elem, 4, 3) node coords
    vols = np.abs(np.linalg.det(T)) / 6.0   # (N_elem,)

    # Scatter equal mass share to each of the 4 corner nodes
    m_node = np.zeros(N)
    np.add.at(m_node, conn.ravel(),
              np.repeat(rho * vols / 4.0, 4))

    return np.repeat(m_node, 3)          # expand to x, y, z DOFs


def compute_internal_forces(mesh, u, D, eps_p, sigma_y0, H, mu,
                            failed_elem, enable_fracture, eps_failure):
    """
    Compute the global internal force vector and update plastic state.

    Using element-level radial return for J2 plasticity.

    Args:
        mesh          : skfem MeshTet
        u             : (3·N,) displacement vector  [mm]
        D             : (6, 6) elastic constitutive matrix  [MPa]
        eps_p         : (N_elem,) equivalent plastic strain per element
        sigma_y0      : initial yield stress  [MPa]
        H             : hardening modulus  [MPa]
        mu            : shear modulus  [MPa]
        failed_elem   : (N_elem,) bool – previously failed elements
        enable_fracture: bool
        eps_failure   : failure equivalent plastic strain threshold

    Returns:
        F_int       : (3·N,) internal force vector  [N]
        eps_p_new   : (N_elem,) updated equivalent plastic strain
        sigma_vm    : (N_elem,) element Von Mises stress  [MPa]
        failed_new  : (N_elem,) updated failure flags
    """
    N = mesh.p.shape[1]
    N_elem = mesh.t.shape[1]

    F_int = np.zeros(3 * N)
    eps_p_new = eps_p.copy()
    sigma_vm = np.zeros(N_elem)
    failed_new = failed_elem.copy()

    pts = mesh.p.T    # (N_nodes, 3)

    for e in range(N_elem):
        # Skip already-failed elements (element death)
        if failed_elem[e]:
            continue

        idx = mesh.t[:, e]         # 4 node indices
        nodes = pts[idx]           # (4, 3) node coords

        B, vol = compute_tet_B(nodes)
        if vol < 1e-20:
            continue

        # Element displacement vector (12,)
        u_e = np.empty(12)
        for i, n in enumerate(idx):
            u_e[3*i:3*i+3] = u[3*n:3*n+3]

        # Strain (Voigt notation, 6×1)
        epsilon = B @ u_e

        # Trial stress
        sigma_trial = D @ epsilon

        # Radial return for plasticity
        sigma, eps_p_new[e], _, _ = radial_return(
            sigma_trial, eps_p[e], sigma_y0, H, mu
        )

        # Von Mises from final stress
        p_s = (sigma[0] + sigma[1] + sigma[2]) / 3.0
        s = sigma.copy()
        s[0] -= p_s; s[1] -= p_s; s[2] -= p_s
        sigma_vm[e] = np.sqrt(max(
            1.5 * (s[0]**2 + s[1]**2 + s[2]**2
                   + 2.0*(s[3]**2 + s[4]**2 + s[5]**2)),
            0.0
        ))

        # Check fracture criterion
        if enable_fracture and eps_p_new[e] >= eps_failure:
            failed_new[e] = True
            eps_p_new[e] = eps_failure
            sigma_vm[e] = 0.0
            continue

        # Element internal force scatter: f_e = vol · Bᵀ σ
        f_e = vol * (B.T @ sigma)
        for i, n in enumerate(idx):
            F_int[3*n:3*n+3] += f_e[3*i:3*i+3]

    return F_int, eps_p_new, sigma_vm, failed_new


def sigma_vm_to_nodal(mesh, field):
    """Average an element-valued field to nodes (vectorized)."""
    N    = mesh.p.shape[1]
    nodal = np.zeros(N)
    count = np.zeros(N)
    # mesh.t shape (4, N_elem); .T.ravel() gives element-major ordering
    idx  = mesh.t.T.ravel()                          # (4·N_elem,)
    vals = np.repeat(field.astype(float), 4)         # (4·N_elem,)
    np.add.at(nodal, idx, vals)
    np.add.at(count, idx, np.ones(len(idx)))
    mask = count > 0
    nodal[mask] /= count[mask]
    return nodal


def _voigt_to_full(sv):
    """
    Convert Voigt stress vectors (N, 6) → full symmetric tensors (N, 3, 3).
    Convention: [s11, s22, s33, s12, s23, s13]
    """
    sf = np.zeros((sv.shape[0], 3, 3))
    sf[:, 0, 0] = sv[:, 0];  sf[:, 1, 1] = sv[:, 1];  sf[:, 2, 2] = sv[:, 2]
    sf[:, 0, 1] = sf[:, 1, 0] = sv[:, 3]
    sf[:, 1, 2] = sf[:, 2, 1] = sv[:, 4]
    sf[:, 0, 2] = sf[:, 2, 0] = sv[:, 5]
    return sf


def _full_to_voigt(sf):
    """Convert full symmetric tensors (N, 3, 3) → Voigt (N, 6)."""
    sv = np.empty((sf.shape[0], 6))
    sv[:, 0] = sf[:, 0, 0];  sv[:, 1] = sf[:, 1, 1];  sv[:, 2] = sf[:, 2, 2]
    sv[:, 3] = sf[:, 0, 1];  sv[:, 4] = sf[:, 1, 2];  sv[:, 5] = sf[:, 0, 2]
    return sv


def _precompute_elements(mesh):
    """
    Precompute B matrices, element volumes, DOF index table, and reference
    shape-matrix inverses for all tetrahedral elements.
    Called **once** before the time loop.

    Returns
    -------
    B_all      : (N_elem, 6, 12)  – strain-displacement matrices
    vol_all    : (N_elem,)        – element volumes  [mm³]
    dof_idx    : (N_elem, 12)     – global DOF indices for each element
    Dm_inv_all : (N_elem, 3, 3)   – inverse of reference shape matrices
                                    (columns = reference edge vectors)
                                    Used by the co-rotational formulation.
    """
    pts        = mesh.p.T         # (N_nodes, 3)
    N_elem     = mesh.t.shape[1]
    conn       = mesh.t.T         # (N_elem, 4)
    nodes_batch = pts[conn]       # (N_elem, 4, 3)

    T = np.ones((N_elem, 4, 4))
    T[:, :, 1:] = nodes_batch
    det   = np.linalg.det(T)                   # (N_elem,)
    vol   = np.abs(det) / 6.0
    valid = np.abs(det) > 1e-20

    T_inv = np.zeros((N_elem, 4, 4))
    if np.any(valid):
        T_inv[valid] = np.linalg.inv(T[valid])

    dNdx = T_inv[:, 1, :]         # (N_elem, 4)
    dNdy = T_inv[:, 2, :]
    dNdz = T_inv[:, 3, :]

    B_all = np.zeros((N_elem, 6, 12))
    for i in range(4):
        j = 3 * i
        B_all[:, 0, j    ] = dNdx[:, i]
        B_all[:, 1, j + 1] = dNdy[:, i]
        B_all[:, 2, j + 2] = dNdz[:, i]
        B_all[:, 3, j    ] = dNdy[:, i];  B_all[:, 3, j + 1] = dNdx[:, i]
        B_all[:, 4, j + 1] = dNdz[:, i];  B_all[:, 4, j + 2] = dNdy[:, i]
        B_all[:, 5, j    ] = dNdz[:, i];  B_all[:, 5, j + 2] = dNdx[:, i]

    B_all[~valid] = 0.0
    vol[~valid]   = 0.0

    # DOF index table: for each element, 12 global DOF indices
    dof_idx = np.repeat(conn * 3, 3, axis=1) + np.tile([0, 1, 2], 4)

    # Reference shape-matrix inverse  Dm_inv  (used by co-rotational kernel)
    # Dm = [x1-x0 | x2-x0 | x3-x0]  columns = reference edge vectors
    e0_ref     = nodes_batch[:, 0, :]                       # (N_elem, 3)
    Dm_all     = np.stack(
        [nodes_batch[:, k, :] - e0_ref for k in range(1, 4)], axis=2
    )                                                        # (N_elem, 3, 3)
    Dm_inv_all = np.zeros_like(Dm_all)
    if np.any(valid):
        Dm_inv_all[valid] = np.linalg.inv(Dm_all[valid])

    return B_all, vol, dof_idx, Dm_inv_all


def _radial_return_vec(sigma_trial, eps_p_eq, sigma_y0, H, mu):
    """
    Vectorized J2 von Mises radial return with linear isotropic hardening.

    Parameters
    ----------
    sigma_trial : (N_elem, 6)  trial stress vectors
    eps_p_eq    : (N_elem,)    equivalent plastic strain (accumulated)

    Returns
    -------
    sigma     : (N_elem, 6)  corrected Cauchy stress
    eps_p_new : (N_elem,)    updated equivalent plastic strain
    """
    p = (sigma_trial[:, 0] + sigma_trial[:, 1] + sigma_trial[:, 2]) / 3.0
    s = sigma_trial.copy()
    s[:, 0] -= p;  s[:, 1] -= p;  s[:, 2] -= p

    s_eq = np.sqrt(np.maximum(
        1.5 * (s[:, 0]**2 + s[:, 1]**2 + s[:, 2]**2
               + 2.0 * (s[:, 3]**2 + s[:, 4]**2 + s[:, 5]**2)),
        0.0))

    sigma_y   = sigma_y0 + H * eps_p_eq
    is_pl     = s_eq > sigma_y
    delta_lam = np.where(is_pl, (s_eq - sigma_y) / (3.0 * mu + H), 0.0)
    s_eq_safe = np.where(s_eq > 1e-30, s_eq, 1.0)
    scale     = np.where(is_pl,
                         1.0 - delta_lam * 3.0 * mu / s_eq_safe,
                         1.0)

    sigma_out = np.empty_like(sigma_trial)
    sigma_out[:, 0] = p + scale * s[:, 0]
    sigma_out[:, 1] = p + scale * s[:, 1]
    sigma_out[:, 2] = p + scale * s[:, 2]
    sigma_out[:, 3] = scale * s[:, 3]
    sigma_out[:, 4] = scale * s[:, 4]
    sigma_out[:, 5] = scale * s[:, 5]

    return sigma_out, eps_p_eq + delta_lam


def _compute_forces_vec(delta_u, sigma_elem, D, eps_p, sigma_y0, H, mu,
                        failed_elem, enable_fracture, eps_failure,
                        B_all, vol_all, dof_idx, N_nodes):
    """
    Vectorized internal force assembly using incremental (hypoelastic) stress
    update.  Replaces the previous total-strain approach which incorrectly
    re-applied cumulative elastic energy in every step, causing spurious
    plastic strain accumulation and mass element deletion.

    Parameters
    ----------
    delta_u    : (3·N,)       incremental displacement vector for this step
                              (= dt * v_half, already BC-enforced)
    sigma_elem : (N_elem, 6)  per-element Cauchy stress from the PREVIOUS step
    B_all      : (N_elem, 6, 12)
    vol_all    : (N_elem,)
    dof_idx    : (N_elem, 12)  precomputed global DOF index table
    N_nodes    : int

    Returns
    -------
    F_int, eps_p_new, sigma_vm, failed_new, sigma_out
        sigma_out : (N_elem, 6) updated per-element Cauchy stress
    """
    # 1. Gather incremental element displacements
    du_elems = delta_u[dof_idx]                               # (N_elem, 12)

    # 2. Incremental strains  Δε = B Δu
    delta_epsilon = np.einsum('eij,ej->ei', B_all, du_elems)  # (N_elem, 6)

    # 3. Trial stresses: σ_trial = σ_old + D Δε  (incremental hypoelastic update)
    #    Using the stored corrected stress as the starting point ensures that
    #    plastic corrections from all previous steps are correctly carried over.
    sigma_trial = sigma_elem + delta_epsilon @ D.T            # (N_elem, 6)

    # 4. Batch plasticity return mapping
    sigma, eps_p_new = _radial_return_vec(sigma_trial, eps_p, sigma_y0, H, mu)

    # 5. Von Mises stress
    p_s     = (sigma[:, 0] + sigma[:, 1] + sigma[:, 2]) / 3.0
    sv      = sigma.copy()
    sv[:, 0] -= p_s;  sv[:, 1] -= p_s;  sv[:, 2] -= p_s
    sigma_vm = np.sqrt(np.maximum(
        1.5 * (sv[:, 0]**2 + sv[:, 1]**2 + sv[:, 2]**2
               + 2.0 * (sv[:, 3]**2 + sv[:, 4]**2 + sv[:, 5]**2)),
        0.0))

    # 6. Fracture check
    failed_new = failed_elem.copy()
    if enable_fracture:
        new_fail = (~failed_elem) & (eps_p_new >= eps_failure)
        failed_new   |= new_fail
        eps_p_new[new_fail] = eps_failure

    # 7. Zero out failed elements
    all_failed         = failed_elem | failed_new
    sigma[all_failed]  = 0.0
    sigma_vm[all_failed] = 0.0

    # 8. Element internal forces  f_e = vol · Bᵀ σ
    f_elems = vol_all[:, None] * np.einsum('eji,ej->ei', B_all, sigma)  # (N_elem, 12)
    f_elems[all_failed] = 0.0

    # NaN / Inf guard – any non-finite value would propagate and blow up
    # the whole simulation; replace silently with zero.
    bad = ~np.isfinite(f_elems)
    if np.any(bad):
        f_elems[bad] = 0.0

    # 9. Scatter to global DOF vector
    F_int = np.zeros(3 * N_nodes)
    np.add.at(F_int, dof_idx.ravel(), f_elems.ravel())

    # 10. Return updated stress tensor (zero for newly/previously failed elements)
    sigma_out = sigma.copy()
    sigma_out[all_failed] = 0.0

    return F_int, eps_p_new, sigma_vm, failed_new, sigma_out


def _compute_forces_corot_vec(delta_u, u_total, sigma_elem, D, eps_p,
                              sigma_y0, H, mu,
                              failed_elem, enable_fracture, eps_failure,
                              B_all, vol_all, dof_idx, Dm_inv_all,
                              pts_ref, conn, N_nodes):
    """
    Co-rotational explicit force assembly for large-rotation crash simulations.

    For each tetrahedral element this function:
      1. Builds the current shape matrix Ds and deformation gradient F = Ds · Dm⁻¹.
      2. Extracts the rigid-body rotation R via polar decomposition  F = R · U
         (computed through SVD:  F = P · Σ · Qᵀ  →  R = P · Qᵀ).
      3. Unrotates the stored Cauchy stress to the element-local frame:
             σ_local = Rᵀ · σ_global · R
      4. Co-rotates the incremental nodal displacement to the local frame:
             Δu_local_i = Rᵀ · Δu_global_i   for each of the 4 corner nodes.
      5. Computes incremental strain and trial stress in the local frame:
             Δε = B · Δu_local
             σ_trial = σ_local + D · Δε
      6. Applies the J2 von Mises radial-return algorithm (identical to the
         standard path — rotation-invariance means plasticity is unaffected).
      7. Rotates the updated local stress back to the global frame:
             σ_global_new = R · σ_local_new · Rᵀ
      8. Computes element nodal forces in the local frame and rotates them back:
             f_global_i = R · f_local_i

    Why this matters for crashes
    ----------------------------
    The standard (non-co-rotational) solver computes B_all **once** at t = 0 and
    uses it unchanged throughout.  When an element undergoes a large rigid-body
    rotation (e.g. a bumper folding 90°) the fixed B matrix produces wrong strain
    components (shear instead of axial) causing artificial stiffening and spurious
    stress blow-up.  The co-rotational correction removes this error entirely.

    Parameters are identical to _compute_forces_vec with the additions:
      u_total    : (3·N,)       total displacement (needed for current positions)
      Dm_inv_all : (N_elem, 3, 3)  reference shape-matrix inverses
      pts_ref    : (N_nodes, 3)    reference (undeformed) node coordinates
      conn       : (N_elem, 4)     element connectivity (node indices)
    """
    N_elem = B_all.shape[0]

    # ── 1. Current node positions ─────────────────────────────────────────────
    x_cur = pts_ref + u_total.reshape(-1, 3)              # (N_nodes, 3)
    x_e   = x_cur[conn]                                   # (N_elem, 4, 3)

    # ── 2. Deformation gradient F = Ds · Dm⁻¹ ────────────────────────────────
    x0_e = x_e[:, 0, :]                                   # (N_elem, 3)
    Ds   = np.stack(
        [x_e[:, k, :] - x0_e for k in range(1, 4)], axis=2
    )                                                      # (N_elem, 3, 3)
    F    = np.einsum('eij,ejk->eik', Ds, Dm_inv_all)      # (N_elem, 3, 3)

    # ── 3. Rotation extraction via vectorised Gram-Schmidt ────────────────────
    # GS orthonormalises the columns of F to give a proper rotation matrix.
    # This is ~7× faster than np.linalg.svd for large batches of 3×3 matrices
    # and produces results that match the polar-decomposition rotation to
    # O(ε) where ε is the element strain — well within T4 modelling accuracy.
    #
    # Algorithm:
    #   e0 = normalise(F[:,0])
    #   e1 = normalise(F[:,1] − (F[:,1]·e0) e0)
    #   e2 = e0 × e1  (right-hand orthonormal triad)
    #   R  = [e0 | e1 | e2]   (det = +1 by construction; flip e2 if inverted)
    _c0 = F[:, :, 0]
    _n0 = np.linalg.norm(_c0, axis=1, keepdims=True)
    e0  = _c0 / np.maximum(_n0, 1e-30)

    _c1  = F[:, :, 1]
    _e1t = _c1 - np.einsum('ni,ni->n', _c1, e0)[:, np.newaxis] * e0
    _n1  = np.linalg.norm(_e1t, axis=1, keepdims=True)
    e1   = _e1t / np.maximum(_n1, 1e-30)

    e2 = np.cross(e0, e1)                                 # unit, right-hand

    R = np.stack([e0, e1, e2], axis=2)                    # (N_elem, 3, 3)

    # Inverted elements (det < 0): flip third column to restore proper rotation
    bad = np.linalg.det(R) < 0
    if np.any(bad):
        R[bad, :, 2] *= -1

    R_T = R.transpose(0, 2, 1)                             # (N_elem, 3, 3)  R^T

    # ── 4. Unrotate stored global stress → element local frame ────────────────
    # σ_local = Rᵀ · σ_global · R   (batch 3-D similarity transform)
    sg_full   = _voigt_to_full(sigma_elem)                 # (N_elem, 3, 3)
    sl_full   = R_T @ sg_full @ R                          # (N_elem, 3, 3)
    sigma_loc = _full_to_voigt(sl_full)                    # (N_elem, 6)

    # ── 5. Co-rotate incremental displacements ────────────────────────────────
    # Δu_local_i = Rᵀ · Δu_global_i  for each corner node i ∈ {0..3}
    du_e       = delta_u[dof_idx]                          # (N_elem, 12)
    du_4x3     = du_e.reshape(N_elem, 4, 3)               # (N_elem, 4, 3)
    # einsum: du_local[e,n,i] = Σ_j R_T[e,i,j] * du[e,n,j]
    du_local_4x3 = np.einsum('eij,enj->eni', R_T, du_4x3) # (N_elem, 4, 3)
    du_local     = du_local_4x3.reshape(N_elem, 12)

    # ── 6. Strain, trial stress, J2 return (same as standard path) ────────────
    delta_eps   = np.einsum('eij,ej->ei', B_all, du_local) # (N_elem, 6)
    sigma_trial = sigma_loc + delta_eps @ D.T              # (N_elem, 6)

    sigma_c, eps_p_new = _radial_return_vec(sigma_trial, eps_p, sigma_y0, H, mu)

    # Von Mises (rotation-invariant, computed from local stress)
    p_s      = (sigma_c[:, 0] + sigma_c[:, 1] + sigma_c[:, 2]) / 3.0
    sv_dev   = sigma_c.copy()
    sv_dev[:, 0] -= p_s;  sv_dev[:, 1] -= p_s;  sv_dev[:, 2] -= p_s
    sigma_vm = np.sqrt(np.maximum(
        1.5 * (sv_dev[:, 0]**2 + sv_dev[:, 1]**2 + sv_dev[:, 2]**2
               + 2.0 * (sv_dev[:, 3]**2 + sv_dev[:, 4]**2 + sv_dev[:, 5]**2)),
        0.0))

    # Fracture criterion
    failed_new = failed_elem.copy()
    if enable_fracture:
        new_fail         = (~failed_elem) & (eps_p_new >= eps_failure)
        failed_new      |= new_fail
        eps_p_new[new_fail] = eps_failure

    all_failed          = failed_elem | failed_new
    sigma_c[all_failed] = 0.0
    sigma_vm[all_failed] = 0.0

    # ── 7. Rotate corrected stress back to global frame ───────────────────────
    sc_full    = _voigt_to_full(sigma_c)                   # (N_elem, 3, 3)
    sg_new     = R @ sc_full @ R_T                         # (N_elem, 3, 3)
    sigma_out  = _full_to_voigt(sg_new)                    # (N_elem, 6)
    sigma_out[all_failed] = 0.0

    # ── 8. Nodal forces: compute in local frame, rotate to global ─────────────
    f_local   = vol_all[:, None] * np.einsum(
        'eji,ej->ei', B_all, sigma_c)                      # (N_elem, 12)
    f_local[all_failed] = 0.0

    bad_f = ~np.isfinite(f_local)
    if np.any(bad_f):
        f_local[bad_f] = 0.0

    # f_global_i = R · f_local_i  for each corner node
    # einsum: f_global[e,n,i] = Σ_j R[e,i,j] * f_local[e,n,j]
    f_l4x3     = f_local.reshape(N_elem, 4, 3)
    f_g4x3     = np.einsum('eij,enj->eni', R, f_l4x3)     # (N_elem, 4, 3)
    f_global   = f_g4x3.reshape(N_elem, 12)

    # Scatter to global DOF vector
    F_int = np.zeros(3 * N_nodes)
    np.add.at(F_int, dof_idx.ravel(), f_global.ravel())

    return F_int, eps_p_new, sigma_vm, failed_new, sigma_out


def _compute_cfl_dt(mesh, E, nu, rho, safety=0.5):
    """
    Element-wise CFL stability limit for the central-difference scheme.

    Uses the LS-DYNA formula for solid elements:
        dt_crit = safety · L_e / c_d

    where
        L_e = 3·V / A_max   (tet characteristic length = minimum altitude)
        c_d = sqrt((λ + 2μ) / ρ)  (dilatational / P-wave speed)

    This is more conservative than using the minimum edge length and the bar
    wave speed sqrt(E/ρ), which over-estimates the stable step size by up to
    50 % on flat tetrahedra.

    Args:
        mesh   : skfem MeshTet
        E      : Young's modulus  [MPa = N/mm²]
        nu     : Poisson's ratio  [–]
        rho    : density          [tonne/mm³]
        safety : Courant safety factor (< 1; 0.5 is standard for solid FEM)

    Returns:
        dt_crit : float  [ms]
    """
    # Dilatational wave speed (faster than bar speed, gives smaller dt)
    lam    = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu_val = E / (2.0 * (1.0 + nu))
    c_d    = float(np.sqrt(max(lam + 2.0 * mu_val, 1.0) / max(rho, 1e-30)))
    # Unit analysis: sqrt([N/mm²] / [t/mm³]) = sqrt([t·mm/ms²·mm²] / [t/mm³])
    # where 1 N = 1e-6 t·mm/ms² in the mm-tonne-ms system  →  this gives mm/s,
    # not mm/ms.  Multiplying the final dt by 1000 converts the result to ms.
    # Net effect: safety * L_e[mm] / c_d[mm/s] * 1000 = safety * L_e / c_d_ms [ms].

    pts  = mesh.p.T      # (N_nodes, 3)
    conn = mesh.t.T      # (N_elem, 4)
    node_coords = pts[conn]  # (N_elem, 4, 3)

    # Element volumes (same calculation as assemble_lumped_mass)
    T = np.ones((conn.shape[0], 4, 4))
    T[:, :, 1:] = node_coords
    vols = np.abs(np.linalg.det(T)) / 6.0   # (N_elem,)

    # Compute area of each of the 4 faces per element
    # Face opposite node i uses the other 3 nodes
    face_node_sets = [(1, 2, 3), (0, 2, 3), (0, 1, 3), (0, 1, 2)]
    max_area = np.zeros(conn.shape[0])
    for fi, fj, fk in face_node_sets:
        ab = node_coords[:, fj, :] - node_coords[:, fi, :]  # (N_elem, 3)
        ac = node_coords[:, fk, :] - node_coords[:, fi, :]  # (N_elem, 3)
        cross = np.cross(ab, ac)                             # (N_elem, 3)
        area  = 0.5 * np.linalg.norm(cross, axis=1)         # (N_elem,)
        max_area = np.maximum(max_area, area)

    # Characteristic length L_e = 3·V / A_max  (altitude to the largest face (most conservative CFL length))
    valid = (max_area > 1e-20) & (vols > 1e-20)
    L_e   = np.full(conn.shape[0], np.inf)
    L_e[valid] = 3.0 * vols[valid] / max_area[valid]

    min_L = float(np.min(L_e[valid])) if np.any(valid) else 1e-4
    if min_L < 1e-20 or c_d < 1e-20:
        return 1e-4   # fallback

    # h / c_d is in seconds; × 1000 → ms
    return safety * min_L / c_d * 1000.0


def _compute_contact_dt_limit(M_diag, boundary_nodes, k_penalty, safety=0.8):
    """
    Estimate the explicit stability limit introduced by penalty contact.

    A penalty contact constraint behaves like a spring with stiffness
    k_penalty [N/mm].  For a single-DOF oscillator the central-difference
    stability limit is dt <= 2 * sqrt(m / k).  With mass in tonne and
    stiffness in N/mm the unit conversion contributes a factor of 2000.
    """
    if k_penalty <= 1e-30 or boundary_nodes is None or len(boundary_nodes) == 0:
        return np.inf

    node_masses = M_diag[0::3]
    min_mass = float(np.min(node_masses[boundary_nodes]))
    if min_mass <= 1e-30:
        return np.inf

    return safety * 2000.0 * np.sqrt(min_mass / k_penalty)


def _apply_mass_scaling(M_diag, mesh, E, nu, rho, dt_target, threshold=0.05):
    """
    Selective mass scaling for the central-difference explicit solver.

    For every tetrahedral element whose natural CFL time-step  dt_e  is smaller
    than the target *dt_target*, the element’s share of the lumped mass matrix
    is scaled by the factor

        s_e = (dt_target / dt_e)²

    so that its new CFL limit equals *dt_target*.  Only “critical” elements
    (dt_e < dt_target) are affected; well-shaped elements with dt_e ≥ dt_target
    are left unchanged.

    Reference: Belytschko, Liu & Moran “Nonlinear FE for Continua and
    Structures”, Section 6.3 (Explicit time integration with mass scaling).

    Parameters
    ----------
    M_diag    : (3·N,) lumped mass diagonal  [tonne]
    mesh      : skfem MeshTet
    E         : Young’s modulus  [MPa]
    nu        : Poisson’s ratio  [–]
    rho       : mass density  [tonne/mm³]
    dt_target : float — target time step  [ms]  (= end_time / requested steps)
    threshold : fraction of total mass increase that triggers a warning
                (default 0.05 = 5 %)

    Returns
    -------
    M_scaled   : (3·N,) scaled mass diagonal
    scale_info : dict with 'n_scaled', 'mass_increase_fraction',
                 'max_scale_factor', 'dt_min_scaled'
    """
    lam_s  = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu_s   = E / (2.0 * (1.0 + nu))
    c_d    = float(np.sqrt(max(lam_s + 2.0 * mu_s, 1.0) / max(rho, 1e-30)))

    pts      = mesh.p.T       # (N_nodes, 3)
    conn     = mesh.t.T       # (N_elem, 4)
    N_nodes  = pts.shape[0]
    N_elem   = conn.shape[0]
    node_coords = pts[conn]   # (N_elem, 4, 3)

    # Element volumes  (same formula as assemble_lumped_mass)
    T = np.ones((N_elem, 4, 4))
    T[:, :, 1:] = node_coords
    vols = np.abs(np.linalg.det(T)) / 6.0

    # Maximum face area per element  (same as _compute_cfl_dt)
    face_sets = [(1, 2, 3), (0, 2, 3), (0, 1, 3), (0, 1, 2)]
    max_area  = np.zeros(N_elem)
    for fi, fj, fk in face_sets:
        ab   = node_coords[:, fj, :] - node_coords[:, fi, :]
        ac   = node_coords[:, fk, :] - node_coords[:, fi, :]
        area = 0.5 * np.linalg.norm(np.cross(ab, ac), axis=1)
        max_area = np.maximum(max_area, area)

    valid = (max_area > 1e-20) & (vols > 1e-20)
    L_e   = np.full(N_elem, np.inf)
    L_e[valid] = 3.0 * vols[valid] / max_area[valid]

    # Per-element CFL stability limit  [ms]
    dt_e = np.full(N_elem, np.inf)
    if c_d > 1e-20:
        dt_e[valid] = 0.5 * L_e[valid] / c_d * 1000.0   # safety=0.5; *1000 converts mm/s → ms

    # Critical elements: natural dt_e < user-requested dt_target
    critical  = valid & (dt_e < dt_target)
    scale_e   = np.ones(N_elem)
    safe_dt_e = np.where(dt_e > 1e-30, dt_e, 1e-30)
    scale_e[critical] = (dt_target / safe_dt_e[critical]) ** 2

    # Distribute scaled mass to nodes via row-sum lumping
    m_node_orig   = np.zeros(N_nodes)
    m_node_scaled = np.zeros(N_nodes)
    np.add.at(m_node_orig,   conn.ravel(), np.repeat(rho * vols / 4.0, 4))
    np.add.at(m_node_scaled, conn.ravel(), np.repeat(rho * vols * scale_e / 4.0, 4))

    M_scaled = np.repeat(m_node_scaled, 3)

    # Diagnostics
    total_orig    = float(np.sum(m_node_orig))
    total_scaled  = float(np.sum(m_node_scaled))
    mass_increase = (total_scaled - total_orig) / max(total_orig, 1e-30)
    max_sf        = float(np.max(scale_e[critical])) if np.any(critical) else 1.0
    dt_scaled     = dt_e * np.sqrt(scale_e)
    dt_min_scaled = float(np.min(dt_scaled[valid])) if np.any(valid) else dt_target

    return M_scaled, {
        'n_scaled':               int(np.sum(critical)),
        'mass_increase_fraction': mass_increase,
        'max_scale_factor':       max_sf,
        'dt_min_scaled':          dt_min_scaled,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Contact mechanics helper
# ─────────────────────────────────────────────────────────────────────────────

def compute_penalty_contact(u, pts_initial, boundary_nodes, thickness, k_penalty,
                             surf_facets=None):
    """
    Penalty contact with node-to-node broad phase + optional node-to-triangle
    narrow phase.

    Broad phase (always active)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~
    O(N_b log N_b) cKDTree sphere overlap: each pair of surface nodes whose
    current distance falls below ``thickness`` receives an equal-and-opposite
    repulsive force (linear spring law).

    Narrow phase (active when ``surf_facets`` is supplied)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Fixes *contact phasing*: a sharp corner can slip through the centre of a
    flat face because every corner node of that face is farther than
    ``thickness`` — the broad phase misses it entirely.  The narrow phase
    projects each surface node onto nearby surface triangles; if the projection
    lands inside the triangle and the signed distance is penetrating (negative),
    a restorative force is applied along the facet outward normal.

    Parameters
    ----------
    u              : (3N,)    current flat displacement vector  [mm]
    pts_initial    : (N, 3)   reference nodal coordinates  [mm]
    boundary_nodes : (N_b,)   integer indices of surface (boundary) nodes
    thickness      : float    contact search radius  [mm]
    k_penalty      : float    penalty stiffness  [N/mm]
    surf_facets    : (M, 3) int | None
                    Triangular surface facets (vertex index triples) for the
                    node-to-triangle narrow phase.  Pass None (default) to use
                    the original node-to-node-only algorithm.

    Returns
    -------
    F_contact : (3N,)  flat contact force vector to be added to the equation
                       of motion.
    """
    N     = pts_initial.shape[0]
    x_cur = pts_initial + u.reshape(N, 3)           # current positions (N, 3)
    F_3n  = np.zeros((N, 3))

    if len(boundary_nodes) < 2:
        return F_3n.flatten()

    surf_pts = x_cur[boundary_nodes]                # (N_b, 3)

    # ── Broad phase: O(N_b log N_b) node-to-node sphere overlap ─────────────
    tree  = cKDTree(surf_pts)
    pairs = tree.query_pairs(r=thickness, output_type='ndarray')  # (P, 2)

    if len(pairs) > 0:
        li = pairs[:, 0];  lj = pairs[:, 1]
        gi = boundary_nodes[li];  gj = boundary_nodes[lj]
        vec  = x_cur[gj] - x_cur[gi]
        dist = np.linalg.norm(vec, axis=1)
        ok   = dist > 1e-12
        if np.any(ok):
            gi, gj, vec, dist = gi[ok], gj[ok], vec[ok], dist[ok]
            delta  = thickness - dist
            active = delta > 0.0
            if np.any(active):
                normal = vec[active] / dist[active, None]
                f_vec  = (k_penalty * delta[active])[:, None] * normal
                np.add.at(F_3n, gi[active], -f_vec)
                np.add.at(F_3n, gj[active],  f_vec)

    # ── Node-to-triangle narrow phase (contact phasing fix) ──────────────────
    # Catches sharp-corner-into-flat-face penetration that the node-to-node
    # broad phase misses when the corner clears all triangle vertices but
    # projects inside the triangle area.
    if surf_facets is not None and len(surf_facets) > 0:
        fa = x_cur[surf_facets[:, 0]]               # (M, 3) face vertices A
        fb = x_cur[surf_facets[:, 1]]               # (M, 3) face vertices B
        fc = x_cur[surf_facets[:, 2]]               # (M, 3) face vertices C
        centroids = (fa + fb + fc) / 3.0            # (M, 3)

        tri_tree = cKDTree(centroids)
        # Each surface node queries nearby facet centroids within 2×thickness
        hits = tri_tree.query_ball_point(surf_pts, r=2.0 * thickness)

        for b_idx, tri_list in enumerate(hits):
            if not tri_list:
                continue
            g_node = boundary_nodes[b_idx]
            p      = x_cur[g_node]

            for t_idx in tri_list:
                ia, ib, ic = surf_facets[t_idx]
                if g_node == ia or g_node == ib or g_node == ic:
                    continue                         # self-facet — skip

                a_ = x_cur[ia];  b_ = x_cur[ib];  c_ = x_cur[ic]
                ab = b_ - a_;    ac = c_ - a_;    ap = p  - a_
                n_raw = np.cross(ab, ac)
                n_len = float(np.linalg.norm(n_raw))
                if n_len < 1e-30:
                    continue
                n_unit = n_raw / n_len
                d      = float(np.dot(ap, n_unit))  # signed dist (+ve = outside)

                # Only process nodes that are slightly penetrating (d < 0)
                if d > 0.0 or d < -2.0 * thickness:
                    continue

                # Barycentric test: is the projection inside the triangle?
                p_proj = p - d * n_unit
                ap2    = p_proj - a_
                d00 = float(np.dot(ac, ac));  d01 = float(np.dot(ac, ab))
                d02 = float(np.dot(ac, ap2)); d11 = float(np.dot(ab, ab))
                d12 = float(np.dot(ab, ap2))
                det = d00 * d11 - d01 * d01
                if abs(det) < 1e-30:
                    continue
                u_ = (d11 * d02 - d01 * d12) / det
                v_ = (d00 * d12 - d01 * d02) / det
                _tol = 0.05             # 5 % border tolerance for near-edge hits
                if u_ < -_tol or v_ < -_tol or u_ + v_ > 1.0 + _tol:
                    continue            # projection outside triangle

                # Penetration confirmed — apply corrective force along outward normal
                f_rep = (k_penalty * (-d)) * n_unit     # pushes node back out
                F_3n[g_node] += f_rep
                # Distribute equal reaction to the three triangle vertices
                F_3n[ia] -= f_rep / 3.0
                F_3n[ib] -= f_rep / 3.0
                F_3n[ic] -= f_rep / 3.0

    return F_3n.flatten()


def _update_boundary_nodes(mesh, failed_elem):
    """
    Recompute the set of surface (boundary) nodes after element deletion.

    A face is on the *current* surface when it belongs to exactly one *live*
    element.  Nodes of such faces are the active contact surface.

    Parameters
    ----------
    mesh        : skfem MeshTet
    failed_elem : (N_elem,) bool  – True = deleted element

    Returns
    -------
    boundary_nodes : (N_b,) int array  – sorted unique surface node indices
    """
    conn      = mesh.t.T                          # (N_elem, 4)
    live_conn = conn[~failed_elem]                # (N_live, 4)

    # The 4 triangular faces of a tet (local node combinations)
    FACE_IDX = [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]

    if len(live_conn) == 0:
        return np.array([], dtype=int), np.zeros((0, 3), dtype=int)

    all_faces = []
    for fi, fj, fk in FACE_IDX:
        faces = np.stack([live_conn[:, fi],
                          live_conn[:, fj],
                          live_conn[:, fk]], axis=1)
        all_faces.append(np.sort(faces, axis=1))

    # Fast uncompiled C-backend hashing for unique faces
    all_faces = np.vstack(all_faces)  # (4 * N_live, 3)
    unique_faces, counts = np.unique(all_faces, axis=0, return_counts=True)
    exposed_faces = unique_faces[counts == 1]

    if len(exposed_faces) == 0:
        return np.array([], dtype=int), np.zeros((0, 3), dtype=int)

    surf_nodes_set = np.unique(exposed_faces)
    return surf_nodes_set.astype(int), exposed_faces.astype(int)

