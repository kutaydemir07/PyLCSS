# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

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

    # ── 3. Polar decomposition  F = R · U  via SVD ────────────────────────────
    # np.linalg.svd(F) = (P, S, Qh) where F = P · diag(S) · Qh
    # Rotation:  R = P · Qh
    # Degenerate elements get identity rotation to avoid NaN propagation.
    try:
        P, _S, Qh = np.linalg.svd(F)
    except np.linalg.LinAlgError:
        P  = np.broadcast_to(np.eye(3), (N_elem, 3, 3)).copy()
        Qh = np.broadcast_to(np.eye(3), (N_elem, 3, 3)).copy()

    R = P @ Qh                                            # (N_elem, 3, 3)

    # Fix reflections: det(R) must be +1, not −1 (can happen for inverted elements)
    det_R = np.linalg.det(R)
    bad   = det_R < 0
    if np.any(bad):
        P_fix              = P.copy()
        P_fix[bad, :, -1] *= -1                            # flip last column
        R[bad]             = P_fix[bad] @ Qh[bad]

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
    # c_d is in mm/s  (sqrt([N/mm²] / [t/mm³]) = sqrt(mm²/s²))

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

    # Characteristic length L_e = 3·V / A_max  (minimum altitude of tet)
    valid = (max_area > 1e-20) & (vols > 1e-20)
    L_e   = np.full(conn.shape[0], np.inf)
    L_e[valid] = 3.0 * vols[valid] / max_area[valid]

    min_L = float(np.min(L_e[valid])) if np.any(valid) else 1e-4
    if min_L < 1e-20 or c_d < 1e-20:
        return 1e-4   # fallback

    # h / c_d is in seconds; × 1000 → ms
    return safety * min_L / c_d * 1000.0


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
    scale_info : dict with 'n_scaled', 'mass_increase_fraction', 'max_scale_factor'
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
        dt_e[valid] = 0.5 * L_e[valid] / c_d * 1000.0   # safety = 0.5

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

    return M_scaled, {
        'n_scaled':               int(np.sum(critical)),
        'mass_increase_fraction': mass_increase,
        'max_scale_factor':       max_sf,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Contact mechanics helper
# ─────────────────────────────────────────────────────────────────────────────

def compute_penalty_contact(u, pts_initial, boundary_nodes, thickness, k_penalty):
    """
    Vectorized node-to-node penalty contact algorithm using scipy.spatial.cKDTree.

    Treats each boundary node as a sphere of radius ``thickness/2``.  When two
    spheres overlap the algorithm applies equal-and-opposite repulsive forces
    proportional to the penetration depth (linear spring law):

        F_contact = k_penalty × (thickness − dist) × n̂_ij

    This is the CPU analogue of the GPU spatial-hash contact kernel in
    ``_run_neo_hookean_sim``.  It runs in O(N_b log N_b) time thanks to the
    C-backend cKDTree broad phase (N_b = number of boundary nodes).

    Parameters
    ----------
    u              : (3·N,)  current flat displacement vector  [mm]
    pts_initial    : (N, 3)  reference nodal coordinates  [mm]
    boundary_nodes : (N_b,)  integer indices of surface (boundary) nodes
    thickness      : float   contact search radius  [mm]
                             ≈ 0.2 × minimum element characteristic length
    k_penalty      : float   penalty stiffness  [N/mm]
                             ≈ 0.1 × K_bulk × L_min  (caller sets this)

    Returns
    -------
    F_contact : (3·N,)  flat contact force vector to be added to F_int
    """
    N     = pts_initial.shape[0]
    u_3n  = u.reshape(N, 3)
    x_cur = pts_initial + u_3n          # current positions: (N, 3)

    # Surface node current positions
    surf_pts = x_cur[boundary_nodes]    # (N_b, 3)

    # ── Broad phase: O(N_b log N_b) via C-backend tree ───────────────────────
    tree  = cKDTree(surf_pts)
    pairs = tree.query_pairs(r=thickness, output_type='ndarray')  # (P, 2)

    F_3n = np.zeros((N, 3))

    if len(pairs) == 0:
        return F_3n.flatten()

    # ── Narrow phase: fully vectorized ───────────────────────────────────────
    li = pairs[:, 0];  lj = pairs[:, 1]       # local indices into surf_pts
    gi = boundary_nodes[li]                   # global node indices
    gj = boundary_nodes[lj]

    # Inter-node vector (i → j) and Euclidean distance
    vec  = x_cur[gj] - x_cur[gi]             # (P, 3)
    dist = np.linalg.norm(vec, axis=1)        # (P,)

    # Guard: skip coincident nodes
    ok   = dist > 1e-12
    if not np.any(ok):
        return F_3n.flatten()
    gi, gj, vec, dist = gi[ok], gj[ok], vec[ok], dist[ok]

    # Signed penetration depth δ = thickness − dist  (positive when overlapping)
    delta  = thickness - dist                         # (P,)

    # Unit normal i → j
    normal = vec / dist[:, None]                      # (P, 3)

    # Force magnitude: F = k · δ
    f_vec  = (k_penalty * delta)[:, None] * normal    # (P, 3)

    # Scatter: node i pushed away from j (−f_vec), node j pushed away from i (+f_vec)
    np.add.at(F_3n, gi, -f_vec)
    np.add.at(F_3n, gj,  f_vec)

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
        return np.array([], dtype=int)

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
        return np.array([], dtype=int)

    surf_nodes_set = np.unique(exposed_faces)
    return surf_nodes_set.astype(int)


# ─────────────────────────────────────────────────────────────────────────────
# Crash material presets (yield strength / tangent modulus added to base data)
# ─────────────────────────────────────────────────────────────────────────────

CRASH_MATERIAL_PRESETS = {
    # Preset name: {E [MPa], nu, rho [t/mm³], yield [MPa], H [MPa], eps_f}
    'Custom': {
        'E': 210000.0, 'nu': 0.30, 'rho': 7.85e-9,
        'yield_strength': 250.0, 'tangent_modulus': 2100.0, 'failure_strain': 0.20,
    },
    'Steel (Structural A36)': {
        'E': 200000.0, 'nu': 0.29, 'rho': 7.85e-9,
        'yield_strength': 250.0, 'tangent_modulus': 2000.0, 'failure_strain': 0.20,
    },
    'Steel (High-Strength DP780)': {
        'E': 210000.0, 'nu': 0.30, 'rho': 7.85e-9,
        'yield_strength': 480.0, 'tangent_modulus': 3000.0, 'failure_strain': 0.15,
    },
    'Steel (Ultra-High UHSS 1500)': {
        'E': 210000.0, 'nu': 0.30, 'rho': 7.85e-9,
        'yield_strength': 1200.0, 'tangent_modulus': 4000.0, 'failure_strain': 0.08,
    },
    'Aluminum 6061-T6': {
        'E': 68900.0, 'nu': 0.33, 'rho': 2.70e-9,
        'yield_strength': 276.0, 'tangent_modulus': 690.0,  'failure_strain': 0.12,
    },
    'Aluminum 5052-H32 (Crush)': {
        'E': 70300.0, 'nu': 0.33, 'rho': 2.68e-9,
        'yield_strength': 193.0, 'tangent_modulus': 500.0,  'failure_strain': 0.14,
    },
    'CFRP (Quasi-Isotropic)': {
        'E': 70000.0, 'nu': 0.30, 'rho': 1.55e-9,
        'yield_strength': 600.0, 'tangent_modulus': 0.0,    'failure_strain': 0.015,
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Node 1: CrashMaterialNode
# ─────────────────────────────────────────────────────────────────────────────

class CrashMaterialNode(CadQueryNode):
    """
    Material definition for crash / impact simulation.

    Extends the standard elastic material with plasticity parameters:
    - Yield strength (von Mises)
    - Isotropic hardening modulus (tangent slope after yield)
    - Failure / fracture strain (element deletion threshold)

    Presets cover common automotive and structural crash materials.
    """

    __identifier__ = 'com.cad.sim.crash_material'
    NODE_NAME = 'Crash Material'

    def __init__(self):
        super().__init__()
        self.add_output('crash_material', color=(255, 150, 50))

        self.create_property(
            'preset', 'Steel (Structural A36)',
            widget_type='combo',
            items=list(CRASH_MATERIAL_PRESETS.keys())
        )
        # ---------- elastic ----------
        self.create_property('youngs_modulus',  210000.0, widget_type='float')  # MPa
        self.create_property('poissons_ratio',  0.3,      widget_type='float')
        self.create_property('density',         7.85e-9,  widget_type='float')  # t/mm³
        # ---------- plasticity ----------
        self.create_property('yield_strength',  250.0,    widget_type='float')  # MPa
        self.create_property('tangent_modulus', 2000.0,   widget_type='float')  # MPa
        self.create_property('failure_strain',  0.20,     widget_type='float')  # m/m
        self.create_property('enable_fracture', True,     widget_type='checkbox')

    def run(self):
        preset = self.get_property('preset')
        if preset != 'Custom' and preset in CRASH_MATERIAL_PRESETS:
            p = CRASH_MATERIAL_PRESETS[preset]
            E   = p['E']
            nu  = p['nu']
            rho = p['rho']
            sy  = p['yield_strength']
            H   = p['tangent_modulus']
            ef  = p['failure_strain']
        else:
            E   = float(self.get_property('youngs_modulus'))
            nu  = float(self.get_property('poissons_ratio'))
            rho = float(self.get_property('density'))
            sy  = float(self.get_property('yield_strength'))
            H   = float(self.get_property('tangent_modulus'))
            ef  = float(self.get_property('failure_strain'))

        return {
            'E':              E,
            'nu':             nu,
            'rho':            rho,
            'yield_strength': sy,
            'tangent_modulus': H,
            'failure_strain': ef,
            'enable_fracture': bool(self.get_property('enable_fracture')),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Node 2: ImpactConditionNode
# ─────────────────────────────────────────────────────────────────────────────

class ImpactConditionNode(CadQueryNode):
    """
    Defines the crash / impact loading condition.

    The impact is represented as an **initial velocity field** applied to
    the nodes of the specified impact face (or the whole body if no face
    is provided). This is the standard approach for drop tests and
    barrier-impact simulations.

    Units: mm / ms = m/s  (consistent with mm–tonne–N–MPa–ms system).

    Tip:
    - 10 km/h ≈ 2 778 mm/ms → velocity_z = -2.778
    - 56 km/h (NCAP) ≈ 15 556 mm/ms → velocity_z = -15.556
    - Keep magnitude ≤ ~50 mm/ms for typical structural crash.
    """

    __identifier__ = 'com.cad.sim.impact'
    NODE_NAME = 'Impact Condition'

    def __init__(self):
        super().__init__()
        self.add_input('impact_face', color=(255, 100, 100))
        self.add_output('impact', color=(255, 200, 0))

        # Velocity components in mm/ms (= m/s)
        self.create_property('velocity_x',      0.0,  widget_type='float')
        self.create_property('velocity_y',      0.0,  widget_type='float')
        self.create_property('velocity_z',     -1.0,  widget_type='float')  # 1 m/s default
        # Node-selection tolerance (mm) – nodes within this distance of the
        # impact face are given the initial velocity.
        self.create_property('node_tolerance',  2.0,  widget_type='float')

    def run(self):
        face_data = self.get_input_value('impact_face', None)

        face_list = []
        if face_data is not None:
            if isinstance(face_data, dict):
                flist = face_data.get('faces', [])
                if not flist and face_data.get('face') is not None:
                    flist = [face_data['face']]
                face_list = flist
            elif hasattr(face_data, 'vals'):
                face_list = face_data.vals()
            else:
                face_list = [face_data]

        return {
            'face_list':      face_list,
            'velocity':       np.array([
                float(self.get_property('velocity_x')),
                float(self.get_property('velocity_y')),
                float(self.get_property('velocity_z')),
            ]),
            'node_tolerance': float(self.get_property('node_tolerance')),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Node 3: CrashSolverNode
# ─────────────────────────────────────────────────────────────────────────────

class CrashSolverNode(CadQueryNode):
    """
    Explicit Transient Crash / Impact Solver.

    Algorithm
    ---------
    Central-Difference (Leapfrog) time integration:
        v(n+½) = v(n-½) + Δt · a(n)
        u(n+1) = u(n)   + Δt · v(n+½)
        a(n+1) = M⁻¹ · [F_ext - F_int(u(n+1)) - C·v(n+½)]

    Mass matrix: diagonal (row-sum lumped) → explicit inversion.
    Damping:     Rayleigh mass-proportional  C ≈ α · M.
    Plasticity:  J2 von Mises, isotropic hardening, radial return.
    Fracture:    Element deletion when ε_p ≥ ε_failure.

    Inputs
    ------
    mesh          → from MeshNode (Netgen tet mesh)
    crash_material→ from CrashMaterialNode
    constraints   → one or more ConstraintNode outputs (multi-input)
    impact        → from ImpactConditionNode

    Output
    ------
    dict with keys compatible with the standard CAD Viewer:
        'type'            : 'crash'
        'mesh'            : deformed skfem mesh (for VTK rendering)
        'displacement'    : flat displacement array (same format as FEA Solver)
        'stress'          : selected field for colouring (see *visualization*)
        'visualization_mode': mode string passed to viewer
        plus crash-specific diagnostics:
        'peak_displacement', 'peak_stress', 'absorbed_energy',
        'n_failed', 'time', 'energy_kinetic', 'energy_strain', 'energy_plastic'
    """

    __identifier__ = 'com.cad.sim.crash_solver'
    NODE_NAME = 'Crash Solver'

    def __init__(self):
        super().__init__()
        self.add_input('mesh',           color=(200, 100, 200))
        self.add_input('crash_material', color=(255, 150, 50))
        self.add_input('constraints',    color=(255, 100, 100), multi_input=True)
        self.add_input('impact',         color=(255, 200,   0))
        self.add_output('crash_results', color=(0, 220, 255))

        # ── Simulation time ──────────────────────────────────────────────────
        # Total simulation duration in milliseconds.
        # Rule of thumb: use 2–10 × (L / c_wave) where c_wave ≈ √(E/ρ).
        # For aluminum at 5 m/s impact on a 150 mm body: ~2–5 ms.
        self.create_property('end_time',   2.0,  widget_type='float')   # ms
        # Minimum number of time steps requested.  The solver will automatically
        # increase this to satisfy the CFL stability condition (dt ≤ h_min/c_wave).
        # The value here is only used when it is already stable; otherwise the
        # CFL-derived step count is used instead.
        self.create_property('time_steps', 500,  widget_type='int')     # minimum steps

        # ── Damping (Rayleigh, mass-proportional only) ───────────────────────
        # α ≈ 2·ζ·ω_min  where ζ ≈ 0.05 (5%) and ω_min ≈ first natural freq.
        # Default 10 s⁻¹ gives ~5% damping at frequencies above ~32 Hz.
        self.create_property('damping_alpha', 10.0, widget_type='float')

        # ── Visualization field ───────────────────────────────────────────────
        self.create_property(
            'visualization', 'Von Mises Stress',
            widget_type='combo',
            items=['Von Mises Stress', 'Displacement', 'Plastic Strain', 'Failed Elements']
        )

        # ── Playback frames ───────────────────────────────────────────────────
        # Number of animation frames to record (equally spaced in time).
        # 0 = record every step (expensive), -1 = disable recording.
        self.create_property('n_frames', 120, widget_type='int')

        # ── Deformation scale (visualisation only) ────────────────────────────
        # Multiplies the displayed displacement so small deformations are easy
        # to see.  Does not affect the physics or result diagnostics.
        # 1.0 = true scale;  3.0 = typical for structural crash visualisation.
        self.create_property('disp_scale', 3.0, widget_type='float')

        # ── Co-rotational large-rotation correction ───────────────────────────
        # When enabled each element's rigid-body rotation R is extracted every
        # time step via polar decomposition of  F = R·U.  Stresses and forces
        # are expressed in the element-local frame (B matrices remain constant)
        # then rotated back to global — eliminating the artificial stiffening /
        # stress blow-up that the small-strain formulation produces when elements
        # rotate more than ~10°.   Recommended: True for any crash simulation.
        # Cost: ~2-3× more compute per step versus the small-strain path.
        self.create_property('enable_corotation', True, widget_type='checkbox')

        # ── Self-contact (CPU penalty) ────────────────────────────────────────
        # Prevents mesh surfaces from phasing through each other.
        # Uses cKDTree broad phase (O(N_b log N_b)) + vectorized narrow phase.
        # Enable only when folding / self-impact is expected; adds cost per step.
        self.create_property('enable_contact',  False, widget_type='checkbox')
        # Penalty stiffness multiplier relative to bulk modulus × L_min.
        # Higher values = stiffer contact (less penetration) but smaller stable dt.
        self.create_property('contact_stiffness', 0.1, widget_type='float')
        # Contact search radius as a fraction of the minimum element size.
        # Smaller = less false positives; too small = missed contacts.
        self.create_property('contact_thickness', 0.2, widget_type='float')
        # Rebuild the boundary-node list every N steps (1 = every step; more
        # expensive but mandatory when fracture is enabled and elements are
        # being deleted, exposing new interior nodes to contact).
        self.create_property('contact_update_interval', 10, widget_type='int')

        # ── Mass Scaling ──────────────────────────────────────────────────────
        # Artificially increases the mass of elements whose natural CFL
        # time-step is smaller than the requested step size (end_time/steps),
        # by a factor s_e = (dt_target/dt_e)².  Only critical (very small)
        # elements are affected; well-conditioned elements are unchanged.
        # This allows a larger global dt and fewer time steps on fine meshes.
        # Risk: increases total model mass.  A warning is printed when the mass
        # increase exceeds the threshold fraction (default 5 % — industry guideline).
        # Disable for quasi-static or inertia-sensitive analyses.
        self.create_property('enable_mass_scaling',    False, widget_type='checkbox')
        self.create_property('mass_scaling_threshold', 0.05,  widget_type='float')

    # ─────────────────────────────────────────────────────────────────────────

    def run(self):
        print("Crash Solver: Starting explicit transient crash simulation...")

        # ── 1. Gather inputs ─────────────────────────────────────────────────

        mesh     = self.get_input_value('mesh',           None)
        material = self.get_input_value('crash_material', None)
        impact   = self.get_input_value('impact',         None)

        def _flatten(items):
            flat = []
            if not items:
                return flat
            for it in items:
                if isinstance(it, list):
                    flat.extend(it)
                elif it is not None:
                    flat.append(it)
            return flat

        constraints = _flatten(self.get_input_list('constraints'))

        if mesh is None:
            self.set_error("Crash Solver: mesh input is not connected.")
            return None
        if material is None:
            self.set_error("Crash Solver: crash_material input is not connected.")
            return None

        # ── 2. Material parameters ───────────────────────────────────────────

        E        = float(material.get('E',               210000.0))
        nu       = float(material.get('nu',              0.30))
        rho      = float(material.get('rho',             7.85e-9))
        sigma_y0 = float(material.get('yield_strength',  250.0))
        H_hard   = float(material.get('tangent_modulus', 2000.0))
        eps_fail = float(material.get('failure_strain',  0.20))
        do_frac  = bool(material.get('enable_fracture',  True))

        _, mu = lam_lame(E, nu)       # shear modulus
        D = build_D_matrix(E, nu)

        # ── 3. Mesh topology ─────────────────────────────────────────────────

        p = mesh.p        # (3, N_nodes)
        t = mesh.t        # (4, N_elem)  – tet connectivity
        N      = p.shape[1]
        N_elem = t.shape[1]
        x, y, z = p

        print(f"Crash Solver: {N} nodes, {N_elem} elements.")

        # ── 4. Lumped mass matrix ─────────────────────────────────────────────

        M_diag      = assemble_lumped_mass(mesh, rho)
        total_mass  = np.sum(M_diag[0::3])          # scalar ≡ sum of nodal masses
        print(f"Crash Solver: Total mass = {total_mass * 1e3:.4f} kg")

        # Guard against zero-mass (bad mesh / wrong density)
        if total_mass < 1e-30:
            self.set_error("Crash Solver: computed mass is effectively zero – "
                           "check mesh and density units (expected tonne/mm³).")
            return None

        # ── 5. Boundary conditions (fixed DOFs) ──────────────────────────────

        fixed_dofs_set: set = set()

        from cadquery import Vector

        for constr in constraints:
            if not constr:
                continue
            fixed_dof_indices = constr.get('fixed_dofs', [0, 1, 2])
            geoms = constr.get('geometries', [constr.get('geometry')])
            geoms = [g for g in geoms if g is not None]
            if not geoms:
                print("Crash Solver: WARNING - constraint has no geometry, skipping.")
                continue

            # ── Per-face bounding-box pre-filter ──────────────────────────────
            tol = 1.5
            try:
                bboxes  = [g.BoundingBox() for g in geoms]
                xmin_bb = min(b.xmin for b in bboxes) - tol
                xmax_bb = max(b.xmax for b in bboxes) + tol
                ymin_bb = min(b.ymin for b in bboxes) - tol
                ymax_bb = max(b.ymax for b in bboxes) + tol
                zmin_bb = min(b.zmin for b in bboxes) - tol
                zmax_bb = max(b.zmax for b in bboxes) + tol
            except Exception as e:
                print(f"Crash Solver: WARNING - BoundingBox failed ({e}), skipping constraint.")
                continue

            in_bb = ((x >= xmin_bb) & (x <= xmax_bb) &
                     (y >= ymin_bb) & (y <= ymax_bb) &
                     (z >= zmin_bb) & (z <= zmax_bb))

            candidates = np.where(in_bb)[0]
            print(f"Crash Solver: Constraint BBox candidates: {len(candidates)}")

            for i in candidates:
                px, py, pz = float(x[i]), float(y[i]), float(z[i])
                pt = Vector(px, py, pz)
                matched = False
                for g in geoms:
                    # Primary check: distanceTo
                    try:
                        if g.distanceTo(pt) <= tol:
                            matched = True
                            break
                    except Exception:
                        pass
                    # Fallback: per-face tight-bbox proximity
                    if not matched:
                        try:
                            fb = g.BoundingBox()
                            if (fb.xmin - tol <= px <= fb.xmax + tol and
                                    fb.ymin - tol <= py <= fb.ymax + tol and
                                    fb.zmin - tol <= pz <= fb.zmax + tol):
                                matched = True
                                break
                        except Exception:
                            pass
                if matched:
                    for d in fixed_dof_indices:
                        fixed_dofs_set.add(3 * i + d)

        fixed_dofs = np.array(sorted(fixed_dofs_set), dtype=int)
        print(f"Crash Solver: {len(fixed_dofs)} constrained DOFs.")

        # ── 6. Initial conditions (velocity impact) ───────────────────────────

        u = np.zeros(3 * N)    # displacement
        v = np.zeros(3 * N)    # velocity
        impact_nodes: set = set()   # nodes receiving initial velocity
        v_imp = np.zeros(3)         # impact velocity vector

        if impact is not None:
            v_imp      = np.asarray(impact.get('velocity', [0.0, 0.0, 0.0]),
                                    dtype=float)
            face_list  = impact.get('face_list', [])
            face_tol   = float(impact.get('node_tolerance', 2.0))

            impact_nodes: set = set()

            if face_list:
                # Build a single merged bounding-box for all impact faces
                try:
                    bboxes = [f.BoundingBox() for f in face_list]
                    imp_xmin = min(b.xmin for b in bboxes) - face_tol
                    imp_xmax = max(b.xmax for b in bboxes) + face_tol
                    imp_ymin = min(b.ymin for b in bboxes) - face_tol
                    imp_ymax = max(b.ymax for b in bboxes) + face_tol
                    imp_zmin = min(b.zmin for b in bboxes) - face_tol
                    imp_zmax = max(b.zmax for b in bboxes) + face_tol
                    in_imp_bb = ((x >= imp_xmin) & (x <= imp_xmax) &
                                 (y >= imp_ymin) & (y <= imp_ymax) &
                                 (z >= imp_zmin) & (z <= imp_zmax))
                    imp_candidates = np.where(in_imp_bb)[0]
                except Exception as e:
                    print(f"Crash Solver: WARNING - impact face bbox failed ({e}), applying to all nodes.")
                    imp_candidates = np.arange(N)

                for ni in imp_candidates:
                    px, py, pz = float(x[ni]), float(y[ni]), float(z[ni])
                    pt = Vector(px, py, pz)
                    matched = False
                    for face in face_list:
                        # Primary: distanceTo
                        try:
                            if face.distanceTo(pt) <= face_tol:
                                matched = True
                                break
                        except Exception:
                            pass
                        # Fallback: per-face bbox proximity
                        if not matched:
                            try:
                                fb = face.BoundingBox()
                                if (fb.xmin - face_tol <= px <= fb.xmax + face_tol and
                                        fb.ymin - face_tol <= py <= fb.ymax + face_tol and
                                        fb.zmin - face_tol <= pz <= fb.zmax + face_tol):
                                    matched = True
                                    break
                            except Exception:
                                pass
                    if matched:
                        impact_nodes.add(ni)
            else:
                # No face specified → apply to all free nodes
                impact_nodes = set(range(N))

            for ni in impact_nodes:
                for d in range(3):
                    gdof = 3 * ni + d
                    if gdof not in fixed_dofs_set:
                        v[gdof] = v_imp[d]

            print(f"Crash Solver: Initial velocity applied to {len(impact_nodes)} nodes "
                  f"({v_imp} mm/ms).")

        # Enforce BCs on initial state
        v[fixed_dofs] = 0.0

        # Track impact DOFs for velocity ramp (avoids shock from step-function IC)
        # Collect the free DOFs of impact nodes that actually received velocity.
        impact_dof_list: list[int] = []
        impact_dof_vals: list[float] = []
        if len(impact_nodes) > 0 and np.any(v_imp != 0.0):
            for ni in impact_nodes:
                for d in range(3):
                    gdof = 3 * ni + d
                    if gdof not in fixed_dofs_set and v_imp[d] != 0.0:
                        impact_dof_list.append(gdof)
                        impact_dof_vals.append(v_imp[d])
        impact_dof_arr = np.array(impact_dof_list, dtype=int)
        impact_vel_arr = np.array(impact_dof_vals, dtype=float)
        # Ramp length: 50 steps or 3 % of total, whichever is larger (min 10)
        # Start with ZERO velocity; the ramp builds it up smoothly.
        if len(impact_dof_arr) > 0:
            v[impact_dof_arr] = 0.0   # will ramp up in the loop

        # ── 7. Time integration parameters ───────────────────────────────────

        end_time    = float(self.get_property('end_time'))
        n_steps_req = max(int(self.get_property('time_steps')), 1)
        dt_req      = end_time / n_steps_req
        alpha       = float(self.get_property('damping_alpha'))

        # ── Optional mass scaling — applied BEFORE the CFL check so that
        #    scaled elements count toward the global stable time step. ─────────
        _enable_ms    = bool(self.get_property('enable_mass_scaling'))
        _ms_threshold = float(self.get_property('mass_scaling_threshold'))
        if _enable_ms:
            M_diag, _ms_info = _apply_mass_scaling(
                M_diag, mesh, E, nu, rho, dt_req, _ms_threshold
            )
            print(
                f"Crash Solver: Mass scaling ON — "
                f"{_ms_info['n_scaled']} element(s) scaled, "
                f"Δmass = {_ms_info['mass_increase_fraction'] * 100:.2f}%, "
                f"max scale factor = {_ms_info['max_scale_factor']:.2f}×."
            )
            if _ms_info['mass_increase_fraction'] > _ms_threshold:
                print(
                    f"  WARNING: total mass increased by "
                    f"{_ms_info['mass_increase_fraction'] * 100:.1f}% "
                    f"(threshold {_ms_threshold * 100:.0f}%). "
                    "Results may be physically inaccurate for high-rate dynamic events."
                )
            # Update total_mass after scaling
            total_mass = float(np.sum(M_diag[0::3]))

        # CFL stability check — explicit central-difference requires
        #   dt ≤ L_e / c_d  (with a safety margin).
        # If the requested dt violates this, auto-reduce it.
        dt_cfl = _compute_cfl_dt(mesh, E, nu, rho, safety=0.5)
        if dt_req > dt_cfl:
            print(f"Crash Solver: WARNING – requested Δt ({dt_req:.4e} ms) exceeds "
                  f"CFL stable limit ({dt_cfl:.4e} ms).  Auto-correcting.")
            dt      = dt_cfl
            n_steps = max(int(np.ceil(end_time / dt)), 1)
        else:
            dt      = dt_req
            n_steps = n_steps_req

        viz_mode = self.get_property('visualization')

        # Playback frame recording
        n_frames      = int(self.get_property('n_frames'))
        if n_frames <= 0:
            n_frames = 0
        frame_interval = max(1, n_steps // max(n_frames, 1)) if n_frames > 0 else n_steps + 1
        frames: list = []   # list of {displacement, stress_vm, eps_p, failed, time}

        print(f"Crash Solver: Δt = {dt:.4e} ms,  steps = {n_steps}  "
              f"(CFL limit = {dt_cfl:.4e} ms)")

        # Per-element plastic state
        eps_p       = np.zeros(N_elem)
        failed_elem = np.zeros(N_elem, dtype=bool)

        # Per-element Cauchy stress tensor (6-component Voigt notation).
        # Tracked incrementally so the hypoelastic update σ += D·Δε never
        # re-applies accumulated elastic energy from previous steps.
        sigma_elem  = np.zeros((N_elem, 6))

        # History buffers (sampled)
        sample_interval = max(1, n_steps // 200)
        t_hist     = []
        KE_hist    = []
        SE_hist    = []
        PE_hist    = []

        # Plastic dissipation accumulator
        absorbed_energy = 0.0

        # ── Energy balance accumulators ───────────────────────────────────────
        # W_ext_cumulative : work done on the system by the prescribed-velocity
        #   impact BC (the external driver / impactor).  In explicit solvers the
        #   correct energy balance is:
        #     KE + SE_elastic + W_plastic_diss = KE_0 + W_ext_input
        #   Significant deviation (> ~10-15 %) indicates numerical instability or
        #   a time-step that is too large.
        # W_ext is computed incrementally each step as:
        #   dW_ext = (F_int + F_damp)[impact_dofs] · delta_u[impact_dofs]
        #   i.e. the work done by the constraint force overcoming structural
        #   and damping resistance at the prescribed DOFs.
        W_ext_cumulative = 0.0   # cumulative external work input  [N·mm]
        EB_hist          = []    # energy balance error ratio at each sample point
        _max_eb_error    = 0.0   # peak energy balance error (for post-summary)

        # ── 8a. Precompute element geometry (B matrices, volumes, DOF table) ─
        #        This is done ONCE; the mesh does not deform the connectivity.

        print("Crash Solver: Precomputing element geometry...")
        B_all, vol_all, dof_idx, Dm_inv_all = _precompute_elements(mesh)
        pts_ref = mesh.p.T.copy()          # (N_nodes, 3) reference coords
        conn    = mesh.t.T.copy()          # (N_elem, 4) connectivity
        print(f"Crash Solver: Precompute done. Starting time loop...")

        # Co-rotational flag
        enable_corotation = bool(self.get_property('enable_corotation'))
        print(f"Crash Solver: Co-rotational formulation = {enable_corotation}")

        # ── 8b-contact. Contact algorithm setup ──────────────────────────────
        enable_contact      = bool(self.get_property('enable_contact'))
        k_cf                = float(self.get_property('contact_stiffness'))
        ct_frac             = float(self.get_property('contact_thickness'))
        contact_update_int  = max(1, int(self.get_property('contact_update_interval')))

        # Characteristic element length (cube-root of median volume).
        _valid_vols = vol_all[vol_all > 1e-20]
        L_min = float(np.median(_valid_vols) ** (1.0 / 3.0)) if _valid_vols.size else 1.0

        # Bulk modulus K = E / (3(1−2ν))
        K_bulk      = E / (3.0 * (1.0 - 2.0 * nu))
        k_penalty   = k_cf * K_bulk * L_min          # [N/mm]
        ct_thickness = ct_frac * L_min               # search radius [mm]

        # Initial boundary-node list
        boundary_nodes = mesh.boundary_nodes() if enable_contact else np.array([], dtype=int)

        if enable_contact:
            print(f"Crash Solver: Self-contact ON  "
                  f"k_pen={k_penalty:.3e} N/mm  "
                  f"thickness={ct_thickness:.4f} mm  "
                  f"surface nodes={len(boundary_nodes)}")

        # Velocity ramp parameters
        n_ramp = max(10, min(50, n_steps // 20))  # ramp over first ~5 % of steps

        # Peak stress tracker (updated every step; not just the final value)
        peak_vm = 0.0

        # ── 8b. Initial internal forces and acceleration ──────────────────────
        # delta_u is zero at t=0 (no displacement yet), so F_int = 0 as well;
        # this call is kept for consistency and to initialise sigma_elem.
        _delta_u_init = np.zeros(3 * N)
        if enable_corotation:
            F_int, eps_p, sigma_vm, failed_elem, sigma_elem = _compute_forces_corot_vec(
                _delta_u_init, u, sigma_elem, D, eps_p, sigma_y0, H_hard, mu,
                failed_elem, do_frac, eps_fail,
                B_all, vol_all, dof_idx, Dm_inv_all, pts_ref, conn, N
            )
        else:
            F_int, eps_p, sigma_vm, failed_elem, sigma_elem = _compute_forces_vec(
                _delta_u_init, sigma_elem, D, eps_p, sigma_y0, H_hard, mu, failed_elem,
                do_frac, eps_fail, B_all, vol_all, dof_idx, N
            )

        F_damp = alpha * M_diag * v
        a = (- F_int - F_damp) / M_diag     # F_ext = 0 (velocity-driven impact)
        a[fixed_dofs] = 0.0

        # Initial kinetic energy (typically 0 because the velocity ramp starts
        # at v = 0; recorded here as the energy-balance reference baseline).
        KE_0 = 0.5 * float(np.dot(M_diag, v * v))

        # ── 9. Central-difference leapfrog time loop ──────────────────────────

        for step in range(n_steps):

            # v(n+½) = v(n-½) + Δt · a(n)
            v += dt * a

            # Prescribed velocity boundary condition for impact nodes.
            #
            # Physical interpretation: the impactor (e.g. a rigid barrier or
            # drop-weight) maintains a constant velocity throughout the event.
            # Only the first n_ramp steps use a smooth ramp to avoid the
            # initial shock that would otherwise cause immediate fracture from
            # a step-function velocity.  After the ramp, the nodes are HELD at
            # the full impact velocity for the rest of the simulation — this is
            # what drives sustained compression / crushing.
            #
            # Without this, the impact nodes are released after the ramp and
            # the elastic restoring force immediately decelerates them back to
            # ~0, producing <0.1 mm displacement for a 2 ms / 5 m·s⁻¹ run.
            if len(impact_dof_arr) > 0:
                if step < n_ramp:
                    ramp_factor = (step + 1) / n_ramp
                    v[impact_dof_arr] = ramp_factor * impact_vel_arr
                else:
                    v[impact_dof_arr] = impact_vel_arr   # sustain prescribed velocity

            v[fixed_dofs] = 0.0

            # Velocity cap – must be applied HERE, before the displacement
            # update, so that u is never integrated with an uncapped velocity.
            # (A cap placed after u += dt·v is useless: the damage is already
            # done to u, and the huge u drives huge F_int → explosive a.)
            #
            # Physical upper bound on particle velocity in a stress wave:
            #   v_max ≈ σ_y / (ρ·c_d)  ≈ impact velocity for typical metals.
            # A safety factor of 5× allows wave reflections and inertia
            # oscillations while bounding spurious velocities tightly enough
            # so that ghost-node displacement stays within a few mm.
            # Also enforce a minimum floor of 50 mm/ms so slow-impact sims
            # are not over-constrained.
            _v_cap = max(5.0 * float(np.max(np.abs(v_imp))), 50.0)
            np.clip(v, -_v_cap, _v_cap, out=v)
            v[fixed_dofs] = 0.0

            # Incremental displacement for this step (used by the hypoelastic
            # stress update inside _compute_forces_vec).  Must be captured
            # AFTER all velocity modifications and BEFORE the position update.
            delta_u = dt * v

            # u(n+1) = u(n) + Δt · v(n+½)
            u += delta_u
            u[fixed_dofs] = 0.0

            # Internal forces and plasticity update
            if enable_corotation:
                F_int, eps_p_new, sigma_vm_new, failed_new, sigma_elem = \
                    _compute_forces_corot_vec(
                        delta_u, u, sigma_elem, D, eps_p, sigma_y0, H_hard, mu,
                        failed_elem, do_frac, eps_fail,
                        B_all, vol_all, dof_idx, Dm_inv_all, pts_ref, conn, N
                    )
            else:
                F_int, eps_p_new, sigma_vm_new, failed_new, sigma_elem = \
                    _compute_forces_vec(
                        delta_u, sigma_elem, D, eps_p, sigma_y0, H_hard, mu,
                        failed_elem, do_frac, eps_fail, B_all, vol_all, dof_idx, N
                    )

            # Progress report every 10 %
            if step % max(1, n_steps // 10) == 0:
                pct = 100 * step // n_steps
                KE_now = 0.5 * float(np.dot(M_diag, v * v))
                print(f"Crash Solver: {pct:3d}%  step {step}/{n_steps}  "
                      f"KE={KE_now:.3e} N·mm  failed={int(np.sum(failed_new))}")

            # Plastic dissipation: W_p ≈ Σ_e Δε_p · σ_y(ε_p) · V_e
            # Only sum over active (non-failed) elements; only take positive
            # plastic-strain increments; guard against non-finite values that
            # would corrupt the accumulator during any transient instability.
            _active      = ~failed_elem
            _deps_p      = np.maximum(eps_p_new - eps_p, 0.0)   # positive only
            _sigma_y_now = sigma_y0 + H_hard * eps_p
            _dW          = _deps_p * _sigma_y_now * vol_all
            absorbed_energy += float(np.nansum(
                np.where(_active & np.isfinite(_dW), _dW, 0.0)
            ))

            # Track peak Von Mises stress over the whole simulation
            if np.any(~failed_new):
                peak_vm = max(peak_vm, float(np.max(sigma_vm_new[~failed_new])))

            # Detect newly deleted elements (this step)
            _new_failures = failed_new & ~failed_elem

            eps_p       = eps_p_new
            sigma_vm    = sigma_vm_new
            failed_elem = failed_new

            # ── Ghost-node suppression ─────────────────────────────────────
            # When elements are deleted the nodes they shared may become
            # completely disconnected from the live mesh.  Without any spring
            # force these "ghost" nodes free-fly at whatever velocity they
            # happened to have, producing astronomically large displacements
            # and corrupting the energy accumulators.  Zero their kinematics
            # every time a new deletion batch occurs.
            if np.any(_new_failures):
                _live_mask = ~failed_elem
                if np.any(_live_mask):
                    _live_nodes = np.unique(mesh.t[:, _live_mask])
                else:
                    _live_nodes = np.array([], dtype=int)
                _dead_nodes = np.setdiff1d(np.arange(N), _live_nodes)
                if _dead_nodes.size:
                    _dofs_dead = (np.repeat(_dead_nodes * 3, 3) +
                                  np.tile(np.arange(3), _dead_nodes.size))
                    v[_dofs_dead] = 0.0      # kill velocity → no free-flight
                    # NOTE: do NOT zero u here.  Displacement already accumulated
                    # by these nodes is physically real deformation that happened
                    # before element death.  Zeroing it erases the deformation
                    # history and makes the structure appear to have never moved.

                # Zero the stored stress for newly deleted elements so the
                # incremental update does not start from a stale high-stress
                # state if those element slots are ever re-used.
                sigma_elem[_new_failures] = 0.0

            # ── Contact: self-contact penalty forces ─────────────────────
            # boundary_nodes is kept up-to-date on two triggers:
            #   1) Periodically (contact_update_interval steps) so gradual
            #      buckling exposes new surface nodes incrementally.
            #   2) Immediately when new element failures occur, so freshly
            #      exposed interior nodes join the contact surface at once
            #      (prevents ghost-node tunnelling through neighbours).
            if enable_contact:
                should_rebuild = (
                    (step % contact_update_int == 0) or
                    (do_frac and np.any(_new_failures))
                )
                if should_rebuild:
                    boundary_nodes = _update_boundary_nodes(mesh, failed_elem)

                F_contact = compute_penalty_contact(
                    u, mesh.p.T, boundary_nodes, ct_thickness, k_penalty
                )
            else:
                F_contact = 0.0

            # Rayleigh damping force
            F_damp = alpha * M_diag * v

            # ── Energy balance: work done by impactor (kinematic BC) ──────────
            # The prescribed-velocity constraint at impact nodes requires an
            # external force from the impactor to overcome the structural
            # resistance F_int and damping force F_damp at those DOFs.
            #
            # Sign convention:  in this solver  a = (-F_int - F_damp) / M,
            # so F_int and F_damp are RESTORING forces (they oppose motion and
            # point in the -impact direction).  The constraint force the
            # impactor must supply is:
            #     F_ext = -(F_int + F_damp)   [at impact nodes, after ramp]
            # giving POSITIVE work for positive displacement:
            #     dW_ext = F_ext · Δu = -(F_int + F_damp) · Δu  > 0
            #
            # Using +F_int·Δu (wrong sign) would give negative W_ext →
            # E_ref collapses toward zero → spurious 1000s-% EB errors.
            if len(impact_dof_arr) > 0 and step >= n_ramp:
                _F_constr = F_int[impact_dof_arr] + F_damp[impact_dof_arr]
                W_ext_cumulative -= float(np.dot(_F_constr, delta_u[impact_dof_arr]))

            # New acceleration (contact force enters with positive sign — it is
            # already a restoring/repulsive force, not an internal stress)
            a = (- F_int + F_contact - F_damp) / M_diag
            a[fixed_dofs] = 0.0

            # Sample history
            if step % sample_interval == 0:
                KE = 0.5 * float(np.dot(M_diag, v * v))
                SE = 0.5 * float(np.dot(u, F_int))
                t_hist.append(step * dt)
                KE_hist.append(KE)
                SE_hist.append(max(SE, 0.0))
                PE_hist.append(absorbed_energy)

                # ── Energy balance check ───────────────────────────────────
                # Correct balance:  KE + SE_elastic + PD = KE_0 + W_ext_input
                # The check is only useful once meaningful energy is in the
                # system.  Guard against near-zero E_ref (early ramp phase
                # or trivially elastic steps) with a 1 N·mm minimum:  below
                # this level the ratio has no practical meaning.
                # Warn only when the error first crosses the threshold or has
                # risen by >5% since the last warning, to avoid log spam.
                E_total = KE + max(SE, 0.0) + absorbed_energy
                E_ref   = KE_0 + W_ext_cumulative
                _eb_min_energy = max(1.0, KE_0 * 10.0)  # meaningful energy floor [N·mm]
                if E_ref > _eb_min_energy and E_total > _eb_min_energy and step >= n_ramp:
                    _eb_err = abs(E_total - E_ref) / E_ref
                    _prev_max = _max_eb_error
                    _max_eb_error = max(_max_eb_error, _eb_err)
                    EB_hist.append(_eb_err)
                    # Only emit a new warning when the error first crosses a
                    # decade boundary (15 / 30 / 50 / 100 %) to avoid flooding.
                    _thresholds = [0.50, 0.30, 0.15]
                    for _thr in _thresholds:
                        if _eb_err > _thr and _prev_max <= _thr:
                            logger.warning(
                                f"Crash Solver: Energy balance error crossed "
                                f"{_thr*100:.0f}% threshold "
                                f"({_eb_err * 100:.1f}% at t = {step * dt:.4f} ms). "
                                "This may indicate numerical instability — consider "
                                "reducing the time step or improving mesh quality."
                            )
                            break
                else:
                    EB_hist.append(0.0)

            # Record animation frame
            if n_frames > 0 and (step % frame_interval == 0 or step == n_steps - 1):
                nvm = sigma_vm_to_nodal(mesh, sigma_vm)
                nep = sigma_vm_to_nodal(mesh, eps_p)
                nfa = sigma_vm_to_nodal(mesh, failed_elem.astype(float))
                frames.append({
                    'displacement': u.copy(),
                    'stress_vm':    nvm,
                    'eps_p':        nep,
                    'failed':       nfa,
                    'time':         (step + 1) * dt,
                })

        # ── 10. Post-processing ───────────────────────────────────────────────

        # Max displacement over ALL nodes.  Ghost nodes have their velocity
        # zeroed when their last element is deleted, so their displacement is
        # frozen at the last physically meaningful position – that IS part of
        # the real deformation field and should be included in the report.
        disp_3n     = u.reshape(N, 3)
        disp_mag    = np.linalg.norm(disp_3n, axis=1)
        max_disp_live = float(np.max(disp_mag)) if N > 0 else 0.0

        print(f"Crash Solver: Simulation complete.")
        print(f"  Peak VM stress    = {peak_vm:.1f} MPa")
        print(f"  Max displacement  = {max_disp_live:.3f} mm")
        print(f"  Failed elements   = {int(np.sum(failed_elem))}")
        print(f"  Absorbed energy   ≈ {absorbed_energy:.3f} N·mm")

        # Energy balance summary
        if _max_eb_error > 0.0:
            _eb_status = (
                "PASS" if _max_eb_error <= 0.10 else
                "WARN" if _max_eb_error <= 0.20 else
                "FAIL"
            )
            print(
                f"  Energy balance    = {_eb_status}  "
                f"(max error {_max_eb_error * 100:.1f}%)"
            )
            if _max_eb_error > 0.20:
                print(
                    "  WARNING: Large energy balance error may indicate numerical "
                    "instability.  Consider reducing the time step or refining the mesh."
                )

        # Nodal Von Mises (element average → nodes)
        nodal_vm = sigma_vm_to_nodal(mesh, sigma_vm)

        # Nodal equivalent plastic strain (element → nodes)
        nodal_eps_p = sigma_vm_to_nodal(mesh, eps_p)

        # Nodal failed-element flag (any connected element failed)
        nodal_failed = sigma_vm_to_nodal(mesh, failed_elem.astype(float))

        # disp_3n / disp_mag already computed in the max-displacement block above.

        # Select the primary field for the viewer colour map
        if viz_mode == 'Von Mises Stress':
            view_field = nodal_vm
        elif viz_mode == 'Displacement':
            view_field = disp_mag
        elif viz_mode == 'Plastic Strain':
            view_field = nodal_eps_p
        elif viz_mode == 'Failed Elements':
            view_field = nodal_failed
        else:
            view_field = nodal_vm

        # The viewer expects displacement in this specific flat order:
        #   [u0x, u0y, u0z, u1x, u1y, u1z, ...]
        # which reshapes to (3, N) via reshape((3, N), order='F').
        # Our layout is already [3*i, 3*i+1, 3*i+2] per node → pass directly.
        displacement_flat = u   # shape (3*N,), Viewer reshapes with order='F'

        return {
            # ── Viewer-compatible fields ──────────────────────────────────
            'type':               'crash',
            'mesh':               mesh,
            'displacement':       displacement_flat,
            'stress':             view_field,
            'visualization_mode': viz_mode,
            'disp_scale':         float(self.get_property('disp_scale')),

            # ── Animation frames (for playback) ──────────────────────────
            'frames':             frames,

            # ── Crash-specific diagnostics ────────────────────────────────
            'element_stress':     sigma_vm,
            'plastic_strain':     eps_p,
            'failed_elements':    failed_elem,

            # ── Energy history ────────────────────────────────────────────
            'time':               np.array(t_hist),
            'energy_kinetic':     np.array(KE_hist),
            'energy_strain':      np.array(SE_hist),
            'energy_plastic':     np.array(PE_hist),
            'energy_balance':     np.array(EB_hist),   # error ratio (0 = perfect balance)

            # ── Scalar summary ────────────────────────────────────────────
            'peak_displacement':  max_disp_live,
            'peak_stress':        peak_vm,
            'absorbed_energy':    float(absorbed_energy),
            'n_failed':           int(np.sum(failed_elem)),
            'energy_balance_max_error': float(_max_eb_error),
        }


# ─────────────────────────────────────────────────────────────────────────────
# GPU / Taichi  –  Neo-Hookean Hyperelastic + J2 Elasto-Plastic Solver
# ─────────────────────────────────────────────────────────────────────────────
#
# Mathematical formulation
# ─────────────────────────
#  Deformation gradient:    F   = Ds · Dm⁻¹
#  Volume ratio:            J   = det(F)
#  1st Piola-Kirchhoff:     P   = μ(F − F⁻ᵀ) + λ·ln(J)·F⁻ᵀ   [Neo-Hookean]
#  Kirchhoff stress:        τ   = P · Fᵀ
#  Cauchy VM stress:        σ_vm = √(3/2 dev(τ/J):dev(τ/J))
#  Plasticity return map:   if σ_vm > σ_y + H·ε_p :
#                               s = (σ_y + H·ε_p) / σ_vm
#                               P ← P·s  (deviatoric scale-back)
#                               ε_p += (σ_vm − σ_y_curr) / (3μ + H)
#  Element death:           if ε_p ≥ ε_failure → fail (zero forces)
#  Nodal force matrix:      H   = −W₀ · P · Dm⁻ᵀ
#  Scatter to 4 tet nodes:  f₁..₃ = columns of H,  f₀ = −(f₁+f₂+f₃)
#
# Contact mechanics (penalty)
# ────────────────────────────
#  Ground plane:            F_y = k_p · max(0, y_floor − y_i)
#  Self-contact (spatial hash):
#    Build: each surface face → hashed to centroid grid cell
#    Query: for each surface node, search 5×5×5 neighbourhood
#    Force: d = n̂·(x_i − x_face_vertex),  if d < 0 → F = k_p·|d|·n̂
#
# Time integration: Central-Difference leapfrog (same as CrashSolverNode)
# Parallelisation:  @ti.kernel → compiled to CUDA / Vulkan / Metal / CPU
# ─────────────────────────────────────────────────────────────────────────────

# Spatial-hash grid dimensions (compile-time constants used inside ti.kernel)
_SC_GRID_RES:     int = 32    # cells per axis  (32³ = 32 768 cells)
_SC_MAX_PER_CELL: int = 32    # max surface faces stored per cell

_taichi_init_done: bool = False
_taichi_available: bool = False


def _try_init_taichi() -> bool:
    """
    One-shot Taichi runtime initialisation.

    Tries hardware backends in priority order: CUDA → Vulkan → CPU.
    Stores the result in module-level flags so subsequent calls are free.
    Returns True when Taichi is ready to use.
    """
    global _taichi_init_done, _taichi_available
    if _taichi_init_done:
        return _taichi_available
    _taichi_init_done = True
    try:
        import taichi as ti                                 # noqa: F401
        for arch in (ti.cuda, ti.vulkan, ti.cpu):
            try:
                ti.init(arch=arch, log_level=ti.WARN, default_fp=ti.f64)
                _taichi_available = True
                logger.info("Taichi backend active: %s", arch)
                break
            except Exception:
                continue
        else:
            _taichi_available = False
            logger.warning("Taichi: no suitable backend found.")
    except ImportError:
        logger.warning("Taichi not installed.  Install with: pip install taichi")
    except Exception as exc:
        logger.warning("Taichi init error: %s", exc)
    return _taichi_available


def _run_neo_hookean_sim(
        mesh,
        mu: float,
        la: float,
        rho: float,
        dt: float,
        n_steps: int,
        alpha: float,
        n_frames: int,
        fixed_dofs_set: set,
        impact_dof_arr: np.ndarray,
        impact_vel_arr: np.ndarray,
        n_ramp: int,
        # ── Plasticity ──────────────────────────────────────────────────────
        sigma_y0: float = 1e30,         # yield stress [MPa]  (1e30 = no yield)
        H_hard:   float = 0.0,          # isotropic hardening modulus [MPa]
        eps_failure: float = 1e30,      # failure strain threshold (1e30 = no fracture)
        enable_fracture: bool = False,
        # ── Contact ─────────────────────────────────────────────────────────
        k_contact_factor: float = 10.0, # penalty stiffness = k_contact_factor × bulk
        enable_floor:     bool  = True, # rigid ground plane at y = y_floor
        y_floor:          float = 0.0,  # floor y-coordinate [mm]
        enable_self_contact: bool = False,
        # ────────────────────────────────────────────────────────────────────
        disp_scale: float = 1.0,
) -> dict:
    """
    GPU-accelerated explicit tetrahedral FEM with:
      • Neo-Hookean hyperelasticity (large deformation)
      • J2 elasto-plasticity with isotropic hardening (GPU radial return)
      • Element deletion at fracture strain
      • Rigid ground-plane penalty contact
      • Self-contact via spatial-hash penalty (optional)

    Parameters / Returns: see CrashSolverGPUNode docstring.
    """
    import taichi as ti

    GR  = _SC_GRID_RES
    MPC = _SC_MAX_PER_CELL

    N      = mesh.p.shape[1]
    N_elem = mesh.t.shape[1]
    pts    = mesh.p.T           # (N,   3) reference positions
    conn   = mesh.t.T           # (Ne,  4) tet connectivity

    # ── 1. Reference geometry: Dm⁻¹ and W₀ ──────────────────────────────────
    e0     = pts[conn[:, 0]]
    Dm_all = np.stack(
        [pts[conn[:, k]] - e0 for k in range(1, 4)], axis=2
    ).astype(np.float64)                                  # (Ne, 3, 3)

    det_Dm = np.linalg.det(Dm_all)
    W_0    = np.abs(det_Dm) / 6.0
    valid  = W_0 > 1e-20

    Dm_inv_all = np.zeros_like(Dm_all)
    if np.any(valid):
        Dm_inv_all[valid] = np.linalg.inv(Dm_all[valid])

    # ── 2. Lumped nodal mass ──────────────────────────────────────────────────
    M_np = np.zeros(N, dtype=np.float64)
    np.add.at(M_np, conn.ravel(), np.repeat(rho * W_0 / 4.0, 4))
    M_np = np.maximum(M_np, 1e-30)

    # Penalty stiffness = k_contact_factor × bulk modulus K = λ + 2/3 μ
    K_bulk  = la + 2.0 / 3.0 * mu
    k_pen   = k_contact_factor * K_bulk

    # ── 3. BC encoding ───────────────────────────────────────────────────────
    bc_kind_np = np.zeros((N, 3), dtype=np.int32)   # 0=free, 1=fixed, 2=prescribed
    bc_val_np  = np.zeros((N, 3), dtype=np.float64)

    for dof in fixed_dofs_set:
        ni, d = divmod(int(dof), 3)
        bc_kind_np[ni, d] = 1

    for dof_i, val in zip(impact_dof_arr, impact_vel_arr):
        ni, d = divmod(int(dof_i), 3)
        if bc_kind_np[ni, d] == 0:
            bc_kind_np[ni, d] = 2
            bc_val_np[ni, d]  = float(val)

    # ── 4. Surface face extraction for contact ────────────────────────────────
    # skfem: mesh.facets[:, mesh.boundary_facets()] → (3, N_surf_f) node indices
    surf_f_idx    = mesh.boundary_facets()                  # boundary facet indices
    surf_tris_np  = mesh.facets[:, surf_f_idx].T.astype(np.int32)  # (Ns, 3)
    N_surf        = surf_tris_np.shape[0]

    # Outward unit normals in reference config (pointing outward from solid)
    _p0  = pts[surf_tris_np[:, 0]]
    _e1  = pts[surf_tris_np[:, 1]] - _p0
    _e2  = pts[surf_tris_np[:, 2]] - _p0
    _cr  = np.cross(_e1, _e2)
    _ln  = np.linalg.norm(_cr, axis=1, keepdims=True)
    _ln  = np.where(_ln < 1e-20, 1.0, _ln)
    surf_norms_np = (_cr / _ln).astype(np.float64)          # (Ns, 3)

    # Boolean mask: is node on the surface?
    surf_node_flag_np = np.zeros(N, dtype=np.int32)
    surf_node_flag_np[np.unique(surf_tris_np.ravel())] = 1

    # Spatial hash bounding box (padded 10 %)
    _pad      = 0.1 * max(np.ptp(pts[:, 0]),
                           np.ptp(pts[:, 1]),
                           np.ptp(pts[:, 2]), 1.0)
    bbox_min  = pts.min(axis=0) - _pad
    bbox_max  = pts.max(axis=0) + _pad
    cell_size = float(np.max(bbox_max - bbox_min) / (GR - 1))

    # ── 5. Taichi field allocation ────────────────────────────────────────────
    x_ti    = ti.Vector.field(3, dtype=ti.f64, shape=N)
    v_ti    = ti.Vector.field(3, dtype=ti.f64, shape=N)
    f_ti    = ti.Vector.field(3, dtype=ti.f64, shape=N)
    m_ti    = ti.field(dtype=ti.f64, shape=N)
    elem_ti = ti.Vector.field(4, dtype=ti.i32, shape=N_elem)
    B_ti    = ti.Matrix.field(3, 3, dtype=ti.f64, shape=N_elem)
    W_ti    = ti.field(dtype=ti.f64, shape=N_elem)
    bkind   = ti.Vector.field(3, dtype=ti.i32, shape=N)
    bval    = ti.Vector.field(3, dtype=ti.f64, shape=N)

    # Plasticity state (per element)
    eps_p_ti   = ti.field(dtype=ti.f64, shape=N_elem)   # equiv plastic strain
    failed_ti  = ti.field(dtype=ti.i32, shape=N_elem)   # 0=alive, 1=dead

    # Contact: surface geometry
    surf_faces_ti = ti.Vector.field(3, dtype=ti.i32, shape=N_surf)
    surf_norms_ti = ti.Vector.field(3, dtype=ti.f64, shape=N_surf)
    surf_flag_ti  = ti.field(dtype=ti.i32, shape=N)     # 1 = on boundary

    # Spatial hash grid (face centroid → cell)
    grid_cnt  = ti.field(dtype=ti.i32, shape=(GR, GR, GR))
    grid_data = ti.field(dtype=ti.i32, shape=(GR, GR, GR, MPC))

    x_ti.from_numpy(pts.copy())
    v_ti.from_numpy(np.zeros((N, 3), dtype=np.float64))
    f_ti.from_numpy(np.zeros((N, 3), dtype=np.float64))
    m_ti.from_numpy(M_np)
    elem_ti.from_numpy(conn.astype(np.int32))
    B_ti.from_numpy(Dm_inv_all)
    W_ti.from_numpy(W_0.astype(np.float64))
    bkind.from_numpy(bc_kind_np)
    bval.from_numpy(bc_val_np)
    eps_p_ti.from_numpy(np.zeros(N_elem, dtype=np.float64))
    failed_ti.from_numpy(np.zeros(N_elem, dtype=np.int32))
    surf_faces_ti.from_numpy(surf_tris_np)
    surf_norms_ti.from_numpy(surf_norms_np)
    surf_flag_ti.from_numpy(surf_node_flag_np)

    # Capture Python scalars for use inside Taichi kernels
    # (Taichi traces these once at kernel-specialisation time)
    _mu   = float(mu)
    _la   = float(la)
    _sy0  = float(sigma_y0)
    _H    = float(H_hard)
    _ef   = float(eps_failure)
    _kpen = float(k_pen)
    _yfl  = float(y_floor)
    _bmx  = float(bbox_min[0])
    _bmy  = float(bbox_min[1])
    _bmz  = float(bbox_min[2])
    _cs   = float(cell_size)

    # ── 6. Taichi GPU kernels ─────────────────────────────────────────────────

    @ti.kernel
    def k_zero_forces():
        for i in f_ti:
            f_ti[i] = ti.Vector([0.0, 0.0, 0.0])

    @ti.kernel
    def k_internal_forces():
        """
        Neo-Hookean internal forces with in-kernel J2 plasticity return mapping.

        For each tet element (all in parallel):
          1. Compute F = Ds · Dm⁻¹, J = det(F)
          2. Compute P (Neo-Hookean PK1 stress)
          3. Compute Kirchhoff stress τ = P · Fᵀ  → von Mises σ_vm
          4. If σ_vm > yield surface: scale P back (return mapping), update ε_p
          5. If ε_p ≥ ε_failure: mark element dead, skip force scatter
          6. Assemble H = −W₀ · P · Dm⁻ᵀ, scatter to 4 nodes (atomic)
        """
        for e in range(N_elem):
            if W_ti[e] < 1e-20:
                continue
            if failed_ti[e] == 1:
                continue

            idx = elem_ti[e]
            x0 = x_ti[idx[0]]; x1 = x_ti[idx[1]]
            x2 = x_ti[idx[2]]; x3 = x_ti[idx[3]]

            Ds  = ti.Matrix.cols([x1 - x0, x2 - x0, x3 - x0])
            F   = Ds @ B_ti[e]
            J   = F.determinant()
            if J <= 1e-8:
                continue

            F_inv_T = F.inverse().transpose()

            # ── Neo-Hookean PK1 stress ──────────────────────────────────────
            P = _mu * (F - F_inv_T) + _la * ti.log(J) * F_inv_T

            # ── J2 Plasticity return mapping ────────────────────────────────
            # Kirchhoff stress tensor τ = P · Fᵀ  (symmetric 3×3)
            tau = P @ F.transpose()

            # Hydrostatic pressure from Kirchhoff
            p_tau = (tau[0, 0] + tau[1, 1] + tau[2, 2]) / 3.0

            # Deviatoric Kirchhoff
            s00 = tau[0, 0] - p_tau
            s11 = tau[1, 1] - p_tau
            s22 = tau[2, 2] - p_tau
            s01 = 0.5 * (tau[0, 1] + tau[1, 0])
            s02 = 0.5 * (tau[0, 2] + tau[2, 0])
            s12 = 0.5 * (tau[1, 2] + tau[2, 1])

            # von Mises from Kirchhoff (= J × Cauchy VM)
            vm_kirch = ti.sqrt(ti.max(
                1.5 * (s00*s00 + s11*s11 + s22*s22
                       + 2.0*(s01*s01 + s02*s02 + s12*s12)),
                0.0
            ))
            # Cauchy von Mises = Kirchhoff VM / J
            sigma_vm_e = vm_kirch / J

            # Current yield stress
            sigma_y_curr = _sy0 + _H * eps_p_ti[e]

            # Radial return: scale P back to yield surface
            if sigma_vm_e > sigma_y_curr and sigma_vm_e > 1e-12:
                scale     = sigma_y_curr / sigma_vm_e
                delta_ep  = (sigma_vm_e - sigma_y_curr) / (3.0 * _mu + _H)
                eps_p_ti[e] = eps_p_ti[e] + delta_ep
                P = P * scale   # exact for affine scaling; approximate for general

                # Element death check (compile-time eliminated when fracture disabled)
                if ti.static(enable_fracture):
                    if eps_p_ti[e] >= _ef:
                        failed_ti[e] = 1
                        continue

            # ── Nodal force matrix H = −W₀ · P · Dm⁻ᵀ ──────────────────────
            H_mat = -W_ti[e] * P @ B_ti[e].transpose()

            f1 = ti.Vector([H_mat[0, 0], H_mat[1, 0], H_mat[2, 0]])
            f2 = ti.Vector([H_mat[0, 1], H_mat[1, 1], H_mat[2, 1]])
            f3 = ti.Vector([H_mat[0, 2], H_mat[1, 2], H_mat[2, 2]])
            f0 = -(f1 + f2 + f3)

            # Atomic scatter to shared nodes
            f_ti[idx[0]] += f0
            f_ti[idx[1]] += f1
            f_ti[idx[2]] += f2
            f_ti[idx[3]] += f3

    @ti.kernel
    def k_grid_build():
        """
        Rebuild the spatial hash each step: hash each surface face by its
        centroid into a 3-D grid cell.  Updates surface normals to current
        deformed configuration simultaneously.
        """
        # Zero all cell counts
        for cx, cy, cz in ti.ndrange(GR, GR, GR):
            grid_cnt[cx, cy, cz] = 0

        # Insert each surface face
        for fi in range(N_surf):
            fni = surf_faces_ti[fi]
            p0  = x_ti[fni[0]]
            p1  = x_ti[fni[1]]
            p2  = x_ti[fni[2]]

            # Update deformed face normal
            e1    = p1 - p0
            e2    = p2 - p0
            n_raw = e1.cross(e2)
            n_len = n_raw.norm()
            if n_len > 1e-20:
                surf_norms_ti[fi] = n_raw / n_len

            # Hash to centroid cell
            cen = (p0 + p1 + p2) / 3.0
            cx  = int((cen[0] - _bmx) / _cs)
            cy  = int((cen[1] - _bmy) / _cs)
            cz  = int((cen[2] - _bmz) / _cs)
            cx  = ti.max(0, ti.min(GR - 1, cx))
            cy  = ti.max(0, ti.min(GR - 1, cy))
            cz  = ti.max(0, ti.min(GR - 1, cz))

            slot = ti.atomic_add(grid_cnt[cx, cy, cz], 1)
            if slot < MPC:
                grid_data[cx, cy, cz, slot] = fi

    @ti.kernel
    def k_contact():
        """
        Penalty contact forces:
          A) Ground-plane rigid contact (always active when enable_floor=True)
          B) Self-contact via spatial hash (active when enable_self_contact=True)

        Force law:  F_contact = k_pen × penetration_depth × outward_normal
        Applied to the penetrating node; equal-&-opposite split across face nodes.
        """
        for ni in range(N):
            xi = x_ti[ni]

            # ── A. Ground plane ─────────────────────────────────────────────
            if ti.static(enable_floor):
                pen_y = _yfl - xi[1]
                if pen_y > 0.0:
                    f_ti[ni] += ti.Vector([0.0, _kpen * pen_y, 0.0])

            # ── B. Self-contact ─────────────────────────────────────────────
            if ti.static(enable_self_contact):
                if surf_flag_ti[ni] == 0:
                    continue   # only test boundary nodes against faces

                # Grid cell of this node
                cx0 = int((xi[0] - _bmx) / _cs)
                cy0 = int((xi[1] - _bmy) / _cs)
                cz0 = int((xi[2] - _bmz) / _cs)

                # Search 5×5×5 neighbourhood (compile-time unroll)
                for dcx in ti.static(range(-2, 3)):
                    for dcy in ti.static(range(-2, 3)):
                        for dcz in ti.static(range(-2, 3)):
                            cx = cx0 + dcx
                            cy = cy0 + dcy
                            cz = cz0 + dcz
                            if 0 <= cx < GR and 0 <= cy < GR and 0 <= cz < GR:
                                n_faces = ti.min(grid_cnt[cx, cy, cz], MPC)
                                for fi in range(n_faces):
                                    fid = grid_data[cx, cy, cz, fi]
                                    fni = surf_faces_ti[fid]

                                    # Skip face sharing this node (self-element)
                                    if (ni == fni[0] or
                                            ni == fni[1] or
                                            ni == fni[2]):
                                        continue

                                    # Skip dead faces (all nodes of failed elem)
                                    # (approximation: check first contact node)
                                    if surf_flag_ti[fni[0]] == 0:
                                        continue

                                    # Signed distance from node to face plane
                                    n_hat = surf_norms_ti[fid]
                                    pa    = x_ti[fni[0]]
                                    d     = n_hat.dot(xi - pa)

                                    # Penetration: node on inside of face (d < 0)
                                    if d < 0.0:
                                        pen_d   = -d
                                        f_cont  = _kpen * pen_d * n_hat
                                        # Push node out
                                        f_ti[ni] += f_cont
                                        # Equal-&-opposite to face nodes (1/3 each)
                                        f_ti[fni[0]] -= f_cont * (1.0 / 3.0)
                                        f_ti[fni[1]] -= f_cont * (1.0 / 3.0)
                                        f_ti[fni[2]] -= f_cont * (1.0 / 3.0)

    @ti.kernel
    def k_advance(dt_: ti.f64, alpha_: ti.f64, ramp_: ti.f64):
        """
        Central-difference velocity + position update with BC enforcement.
        """
        for i in range(N):
            acc = f_ti[i] / m_ti[i] - alpha_ * v_ti[i]
            v_ti[i] += dt_ * acc

            for d in ti.static(range(3)):
                kk = bkind[i][d]
                if kk == 1:
                    v_ti[i][d] = 0.0
                elif kk == 2:
                    v_ti[i][d] = ramp_ * bval[i][d]

            x_ti[i] += dt_ * v_ti[i]

    # ── 7. Time-loop set-up ───────────────────────────────────────────────────
    frame_interval = max(1, n_steps // max(n_frames, 1)) if n_frames > 0 else n_steps + 1
    frames: list  = []
    sample_int    = max(1, n_steps // 200)
    t_hist:  list = []
    KE_hist: list = []
    PE_hist: list = []  # plastic dissipation

    absorbed_energy = 0.0

    print(f"GPU Solver (plasticity={'on' if sigma_y0 < 1e29 else 'off'}, "
          f"fracture={'on' if enable_fracture else 'off'}, "
          f"floor={'on' if enable_floor else 'off'}, "
          f"self-contact={'on' if enable_self_contact else 'off'}): "
          f"{N} nodes, {N_elem} elements, {n_steps} steps, dt={dt:.4e} ms")

    # Prime the force fields before entering the loop
    k_zero_forces()
    if enable_self_contact:
        k_grid_build()
    k_internal_forces()
    k_contact()

    # ── 8. Central-difference leapfrog ───────────────────────────────────────
    for step in range(n_steps):

        ramp = min(1.0, (step + 1) / max(n_ramp, 1))
        k_advance(dt, alpha, ramp)
        k_zero_forces()
        if enable_self_contact:
            k_grid_build()
        k_internal_forces()
        k_contact()

        if step % max(1, n_steps // 10) == 0:
            pct = 100 * step // n_steps
            n_dead = int(failed_ti.to_numpy().sum())
            print(f"GPU: {pct:3d}%  step {step}/{n_steps}  "
                  f"failed_elem={n_dead}")

        if step % sample_int == 0 or step == n_steps - 1:
            v_np   = v_ti.to_numpy()
            ep_np  = eps_p_ti.to_numpy()
            KE     = 0.5 * float(np.einsum('i,ij->', M_np, v_np ** 2))
            # Incremental plastic dissipation (approximate)
            _sigma_y_e = np.minimum(sigma_y0 + H_hard * ep_np, sigma_y0 * 10)
            PE_step    = float(np.sum(ep_np * _sigma_y_e * W_0 * rho))
            absorbed_energy = PE_step
            t_hist.append(step * dt)
            KE_hist.append(KE)
            PE_hist.append(absorbed_energy)

        if n_frames > 0 and (step % frame_interval == 0 or step == n_steps - 1):
            x_fr  = x_ti.to_numpy()
            ep_fr = eps_p_ti.to_numpy()
            fd_fr = failed_ti.to_numpy().astype(float)
            u_fr  = (x_fr - pts).flatten()
            nep   = sigma_vm_to_nodal(mesh, ep_fr)
            nfd   = sigma_vm_to_nodal(mesh, fd_fr)
            frames.append({
                'displacement': u_fr,
                'stress_vm':    np.zeros(N),
                'eps_p':        nep,
                'failed':       nfd,
                'time':         (step + 1) * dt,
            })

    # ── 9. Post-processing ────────────────────────────────────────────────────
    x_final  = x_ti.to_numpy()
    ep_final = eps_p_ti.to_numpy()
    fd_final = failed_ti.to_numpy().astype(bool)

    u_3n    = x_final - pts
    u_flat  = u_3n.flatten()
    disp_mag = np.linalg.norm(u_3n, axis=1)

    # Von Mises from deformed Green-Lagrange strain (display only)
    _x0b  = x_final[conn[:, 0]]
    _Ds_f = np.stack(
        [x_final[conn[:, k]] - _x0b for k in range(1, 4)], axis=2
    ).astype(np.float64)
    _F_f  = _Ds_f @ Dm_inv_all
    _FtF  = np.einsum('eji,ejk->eik', _F_f, _F_f)
    _E_gl = 0.5 * (_FtF - np.eye(3)[None])
    _trE  = np.trace(_E_gl, axis1=1, axis2=2)[:, None, None]
    _devE = _E_gl - (_trE / 3.0) * np.eye(3)[None]
    _eps_vm  = np.sqrt(np.maximum(
        2.0 / 3.0 * np.einsum('eij,eij->e', _devE, _devE), 0.0
    ))
    vm_stress          = 2.0 * mu * _eps_vm
    vm_stress[~valid]  = 0.0
    vm_stress[fd_final] = 0.0

    nodal_vm = sigma_vm_to_nodal(mesh, vm_stress)
    nodal_ep = sigma_vm_to_nodal(mesh, ep_final)
    nodal_fd = sigma_vm_to_nodal(mesh, fd_final.astype(float))
    peak_vm  = float(np.max(vm_stress)) if vm_stress.size else 0.0
    peak_d   = float(np.max(disp_mag))  if disp_mag.size  else 0.0

    print(f"GPU Solver: Done.  MaxDisp={peak_d:.3f} mm  "
          f"PeakVM≈{peak_vm:.1f} MPa  Failed={int(fd_final.sum())}")

    return {
        'type':               'crash',
        'mesh':               mesh,
        'displacement':       u_flat,
        'stress':             nodal_vm,
        'visualization_mode': 'Von Mises Stress',
        'disp_scale':         disp_scale,
        'frames':             frames,
        'element_stress':     vm_stress,
        'plastic_strain':     ep_final,
        'failed_elements':    fd_final,
        'time':               np.array(t_hist),
        'energy_kinetic':     np.array(KE_hist),
        'energy_strain':      np.zeros(len(t_hist)),
        'energy_plastic':     np.array(PE_hist),
        'peak_displacement':  peak_d,
        'peak_stress':        peak_vm,
        'absorbed_energy':    float(absorbed_energy),
        'n_failed':           int(fd_final.sum()),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Node 4: CrashSolverGPUNode
# ─────────────────────────────────────────────────────────────────────────────

class CrashSolverGPUNode(CadQueryNode):
    """
    GPU-Accelerated Explicit Crash Solver.

    Physics
    -------
    **Constitutive model** – Neo-Hookean hyperelasticity + J2 elasto-plasticity:

        F = Ds · Dm⁻¹          (deformation gradient)
        J = det(F)             (volume ratio)
        P = μ(F−F⁻ᵀ) + λln(J)F⁻ᵀ   (Neo-Hookean PK1 stress)
        τ = P · Fᵀ             (Kirchhoff stress)
        σ_vm = √(3/2 s:s)/J   (von Mises, Cauchy)
        if σ_vm > σ_y + H·εp: radial return → scale P, accumulate εp
        if εp ≥ εf: element deleted (zero force contribution)

    **Contact mechanics** – penalty method:

        F_contact = k_pen · d · n̂   (d = penetration depth)
        k_pen = contact_stiffness_factor × (λ + 2/3 μ)   [bulk modulus scaled]

        Ground plane: rigid floor at configurable y-coordinate.
        Self-contact: spatial hash (32³ grid, face-based, 5×5×5 neighbourhood).

    **Time integration** – Central-Difference leapfrog (same as CrashSolverNode).

    Inputs / Outputs
    ----------------
    Fully compatible with CrashSolverNode wiring.

    Notes
    -----
    Requires ``pip install taichi``.  On first run Taichi JIT-compiles
    all kernels (~3 s); subsequent runs are fast.
    """

    __identifier__ = 'com.cad.sim.crash_solver_gpu'
    NODE_NAME      = 'Crash Solver (GPU)'

    def __init__(self):
        super().__init__()
        self.add_input('mesh',           color=(200, 100, 200))
        self.add_input('crash_material', color=(255, 150,  50))
        self.add_input('constraints',    color=(255, 100, 100), multi_input=True)
        self.add_input('impact',         color=(255, 200,   0))
        self.add_output('crash_results', color=(0, 180, 255))

        # ── Simulation time ──────────────────────────────────────────────────
        self.create_property('end_time',          2.0,   widget_type='float')
        self.create_property('time_steps',        500,   widget_type='int')

        # ── Damping ──────────────────────────────────────────────────────────
        self.create_property('damping_alpha',     10.0,  widget_type='float')

        # ── Ground-plane contact ─────────────────────────────────────────────
        self.create_property('enable_floor',      True,  widget_type='checkbox')
        self.create_property('floor_y',           0.0,   widget_type='float')

        # ── Self-contact ─────────────────────────────────────────────────────
        self.create_property('enable_self_contact', False, widget_type='checkbox')
        self.create_property('contact_stiffness',   10.0,  widget_type='float')

        # ── Animation / display ───────────────────────────────────────────────
        self.create_property('n_frames',          120,   widget_type='int')
        self.create_property('disp_scale',        1.0,   widget_type='float')
        self.create_property(
            'visualization', 'Von Mises Stress',
            widget_type='combo',
            items=['Von Mises Stress', 'Displacement', 'Plastic Strain',
                   'Failed Elements']
        )

    # ─────────────────────────────────────────────────────────────────────────

    def run(self):
        print("GPU Crash Solver: Starting hybrid Neo-Hookean / J2-elasto-plastic "
              "simulation with penalty contact...")

        # ── 1. Require Taichi ─────────────────────────────────────────────────
        if not _try_init_taichi():
            self.set_error(
                "GPU Crash Solver: Taichi is not available.  "
                "Install it with  pip install taichi  and restart."
            )
            return None

        # ── 2. Gather inputs ──────────────────────────────────────────────────
        mesh     = self.get_input_value('mesh',           None)
        material = self.get_input_value('crash_material', None)
        impact   = self.get_input_value('impact',         None)

        def _flatten(items):
            flat = []
            if not items:
                return flat
            for it in items:
                if isinstance(it, list): flat.extend(it)
                elif it is not None:     flat.append(it)
            return flat

        constraints = _flatten(self.get_input_list('constraints'))

        if mesh is None:
            self.set_error("GPU Crash Solver: mesh input is not connected.")
            return None
        if material is None:
            self.set_error("GPU Crash Solver: crash_material input is not connected.")
            return None

        # ── 3. Material parameters ────────────────────────────────────────────
        E        = float(material.get('E',               210000.0))
        nu       = float(material.get('nu',              0.30))
        rho      = float(material.get('rho',             7.85e-9))
        sigma_y0 = float(material.get('yield_strength',  250.0))
        H_hard   = float(material.get('tangent_modulus', 2000.0))
        eps_fail = float(material.get('failure_strain',  0.20))
        do_frac  = bool(material.get('enable_fracture',  True))

        _, mu = lam_lame(E, nu)
        la    = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

        # ── 4. Mesh ───────────────────────────────────────────────────────────
        p = mesh.p
        N = p.shape[1]
        x, y, z = p
        print(f"GPU Crash Solver: {N} nodes, {mesh.t.shape[1]} elements.")

        # ── 5. Fixed DOFs ─────────────────────────────────────────────────────
        fixed_dofs_set: set = set()
        from cadquery import Vector

        for constr in constraints:
            if not constr:
                continue
            fixed_dof_indices = constr.get('fixed_dofs', [0, 1, 2])
            geoms = constr.get('geometries', [constr.get('geometry')])
            geoms = [g for g in geoms if g is not None]
            if not geoms:
                continue
            tol = 1.5
            try:
                bboxes  = [g.BoundingBox() for g in geoms]
                xmin_bb = min(b.xmin for b in bboxes) - tol
                xmax_bb = max(b.xmax for b in bboxes) + tol
                ymin_bb = min(b.ymin for b in bboxes) - tol
                ymax_bb = max(b.ymax for b in bboxes) + tol
                zmin_bb = min(b.zmin for b in bboxes) - tol
                zmax_bb = max(b.zmax for b in bboxes) + tol
            except Exception:
                continue
            in_bb = ((x >= xmin_bb) & (x <= xmax_bb) &
                     (y >= ymin_bb) & (y <= ymax_bb) &
                     (z >= zmin_bb) & (z <= zmax_bb))
            for i in np.where(in_bb)[0]:
                pt = Vector(float(x[i]), float(y[i]), float(z[i]))
                for g in geoms:
                    try:
                        if g.distanceTo(pt) <= tol:
                            for d in fixed_dof_indices:
                                fixed_dofs_set.add(3 * int(i) + d)
                            break
                    except Exception:
                        pass

        print(f"GPU Crash Solver: {len(fixed_dofs_set)} constrained DOFs.")

        # ── 6. Impact velocity BCs ────────────────────────────────────────────
        impact_dof_list: list = []
        impact_dof_vals: list = []
        v_imp = np.zeros(3)

        if impact is not None:
            v_imp     = np.asarray(impact.get('velocity', [0, 0, 0]), dtype=float)
            face_list = impact.get('face_list', [])
            face_tol  = float(impact.get('node_tolerance', 2.0))
            if face_list:
                try:
                    bboxes = [f.BoundingBox() for f in face_list]
                    ixmin  = min(b.xmin for b in bboxes) - face_tol
                    ixmax  = max(b.xmax for b in bboxes) + face_tol
                    iymin  = min(b.ymin for b in bboxes) - face_tol
                    iymax  = max(b.ymax for b in bboxes) + face_tol
                    izmin  = min(b.zmin for b in bboxes) - face_tol
                    izmax  = max(b.zmax for b in bboxes) + face_tol
                    cands  = np.where(
                        (x >= ixmin) & (x <= ixmax) &
                        (y >= iymin) & (y <= iymax) &
                        (z >= izmin) & (z <= izmax)
                    )[0]
                except Exception:
                    cands = np.arange(N)
            else:
                cands = np.arange(N)

            for ni in cands:
                pt = Vector(float(x[ni]), float(y[ni]), float(z[ni]))
                matched = not face_list
                for face in face_list:
                    try:
                        if face.distanceTo(pt) <= face_tol:
                            matched = True; break
                    except Exception:
                        pass
                if matched:
                    for d in range(3):
                        if v_imp[d] != 0.0:
                            gdof = 3 * int(ni) + d
                            if gdof not in fixed_dofs_set:
                                impact_dof_list.append(gdof)
                                impact_dof_vals.append(v_imp[d])

        impact_dof_arr = np.array(impact_dof_list, dtype=np.int64)
        impact_vel_arr = np.array(impact_dof_vals, dtype=np.float64)
        print(f"GPU Crash Solver: Impact BCs on {len(impact_dof_list)} DOFs "
              f"({v_imp} mm/ms).")

        # ── 7. Time parameters ────────────────────────────────────────────────
        end_time    = float(self.get_property('end_time'))
        n_steps_req = max(int(self.get_property('time_steps')), 1)
        alpha       = float(self.get_property('damping_alpha'))
        n_frames    = int(self.get_property('n_frames'))
        disp_scale  = float(self.get_property('disp_scale'))

        dt_req = end_time / n_steps_req
        dt_cfl = _compute_cfl_dt(mesh, E, nu, rho, safety=0.5)
        if dt_req > dt_cfl:
            print(f"GPU Crash Solver: AUTO Δt {dt_req:.4e} → {dt_cfl:.4e} ms.")
            dt      = dt_cfl
            n_steps = max(int(np.ceil(end_time / dt)), 1)
        else:
            dt      = dt_req
            n_steps = n_steps_req

        n_ramp = max(10, min(50, n_steps // 20))
        print(f"GPU Crash Solver: Δt={dt:.4e} ms, steps={n_steps}, "
              f"CFL={dt_cfl:.4e} ms")

        # ── 8. Contact settings ───────────────────────────────────────────────
        en_floor    = bool(self.get_property('enable_floor'))
        y_fl        = float(self.get_property('floor_y'))
        en_self     = bool(self.get_property('enable_self_contact'))
        k_cf        = float(self.get_property('contact_stiffness'))

        # ── 9. Visualization field selection ─────────────────────────────────
        viz_mode  = self.get_property('visualization')

        # ── 10. Delegate to GPU simulation ───────────────────────────────────
        result = _run_neo_hookean_sim(
            mesh             = mesh,
            mu               = mu,
            la               = la,
            rho              = rho,
            dt               = dt,
            n_steps          = n_steps,
            alpha            = alpha,
            n_frames         = n_frames,
            fixed_dofs_set   = fixed_dofs_set,
            impact_dof_arr   = impact_dof_arr,
            impact_vel_arr   = impact_vel_arr,
            n_ramp           = n_ramp,
            sigma_y0         = sigma_y0,
            H_hard           = H_hard,
            eps_failure      = eps_fail,
            enable_fracture  = do_frac,
            k_contact_factor = k_cf,
            enable_floor     = en_floor,
            y_floor          = y_fl,
            enable_self_contact = en_self,
            disp_scale       = disp_scale,
        )

        if result is None:
            return None

        # Select the viewer field based on visualization mode
        N_nodes = mesh.p.shape[1]
        if viz_mode == 'Displacement':
            u_3n     = result['displacement'].reshape(N_nodes, 3)
            result['stress'] = np.linalg.norm(u_3n, axis=1)
        elif viz_mode == 'Plastic Strain':
            result['stress'] = sigma_vm_to_nodal(
                mesh, result['plastic_strain'])
        elif viz_mode == 'Failed Elements':
            result['stress'] = sigma_vm_to_nodal(
                mesh, result['failed_elements'].astype(float))
        # 'Von Mises Stress' is the default already set by _run_neo_hookean_sim

        result['visualization_mode'] = viz_mode
        return result

