# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.
"""GPU/Taichi crash-simulation kernels — Neo-Hookean + J2 plasticity.

Contains Taichi-compiled explicit solver (_run_neo_hookean_sim) and its
runtime-initialisation helper (_try_init_taichi).  Import from here rather
than importing Taichi directly in solver_gpu.py.
"""
import numpy as np
import logging

logger = logging.getLogger(__name__)

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

    # Downgrade P2 / higher-order meshes to linear P1 for explicit dynamics.
    if mesh.t.shape[0] > 4:
        import skfem as _skfem_nh
        print("Neo-Hookean Sim: Downgrading higher-order mesh to linear P1 "
              "for explicit integration.")
        mesh = _skfem_nh.MeshTet(mesh.p, mesh.t[:4, :])

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

    # Spatial hash bounding box (padded 50 %).
    # A 10 % pad is insufficient for high-speed crashes where debris can eject
    # well beyond the original envelope; nodes outside the box collapse into a
    # single dense hash cell, destroying O(1) contact lookup performance.
    _pad      = 0.5 * max(np.ptp(pts[:, 0]),
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
    prev_ep_sample = np.zeros(N_elem, dtype=np.float64)

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
            # Incremental plastic dissipation (approximate, units N·mm).
            _dep_np = np.maximum(ep_np - prev_ep_sample, 0.0)
            _sigma_y_e = sigma_y0 + H_hard * prev_ep_sample
            absorbed_energy += float(np.sum(_dep_np * _sigma_y_e * W_0))
            prev_ep_sample = ep_np.copy()
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

