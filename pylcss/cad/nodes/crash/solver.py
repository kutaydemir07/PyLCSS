# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.
"""CPU explicit crash/impact solver node (central-difference leapfrog + J2 plasticity)."""
import numpy as np
import logging
from scipy.spatial import cKDTree
from pylcss.cad.core.base_node import CadQueryNode
from pylcss.cad.nodes.fem import MATERIAL_DATABASE, lam_lame

logger = logging.getLogger(__name__)
from pylcss.cad.nodes.crash._mechanics import (
    build_D_matrix, assemble_lumped_mass, sigma_vm_to_nodal,
    _precompute_elements, _radial_return_vec,
    _compute_forces_vec, _compute_forces_corot_vec,
    _compute_cfl_dt, _compute_contact_dt_limit, _apply_mass_scaling,
    compute_penalty_contact, _update_boundary_nodes,
)

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
        if impact is None:
            self.set_error(
                "Crash Solver: impact input is not connected.\n"
                "Connect an ImpactConditionNode to define an initial velocity.\n"
                "Without it the simulation starts with v=0 and F_ext=0, "
                "producing zero stress for all time steps."
            )
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

        # Downgrade P2 / higher-order meshes to linear P1 for explicit dynamics.
        # Row-summing a P2 consistent mass matrix gives zero (or negative) corner
        # masses, immediately blowing up a = F/m.  The first 4 rows of mesh.t
        # are always the corner vertices, so simple slicing is sufficient.
        if mesh.t.shape[0] > 4:
            import skfem as _skfem_crash
            print("Crash Solver: Downgrading higher-order mesh to linear P1 "
                  "for explicit integration.")
            mesh = _skfem_crash.MeshTet(mesh.p, mesh.t[:4, :])

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

        # ── 5b. Rigid-wall planes derived from Roller / Fixed constraints ─────
        # For each constraint that locks exactly one translational DOF (Roller),
        # build a rigid axial wall so that FREE interior nodes cannot penetrate
        # past the constrained face during dynamic crushing.
        # Format: list of (dof_axis: int, wall_coord: float, sign: +1 or -1)
        #   sign=+1  → wall blocks motion in the +axis direction (x <= wall_coord)
        #   sign=-1  → wall blocks motion in the -axis direction (x >= wall_coord)
        _rigid_walls = []  # [(axis, coord, sign), ...]
        for constr in constraints:
            if not constr:
                continue
            fdofs = constr.get('fixed_dofs', [])
            geoms = constr.get('geometries', [constr.get('geometry')])
            geoms = [g for g in (geoms or []) if g is not None]
            if not geoms:
                continue
            # Only add a wall for single-axis Roller constraints (exactly 1 DOF)
            if len(fdofs) != 1:
                continue
            axis = fdofs[0]   # 0=X, 1=Y, 2=Z
            try:
                bboxes = [g.BoundingBox() for g in geoms]
            except Exception:
                continue
            # Determine wall coordinate and direction from face centroid vs model centre
            axis_coords_all = p[axis, :]   # all node coordinates along this axis
            _model_mid = float(np.mean(axis_coords_all))
            _face_coords = []
            for bb in bboxes:
                _face_coords.append((getattr(bb, 'xmin' if axis==0 else 'ymin' if axis==1 else 'zmin')
                                     + getattr(bb, 'xmax' if axis==0 else 'ymax' if axis==1 else 'zmax')) / 2.0)
            _face_mid = float(np.mean(_face_coords))
            if _face_mid >= _model_mid:
                # Wall is on the positive side — prevents motion past it in +axis
                _wall_pos = max(getattr(bb, 'xmax' if axis==0 else 'ymax' if axis==1 else 'zmax') for bb in bboxes)
                _rigid_walls.append((axis, float(_wall_pos), +1))
                print(f"Crash Solver: Rigid wall axis={axis} coord={_wall_pos:.2f} (blocks +motion).")
            else:
                # Wall is on the negative side — prevents motion past it in -axis
                _wall_neg = min(getattr(bb, 'xmin' if axis==0 else 'ymin' if axis==1 else 'zmin') for bb in bboxes)
                _rigid_walls.append((axis, float(_wall_neg), -1))
                print(f"Crash Solver: Rigid wall axis={axis} coord={_wall_neg:.2f} (blocks -motion).")

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
        _ms_info = None
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

        # Bulk/material CFL limit.  If selective mass scaling was applied,
        # use the scaled minimum element time step rather than the original
        # material-density limit.
        dt_cfl_bulk = _compute_cfl_dt(mesh, E, nu, rho, safety=0.5)
        if _enable_ms and _ms_info is not None:
            dt_cfl_bulk = max(dt_cfl_bulk, float(_ms_info['dt_min_scaled']))

        # Contact can introduce a stiffer stability limit than the material
        # wave-speed CFL bound, so include it in the final time-step selection.
        enable_contact = bool(self.get_property('enable_contact'))
        k_cf = float(self.get_property('contact_stiffness'))
        ct_frac = float(self.get_property('contact_thickness'))
        contact_update_int = max(1, int(self.get_property('contact_update_interval')))

        _bnd_facet_mask = mesh.f2t[1, :] == -1
        boundary_facets_all = mesh.facets[:, _bnd_facet_mask].T.astype(int)
        boundary_nodes_all = np.unique(boundary_facets_all) if len(boundary_facets_all) > 0 else np.array([], dtype=int)

        conn_tmp = mesh.t.T
        T_tmp = np.ones((conn_tmp.shape[0], 4, 4))
        T_tmp[:, :, 1:] = mesh.p.T[conn_tmp]
        vol_tmp = np.abs(np.linalg.det(T_tmp)) / 6.0
        valid_tmp = vol_tmp[vol_tmp > 1e-20]
        L_min = float(np.median(valid_tmp) ** (1.0 / 3.0)) if valid_tmp.size else 1.0
        K_bulk = E / (3.0 * (1.0 - 2.0 * nu))
        k_penalty = k_cf * K_bulk * L_min
        dt_cfl_contact = _compute_contact_dt_limit(M_diag, boundary_nodes_all, k_penalty) if enable_contact else np.inf
        dt_cfl = min(dt_cfl_bulk, dt_cfl_contact)

        if dt_req > dt_cfl:
            _limiter = 'contact' if dt_cfl_contact < dt_cfl_bulk else 'bulk'
            print(f"Crash Solver: WARNING – requested Δt ({dt_req:.4e} ms) exceeds "
                  f"stable limit ({dt_cfl:.4e} ms, {_limiter}-controlled).  Auto-correcting.")
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
              f"(bulk CFL = {dt_cfl_bulk:.4e} ms, contact CFL = {dt_cfl_contact:.4e} ms)")

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

        # ── T4 volumetric locking diagnostic ─────────────────────────────────
        # Linear T4 (4-node tet / P1) elements have only one volumetric strain
        # mode per element.  J2 plasticity is isochoric (ΔV=0 during plastic
        # flow), which forces volumetric deformation entirely through this single
        # mode → over-constraint → artificial stiffness (locking).
        # Symptoms: stress 3–5× analytical, negligible plastic strain, response
        # insensitive to impact velocity.  Mitigation (not in this solver):
        #   F-bar mean-dilation, NBS nodal pressure averaging, or T10/P2 elements.
        # FIX #5: The original condition (sigma_y0 < 1e15) was always True because
        # yield stress is never a quadrillion MPa.  Correct logic: warn about
        # near-incompressible locking (ν≥0.45) OR when both plasticity is active
        # AND ν is high enough to cause significant volumetric over-constraint.
        # J2 plasticity (isochoric) amplifies locking, but at low ν it is mild;
        # severe locking only appears when ν≥0.35 with plasticity, or ν≥0.45 alone.
        _is_near_incompressible = nu >= 0.45
        _plasticity_locking_risk = (sigma_y0 < E) and (nu >= 0.35)  # plastic flow + vol locking
        _locking_risk = _is_near_incompressible or _plasticity_locking_risk
        if _locking_risk:
            _reasons = []
            if _is_near_incompressible:
                _reasons.append(f"\u03bd={nu:.3f} \u2265 0.45 (near-incompressible \u2014 volumetric locking)")
            if _plasticity_locking_risk:
                _reasons.append(
                    f"\u03c3_y={sigma_y0:.0f} MPa < E={E:.0f} MPa: J2 plastic flow is isochoric "
                    f"and amplifies volumetric locking at \u03bd={nu:.3f} \u2265 0.35"
                )
            _lock_warn = (
                "Crash Solver: WARNING \u2014 T4 (linear tet) volumetric locking risk detected.\n"
                + "\n".join(f"  \u2022 {r}" for r in _reasons) + "\n"
                "  Symptoms: stress 3\u20135\u00d7 analytical, under-predicted plastic strain.\n"
                "  Mitigation: use a coarser mesh with larger elements, or switch to\n"
                "  T10/P2 elements once supported."
            )
            print(_lock_warn)
            logger.warning(_lock_warn)

        # ── 8b-contact. Contact algorithm setup ──────────────────────────────
        # Characteristic element length (cube-root of median volume).
        _valid_vols = vol_all[vol_all > 1e-20]
        L_min = float(np.median(_valid_vols) ** (1.0 / 3.0)) if _valid_vols.size else L_min
        k_penalty   = k_cf * K_bulk * L_min          # [N/mm]
        ct_thickness = ct_frac * L_min               # search radius [mm]

        # Initial boundary-node list and surface facets for contact
        if enable_contact:
            boundary_facets = boundary_facets_all
            boundary_nodes  = boundary_nodes_all
        else:
            boundary_nodes  = np.array([], dtype=int)
            boundary_facets = np.zeros((0, 3), dtype=int)

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

            # ── Rigid wall enforcement ────────────────────────────────────
            # Prevent free nodes from penetrating the constrained wall plane.
            # Uses a hard kinematic clamp: if deformed coordinate overshoots
            # the wall, pull it back and zero the normal velocity component.
            if _rigid_walls:
                for (_waxis, _wcoord, _wsign) in _rigid_walls:
                    _def_coord = p[_waxis, :] + u[_waxis::3][:N]
                    if _wsign == +1:  # wall blocks +axis motion
                        _pentr = _def_coord > _wcoord
                    else:             # wall blocks -axis motion
                        _pentr = _def_coord < _wcoord
                    _pentr_idx = np.where(_pentr)[0]
                    for _ni in _pentr_idx:
                        _gdof = 3 * _ni + _waxis
                        if _gdof in fixed_dofs_set:
                            continue  # constrained node, already handled
                        u[_gdof] = _wcoord - p[_waxis, _ni]   # clamp to wall
                        v[_gdof] = 0.0                          # kill normal vel

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
                    boundary_nodes, boundary_facets = _update_boundary_nodes(mesh, failed_elem)

                F_contact = compute_penalty_contact(
                    u, mesh.p.T, boundary_nodes, ct_thickness, k_penalty,
                    surf_facets=boundary_facets
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

            # ── Artificial bulk viscosity (von Neumann–Richtmyer / Wilkins) ─
            # Damps numerical ringing behind compressive shock fronts by adding
            # a viscous pressure to volumetrically compressing elements only.
            #   Q_e = \u03c1 L_e (C_q L_e \u03b5\u0307_v\u00b2 + C_l c_d |\u03b5\u0307_v|)  if \u03b5\u0307_v < 0 (compression)
            #   Q_e = 0                                              otherwise
            # C_q \u2248 1.5 (strong shocks), C_l \u2248 0.06 (weak shocks / ringing).
            _C_q_bv     = 1.5
            _C_l_bv     = 0.06
            _c_d_bv     = np.sqrt(
                E * (1.0 - nu) /
                max(rho * (1.0 + nu) * (1.0 - 2.0 * nu), 1e-30)
            )
            _delta_u_e  = delta_u[dof_idx]                        # (N_el, 12)
            # Volumetric row = sum of \u03b5_xx, \u03b5_yy, \u03b5_zz rows of B
            _B_vol      = B_all[:, 0, :] + B_all[:, 1, :] + B_all[:, 2, :]  # (N_el, 12)
            _eps_v_rate = (
                np.einsum('ij,ij->i', _B_vol, _delta_u_e) / max(dt, 1e-30)
            )                                                      # (N_el,) vol. strain rate
            _L_bv       = np.cbrt(np.maximum(vol_all, 0.0))       # V^{1/3} char. length
            _Q_bv       = np.where(
                (_eps_v_rate < 0.0) & (vol_all > 1e-20) & ~failed_elem,
                rho * _L_bv * (
                    _C_q_bv * _L_bv * _eps_v_rate ** 2
                    + _C_l_bv * _c_d_bv * np.abs(_eps_v_rate)
                ),
                0.0,
            )                                                      # (N_el,)
            # Scatter hydrostatic pressure force: F_bv_e = Q_e V_e B_vol^T
            F_bv = np.zeros(3 * N)
            _Q_V = _Q_bv * vol_all                                 # (N_el,)
            np.add.at(F_bv, dof_idx.ravel(),
                      (_Q_V[:, np.newaxis] * _B_vol).ravel())
            F_bv[fixed_dofs] = 0.0

            # New acceleration (contact + bulk-viscosity enter with sign that
            # resists motion; F_bv already opposes further compression)
            a = (- F_int - F_bv + F_contact - F_damp) / M_diag
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
        # 'absorbed_energy' is cumulative plastic dissipation (N·mm = mJ).
        # NOTE: this is the work done *against* plastic deformation, not the total
        # external work input (which equals absorbed_energy + remaining KE + SE).
        # It is reliable for comparative ranking but may be inflated by T4 locking.
        print(f"  Plastic dissipation ≈ {absorbed_energy:.1f} N·mm  ({absorbed_energy*1e-3:.2f} J)")
        if W_ext_cumulative > 0:
            print(f"  External work input ≈ {W_ext_cumulative:.1f} N·mm  ({W_ext_cumulative*1e-3:.2f} J)")

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
