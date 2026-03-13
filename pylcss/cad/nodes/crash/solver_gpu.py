# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.
"""GPU-accelerated crash solver node (Taichi backend)."""
import numpy as np
import logging
from pylcss.cad.core.base_node import CadQueryNode
from pylcss.cad.nodes.fem import lam_lame
from pylcss.cad.nodes.crash._mechanics import sigma_vm_to_nodal
from pylcss.cad.nodes.crash._gpu_kernels import (
    _try_init_taichi, _taichi_available, _run_neo_hookean_sim,
)

logger = logging.getLogger(__name__)

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
        if impact is None:
            self.set_error(
                "GPU Crash Solver: impact input is not connected.\n"
                "Connect an ImpactConditionNode to define an initial velocity.\n"
                "Without it the simulation starts with v=0 and F_ext=0, "
                "producing zero stress for all time steps."
            )
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
        # Downgrade P2 / higher-order meshes to linear P1 for explicit dynamics.
        if mesh.t.shape[0] > 4:
            import skfem as _skfem_gpu
            print("GPU Crash Solver: Downgrading higher-order mesh to linear P1 "
                  "for explicit integration.")
            mesh = _skfem_gpu.MeshTet(mesh.p, mesh.t[:4, :])

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

