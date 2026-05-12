# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.
"""FEM topology-optimisation node - SIMP driven by CalculiX-in-the-loop.

Each iteration the design field ``rho_e`` (one value per tet) is fed to a
single CalculiX static solve using binned modified-SIMP moduli.  Sensitivities
and objective values are then computed analytically from the displacement field
as per-element base-stiffness energies, avoiding FRD ENER nodal projection and
material-bin deprojection errors.
"""
import logging
from typing import Any, Mapping, Tuple

import numpy as np
from scipy.spatial import cKDTree

from pylcss.cad.core.base_node import CadQueryNode
from pylcss.cad.nodes.fem._helpers import (
    build_filter_matrix, sensitivity_filter, density_filter_3d,
    density_filter_chainrule, heaviside_projection, mma_update, shape_recovery,
)

logger = logging.getLogger(__name__)


class TopologyOptimizationNode(CadQueryNode):
    """SIMP topology optimisation driven by CalculiX evaluations."""
    __identifier__ = 'com.cad.sim.topopt'
    NODE_NAME = 'Topology Opt'

    def __init__(self):
        super().__init__()
        self.add_input('mesh',        color=(200, 100, 200))
        self.add_input('material',    color=(200, 200, 200))
        self.add_input('constraints', color=(255, 100, 100), multi_input=True)
        self.add_input('loads',       color=(255, 255,   0), multi_input=True)

        self.add_output('optimized_mesh',  color=(200, 100, 200))
        self.add_output('recovered_shape', color=(100, 255, 100))

        self.create_property('vol_frac',       0.4,  widget_type='float')
        self.create_property('iterations',     50,   widget_type='int')
        self.create_property('filter_radius',  3.0,  widget_type='float')
        self.create_property('density_cutoff', 0.3,  widget_type='float')
        self.create_property('shape_recovery', True, widget_type='bool')

        self.create_property('visualization', 'Density', widget_type='combo',
                             items=['Density', 'Recovered Shape', 'Von Mises Stress'])
        self.create_property('symmetry_x', None, widget_type='float')
        self.create_property('symmetry_y', None, widget_type='float')
        self.create_property('symmetry_z', None, widget_type='float')

        self.create_property('penal',           3.0,   widget_type='float')
        self.create_property('move_limit',      0.2,   widget_type='float')
        self.create_property('min_density',     0.001, widget_type='float')
        self.create_property('convergence_tol', 0.02,  widget_type='float')
        self.create_property('recovery_resolution', 100, widget_type='int')
        self.create_property('smoothing_iterations',  3, widget_type='int')

        self.create_property('filter_type',   'density', widget_type='combo',
                             items=['sensitivity', 'density'])
        self.create_property('update_scheme', 'MMA',     widget_type='combo',
                             items=['MMA', 'OC'])
        self.create_property('projection', 'None', widget_type='combo',
                             items=['None', 'Heaviside'])
        self.create_property('heaviside_beta', 4.0, widget_type='float')
        self.create_property('heaviside_eta',  0.5, widget_type='float')
        self.create_property('continuation',   True, widget_type='bool')

        # SIMP modulus binning — fewer bins = smaller deck, more quantisation error.
        self.create_property('simp_bins', 32, widget_type='int')

        # External CalculiX dispatch settings.
        self.create_property('external_solver_path', '',  widget_type='text')
        self.create_property('external_work_dir',    '',  widget_type='text')
        self.create_property('external_timeout_s',   3600.0, widget_type='float')

        # Deprecated — kept so projects saved before the CalculiX-only cut load
        # cleanly. The in-house skfem path no longer exists; element type is
        # fixed to CalculiX C3D4 tets regardless of this property.
        self.create_property('element_type', 'Fast (Linear P1)', widget_type='combo',
                             items=['Fast (Linear P1)', 'Accurate (Quadratic P2)'])

    # ──────────────────────────────────────────────────────────────────
    def run(self, progress_callback=None):
        logger.info("TopOpt: starting CalculiX-coupled SIMP optimisation.")

        mesh = self.get_input_value('mesh', None)
        material = self.get_input_value('material', None)
        constraint_list = _flatten(self.get_input_list('constraints'))
        load_list       = _flatten(self.get_input_list('loads'))

        missing = []
        if mesh is None:        missing.append("mesh")
        if material is None:    missing.append("material")
        if not constraint_list: missing.append("at least one constraint")
        if not load_list:       missing.append("at least one load")
        if missing:
            msg = "TopOpt requires " + ", ".join(missing) + "."
            self.set_error(msg)
            return None

        vol_frac      = float(self.get_input_value('vol_frac',      'vol_frac'))
        max_iter      = int(self.get_property('iterations') or 1)
        filter_radius = float(self.get_input_value('filter_radius', 'filter_radius'))
        penal         = float(self.get_input_value('penal',         'penal'))
        move          = float(self.get_input_value('move_limit',    'move_limit'))
        rho_min       = float(self.get_input_value('min_density',   'min_density'))
        conv_tol      = float(self.get_input_value('convergence_tol', 'convergence_tol'))
        n_bins        = max(2, int(self.get_property('simp_bins') or 32))

        sym_x = self.get_input_value('symmetry_x', 'symmetry_x')
        sym_y = self.get_input_value('symmetry_y', 'symmetry_y')
        sym_z = self.get_input_value('symmetry_z', 'symmetry_z')

        filter_type      = self.get_property('filter_type')
        update_scheme    = self.get_property('update_scheme')
        projection_type  = self.get_property('projection')
        heaviside_beta   = float(self.get_property('heaviside_beta') or 4.0)
        heaviside_eta    = float(self.get_property('heaviside_eta')  or 0.5)
        use_continuation = bool(self.get_property('continuation'))
        shape_recovery_on = bool(self.get_property('shape_recovery'))
        recovery_res      = int(self.get_property('recovery_resolution') or 100)
        smoothing_iter    = int(self.get_property('smoothing_iterations') or 3)
        density_cutoff    = float(self.get_property('density_cutoff') or 0.3)

        # ── per-element geometry ─────────────────────────────────────────
        tets = np.asarray(mesh.t).T[:, :4].astype(int)   # (n_elem, 4)
        n_elem = tets.shape[0]
        coords = np.asarray(mesh.p).T                     # (n_node, 3)
        v0 = coords[tets[:, 0]]
        edges = np.stack(
            [coords[tets[:, 1]] - v0, coords[tets[:, 2]] - v0, coords[tets[:, 3]] - v0],
            axis=-1,
        )
        elem_vol = np.abs(np.linalg.det(edges)) / 6.0   # (n_elem,)
        total_vol = float(np.sum(elem_vol))
        target_vol = total_vol * vol_frac
        centroids = coords[tets].mean(axis=1)            # (n_elem, 3)

        # ── symmetry maps ────────────────────────────────────────────────
        sym_maps = []
        for axis, sym_val in enumerate([sym_x, sym_y, sym_z]):
            if sym_val is None:
                continue
            tree = cKDTree(centroids)
            mirror = centroids.copy()
            mirror[:, axis] = 2.0 * float(sym_val) - mirror[:, axis]
            dists, indices = tree.query(mirror)
            valid = dists < 0.1
            sym_maps.append((np.where(valid)[0], indices[valid]))

        # ── filter ───────────────────────────────────────────────────────
        H, H_sum = (None, None)
        if filter_radius > 0 and filter_type in {'density', 'sensitivity'}:
            logger.info("TopOpt: precomputing sparse filter matrix...")
            H, H_sum = build_filter_matrix(centroids, filter_radius)

        # ── initial density ──────────────────────────────────────────────
        densities = np.full(n_elem, vol_frac, dtype=float)
        _apply_symmetry(densities, sym_maps)

        # MMA history
        xold1 = densities.copy()
        xold2 = densities.copy()
        low = np.zeros(n_elem)
        upp = np.zeros(n_elem)

        beta_used = heaviside_beta
        eta_used = heaviside_eta
        prev_compliance = 0.0
        plateau = 0
        converged_runs = 0
        last_result: dict = {}
        densities_phys = densities.copy()
        d_proj = np.ones_like(densities)

        # ── optimisation loop ────────────────────────────────────────────
        for loop in range(max_iter):
            if projection_type == 'Heaviside':
                beta_used = (
                    min(heaviside_beta, 2 ** (4.0 * loop / max(max_iter, 1)))
                    if use_continuation else heaviside_beta
                )
            densities_phys, d_proj = _physical_density_from_design(
                densities, filter_type, filter_radius, H, H_sum,
                projection_type, beta_used, eta_used,
            )

            if progress_callback is not None:
                try:
                    progress_callback(mesh, densities_phys, loop, max_iter)
                except Exception:
                    pass

            try:
                last_result = self._run_calculix_iter(
                    mesh, material, constraint_list, load_list,
                    densities_phys, penal, rho_min, n_bins,
                )
            except Exception as exc:
                logger.error("TopOpt: CalculiX iteration %d failed: %s", loop, exc)
                self.set_error(f"TopOpt: CalculiX iteration {loop} failed: {exc}")
                break
            if not last_result:
                break

            flat_disp = last_result.get('displacement', None)
            if flat_disp is None:
                logger.error("TopOpt iter %d: CalculiX returned no displacement.", loop)
                self.set_error(f"TopOpt: CalculiX iteration {loop} returned no displacement field.")
                break

            # Exact base-stiffness element compliance, computed analytically from
            # the displacement field and the BASE Lamé constants.  Bypasses the
            # nodal-projection / binning errors in CalculiX's *EL FILE ENER block.
            #
            #   u_e^T K_e^0 u_e  =  V_e · ε_voigt · D_base · ε_voigt   (linear tet)
            elem_2W_base = _compute_base_strain_energies(
                tets, coords, elem_vol, material, np.asarray(flat_disp, dtype=float),
            )
            simp_factor, dsimp_drho = _simp_factor_and_derivative(densities_phys, rho_min, penal)
            c_val = float(np.sum(simp_factor * elem_2W_base))
            dc = -dsimp_drho * elem_2W_base

            if projection_type == 'Heaviside':
                dc = dc * d_proj
            if filter_type == 'density' and filter_radius > 0 and H is not None:
                dc = density_filter_chainrule(dc, H, H_sum)
            elif filter_type == 'sensitivity' and filter_radius > 0 and H is not None:
                dc = sensitivity_filter(dc, H, H_sum, densities=densities_phys)

            dvol_base = elem_vol * d_proj if projection_type == 'Heaviside' else elem_vol
            dvol = (
                density_filter_chainrule(dvol_base, H, H_sum)
                if filter_type == 'density' and filter_radius > 0 and H is not None
                else dvol_base
            )

            if update_scheme == 'MMA':
                current_vol = float(np.sum(densities_phys * elem_vol))
                vol_constraint = current_vol - target_vol

                # Adaptive move limit — wider in the early iterations.
                ramp_iters = 20
                move_factor = 1.0 + max(0.0, 1.5 * (1.0 - loop / ramp_iters))
                effective_move = min(0.5, move * move_factor)

                rho_new, low, upp = mma_update(
                    n_elem, loop, densities, rho_min, 1,
                    xold1, xold2, c_val, dc, vol_constraint, dvol,
                    low, upp, move=effective_move,
                )
                xold2 = xold1.copy()
                xold1 = densities.copy()
            else:
                def _candidate_volume(candidate):
                    candidate_phys, _ = _physical_density_from_design(
                        candidate, filter_type, filter_radius, H, H_sum,
                        projection_type, beta_used, eta_used,
                    )
                    return float(np.sum(candidate_phys * elem_vol))

                rho_new = _oc_update(
                    densities, dc, dvol, target_vol, rho_min, move,
                    volume_fn=_candidate_volume,
                )

            change = float(np.max(np.abs(rho_new - densities)))
            densities = rho_new
            _apply_symmetry(densities, sym_maps)

            densities_phys_log, _ = _physical_density_from_design(
                densities, filter_type, filter_radius, H, H_sum,
                projection_type, beta_used, eta_used,
            )
            vol_frac_now = float(np.sum(densities_phys_log * elem_vol) / total_vol)
            logger.info(
                "TopOpt iter %d: Δρ_max=%.4f, vol_frac=%.3f, compliance=%.4e",
                loop, change, vol_frac_now, c_val,
            )

            if change < conv_tol:
                converged_runs += 1
                if converged_runs >= 3:
                    logger.info("TopOpt: converged at iteration %d (Δρ plateau).", loop + 1)
                    break
            else:
                converged_runs = 0

            if loop > 0 and prev_compliance > 0:
                rel = abs(c_val - prev_compliance) / (abs(prev_compliance) + 1e-10)
                if rel < 1e-3:
                    plateau += 1
                    if plateau >= 5:
                        logger.info("TopOpt: converged at iteration %d (objective plateau).", loop + 1)
                        break
                else:
                    plateau = 0
            prev_compliance = c_val

        # ── final solve at projected densities for clean stress field ────
        densities_final = densities.copy()
        densities_phys_final, _ = _physical_density_from_design(
            densities, filter_type, filter_radius, H, H_sum,
            projection_type, beta_used, eta_used,
        )
        densities_final = densities_phys_final

        stress = None
        final_compliance = float(prev_compliance)
        try:
            final_result = self._run_calculix_iter(
                mesh, material, constraint_list, load_list,
                densities_phys_final, penal, rho_min, n_bins,
            )
            stress = np.asarray(final_result.get('stress', np.zeros(coords.shape[0])), dtype=float)
            final_disp = final_result.get('displacement', None)
            if final_disp is not None:
                final_elem_2W_base = _compute_base_strain_energies(
                    tets, coords, elem_vol, material, np.asarray(final_disp, dtype=float),
                )
                final_simp_factor, _ = _simp_factor_and_derivative(
                    densities_phys_final, rho_min, penal,
                )
                final_compliance = float(np.sum(final_simp_factor * final_elem_2W_base))
        except Exception as exc:
            logger.warning("TopOpt: final stress recovery failed: %s", exc)

        recovered = None
        if shape_recovery_on:
            try:
                verts, faces = shape_recovery(
                    mesh, densities_final, density_cutoff,
                    smoothing_iterations=smoothing_iter,
                    resolution=recovery_res,
                )
                if verts is not None and faces is not None:
                    recovered = {'vertices': verts, 'faces': faces}
            except Exception as exc:
                logger.warning("TopOpt: shape recovery failed: %s", exc)

        logger.info(
            "TopOpt: complete. final physical vol_frac = %.3f",
            float(np.sum(densities_final * elem_vol) / total_vol),
        )
        final_volume = float(np.sum(densities_final * elem_vol))
        material_density = float(material.get('rho', material.get('density', 0.0)))
        return {
            'mesh':            mesh,
            'density':         densities_final,
            'design_density':  densities,
            'stress':          stress,
            'recovered_shape': recovered,
            'type':            'topopt',
            'compliance':      final_compliance,
            'volume':          final_volume,
            'mass':            final_volume * material_density,
            'visualization_mode': self.get_property('visualization'),
            'density_cutoff':  density_cutoff,
        }

    # ──────────────────────────────────────────────────────────────────
    def _run_calculix_iter(
        self, mesh, material, constraints, loads,
        densities_phys, penal, rho_min, n_bins,
    ):
        from pylcss.solver_backends import (
            ExternalRunConfig, run_calculix_topopt_iteration,
        )
        config = ExternalRunConfig(
            executable=(self.get_property('external_solver_path') or None),
            work_dir=(self.get_property('external_work_dir') or None),
            keep_files=False,
            run_solver=True,
            timeout_s=float(self.get_property('external_timeout_s') or 3600.0),
            job_name='pylcss_topopt_iter',
        )
        return run_calculix_topopt_iteration(
            mesh=mesh, material=material,
            constraints=constraints, loads=loads,
            densities=densities_phys,
            p_penal=penal, rho_min=rho_min,
            config=config, n_bins=n_bins,
        )


# ──────────────────────────────────────────────────────────────────────
def _flatten(items):
    out: list = []
    if not items:
        return out
    for it in items:
        if isinstance(it, list):
            out.extend(_flatten(it))
        elif it is not None:
            out.append(it)
    return out


def _apply_symmetry(densities: np.ndarray, sym_maps):
    """Average mirrored element pairs in place (volume-preserving)."""
    for src, dst in sym_maps:
        avg = 0.5 * (densities[src] + densities[dst])
        densities[src] = avg
        densities[dst] = avg


def _compute_base_strain_energies(
    tets: np.ndarray,
    coords: np.ndarray,
    elem_vol: np.ndarray,
    material: Mapping[str, Any],
    flat_disp: np.ndarray,
) -> np.ndarray:
    """Analytical ``u_e^T K_e^0 u_e`` per linear-tet element.

    Returns the *base*-stiffness compliance contribution for every element —
    i.e. what the contribution would be if the modulus were ``E_0``
    everywhere — using the actual current displacement field.  This is the
    quantity SIMP sensitivity multiplies by ``-p · ρ_e^{p-1}`` and is robust
    to (a) how CalculiX binned the per-element modulus in the deck and
    (b) any nodal-projection smoothing that happens on ``*EL FILE``.
    """
    n_elem = tets.shape[0]
    v0 = coords[tets[:, 0]]
    e1 = coords[tets[:, 1]] - v0
    e2 = coords[tets[:, 2]] - v0
    e3 = coords[tets[:, 3]] - v0
    J = np.stack([e1, e2, e3], axis=-1)                # (n_elem, 3, 3)
    J_inv = np.linalg.inv(J)                            # (n_elem, 3, 3)

    grad_N_natural = np.array(
        [
            [-1.0, -1.0, -1.0],
            [ 1.0,  0.0,  0.0],
            [ 0.0,  1.0,  0.0],
            [ 0.0,  0.0,  1.0],
        ],
        dtype=float,
    )
    # grad_x N_i  =  J^{-T} · grad_xi N_i
    # → grad_N_global[e, i, k] = Σ_j J_inv[e, j, k] · grad_N_natural[i, j]
    grad_N_global = np.einsum('ejk,ij->eik', J_inv, grad_N_natural)

    B = np.zeros((n_elem, 6, 12), dtype=float)
    for i in range(4):
        dNdx = grad_N_global[:, i, 0]
        dNdy = grad_N_global[:, i, 1]
        dNdz = grad_N_global[:, i, 2]
        c = 3 * i
        B[:, 0, c + 0] = dNdx
        B[:, 1, c + 1] = dNdy
        B[:, 2, c + 2] = dNdz
        B[:, 3, c + 0] = dNdy
        B[:, 3, c + 1] = dNdx
        B[:, 4, c + 1] = dNdz
        B[:, 4, c + 2] = dNdy
        B[:, 5, c + 0] = dNdz
        B[:, 5, c + 2] = dNdx

    E  = float(material.get('E', 210000.0))
    nu = float(material.get('nu', material.get('poissons_ratio', 0.3)))
    G   = E / (2.0 * (1.0 + nu))
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    D = np.array(
        [
            [lam + 2 * G, lam,         lam,         0.0, 0.0, 0.0],
            [lam,         lam + 2 * G, lam,         0.0, 0.0, 0.0],
            [lam,         lam,         lam + 2 * G, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, G,   0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, G,   0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, G  ],
        ],
        dtype=float,
    )

    disp_xyz = flat_disp.reshape(-1, 3)
    u_e = disp_xyz[tets].reshape(n_elem, 12)

    eps = np.einsum('eij,ej->ei', B, u_e)               # (n_elem, 6)
    sig = np.einsum('ij,ej->ei', D, eps)                # (n_elem, 6)
    return np.einsum('ei,ei->e', eps, sig) * elem_vol   # (n_elem,)


def _simp_factor_and_derivative(
    densities_phys: np.ndarray,
    rho_min: float,
    penal: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return modified-SIMP stiffness factors and density derivatives."""
    rho_eval = np.clip(np.asarray(densities_phys, dtype=float), float(rho_min), 1.0)
    factor = float(rho_min) + (1.0 - float(rho_min)) * (rho_eval ** float(penal))
    derivative = (
        (1.0 - float(rho_min))
        * float(penal)
        * (rho_eval ** (float(penal) - 1.0))
    )
    return factor, derivative


def _physical_density_from_design(
    densities: np.ndarray,
    filter_type: str,
    filter_radius: float,
    H,
    H_sum,
    projection_type: str,
    beta: float,
    eta: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Map design densities to physical densities and projection derivative."""
    densities_phys = densities
    if filter_type == 'density' and filter_radius > 0 and H is not None:
        densities_phys = density_filter_3d(densities, H, H_sum)

    d_proj = np.ones_like(densities_phys)
    if projection_type == 'Heaviside':
        densities_phys, d_proj = heaviside_projection(densities_phys, beta, eta)
    return densities_phys, d_proj


def _oc_update(densities, dc, dvol, target_vol, rho_min, move, volume_fn=None):
    """Optimality-criteria update for compliance minimisation."""
    safe_dvol = np.maximum(np.asarray(dvol, dtype=float), 1e-30)
    sensitivity = np.maximum(-dc / safe_dvol, 1e-12)
    if volume_fn is None:
        volume_fn = lambda candidate: float(np.sum(candidate * safe_dvol))
    l1, l2 = 0.0, 1e9
    rho_new = densities.copy()
    while (l2 - l1) / (l1 + l2 + 1e-10) > 1e-3:
        lmid = 0.5 * (l1 + l2)
        term = np.sqrt(sensitivity / lmid)
        rho_new = densities * term
        rho_new = np.clip(rho_new, np.maximum(0, densities - move), np.minimum(1, densities + move))
        rho_new = np.clip(rho_new, rho_min, 1.0)
        if float(volume_fn(rho_new)) - target_vol > 0:
            l1 = lmid
        else:
            l2 = lmid
    return rho_new
