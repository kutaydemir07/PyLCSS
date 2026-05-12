# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.
"""FEM shape-optimisation node — CalculiX-coupled mesh-morphing optimisation.

Two sensitivity methods drive the boundary motion:

Biological Stress Leveling (Fully-Stressed Design heuristic)
    sens_i = (mean(σ) − σ_i) / mean(σ)
    Inspired by Wolff's law of bone remodelling: add material where stress
    is high, remove it where stress is low.  Heuristic — does **not**
    rigorously minimise compliance.

Adjoint Compliance (Hadamard–Zolesio boundary shape derivative)
    dC/dV_n = −2 W(u)  with W = strain-energy density.
    Drives the free boundary toward uniform W(u), the optimality condition
    for minimum-compliance shape design under a volume constraint.  W is
    read directly from CalculiX (``*EL FILE\\nENER``, nodal-averaged).
"""
import json
import logging

import numpy as np

from pylcss.cad.core.base_node import CadQueryNode

logger = logging.getLogger(__name__)


class ShapeOptimizationNode(CadQueryNode):
    """Boundary mesh-morphing shape optimisation driven by CalculiX results."""
    __identifier__ = 'com.cad.sim.shapeopt'
    NODE_NAME = 'Shape Optimization'

    def __init__(self):
        super().__init__()
        self.add_input('mesh',        color=(200, 100, 200))
        self.add_input('material',    color=(200, 200, 200))
        self.add_input('constraints', color=(255, 100, 100), multi_input=True)
        self.add_input('loads',       color=(255, 255,   0), multi_input=True)

        self.add_output('optimized_mesh', color=(200, 100, 200))
        self.add_output('result',         color=(200, 100, 200))

        self.create_property('objective', 'Min Max Stress', widget_type='combo',
                             items=['Min Max Stress', 'Min Compliance', 'Uniform Stress'])

        self.create_property('max_iterations',  20,  widget_type='int')
        self.create_property('step_size',       0.1, widget_type='float')
        self.create_property('smoothing_weight', 0.5, widget_type='float')
        self.create_property('convergence_tol', 0.01, widget_type='float')

        self.create_property('volume_preservation', True, widget_type='bool')
        self.create_property('max_displacement',     5.0, widget_type='float')
        self.create_property('fixed_faces', '[]', widget_type='text')

        self.create_property('sensitivity_method', 'Biological Stress Leveling',
                             widget_type='combo',
                             items=['Biological Stress Leveling', 'Adjoint Compliance'])

        self.create_property('visualization', 'Stress', widget_type='combo',
                             items=['Stress', 'Displacement', 'Shape Change'])

        # External-solver pass-through.
        self.create_property('external_solver_path', '',  widget_type='text')
        self.create_property('external_work_dir',    '',  widget_type='text')
        self.create_property('external_timeout_s',   3600.0, widget_type='float')

    # ──────────────────────────────────────────────────────────────────
    # main loop
    # ──────────────────────────────────────────────────────────────────
    def run(self, progress_callback=None):
        import skfem  # mesh data type only; no solver use

        logger.info("ShapeOpt: starting CalculiX-coupled shape optimisation.")

        mesh_input = self.get_input_value('mesh', None)
        material   = self.get_input_value('material', None)
        constraint_list = _flatten(self.get_input_list('constraints'))
        load_list       = _flatten(self.get_input_list('loads'))

        mesh = mesh_input['mesh'] if isinstance(mesh_input, dict) and 'mesh' in mesh_input else mesh_input

        missing = []
        if mesh is None:           missing.append("mesh")
        if material is None:       missing.append("material")
        if not constraint_list:    missing.append("at least one constraint")
        if not load_list:          missing.append("at least one load")
        if missing:
            msg = "ShapeOpt requires " + ", ".join(missing) + "."
            self.set_error(msg)
            return None

        obj_type        = self.get_property('objective')
        max_iter        = int(self.get_property('max_iterations') or 1)
        step_size       = float(self.get_property('step_size') or 0.1)
        smooth_weight   = float(self.get_property('smoothing_weight') or 0.5)
        conv_tol        = float(self.get_property('convergence_tol') or 0.01)
        preserve_volume = bool(self.get_property('volume_preservation'))
        max_disp_allowed = float(self.get_property('max_displacement') or 0.0)

        try:
            fixed_faces = json.loads(self.get_property('fixed_faces') or '[]')
        except json.JSONDecodeError:
            fixed_faces = []

        original_points = mesh.p.copy()
        current_points  = mesh.p.copy()

        f2t = mesh.f2t
        boundary_facets = (
            np.where(f2t[1, :] == -1)[0] if f2t.shape[0] == 2
            else mesh.facets_satisfying(lambda _x: True)
        )
        boundary_nodes = np.unique(mesh.facets[:, boundary_facets].flatten())

        fixed_nodes = self._get_fixed_nodes(mesh, constraint_list, fixed_faces)
        moveable_nodes = np.setdiff1d(boundary_nodes, fixed_nodes)
        logger.info("ShapeOpt: %d moveable boundary nodes, %d fixed",
                    len(moveable_nodes), len(fixed_nodes))

        history = {'iterations': [], 'objective': [], 'max_stress': [], 'volume': []}
        initial_volume = _tet_mesh_volume(mesh)
        best_objective = float('inf')
        best_points = current_points.copy()

        # Precomputed adjacency (used by interior-Laplacian morphing each iter).
        n_pts = mesh.p.shape[1]
        adjacency = [[] for _ in range(n_pts)]
        for ec in mesh.t.T:
            for na in ec:
                for nb in ec:
                    if na != nb:
                        adjacency[na].append(int(nb))
        adjacency = [np.unique(a) for a in adjacency]
        bnd_set = set(boundary_nodes.tolist())
        interior_nodes = np.array(
            [i for i in range(n_pts) if i not in bnd_set], dtype=int
        )

        for itr in range(max_iter):
            mesh_curr = skfem.MeshTet(current_points, mesh.t)

            fea = self._run_calculix(mesh_curr, material, constraint_list, load_list)
            if fea is None:
                logger.warning("ShapeOpt iter %d: CalculiX evaluation failed, stopping.", itr)
                break

            stress     = np.asarray(fea.get('stress', np.zeros(n_pts)), dtype=float)
            ener_nodal = np.asarray(fea.get('ener_nodal', np.zeros(n_pts)), dtype=float)
            compliance = float(fea.get('compliance', 0.0))
            max_stress = float(np.max(stress)) if stress.size else 0.0
            curr_vol   = _tet_mesh_volume(mesh_curr)

            if obj_type == 'Min Max Stress':
                obj = max_stress
            elif obj_type == 'Min Compliance':
                obj = compliance
            elif obj_type == 'Uniform Stress':
                obj = float(np.std(stress))
            else:
                obj = max_stress

            history['iterations'].append(itr)
            history['objective'].append(obj)
            history['max_stress'].append(max_stress)
            history['volume'].append(curr_vol)
            logger.info(
                "ShapeOpt iter %d: obj=%.4f, max σ=%.2f MPa, V=%.4f",
                itr, obj, max_stress, curr_vol,
            )

            if itr > 0 and abs(obj - best_objective) / (abs(best_objective) + 1e-10) < conv_tol:
                logger.info("ShapeOpt: converged at iteration %d", itr)
                break

            if obj < best_objective:
                best_objective = obj
                best_points = current_points.copy()

            sensitivities = self._compute_shape_sensitivity(
                stress, ener_nodal, moveable_nodes,
            )
            sensitivities = self._laplacian_smooth(
                sensitivities, moveable_nodes, mesh_curr, smooth_weight
            )

            normals = self._compute_boundary_normals(mesh_curr, moveable_nodes)
            move = -step_size * sensitivities[:, np.newaxis] * normals

            # Capture reference winding once per iteration so we can detect
            # element inversion by sign-flip (robust to all-negative meshes).
            ref_signs = self._signed_volume_signs(current_points, mesh.t)

            alpha = 1.0
            accepted = False
            for _ in range(5):
                trial_points = current_points.copy()
                for i, node_idx in enumerate(moveable_nodes):
                    if i < len(move):
                        trial_points[:, node_idx] += alpha * move[i]
                        if max_disp_allowed > 0.0:
                            total_move = trial_points[:, node_idx] - original_points[:, node_idx]
                            total_norm = float(np.linalg.norm(total_move))
                            if total_norm > max_disp_allowed:
                                trial_points[:, node_idx] = (
                                    original_points[:, node_idx]
                                    + total_move * (max_disp_allowed / total_norm)
                                )
                if not self._is_mesh_inverted(trial_points, mesh.t, ref_signs):
                    current_points = trial_points
                    accepted = True
                    break
                alpha *= 0.5
            if not accepted:
                logger.warning("ShapeOpt: every backtracking step inverted the mesh; stopping.")
                break

            current_points = self._relax_interior_nodes(
                current_points, adjacency, interior_nodes, n_relax=3
            )

            if preserve_volume:
                new_vol = _tet_mesh_volume(skfem.MeshTet(current_points, mesh.t))
                if new_vol > 1e-12:
                    scale = (initial_volume / new_vol) ** (1.0 / 3.0)
                    for node_idx in moveable_nodes:
                        disp = current_points[:, node_idx] - original_points[:, node_idx]
                        current_points[:, node_idx] = original_points[:, node_idx] + disp * scale

            if progress_callback is not None:
                try:
                    progress_callback(mesh_curr, stress, itr, max_iter)
                except Exception:
                    pass

        optimized_mesh = skfem.MeshTet(best_points, mesh.t)
        final_fea = self._run_calculix(optimized_mesh, material, constraint_list, load_list)

        logger.info("ShapeOpt: complete. final objective=%.4f", best_objective)

        return {
            'mesh':         optimized_mesh,
            'stress':       final_fea.get('stress') if final_fea else None,
            'displacement': final_fea.get('displacement') if final_fea else None,
            'history':      history,
            'type':         'shapeopt',
            'visualization_mode': self.get_property('visualization'),
        }

    # ──────────────────────────────────────────────────────────────────
    # CalculiX bridge
    # ──────────────────────────────────────────────────────────────────
    def _run_calculix(self, mesh, material, constraints, loads):
        from pylcss.solver_backends import (
            ExternalRunConfig, SolverBackendError, run_calculix_static,
        )
        config = ExternalRunConfig(
            executable=(self.get_property('external_solver_path') or None),
            work_dir=(self.get_property('external_work_dir') or None),
            keep_files=False,
            run_solver=True,
            timeout_s=float(self.get_property('external_timeout_s') or 3600.0),
            job_name='pylcss_shapeopt_eval',
        )
        try:
            return run_calculix_static(
                mesh=mesh, material=material,
                constraints=constraints, loads=loads,
                config=config, visualization_mode='Von Mises Stress',
            )
        except SolverBackendError as exc:
            logger.warning("ShapeOpt: CalculiX backend error: %s", exc)
            return None

    # ──────────────────────────────────────────────────────────────────
    # sensitivity & mesh helpers (FEA-independent)
    # ──────────────────────────────────────────────────────────────────
    def _compute_shape_sensitivity(self, stress, ener_nodal, moveable_nodes):
        """Return per-moveable-node sensitivity in [-1, 1]; sign convention:
        positive ⇒ move inward (remove material), negative ⇒ outward."""
        method = self.get_property('sensitivity_method')
        sens = np.zeros(len(moveable_nodes))

        if method == 'Adjoint Compliance' and ener_nodal.size > 0 and float(np.mean(ener_nodal)) > 1e-30:
            mean_w = float(np.mean(ener_nodal)) + 1e-30
            for i, n in enumerate(moveable_nodes):
                w_i = float(ener_nodal[n]) if n < len(ener_nodal) else mean_w
                sens[i] = (mean_w - w_i) / mean_w
        else:
            mean_s = float(np.mean(stress)) + 1e-30
            for i, n in enumerate(moveable_nodes):
                s_i = float(stress[n]) if n < len(stress) else mean_s
                sens[i] = (mean_s - s_i) / mean_s

        peak = float(np.max(np.abs(sens))) if sens.size else 0.0
        if peak > 1e-10:
            sens = sens / peak
        return sens

    @staticmethod
    def _signed_volume_signs(points, t):
        try:
            v0 = points[:, t[0, :]]
            e1 = points[:, t[1, :]] - v0
            e2 = points[:, t[2, :]] - v0
            e3 = points[:, t[3, :]] - v0
            vols = np.sum(np.cross(e1, e2, axis=0) * e3, axis=0) / 6.0
            return np.sign(vols)
        except Exception:
            return None

    @staticmethod
    def _is_mesh_inverted(points, t, ref_signs):
        try:
            v0 = points[:, t[0, :]]
            e1 = points[:, t[1, :]] - v0
            e2 = points[:, t[2, :]] - v0
            e3 = points[:, t[3, :]] - v0
            vols = np.sum(np.cross(e1, e2, axis=0) * e3, axis=0) / 6.0
            if ref_signs is not None:
                return bool(np.any(np.sign(vols) != ref_signs))
            return bool(np.min(np.abs(vols)) <= 1e-9)
        except Exception:
            return True

    @staticmethod
    def _laplacian_smooth(values, nodes, mesh, weight):
        if weight <= 0:
            return values
        smoothed = values.copy()
        node_to_idx = {int(node): i for i, node in enumerate(nodes)}
        for i, node_idx in enumerate(nodes):
            elem_mask = np.any(mesh.t == node_idx, axis=0)
            neighbors = np.unique(mesh.t[:, elem_mask].flatten())
            boundary_neighbors = [int(n) for n in neighbors if int(n) in node_to_idx and int(n) != int(node_idx)]
            if boundary_neighbors:
                avg = float(np.mean([values[node_to_idx[n]] for n in boundary_neighbors]))
                smoothed[i] = (1 - weight) * values[i] + weight * avg
        return smoothed

    @staticmethod
    def _relax_interior_nodes(pts, adjacency, interior_nodes, n_relax=3):
        if len(interior_nodes) == 0:
            return pts.copy()
        pts_new = pts.copy()
        for _ in range(n_relax):
            pts_pass = pts_new.copy()
            for n_idx in interior_nodes:
                nb = adjacency[n_idx]
                if len(nb) == 0:
                    continue
                pts_pass[:, n_idx] = np.mean(pts_new[:, nb], axis=1)
            pts_new = pts_pass
        return pts_new

    def _compute_boundary_normals(self, mesh, boundary_nodes):
        normals = np.zeros((len(boundary_nodes), 3))
        boundary_set = set(int(f) for f in mesh.boundary_facets().tolist())
        mesh_center = np.mean(mesh.p, axis=1)
        for i, node_idx in enumerate(boundary_nodes):
            facet_mask = np.any(mesh.facets == node_idx, axis=0)
            adjacent = [int(f) for f in np.where(facet_mask)[0] if int(f) in boundary_set]
            n_acc = np.zeros(3)
            for f_idx in adjacent:
                facet_nodes = mesh.facets[:, f_idx]
                pts = mesh.p[:, facet_nodes]
                v1 = pts[:, 1] - pts[:, 0]
                v2 = pts[:, 2] - pts[:, 0]
                normal = np.cross(v1, v2)
                norm = np.linalg.norm(normal)
                if norm > 1e-10:
                    face_center = np.mean(pts, axis=1)
                    if float(np.dot(normal, face_center - mesh_center)) < 0.0:
                        normal = -normal
                    n_acc += normal / norm
            norm = np.linalg.norm(n_acc)
            normals[i] = n_acc / norm if norm > 1e-10 else np.array([0.0, 0.0, 1.0])
        return normals

    def _get_fixed_nodes(self, mesh, constraint_list, fixed_face_conditions):
        """Boundary nodes that should never move: those tied to BCs + any user-tagged faces."""
        from pylcss.solver_backends.common import (
            dict_geometries,
            nodes_matching_condition,
            nodes_matching_geometries,
        )

        fixed = np.array([], dtype=np.int64)

        for c in constraint_list:
            if not c:
                continue
            geoms = dict_geometries(c)
            condition = str(c.get("condition") or "").strip()
            try:
                if geoms:
                    fixed = np.union1d(fixed, nodes_matching_geometries(mesh, geoms))
                elif condition:
                    fixed = np.union1d(
                        fixed,
                        nodes_matching_condition(mesh, condition, label="ShapeOpt fixed constraint"),
                    )
            except Exception as exc:
                logger.debug("ShapeOpt: fixed-node detection skipped one constraint: %s", exc)

        # User-tagged fixed face conditions are not exported to CalculiX, so this
        # path is kept only for compatibility with old projects — we ignore string
        # conditions silently.  Real fixed-face tagging should go through a
        # SelectFaceNode → ConstraintNode pipeline instead.
        if fixed_face_conditions:
            logger.info(
                "ShapeOpt: fixed_faces string conditions are no longer evaluated; "
                "use SelectFaceNode-driven constraints to pin geometry."
            )
        return fixed.astype(np.int64)


# ──────────────────────────────────────────────────────────────────────
# helpers
# ──────────────────────────────────────────────────────────────────────
def _flatten(items):
    out = []
    if not items:
        return out
    for it in items:
        if isinstance(it, list):
            out.extend(_flatten(it))
        elif it is not None:
            out.append(it)
    return out


def _tet_mesh_volume(mesh) -> float:
    tets = np.asarray(mesh.t).T[:, :4].astype(int)
    coords = np.asarray(mesh.p).T
    if tets.shape[0] == 0:
        return 0.0
    v0 = coords[tets[:, 0]]
    edges = np.stack(
        [coords[tets[:, 1]] - v0, coords[tets[:, 2]] - v0, coords[tets[:, 3]] - v0],
        axis=-1,
    )
    return float(np.sum(np.abs(np.linalg.det(edges)) / 6.0))
