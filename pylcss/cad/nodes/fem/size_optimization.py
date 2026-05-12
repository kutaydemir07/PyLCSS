# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.
"""FEM parametric size-optimisation node — CalculiX-coupled.

Each design evaluation is a full CAD-rebuild → mesh → CalculiX-static cycle.
Gradient-based optimizers (SLSQP, L-BFGS-B, trust-constr) drive that cycle
``n_params + 1`` times per step; gradient-free methods (COBYLA, Nelder-Mead)
issue one evaluation per step.
"""
import json
import logging

import numpy as np

from pylcss.cad.core.base_node import CadQueryNode

logger = logging.getLogger(__name__)


def _tet_mesh_volume(mesh) -> float:
    """Sum of tetrahedral element volumes (mm³ for the standard unit system)."""
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


class SizeOptimizationNode(CadQueryNode):
    """Optimise parametric CAD dimensions with a CalculiX-in-the-loop solver."""
    __identifier__ = 'com.cad.sim.sizeopt'
    NODE_NAME = 'Size Optimization'

    def __init__(self):
        super().__init__()
        self.add_input('shape',       color=(100, 255, 100))
        self.add_input('material',    color=(200, 200, 200))
        self.add_input('constraints', color=(255, 100, 100), multi_input=True)
        self.add_input('loads',       color=(255, 255,   0), multi_input=True)

        self.add_output('optimized_shape',     color=(100, 255, 100))
        self.add_output('optimal_parameters',  color=(180, 180,   0))
        self.add_output('result',              color=(200, 100, 200))

        self.create_property('objective', 'Min Weight', widget_type='combo',
                             items=['Min Weight', 'Min Compliance', 'Min Max Stress'])
        self.create_property('parameters', '["wall_thickness"]', widget_type='text')
        self.create_property('bounds', '{"wall_thickness": [1.0, 20.0]}', widget_type='text')

        self.create_property('max_stress',        250.0, widget_type='float')
        self.create_property('max_volume',          0.0, widget_type='float')
        self.create_property('min_safety_factor',   1.5, widget_type='float')

        self.create_property('max_iterations', 50,    widget_type='int')
        self.create_property('tolerance',      1e-4,  widget_type='float')
        self.create_property('optimizer', 'COBYLA', widget_type='combo',
                             items=['COBYLA', 'Nelder-Mead', 'SLSQP', 'L-BFGS-B', 'trust-constr', 'Powell'])
        # FD step for gradient-based methods. Each FD perturbation = one full
        # CAD→Mesh→CalculiX cycle.
        self.create_property('gradient_step', 0.05, widget_type='float')

        self.create_property('element_size', 2.0, widget_type='float')

        # External-solver pass-through.
        self.create_property('external_solver_path', '',  widget_type='text')
        self.create_property('external_work_dir',    '',  widget_type='text')
        self.create_property('external_timeout_s',   3600.0, widget_type='float')

    def run(self, progress_callback=None):
        from scipy.optimize import minimize

        logger.info("SizeOpt: starting CalculiX-coupled size optimisation.")

        shape_node = self._get_upstream_shape_node()
        material = self.get_input_value('material', None)
        constraint_list = _flatten(self.get_input_list('constraints'))
        load_list       = _flatten(self.get_input_list('loads'))

        missing = []
        if shape_node is None:        missing.append("parametric shape upstream")
        if material is None:          missing.append("material")
        if not constraint_list:       missing.append("at least one constraint")
        if not load_list:             missing.append("at least one load")
        if missing:
            msg = "SizeOpt requires " + ", ".join(missing) + "."
            self.set_error(msg)
            logger.warning(msg)
            return None

        try:
            param_names = json.loads(self.get_property('parameters'))
            bounds_dict = json.loads(self.get_property('bounds'))
        except json.JSONDecodeError as exc:
            self.set_error(f"SizeOpt: invalid JSON in parameters/bounds: {exc}")
            return None

        bounds: list = []
        initial_values: list = []
        for name in param_names:
            if name in bounds_dict:
                lo, hi = bounds_dict[name]
                bounds.append((float(lo), float(hi)))
                try:
                    initial_values.append(float(shape_node.get_property(name)))
                except Exception:
                    initial_values.append(0.5 * (float(lo) + float(hi)))
            else:
                bounds.append((0.1, 100.0))
                initial_values.append(10.0)
        x0 = np.array(initial_values, dtype=float)

        obj_type      = self.get_property('objective')
        max_stress    = float(self.get_property('max_stress') or 0.0)
        max_vol       = float(self.get_property('max_volume') or 0.0)
        max_iter      = int(self.get_property('max_iterations') or 1)
        tol           = float(self.get_property('tolerance') or 1e-4)
        optimizer     = self.get_property('optimizer')
        gradient_step = float(self.get_property('gradient_step') or 0.05)
        elem_size     = float(self.get_property('element_size') or 2.0)

        if optimizer in {'SLSQP', 'L-BFGS-B', 'trust-constr'}:
            logger.warning(
                "SizeOpt: gradient-based optimizer '%s' selected. Each FD "
                "perturbation runs one full CAD→Mesh→CalculiX cycle; meshing "
                "topology changes across perturbations corrupt the gradient. "
                "Prefer COBYLA for ≥5 parameters.",
                optimizer,
            )

        history = {'iterations': [], 'objective': [], 'stress': [], 'volume': []}
        iter_count = [0]

        def evaluate(x: np.ndarray):
            for i, name in enumerate(param_names):
                try:
                    shape_node.set_property(name, float(x[i]))
                except Exception as exc:
                    logger.warning("SizeOpt: could not set %s=%s: %s", name, x[i], exc)
            try:
                shape = shape_node.run()
                if shape is None:
                    return None
            except Exception as exc:
                logger.warning("SizeOpt: shape rebuild failed: %s", exc)
                return None
            try:
                from pylcss.cad.nodes.fem.mesh import MeshNode
                mesh_node = MeshNode()
                mesh_node._inputs = {'shape': shape}
                mesh_node.set_property('element_size', elem_size)
                mesh = mesh_node.run()
                if mesh is None:
                    return None
            except Exception as exc:
                logger.warning("SizeOpt: meshing failed: %s", exc)
                return None
            try:
                fea = self._run_calculix(mesh, material, constraint_list, load_list)
                if fea is None:
                    return None
                stress = float(np.max(fea.get('stress', np.zeros(1))))
                volume = _tet_mesh_volume(mesh)
                compliance = float(fea.get('compliance', 0.0))
                return stress, volume, compliance
            except Exception as exc:
                logger.warning("SizeOpt: CalculiX evaluation failed: %s", exc)
                return None

        def objective(x):
            res = evaluate(x)
            if res is None:
                return 1e10
            stress, volume, compliance = res
            iter_count[0] += 1
            history['iterations'].append(iter_count[0])
            history['stress'].append(stress)
            history['volume'].append(volume)
            if obj_type == 'Min Weight':
                obj = volume
            elif obj_type == 'Min Compliance':
                obj = compliance
            elif obj_type == 'Min Max Stress':
                obj = stress
            else:
                obj = volume
            history['objective'].append(obj)
            logger.info(
                "SizeOpt iter %d: obj=%.4f, stress=%.2f MPa, vol=%.4f",
                iter_count[0], obj, stress, volume,
            )
            if progress_callback is not None:
                try:
                    progress_callback(iter_count[0], max_iter, obj)
                except Exception:
                    pass
            return obj

        def stress_constraint(x):
            res = evaluate(x)
            if res is None:
                return -1e10
            return max_stress - res[0]

        def volume_constraint(x):
            res = evaluate(x)
            if res is None:
                return -1e10
            return max_vol - res[1]

        constraints_list: list = []
        if max_stress > 0:
            constraints_list.append({'type': 'ineq', 'fun': stress_constraint})
        if max_vol > 0:
            constraints_list.append({'type': 'ineq', 'fun': volume_constraint})

        try:
            if optimizer in {'SLSQP', 'trust-constr'}:
                result = minimize(
                    objective, x0, method=optimizer,
                    bounds=bounds, constraints=constraints_list,
                    options={'maxiter': max_iter, 'disp': True,
                             'ftol': tol, 'eps': gradient_step},
                )
            elif optimizer == 'L-BFGS-B':
                result = minimize(
                    objective, x0, method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': max_iter, 'disp': True, 'eps': gradient_step},
                )
            elif optimizer == 'COBYLA':
                cons = list(constraints_list)
                for j, (lb, ub) in enumerate(bounds):
                    cons.append({'type': 'ineq',
                                 'fun': lambda x, j=j, lb=lb: x[j] - lb})
                    cons.append({'type': 'ineq',
                                 'fun': lambda x, j=j, ub=ub: ub - x[j]})
                result = minimize(
                    objective, x0, method='COBYLA',
                    constraints=cons,
                    options={'maxiter': max_iter, 'disp': True,
                             'rhobeg': gradient_step * 10, 'catol': tol},
                )
            elif optimizer == 'Nelder-Mead':
                result = minimize(
                    objective, x0, method='Nelder-Mead',
                    options={'maxiter': max_iter, 'disp': True,
                             'xatol': tol, 'fatol': tol},
                )
            else:
                result = minimize(
                    objective, x0, method=optimizer,
                    bounds=bounds,
                    options={'maxiter': max_iter, 'disp': True},
                )
            optimal_x = result.x
            success = bool(result.success)
        except Exception as exc:
            logger.error("SizeOpt: optimisation failed: %s", exc)
            optimal_x = x0
            success = False

        optimal_params: dict = {}
        for i, name in enumerate(param_names):
            optimal_params[name] = float(optimal_x[i])
            try:
                shape_node.set_property(name, float(optimal_x[i]))
            except Exception:
                pass
        try:
            final_shape = shape_node.run()
        except Exception:
            final_shape = None

        logger.info("SizeOpt: %s. optimal_parameters=%s",
                    "succeeded" if success else "stopped without success",
                    optimal_params)

        return {
            'optimized_shape':    final_shape,
            'optimal_parameters': optimal_params,
            'history':            history,
            'success':            success,
            'type':               'sizeopt',
        }

    # ---------- internals ----------

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
            job_name='pylcss_sizeopt_eval',
        )
        try:
            return run_calculix_static(
                mesh=mesh, material=material,
                constraints=constraints, loads=loads,
                config=config, visualization_mode='Von Mises Stress',
            )
        except SolverBackendError as exc:
            logger.warning("SizeOpt: CalculiX backend error: %s", exc)
            return None

    def _get_upstream_shape_node(self):
        """Walk back from the 'shape' input to the first parametric CAD node."""
        try:
            port = self.get_input('shape')
            if not port:
                return None
            queue = [port]
            visited = set()
            while queue:
                current = queue.pop(0)
                for connected in current.connected_ports():
                    node = connected.node()
                    if node in visited:
                        continue
                    visited.add(node)
                    node_type = getattr(node, 'NODE_NAME', '')
                    if node_type in {'Mesh', 'Material', 'Filter', 'Remesh Surface'}:
                        for inp in node.inputs().values():
                            queue.append(inp)
                    else:
                        return node
        except Exception as exc:
            logger.debug("SizeOpt: upstream-shape search failed: %s", exc)
        return None


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
