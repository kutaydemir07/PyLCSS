# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.
"""FEM parametric size-optimisation node (wall thickness, fillet radii, etc.)."""
import numpy as np
import logging
import os
import tempfile
import sys
import contextlib
from scipy.spatial import cKDTree
import skfem
from skfem import *
from skfem.helpers import sym_grad, ddot, trace
try:
    from simpleeval import simple_eval
except ImportError:
    simple_eval = None
from pylcss.config import simulation_config
from pylcss.cad.core.base_node import CadQueryNode

logger = logging.getLogger(__name__)
from pylcss.cad.nodes.fem._helpers import (
    lam_lame, tr, suppress_output, OCCGeometry,
    _find_matching_boundary_facets,
)

class SizeOptimizationNode(CadQueryNode):
    """
    Size Optimization Node - Optimizes parametric dimensions of CAD geometry.
    
    This node optimizes design variables such as wall thickness, fillet radii,
    and other dimensions while satisfying stress and volume constraints.
    Uses SciPy optimization algorithms (SLSQP, L-BFGS-B, etc.).
    """
    __identifier__ = 'com.cad.sim.sizeopt'
    NODE_NAME = 'Size Optimization'

    def __init__(self):
        super().__init__()
        # Inputs
        self.add_input('shape', color=(100, 255, 100))  # Parametric shape to optimize
        self.add_input('material', color=(200, 200, 200))
        self.add_input('constraints', color=(255, 100, 100))
        self.add_input('loads', color=(255, 255, 0))
        
        # Outputs
        self.add_output('optimized_shape', color=(100, 255, 100))
        self.add_output('optimal_parameters', color=(180, 180, 0))
        self.add_output('result', color=(200, 100, 200))
        
        # Optimization objective
        self.create_property('objective', 'Min Weight', widget_type='combo',
                             items=['Min Weight', 'Min Compliance', 'Min Max Stress'])
        
        # Design variables - JSON format: ["param1", "param2", ...]
        self.create_property('parameters', '["wall_thickness"]', widget_type='text')
        
        # Bounds - JSON format: {"param1": [min, max], ...}
        self.create_property('bounds', '{"wall_thickness": [1.0, 20.0]}', widget_type='text')
        
        # Constraints
        self.create_property('max_stress', 250.0, widget_type='float')  # MPa
        self.create_property('max_volume', 0.0, widget_type='float')  # 0 = no constraint
        self.create_property('min_safety_factor', 1.5, widget_type='float')
        
        # Optimization parameters
        self.create_property('max_iterations', 50, widget_type='int')
        self.create_property('tolerance', 1e-4, widget_type='float')
        self.create_property('optimizer', 'COBYLA', widget_type='combo',
                             items=['COBYLA', 'Nelder-Mead', 'SLSQP', 'L-BFGS-B', 'trust-constr', 'Powell'])
        # Finite-difference step for gradient-based solvers (SLSQP / L-BFGS-B).
        # Each FD perturbation costs one full CAD→Mesh→FEA loop, so keep this
        # coarse (1e-2..1e-1) unless you need high-accuracy gradients.
        # Gradient-free methods (COBYLA, Nelder-Mead) ignore this setting entirely.
        self.create_property('gradient_step', 0.05, widget_type='float')
        
        # Mesh settings
        self.create_property('element_size', 2.0, widget_type='float')

    def run(self, progress_callback=None):
        """Execute size optimization."""
        import json
        from scipy.optimize import minimize, NonlinearConstraint
        
        logger.info("SizeOpt: Starting size optimization...")
        
        # Get inputs
        shape_node = self._get_upstream_shape_node()
        material = self.get_input_value('material', None)
        constraint_data = self.get_input_value('constraints', None)
        load_data = self.get_input_value('loads', None)
        
        if not all([shape_node, material, constraint_data, load_data]):
            logger.warning("SizeOpt: Missing inputs!")
            return None
        
        # Parse JSON parameters
        try:
            param_names = json.loads(self.get_property('parameters'))
            bounds_dict = json.loads(self.get_property('bounds'))
        except json.JSONDecodeError as e:
            logger.error(f"SizeOpt: Invalid JSON in parameters/bounds: {e}")
            return None
        
        # Build bounds array
        bounds = []
        initial_values = []
        for param in param_names:
            if param in bounds_dict:
                b = bounds_dict[param]
                bounds.append((b[0], b[1]))
                # Get initial value from shape node property
                try:
                    initial = shape_node.get_property(param)
                    initial_values.append(float(initial))
                except:
                    initial_values.append((b[0] + b[1]) / 2)  # Midpoint
            else:
                bounds.append((0.1, 100.0))  # Default bounds
                initial_values.append(10.0)
        
        x0 = np.array(initial_values)
        
        # Optimization settings
        obj_type = self.get_property('objective')
        max_stress = self.get_property('max_stress')
        max_vol = self.get_property('max_volume')
        max_iter = self.get_property('max_iterations')
        tol = self.get_property('tolerance')
        optimizer = self.get_property('optimizer')
        gradient_step = float(self.get_property('gradient_step'))

        # ── Gradient-based optimizer cost & validity warning ─────────────────
        _FD_METHODS = {'SLSQP', 'L-BFGS-B', 'trust-constr'}
        if optimizer in _FD_METHODS:
            _fd_warn = (
                f"SizeOpt: WARNING — gradient-based optimizer '{optimizer}' is selected.  "
                "Each finite-difference perturbation requires one full "
                "CAD → mesh → FEA evaluation cycle, so the iteration cost is "
                "(n_params + 1) × FEA per step — easily 10–100× slower than "
                "COBYLA for 5+ parameters.  More critically: if meshing topology "
                "changes between perturbed designs (different element count / "
                "connectivity), the gradient estimate is corrupted and the optimizer "
                "converges to a false optimum.  Switch to COBYLA or Nelder-Mead "
                "unless analytic sensitivities are available."
            )
            print(_fd_warn)
            logger.warning(_fd_warn)
        elem_size = self.get_property('element_size')
        
        # History tracking
        history = {'iterations': [], 'objective': [], 'stress': [], 'volume': []}
        iteration_count = [0]  # Use list for closure
        
        def evaluate_design(x):
            """Evaluate a design point by running FEA."""
            # Update shape parameters
            for i, param in enumerate(param_names):
                try:
                    shape_node.set_property(param, float(x[i]))
                except Exception as e:
                    logger.warning(f"SizeOpt: Could not set {param}: {e}")
            
            # Rebuild geometry
            try:
                shape = shape_node.run()
                if shape is None:
                    return None, None, None
            except Exception as e:
                logger.warning(f"SizeOpt: Shape rebuild failed: {e}")
                return None, None, None
            
            # Mesh the shape
            try:
                # MeshNode is defined in this same file — no import needed.
                mesh_node = MeshNode()
                mesh_node._inputs = {'shape': shape}
                mesh_node.set_property('element_size', elem_size)
                mesh = mesh_node.run()
                if mesh is None:
                    return None, None, None
            except Exception as e:
                logger.warning(f"SizeOpt: Meshing failed: {e}")
                return None, None, None
            
            # Run FEA
            try:
                result = self._run_fea(mesh, material, constraint_data, load_data)
                if result is None:
                    return None, None, None
                
                max_vm_stress = np.max(result.get('stress', [0]))
                volume = self._compute_volume(mesh)
                compliance = result.get('compliance', 0)
                
                return max_vm_stress, volume, compliance
                
            except Exception as e:
                logger.warning(f"SizeOpt: FEA failed: {e}")
                return None, None, None
        
        def objective(x):
            """Objective function."""
            stress, volume, compliance = evaluate_design(x)
            if stress is None:
                return 1e10  # Penalty for failed evaluation
            
            iteration_count[0] += 1
            
            # Track history
            history['iterations'].append(iteration_count[0])
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
            logger.info(f"SizeOpt Iter {iteration_count[0]}: Obj={obj:.4f}, Stress={stress:.2f}, Vol={volume:.4f}")
            
            return obj
        
        def stress_constraint(x):
            """Stress constraint: max_stress - current_stress >= 0."""
            stress, _, _ = evaluate_design(x)
            if stress is None:
                return -1e10  # Highly infeasible
            return max_stress - stress
        
        def volume_constraint(x):
            """Volume constraint: max_volume - current_volume >= 0."""
            _, volume, _ = evaluate_design(x)
            if volume is None:
                return -1e10
            return max_vol - volume
        
        # Set up constraints for SLSQP
        constraints_list = []
        if max_stress > 0:
            constraints_list.append({'type': 'ineq', 'fun': stress_constraint})
        if max_vol > 0:
            constraints_list.append({'type': 'ineq', 'fun': volume_constraint})
        
        # Run optimization
        logger.info(f"SizeOpt: Running {optimizer} optimization...")
        try:
            if optimizer in ['SLSQP', 'trust-constr']:
                # Gradient-based: SciPy uses finite-differences internally.
                # Each FD perturbation = one full CAD/Mesh/FEA loop per parameter.
                # 'eps' controls the FD step size; expose it so users can tune it.
                result = minimize(
                    objective, x0, method=optimizer,
                    bounds=bounds, constraints=constraints_list,
                    options={'maxiter': max_iter, 'disp': True, 'ftol': tol,
                             'eps': gradient_step}
                )
            elif optimizer == 'L-BFGS-B':
                result = minimize(
                    objective, x0, method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': max_iter, 'disp': True, 'eps': gradient_step}
                )
            elif optimizer == 'COBYLA':
                # Gradient-free: no CAD/Mesh/FEA calls for gradient estimation.
                # Bounds are converted to inequality constraints because COBYLA
                # does not accept a 'bounds' argument.
                cobyla_cons = list(constraints_list)
                for _j, (_lb, _ub) in enumerate(bounds):
                    j_cap = _j  # capture loop variable
                    cobyla_cons.append({'type': 'ineq',
                                        'fun': lambda x, j=j_cap, lb=_lb: x[j] - lb})
                    cobyla_cons.append({'type': 'ineq',
                                        'fun': lambda x, j=j_cap, ub=_ub: ub - x[j]})
                # rhobeg ~ initial trust-region radius; roughly 10x gradient_step works well.
                result = minimize(
                    objective, x0, method='COBYLA',
                    constraints=cobyla_cons,
                    options={'maxiter': max_iter, 'disp': True,
                             'rhobeg': gradient_step * 10, 'catol': tol}
                )
            elif optimizer == 'Nelder-Mead':
                # Pure gradient-free simplex search; ignores constraints.
                result = minimize(
                    objective, x0, method='Nelder-Mead',
                    options={'maxiter': max_iter, 'disp': True,
                             'xatol': tol, 'fatol': tol}
                )
            else:  # Powell or other gradient-free
                result = minimize(
                    objective, x0, method=optimizer,
                    bounds=bounds,
                    options={'maxiter': max_iter, 'disp': True}
                )
            
            optimal_x = result.x
            success = result.success
            
        except Exception as e:
            logger.error(f"SizeOpt: Optimization failed: {e}")
            optimal_x = x0
            success = False
        
        # Set optimal parameters and rebuild final shape
        optimal_params = {}
        for i, param in enumerate(param_names):
            optimal_params[param] = float(optimal_x[i])
            try:
                shape_node.set_property(param, float(optimal_x[i]))
            except:
                pass
        
        final_shape = shape_node.run()
        
        logger.info(f"SizeOpt: Optimization {'succeeded' if success else 'completed'}")
        logger.info(f"SizeOpt: Optimal parameters: {optimal_params}")
        
        return {
            'optimized_shape': final_shape,
            'optimal_parameters': optimal_params,
            'history': history,
            'success': success,
            'type': 'sizeopt'
        }
    
    def _get_upstream_shape_node(self):
        """Get the upstream parametric shape node by traversing graph."""
        try:
            # Start search from 'shape' input
            port = self.get_input('shape')
            if not port: return None
            
            queue = [port]
            visited = set()
            
            while queue:
                curr_port = queue.pop(0)
                connected = curr_port.connected_ports()
                
                for cp in connected:
                    node = cp.node()
                    if node in visited: continue
                    visited.add(node)
                    
                    # Check if this node has the properties we want to optimize
                    # A robust way is to check if it has the parameters listed in 'parameters' property
                    # But we don't know them yet.
                    # Heuristic: If it has custom properties and is not a simulation node.
                    # Simple heuristic: Skip MeshNode, MaterialNode, etc.
                    
                    node_type = getattr(node, 'NODE_NAME', '')
                    if node_type in ['Mesh', 'Material', 'Filter', 'Remesh Surface']:
                        # Traverse upstream inputs of this node
                        for input_port in node.inputs().values():
                             queue.append(input_port)
                    else:
                        # Found a candidate node (e.g. Box, Cylinder, Script)
                        return node
                        
        except Exception as e:
            logger.debug(f"SizeOpt: Upstream search failed: {e}")
            pass
        return None
    
    def _run_fea(self, mesh, material, constraint_data, load_data):
        """Run FEA analysis on the mesh."""
        try:
            E = material['E']
            nu = material['nu']
            lam, mu = lam_lame(E, nu)
            
            # Setup basis
            e_vec = ElementVector(ElementTetP1())
            basis = Basis(mesh, e_vec)
            
            @BilinearForm
            def stiffness(u, v, w):
                def epsilon(w):
                    return sym_grad(w)
                return 2*mu*ddot(epsilon(u), epsilon(v)) + lam*tr(epsilon(u))*tr(epsilon(v))
            
            K = stiffness.assemble(basis)
            
            # Apply boundary conditions
            dofs_to_fix = self._apply_bc(mesh, basis, constraint_data)
            
            # Apply loads
            f = self._apply_loads(mesh, basis, load_data)
            
            # Solve
            with suppress_output():
                u = solve(*condense(K, f, D=dofs_to_fix))
            
            # Compute compliance
            compliance = float(f @ u)
            
            # Compute stress
            basis_p1 = Basis(mesh, ElementTetP1())
            
            @LinearForm
            def von_mises(v, w):
                u_interp = w['u']
                grad_u = u_interp.grad
                E11 = grad_u[0, 0]
                E22 = grad_u[1, 1]
                E33 = grad_u[2, 2]
                E12 = 0.5*(grad_u[0, 1] + grad_u[1, 0])
                E23 = 0.5*(grad_u[1, 2] + grad_u[2, 1])
                E13 = 0.5*(grad_u[0, 2] + grad_u[2, 0])
                trE = E11 + E22 + E33
                S11 = 2*mu*E11 + lam*trE
                S22 = 2*mu*E22 + lam*trE
                S33 = 2*mu*E33 + lam*trE
                S12 = 2*mu*E12
                S23 = 2*mu*E23
                S13 = 2*mu*E13
                vm = np.sqrt(0.5 * ((S11-S22)**2 + (S22-S33)**2 + (S33-S11)**2 + 6*(S12**2 + S23**2 + S13**2)))
                return vm * v
            
            @BilinearForm
            def mass(u, v, w):
                return u * v
            
            M = mass.assemble(basis_p1)
            b = von_mises.assemble(basis_p1, u=basis.interpolate(u))
            with suppress_output():
                stress = solve(M, b)
            stress = np.abs(stress)
            
            return {
                'displacement': u,
                'stress': stress,
                'compliance': compliance
            }
            
        except Exception as e:
            logger.warning(f"SizeOpt FEA failed: {e}")
            return None
    
    def _apply_bc(self, mesh, basis, constraint_data):
        """Apply boundary conditions and return DOFs to fix."""
        dofs_to_fix = np.array([], dtype=np.int64)

        if constraint_data is None:
            return dofs_to_fix

        # Handle both single constraint and list of constraints
        constraints = constraint_data if isinstance(constraint_data, list) else [constraint_data]

        for c in constraints:
            if c is None:
                continue

            fixed_dof_indices = c.get('fixed_dofs', [0, 1, 2])

            # --- Geometry-based constraint (from ConstraintNode / SelectFaceNode) ---
            geoms = c.get('geometries', None)
            if geoms is None and 'geometry' in c:
                geoms = [c['geometry']]
            geoms = [g for g in (geoms or []) if g is not None]

            if geoms:
                tolerance = 1.5
                from cadquery import Vector as _CQVector
                def _is_on_face(pts):
                    px, py, pz = pts[0], pts[1], pts[2]
                    mask = np.zeros(len(px), dtype=bool)
                    for i in range(len(px)):
                        pt = _CQVector(float(px[i]), float(py[i]), float(pz[i]))
                        for g in geoms:
                            try:
                                if g.distanceTo(pt) <= tolerance:
                                    mask[i] = True
                                    break
                            except Exception:
                                try:
                                    bb = g.BoundingBox()
                                    if (bb.xmin - tolerance <= px[i] <= bb.xmax + tolerance and
                                            bb.ymin - tolerance <= py[i] <= bb.ymax + tolerance and
                                            bb.zmin - tolerance <= pz[i] <= bb.zmax + tolerance):
                                        mask[i] = True
                                        break
                                except Exception:
                                    pass
                    return mask

                try:
                    facet_dofs = basis.get_dofs(_is_on_face)
                    for dof_idx in fixed_dof_indices:
                        dofs_to_fix = np.union1d(dofs_to_fix, facet_dofs.nodal[f'u^{dof_idx+1}'])
                except Exception as e:
                    logger.warning(f"SizeOpt BC geometry failed: {e}")
                continue  # Skip legacy condition check when geometry is present

            # --- Legacy string-based constraint ---
            condition = c.get('condition', '')
            if condition:
                try:
                    def make_cond(cond_str):
                        def condition_func(x):
                            if simple_eval is not None:
                                x_val, y_val, z_val = x
                                return simple_eval(cond_str,
                                                  names={'x': x_val, 'y': y_val, 'z': z_val},
                                                  functions={'sin': np.sin, 'cos': np.cos,
                                                             'abs': abs, 'sqrt': np.sqrt})
                            return eval(cond_str, {'x': x, 'np': np})
                        return condition_func

                    facet_dofs = basis.get_dofs(make_cond(condition))
                    for dof_idx in fixed_dof_indices:
                        dofs_to_fix = np.union1d(dofs_to_fix, facet_dofs.nodal[f'u^{dof_idx+1}'])
                except Exception as e:
                    logger.warning(f"SizeOpt BC condition failed: {e}")

        return dofs_to_fix
    
    def _apply_loads(self, mesh, basis, load_data):
        """Apply loads and return force vector."""
        f = basis.zeros()

        if load_data is None:
            return f

        loads = load_data if isinstance(load_data, list) else [load_data]

        for load in loads:
            if load is None:
                continue

            vector = load.get('vector', [0, 0, 0])

            # --- Geometry-based load (from LoadNode / SelectFaceNode) ---
            # This path was missing entirely, causing loads to be silently ignored
            # whenever a user connected a SelectFaceNode instead of typing a condition.
            geoms = load.get('geometries', None)
            if geoms is None and 'geometry' in load:
                geoms = [load['geometry']]
            geoms = [g for g in (geoms or []) if g is not None]

            if geoms:
                f_traction = _assemble_traction_force(mesh, basis, geoms, vector)
                if f_traction is not None:
                    f += f_traction
                else:
                    logger.warning(
                        f"SizeOpt: FacetBasis traction failed for load {vector} — "
                        "no mesh facets matched the selected geometry face."
                    )
                continue  # Skip legacy condition check when geometry is present

            # --- Legacy string-based load ---
            condition = load.get('condition', '')
            if condition:
                try:
                    def make_cond(cond_str):
                        def condition_func(x):
                            if simple_eval is not None:
                                x_val, y_val, z_val = x
                                return simple_eval(cond_str,
                                                  names={'x': x_val, 'y': y_val, 'z': z_val},
                                                  functions={'sin': np.sin, 'cos': np.cos,
                                                             'abs': abs, 'sqrt': np.sqrt})
                            return eval(cond_str, {'x': x, 'np': np})
                        return condition_func

                    facet_dofs = basis.get_dofs(make_cond(condition))
                    n_nodes = len(facet_dofs.nodal['u^1'])
                    for i, v in enumerate(vector):
                        f[facet_dofs.nodal[f'u^{i+1}']] = v / n_nodes
                except Exception as e:
                    logger.warning(f"SizeOpt load application failed: {e}")

        return f
    
    def _compute_volume(self, mesh):
        """Compute mesh volume."""
        try:
            basis0 = Basis(mesh, ElementTetP0())
            
            @LinearForm
            def unit(v, w):
                return 1.0 * v
            
            volumes = unit.assemble(basis0)
            return float(np.sum(volumes))
        except:
            return 0.0


