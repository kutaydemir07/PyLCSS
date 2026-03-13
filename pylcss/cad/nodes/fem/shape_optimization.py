# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.
"""FEM shape-optimisation node (biological stress leveling + adjoint compliance)."""
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

class ShapeOptimizationNode(CadQueryNode):
    """
    Shape Optimization Node - Optimizes boundary geometry via mesh morphing.

    Two sensitivity methods are available:

    Biological Stress Leveling (classical Fully-Stressed Design)
        Moves boundary nodes to drive the surface toward a *uniform* stress
        state: inward at low-stress zones (remove material) and outward at
        high-stress zones (add material).  Works well for stress-concentration
        reduction (fillets, notches) but is **not** rigorously minimising
        compliance.  It is a heuristic inspired by Wolff's Law of bone
        remodelling and should be labelled as such in publications.

    Adjoint Compliance (Hadamard-Zolesio shape derivative)
        Uses the mathematically exact shape derivative of compliance
            dC/dV_n  ≈  -2 W(u)  on the free boundary
        where W(u) = ½ σ:ε is the strain energy density.  Under a
        volume-preservation constraint this drives the boundary toward
        a *uniform strain-energy density* state, which IS the true
        optimality condition for minimum-compliance shape optimisation.
    """
    __identifier__ = 'com.cad.sim.shapeopt'
    NODE_NAME = 'Shape Optimization'

    def __init__(self):
        super().__init__()
        # Inputs
        self.add_input('mesh', color=(200, 100, 200))
        self.add_input('material', color=(200, 200, 200))
        self.add_input('constraints', color=(255, 100, 100))
        self.add_input('loads', color=(255, 255, 0))
        
        # Outputs
        self.add_output('optimized_mesh', color=(200, 100, 200))
        self.add_output('result', color=(200, 100, 200))
        
        # Optimization objective
        self.create_property('objective', 'Min Max Stress', widget_type='combo',
                             items=['Min Max Stress', 'Min Compliance', 'Uniform Stress'])
        
        # Optimization parameters
        self.create_property('max_iterations', 20, widget_type='int')
        self.create_property('step_size', 0.1, widget_type='float')  # Boundary movement scale
        self.create_property('smoothing_weight', 0.5, widget_type='float')  # Laplacian regularization
        self.create_property('convergence_tol', 0.01, widget_type='float')
        
        # Constraints
        self.create_property('volume_preservation', True, widget_type='bool')
        self.create_property('max_displacement', 5.0, widget_type='float')  # Max node movement
        
        # Fixed regions - JSON format: condition strings for faces that shouldn't move
        self.create_property('fixed_faces', '[]', widget_type='text')

        # Sensitivity method
        self.create_property('sensitivity_method', 'Biological Stress Leveling',
                             widget_type='combo',
                             items=['Biological Stress Leveling', 'Adjoint Compliance'])

        # Visualization
        self.create_property('visualization', 'Stress', widget_type='combo',
                             items=['Stress', 'Displacement', 'Shape Change'])

    def run(self, progress_callback=None):
        """Execute shape optimization."""
        import json
        
        logger.info("ShapeOpt: Starting shape optimization...")
        
        # Get inputs
        mesh_input = self.get_input_value('mesh', None)
        material = self.get_input_value('material', None)
        constraint_data = self.get_input_value('constraints', None)
        load_data = self.get_input_value('loads', None)
        
        # Extract mesh from RemeshNode dict output if needed
        if isinstance(mesh_input, dict) and 'mesh' in mesh_input:
            mesh = mesh_input['mesh']
            logger.info("ShapeOpt: Extracted mesh from RemeshNode output")
        else:
            mesh = mesh_input
        
        if not all([mesh, material, constraint_data, load_data]):
            logger.warning("ShapeOpt: Missing inputs!")
            return None
        
        # Get parameters
        obj_type = self.get_property('objective')
        max_iter = self.get_property('max_iterations')
        step_size = self.get_property('step_size')
        smooth_weight = self.get_property('smoothing_weight')
        conv_tol = self.get_property('convergence_tol')
        preserve_volume = self.get_property('volume_preservation')
        max_disp = self.get_property('max_displacement')
        
        # Parse fixed faces
        try:
            fixed_faces = json.loads(self.get_property('fixed_faces'))
        except:
            fixed_faces = []
        
        # Material properties
        E = material['E']
        nu = material['nu']
        lam, mu = lam_lame(E, nu)
        
        # Create a copy of mesh points for modification
        original_points = mesh.p.copy()
        current_points = mesh.p.copy()
        
        # Find boundary nodes
        # Use topology: boundary facets are those connected to only one element
        # mesh.f2t is (2, N_facets). -1 indicates no neighbor.
        f2t = mesh.f2t
        if f2t.shape[0] == 2:
            # Standard skfem
            boundary_facets = np.where(f2t[1, :] == -1)[0]
        else:
            # Fallback (very old skfem?)
            boundary_facets = mesh.facets_satisfying(lambda x: True)
            
        boundary_nodes = np.unique(mesh.facets[:, boundary_facets].flatten())
        
        # Find fixed boundary nodes (from constraints)
        fixed_nodes = self._get_fixed_nodes(mesh, constraint_data, fixed_faces)
        
        # Moveable boundary nodes
        moveable_nodes = np.setdiff1d(boundary_nodes, fixed_nodes)
        
        logger.info(f"ShapeOpt: {len(moveable_nodes)} moveable boundary nodes, {len(fixed_nodes)} fixed")
        
        # Compute boundary node normals
        normals = self._compute_boundary_normals(mesh, moveable_nodes)
        
        # History tracking
        history = {'iterations': [], 'objective': [], 'max_stress': [], 'volume': []}
        
        # Initial volume
        initial_volume = self._compute_volume(mesh)
        
        best_objective = float('inf')
        best_points = current_points.copy()

        # ── Precompute adjacency and interior nodes for Laplacian morphing ──────
        # Done once — connectivity never changes during shape optimisation.
        _n_pts_so = mesh.p.shape[1]
        _adj_so   = [[] for _ in range(_n_pts_so)]
        for _ec in mesh.t.T:
            for _na in _ec:
                for _nb in _ec:
                    if _na != _nb:
                        _adj_so[_na].append(int(_nb))
        _adj_so = [np.unique(a) for a in _adj_so]
        _bnd_set_so = set(boundary_nodes.tolist())
        _interior_nodes_so = np.array(
            [i for i in range(_n_pts_so) if i not in _bnd_set_so], dtype=int
        )
        logger.info(
            f"ShapeOpt: {len(_interior_nodes_so)} interior nodes will be "
            "Laplacian-morphed each iteration."
        )

        for itr in range(max_iter):
            # Create mesh with current points
            mesh_curr = skfem.MeshTet(current_points, mesh.t)
            
            # Run FEA
            result = self._run_fea(mesh_curr, lam, mu, constraint_data, load_data)
            if result is None:
                logger.warning(f"ShapeOpt Iter {itr}: FEA failed, stopping")
                break
            
            stress = result['stress']
            compliance = result['compliance']
            max_stress = np.max(stress)
            current_volume = self._compute_volume(mesh_curr)
            
            # Compute objective
            if obj_type == 'Min Max Stress':
                obj = max_stress
            elif obj_type == 'Min Compliance':
                obj = compliance
            elif obj_type == 'Uniform Stress':
                obj = np.std(stress)  # Minimize stress variance
            else:
                obj = max_stress
            
            # Track history
            history['iterations'].append(itr)
            history['objective'].append(obj)
            history['max_stress'].append(max_stress)
            history['volume'].append(current_volume)
            
            logger.info(f"ShapeOpt Iter {itr}: Obj={obj:.4f}, MaxStress={max_stress:.2f}, Vol={current_volume:.4f}")
            
            # Check convergence
            if itr > 0:
                rel_change = abs(obj - best_objective) / (abs(best_objective) + 1e-10)
                if rel_change < conv_tol:
                    logger.info(f"ShapeOpt: Converged at iteration {itr}")
                    break
            
            if obj < best_objective:
                best_objective = obj
                best_points = current_points.copy()
            
            # Compute shape sensitivities on boundary nodes
            sensitivities = self._compute_shape_sensitivity(
                mesh_curr, result, moveable_nodes, normals, obj_type, lam, mu
            )
            
            # Apply Laplacian smoothing for regularization
            sensitivities = self._laplacian_smooth(
                sensitivities, moveable_nodes, mesh_curr, smooth_weight
            )
            
            # Update boundary node positions
            # Fix broadcasting: sens (N,) * normals (N,3) needs sens (N,1)
            move = -step_size * sensitivities[:, np.newaxis] * normals

            # Compute reference signed element volumes BEFORE the trial move.
            # We compare signs rather than using a fixed threshold so that meshes
            # with right-hand vs left-hand winding (all-negative Jacobians) are
            # handled correctly — a pure threshold of 1e-9 always triggers on such
            # meshes and permanently blocks every update.
            try:
                _p_ref = current_points
                _t_ref = mesh.t
                _e1r = _p_ref[:, _t_ref[1]] - _p_ref[:, _t_ref[0]]
                _e2r = _p_ref[:, _t_ref[2]] - _p_ref[:, _t_ref[0]]
                _e3r = _p_ref[:, _t_ref[3]] - _p_ref[:, _t_ref[0]]
                _ref_vols = np.sum(np.cross(_e1r, _e2r, axis=0) * _e3r, axis=0) / 6.0
                _ref_signs = np.sign(_ref_vols)
            except Exception:
                _ref_vols = None
                _ref_signs = None

            # Simple line search / backtracking to prevent inverted elements
            alpha = 1.0
            for backtrack in range(5):
                trial_points = current_points.copy()
                for i, node_idx in enumerate(moveable_nodes):
                     if i < len(move):
                        trial_points[:, node_idx] += alpha * move[i]
                
                # Check mesh quality (Jacobian)
                trial_mesh = skfem.MeshTet(trial_points, mesh.t)
                quality_ok = True
                
                try:
                    # Calculate element volumes (signed)
                    # Skfem doesn't expose jacobian directly easily on mesh object, 
                    # but element volumes < 0 means inversion.
                    mapping = skfem.mapping.MappingAffine(trial_mesh)
                    # evaluating determinant at quadrature points?
                    # Simpler: use element_finder or base geometry approach.
                    # Or just check if mapping.detJ is positive? 
                    # MappingAffine calculates J based on vertices.
                    
                    # detJ is roughly 6 * Volume for Tet
                    # Let's use internal method if available or verify volumes.
                    # For Tet mesh, mesh.element_volumes is available in newer skfem?
                    # No, but we can compute it.
                    
                    # Manual volume check:
                    # V = 1/6 * det([x1-x0, x2-x0, x3-x0])
                    # Vectorized:
                    p = trial_points
                    t = mesh.t
                    v0 = p[:, t[0, :]]
                    v1 = p[:, t[1, :]]
                    v2 = p[:, t[2, :]]
                    v3 = p[:, t[3, :]]
                    
                    # Edge vectors
                    e1 = v1 - v0
                    e2 = v2 - v0
                    e3 = v3 - v0
                    
                    # Mixed product
                    cross = np.cross(e1, e2, axis=0)
                    vols = np.sum(cross * e3, axis=0) / 6.0
                    
                    # Detect inversion by sign-flip relative to reference orientation.
                    # A fixed threshold (< 1e-9) incorrectly triggers on left-hand-wound
                    # meshes where all reference volumes are legitimately negative.
                    if _ref_signs is not None:
                        inverted = np.any(np.sign(vols) != _ref_signs)
                    else:
                        inverted = np.min(np.abs(vols)) <= 1e-9
                    if inverted:
                        min_abs = np.min(np.abs(vols))
                        logger.warning(f"ShapeOpt: Mesh inversion detected (min |vol| {min_abs:.2e}). Backtracking...")
                        quality_ok = False
                except Exception:
                     pass
                
                if quality_ok:
                    current_points = trial_points
                    break
                else:
                    alpha *= 0.5
            else:
                logger.warning("ShapeOpt: Step failed quality check completely. Stopping.")
                break

            # ── Interior-node Laplacian relaxation (mesh morphing) ────────────
            # Propagate boundary deformation into the mesh interior to avoid
            # shear-mode element inversion caused by surface nodes moving
            # while interior nodes stay fixed.  3 passes distribute the
            # deformation smoothly with negligible computational overhead.
            current_points = self._relax_interior_nodes(
                current_points, _adj_so, _interior_nodes_so, n_relax=3
            )

            # Volume preservation
            if preserve_volume:
                new_volume = self._compute_volume(skfem.MeshTet(current_points, mesh.t))
                scale = (initial_volume / new_volume) ** (1/3)
                # Scale only boundary movements, not entire mesh
                for node_idx in moveable_nodes:
                    disp = current_points[:, node_idx] - original_points[:, node_idx]
                    current_points[:, node_idx] = original_points[:, node_idx] + disp * scale
            
            # Progress callback
            if progress_callback:
                try:
                    progress_callback(mesh_curr, stress, itr, max_iter)
                except:
                    pass
        
        # Create final optimized mesh
        optimized_mesh = skfem.MeshTet(best_points, mesh.t)
        
        # Final FEA on optimized mesh
        final_result = self._run_fea(optimized_mesh, lam, mu, constraint_data, load_data)
        
        logger.info(f"ShapeOpt: Optimization complete. Final objective: {best_objective:.4f}")
        
        return {
            'mesh': optimized_mesh,
            'stress': final_result['stress'] if final_result else None,
            'displacement': final_result['displacement'] if final_result else None,
            'history': history,
            'type': 'shapeopt',
            'visualization_mode': self.get_property('visualization')
        }
    
    def _get_fixed_nodes(self, mesh, constraint_data, fixed_faces):
        """Get nodes that should not move (from constraints and fixed faces)."""
        fixed = np.array([], dtype=np.int64)
        x, y, z = mesh.p
        node_coords = np.column_stack((x, y, z))
        
        if constraint_data:
            constraints = constraint_data if isinstance(constraint_data, list) else [constraint_data]
            for c in constraints:
                if c is None:
                    continue

                # Support both 'geometry' (single) and 'geometries' (multi-face) keys.
                if 'geometry' in c or 'geometries' in c:
                    geoms_gfn = c.get('geometries', None)
                    if geoms_gfn is None and 'geometry' in c:
                        geoms_gfn = [c['geometry']]
                    geoms_gfn = [g for g in (geoms_gfn or []) if g is not None]

                    # Use 1.5 mm tolerance (was 1e-3 — far too tight for coarse meshes).
                    tolerance = 1.5
                    from cadquery import Vector as _CQVector_gfn

                    if geoms_gfn:
                        bbox_list_gfn = [g.BoundingBox() for g in geoms_gfn]
                        xmin_gfn = min(b.xmin for b in bbox_list_gfn) - tolerance
                        xmax_gfn = max(b.xmax for b in bbox_list_gfn) + tolerance
                        ymin_gfn = min(b.ymin for b in bbox_list_gfn) - tolerance
                        ymax_gfn = max(b.ymax for b in bbox_list_gfn) + tolerance
                        zmin_gfn = min(b.zmin for b in bbox_list_gfn) - tolerance
                        zmax_gfn = max(b.zmax for b in bbox_list_gfn) + tolerance

                        for i, coord in enumerate(node_coords):
                            if not (xmin_gfn <= coord[0] <= xmax_gfn and
                                    ymin_gfn <= coord[1] <= ymax_gfn and
                                    zmin_gfn <= coord[2] <= zmax_gfn):
                                continue
                            pt = _CQVector_gfn(coord[0], coord[1], coord[2])
                            for g in geoms_gfn:
                                try:
                                    if g.distanceTo(pt) <= tolerance:
                                        fixed = np.union1d(fixed, [i])
                                        break
                                except Exception:
                                    try:
                                        bb = g.BoundingBox()
                                        if (bb.xmin - tolerance <= coord[0] <= bb.xmax + tolerance and
                                                bb.ymin - tolerance <= coord[1] <= bb.ymax + tolerance and
                                                bb.zmin - tolerance <= coord[2] <= bb.zmax + tolerance):
                                            fixed = np.union1d(fixed, [i])
                                            break
                                    except Exception:
                                        pass
                
                elif 'condition' in c and c['condition']:
                    # LEGACY: String-based condition
                    condition = c['condition']
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
                        facets = mesh.facets_satisfying(make_cond(condition))
                        fixed = np.union1d(fixed, np.unique(mesh.facets[:, facets].flatten()))
                    except Exception as e:
                        logger.warning(f"ShapeOpt: BC condition failed: {e}")

        # Additional fixed faces from property (string conditions)
        for face_cond in fixed_faces:
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
                facets = mesh.facets_satisfying(make_cond(face_cond))
                fixed = np.union1d(fixed, np.unique(mesh.facets[:, facets].flatten()))
            except Exception as e:
                logger.warning(f"ShapeOpt: Fixed face condition failed: {e}")
        
        return fixed.astype(np.int64)
    
    def _compute_boundary_normals(self, mesh, boundary_nodes):
        """Compute outward normals at boundary nodes."""
        normals = np.zeros((len(boundary_nodes), 3))
        
        # For each boundary node, average normals of adjacent boundary facets
        for i, node_idx in enumerate(boundary_nodes):
            # Find facets containing this node
            facet_mask = np.any(mesh.facets == node_idx, axis=0)
            adjacent_facets = np.where(facet_mask)[0]
            
            node_normal = np.zeros(3)
            for facet_idx in adjacent_facets:
                # Get facet vertices
                facet_nodes = mesh.facets[:, facet_idx]
                pts = mesh.p[:, facet_nodes]
                
                # Compute facet normal
                v1 = pts[:, 1] - pts[:, 0]
                v2 = pts[:, 2] - pts[:, 0]
                normal = np.cross(v1, v2)
                norm = np.linalg.norm(normal)
                if norm > 1e-10:
                    node_normal += normal / norm
            
            # Normalize
            norm = np.linalg.norm(node_normal)
            if norm > 1e-10:
                normals[i] = node_normal / norm
            else:
                normals[i] = np.array([0, 0, 1])
        
        return normals
    
    def _compute_shape_sensitivity(self, mesh, result, moveable_nodes, normals, obj_type, lam, mu):
        """Compute shape sensitivity at boundary nodes.

        The update rule applied by the caller is:
            move = -step_size * sensitivities[:, newaxis] * normals
        so a *positive* sensitivity moves the node *inward* (removes material)
        and a *negative* sensitivity moves it *outward* (adds material).

        Biological Stress Leveling
        --------------------------
        sens_i = (sigma_i - sigma_mean) / sigma_mean
        High-stress nodes  (sens > 0) => inward? -- NO.  For biological growth
        we WANT to add material at high-stress zones, so we negate:
            sens_i = (sigma_mean - sigma_i) / sigma_mean
        => high stress: sens < 0 => move outward  (add material, reduce stress) OK
        => low  stress: sens > 0 => move inward   (remove material)             OK

        Adjoint Compliance  (Hadamard-Zolesio boundary shape derivative)
        ----------------------------------------------------------------
        dC/dV_n = -2 W(u)  where W = strain energy density = ½ σ:ε
        To *minimise* C under a volume constraint we move outward where W is
        above average (add material where the structure is working hardest) and
        inward where W is below average.  Using the same sign convention:
            sens_i = (W_mean - W_i) / W_mean
        => high SED: sens < 0 => move outward (add material, reduce compliance)
        => low  SED: sens > 0 => move inward  (remove idle material)
        """
        sensitivity_method = self.get_property('sensitivity_method')

        stress = result['stress']
        sensitivities = np.zeros(len(moveable_nodes))

        if sensitivity_method == 'Adjoint Compliance' and 'sed' in result:
            # --- Adjoint Compliance (Hadamard shape derivative) ---
            sed = result['sed']  # strain energy density, nodal projection
            mean_sed = np.mean(sed) + 1e-30
            for i, node_idx in enumerate(moveable_nodes):
                w_i = float(sed[node_idx]) if node_idx < len(sed) else mean_sed
                sensitivities[i] = (mean_sed - w_i) / mean_sed  # sign: outward at high SED
        else:
            # --- Biological Stress Leveling (Fully-Stressed Design heuristic) ---
            # NOTE: pure stress-levelling; NOT a rigorous compliance minimiser.
            mean_stress = np.mean(stress) + 1e-30
            for i, node_idx in enumerate(moveable_nodes):
                sigma_i = float(stress[node_idx]) if node_idx < len(stress) else mean_stress
                # Positive at low-stress (remove material), negative at high-stress (add)
                sensitivities[i] = (mean_stress - sigma_i) / mean_stress

        # Normalise to [-1, 1] to decouple from absolute magnitude
        sens_max = np.max(np.abs(sensitivities))
        if sens_max > 1e-10:
            sensitivities /= sens_max

        return sensitivities
    
    def _laplacian_smooth(self, values, nodes, mesh, weight):
        """Apply Laplacian smoothing to sensitivity values."""
        if weight <= 0:
            return values
        
        smoothed = values.copy()
        
        # Build node adjacency for boundary nodes
        node_to_idx = {node: i for i, node in enumerate(nodes)}
        
        for i, node_idx in enumerate(nodes):
            # Find adjacent boundary nodes
            # Simple approximation: nodes in same elements
            elem_mask = np.any(mesh.t == node_idx, axis=0)
            adjacent_nodes = np.unique(mesh.t[:, elem_mask].flatten())
            adjacent_boundary = [n for n in adjacent_nodes if n in node_to_idx and n != node_idx]
            
            if adjacent_boundary:
                neighbor_vals = [values[node_to_idx[n]] for n in adjacent_boundary]
                avg = np.mean(neighbor_vals)
                smoothed[i] = (1 - weight) * values[i] + weight * avg
        
        return smoothed

    def _relax_interior_nodes(self, pts, adjacency, interior_nodes, n_relax=3):
        """
        Propagate boundary-node motion into the mesh interior via Laplacian
        smoothing.  Called once per shape-optimisation iteration after the
        boundary nodes have been updated.

        Only *interior* nodes (those not on the mesh boundary) are repositioned;
        boundary nodes — both fixed and moveable — are left untouched so that
        the shape update is not undone.

        Parameters
        ----------
        pts            : (3, N) ndarray  — current node coordinates
        adjacency      : list of (K_i,) int arrays  — per-node neighbour lists,
                         precomputed once before the optimisation loop
        interior_nodes : (N_i,) int array  — node indices to be smoothed
        n_relax        : int  — number of Jacobi passes (default 3)

        Returns
        -------
        pts_new : (3, N) ndarray  — updated node coordinates
        """
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

    def _run_fea(self, mesh, lam, mu, constraint_data, load_data):
        """Run FEA on mesh."""
        try:
            e_vec = ElementVector(ElementTetP1())
            basis = Basis(mesh, e_vec)
            
            @BilinearForm
            def stiffness(u, v, w):
                def epsilon(w):
                    return sym_grad(w)
                return 2*mu*ddot(epsilon(u), epsilon(v)) + lam*tr(epsilon(u))*tr(epsilon(v))
            
            K = stiffness.assemble(basis)
            
            # Apply BC - handle both geometry-based and string-based conditions
            dofs_to_fix = np.array([], dtype=np.int64)
            x, y, z = mesh.p
            
            if constraint_data:
                constraints = constraint_data if isinstance(constraint_data, list) else [constraint_data]
                for c in constraints:
                    if c is None:
                        continue
                    
                    fixed_dof_indices = c.get('fixed_dofs', [0, 1, 2])
                    
                    # Geometry-based constraint — supports both single 'geometry'
                    # and multi-face 'geometries' keys.  Uses basis.get_dofs() so
                    # that mid-edge / face-centre DOFs of P2 elements are captured.
                    if 'geometry' in c or 'geometries' in c:
                        geoms_sh = c.get('geometries', None)
                        if geoms_sh is None and 'geometry' in c:
                            geoms_sh = [c['geometry']]
                        geoms_sh = [g for g in (geoms_sh or []) if g is not None]

                        if geoms_sh:
                            tolerance = 1.5
                            from cadquery import Vector as _CQVector
                            def _is_on_face_sh(pts):
                                px, py, pz = pts[0], pts[1], pts[2]
                                mask = np.zeros(len(px), dtype=bool)
                                for i in range(len(px)):
                                    pt = _CQVector(float(px[i]), float(py[i]), float(pz[i]))
                                    for g in geoms_sh:
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
                                facet_dofs_sh = basis.get_dofs(_is_on_face_sh)
                                for dof_idx in fixed_dof_indices:
                                    dofs_to_fix = np.union1d(
                                        dofs_to_fix, facet_dofs_sh.nodal[f'u^{dof_idx+1}']
                                    )
                                logger.debug(
                                    f"ShapeOpt: Fixed {sum(len(facet_dofs_sh.nodal[f'u^{k+1}']) for k in range(3))} "
                                    "DOFs from geometry via get_dofs"
                                )
                            except Exception as e:
                                logger.warning(f"ShapeOpt BC geometry failed: {e}")
                        
                    elif 'condition' in c and c['condition']:
                        # LEGACY: String-based condition
                        condition = c['condition']
                        try:
                            def make_cond(cond_str):
                                def condition_func(x):
                                    return eval(cond_str, {'x': x, 'np': np})
                                return condition_func
                            facet_dofs = basis.get_dofs(make_cond(condition))
                            for dof_idx in fixed_dof_indices:
                                dofs_to_fix = np.union1d(dofs_to_fix, facet_dofs.nodal[f'u^{dof_idx+1}'])
                        except Exception as e:
                            logger.warning(f"ShapeOpt BC condition failed: {e}")
            
            # Apply loads - handle both geometry and condition
            f = basis.zeros()
            if load_data:
                loads = load_data if isinstance(load_data, list) else [load_data]
                for load in loads:
                    if load is None:
                        continue
                    
                    vector = load.get('vector', [0, 0, 0])
                    
                    if 'geometries' in load or 'geometry' in load:
                        # Geometry-based load via area-weighted FacetBasis traction.
                        # Equal nodal distribution is NOT used — on unstructured meshes
                        # it concentrates force artificially on dense mesh regions.
                        geoms_sh = load.get('geometries', None)
                        if geoms_sh is None and 'geometry' in load:
                            geoms_sh = [load['geometry']]
                        geoms_sh = [g for g in (geoms_sh or []) if g is not None]

                        if geoms_sh:
                            f_traction = _assemble_traction_force(
                                mesh, basis, geoms_sh, vector
                            )
                            if f_traction is not None:
                                f += f_traction
                                logger.debug(
                                    f"ShapeOpt: Load {vector} applied via FacetBasis traction."
                                )
                            else:
                                logger.error(
                                    f"ShapeOpt: FacetBasis traction failed for load {vector}. "
                                    "No load applied — check geometry face vs mesh boundary."
                                )
                        
                    elif 'condition' in load and load['condition']:
                        # LEGACY: String-based condition
                        condition = load['condition']
                        try:
                            def make_cond(cond_str):
                                def condition_func(x):
                                    return eval(cond_str, {'x': x, 'np': np})
                                return condition_func
                            facet_dofs = basis.get_dofs(make_cond(condition))
                            n_nodes = len(facet_dofs.nodal['u^1'])
                            for i, v in enumerate(vector):
                                if n_nodes > 0:
                                    f[facet_dofs.nodal[f'u^{i+1}']] = v / n_nodes
                        except Exception as e:
                            logger.warning(f"ShapeOpt load failed: {e}")
            
            # Check if we have valid BCs
            if len(dofs_to_fix) == 0:
                logger.warning("ShapeOpt: No DOFs fixed - check constraint setup")
            
            # Solve
            with suppress_output():
                u = solve(*condense(K, f, D=dofs_to_fix))
            
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

            # --- Strain Energy Density (for Adjoint Compliance sensitivity) ---
            # W = mu * (eps:eps) + (lam/2) * (tr eps)^2  (linear elasticity)
            # We project it onto the same P1 nodal space via L2 projection.
            sed = np.zeros_like(stress)  # default: zero if anything fails
            try:
                @LinearForm
                def sed_form(v, w):
                    u_i = w['u']
                    g = u_i.grad
                    E11 = g[0, 0]; E22 = g[1, 1]; E33 = g[2, 2]
                    E12 = 0.5 * (g[0, 1] + g[1, 0])
                    E23 = 0.5 * (g[1, 2] + g[2, 1])
                    E13 = 0.5 * (g[0, 2] + g[2, 0])
                    trE = E11 + E22 + E33
                    W = (mu * (E11**2 + E22**2 + E33**2
                               + 2*E12**2 + 2*E23**2 + 2*E13**2)
                         + 0.5 * lam * trE**2)
                    return W * v
                b_sed = sed_form.assemble(basis_p1, u=basis.interpolate(u))
                with suppress_output():
                    sed = solve(M, b_sed)   # reuse mass matrix from Von Mises
                sed = np.abs(sed)
            except Exception as _sed_err:
                logger.debug(f"ShapeOpt: SED computation skipped: {_sed_err}")

            return {
                'displacement': u,
                'stress': stress,
                'sed': sed,
                'compliance': compliance
            }
            
        except Exception as e:
            logger.warning(f"ShapeOpt FEA failed: {e}")
            return None
    
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
