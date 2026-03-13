# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.
"""FEM topology optimization node — SIMP method with filters and shape recovery."""
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
    lam_lame, tr,
    build_filter_matrix, sensitivity_filter, density_filter_3d,
    density_filter_chainrule, heaviside_projection, mma_update, shape_recovery,
    suppress_output, OCCGeometry,
)

class TopologyOptimizationNode(CadQueryNode):
    """Performs Topology Optimization (SIMP Method) with sensitivity filtering and shape recovery."""
    __identifier__ = 'com.cad.sim.topopt'
    NODE_NAME = 'Topology Opt'

    def __init__(self):
        super().__init__()
        self.add_input('mesh', color=(200, 100, 200))
        self.add_input('material', color=(200, 200, 200))
        self.add_input('constraints', color=(255, 100, 100), multi_input=True)
        self.add_input('loads', color=(255, 255, 0), multi_input=True)
        self.add_output('optimized_mesh', color=(200, 100, 200))
        self.add_output('recovered_shape', color=(100, 255, 100))
        self.create_property('vol_frac', 0.4, widget_type='float')
        self.create_property('iterations', 50, widget_type='int')  # Reasonable default
        self.create_property('filter_radius', 3.0, widget_type='float')  # Should be 2-3x element size
        self.create_property('density_cutoff', 0.3, widget_type='float')
        self.create_property('shape_recovery', True, widget_type='bool')
        self.create_property('visualization', 'Density', widget_type='combo', items=['Density', 'Recovered Shape', 'Von Mises Stress'])
        # NEW: Symmetry properties
        self.create_property('symmetry_x', None, widget_type='float')  # None means no symmetry
        self.create_property('symmetry_y', None, widget_type='float')
        self.create_property('symmetry_z', None, widget_type='float')
        # NEW: Optimization parameters (previously hardcoded)
        self.create_property('penal', 3.0, widget_type='float')  # SIMP penalization exponent
        self.create_property('move_limit', 0.2, widget_type='float')  # Max density change per iteration
        self.create_property('min_density', 0.001, widget_type='float')  # Minimum element density
        self.create_property('convergence_tol', 0.02, widget_type='float')  # Convergence threshold
        self.create_property('recovery_resolution', 100, widget_type='int')  # Grid resolution for shape recovery
        self.create_property('smoothing_iterations', 3, widget_type='int')  # Gaussian smoothing passes
        # NEW: Filter type and update scheme selection
        self.create_property('filter_type', 'density', widget_type='combo', items=['sensitivity', 'density'])
        self.create_property('update_scheme', 'MMA', widget_type='combo', items=['MMA', 'OC'])

        # Element type for displacement basis.
        # Linear P1 is fast (recommended for many iterations) but exhibits
        # shear/volumetric locking in bending, making structures appear stiffer
        # than they really are and producing overly thin optimised members.
        # Quadratic P2 is significantly more accurate but ~4-8x slower per solve.
        self.create_property('element_type', 'Fast (Linear P1)', widget_type='combo',
                             items=['Fast (Linear P1)', 'Accurate (Quadratic P2)'])

        # NEW: Heaviside projection for sharper boundaries
        self.create_property('projection', 'None', widget_type='combo', 
                             items=['None', 'Heaviside'])
        self.create_property('heaviside_beta', 4.0, widget_type='float')  # Sharpness (1-64)
        self.create_property('heaviside_eta', 0.5, widget_type='float')   # Threshold (0-1)
        self.create_property('continuation', True, widget_type='bool')   # Gradually increase beta


    def run(self, progress_callback=None):
        logger.info("TopOpt: Optimization started.")
        
        # Helper to flatten inputs for list validation
        def flatten_inputs(inputs):
            res = []
            for item in inputs:
                if isinstance(item, list):
                    res.extend(flatten_inputs(item))
                elif item is not None:
                    res.append(item)
            return res

        mesh = self.get_input_value('mesh', None)
        material = self.get_input_value('material', None)
        
        # Fetch lists and validate
        constraint_list = flatten_inputs(self.get_input_list('constraints'))
        load_list = flatten_inputs(self.get_input_list('loads'))
        
        # Resolve parametric inputs with fallbacks to properties
        vol_frac = self.get_input_value('vol_frac', 'vol_frac')
        vol_frac = float(vol_frac)
        max_iter = self.get_property('iterations')
        filter_radius = self.get_input_value('filter_radius', 'filter_radius')
        filter_radius = float(filter_radius)
        shape_recovery_enabled = self.get_property('shape_recovery')
        
        # NEW: Get symmetry plane coordinates
        sym_x = self.get_input_value('symmetry_x', 'symmetry_x')
        sym_y = self.get_input_value('symmetry_y', 'symmetry_y')
        sym_z = self.get_input_value('symmetry_z', 'symmetry_z')
        
        # NEW: Get optimization parameters (previously hardcoded)
        penal = self.get_input_value('penal', 'penal')
        penal = float(penal)
        move = self.get_input_value('move_limit', 'move_limit')
        move = float(move)
        rho_min = self.get_input_value('min_density', 'min_density')
        rho_min = float(rho_min)
        conv_tol = self.get_input_value('convergence_tol', 'convergence_tol')
        conv_tol = float(conv_tol)
        recovery_res = self.get_property('recovery_resolution')
        smoothing_iter = self.get_property('smoothing_iterations')
        
        if not (mesh and material and constraint_list and load_list):
            logger.warning(f"TopOpt: Missing inputs! Mesh:{mesh is not None}, Mat:{material is not None}, Cons:{len(constraint_list)} items, Load:{len(load_list)} items")
            return None

        logger.info("TopOpt: Inputs confirmed. Setting up basis...")

        # 1. Setup Basis (Vector for displacement, Scalar P0 for density)
        _elem_type = self.get_property('element_type')
        if _elem_type == 'Accurate (Quadratic P2)':
            logger.info("TopOpt: Using quadratic P2 elements (accurate, slower).")
            e_vec = ElementVector(ElementTetP2())
        else:
            logger.warning(
                "TopOpt: Using linear P1 elements (P1/T4).  In bending-dominated "
                "topologies these elements exhibit volumetric locking — the optimizer "
                "may converge to an overly stiff, strut-dominated layout that routes "
                "load through shear members to avoid bending.  If the result looks "
                "unexpectedly truss-like or far stiffer than the P2 result, switch the "
                "Element Type property to 'Accurate (Quadratic P2)' to eliminate locking."
            )
            e_vec = ElementVector(ElementTetP1())
        basis = Basis(mesh, e_vec)
        
        # Density basis (P0 - constant per element)
        basis0 = basis.with_element(ElementTetP0())
        
        # Get element centroids for filtering and symmetry
        centroids = mesh.p[:, mesh.t].mean(axis=1).T
        
        # PRE-CALCULATE SYMMETRY MAPPINGS (The Fast Way - O(N log N) once)
        sym_map_x = None
        sym_map_y = None
        sym_map_z = None
        
        if sym_x is not None or sym_y is not None or sym_z is not None:
            from scipy.spatial import cKDTree
            tree = cKDTree(centroids)  # Build spatial tree once
            
            if sym_x is not None:
                # Create mirror points across x = sym_x plane
                mirror_pts = centroids.copy()
                mirror_pts[:, 0] = 2 * sym_x - mirror_pts[:, 0]
                # Query nearest neighbors for all points at once
                dists, indices = tree.query(mirror_pts)
                # Filter valid symmetric pairs (within tolerance)
                # FIX #2: Tolerance raised from 1e-3 to 0.1 mm.
                # Mirror-point centroids computed from floating-point arithmetic can
                # differ by 0.01–0.05 mm from their true mirror, causing all symmetric
                # pairs to be missed with the original 0.001 mm tolerance.
                valid_mask = dists < 0.1
                sym_map_x = (np.where(valid_mask)[0], indices[valid_mask])
            
            if sym_y is not None:
                # Create mirror points across y = sym_y plane
                mirror_pts = centroids.copy()
                mirror_pts[:, 1] = 2 * sym_y - mirror_pts[:, 1]
                # Query nearest neighbors
                dists, indices = tree.query(mirror_pts)
                valid_mask = dists < 0.1
                sym_map_y = (np.where(valid_mask)[0], indices[valid_mask])
            
            if sym_z is not None:
                # Create mirror points across z = sym_z plane
                mirror_pts = centroids.copy()
                mirror_pts[:, 2] = 2 * sym_z - mirror_pts[:, 2]
                # Query nearest neighbors
                dists, indices = tree.query(mirror_pts)
                valid_mask = dists < 0.1
                sym_map_z = (np.where(valid_mask)[0], indices[valid_mask])
        
        # Initialize density
        # Start with uniform density equal to volume fraction
        densities = np.ones(basis0.N) * vol_frac
        
        # Apply symmetry constraints to initial density (fast version)
        # Average the mirrored pair — preserves volume fraction and avoids the
        # progressive density loss that np.minimum causes when the two sides
        # differ (e.g. asymmetric sensitivity at early iterations).
        if sym_map_x is not None:
            src_indices, target_indices = sym_map_x
            avg_densities = 0.5 * (densities[src_indices] + densities[target_indices])
            densities[src_indices] = avg_densities
            densities[target_indices] = avg_densities
        
        if sym_map_y is not None:
            src_indices, target_indices = sym_map_y
            avg_densities = 0.5 * (densities[src_indices] + densities[target_indices])
            densities[src_indices] = avg_densities
            densities[target_indices] = avg_densities
        
        if sym_map_z is not None:
            src_indices, target_indices = sym_map_z
            avg_densities = 0.5 * (densities[src_indices] + densities[target_indices])
            densities[src_indices] = avg_densities
            densities[target_indices] = avg_densities
        
        # Material
        E_mat = material['E']
        nu_mat = material['nu']
        lam_val, mu_val = lam_lame(E_mat, nu_mat)
        
        # Create constant fields for material properties
        lam_field = basis0.zeros() + lam_val
        mu_field = basis0.zeros() + mu_val
        
        # Boundary Conditions (Fixed)
        x, y, z = mesh.p
        debug_constraints = []
        debug_loads = []
        
        fixed_dofs = np.array([], dtype=int)
        # Prescribed displacement vector for non-zero Displacement BCs.
        # Initialised here (before the loop); values are filled in below.
        u_prescribed = np.zeros(basis.N)

        # --- PROCESS CONSTRAINTS ---
        try:
            for c in constraint_list:
                if not c: continue

                fixed_dof_indices = c.get('fixed_dofs', [0, 1, 2])
                disp_vals = c.get('displacement', None)  # [dx, dy, dz] or None

                # Handle both 'geometries' (list, from SelectFaceNode) and legacy 'geometry' (single)
                geoms = c.get('geometries', None)
                if geoms is None and 'geometry' in c:
                    geoms = [c['geometry']]
                geoms = [g for g in (geoms or []) if g is not None]

                if geoms:
                    tolerance = 1.5

                    # Use skfem's native DOF locator so that mid-edge / face-centre
                    # DOFs introduced by quadratic (P2) elements are also captured.
                    # FIX #3b: Use default-argument capture (_geoms=geoms) to avoid
                    # the Python late-binding closure bug.
                    from cadquery import Vector as _CQVector
                    def _is_on_face(pts, _geoms=geoms, _tol=tolerance):
                        px, py, pz = pts[0], pts[1], pts[2]
                        mask = np.zeros(len(px), dtype=bool)
                        for i in range(len(px)):
                            pt = _CQVector(float(px[i]), float(py[i]), float(pz[i]))
                            for g in _geoms:
                                try:
                                    if g.distanceTo(pt) <= _tol:
                                        mask[i] = True
                                        break
                                except Exception:
                                    try:
                                        bb = g.BoundingBox()
                                        if (bb.xmin - _tol <= px[i] <= bb.xmax + _tol and
                                                bb.ymin - _tol <= py[i] <= bb.ymax + _tol and
                                                bb.zmin - _tol <= pz[i] <= bb.zmax + _tol):
                                            mask[i] = True
                                            break
                                    except Exception:
                                        pass
                        return mask

                    facet_dofs = basis.get_dofs(_is_on_face)
                    for dof_idx in fixed_dof_indices:
                        dofs_nodal = facet_dofs.nodal[f'u^{dof_idx+1}']
                        dofs_facet = facet_dofs.facet.get(f'u^{dof_idx+1}', np.array([], dtype=int))
                        dofs = np.concatenate([dofs_nodal, dofs_facet])
                        fixed_dofs = np.union1d(fixed_dofs, dofs)
                        if disp_vals is not None:
                            u_prescribed[dofs] = float(disp_vals[dof_idx])

                    # Debug Viz (sampled from vertex DOFs)
                    try:
                        nodal_dofs = basis.nodal_dofs
                        vertex_dof_set = set(facet_dofs.nodal['u^1'].tolist())
                        fixed_nodes_approx = [n for n in range(mesh.p.shape[1])
                                              if int(nodal_dofs[0, n]) in vertex_dof_set]
                        step = max(1, len(fixed_nodes_approx) // 50)
                        for i in range(0, len(fixed_nodes_approx), step):
                            idx = fixed_nodes_approx[i]
                            debug_constraints.append({'pos': mesh.p[:, idx].tolist()})
                    except Exception:
                        pass
                        
                elif 'condition' in c and c['condition']:
                    # Legacy string condition
                    cond_str = c['condition']
                    try:
                        def constraint_func(p):
                            x, y, z = p
                            return eval(cond_str, {'x': x, 'y': y, 'z': z, 'np': np})
                        
                        facet_dofs = basis.get_dofs(constraint_func)
                        for dof_idx in fixed_dof_indices:
                             fixed_dofs = np.union1d(fixed_dofs, facet_dofs.nodal[f'u^{dof_idx+1}'])
                        
                        # Debug Viz via mask
                        mask = constraint_func(mesh.p)
                        if isinstance(mask, np.ndarray):
                            fixed_nodes_indices = np.where(mask)[0]
                            step = max(1, len(fixed_nodes_indices) // 50)
                            for i in range(0, len(fixed_nodes_indices), step):
                                idx = fixed_nodes_indices[i]
                                debug_constraints.append({'pos': mesh.p[:, idx].tolist()})
                    except Exception as e:
                        logger.warning(f"TopOpt: Constraint condition failed: {e}")

            fixed_dofs = fixed_dofs.astype(int)

        except Exception as e:
            logger.warning(f"TopOpt: Constraint processing error: {e}")
            fixed_dofs = np.array([], dtype=int)


        # --- PROCESS LOADS ---
        f = np.zeros(basis.N)
        
        try:
            for l in load_list:
                if not l: continue
                
                vector = l.get('vector', [0, 0, 0])
                load_type = l.get('type', 'force')
                pressure = l.get('pressure', 0.0) # For pressure loads
                
                # Handle both 'geometries' (list, from LoadNode) and legacy 'geometry' (single)
                geoms = l.get('geometries', None)
                if geoms is None and 'geometry' in l:
                    geoms = [l['geometry']]
                geoms = [g for g in (geoms or []) if g is not None]
                
                if geoms:
                    # ----------------------------------------------------------------
                    # PRIMARY PATH: area-weighted FacetBasis traction integration.
                    # ----------------------------------------------------------------
                    f_traction = _assemble_traction_force(mesh, basis, geoms, vector)
                    if f_traction is not None:
                        f += f_traction
                    else:
                        # Do NOT fall back to equal nodal distribution — on unstructured meshes
                        # equal nodal weighting concentrates force on dense mesh regions and
                        # produces artificial stress spikes.  Skip load and surface the error.
                        logger.error(
                            f"TopOpt: FacetBasis traction assembly failed for load {vector}. "
                            "No load applied — check that the selected geometry face "
                            "coincides with the mesh boundary."
                        )

                    # Debug Viz (uses bounding box centre — lightweight)
                    bbox_list_dbg = [g.BoundingBox() for g in geoms]
                    cx = sum(b.xmin + b.xmax for b in bbox_list_dbg) / (2 * len(bbox_list_dbg))
                    cy = sum(b.ymin + b.ymax for b in bbox_list_dbg) / (2 * len(bbox_list_dbg))
                    cz = sum(b.zmin + b.zmax for b in bbox_list_dbg) / (2 * len(bbox_list_dbg))
                    debug_loads.append({'start': [cx, cy, cz], 'vector': list(vector)})
                        
                elif 'condition' in l and l['condition']:
                    # Legacy string load
                    load_cond = l['condition']
                    try:
                        matching_nodes_indices = np.where(eval(load_cond, {'x': x, 'y': y, 'z': z, 'np': np}))[0]
                        n_load_nodes = len(matching_nodes_indices)
                        if n_load_nodes > 0:
                            fx, fy, fz = [val / n_load_nodes for val in vector]
                            nodal_dofs = basis.nodal_dofs
                            for node_idx in matching_nodes_indices:
                                f[nodal_dofs[0, node_idx]] += fx
                                f[nodal_dofs[1, node_idx]] += fy
                                f[nodal_dofs[2, node_idx]] += fz
                            
                            step = max(1, n_load_nodes // 20)
                            for i in range(0, n_load_nodes, step):
                                idx = matching_nodes_indices[i]
                                debug_loads.append({
                                    'start': mesh.p[:, idx].tolist(),
                                    'vector': vector
                                })
                    except Exception as e:
                        logger.warning(f"TopOpt: Load condition failed: {e}")
                        
        except Exception as e:
            logger.warning(f"TopOpt: Load processing error: {e}")


        # Stiffness Form with Density Penalization
        @BilinearForm
        def stiffness(u, v, w):
            # penal and rho_min are captured from closure
            # Manual linear elasticity to avoid version issues
            def epsilon(w):
                return sym_grad(w)

            E = epsilon(u)
            D = epsilon(v)
            
            # sigma(u) : epsilon(v)
            # 2*mu*E:D + lam*tr(E)*tr(D)
            term1 = 2.0 * mu_val * ddot(E, D)
            term2 = lam_val * tr(E) * tr(D)
            
            # Add small epsilon to avoid singularity
            return (rho_min + w['rho'] ** penal) * (term1 + term2)
            
        # Element Compliance Functional for Sensitivity Analysis
        # --------------------------------------------------------
        # What we actually compute here is u_e^T k_e u_e per element, which equals
        # 2 × (strain energy per element) = element compliance contribution.
        # This is the correct quantity for SIMP sensitivity:
        #   dc/d(rho_e) = -p * rho_e^(p-1) * (u_e^T k_e u_e)
        # The name 'element_compliance' prevents the common confusion with
        # the factor-of-two difference between strain energy and compliance.
        @Functional
        def element_compliance(w):
            def epsilon(w):
                return sym_grad(w)

            u = w['u']
            E = epsilon(u)

            # = integral_e (2*mu*E:E + lam*tr(E)^2) dV
            term1 = 2.0 * mu_val * ddot(E, E)
            term2 = lam_val * tr(E) * tr(E)
            return term1 + term2

        # Element Volumes for OC
        @Functional
        def unit_one(w):
            return 1.0
        volumes = unit_one.elemental(basis0)
        total_vol = np.sum(volumes)
        target_vol = total_vol * vol_frac

        # Get element centroids for filtering
        centroids = mesh.p[:, mesh.t].mean(axis=1).T

        # Get filter and update scheme settings
        filter_type = self.get_property('filter_type')
        update_scheme = self.get_property('update_scheme')
        
        # Initialize MMA history variables
        m = 1  # One constraint (volume)
        n = len(densities)
        xold1 = densities.copy()
        xold2 = densities.copy()
        low = np.zeros(n)
        upp = np.zeros(n)
        beta_schedule = float(self.get_property('heaviside_beta'))
        
        # Filter precomputation
        H, H_sum = None, None
        if filter_radius > 0 and filter_type in ['density', 'sensitivity']:
            logger.info("TopOpt: Precomputing sparse filter matrix...")
            H, H_sum = build_filter_matrix(centroids, filter_radius)

        # NEW: Precompute Base Stiffness Matrix (Performance Optimization)
        # We pre-evaluate the element stiffness matrices k_e for unscaled density.
        logger.info("TopOpt: Precomputing base element stiffness matrices for fast assembly...")
        
        @BilinearForm
        def stiffness_base(u, v, w):
            E = sym_grad(u)
            D = sym_grad(v)
            term1 = 2.0 * mu_val * ddot(E, D)
            term2 = lam_val * tr(E) * tr(D)
            return term1 + term2

        # _assemble returns (indices, data, shape, bshape)
        # where data is flattened 'C' from (Nbfun, Nbfun, n_elem)
        res_K = stiffness_base._assemble(basis)
        I_indices, J_indices = res_K[0][0], res_K[0][1]
        K_base_data = res_K[1]
        Nbfun_sq = res_K[3][0] * res_K[3][1]
        
        # Optimization Loop
        logger.info(f"TopOpt: Optimization loop started. Type: {update_scheme}, Filter: {filter_type}, Max iter: {max_iter}")
        
        from scipy.sparse import coo_matrix
        
        for loop in range(max_iter):
            # Track consecutive convergence (require 3 consecutive low-change iterations)
            if loop == 0:
                consecutive_converged = 0
                prev_compliance = 0.0
                obj_plateau_count = 0

            # 1. Apply Density Filter (if enabled)
            densities_phys = densities
            if filter_type == 'density' and filter_radius > 0:
                densities_phys = density_filter_3d(densities, H, H_sum)
            
            # 1b. Apply Heaviside projection (if enabled)
            projection_type = self.get_property('projection')
            d_proj = np.ones_like(densities_phys)  # Default derivative
            
            if projection_type == 'Heaviside':
                beta = float(self.get_property('heaviside_beta'))
                eta = float(self.get_property('heaviside_eta'))
                continuation = self.get_property('continuation')
                
                # Continuation: gradually increase beta for stability
                if continuation:
                    # Start with beta=1, double each quartile of iterations
                    # e.g., iter 0-3: beta=1, iter 4-7: beta=2, etc.
                    beta_schedule = min(beta, 2 ** (4 * loop / max_iter))
                else:
                    beta_schedule = beta
                
                densities_phys, d_proj = heaviside_projection(densities_phys, beta_schedule, eta)
            
            # --- REAL-TIME VISUALIZATION CALLBACK ---
            if progress_callback:
                # Send physical densities (what actually matters)
                # We catch exceptions to prevent UI errors from crashing the solver
                try:
                    progress_callback(mesh, densities_phys, loop, max_iter)
                except Exception:
                    pass
            
            # 2. FE Analysis (FAST PATH)
            # Scale the precomputed element matrices
            density_penalty = rho_min + densities_phys ** penal
            # Tile penalty over the Nbfun^2 components (since n_elem is innermost dimension in C-flatten)
            V_data = K_base_data * np.tile(density_penalty, Nbfun_sq)
            
            # Build fast sparse global stiffness matrix
            K_coo = coo_matrix((V_data, (I_indices, J_indices)), shape=(basis.N, basis.N))
            K = K_coo.tocsr()
            
            # Solve
            try:
                with suppress_output():
                    u = solve(*condense(K, f, x=u_prescribed, D=fixed_dofs))
            except Exception as e:
                logger.error(f"Solver failed at iter {loop}: {e}")
                break

            # 3. Sensitivity Analysis
            # NOTE: the Functional is named 'element_compliance' (not 'strain_energy') to
            # reflect that term1+term2 = u_e^T k_e u_e = 2*strain_energy_e = element compliance.
            energies = element_compliance.elemental(basis, u=basis.interpolate(u))
            
            # dc/drho = -p * rho^(p-1) * energy  (elemental() already integrates over vol)
            dc = -penal * (densities_phys ** (penal - 1)) * energies

            # Apply Heaviside projection chain rule BEFORE the density filter.
            # d_proj = dH/d(rho_filtered); without this the gradient is incorrect
            # whenever Heaviside projection is active.
            if projection_type == 'Heaviside':
                dc = dc * d_proj

            # Apply Filter to Sensitivities
            if filter_type == 'density' and filter_radius > 0:
                # Chain rule for density filter
                dc = density_filter_chainrule(dc, H, H_sum)
            elif filter_type == 'sensitivity' and filter_radius > 0:
                # Heuristic sensitivity filter
                dc = sensitivity_filter(dc, H, H_sum, densities=densities_phys)

            # 4. Update Design Variables
            if update_scheme == 'MMA':
                # Objective: Compliance (minimize)
                # Constraint: Volume - Target <= 0
                
                # Compliance value — energies is already integrated over element vol,
                # so do NOT multiply by volumes again.
                c = np.sum(densities_phys**penal * energies)

                # Volume constraint value (physical densities drive the actual volume)
                current_vol = np.sum(densities_phys * volumes)
                vol_constraint = current_vol - target_vol

                # Volume constraint gradient dV/dx, accounting for both Heaviside and
                # density-filter chain rules in the correct order.
                # dV/d(phys_e) = volumes[e]
                # dV/d(filt_e) = volumes[e] * d_proj[e]   (Heaviside chain rule)
                # dV/d(x_j)   = sum_e dV/d(filt_e) * d(filt_e)/d(x_j)  (filter chain rule)
                dvol_base = volumes * d_proj if projection_type == 'Heaviside' else volumes
                dvol = dvol_base
                if filter_type == 'density' and filter_radius > 0:
                    dvol = density_filter_chainrule(dvol_base, H, H_sum)
                # Adaptive move limit: wider in early iterations for faster convergence
                ramp_iters = 20
                move_factor = 1.0 + max(0.0, 1.5 * (1.0 - loop / ramp_iters))
                effective_move = min(0.5, move * move_factor)
                
                # Call MMA (xmin=rho_min to prevent singularity)
                rho_new, low, upp = mma_update(n, loop, densities, rho_min, 1, xold1, xold2, 
                                             c, dc, vol_constraint, dvol, low, upp, move=effective_move)
                
                # Update history
                xold2 = xold1.copy()
                xold1 = densities.copy()
                
            else: # OC Update (Legacy)
                # Sensitivity per unit volume (positive for OC formula usually involves -dc)
                sensitivity = -dc # Make positive for OC
                
                l1, l2 = 0.0, 1e9
                rho_new = densities # Default
                
                while (l2 - l1) / (l1 + l2 + 1e-10) > 1e-3:
                    lmid = 0.5 * (l2 + l1)
                    term = np.sqrt(sensitivity / lmid)
                    rho_new = densities * term
                    rho_new = np.clip(rho_new, np.maximum(0, densities - move), np.minimum(1, densities + move))
                    rho_new = np.clip(rho_new, rho_min, 1.0)
                    
                    # For OC, we check volume on the NEW density
                    # If density filter is active, we should strictly filter rho_new to check volume...
                    # But OC is heuristic anyway. Let's just check raw volume conservation for OC.
                    if np.sum(rho_new * volumes) - target_vol > 0:
                        l1 = lmid
                    else:
                        l2 = lmid

            change = np.max(np.abs(rho_new - densities))
            densities = rho_new
            
            # Apply Symmetry (Projection)
            # Average mirrored pairs to preserve volume fraction and prevent
            # progressive density loss from min-projection ratcheting.
            if sym_map_x is not None:
                src_indices, target_indices = sym_map_x
                avg_densities = 0.5 * (densities[src_indices] + densities[target_indices])
                densities[src_indices] = avg_densities
                densities[target_indices] = avg_densities
            if sym_map_y is not None:
                src_indices, target_indices = sym_map_y
                avg_densities = 0.5 * (densities[src_indices] + densities[target_indices])
                densities[src_indices] = avg_densities
                densities[target_indices] = avg_densities
            if sym_map_z is not None:
                src_indices, target_indices = sym_map_z
                avg_densities = 0.5 * (densities[src_indices] + densities[target_indices])
                densities[src_indices] = avg_densities
                densities[target_indices] = avg_densities

            # Compute compliance for logging
            # energies already contains the integrated element volume (skfem
            # .elemental() folds the Jacobian into the quadrature), so we must
            # NOT multiply by `volumes` again — that would be double-counting.
            obj_val = np.sum(densities_phys**penal * energies) if update_scheme != 'MMA' else c
            logger.info(f"Iter {loop}: Change {change:.4f}, Vol {np.sum(densities*volumes)/total_vol:.2f}, Compliance {obj_val:.4e}")
            
            # Check convergence: density change OR objective plateau
            if change < conv_tol:
                consecutive_converged += 1
                if consecutive_converged >= 3:
                    logger.info(f"TopOpt: Converged after {loop + 1} iterations (3 consecutive low-change iterations)")
                    break
            else:
                consecutive_converged = 0
            
            # Objective-based convergence: stop if compliance barely changes
            if loop > 0 and prev_compliance > 0:
                rel_change = abs(obj_val - prev_compliance) / (abs(prev_compliance) + 1e-10)
                if rel_change < 1e-3:
                    obj_plateau_count += 1
                    if obj_plateau_count >= 5:
                        logger.info(f"TopOpt: Converged after {loop + 1} iterations (objective plateau, rel_change < 0.1%)")
                        break
                else:
                    obj_plateau_count = 0
            prev_compliance = obj_val
        
        # Calculate final stress for visualization if requested
        stress = None
        densities_final = densities.copy()
        try:
            # Re-solve with final density (FAST PATH)
            densities_phys = densities
            if filter_type == 'density' and filter_radius > 0:
                densities_phys = density_filter_3d(densities, H, H_sum)
            if projection_type == 'Heaviside':
                densities_phys, _ = heaviside_projection(densities_phys, beta_schedule, eta)
            densities_final = densities_phys.copy()

            density_penalty = rho_min + densities_phys ** penal
            V_data = K_base_data * np.tile(density_penalty, Nbfun_sq)
            K_coo = coo_matrix((V_data, (I_indices, J_indices)), shape=(basis.N, basis.N))
            K = K_coo.tocsr()
            
            with suppress_output():
                u = solve(*condense(K, f, x=u_prescribed, D=fixed_dofs))
            
            # Project stress
            basis_p1 = Basis(mesh, ElementTetP1(), quadrature=basis.quadrature)
            
            @LinearForm
            def von_mises(v, w):
                def epsilon(w):
                    return sym_grad(w)
                E = epsilon(w['u'])
                E11, E12, E13 = E[0,0], E[0,1], E[0,2]
                E21, E22, E23 = E[1,0], E[1,1], E[1,2]
                E31, E32, E33 = E[2,0], E[2,1], E[2,2]
                trE = E11 + E22 + E33
                S11 = 2*mu_val*E11 + lam_val*trE
                S22 = 2*mu_val*E22 + lam_val*trE
                S33 = 2*mu_val*E33 + lam_val*trE
                S12 = 2*mu_val*E12
                S23 = 2*mu_val*E23
                S13 = 2*mu_val*E13
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
        except Exception as e:
            logger.warning(f"TopOpt stress calc failed: {e}")

        # Shape recovery for manufacturable geometry
        # IMPORTANT: Use physical (filtered + projected) densities, not raw design variables.
        # Raw design variables can have many intermediate values that the filter resolves
        # into clean 0/1; using them directly would kill legitimate structural members.
        recovered_shape = None
        if shape_recovery_enabled:
            verts, faces = shape_recovery(mesh, densities_final, self.get_property('density_cutoff'), 
                                          smoothing_iterations=int(smoothing_iter),
                                          resolution=int(recovery_res))
            if verts is not None and faces is not None:
                recovered_shape = {'vertices': verts, 'faces': faces}
        
        logger.info(f"TopOpt: Optimization complete. Final physical VolFrac: {np.mean(densities_final):.3f}")
        return {
            'mesh': mesh,
            'density': densities_final,
            'design_density': densities,
            'stress': stress,
            'recovered_shape': recovered_shape,
            'type': 'topopt',
            'visualization_mode': self.get_property('visualization'),
            # Pass cutoff so the viewer can actually remove low-density material
            'density_cutoff': self.get_property('density_cutoff'),
            'debug_loads': debug_loads,
            'debug_constraints': debug_constraints
        }


