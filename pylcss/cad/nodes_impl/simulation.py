# Copyright (c) 2025 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

import numpy as np
from pylcss.cad.core.base_node import CadQueryNode
import skfem
from skfem import *
from skfem.helpers import sym_grad, ddot, trace
import os
import tempfile
from scipy.spatial import cKDTree
import logging
import sys
import contextlib

try:
    from simpleeval import simple_eval
except ImportError:
    simple_eval = None  # Fallback if not installed

from pylcss.config import simulation_config

logger = logging.getLogger(__name__)

@contextlib.contextmanager
def suppress_output():
    """Context manager to suppress stdout and stderr."""
    if not simulation_config.SUPPRESS_EXTERNAL_LIBRARY_OUTPUT:
        yield
        return
        
    # Save the current stdout and stderr
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    
    # Redirect to null
    sys.stdout = open(os.devnull, 'w')
    sys.stderr = open(os.devnull, 'w')
    
    try:
        yield
    finally:
        # Restore stdout and stderr
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = old_stdout
        sys.stderr = old_stderr

# ADD: Netgen imports
try:
    from netgen.occ import OCCGeometry
    import netgen.meshing as ngmeshing
except ImportError:
    OCCGeometry = None

# Alias trace to tr if needed, or just use trace
tr = trace

def sensitivity_filter(sensitivities, centers, r_min):
    """
    Apply sensitivity filtering using a KD-Tree for O(N log N) performance.
    """
    # Build spatial tree (very fast)
    tree = cKDTree(centers)
    
    # Query all neighbors within r_min for all points at once
    # returns a sparse list of neighbors
    neighbors_list = tree.query_ball_point(centers, r_min)
    
    filtered = np.zeros_like(sensitivities)
    
    for i, neighbors in enumerate(neighbors_list):
        if not neighbors:
            filtered[i] = sensitivities[i]
            continue
            
        # Get indices and coordinates of neighbors
        indices = np.array(neighbors)
        neighbor_centers = centers[indices]
        
        # Calculate distances
        dist = np.linalg.norm(neighbor_centers - centers[i], axis=1)
        
        # Linear weight decay: max(0, r_min - d)
        weights = np.maximum(0, r_min - dist)
        
        # Avoid division by zero
        weight_sum = np.sum(weights)
        if weight_sum > 1e-10:
            filtered[i] = np.sum(weights * sensitivities[indices]) / weight_sum
        else:
            filtered[i] = sensitivities[i]
            
    return filtered

def density_filter_3d(densities, centroids, r_min):
    """
    Apply density filter using spatial tree for 3D elements.
    Returns both filtered densities and the filter weights matrix for chain rule.
    """
    tree = cKDTree(centroids)
    neighbors_list = tree.query_ball_point(centroids, r_min)
    
    n_elem = len(densities)
    filtered = np.zeros(n_elem)
    
    # Store weight sums for chain rule
    weight_sums = np.zeros(n_elem)
    
    for i, neighbors in enumerate(neighbors_list):
        if not neighbors:
            filtered[i] = densities[i]
            weight_sums[i] = 1.0
            continue
        
        indices = np.array(neighbors)
        neighbor_centroids = centroids[indices]
        dist = np.linalg.norm(neighbor_centroids - centroids[i], axis=1)
        weights = np.maximum(0, r_min - dist)
        weight_sum = np.sum(weights)
        
        if weight_sum > 1e-10:
            filtered[i] = np.sum(weights * densities[indices]) / weight_sum
            weight_sums[i] = weight_sum
        else:
            filtered[i] = densities[i]
            weight_sums[i] = 1.0
    
    return filtered, weight_sums

def density_filter_chainrule(dc, densities, filtered_densities, centroids, r_min, weight_sums):
    """
    Apply chain rule for density filter sensitivities.
    dc_tilde_j = sum_i (H_ij / sum_k H_ik) * dc_i / rho_j
    """
    tree = cKDTree(centroids)
    neighbors_list = tree.query_ball_point(centroids, r_min)
    
    n_elem = len(densities)
    dc_filtered = np.zeros(n_elem)
    
    for j, neighbors in enumerate(neighbors_list):
        if not neighbors:
            dc_filtered[j] = dc[j]
            continue
        
        # For each element j, sum contributions from elements i that include j in their filter
        total = 0.0
        indices = np.array(neighbors)
        for i in indices:
            # Element i includes j in its filter neighborhood
            dist = np.linalg.norm(centroids[i] - centroids[j])
            H_ij = max(0, r_min - dist)
            if weight_sums[i] > 1e-10 and densities[j] > 1e-10:
                total += H_ij / weight_sums[i] * dc[i] * densities[j]
        
        if densities[j] > 1e-10:
            dc_filtered[j] = total / densities[j]
        else:
            dc_filtered[j] = dc[j]
    
    return dc_filtered

def mma_update(n, itr, xval, xmin, xmax, xold1, xold2, f0val, df0dx, 
               fval, dfdx, low, upp, move=0.2):
    """
    Simplified MMA update for topology optimization with single volume constraint.
    Adapted from topopt-Documentation/src_Compliance/topopt.py
    
    Parameters:
    -----------
    n : int - number of design variables
    itr : int - current iteration
    xval : array - current design variables
    xmin, xmax : arrays - bounds
    xold1, xold2 : arrays - previous designs
    f0val : float - objective value
    df0dx : array - objective gradient
    fval : float - constraint value (volume - target)
    dfdx : array - constraint gradient
    low, upp : arrays - MMA asymptotes
    move : float - move limit
    
    Returns:
    --------
    xnew, low, upp : updated design and asymptotes
    """
    asyinit = 0.5
    asyincr = 1.2
    asydecr = 0.7
    albefa = 0.1
    
    eeen = np.ones(n)
    
    # Calculate asymptotes
    if itr <= 2:
        low = xval - asyinit * (xmax - xmin)
        upp = xval + asyinit * (xmax - xmin)
    else:
        # Check oscillation
        zzz = (xval - xold1) * (xold1 - xold2)
        factor = np.ones(n)
        factor[zzz > 0] = asyincr
        factor[zzz < 0] = asydecr
        
        low = xval - factor * (xold1 - low)
        upp = xval + factor * (upp - xold1)
        
        # Bounds on asymptotes
        lowmin = xval - 10 * (xmax - xmin)
        lowmax = xval - 0.01 * (xmax - xmin)
        uppmin = xval + 0.01 * (xmax - xmin)
        uppmax = xval + 10 * (xmax - xmin)
        
        low = np.maximum(low, lowmin)
        low = np.minimum(low, lowmax)
        upp = np.minimum(upp, uppmax)
        upp = np.maximum(upp, uppmin)
    
    # Bounds alfa and beta
    alfa = np.maximum(low + albefa * (xval - low), xval - move * (xmax - xmin))
    alfa = np.maximum(alfa, xmin)
    
    beta = np.minimum(upp - albefa * (upp - xval), xval + move * (xmax - xmin))
    beta = np.minimum(beta, xmax)
    
    # MMA approximation
    ux1 = upp - xval
    xl1 = xval - low
    ux2 = ux1 * ux1
    xl2 = xl1 * xl1
    
    # p0, q0 for objective
    p0 = np.maximum(df0dx, 0) * ux2
    q0 = np.maximum(-df0dx, 0) * xl2
    
    # P, Q for constraint
    P = np.maximum(dfdx, 0) * ux2
    Q = np.maximum(-dfdx, 0) * xl2
    
    # Solve subproblem using bisection on Lagrange multiplier
    l1, l2 = 0.0, 1e9
    
    for _ in range(100):
        lmid = 0.5 * (l1 + l2)
        
        # Compute xnew from MMA subproblem
        plam = p0 + lmid * P
        qlam = q0 + lmid * Q
        
        # Optimal x from KKT conditions
        sqrt_term = np.sqrt(plam / (qlam + 1e-10))
        xnew = (low * sqrt_term + upp) / (1 + sqrt_term)
        
        # Clamp to bounds
        xnew = np.maximum(xnew, alfa)
        xnew = np.minimum(xnew, beta)
        
        # Check volume constraint
        vol_constraint = np.sum(dfdx * xnew) + fval
        
        if vol_constraint > 0:
            l1 = lmid
        else:
            l2 = lmid
        
        if (l2 - l1) / (l1 + l2 + 1e-10) < 1e-4:
            break
    
    return xnew, low, upp

def shape_recovery(mesh, densities, cutoff, smoothing_iterations=3):
    """Recover manufacturable geometry from density field using isosurface extraction."""
    try:
        from skimage import measure
        import scipy.ndimage as ndi

        # Get element centroids
        centroids = mesh.p[:, mesh.t].mean(axis=1)

        # Create 3D grid for marching cubes
        x_min, x_max = centroids[0].min(), centroids[0].max()
        y_min, y_max = centroids[1].min(), centroids[1].max()
        z_min, z_max = centroids[2].min(), centroids[2].max()

        # Create regular grid
        nx, ny, nz = 50, 50, 50
        x = np.linspace(x_min, x_max, nx)
        y = np.linspace(y_min, y_max, ny)
        z = np.linspace(z_min, z_max, nz)

        # Interpolate densities onto grid
        from scipy.interpolate import griddata
        grid_x, grid_y, grid_z = np.meshgrid(x, y, z, indexing='ij')
        grid_points = np.column_stack((grid_x.ravel(), grid_y.ravel(), grid_z.ravel()))

        grid_densities = griddata(centroids.T, densities, grid_points,
                                method='linear', fill_value=0).reshape((nx, ny, nz))

        # Smooth the density field
        for _ in range(smoothing_iterations):
            grid_densities = ndi.gaussian_filter(grid_densities, sigma=1)

        # Sanitize cutoff to be within data range to avoid crash
        d_min, d_max = grid_densities.min(), grid_densities.max()
        effective_cutoff = cutoff
        if effective_cutoff < d_min:
            effective_cutoff = d_min + 1e-5
        elif effective_cutoff > d_max:
            effective_cutoff = d_max - 1e-5
            
        # Extract isosurface if range permits
        if d_max > d_min:
            verts, faces, _, _ = measure.marching_cubes(grid_densities, level=effective_cutoff)
        else:
            return None, None

        # Scale back to original coordinates
        verts[:, 0] = verts[:, 0] * (x_max - x_min) / (nx - 1) + x_min
        verts[:, 1] = verts[:, 1] * (y_max - y_min) / (ny - 1) + y_min
        verts[:, 2] = verts[:, 2] * (z_max - z_min) / (nz - 1) + z_min

        # NEW: Filter disconnected components - keep only the largest connected component
        if len(verts) > 0 and len(faces) > 0:
            try:
                from scipy.sparse import lil_matrix, csr_matrix
                from scipy.sparse.csgraph import connected_components
                
                # Build face adjacency matrix
                n_faces = len(faces)
                adj_matrix = lil_matrix((n_faces, n_faces))  # Use LIL for efficient construction
                
                # Create vertex-to-faces mapping
                vertex_to_faces = {}
                for i, face in enumerate(faces):
                    for v in face:
                        if v not in vertex_to_faces:
                            vertex_to_faces[v] = []
                        vertex_to_faces[v].append(i)
                
                # Connect faces that share vertices
                for face_list in vertex_to_faces.values():
                    if len(face_list) > 1:
                        for i in range(len(face_list)):
                            for j in range(i+1, len(face_list)):
                                adj_matrix[face_list[i], face_list[j]] = 1
                                adj_matrix[face_list[j], face_list[i]] = 1
                
                # Convert to CSR for connected_components (optional but good practice)
                adj_matrix = adj_matrix.tocsr()
                
                # Find connected components
                n_components, labels = connected_components(adj_matrix, directed=False)
                
                if n_components > 1:
                    # Calculate component sizes
                    component_sizes = np.bincount(labels)
                    
                    # Keep only the largest component
                    largest_component = np.argmax(component_sizes)
                    keep_faces = labels == largest_component
                    
                    # Filter faces
                    faces_filtered = faces[keep_faces]
                    
                    # Create vertex mapping for filtered mesh
                    unique_verts = np.unique(faces_filtered)
                    vert_map = {old_idx: new_idx for new_idx, old_idx in enumerate(unique_verts)}
                    
                    # Remap face indices
                    faces_remapped = np.array([[vert_map[v] for v in face] for face in faces_filtered])
                    verts_filtered = verts[unique_verts]
                    
                    verts, faces = verts_filtered, faces_remapped
                    
            except Exception as e:
                print(f"Component filtering failed: {e}")
                # Continue with unfiltered mesh

        return verts, faces

    except ImportError:
        print("Shape recovery requires scikit-image. Install with: pip install scikit-image")
        return None, None
    except Exception as e:
        print(f"Shape recovery failed: {e}")
        return None, None

def lam_lame(E, nu):
    """Convert Young's modulus and Poisson's ratio to Lame parameters."""
    return E * nu / ((1 + nu) * (1 - 2 * nu)), E / (2 * (1 + nu))

class MaterialNode(CadQueryNode):
    """Defines material properties."""
    __identifier__ = 'com.cad.sim.material'
    NODE_NAME = 'Material'

    def __init__(self):
        super().__init__()
        self.add_output('material', color=(200, 200, 200))
        
        # Add Inputs for parametric material properties
        self.add_input('youngs_modulus', color=(180, 180, 0))
        self.add_input('poissons_ratio', color=(180, 180, 0))
        self.add_input('density', color=(180, 180, 0))
        
        # Keep properties as defaults
        self.create_property('youngs_modulus', 210000.0, widget_type='float') # MPa
        self.create_property('poissons_ratio', 0.3, widget_type='float')
        self.create_property('density', 7.85e-9, widget_type='float') # tonne/mm^3

    def run(self):
        # Resolve inputs with fallback to properties
        E = self.get_input_value('youngs_modulus', 'youngs_modulus')
        nu = self.get_input_value('poissons_ratio', 'poissons_ratio')
        rho = self.get_input_value('density', 'density')
        
        return {
            'E': float(E),
            'nu': float(nu),
            'rho': float(rho)
        }

class MeshNode(CadQueryNode):
    """Generates a finite element mesh from a shape using Netgen."""
    __identifier__ = 'com.cad.sim.mesh'
    NODE_NAME = 'Generate Mesh (Netgen)'

    def __init__(self):
        super().__init__()
        self.add_input('shape', color=(100, 255, 100))
        self.add_input('element_size', color=(180, 180, 0))
        # NEW: Local refinement inputs
        self.add_input('refinement_faces', color=(255, 100, 100))  # List of faces for refinement
        self.add_input('refinement_size', color=(255, 100, 100))   # Smaller element size for refinement
        self.add_output('mesh', color=(200, 100, 200))
        self.create_property('element_size', 2.0, widget_type='float')
        self.create_property('refinement_size', 0.5, widget_type='float')  # Finer mesh for critical areas

    def run(self):
        if OCCGeometry is None:
            print("Error: 'netgen-occ' is not installed.")
            return None

        shape = self.get_input_shape('shape')
        # Resolve element size input with fallback to property
        size = self.get_input_value('element_size', 'element_size')
        size = float(size)
        
        # NEW: Get refinement parameters
        refinement_faces = self.get_input_value('refinement_faces', None)
        refinement_size = self.get_input_value('refinement_size', 'refinement_size')
        refinement_size = float(refinement_size)
        
        if not shape:
            return None

        # Handle assemblies by converting to compound
        if hasattr(shape, 'toCompound'):
            try:
                shape = shape.toCompound()
            except Exception as e:
                print(f"MeshNode: Error converting assembly to compound: {e}")
                return None

        # Optimized temporary file handling for performance
        # Try to use RAM disk if available (significant speedup for optimization loops)
        temp_base = None
        try:
            # Check for common RAM disk locations
            ram_disk_paths = ['R:\\', 'Z:\\', '/tmp/', '/dev/shm/']
            for path in ram_disk_paths:
                if os.path.exists(path) and os.access(path, os.W_OK):
                    temp_base = path
                    break
            
            # Fallback to system temp directory
            if temp_base is None:
                temp_base = tempfile.gettempdir()
            
            # Create temporary files in optimized location
            with tempfile.NamedTemporaryFile(suffix=".step", dir=temp_base, delete=False) as step_file:
                step_path = step_file.name
            
            msh_path = step_path.replace(".step", ".msh")
            
            try:
                # 1. Export CadQuery shape to STEP
                if hasattr(shape, 'val'):
                    shape.val().exportStep(step_path)
                else:
                    shape.exportStep(step_path)
                
                # 2. Load Geometry with Netgen and generate mesh (suppress verbose output)
                with suppress_output():
                    geo = OCCGeometry(step_path)
                    
                    # NEW: Apply local mesh refinement if specified
                    if refinement_faces is not None:
                        try:
                            # refinement_faces should be a list of face geometries
                            face_list = refinement_faces.vals() if hasattr(refinement_faces, 'vals') else [refinement_faces]
                            
                            for face in face_list:
                                if hasattr(face, 'hashCode'):
                                    # Set finer mesh size on specific faces
                                    geo.SetFaceMaxH(face.hashCode(), refinement_size)
                        except Exception as e:
                            print(f"MeshNode: Local refinement setup failed: {e}")
                    
                    # 3. Generate Mesh
                    # maxh controls the global element size
                    ng_mesh = geo.GenerateMesh(maxh=size)
                    
                    # 4. Export to Gmsh format (Version 2 is most compatible with skfem/meshio)
                    # Netgen's Export function takes the filename and the format string
                    ng_mesh.Export(msh_path, "Gmsh2 Format")
                
                # 5. Load into skfem
                mesh = Mesh.load(msh_path)
                
            except Exception as e:
                print(f"Meshing failed: {e}")
                return None
                
            finally:
                # Clean up temporary files immediately
                try:
                    if os.path.exists(step_path):
                        os.remove(step_path)
                    if os.path.exists(msh_path):
                        os.remove(msh_path)
                except OSError:
                    pass  # Ignore cleanup errors
        
        except Exception as e:
            print(f"MeshNode: Temporary file creation failed: {e}")
            return None
        
        return mesh

class ConstraintNode(CadQueryNode):
    """Applies a fixed support to a specific geometric face."""
    __identifier__ = 'com.cad.sim.constraint'
    NODE_NAME = 'FEA Fixed Support (Face)'

    def __init__(self):
        super().__init__()
        self.add_input('mesh', color=(200, 100, 200))
        # NEW: Input for the specific face geometry to fix
        self.add_input('target_face', color=(100, 200, 255))
        self.add_output('constraints', color=(255, 100, 100))
        # Keep string condition as fallback for backward compatibility
        self.create_property('condition', '', widget_type='text')

    def run(self):
        mesh = self.get_input_value('mesh', None)
        target_wp = self.get_input_value('target_face', None)  # This is a Workplane object
        fallback_condition = self.get_property('condition')
        
        # Debug logging
        print(f"[ConstraintNode DEBUG] mesh={mesh is not None}, target_wp={target_wp}, fallback='{fallback_condition}'")

        if mesh is None:
            return None

        # If no face input provided, use fallback string condition
        if target_wp is None:
            if not fallback_condition:
                print("[ConstraintNode DEBUG] No target_face and no fallback condition - returning None")
                return None
            return {'type': 'fixed', 'condition': fallback_condition}

        # Extract face from SelectFaceNode dict format {'workplane': ..., 'face': ...}
        try:
            if isinstance(target_wp, dict) and 'face' in target_wp:
                face_obj = target_wp['face']
            else:
                # Fallback: try to get vals from workplane (legacy support)
                face_objs = target_wp.vals() if hasattr(target_wp, 'vals') else []
                if not face_objs:
                    self.set_error("No faces found in target face input")
                    return None
                face_obj = face_objs[0]
            
            print(f"[ConstraintNode DEBUG] Extracted face_obj: {type(face_obj).__name__}")
            return {'type': 'fixed', 'geometry': face_obj}

        except Exception as e:
            self.set_error(f"Constraint setup failed: {e}")
            return None

class LoadNode(CadQueryNode):
    """Applies a load to a specific geometric face."""
    __identifier__ = 'com.cad.sim.load'
    NODE_NAME = 'FEA Load (Face)'

    def __init__(self):
        super().__init__()
        self.add_input('mesh', color=(200, 100, 200))
        # NEW: Input for the specific face geometry to load
        self.add_input('target_face', color=(100, 200, 255))

        # Add inputs for parametric force components
        self.add_input('force_x', color=(255, 255, 0))
        self.add_input('force_y', color=(255, 255, 0))
        self.add_input('force_z', color=(255, 255, 0))

        self.add_output('loads', color=(255, 255, 0))
        # Keep string condition as fallback for backward compatibility
        self.create_property('condition', '', widget_type='text')
        self.create_property('force_x', 0.0, widget_type='float')
        self.create_property('force_y', -1000.0, widget_type='float')
        self.create_property('force_z', 0.0, widget_type='float')

    def run(self):
        mesh = self.get_input_value('mesh', None)
        target_wp = self.get_input_value('target_face', None)  # This is a Workplane object

        # Resolve force inputs with fallback to properties
        fx = self.get_input_value('force_x', 'force_x')
        fy = self.get_input_value('force_y', 'force_y')
        fz = self.get_input_value('force_z', 'force_z')

        fallback_condition = self.get_property('condition')
        
        # Debug logging
        print(f"[LoadNode DEBUG] mesh={mesh is not None}, target_wp={target_wp}, fallback='{fallback_condition}'")

        if mesh is None:
            return None

        # If no face input provided, use fallback string condition
        if target_wp is None:
            if not fallback_condition:
                print("[LoadNode DEBUG] No target_face and no fallback condition - returning None")
                return None
            return {
                'type': 'force',
                'condition': fallback_condition,
                'vector': [float(fx), float(fy), float(fz)]
            }

        # Extract face from SelectFaceNode dict format {'workplane': ..., 'face': ...}
        try:
            if isinstance(target_wp, dict) and 'face' in target_wp:
                face_obj = target_wp['face']
            else:
                # Fallback: try to get vals from workplane (legacy support)
                face_objs = target_wp.vals() if hasattr(target_wp, 'vals') else []
                if not face_objs:
                    self.set_error("No faces found in target face input")
                    return None
                face_obj = face_objs[0]
            
            print(f"[LoadNode DEBUG] Extracted face_obj: {type(face_obj).__name__}")
            bb = face_obj.BoundingBox()
            tol = 1e-3

            # Create condition string that selects nodes within the face's bounding box
            cond_str = (f"(x >= {bb.xmin - tol}) & (x <= {bb.xmax + tol}) & "
                        f"(y >= {bb.ymin - tol}) & (y <= {bb.ymax + tol}) & "
                        f"(z >= {bb.zmin - tol}) & (z <= {bb.zmax + tol})")

            return {
                'type': 'force',
                'condition': cond_str,
                'vector': [float(fx), float(fy), float(fz)]
            }

        except Exception as e:
            self.set_error(f"Load setup failed: {e}")
            return None


class PressureLoadNode(CadQueryNode):
    """Applies a pressure load to a specific geometric face."""
    __identifier__ = 'com.cad.sim.pressure_load'
    NODE_NAME = 'FEA Pressure Load'

    def __init__(self):
        super().__init__()
        self.add_input('mesh', color=(200, 100, 200))
        # Input for the specific face geometry to apply pressure
        self.add_input('target_face', color=(100, 200, 255))
        # Pressure magnitude (positive = outward, negative = inward)
        self.add_input('pressure', color=(255, 255, 0))

        self.add_output('loads', color=(255, 255, 0))
        self.create_property('pressure', 1000000.0, widget_type='float')  # 1 MPa default

    def run(self):
        mesh = self.get_input_value('mesh', None)
        target_wp = self.get_input_value('target_face', None)
        pressure = self.get_input_value('pressure', 'pressure')

        if mesh is None or target_wp is None:
            return None

        try:
            # Extract the face geometry
            face_objs = target_wp.vals()
            if not face_objs:
                self.set_error("No faces found in target face input")
                return None

            face_obj = face_objs[0]  # Take the first face if multiple

            return {
                'type': 'pressure',
                'geometry': face_obj,
                'pressure': float(pressure)
            }

        except Exception as e:
            self.set_error(f"Pressure load setup failed: {e}")
            return None


class SolverNode(CadQueryNode):
    """Solves the FEA problem."""
    __identifier__ = 'com.cad.sim.solver'
    NODE_NAME = 'FEA Solver'

    def __init__(self):
        super().__init__()
        self.add_input('mesh', color=(200, 100, 200))
        self.add_input('material', color=(200, 200, 200))
        self.add_input('constraints', color=(255, 100, 100))
        self.add_input('loads', color=(255, 255, 0))
        self.add_output('results', color=(0, 255, 255))
        self.create_property('visualization', 'Von Mises Stress', widget_type='combo', items=['Von Mises Stress', 'Displacement'])

    def run(self):
        mesh = self.get_input_value('mesh', None)
        material = self.get_input_value('material', None)
        constraint = self.get_input_value('constraints', None)
        load = self.get_input_value('loads', None)
        
        if not (mesh and material and constraint and load):
            return None

        # 1. Define Element and Basis (Vector)
        # CHANGE: Use P2 (Quadratic) elements for accuracy (prevents locking)
        e = ElementVector(ElementTetP2())
        basis = Basis(mesh, e)

        # 2. Define Physics (Linear Elasticity)
        E_mat = material['E']
        nu_mat = material['nu']
        
        # Lame parameters
        lam_val, mu_val = lam_lame(E_mat, nu_mat)
        
        # Create constant fields for material properties
        # We need a scalar basis for parameters
        basis0 = basis.with_element(ElementTetP0())
        lam_field = basis0.zeros() + lam_val
        mu_field = basis0.zeros() + mu_val
        
        lam_interp = basis0.interpolate(lam_field)
        mu_interp = basis0.interpolate(mu_field)

        @BilinearForm
        def stiffness(u, v, w):
            def epsilon(w):
                return sym_grad(w)
            E = epsilon(u)
            D = epsilon(v)
            return 2.0 * w['mu'] * ddot(E, D) + w['lam'] * tr(E) * tr(D)

        # 3. Assemble Stiffness Matrix
        K = stiffness.assemble(basis, lam=lam_interp, mu=mu_interp)

        # 4. Apply Boundary Conditions
        x, y, z = mesh.p
        
        # Fixed Dofs
        try:
            if 'geometry' in constraint:
                # NEW: Geometry-based constraint selection (more accurate)
                face_shape = constraint['geometry']
                
                # Get node coordinates
                node_coords = np.column_stack((x, y, z))
                
                # For each node, check if it's close to the face
                # This is more accurate than bounding box approach
                fixed_nodes = []
                tolerance = 1e-3  # Distance tolerance
                
                # OPTIMIZATION: Calculate bounding box once outside the loop
                bb = face_shape.BoundingBox()
                xmin, xmax = bb.xmin - tolerance, bb.xmax + tolerance
                ymin, ymax = bb.ymin - tolerance, bb.ymax + tolerance
                zmin, zmax = bb.zmin - tolerance, bb.zmax + tolerance
                
                for i, coord in enumerate(node_coords):
                    # Quick bounding box check (pure Python - very fast)
                    if (xmin <= coord[0] <= xmax and
                        ymin <= coord[1] <= ymax and
                        zmin <= coord[2] <= zmax):
                        
                        # More precise check: distance to face
                        # This requires OCC geometric distance calculation
                        try:
                            # Use CadQuery's distance calculation
                            from cadquery import Vector
                            point = Vector(coord[0], coord[1], coord[2])
                            distance = face_shape.distanceTo(point)
                            if distance <= tolerance:
                                fixed_nodes.append(i)
                        except:
                            # Fallback: if distance calculation fails, use bounding box
                            fixed_nodes.append(i)
                
                # Convert node indices to DOF indices
                fixed_dofs = []
                nodal_dofs = basis.nodal_dofs  # (3, N_nodes)
                for node_idx in fixed_nodes:
                    for dof_idx in nodal_dofs[:, node_idx]:
                        fixed_dofs.append(dof_idx)
                
                fixed_dofs = np.array(fixed_dofs, dtype=int)
                
            elif 'condition' in constraint:
                # LEGACY: String-based constraint (fallback)
                cond_str = constraint['condition']
                
                # Secure evaluation using simpleeval
                if simple_eval is not None:
                    # Define allowed names and functions for safe evaluation
                    names = {'x': x, 'y': y, 'z': z}
                    functions = {'sin': np.sin, 'cos': np.cos, 'abs': np.abs, 'sqrt': np.sqrt}
                    
                    def constraint_func(p):
                        x_val, y_val, z_val = p
                        names.update({'x': x_val, 'y': y_val, 'z': z_val})
                        return simple_eval(cond_str, names=names, functions=functions)
                else:
                    # Fallback to restricted eval if simpleeval not available
                    def constraint_func(p):
                        x_val, y_val, z_val = p
                        return eval(cond_str, {'x': x_val, 'y': y_val, 'z': z_val, 'np': np})
                
                fixed_dofs = basis.get_dofs(constraint_func).all()
            else:
                fixed_dofs = np.array([], dtype=int)
                
        except Exception as e:
            print(f"Error processing constraint: {e}")
            fixed_dofs = np.array([], dtype=int)

        # 5. Apply Loads
        f = np.zeros(basis.N)
        
        try:
            if load['type'] == 'pressure':
                # NEW: Handle pressure loads
                face_shape = load['geometry']
                pressure = load['pressure']
                
                # Get node coordinates
                node_coords = np.column_stack((x, y, z))
                
                # Find nodes on the pressure face
                pressure_nodes = []
                tolerance = 1e-3
                
                # OPTIMIZATION: Pre-calculate bounding box for fallback
                bb = face_shape.BoundingBox()
                xmin, xmax = bb.xmin - tolerance, bb.xmax + tolerance
                ymin, ymax = bb.ymin - tolerance, bb.ymax + tolerance
                zmin, zmax = bb.zmin - tolerance, bb.zmax + tolerance
                
                for i, coord in enumerate(node_coords):
                    try:
                        from cadquery import Vector
                        point = Vector(coord[0], coord[1], coord[2])
                        distance = face_shape.distanceTo(point)
                        if distance <= tolerance:
                            pressure_nodes.append(i)
                    except:
                        # Fallback: use bounding box (pre-calculated - fast)
                        if (xmin <= coord[0] <= xmax and
                            ymin <= coord[1] <= ymax and
                            zmin <= coord[2] <= zmax):
                            pressure_nodes.append(i)
                
                if pressure_nodes:
                    # Calculate pressure force on each node
                    # For simplicity, distribute pressure evenly across face nodes
                    # In reality, this should integrate pressure over face area
                    n_pressure_nodes = len(pressure_nodes)
                    face_area = face_shape.Area()
                    
                    if face_area > 0:
                        # Total force = pressure * area
                        total_force = pressure * face_area
                        # Distribute to nodes (simplified - should use shape functions)
                        force_per_node = total_force / n_pressure_nodes
                        
                        # Get face normal for direction
                        try:
                            # Get normal vector of the face
                            normal = face_shape.normalAt()
                            fx_per_node = force_per_node * normal.x
                            fy_per_node = force_per_node * normal.y
                            fz_per_node = force_per_node * normal.z
                        except:
                            # Fallback: assume normal pressure (outward)
                            fx_per_node = fy_per_node = 0.0
                            fz_per_node = force_per_node
                        
                        nodal_dofs = basis.nodal_dofs
                        for node_idx in pressure_nodes:
                            dof_x = nodal_dofs[0, node_idx]
                            dof_y = nodal_dofs[1, node_idx]
                            dof_z = nodal_dofs[2, node_idx]
                            
                            f[dof_x] += fx_per_node
                            f[dof_y] += fy_per_node
                            f[dof_z] += fz_per_node
                            
            elif load['type'] == 'force':
                # LEGACY: Handle force loads
                load_vec = load['vector']
                load_cond = load['condition']
                
                # Secure evaluation using simpleeval
                if simple_eval is not None:
                    # Use numpy broadcasting with simple_eval - evaluate condition for all points
                    try:
                        # Create arrays for vectorized evaluation
                        x_arr = np.asarray(x)
                        y_arr = np.asarray(y) 
                        z_arr = np.asarray(z)
                        
                        # For simple conditions, try to evaluate vectorized
                        # This is a compromise - for complex conditions, fall back to loop
                        condition_results = []
                        for i in range(len(x_arr)):
                            names = {'x': float(x_arr[i]), 'y': float(y_arr[i]), 'z': float(z_arr[i])}
                            functions = {'sin': np.sin, 'cos': np.cos, 'abs': abs, 'sqrt': np.sqrt}
                            result = simple_eval(load_cond, names=names, functions=functions)
                            condition_results.append(bool(result))
                        
                        matching_nodes_indices = np.where(condition_results)[0]
                    except:
                        # Fallback to old eval if simpleeval fails
                        matching_nodes_indices = np.where(eval(load_cond, {'x': x, 'y': y, 'z': z, 'np': np}))[0]
                else:
                    # Fallback to restricted eval if simpleeval not available
                    matching_nodes_indices = np.where(eval(load_cond, {'x': x, 'y': y, 'z': z, 'np': np}))[0]
                
                n_load_nodes = len(matching_nodes_indices)
                
                if n_load_nodes > 0:
                    # FIXED: Simple uniform load distribution (area weighting was causing issues)
                    fx_total, fy_total, fz_total = load_vec
                    fx_per_node = fx_total / n_load_nodes
                    fy_per_node = fy_total / n_load_nodes
                    fz_per_node = fz_total / n_load_nodes
                    
                    nodal_dofs = basis.nodal_dofs
                    for node_idx in matching_nodes_indices:
                        dof_x = nodal_dofs[0, node_idx]
                        dof_y = nodal_dofs[1, node_idx]
                        dof_z = nodal_dofs[2, node_idx]
                        
                        f[dof_x] += fx_per_node
                        f[dof_y] += fy_per_node
                        f[dof_z] += fz_per_node
                        
        except Exception as e:
            print(f"Load application error: {e}")

        # 6. Solve
        try:
            with suppress_output():
                u = solve(*condense(K, f, D=fixed_dofs))
            logger.info(f"FEA Solve Complete. Max Displacement: {np.max(np.abs(u)):.6e}")
        except Exception as e:
            print(f"Solver failed: {e}")
            return None

        # 7. Calculate Von Mises Stress
        try:
            # Create a scalar P1 basis for stress visualization
            basis_p1 = basis.with_element(ElementTetP1())
            
            @LinearForm
            def von_mises(v, w):
                # Reconstruct stress tensor from strain
                def epsilon(w):
                    return sym_grad(w)
                
                E = epsilon(w['u'])
                mu = w['mu']
                lam = w['lam']
                
                # Components of Strain Tensor E
                E11, E12, E13 = E[0,0], E[0,1], E[0,2]
                E21, E22, E23 = E[1,0], E[1,1], E[1,2]
                E31, E32, E33 = E[2,0], E[2,1], E[2,2]
                
                trE = E11 + E22 + E33
                
                # Components of Stress Tensor S
                # S_ij = 2*mu*E_ij + lam*tr(E)*delta_ij
                S11 = 2*mu*E11 + lam*trE
                S22 = 2*mu*E22 + lam*trE
                S33 = 2*mu*E33 + lam*trE
                S12 = 2*mu*E12
                S23 = 2*mu*E23
                S13 = 2*mu*E13
                
                # Von Mises Stress
                # sqrt(0.5 * ((S11-S22)^2 + (S22-S33)^2 + (S33-S11)^2 + 6*(S12^2 + S23^2 + S13^2)))
                vm = np.sqrt(0.5 * ((S11-S22)**2 + (S22-S33)**2 + (S33-S11)**2 + 6*(S12**2 + S23**2 + S13**2)))
                return vm * v

            # Assemble Mass Matrix for P1 basis
            @BilinearForm
            def mass(u, v, w):
                return u * v
            
            M = mass.assemble(basis_p1)
            
            # Assemble Load Vector (Projected Stress)
            # Note: assemble expects keyword arguments for extra parameters, not a dict 'w'
            b = von_mises.assemble(basis_p1, 
                u=basis.interpolate(u),
                mu=basis_p1.zeros() + mu_val,
                lam=basis_p1.zeros() + lam_val
            )
            
            with suppress_output():
                stress = solve(M, b)
            
            # Ensure stress is positive (numerical errors might make it slightly negative)
            stress = np.abs(stress)
            logger.info(f"Stress Calc Complete. Max Stress: {np.max(stress):.6e}")
            
        except Exception as e:
            print(f"Stress calculation failed: {e}")
            stress = None

        return {
            'mesh': mesh,
            'displacement': u,
            'stress': stress,
            'type': 'fea',
            'visualization_mode': self.get_property('visualization')
        }

class TopologyOptimizationNode(CadQueryNode):
    """Performs Topology Optimization (SIMP Method) with sensitivity filtering and shape recovery."""
    __identifier__ = 'com.cad.sim.topopt'
    NODE_NAME = 'Topology Opt'

    def __init__(self):
        super().__init__()
        self.add_input('mesh', color=(200, 100, 200))
        self.add_input('material', color=(200, 200, 200))
        self.add_input('constraints', color=(255, 100, 100))
        self.add_input('loads', color=(255, 255, 0))
        self.add_output('optimized_mesh', color=(200, 100, 200))
        self.add_output('recovered_shape', color=(100, 255, 100))
        self.create_property('vol_frac', 0.4, widget_type='float')
        self.create_property('iterations', 15, widget_type='int')
        self.create_property('filter_radius', 1.5, widget_type='float')
        self.create_property('density_cutoff', 0.3, widget_type='float')
        self.create_property('shape_recovery', True, widget_type='bool')
        self.create_property('visualization', 'Density', widget_type='combo', items=['Density', 'Von Mises Stress'])
        # NEW: Symmetry properties
        self.create_property('symmetry_x', None, widget_type='float')  # None means no symmetry
        self.create_property('symmetry_y', None, widget_type='float')
        self.create_property('symmetry_z', None, widget_type='float')
        # NEW: Optimization parameters (previously hardcoded)
        self.create_property('penal', 3.0, widget_type='float')  # SIMP penalization exponent
        self.create_property('move_limit', 0.2, widget_type='float')  # Max density change per iteration
        self.create_property('min_density', 0.001, widget_type='float')  # Minimum element density
        self.create_property('convergence_tol', 0.01, widget_type='float')  # Convergence threshold
        self.create_property('recovery_resolution', 50, widget_type='int')  # Grid resolution for shape recovery
        self.create_property('smoothing_iterations', 3, widget_type='int')  # Gaussian smoothing passes
        # NEW: Filter type and update scheme selection
        self.create_property('filter_type', 'density', widget_type='combo', items=['sensitivity', 'density'])
        self.create_property('update_scheme', 'MMA', widget_type='combo', items=['MMA', 'OC'])


    def run(self):
        logger.info("TopOpt: Optimization started.")
        mesh = self.get_input_value('mesh', None)
        material = self.get_input_value('material', None)
        constraint = self.get_input_value('constraints', None)
        load = self.get_input_value('loads', None)
        
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
        
        if not (mesh and material and constraint and load):
            logger.warning(f"TopOpt: Missing inputs! Mesh:{mesh is not None}, Mat:{material is not None}, Cons:{constraint is not None}, Load:{load is not None}")
            return None

        logger.info("TopOpt: Inputs confirmed. Setting up basis...")

        # 1. Setup Basis (Vector for displacement, Scalar P0 for density)
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
                valid_mask = dists < 1e-3
                sym_map_x = (np.where(valid_mask)[0], indices[valid_mask])
            
            if sym_y is not None:
                # Create mirror points across y = sym_y plane
                mirror_pts = centroids.copy()
                mirror_pts[:, 1] = 2 * sym_y - mirror_pts[:, 1]
                # Query nearest neighbors
                dists, indices = tree.query(mirror_pts)
                valid_mask = dists < 1e-3
                sym_map_y = (np.where(valid_mask)[0], indices[valid_mask])
            
            if sym_z is not None:
                # Create mirror points across z = sym_z plane
                mirror_pts = centroids.copy()
                mirror_pts[:, 2] = 2 * sym_z - mirror_pts[:, 2]
                # Query nearest neighbors
                dists, indices = tree.query(mirror_pts)
                valid_mask = dists < 1e-3
                sym_map_z = (np.where(valid_mask)[0], indices[valid_mask])
        
        # Initialize density
        # Start with uniform density equal to volume fraction
        densities = np.ones(basis0.N) * vol_frac
        
        # Apply symmetry constraints to initial density (fast version)
        if sym_map_x is not None:
            src_indices, target_indices = sym_map_x
            min_densities = np.minimum(densities[src_indices], densities[target_indices])
            densities[src_indices] = min_densities
            densities[target_indices] = min_densities
        
        if sym_map_y is not None:
            src_indices, target_indices = sym_map_y
            min_densities = np.minimum(densities[src_indices], densities[target_indices])
            densities[src_indices] = min_densities
            densities[target_indices] = min_densities
        
        if sym_map_z is not None:
            src_indices, target_indices = sym_map_z
            min_densities = np.minimum(densities[src_indices], densities[target_indices])
            densities[src_indices] = min_densities
            densities[target_indices] = min_densities
        
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
        
        try:
            if 'geometry' in constraint:
                # New geometry-based constraint selection
                face_shape = constraint['geometry']
                node_coords = np.column_stack((x, y, z))
                
                # Get bounding box for initial filtering
                bb = face_shape.BoundingBox()
                tolerance = 1e-3
                xmin, xmax = bb.xmin - tolerance, bb.xmax + tolerance
                ymin, ymax = bb.ymin - tolerance, bb.ymax + tolerance
                zmin, zmax = bb.zmin - tolerance, bb.zmax + tolerance
                
                fixed_nodes = []
                for i, coord in enumerate(node_coords):
                    if (xmin <= coord[0] <= xmax and ymin <= coord[1] <= ymax and zmin <= coord[2] <= zmax):
                        fixed_nodes.append(i)
                
                # Convert node indices to DOF indices
                fixed_dofs = []
                nodal_dofs = basis.nodal_dofs
                for node_idx in fixed_nodes:
                    for dof_idx in nodal_dofs[:, node_idx]:
                        fixed_dofs.append(dof_idx)
                
                fixed_dofs = np.array(fixed_dofs, dtype=int)
                
                # Debug Viz: Sparse sampling of fixed nodes
                step = max(1, len(fixed_nodes) // 50)
                for i in range(0, len(fixed_nodes), step):
                    idx = fixed_nodes[i]
                    debug_constraints.append({'pos': mesh.p[:, idx].tolist()})
                    
                print(f"[TopOpt] Fixed {len(fixed_nodes)} nodes, {len(fixed_dofs)} DOFs")
                
            elif 'condition' in constraint:
                # Legacy condition-based constraint
                cond_str = constraint['condition']
                def constraint_func(p):
                    x, y, z = p
                    return eval(cond_str, {'x': x, 'y': y, 'z': z, 'np': np})
                    
                # We need node indices for visualization, but basis.get_dofs returns DOFs.
                # Let's verify manually using the condition
                mask = constraint_func(mesh.p)
                fixed_nodes_indices = np.where(mask)[0]
                
                fixed_dofs = basis.get_dofs(constraint_func).all()
                
                # Debug Viz
                step = max(1, len(fixed_nodes_indices) // 50)
                for i in range(0, len(fixed_nodes_indices), step):
                    idx = fixed_nodes_indices[i]
                    debug_constraints.append({'pos': mesh.p[:, idx].tolist()})
            else:
                fixed_dofs = np.array([], dtype=int)
        except Exception as e:
            print(f"Error parsing constraint: {e}")
            fixed_dofs = np.array([], dtype=int)

        # Assemble Load Vector f
        f = np.zeros(basis.N)
        load_vec = load['vector']
        load_cond = load['condition']
        
        try:
            matching_nodes_indices = np.where(eval(load_cond, {'x': x, 'y': y, 'z': z, 'np': np}))[0]
            n_load_nodes = len(matching_nodes_indices)
            if n_load_nodes > 0:
                fx, fy, fz = [val / n_load_nodes for val in load_vec]
                nodal_dofs = basis.nodal_dofs
                for node_idx in matching_nodes_indices:
                    f[nodal_dofs[0, node_idx]] += fx
                    f[nodal_dofs[1, node_idx]] += fy
                    f[nodal_dofs[2, node_idx]] += fz
                
                # Debug Viz: Sparse sampling of loaded nodes
                # Use total force vector for visualization (easier to see) or nodal force?
                # Nodal force is very small. Maybe use total load_vec as direction check.
                # Let's pass the nodal vector but scaled up by node count or just unit vector * magnitude?
                # The viewer draws arrows.
                step = max(1, n_load_nodes // 20) # Max 20 arrows
                for i in range(0, n_load_nodes, step):
                    idx = matching_nodes_indices[i]
                    debug_loads.append({
                        'start': mesh.p[:, idx].tolist(),
                        'vector': load_vec # Use total force vector for visualization direction/magnitude
                    })
        except Exception as e:
            print(f"Load application error: {e}")


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
            term1 = 2.0 * w['mu'] * ddot(E, D)
            term2 = w['lam'] * tr(E) * tr(D)
            
            # Add small epsilon to avoid singularity
            return (rho_min + w['rho'] ** penal) * (term1 + term2)
            
        # Energy Functional for Sensitivity
        @Functional
        def strain_energy(w):
            # Manual strain energy density
            def epsilon(w):
                return sym_grad(w)
            
            u = w['u']
            E = epsilon(u)
            
            # 1/2 * sigma : epsilon
            # But we need 2*mu*E:E + lam*tr(E)^2
            # Wait, linear_elasticity form returns sigma:epsilon.
            # Strain energy is 1/2 * sigma : epsilon.
            # But here we just need the term that scales with rho.
            # The compliance is u^T K u.
            # Element energy is u_e^T k_e u_e.
            # This is exactly what the bilinear form evaluates if u=v.
            
            term1 = 2.0 * w['mu'] * ddot(E, E)
            term2 = w['lam'] * tr(E) * tr(E)
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
        
        # Calculate filter weights once for density filter
        density_filter_weights = None
        if filter_type == 'density' and filter_radius > 0:
            # We don't need to return anything here, just checking it runs? 
            # Actually we need weights for chain rule.
            # But the functions compute them on the fly. That's O(N log N) per iter.
            # For max efficiency we should compute weights once.. but let's stick to the function interface for now.
            pass

        # Optimization Loop
        logger.info(f"TopOpt: Optimization loop started. Type: {update_scheme}, Filter: {filter_type}, Max iter: {max_iter}")
        lam_interp = basis0.interpolate(lam_field)
        mu_interp = basis0.interpolate(mu_field)
        
        for loop in range(max_iter):
            # 1. Apply Density Filter (if enabled)
            densities_phys = densities
            weight_sums = None
            if filter_type == 'density' and filter_radius > 0:
                densities_phys, weight_sums = density_filter_3d(densities, centroids, filter_radius)
            
            # 2. FE Analysis (using physical densities)
            rho_interp = basis0.interpolate(densities_phys)
            K = stiffness.assemble(basis, rho=rho_interp, lam=lam_interp, mu=mu_interp)
            
            # Solve
            try:
                with suppress_output():
                    u = solve(*condense(K, f, D=fixed_dofs))
            except Exception as e:
                logger.error(f"Solver failed at iter {loop}: {e}")
                break
            
            # 3. Sensitivity Analysis
            energies = strain_energy.elemental(basis, u=basis.interpolate(u), lam=lam_interp, mu=mu_interp)
            
            # dc/drho = -p * rho^(p-1) * energy
            dc = -penal * (densities_phys ** (penal - 1)) * energies
            
            # Divide by volume (sensitivity per unit volume)
            # Make sure we use the correct volume measure
            dc = dc / volumes
            
            # Apply Filter to Sensitivities
            if filter_type == 'density' and filter_radius > 0:
                # Chain rule for density filter
                dc = density_filter_chainrule(dc, densities, densities_phys, centroids, filter_radius, weight_sums)
            elif filter_type == 'sensitivity' and filter_radius > 0:
                # Heuristic sensitivity filter
                dc = sensitivity_filter(dc, centroids, filter_radius)
            
            # 4. Update Design Variables
            if update_scheme == 'MMA':
                # Objective: Compliance (minimize)
                # Constraint: Volume - Target <= 0
                
                # Compliance value (approximate)
                c = np.sum(densities_phys**penal * energies * volumes)
                
                # Volume constraint value
                current_vol = np.sum(densities_phys * volumes)
                vol_constraint = current_vol - target_vol
                
                # Volume constraint gradient
                # dV/dx = volumes (if no filter) or filtered volumes
                dvol = volumes
                if filter_type == 'density' and filter_radius > 0:
                    # Chain rule for volume constraint gradient
                    # dVol/dx = sum(dV/dy * dy/dx) = sum(volumes * dy/dx)
                    # This is effectively applying the filter to the volumes vector
                    dvol = density_filter_chainrule(volumes, densities, densities_phys, centroids, filter_radius, weight_sums)
                
                # Call MMA
                rho_new, low, upp = mma_update(n, loop, densities, 0, 1, xold1, xold2, 
                                             c, dc, vol_constraint, dvol, low, upp, move=move)
                
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
            if sym_map_x is not None:
                src_indices, target_indices = sym_map_x
                min_densities = np.minimum(densities[src_indices], densities[target_indices])
                densities[src_indices] = min_densities
                densities[target_indices] = min_densities
            if sym_map_y is not None:
                src_indices, target_indices = sym_map_y
                min_densities = np.minimum(densities[src_indices], densities[target_indices])
                densities[src_indices] = min_densities
                densities[target_indices] = min_densities
            if sym_map_z is not None:
                src_indices, target_indices = sym_map_z
                min_densities = np.minimum(densities[src_indices], densities[target_indices])
                densities[src_indices] = min_densities
                densities[target_indices] = min_densities
            
            logger.info(f"Iter {loop}: Change {change:.4f}, Vol {np.sum(densities*volumes)/total_vol:.2f}")
            
            if change < conv_tol:
                break
        
        # Calculate final stress for visualization if requested
        stress = None
        try:
            # Re-solve with final density
            rho_interp = basis0.interpolate(densities)
            K = stiffness.assemble(basis, rho=rho_interp, lam=lam_interp, mu=mu_interp)
            with suppress_output():
                u = solve(*condense(K, f, D=fixed_dofs))
            
            # Project stress
            basis_p1 = basis.with_element(ElementTetP1())
            
            @LinearForm
            def von_mises(v, w):
                def epsilon(w):
                    return sym_grad(w)
                E = epsilon(w['u'])
                mu = w['mu']
                lam = w['lam']
                E11, E12, E13 = E[0,0], E[0,1], E[0,2]
                E21, E22, E23 = E[1,0], E[1,1], E[1,2]
                E31, E32, E33 = E[2,0], E[2,1], E[2,2]
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
            b = von_mises.assemble(basis_p1, u=basis.interpolate(u), mu=basis_p1.zeros()+mu_val, lam=basis_p1.zeros()+lam_val)
            with suppress_output():
                stress = solve(M, b)
            stress = np.abs(stress)
        except Exception as e:
            logger.warning(f"TopOpt stress calc failed: {e}")

        # Shape recovery for manufacturable geometry
        recovered_shape = None
        if shape_recovery_enabled:
            verts, faces = shape_recovery(mesh, densities, self.get_property('density_cutoff'), 
                                          smoothing_iterations=int(smoothing_iter))
            if verts is not None and faces is not None:
                recovered_shape = {'vertices': verts, 'faces': faces}
        
        logger.info(f"TopOpt: Optimization complete. Final VolFrac: {np.mean(densities):.3f}")
        return {
            'mesh': mesh,
            'density': densities,
            'stress': stress,
            'recovered_shape': recovered_shape,
            'type': 'topopt',
            'visualization_mode': self.get_property('visualization'),
            # Pass cutoff so the viewer can actually remove low-density material
            'density_cutoff': self.get_property('density_cutoff'),
            'debug_loads': debug_loads,
            'debug_constraints': debug_constraints
        }
