# Copyright (c) 2026 Kutay Demir.
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

def sensitivity_filter(sensitivities, centers, r_min, densities=None):
    """
    Apply sensitivity filtering using a KD-Tree for O(N log N) performance.
    
    Uses the original Sigmund (2001) formulation:
        ĉ_e = (1 / (ρ_e * Σ_i H_ei)) * Σ_i (H_ei * ρ_i * dc_i)
    
    where H_ei = max(0, r_min - dist(e, i)) is the filter weight.
    This includes the ρ_i / ρ_e density weighting from Bendsøe & Sigmund (2003).
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
            if densities is not None and densities[i] > 1e-10:
                # Sigmund (2001) original: ĉ_e = Σ(H_ei * ρ_i * dc_i) / (ρ_e * Σ H_ei)
                filtered[i] = np.sum(weights * densities[indices] * sensitivities[indices]) / (densities[i] * weight_sum)
            else:
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


def heaviside_projection(densities, beta, eta=0.5):
    """
    Apply Heaviside projection to sharpen material boundaries.
    
    This eliminates gray (intermediate) densities by projecting them
    toward 0 or 1, producing clearer manufacturable geometries.
    
    Parameters:
    -----------
    densities : array - filtered densities in [0, 1]
    beta : float - sharpness parameter (0 = no effect, higher = sharper)
                   Typical values: 1, 2, 4, 8, 16, 32
    eta : float - threshold (default 0.5, densities below go to 0, above go to 1)
    
    Returns:
    --------
    projected : array - projected densities with sharper 0/1 boundaries
    d_proj : array - derivative of projection for sensitivity chain rule
    """
    if beta < 1e-6:
        # No projection
        return densities.copy(), np.ones_like(densities)
    
    # Smooth Heaviside approximation
    # H_beta(x) = (tanh(beta*eta) + tanh(beta*(x - eta))) / (tanh(beta*eta) + tanh(beta*(1-eta)))
    tanh_eta = np.tanh(beta * eta)
    tanh_1_eta = np.tanh(beta * (1 - eta))
    
    numerator = tanh_eta + np.tanh(beta * (densities - eta))
    denominator = tanh_eta + tanh_1_eta
    
    projected = numerator / denominator
    
    # Derivative for chain rule
    # dH/dx = beta * (1 - tanh^2(beta*(x - eta))) / (tanh(beta*eta) + tanh(beta*(1-eta)))
    d_proj = beta * (1 - np.tanh(beta * (densities - eta))**2) / denominator
    
    return projected, d_proj


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
    asyinit = 0.7
    asyincr = 1.2
    asydecr = 0.65
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
        vol_constraint = np.sum(dfdx * (xnew - xval)) + fval
        
        if vol_constraint > 0:
            l1 = lmid
        else:
            l2 = lmid
        
        if (l2 - l1) / (l1 + l2 + 1e-10) < 1e-4:
            break
    
    return xnew, low, upp

def shape_recovery(mesh, densities, cutoff, smoothing_iterations=3, resolution=100):
    """Recover manufacturable geometry from density field using isosurface extraction.

    Improvements over naive marching cubes:
    1. Anisotropic grid — voxel count per axis scales with geometry extents.
    2. Taubin (λ-μ) smoothing — prevents volume shrinkage of pure Laplacian.
    3. Vectorised adjacency & smoothing — ~100× faster for large meshes.
    4. KD-tree inverse-distance interpolation — faster & more robust than griddata.
    5. Adaptive Gaussian sigma — resolution-independent pre-smoothing.
    6. Centroid-level pre-smoothing — feeds a cleaner field into marching cubes.
    7. Zero-padding — ensures closed surfaces at domain boundaries.
    """
    try:
        from skimage import measure
        import scipy.ndimage as ndi

        # ------------------------------------------------------------------
        # 0. Element centroids & domain extents
        # ------------------------------------------------------------------
        centroids = mesh.p[:, mesh.t].mean(axis=1)  # (3, n_elem)

        x_min, x_max = centroids[0].min(), centroids[0].max()
        y_min, y_max = centroids[1].min(), centroids[1].max()
        z_min, z_max = centroids[2].min(), centroids[2].max()

        extents = np.array([x_max - x_min, y_max - y_min, z_max - z_min])
        max_extent = extents.max()

        # Guard against degenerate (flat) dimensions
        extents = np.maximum(extents, max_extent * 0.01)

        # ------------------------------------------------------------------
        # 1. Anisotropic grid resolution — uniform voxel SIZE, not count
        # ------------------------------------------------------------------
        nx = max(10, int(resolution * extents[0] / max_extent))
        ny = max(10, int(resolution * extents[1] / max_extent))
        nz = max(10, int(resolution * extents[2] / max_extent))

        x = np.linspace(x_min, x_max, nx)
        y = np.linspace(y_min, y_max, ny)
        z = np.linspace(z_min, z_max, nz)

        # ------------------------------------------------------------------
        # 6. Pre-smooth density field on FEM centroids
        #    A mild spatial filter on the raw densities before interpolation
        #    feeds cleaner input to marching cubes.
        # ------------------------------------------------------------------
        voxel_size = max_extent / resolution
        pre_smooth_radius = 1.5 * voxel_size  # Reduced from 2.0 to limit boundary erosion
        pre_tree = cKDTree(centroids.T)
        neighbors_list = pre_tree.query_ball_point(centroids.T, pre_smooth_radius)

        smoothed_densities = np.empty_like(densities)
        for i, nbrs in enumerate(neighbors_list):
            if not nbrs:
                smoothed_densities[i] = densities[i]
                continue
            idx = np.array(nbrs)
            dists = np.linalg.norm(centroids[:, idx].T - centroids[:, i], axis=1)
            w = np.maximum(0.0, pre_smooth_radius - dists)
            ws = w.sum()
            smoothed_densities[i] = (w @ densities[idx]) / ws if ws > 1e-10 else densities[i]

        # ------------------------------------------------------------------
        # 4. KD-tree inverse-distance interpolation (replaces griddata)
        # ------------------------------------------------------------------
        grid_x, grid_y, grid_z = np.meshgrid(x, y, z, indexing='ij')
        grid_points = np.column_stack((grid_x.ravel(), grid_y.ravel(), grid_z.ravel()))

        k_neighbors = min(8, len(smoothed_densities))
        dists, idxs = pre_tree.query(grid_points, k=k_neighbors)

        # Inverse-distance weights (avoid div-by-zero)
        weights = 1.0 / (dists + 1e-12)
        weights /= weights.sum(axis=1, keepdims=True)
        grid_densities = (weights * smoothed_densities[idxs]).sum(axis=1).reshape((nx, ny, nz))

        # Nearest-neighbor mask: zero out grid points whose nearest element is clearly void.
        # Reverted to strict cutoff to prevent filling empty spaces.
        dists_nn = dists[:, 0]
        nearest_density = smoothed_densities[idxs[:, 0]].reshape((nx, ny, nz))
        
        # Mask 1: Density cutoff
        void_mask = nearest_density < cutoff
        grid_densities[void_mask] = 0.0

        # Mask 2: Distance cutoff (prevent filling large holes/external space)
        # If a grid point is far from ANY element centroid, it's void.
        # Use robust threshold: max(2*voxel_size, 1.5*mean_element_spacing)
        # Calculate mean element spacing if not known
        # (Approximate by querying nearest neighbor distance for a subset of centroids)
        sample_size = min(100, len(centroids.T))
        d_sample, _ = pre_tree.query(centroids.T[:sample_size], k=2)
        mean_element_spacing = np.mean(d_sample[:, 1])

        dist_threshold = max(2.0 * voxel_size, 1.5 * mean_element_spacing)
        dist_mask = dists_nn > dist_threshold
        grid_densities[dist_mask.reshape((nx, ny, nz))] = 0.0

        # ------------------------------------------------------------------
        # 5. Adaptive Gaussian sigma — scales with voxel size
        #    Keep sigma small to avoid eroding thin structural members.
        # ------------------------------------------------------------------
        sigma_voxels = 0.35
        grid_densities = ndi.gaussian_filter(grid_densities, sigma=sigma_voxels)

        # ------------------------------------------------------------------
        # 7. Zero-padding — closed surfaces at domain boundaries
        # ------------------------------------------------------------------
        grid_densities = np.pad(grid_densities, pad_width=1,
                                mode='constant', constant_values=0)
        nx += 2
        ny += 2
        nz += 2

        # ------------------------------------------------------------------
        # Sanitise cutoff to lie within the data range.
        # Compensate for the systematic volume erosion introduced by
        # Gaussian smoothing + zero-padding by lowering the isosurface
        # level.  The factor 0.75 was empirically tuned so that the
        # exported STL matches the VTK density-threshold visualisation.
        # ------------------------------------------------------------------
        d_min, d_max = grid_densities.min(), grid_densities.max()
        effective_cutoff = cutoff * 0.75
        if effective_cutoff < d_min:
            effective_cutoff = d_min + 1e-5
        elif effective_cutoff > d_max:
            effective_cutoff = d_max - 1e-5

        # Extract isosurface
        if d_max > d_min:
            verts, faces, _, _ = measure.marching_cubes(
                grid_densities, level=effective_cutoff)
        else:
            return None, None

        # ------------------------------------------------------------------
        # 2 + 3. Taubin (λ-μ) smoothing — vectorised
        #
        #   λ  = 0.5   (shrink pass)
        #   μ  = -0.53  (inflate pass — |μ| > λ prevents net shrinkage)
        # ------------------------------------------------------------------
        lambda_factor = 0.5
        mu_factor = -0.53  # |μ| > λ prevents net shrinkage
        smoothing_iterations = min(smoothing_iterations, 2)  # Cap to prevent over-erosion

        for _ in range(smoothing_iterations):
            for factor in (lambda_factor, mu_factor):
                n_verts = len(verts)
                neighbor_sum = np.zeros_like(verts)
                neighbor_count = np.zeros(n_verts)

                # Vectorised edge extraction from faces
                edges = np.vstack([
                    faces[:, [0, 1]],
                    faces[:, [1, 2]],
                    faces[:, [2, 0]],
                ])
                np.add.at(neighbor_sum, edges[:, 0], verts[edges[:, 1]])
                np.add.at(neighbor_count, edges[:, 0], 1)
                np.add.at(neighbor_sum, edges[:, 1], verts[edges[:, 0]])
                np.add.at(neighbor_count, edges[:, 1], 1)

                mask = neighbor_count > 0
                avg = neighbor_sum[mask] / neighbor_count[mask, None]
                verts[mask] += factor * (avg - verts[mask])

        # ------------------------------------------------------------------
        # Scale back to original coordinates
        # (account for the +1 padding offset: vertex index 0 maps to pad,
        #  vertex 1 maps to x_min, etc.)
        # ------------------------------------------------------------------
        verts[:, 0] = (verts[:, 0] - 1) * (x_max - x_min) / (nx - 3) + x_min
        verts[:, 1] = (verts[:, 1] - 1) * (y_max - y_min) / (ny - 3) + y_min
        verts[:, 2] = (verts[:, 2] - 1) * (z_max - z_min) / (nz - 3) + z_min

        # ------------------------------------------------------------------
        # Filter disconnected components — keep largest connected component
        # ------------------------------------------------------------------
        if len(verts) > 0 and len(faces) > 0:
            try:
                from scipy.sparse import lil_matrix
                from scipy.sparse.csgraph import connected_components

                n_faces = len(faces)
                adj_matrix = lil_matrix((n_faces, n_faces))

                # Vertex-to-faces mapping
                vertex_to_faces = {}
                for i, face in enumerate(faces):
                    for v in face:
                        if v not in vertex_to_faces:
                            vertex_to_faces[v] = []
                        vertex_to_faces[v].append(i)

                for face_list in vertex_to_faces.values():
                    if len(face_list) > 1:
                        for i in range(len(face_list)):
                            for j in range(i + 1, len(face_list)):
                                adj_matrix[face_list[i], face_list[j]] = 1
                                adj_matrix[face_list[j], face_list[i]] = 1

                adj_matrix = adj_matrix.tocsr()
                n_components, labels = connected_components(adj_matrix,
                                                            directed=False)

                if n_components > 1:
                    component_sizes = np.bincount(labels)
                    largest = np.argmax(component_sizes)
                    keep = labels == largest

                    faces_filtered = faces[keep]
                    unique_verts = np.unique(faces_filtered)
                    vert_map = {old: new for new, old in enumerate(unique_verts)}
                    faces = np.array([[vert_map[v] for v in f]
                                      for f in faces_filtered])
                    verts = verts[unique_verts]

            except Exception:
                pass  # Continue with unfiltered mesh

        return verts, faces

    except ImportError:
        return None, None
    except Exception:
        return None, None

def lam_lame(E, nu):
    """Convert Young's modulus and Poisson's ratio to Lame parameters."""
    return E * nu / ((1 + nu) * (1 - 2 * nu)), E / (2 * (1 + nu))

# Professional Material Database (E in MPa, density in tonne/mm^3)
MATERIAL_DATABASE = {
    'Custom': {'E': 210000.0, 'nu': 0.30, 'rho': 7.85e-9},
    'Steel (Structural)': {'E': 210000.0, 'nu': 0.30, 'rho': 7.85e-9},
    'Steel (Stainless 304)': {'E': 193000.0, 'nu': 0.29, 'rho': 8.00e-9},
    'Aluminum 6061-T6': {'E': 68900.0, 'nu': 0.33, 'rho': 2.70e-9},
    'Aluminum 7075-T6': {'E': 71700.0, 'nu': 0.33, 'rho': 2.81e-9},
    'Titanium Ti-6Al-4V': {'E': 113800.0, 'nu': 0.34, 'rho': 4.43e-9},
    'Copper (Annealed)': {'E': 110000.0, 'nu': 0.34, 'rho': 8.96e-9},
    'Brass': {'E': 100000.0, 'nu': 0.34, 'rho': 8.50e-9},
    'Cast Iron (Gray)': {'E': 100000.0, 'nu': 0.26, 'rho': 7.20e-9},
    'Magnesium AZ31': {'E': 45000.0, 'nu': 0.35, 'rho': 1.77e-9},
    'Nickel Alloy 718': {'E': 200000.0, 'nu': 0.30, 'rho': 8.19e-9},
    'CFRP (Quasi-Isotropic)': {'E': 70000.0, 'nu': 0.30, 'rho': 1.55e-9},
    'GFRP (E-Glass)': {'E': 25000.0, 'nu': 0.23, 'rho': 1.90e-9},
    'Concrete (Normal)': {'E': 30000.0, 'nu': 0.20, 'rho': 2.40e-9},
    'ABS Plastic': {'E': 2300.0, 'nu': 0.35, 'rho': 1.05e-9},
    'Nylon 6/6': {'E': 2900.0, 'nu': 0.40, 'rho': 1.14e-9},
    'PEEK': {'E': 3600.0, 'nu': 0.38, 'rho': 1.30e-9},
    'Wood (Oak)': {'E': 12000.0, 'nu': 0.35, 'rho': 0.60e-9},
}

class MaterialNode(CadQueryNode):
    """Defines material properties with preset database."""
    __identifier__ = 'com.cad.sim.material'
    NODE_NAME = 'Material'

    def __init__(self):
        super().__init__()
        self.add_output('material', color=(200, 200, 200))
        
        # Add Inputs for parametric material properties
        self.add_input('youngs_modulus', color=(180, 180, 0))
        self.add_input('poissons_ratio', color=(180, 180, 0))
        self.add_input('density', color=(180, 180, 0))
        
        # Preset dropdown
        self.create_property('preset', 'Steel (Structural)', widget_type='combo',
                             items=list(MATERIAL_DATABASE.keys()))
        
        # Keep properties as defaults (editable for Custom)
        self.create_property('youngs_modulus', 210000.0, widget_type='float')  # MPa
        self.create_property('poissons_ratio', 0.3, widget_type='float')
        self.create_property('density', 7.85e-9, widget_type='float')  # tonne/mm^3

    def run(self):
        # Check if using preset or custom
        preset = self.get_property('preset')
        
        if preset != 'Custom' and preset in MATERIAL_DATABASE:
            mat = MATERIAL_DATABASE[preset]
            E = mat['E']
            nu = mat['nu']
            rho = mat['rho']
        else:
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
        
        # Mesh type selection
        self.create_property('mesh_type', 'Tet', widget_type='combo',
                             items=['Tet', 'Tet10'])
        self.create_property('element_size', 2.0, widget_type='float')
        self.create_property('refinement_size', 0.5, widget_type='float')  # Finer mesh for critical areas

    def run(self):
        if OCCGeometry is None:
            self.set_error("Netgen-occ is not installed")
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
            except Exception:
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
                        except Exception:
                            pass
                    
                    # 3. Generate Mesh
                    # maxh controls the global element size
                    ng_mesh = geo.GenerateMesh(maxh=size)
                    
                    # 4. Export to Gmsh format (Version 2 is most compatible with skfem/meshio)
                    # Netgen's Export function takes the filename and the format string
                    ng_mesh.Export(msh_path, "Gmsh2 Format")
                
                # 5. Load into skfem
                mesh = Mesh.load(msh_path)
                
            except Exception:
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
        
        except Exception:
            return None
        
        return mesh

class ConstraintNode(CadQueryNode):
    """Applies boundary constraints (fixed, roller, pinned, displacement) to a face."""
    __identifier__ = 'com.cad.sim.constraint'
    NODE_NAME = 'FEA Constraint (Face)'

    def __init__(self):
        super().__init__()
        self.add_input('mesh', color=(200, 100, 200))
        # Input for the specific face geometry to constrain
        self.add_input('target_face', color=(100, 200, 255))
        self.add_output('constraints', color=(255, 100, 100))
        
        # Constraint type selection
        self.create_property('constraint_type', 'Fixed', widget_type='combo',
                             items=['Fixed', 'Roller X', 'Roller Y', 'Roller Z', 
                                    'Pinned', 'Symmetry X', 'Symmetry Y', 'Symmetry Z',
                                    'Displacement'])
        
        # Displacement values for prescribed BC (used when type is 'Displacement')
        self.create_property('displacement_x', 0.0, widget_type='float')
        self.create_property('displacement_y', 0.0, widget_type='float')
        self.create_property('displacement_z', 0.0, widget_type='float')
        
        # Keep string condition as fallback for backward compatibility
        self.create_property('condition', '', widget_type='text')

    def run(self):
        mesh = self.get_input_value('mesh', None)
        target_wp = self.get_input_value('target_face', None)
        constraint_type = self.get_property('constraint_type')
        fallback_condition = self.get_property('condition')
        
        # Get displacement values
        disp_x = float(self.get_property('displacement_x'))
        disp_y = float(self.get_property('displacement_y'))
        disp_z = float(self.get_property('displacement_z'))
        
        if mesh is None:
            return None
        
        # Map constraint type to DOF constraints
        # 'fixed_dofs' indicates which DOFs (0=x, 1=y, 2=z) are fixed
        # 'free_dofs' indicates which DOFs are free (for roller/pinned)
        constraint_mapping = {
            'Fixed': {'fixed_dofs': [0, 1, 2], 'displacement': None},
            'Roller X': {'fixed_dofs': [0], 'displacement': None},  # Only X fixed
            'Roller Y': {'fixed_dofs': [1], 'displacement': None},  # Only Y fixed
            'Roller Z': {'fixed_dofs': [2], 'displacement': None},  # Only Z fixed
            'Pinned': {'fixed_dofs': [0, 1, 2], 'displacement': None},  # Same as fixed for 3D
            'Symmetry X': {'fixed_dofs': [0], 'displacement': None},  # No motion normal to X
            'Symmetry Y': {'fixed_dofs': [1], 'displacement': None},  # No motion normal to Y
            'Symmetry Z': {'fixed_dofs': [2], 'displacement': None},  # No motion normal to Z
            'Displacement': {'fixed_dofs': [0, 1, 2], 'displacement': [disp_x, disp_y, disp_z]},
        }
        
        constraint_info = constraint_mapping.get(constraint_type, constraint_mapping['Fixed'])

        # If no face input provided, use fallback string condition
        if target_wp is None:
            if not fallback_condition:
                self.set_error("No target face or condition")
                return None
            return {
                'type': constraint_type.lower().replace(' ', '_'),
                'condition': fallback_condition,
                'fixed_dofs': constraint_info['fixed_dofs'],
                'displacement': constraint_info['displacement']
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
            
            return {
                'type': constraint_type.lower().replace(' ', '_'),
                'geometry': face_obj,
                'fixed_dofs': constraint_info['fixed_dofs'],
                'displacement': constraint_info['displacement']
            }

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
        # Input for the specific face geometry to load
        self.add_input('target_face', color=(100, 200, 255))

        # Add inputs for parametric force components
        self.add_input('force_x', color=(255, 255, 0))
        self.add_input('force_y', color=(255, 255, 0))
        self.add_input('force_z', color=(255, 255, 0))

        self.add_output('loads', color=(255, 255, 0))
        
        # Load type selection
        self.create_property('load_type', 'Force', widget_type='combo',
                             items=['Force', 'Moment', 'Gravity', 'Remote Force'])
        
        # Keep string condition as fallback for backward compatibility
        self.create_property('condition', '', widget_type='text')
        
        # Force/load values
        self.create_property('force_x', 0.0, widget_type='float')
        self.create_property('force_y', -1000.0, widget_type='float')
        self.create_property('force_z', 0.0, widget_type='float')
        
        # Moment values (for Moment type)
        self.create_property('moment_x', 0.0, widget_type='float')
        self.create_property('moment_y', 0.0, widget_type='float')
        self.create_property('moment_z', 0.0, widget_type='float')
        
        # Gravity parameters
        self.create_property('gravity_accel', 9810.0, widget_type='float')  # mm/s^2
        self.create_property('gravity_direction', '-Y', widget_type='combo',
                             items=['-Y', '-Z', '-X', '+Y', '+Z', '+X'])

    def run(self):
        mesh = self.get_input_value('mesh', None)
        target_wp = self.get_input_value('target_face', None)  # This is a Workplane object

        # Resolve force inputs with fallback to properties
        fx = self.get_input_value('force_x', 'force_x')
        fy = self.get_input_value('force_y', 'force_y')
        fz = self.get_input_value('force_z', 'force_z')

        fallback_condition = self.get_property('condition')
        
        # Debug logging


        if mesh is None:
            return None

        # If no face input provided, use fallback string condition
        if target_wp is None:
            if not fallback_condition:

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
                
        except Exception:
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
                        
        except Exception:
            pass

        # 6. Solve
        try:
            with suppress_output():
                u = solve(*condense(K, f, D=fixed_dofs))
            logger.info(f"FEA Solve Complete. Max Displacement: {np.max(np.abs(u)):.6e}")
        except Exception:
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
            
        except Exception:
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
        self.add_input('constraints', color=(255, 100, 100), multi_input=True)
        self.add_input('loads', color=(255, 255, 0), multi_input=True)
        self.add_output('optimized_mesh', color=(200, 100, 200))
        self.add_output('recovered_shape', color=(100, 255, 100))
        self.create_property('vol_frac', 0.4, widget_type='float')
        self.create_property('iterations', 50, widget_type='int')  # Reasonable default
        self.create_property('filter_radius', 3.0, widget_type='float')  # Should be 2-3x element size
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
        self.create_property('convergence_tol', 0.02, widget_type='float')  # Convergence threshold
        self.create_property('recovery_resolution', 100, widget_type='int')  # Grid resolution for shape recovery
        self.create_property('smoothing_iterations', 3, widget_type='int')  # Gaussian smoothing passes
        # NEW: Filter type and update scheme selection
        self.create_property('filter_type', 'density', widget_type='combo', items=['sensitivity', 'density'])
        self.create_property('update_scheme', 'MMA', widget_type='combo', items=['MMA', 'OC'])
        
        # NEW: Heaviside projection for sharper boundaries
        self.create_property('projection', 'None', widget_type='combo', 
                             items=['None', 'Heaviside'])
        self.create_property('heaviside_beta', 4.0, widget_type='float')  # Sharpness (1-64)
        self.create_property('heaviside_eta', 0.5, widget_type='float')   # Threshold (0-1)
        self.create_property('continuation', True, widget_type='bool')   # Gradually increase beta


    def run(self, progress_callback=None):
        logger.info("TopOpt: Optimization started.")
        
        # Helper to flatten inputs for list validation
        def flatten_initial(inputs):
            flat = []
            if not inputs: return flat
            for item in inputs:
                if isinstance(item, list):
                    flat.extend(item)
                elif item is not None:
                    flat.append(item)
            return flat

        mesh = self.get_input_value('mesh', None)
        material = self.get_input_value('material', None)
        
        # Fetch lists and validate
        raw_constraints = self.get_input_list('constraints')
        constraints = flatten_initial(raw_constraints)
        
        raw_loads = self.get_input_list('loads')
        loads = flatten_initial(raw_loads)
        
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
        
        if not (mesh and material and constraints and loads):
            logger.warning(f"TopOpt: Missing inputs! Mesh:{mesh is not None}, Mat:{material is not None}, Cons:{len(constraints)} items, Load:{len(loads)} items")
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
        
        # Helper to flatten inputs
        def flatten_inputs(inputs):
            flat = []
            if not inputs: return flat
            for item in inputs:
                if isinstance(item, list):
                    flat.extend(item)
                elif item is not None:
                    flat.append(item)
            return flat

        # Get all constraints and loads
        constraint_list = flatten_inputs(self.get_input_list('constraints'))
        load_list = flatten_inputs(self.get_input_list('loads'))
        
        fixed_dofs = np.array([], dtype=int)
        
        # --- PROCESS CONSTRAINTS ---
        try:
            for c in constraint_list:
                if not c: continue
                
                fixed_dof_indices = c.get('fixed_dofs', [0, 1, 2])
                
                if 'geometry' in c:
                    # Geometry-based constraint selection
                    face_shape = c['geometry']
                    node_coords = np.column_stack((x, y, z))
                    
                    # Bounding box filter
                    bb = face_shape.BoundingBox()
                    tolerance = 1e-3
                    xmin, xmax = bb.xmin - tolerance, bb.xmax + tolerance
                    ymin, ymax = bb.ymin - tolerance, bb.ymax + tolerance
                    zmin, zmax = bb.zmin - tolerance, bb.zmax + tolerance
                    
                    fixed_nodes = []
                    for i, coord in enumerate(node_coords):
                        if (xmin <= coord[0] <= xmax and ymin <= coord[1] <= ymax and zmin <= coord[2] <= zmax):
                            try:
                                from cadquery import Vector
                                point = Vector(coord[0], coord[1], coord[2])
                                distance = face_shape.distanceTo(point)
                                if distance <= tolerance:
                                    fixed_nodes.append(i)
                            except:
                                fixed_nodes.append(i) # Fallback to BB
                    
                    # Convert to DOFs
                    nodal_dofs = basis.nodal_dofs
                    for node_idx in fixed_nodes:
                        for dof_idx in fixed_dof_indices:
                             fixed_dofs = np.union1d(fixed_dofs, [nodal_dofs[dof_idx, node_idx]])
                    
                    # Debug Viz (Sampled)
                    step = max(1, len(fixed_nodes) // 50)
                    for i in range(0, len(fixed_nodes), step):
                        idx = fixed_nodes[i]
                        debug_constraints.append({'pos': mesh.p[:, idx].tolist()})
                        
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
                
                if 'geometry' in l:
                    # Geometry-based load
                    face_shape = l['geometry']
                    node_coords = np.column_stack((x, y, z))
                    
                    # Bounding box filter
                    bb = face_shape.BoundingBox()
                    tolerance = 1e-3
                    xmin, xmax = bb.xmin - tolerance, bb.xmax + tolerance
                    ymin, ymax = bb.ymin - tolerance, bb.ymax + tolerance
                    zmin, zmax = bb.zmin - tolerance, bb.zmax + tolerance
                    
                    load_nodes = []
                    for i, coord in enumerate(node_coords):
                        if (xmin <= coord[0] <= xmax and ymin <= coord[1] <= ymax and zmin <= coord[2] <= zmax):
                            try:
                                from cadquery import Vector
                                point = Vector(coord[0], coord[1], coord[2])
                                distance = face_shape.distanceTo(point)
                                if distance <= tolerance:
                                    load_nodes.append(i)
                            except:
                                load_nodes.append(i)
                    
                    if not load_nodes: continue
                    
                    n_nodes = len(load_nodes)
                    nodal_dofs = basis.nodal_dofs
                    
                    # Distribute load (Simplified Point Load Distribution)
                    fx, fy, fz = [v / n_nodes for v in vector]
                    
                    for node_idx in load_nodes:
                         f[nodal_dofs[0, node_idx]] += fx
                         f[nodal_dofs[1, node_idx]] += fy
                         f[nodal_dofs[2, node_idx]] += fz
                    
                    # Debug Viz
                    step = max(1, n_nodes // 20)
                    for i in range(0, n_nodes, step):
                        idx = load_nodes[i]
                        debug_loads.append({
                            'start': mesh.p[:, idx].tolist(),
                            'vector': vector
                        })
                        
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
            # Track consecutive convergence (require 3 consecutive low-change iterations)
            if loop == 0:
                consecutive_converged = 0
                prev_compliance = 0.0
                obj_plateau_count = 0

            # 1. Apply Density Filter (if enabled)
            densities_phys = densities
            weight_sums = None
            if filter_type == 'density' and filter_radius > 0:
                densities_phys, weight_sums = density_filter_3d(densities, centroids, filter_radius)
            
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
                dc = sensitivity_filter(dc, centroids, filter_radius, densities_phys)
            
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
            
            # Compute compliance for logging
            obj_val = np.sum(densities_phys**penal * energies * volumes) if update_scheme != 'MMA' else c
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
        # IMPORTANT: Use physical (filtered + projected) densities, not raw design variables.
        # Raw design variables can have many intermediate values that the filter resolves
        # into clean 0/1; using them directly would kill legitimate structural members.
        recovered_shape = None
        if shape_recovery_enabled:
            # Recompute final physical densities
            densities_final = densities
            if filter_type == 'density' and filter_radius > 0:
                densities_final, _ = density_filter_3d(densities, centroids, filter_radius)
            if projection_type == 'Heaviside':
                densities_final, _ = heaviside_projection(densities_final, beta_schedule, eta)
            
            verts, faces = shape_recovery(mesh, densities_final, self.get_property('density_cutoff'), 
                                          smoothing_iterations=int(smoothing_iter),
                                          resolution=int(recovery_res))
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


class RemeshNode(CadQueryNode):
    """
    Remesh Node - Converts surface mesh (from TopOpt) to volumetric tetrahedral mesh.
    
    This node bridges TopOpt → ShapeOpt workflow by taking the recovered shape
    (surface triangles) and creating a new volumetric mesh suitable for FEA.
    """
    __identifier__ = 'com.cad.sim.remesh'
    NODE_NAME = 'Remesh Surface'

    def __init__(self):
        super().__init__()
        # Input: TopOpt result containing recovered_shape
        self.add_input('topopt_result', color=(200, 100, 200))
        
        # Output: Volumetric mesh
        self.add_output('mesh', color=(200, 100, 200))
        self.add_output('shape', color=(100, 255, 100))  # CadQuery solid for visualization
        
        # Mesh quality settings
        self.create_property('element_size', 3.0, widget_type='float')
        self.create_property('mesh_quality', 'Medium', widget_type='combo',
                             items=['Coarse', 'Medium', 'Fine', 'Very Fine'])
        
        # Surface repair options
        self.create_property('repair_surface', True, widget_type='bool')
        self.create_property('close_holes', True, widget_type='bool')

    def run(self):
        """Convert recovered shape to volumetric mesh."""
        topopt_result = self.get_input_value('topopt_result', None)
        
        if topopt_result is None:
            logger.warning("RemeshNode: No TopOpt result provided")
            return None
        
        # Extract recovered shape from TopOpt result
        recovered_shape = None
        if isinstance(topopt_result, dict):
            recovered_shape = topopt_result.get('recovered_shape', None)
        
        if recovered_shape is None:
            logger.warning("RemeshNode: No recovered_shape in TopOpt result")
            return None
        
        vertices = recovered_shape.get('vertices', None)
        faces = recovered_shape.get('faces', None)
        
        if vertices is None or faces is None:
            logger.warning("RemeshNode: Invalid recovered_shape format")
            return None
        
        logger.info(f"RemeshNode: Processing surface with {len(vertices)} vertices, {len(faces)} faces")
        
        element_size = self.get_property('element_size')
        quality = self.get_property('mesh_quality')
        
        # Adjust element size based on quality setting
        quality_multipliers = {
            'Coarse': 2.0,
            'Medium': 1.0,
            'Fine': 0.5,
            'Very Fine': 0.25
        }
        effective_size = element_size * quality_multipliers.get(quality, 1.0)
        
        try:
            # Method 1: Create solid from surface mesh and mesh with Netgen
            mesh, solid = self._remesh_via_solid(vertices, faces, effective_size)
            
            if mesh is not None:
                logger.info(f"RemeshNode: Created volumetric mesh with {mesh.nelements} elements")
                return {
                    'mesh': mesh,
                    'shape': solid,
                    'type': 'remesh'
                }
            
            # Method 2 fallback: Direct tetrahedral mesh from surface
            mesh = self._remesh_direct(vertices, faces, effective_size)
            if mesh is not None:
                logger.info(f"RemeshNode: Created mesh via direct method with {mesh.nelements} elements")
                return {
                    'mesh': mesh,
                    'shape': None,
                    'type': 'remesh'
                }
            
        except Exception as e:
            logger.error(f"RemeshNode: Meshing failed: {e}")
        
        return None
    
    def _remesh_via_solid(self, vertices, faces, element_size):
        """Create volumetric mesh by first creating a solid from surface."""
        # Performance guard: very large meshes are slow to sew into B-Rep.
        # Raise the limit to 20 000 faces (typical TopOpt output) with a warning.
        if len(faces) > 20000:
            logger.info(f"RemeshNode: Mesh very complex ({len(faces)} faces > 20000). Skipping solid conversion.")
            return None, None
        if len(faces) > 5000:
            logger.info(f"RemeshNode: Large mesh ({len(faces)} faces). Solid conversion may be slow.")
            
        mesh = None
        solid = None
        
        # Helper to create solid from faces
        try:
            import cadquery as cq
            from OCP.BRepBuilderAPI import BRepBuilderAPI_Sewing, BRepBuilderAPI_MakeSolid
            from OCP.BRep import BRep_Builder
            from OCP.TopoDS import TopoDS_Shell, TopoDS_Compound, TopoDS_Solid
            from OCP.BRepMesh import BRepMesh_IncrementalMesh
            from OCP.gp import gp_Pnt
            from OCP.BRepBuilderAPI import BRepBuilderAPI_MakePolygon, BRepBuilderAPI_MakeFace
            from OCP.ShapeFix import ShapeFix_Solid, ShapeFix_Shell
            
            logger.info("RemeshNode: Attempting surface sewing to create solid...")
            
            # Build shell from triangular faces
            sew = BRepBuilderAPI_Sewing(1e-4)  # Tighter tolerance
            
            for face in faces:
                try:
                    # Get triangle vertices
                    pts = [gp_Pnt(float(vertices[idx, 0]), 
                                  float(vertices[idx, 1]), 
                                  float(vertices[idx, 2])) for idx in face]
                    
                    # Create triangular face
                    poly = BRepBuilderAPI_MakePolygon(pts[0], pts[1], pts[2], True)
                    if not poly.IsDone(): continue
                        
                    wire = poly.Wire()
                    face_builder = BRepBuilderAPI_MakeFace(wire, True)
                    if face_builder.IsDone():
                        sew.Add(face_builder.Face())
                except Exception:
                    continue
            
            sew.Perform()
            sewed_shape = sew.SewedShape()
            
            # Check if we have a shell
            if sewed_shape.ShapeType() == 0: # TopAbs_COMPOUND
                # Try to extract shells
                pass 
                
            # Try to fix shell
            fixer = ShapeFix_Shell(TopoDS_Shell(sewed_shape))
            fixer.Perform()
            shell = fixer.Shell()
            
            # Make solid
            solid_builder = BRepBuilderAPI_MakeSolid()
            solid_builder.Add(shell)
            
            if solid_builder.IsDone():
                occ_solid = solid_builder.Solid()
                
                # Fix solid orientation/volume
                fixer_sol = ShapeFix_Solid(occ_solid)
                fixer_sol.Perform()
                occ_solid = fixer_sol.Solid()
                
                solid = cq.Workplane().add(cq.Shape(occ_solid))
                
                # Mesh with Netgen if available
                if OCCGeometry is not None:
                    try:
                        geo = OCCGeometry(occ_solid)
                        ngmesh = geo.GenerateMesh(maxh=element_size)
                        
                        # Export/Import cycle for skfem
                        with tempfile.NamedTemporaryFile(suffix='.msh', delete=False) as f:
                            ngmesh.Export(f.name, 'Gmsh2 Format')
                            f.close()
                            mesh = skfem.MeshTet.load(f.name)
                            os.unlink(f.name)
                            
                    except Exception as e:
                        logger.warning(f"RemeshNode: Netgen meshing failed: {e}")
            else:
                logger.warning("RemeshNode: Failed to close shell into solid")
                
        except Exception as e:
            logger.warning(f"RemeshNode: Solid creation failed: {e}")
        
        return mesh, solid
    
    def _remesh_direct(self, vertices, faces, element_size):
        """Direct tetrahedral meshing using scipy Delaunay."""
        try:
            from scipy.spatial import Delaunay
            
            # Create 3D Delaunay triangulation of the vertices
            # This creates tetrahedra filling the convex hull
            tri = Delaunay(vertices)
            
            # Filter tetrahedra to keep only those inside the surface
            # Use point-in-mesh test based on face normals + KDTree
            centroids = vertices[tri.simplices].mean(axis=1)
            
            # Simple inside test: keep tetrahedra whose centroids are near original surface
            from scipy.spatial import cKDTree
            face_centers = vertices[faces].mean(axis=1)
            tree = cKDTree(face_centers)
            
            # For each tetrahedron centroid, find distance to nearest face center
            distances, indices = tree.query(centroids)
            
            # Estimate typical edge length from sample faces
            edge_lengths = []
            for face in faces[:min(100, len(faces))]:
                pts = vertices[face]
                vals = [np.linalg.norm(pts[i] - pts[(i+1)%3]) for i in range(3)]
                edge_lengths.extend(vals)
            avg_edge = np.mean(edge_lengths) if edge_lengths else element_size
            
            # Heuristic: keep tets that are "close enough" to the surface shell
            # Ideally we would do a winding number check, but that's expensive in pure python/scipy
            # A threshold of avg_edge * 1.5 usually keeps the bulk without too many outliers
            # Also, check if centroid is "behind" the nearest face (dot product of normal)
            
            valid_mask = np.zeros(len(centroids), dtype=bool)
            
            # Precompute face normals
            v0 = vertices[faces[:, 0]]
            v1 = vertices[faces[:, 1]]
            v2 = vertices[faces[:, 2]]
            normals = np.cross(v1 - v0, v2 - v0)
            norms = np.linalg.norm(normals, axis=1, keepdims=True) + 1e-10
            normals /= norms
            
            for i, (dist, idx) in enumerate(zip(distances, indices)):
                if dist > avg_edge * 3.0: # Too far, definitely outside
                    continue
                
                # Check direction (is centroid 'inside' relative to nearest face?)
                # Vector from face center to centroid
                vec = centroids[i] - face_centers[idx]
                val = np.dot(vec, normals[idx])
                
                # If dot product is negative, it's on the 'back' side of the face (inside)
                # Or if it's very close (within tolerance)
                if val < 0.1 * avg_edge: 
                    valid_mask[i] = True
            
            valid_simplices = tri.simplices[valid_mask]
            
            if len(valid_simplices) > 0:
                mesh = skfem.MeshTet(vertices.T, valid_simplices.T)
                return mesh
                
        except Exception as e:
            logger.debug(f"RemeshNode: Direct method failed: {e}")
        
        return None


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
        self.create_property('optimizer', 'SLSQP', widget_type='combo',
                             items=['SLSQP', 'L-BFGS-B', 'trust-constr', 'Powell'])
        
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
                from pylcss.cad.nodes.fem import MeshNode
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
                result = minimize(
                    objective, x0, method=optimizer,
                    bounds=bounds, constraints=constraints_list,
                    options={'maxiter': max_iter, 'disp': True, 'ftol': tol}
                )
            else:
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
            
            fixed_dofs = c.get('fixed_dofs', [0, 1, 2])
            condition = c.get('condition', '')
            
            if condition:
                try:
                    # Use string condition
                    def make_cond(cond_str):
                        def condition_func(x):
                            return eval(cond_str, {'x': x, 'np': np})
                        return condition_func
                    
                    facet_dofs = basis.get_dofs(make_cond(condition))
                    for dof_idx in fixed_dofs:
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
            condition = load.get('condition', '')
            
            if condition:
                try:
                    def make_cond(cond_str):
                        def condition_func(x):
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


class ShapeOptimizationNode(CadQueryNode):
    """
    Shape Optimization Node - Optimizes boundary geometry via mesh morphing.
    
    This node performs Form Optimization by moving boundary nodes to minimize
    stress concentrations or compliance while maintaining volume constraints.
    Uses Laplacian smoothing for regularization.
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
                    
                    min_vol = np.min(vols)
                    if min_vol <= 1e-9:
                        logger.warning(f"ShapeOpt: Mesh inversion detected (min vol {min_vol:.2e}). Backtracking...")
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
                
                if 'geometry' in c:
                    # NEW: Geometry-based constraint (from SelectFaceNode)
                    face_shape = c['geometry']
                    tolerance = 1e-3
                    
                    bb = face_shape.BoundingBox()
                    xmin, xmax = bb.xmin - tolerance, bb.xmax + tolerance
                    ymin, ymax = bb.ymin - tolerance, bb.ymax + tolerance
                    zmin, zmax = bb.zmin - tolerance, bb.zmax + tolerance
                    
                    for i, coord in enumerate(node_coords):
                        if (xmin <= coord[0] <= xmax and
                            ymin <= coord[1] <= ymax and
                            zmin <= coord[2] <= zmax):
                            try:
                                from cadquery import Vector
                                point = Vector(coord[0], coord[1], coord[2])
                                distance = face_shape.distanceTo(point)
                                if distance <= tolerance:
                                    fixed = np.union1d(fixed, [i])
                            except:
                                fixed = np.union1d(fixed, [i])
                
                elif 'condition' in c and c['condition']:
                    # LEGACY: String-based condition
                    condition = c['condition']
                    try:
                        def make_cond(cond_str):
                            def condition_func(x):
                                return eval(cond_str, {'x': x, 'np': np})
                            return condition_func
                        facets = mesh.facets_satisfying(make_cond(condition))
                        fixed = np.union1d(fixed, np.unique(mesh.facets[:, facets].flatten()))
                    except:
                        pass
        
        # Additional fixed faces from property (string conditions)
        for face_cond in fixed_faces:
            try:
                def make_cond(cond_str):
                    def condition_func(x):
                        return eval(cond_str, {'x': x, 'np': np})
                    return condition_func
                facets = mesh.facets_satisfying(make_cond(face_cond))
                fixed = np.union1d(fixed, np.unique(mesh.facets[:, facets].flatten()))
            except:
                pass
        
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
        """Compute shape sensitivity at boundary nodes."""
        stress = result['stress']
        displacement = result['displacement']
        
        # Simple approximation: sensitivity proportional to local stress
        sensitivities = np.zeros(len(moveable_nodes))
        
        # Map stress to nodes (stress is at nodes from L2 projection)
        for i, node_idx in enumerate(moveable_nodes):
            if node_idx < len(stress):
                sensitivities[i] = stress[node_idx]
            else:
                sensitivities[i] = np.mean(stress)
        
        # Normalize
        sens_max = np.max(np.abs(sensitivities))
        if sens_max > 1e-10:
            sensitivities = sensitivities / sens_max
        
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
                    
                    if 'geometry' in c:
                        # NEW: Geometry-based constraint (from SelectFaceNode)
                        face_shape = c['geometry']
                        node_coords = np.column_stack((x, y, z))
                        
                        fixed_nodes = []
                        tolerance = 1e-3
                        
                        # Bounding box for quick filtering
                        bb = face_shape.BoundingBox()
                        xmin, xmax = bb.xmin - tolerance, bb.xmax + tolerance
                        ymin, ymax = bb.ymin - tolerance, bb.ymax + tolerance
                        zmin, zmax = bb.zmin - tolerance, bb.zmax + tolerance
                        
                        for i, coord in enumerate(node_coords):
                            if (xmin <= coord[0] <= xmax and
                                ymin <= coord[1] <= ymax and
                                zmin <= coord[2] <= zmax):
                                try:
                                    from cadquery import Vector
                                    point = Vector(coord[0], coord[1], coord[2])
                                    distance = face_shape.distanceTo(point)
                                    if distance <= tolerance:
                                        fixed_nodes.append(i)
                                except:
                                    fixed_nodes.append(i)
                        
                        # Convert node indices to DOF indices
                        nodal_dofs = basis.nodal_dofs
                        for node_idx in fixed_nodes:
                            for dof_idx in fixed_dof_indices:
                                dofs_to_fix = np.union1d(dofs_to_fix, [nodal_dofs[dof_idx, node_idx]])
                        
                        logger.debug(f"ShapeOpt: Fixed {len(fixed_nodes)} nodes from geometry")
                        
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
                    
                    if 'geometry' in load:
                        # NEW: Geometry-based load (from SelectFaceNode)
                        face_shape = load['geometry']
                        node_coords = np.column_stack((x, y, z))
                        
                        load_nodes = []
                        tolerance = 1e-3
                        
                        bb = face_shape.BoundingBox()
                        xmin, xmax = bb.xmin - tolerance, bb.xmax + tolerance
                        ymin, ymax = bb.ymin - tolerance, bb.ymax + tolerance
                        zmin, zmax = bb.zmin - tolerance, bb.zmax + tolerance
                        
                        for i, coord in enumerate(node_coords):
                            if (xmin <= coord[0] <= xmax and
                                ymin <= coord[1] <= ymax and
                                zmin <= coord[2] <= zmax):
                                try:
                                    from cadquery import Vector
                                    point = Vector(coord[0], coord[1], coord[2])
                                    distance = face_shape.distanceTo(point)
                                    if distance <= tolerance:
                                        load_nodes.append(i)
                                except:
                                    load_nodes.append(i)
                        
                        if load_nodes:
                            n_nodes = len(load_nodes)
                            nodal_dofs = basis.nodal_dofs
                            for node_idx in load_nodes:
                                for i, v in enumerate(vector):
                                    f[nodal_dofs[i, node_idx]] += v / n_nodes
                            logger.debug(f"ShapeOpt: Applied load to {n_nodes} nodes")
                        
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
            
            return {
                'displacement': u,
                'stress': stress,
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
