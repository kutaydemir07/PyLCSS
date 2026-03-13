# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.
"""Shared helper functions, constants, and utilities for the FEM node package.

Import from here: lam_lame, MATERIAL_DATABASE, suppress_output, OCCGeometry,
build_filter_matrix, sensitivity_filter, density_filter_3d, heaviside_projection,
density_filter_chainrule, _find_matching_boundary_facets, _assemble_traction_force,
_assemble_pressure_force, mma_update, shape_recovery, tr.
"""
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
    """Context manager to suppress stdout **and** C-level stdout/stderr.

    Python's sys.stdout redirect does not silence output written directly to
    file-descriptor 1 (e.g. Netgen's C++ std::cout).  This implementation
    uses os.dup2() to redirect the raw file descriptors so that *all* output
    — including C-extension output — is sent to /dev/null (or NUL on Windows).
    """
    if not simulation_config.SUPPRESS_EXTERNAL_LIBRARY_OUTPUT:
        yield
        return

    # Flush Python-level buffers before redirecting.
    sys.stdout.flush()
    sys.stderr.flush()

    # Save duplicates of the real file descriptors.
    old_stdout_fd = os.dup(1)
    old_stderr_fd = os.dup(2)

    try:
        with open(os.devnull, 'w') as devnull:
            devnull_fd = devnull.fileno()
            # Redirect FD 1 & 2 at the OS level.
            os.dup2(devnull_fd, 1)
            os.dup2(devnull_fd, 2)
            # Also redirect Python-level streams.
            old_py_stdout = sys.stdout
            old_py_stderr = sys.stderr
            sys.stdout = open(os.devnull, 'w')
            sys.stderr = open(os.devnull, 'w')
            try:
                yield
            finally:
                sys.stdout.close()
                sys.stderr.close()
                sys.stdout = old_py_stdout
                sys.stderr = old_py_stderr
    finally:
        # Restore the original file descriptors unconditionally.
        os.dup2(old_stdout_fd, 1)
        os.dup2(old_stderr_fd, 2)
        os.close(old_stdout_fd)
        os.close(old_stderr_fd)

# ADD: Netgen imports
try:
    from netgen.occ import OCCGeometry
    import netgen.meshing as ngmeshing
except ImportError:
    OCCGeometry = None

# Alias trace to tr if needed, or just use trace
tr = trace

def build_filter_matrix(centroids, r_min):
    """
    Build a sparse filter matrix H for O(1) vectorized filter applications.
    H_ij = max(0, r_min - dist(i, j))
    """
    from scipy.sparse import coo_matrix
    tree = cKDTree(centroids)
    neighbors_list = tree.query_ball_point(centroids, r_min)
    
    I, J, V = [], [], []
    for i, neighbors in enumerate(neighbors_list):
        for j in neighbors:
            dist = np.linalg.norm(centroids[i] - centroids[j])
            H_ij = max(0.0, r_min - dist)
            if H_ij > 1e-10:
                I.append(i)
                J.append(j)
                V.append(H_ij)
    
    H = coo_matrix((V, (I, J)), shape=(len(centroids), len(centroids))).tocsr()
    H_sum = np.array(H.sum(axis=1)).flatten()
    return H, H_sum

def sensitivity_filter(sensitivities, H, H_sum, densities=None):
    """
    Apply sensitivity filtering using precomputed sparse matrix H.
    """
    if densities is not None:
        safe_densities = np.maximum(densities, 1e-10)
        return H.dot(safe_densities * sensitivities) / (safe_densities * H_sum)
    else:
        return H.dot(sensitivities) / H_sum

def density_filter_3d(densities, H, H_sum):
    """
    Apply density filter using precomputed sparse matrix H.
    """
    return H.dot(densities) / H_sum


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


def density_filter_chainrule(dc, H, H_sum):
    """
    Apply chain rule for density filter sensitivities using precomputed sparse matrix H.
    """
    return H.T.dot(dc / H_sum)


def _find_matching_boundary_facets(mesh, geoms, tolerance=1.5):
    """
    Locate mesh boundary facets that correspond to the selected CAD faces.

    Returns
    -------
    np.ndarray | None
        Boundary facet indices in mesh.facets, or None if nothing matched.
    """
    from cadquery import Vector as CQVector

    boundary_facet_ids = mesh.boundary_facets()
    if boundary_facet_ids is None or len(boundary_facet_ids) == 0:
        return None

    bf_node_ids = mesh.facets[:, boundary_facet_ids]
    bf_midpoints = mesh.p[:, bf_node_ids].mean(axis=1)

    bbox_list = [g.BoundingBox() for g in geoms]
    xmin_bb = min(b.xmin for b in bbox_list) - tolerance
    xmax_bb = max(b.xmax for b in bbox_list) + tolerance
    ymin_bb = min(b.ymin for b in bbox_list) - tolerance
    ymax_bb = max(b.ymax for b in bbox_list) + tolerance
    zmin_bb = min(b.zmin for b in bbox_list) - tolerance
    zmax_bb = max(b.zmax for b in bbox_list) + tolerance

    in_bb = (
        (bf_midpoints[0] >= xmin_bb) & (bf_midpoints[0] <= xmax_bb) &
        (bf_midpoints[1] >= ymin_bb) & (bf_midpoints[1] <= ymax_bb) &
        (bf_midpoints[2] >= zmin_bb) & (bf_midpoints[2] <= zmax_bb)
    )
    candidate_local_idxs = np.where(in_bb)[0]

    loaded_local_idxs = []
    for li in candidate_local_idxs:
        px, py, pz = bf_midpoints[:, li]
        pt = CQVector(float(px), float(py), float(pz))
        for g in geoms:
            try:
                if g.distanceTo(pt) <= tolerance:
                    loaded_local_idxs.append(li)
                    break
            except Exception:
                loaded_local_idxs.append(li)
                break

    if not loaded_local_idxs:
        return None

    return boundary_facet_ids[np.array(loaded_local_idxs, dtype=np.int32)]


def _assemble_traction_force(mesh, basis, geoms, vector, tolerance=1.5):
    """
    Assemble a nodal force vector using FacetBasis integration (proper Neumann BC).

    Distributes a total force vector over the loaded surface facets via numerical
    integration.  Because integration is area-weighted, dense mesh regions do not
    receive artificially large loads — this corrects the error of dividing the
    total force equally across all matched nodes on unstructured meshes.

    Parameters
    ----------
    mesh      : skfem MeshTet
    basis     : skfem CellBasis (P1 or P2 tetrahedral)
    geoms     : list of CadQuery Face objects — target surface
    vector    : (3,) iterable — total force [Fx, Fy, Fz]
    tolerance : float — geometric snap tolerance (mm)

    Returns
    -------
    f_traction : np.ndarray of shape (basis.N,), or None if assembly failed.
    On None the caller should fall back to equal nodal distribution.
    """
    try:
        loaded_facets = _find_matching_boundary_facets(mesh, geoms, tolerance=tolerance)
        if loaded_facets is None:
            return None

        # Compute total loaded area for traction = force / area
        fi_nodes   = mesh.facets[:, loaded_facets]        # (3, n_loaded)
        fv         = mesh.p[:, fi_nodes]                  # (3, 3, n_loaded)
        e1         = fv[:, 1, :] - fv[:, 0, :]            # (3, n_loaded)
        e2         = fv[:, 2, :] - fv[:, 0, :]            # (3, n_loaded)
        cross      = np.cross(e1.T, e2.T).T               # (3, n_loaded)
        face_areas = 0.5 * np.linalg.norm(cross, axis=0)  # (n_loaded,)
        total_area = float(np.sum(face_areas))

        if total_area < 1e-20:
            return None

        # Constant traction vector (uniform distributed load)
        tx = float(vector[0]) / total_area
        ty = float(vector[1]) / total_area
        tz = float(vector[2]) / total_area

        # FacetBasis restricted to loaded boundary facets only
        fb = FacetBasis(mesh, basis.elem, facets=loaded_facets)

        @LinearForm
        def traction_form(v, w):
            return tx * v[0] + ty * v[1] + tz * v[2]

        return traction_form.assemble(fb)

    except Exception as e:
        logger.warning(
            f"_assemble_traction_force: FacetBasis assembly failed ({e}). "
            "Falling back to nodal load distribution."
        )
        return None


def _assemble_pressure_force(mesh, basis, geoms, pressure, tolerance=1.5):
    """
    Assemble a pressure load using the mesh facet normals.

    Unlike a single centroid normal, this keeps the traction direction
    consistent with the actual boundary facet orientation on curved faces.
    """
    try:
        loaded_facets = _find_matching_boundary_facets(mesh, geoms, tolerance=tolerance)
        if loaded_facets is None:
            return None

        fb = FacetBasis(mesh, basis.elem, facets=loaded_facets)
        pressure = float(pressure)

        @LinearForm
        def pressure_form(v, w):
            return pressure * (w.n[0] * v[0] + w.n[1] * v[1] + w.n[2] * v[2])

        return pressure_form.assemble(fb)

    except Exception as e:
        logger.warning(
            f"_assemble_pressure_force: FacetBasis assembly failed ({e}). "
            "Falling back to equivalent traction force assembly."
        )
        return None


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

