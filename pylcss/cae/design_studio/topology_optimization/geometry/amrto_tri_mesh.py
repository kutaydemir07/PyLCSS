# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Step 1: Generation of the triangular mesh model from topology optimization results.

Reference:
Section 2.3: Generation of the triangular mesh model (step 1)
In AMRTO (Ren et al., 2025):
"The homemade code has been developed using MATLAB to smooth the pseudo-density field,
identify the isosurface with a reference pseudo-density, and generate the triangular mesh
model... Preprocessing techniques, such as Laplacian smoothing, face normal smoothing,
flipping normal error patches, eliminating duplicate patches, merging overlapping points,
and filling gaps, may be used to address possible flaws in the triangular mesh,
resulting in a smooth and watertight mesh model."
"""

from __future__ import annotations

import logging
from typing import Optional, Sequence, Tuple

import numpy as np
from scipy import ndimage as ndi
from skimage import measure
import trimesh

from .amrto_common import AMRTOConfig

logger = logging.getLogger(__name__)


def smooth_pseudo_density_field(
    field: np.ndarray,
    sigma: float = 0.5,
    window_size: int = 3,
    method: str = "gaussian",
) -> np.ndarray:
    """Smooth the pseudo-density field as in AMRTO Section 2.3."""
    grid = np.asarray(field, dtype=float)
    if grid.ndim != 3 or grid.size == 0:
        return grid

    if method == "gaussian" and sigma > 0.0:
        return ndi.gaussian_filter(grid, sigma=float(sigma), mode="nearest")
    elif method == "box" and window_size > 1:
        return ndi.uniform_filter(grid, size=int(window_size), mode="nearest")
    return grid


def extract_isosurface_mesh(
    field: np.ndarray,
    cutoff: float = 0.5,
    bounds: Optional[Tuple[Sequence[float], Sequence[float]]] = None,
    smooth_field: bool = True,
    smooth_sigma: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract isosurface from pseudo-density field with physical bounds mapping."""
    grid = np.asarray(field, dtype=float)
    if grid.ndim != 3 or grid.size == 0:
        raise ValueError(f"Density field must be 3D, got shape {grid.shape}")
    if not np.all(np.isfinite(grid)):
        raise ValueError("Density field contains non-finite values.")

    if smooth_field and smooth_sigma > 0.0:
        grid = smooth_pseudo_density_field(grid, sigma=smooth_sigma)
    # Densities live at voxel centres. Smooth the field before padding: blurring
    # the artificial exterior into the field incorrectly shrinks a full box.
    smoothed = np.pad(grid, pad_width=1, mode="constant", constant_values=0.0)

    cutoff_val = float(cutoff if cutoff is not None else 0.5)
    f_min, f_max = float(np.min(smoothed)), float(np.max(smoothed))
    if not np.isfinite(cutoff_val) or not f_min < cutoff_val < f_max:
        raise ValueError(
            f"No density isosurface at requested cutoff {cutoff_val:g}; "
            f"smoothed field range is [{f_min:g}, {f_max:g}]."
        )
    verts, faces, _, _ = measure.marching_cubes(
        smoothed, level=cutoff_val, step_size=1, allow_degenerate=False,
    )

    # Shift vertices by -1 to undo the 1-voxel padding
    verts = verts - 1.0

    # Map grid coordinates to physical bounds
    if bounds is not None:
        mins = np.asarray(bounds[0], dtype=float).reshape(3)
        maxs = np.asarray(bounds[1], dtype=float).reshape(3)
        if not np.all(np.isfinite([mins, maxs])) or not np.all(maxs > mins):
            raise ValueError("Mesh bounds must be finite and strictly increasing.")
        voxel_size = (maxs - mins) / np.asarray(grid.shape, dtype=float)
        verts = mins + (verts + 0.5) * voxel_size
        # Explicit bounds are the outer voxel faces. Against zero padding a
        # cutoff below 0.5 crosses beyond those faces; retain the physical
        # domain instead of growing an exterior skin outside the source CAD.
        # Higher-cutoff contours and all interior crossings stay unchanged.
        verts = np.clip(verts, mins, maxs)
    else:
        # Match the centred, unit-pitch convention used by passive regions.
        verts = verts + 0.5 - 0.5 * np.asarray(grid.shape, dtype=float)

    return verts.astype(float), faces.astype(np.int64)


def preprocess_and_repair_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    laplacian_iters: int = 2,
    laplacian_lambda: float = 0.3,
    domain_bounds: Optional[Tuple[Sequence[float], Sequence[float]]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Execute preprocessing techniques from AMRTO Section 2.3.

    Includes:
    - Eliminating duplicate patches and merging overlapping points
    - Removing degenerate (zero-area) faces
    - Flipping normal error patches and face normal smoothing
    - Filling small holes
    - Feature-constrained Laplacian smoothing for fair boundary curvature
    """
    if len(vertices) == 0 or len(faces) == 0:
        return vertices, faces

    mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
        process=False,
    )
    # Generic validation deletes collinear triangles and opens a previously
    # closed TPMS surface. Keep a closed input's vertex identities; welding can
    # also join independent bodies that happen to touch geometrically.
    was_closed = bool(mesh.is_watertight)
    if not was_closed:
        mesh.process(validate=False)

    # 1. Eliminate duplicate patches / unreferenced vertices
    if hasattr(mesh, "unique_faces"):
        mesh.update_faces(mesh.unique_faces())
    else:
        unique_idx = trimesh.grouping.unique_rows(np.sort(mesh.faces, axis=1))[0]
        mesh.update_faces(unique_idx)

    if was_closed and np.any(mesh.area_faces == 0):
        mesh = _collapse_zero_area_faces_preserving_topology(mesh)
    elif not was_closed:
        mesh.update_faces(mesh.area_faces > 0)

    mesh.remove_unreferenced_vertices()

    # 2. Fix normals & inversion
    # Closed components include inward-oriented cavity boundaries. Inverting
    # each component independently fills every enclosed void with material.
    # Marching Cubes supplies coherent field orientation; reverse the entire
    # surface only when its aggregate material volume is negative.
    trimesh.repair.fix_normals(mesh, multibody=False)
    trimesh.repair.fix_inversion(mesh, multibody=False)

    # 3. Fill small gaps if non-watertight
    if not mesh.is_watertight:
        try:
            trimesh.repair.fill_holes(mesh)
        except Exception:
            pass

    # 4. Laplacian smoothing as described in AMRTO Section 2.3
    if laplacian_iters > 0:
        try:
            mesh = _smooth_mesh_preserving_features(
                mesh, iterations=int(laplacian_iters), lamb=float(laplacian_lambda),
                domain_bounds=domain_bounds,
            )
        except Exception as exc:
            logger.debug("Laplacian smoothing skipped: %s", exc)

    return np.asarray(mesh.vertices, dtype=float), np.asarray(mesh.faces, dtype=np.int64)


def _smooth_mesh_preserving_features(mesh: trimesh.Trimesh, *, iterations: int,
                                     lamb: float, crease_angle_degrees: float = 30.0,
                                     domain_bounds: Optional[Tuple[Sequence[float], Sequence[float]]] = None,
                                     ) -> trimesh.Trimesh:
    """Fair curved regions without moving sharp edges or planar interfaces.

    Marching Cubes replaces a voxel box corner with 45-degree chamfers, so a
    60-degree crease test misses even its simplest sharp features. Pin vertices
    incident to 30-degree creases and open/nonmanifold boundaries. Coplanar
    vertex stars can move tangentially in their original plane; smooth curved
    stars retain full freedom. Explicit domain-plane vertices are also fixed;
    mesh extrema alone never identify planes. Use normalized inverse-distance
    neighbors.

    Trimesh's volume constraint rescales *every* vertex about the center of
    mass, moving planes and pinned features. Instead restore the signed volume
    of each closed shell along its locally admissible volume gradient. Reject
    a step that reverses a triangle, collapses it, or cannot retain shell volume.
    """
    if not np.isfinite(lamb) or not 0 <= lamb <= 1:
        raise ValueError("Explicit Laplacian smoothing requires 0 <= lambda <= 1")
    if iterations <= 0 or lamb == 0:
        return mesh
    from scipy.sparse import coo_matrix

    original = np.asarray(mesh.vertices).copy()
    faces = np.asarray(mesh.faces)
    normals = np.asarray(mesh.face_normals)
    edge_vertices = np.asarray(mesh.edges_unique)
    edge_counts = np.bincount(mesh.edges_unique_inverse, minlength=len(edge_vertices))
    pinned = np.zeros(len(original), dtype=bool)
    pinned[edge_vertices[edge_counts != 2].ravel()] = True
    creases = mesh.face_adjacency_angles >= np.deg2rad(crease_angle_degrees)
    pinned[mesh.face_adjacency_edges[creases].ravel()] = True
    lower, upper = None, None
    if domain_bounds is not None:
        lower = np.asarray(domain_bounds[0], dtype=float).reshape(3)
        upper = np.asarray(domain_bounds[1], dtype=float).reshape(3)
        if not np.isfinite([lower, upper]).all() or not np.all(upper > lower):
            raise ValueError("Smoothing domain bounds must be finite and strictly increasing")
        tolerance = 64 * np.finfo(float).eps * np.maximum(
            np.maximum(np.abs(lower), np.abs(upper)), upper - lower,
        )
        pinned |= np.any((np.abs(original - lower) <= tolerance)
                         | (np.abs(original - upper) <= tolerance), axis=1)

    # Unit-normal averaging identifies actual coplanar stars, independently of
    # triangle area and mesh scale. Curved stars are not classified as planes.
    normal_sum = np.zeros_like(original)
    np.add.at(normal_sum, faces.ravel(), np.repeat(normals, 3, axis=0))
    count = np.bincount(faces.ravel(), minlength=len(original))
    average_normals = normal_sum / np.maximum(count, 1)[:, None]
    lengths = np.linalg.norm(average_normals, axis=1)
    planar = (count > 0) & (lengths >= 1 - 1e-10) & ~pinned
    plane_normals = average_normals / np.maximum(lengths, np.finfo(float).tiny)[:, None]

    def project(displacements, vertex_ids=None):
        displacements = np.asarray(displacements).copy()
        planar_mask = planar if vertex_ids is None else planar[vertex_ids]
        normals_to_use = plane_normals if vertex_ids is None else plane_normals[vertex_ids]
        pinned_mask = pinned if vertex_ids is None else pinned[vertex_ids]
        displacements[planar_mask] -= (
            np.einsum("ij,ij->i", displacements[planar_mask], normals_to_use[planar_mask])[:, None]
            * normals_to_use[planar_mask]
        )
        displacements[pinned_mask] = 0
        return displacements

    distances = np.linalg.norm(original[edge_vertices[:, 0]] - original[edge_vertices[:, 1]], axis=1)
    if np.any(distances <= 0) or not np.isfinite(distances).all():
        raise ValueError("Smoothing requires positive finite edge lengths")
    rows = edge_vertices.ravel()
    columns = edge_vertices[:, ::-1].ravel()
    weights = np.repeat(1 / distances, 2)
    operator = coo_matrix((weights, (rows, columns)), shape=(len(original), len(original))).tocsr()
    operator = operator.multiply(1 / np.maximum(np.asarray(operator.sum(axis=1)).ravel(),
                                              np.finfo(float).tiny)[:, None]).tocsr()

    shells = []
    if mesh.is_watertight:
        for face_ids in trimesh.graph.connected_components(mesh.face_adjacency, nodes=np.arange(len(faces))):
            shell_faces = faces[face_ids]
            vertex_ids = np.unique(shell_faces)
            local_faces = np.searchsorted(vertex_ids, shell_faces)
            origin = original[vertex_ids].mean(axis=0)
            points = original[vertex_ids][local_faces] - origin
            volume = float(np.einsum("ij,ij->i", points[:, 0],
                                    np.cross(points[:, 1], points[:, 2])).sum() / 6)
            scale = float(np.linalg.norm(np.ptp(original[vertex_ids], axis=0)))
            tolerance = max(abs(volume) * 1e-11, scale ** 3 * np.finfo(float).eps * 32)
            shells.append((vertex_ids, local_faces, origin, volume, tolerance))

    def restore_shell_volumes(candidate):
        for vertex_ids, local_faces, origin, target, tolerance in shells:
            for _ in range(8):
                points = candidate[vertex_ids][local_faces] - origin
                volume = float(np.einsum("ij,ij->i", points[:, 0],
                                        np.cross(points[:, 1], points[:, 2])).sum() / 6)
                difference = target - volume
                if abs(difference) <= tolerance:
                    if target != 0 and target * volume <= 0:
                        return None
                    break
                gradient = np.zeros((len(vertex_ids), 3))
                for corner in range(3):
                    np.add.at(gradient, local_faces[:, corner],
                              np.cross(points[:, (corner + 1) % 3], points[:, (corner + 2) % 3]) / 6)
                gradient = project(gradient, vertex_ids)
                squared = float(np.einsum("ij,ij->", gradient, gradient))
                if not np.isfinite(squared) or squared <= np.finfo(float).tiny:
                    return None
                candidate[vertex_ids] += (difference / squared) * gradient
                # Restore exact references after projection/roundoff.
                candidate[vertex_ids] = original[vertex_ids] + project(
                    candidate[vertex_ids] - original[vertex_ids], vertex_ids,
                )
            else:
                return None
        return candidate

    vertices = original.copy()
    initial_cross = np.cross(original[faces[:, 1]] - original[faces[:, 0]],
                             original[faces[:, 2]] - original[faces[:, 0]])
    minimum_area = np.finfo(float).eps * max(float(np.max(np.ptp(original, axis=0))) ** 2,
                                           np.finfo(float).tiny)
    for _ in range(iterations):
        displacement = project(operator @ vertices - vertices)
        for fraction in (1., .5, .25, .125):
            candidate = original + project(vertices + lamb * fraction * displacement - original)
            candidate = restore_shell_volumes(candidate)
            if candidate is None or not np.isfinite(candidate).all():
                continue
            # Volume restoration can also push a nearby interior vertex past
            # a domain plane. Reject that step rather than clipping after the
            # volume check or silently accepting an exterior material skin.
            if lower is not None and (np.any(candidate < lower) or np.any(candidate > upper)):
                continue
            cross = np.cross(candidate[faces[:, 1]] - candidate[faces[:, 0]],
                             candidate[faces[:, 2]] - candidate[faces[:, 0]])
            if (np.any(np.linalg.norm(cross, axis=1) <= minimum_area)
                    or np.any(np.einsum("ij,ij->i", cross, initial_cross) <= 0)):
                continue
            vertices = candidate
            break
        else:
            logger.debug("Feature-constrained Laplacian step rejected; retaining preceding geometry")
            break
    result = mesh.copy()
    result.vertices = vertices
    return result


def _collapse_zero_area_faces_preserving_topology(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Repair collinear triangles by checked local edge collapses, never holes."""
    from .amrto_quadrangulation import mesh_topology_report, _quad_topology_error

    vertices = np.asarray(mesh.vertices).copy()
    faces = np.asarray(mesh.faces).copy()
    original = mesh_topology_report(vertices, faces)
    # The caller checks combinatorial closure; degeneracy is the defect being
    # repaired. Enforce its original component counts and genus on the result.
    original["is_closed"] = True
    volume = abs(float(mesh.volume))
    maximum_movement = 0.0
    collapsed_edges = 0
    for _ in range(64):
        points = vertices[faces]
        areas = np.linalg.norm(np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0]), axis=1)
        bad_faces = np.flatnonzero(areas == 0)
        if not len(bad_faces):
            break
        changed = False
        for face_index in bad_faces:
            triangle = faces[face_index]
            candidates = sorted((float(np.linalg.norm(vertices[a] - vertices[b])), int(a), int(b))
                                for a, b in zip(triangle, np.roll(triangle, 1)))
            for movement, kept, removed in candidates:
                adjacent_kept = faces[np.any(faces == kept, axis=1)]
                adjacent_removed = faces[np.any(faces == removed, axis=1)]
                shared_faces = faces[np.sum((faces == kept) | (faces == removed), axis=1) == 2]
                kept_neighbors = set(adjacent_kept.flat) - {kept}
                removed_neighbors = set(adjacent_removed.flat) - {removed}
                opposite_vertices = set(shared_faces.flat) - {kept, removed}
                # The closed-manifold link condition rules out collapsing a
                # handle, joining bodies, or deleting a tetrahedral component.
                if (len(shared_faces) != 2 or len(opposite_vertices) != 2
                        or kept_neighbors.intersection(removed_neighbors) != opposite_vertices):
                    continue
                trial = faces.copy()
                trial[trial == removed] = kept
                mask = ((trial[:, 0] != trial[:, 1]) & (trial[:, 1] != trial[:, 2])
                        & (trial[:, 2] != trial[:, 0]))
                if int(np.count_nonzero(~mask)) != 2:
                    continue
                trial = trial[mask]
                # Avoid creating coincident opposite faces, which can happen
                # for a tetrahedron despite the vertex-neighborhood condition.
                local = trial[np.any(trial == kept, axis=1)]
                if len(np.unique(np.sort(local, axis=1), axis=0)) != len(local):
                    continue
                faces = trial
                maximum_movement = max(maximum_movement, movement)
                collapsed_edges += 1
                changed = True
                break
            if changed:
                break
        if not changed:
            break
    repaired = trimesh.Trimesh(vertices, faces, process=False)
    topology = mesh_topology_report(vertices, faces)
    error = _quad_topology_error(original, topology)
    relative_volume_delta = abs(abs(float(repaired.volume)) - volume) / volume if volume else 0.0
    if error or relative_volume_delta > 1e-5:
        raise ValueError("Closed surface degeneracy could not be repaired while preserving topology and volume: "
                         f"{error or 'volume changed by more than 0.001%'}")
    logger.warning("Collapsed %d zero-area triangle edges while preserving every closed component and genus; "
                   "maximum endpoint movement %.9g, relative volume delta %.9g",
                   collapsed_edges, maximum_movement, relative_volume_delta)
    return repaired


def generate_topopt_triangular_mesh(
    field: np.ndarray,
    cutoff: float = 0.5,
    bounds: Optional[Tuple[Sequence[float], Sequence[float]]] = None,
    config: Optional[AMRTOConfig] = None,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Step 1 entry point: Generate a smooth, watertight triangular mesh from TO results."""
    cfg = config or AMRTOConfig(density_cutoff=cutoff)
    verts, faces = extract_isosurface_mesh(
        field,
        cutoff=cfg.density_cutoff,
        bounds=bounds,
        smooth_field=True,
        smooth_sigma=cfg.density_smooth_sigma,
    )

    verts, faces = preprocess_and_repair_mesh(
        verts,
        faces,
        laplacian_iters=cfg.mesh_laplacian_smoothing_iters,
        laplacian_lambda=cfg.mesh_laplacian_lambda,
        domain_bounds=bounds,
    )

    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    vol = abs(float(mesh.volume)) if mesh.is_watertight else None
    components = len(mesh.split(only_watertight=False)) if len(faces) > 0 else 0

    report = {
        "step": 1,
        "vertex_count": len(verts),
        "face_count": len(faces),
        "volume": vol,
        "connected_components": int(components),
        "boundary_shell_count": int(components),
        "component_semantics": "Connected boundary shells; enclosed cavity shells are included",
        "is_watertight": bool(mesh.is_watertight),
        "density_cutoff": float(cfg.density_cutoff),
        "mesh_smoothing_method": "Feature-constrained Laplacian with per-shell signed-volume preservation",
        "mesh_smoothing_crease_angle_degrees": 30.0,
        "explicit_domain_bounds_retained": bounds is not None,
        "implementation_revision": "2026-10-10-preserved-edges",
    }
    return verts, faces, report
