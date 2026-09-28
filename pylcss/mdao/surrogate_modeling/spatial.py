# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Signed-distance and regular-grid operations for geometry operators."""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

try:
    import trimesh

    TRIMESH_AVAILABLE = True
except ImportError:
    TRIMESH_AVAILABLE = False


def compute_sdf(
    points: np.ndarray,
    cells: np.ndarray,
    query_points: np.ndarray,
) -> np.ndarray:
    """Evaluate signed distance to a supported surface or volume mesh."""
    if not TRIMESH_AVAILABLE:
        raise RuntimeError("trimesh is required for signed-distance computation.")

    points = np.asarray(points, dtype=np.float64)
    cells = np.asarray(cells, dtype=np.int64)
    query_points = np.asarray(query_points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not len(points):
        raise ValueError(f"points must have shape (n_nodes, 3); got {points.shape}.")
    if cells.ndim != 2 or cells.shape[1] < 3 or not len(cells):
        raise ValueError(
            f"cells must have shape (n_cells, at_least_3); got {cells.shape}."
        )
    if query_points.ndim != 2 or query_points.shape[1] != 3:
        raise ValueError(
            f"query_points must have shape (n_query, 3); got {query_points.shape}."
        )
    if not np.isfinite(points).all() or not np.isfinite(query_points).all():
        raise ValueError("Mesh and query coordinates must be finite.")
    if np.any(cells < 0) or np.any(cells >= len(points)):
        raise ValueError("cells contains node indices outside the points array.")

    width = cells.shape[1]
    if width in (4, 6, 8, 10):
        faces = _volume_to_surface(cells)
    elif width == 3:
        faces = cells
    else:
        raise ValueError(
            f"Unsupported cell width {width}; expected triangles, tetrahedra, "
            "wedges, hexahedra, or quadratic tetrahedra."
        )
    if not len(faces):
        raise ValueError("The mesh has no exterior surface faces.")

    mesh = trimesh.Trimesh(vertices=points, faces=faces, process=True)
    if mesh.is_watertight:
        if not mesh.is_winding_consistent:
            mesh.fix_normals(multibody=True)
        # trimesh uses positive-inside; operator-learning literature commonly
        # uses negative-inside.
        distance = -mesh.nearest.signed_distance(query_points)
    else:
        logger.debug("Mesh is open; using unsigned distance.")
        _, distance, _ = mesh.nearest.on_surface(query_points)
    return np.asarray(distance, dtype=np.float64)


def _tetra_to_surface(tets: np.ndarray) -> np.ndarray:
    """Extract triangles that occur in exactly one tetrahedron."""
    triangles = np.vstack(
        [
            tets[:, [0, 1, 2]],
            tets[:, [0, 1, 3]],
            tets[:, [0, 2, 3]],
            tets[:, [1, 2, 3]],
        ]
    )
    sorted_triangles = np.sort(triangles, axis=1)
    _, inverse, counts = np.unique(
        sorted_triangles,
        axis=0,
        return_inverse=True,
        return_counts=True,
    )
    return triangles[counts[inverse] == 1]


def _volume_to_surface(cells: np.ndarray) -> np.ndarray:
    """Extract triangulated boundary faces from common volume elements."""
    width = cells.shape[1]
    if width in (4, 10):
        return _tetra_to_surface(cells[:, :4])
    local_faces: tuple[tuple[int, ...], ...]
    if width == 6:
        local_faces = (
            (0, 1, 2),
            (3, 5, 4),
            (0, 3, 4, 1),
            (1, 4, 5, 2),
            (2, 5, 3, 0),
        )
    elif width == 8:
        local_faces = (
            (0, 3, 2, 1),
            (4, 5, 6, 7),
            (0, 1, 5, 4),
            (1, 2, 6, 5),
            (2, 3, 7, 6),
            (3, 0, 4, 7),
        )
    else:
        raise ValueError(f"Unsupported volume cell width: {width}.")

    counts: dict[tuple[int, ...], int] = {}
    oriented: dict[tuple[int, ...], tuple[int, ...]] = {}
    for cell in cells:
        for local in local_faces:
            face = tuple(int(cell[index]) for index in local)
            key = tuple(sorted(face))
            counts[key] = counts.get(key, 0) + 1
            oriented.setdefault(key, face)

    triangles: list[tuple[int, int, int]] = []
    for key, count in counts.items():
        if count != 1:
            continue
        face = oriented[key]
        if len(face) == 3:
            triangles.append(face)
        else:
            triangles.extend([(face[0], face[1], face[2]), (face[0], face[2], face[3])])
    return np.asarray(triangles, dtype=np.int64).reshape(-1, 3)


def make_background_grid(
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    resolution: int = 32,
) -> tuple[np.ndarray, tuple[int, int, int]]:
    """Create a padded uniform 3-D grid over a bounding box."""
    if resolution < 2:
        raise ValueError("resolution must be at least 2.")
    bbox_min, bbox_max = _validate_bbox(bbox_min, bbox_max)
    extent = bbox_max - bbox_min
    padding = extent * 0.1
    lower, upper = bbox_min - padding, bbox_max + padding
    axes = [np.linspace(lower[index], upper[index], resolution) for index in range(3)]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
    return grid.reshape(-1, 3), (resolution, resolution, resolution)


def normalize_grid_coordinates(
    points: np.ndarray,
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Map xyz to ``grid_sample``'s ``(z, y, x)`` order and ``[-1, 1]``."""
    points = np.asarray(points, dtype=np.float64)
    bbox_min, bbox_max = _validate_bbox(bbox_min, bbox_max)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must have shape (n_points, 3); got {points.shape}.")
    extent = bbox_max - bbox_min
    padding = extent * 0.1
    lower, upper = bbox_min - padding, bbox_max + padding
    span = np.where((upper - lower) > 1e-12, upper - lower, 1.0)
    normalized = 2.0 * (points - lower) / span - 1.0
    distance_scale = max(float(np.max(span)) / 2.0, 1e-12)
    return normalized[:, [2, 1, 0]].astype(np.float32), distance_scale


def _validate_bbox(
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    minimum = np.asarray(bbox_min, dtype=np.float64)
    maximum = np.asarray(bbox_max, dtype=np.float64)
    if minimum.shape != (3,) or maximum.shape != (3,):
        raise ValueError("bbox_min and bbox_max must each contain three coordinates.")
    if not np.isfinite(minimum).all() or not np.isfinite(maximum).all():
        raise ValueError("Bounding-box coordinates must be finite.")
    if np.any(maximum < minimum):
        raise ValueError("bbox_max must be greater than or equal to bbox_min.")
    return minimum, maximum


__all__ = [
    "TRIMESH_AVAILABLE",
    "compute_sdf",
    "make_background_grid",
    "normalize_grid_coordinates",
]
