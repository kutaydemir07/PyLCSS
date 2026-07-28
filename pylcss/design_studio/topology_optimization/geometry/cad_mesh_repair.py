# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Repair and validate recovered triangle meshes before STEP conversion."""

from __future__ import annotations

import logging
from collections import defaultdict, deque
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


def _extract_recovered_mesh(payload: Any) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """Return ``(vertices, triangle_faces)`` from a topology result payload."""
    if not isinstance(payload, dict):
        return None

    mesh = payload.get("recovered_shape")
    if mesh is None and "vertices" in payload and "faces" in payload:
        mesh = payload
    if not isinstance(mesh, dict):
        return None

    vertices = mesh.get("vertices")
    faces = mesh.get("faces")
    if vertices is None or faces is None:
        return None

    vertices = np.asarray(vertices, dtype=float)
    faces = np.asarray(faces, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] < 3:
        return None
    if faces.ndim != 2 or faces.shape[1] < 3:
        return None
    if len(vertices) < 4 or len(faces) < 4:
        return None
    return vertices[:, :3], faces[:, :3]


def _drop_degenerate_faces(
    vertices: np.ndarray,
    faces: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Remove invalid, zero-area, repeated-index, and duplicate triangles."""
    vertices = np.asarray(vertices, dtype=float)
    if vertices.ndim != 2 or vertices.shape[1] < 3:
        return np.empty((0, 3), dtype=float), np.empty((0, 3), dtype=np.int64)
    vertices = vertices[:, :3]

    faces = np.asarray(faces, dtype=np.int64)
    if faces.size == 0:
        return vertices, faces.reshape((0, 3))
    if faces.ndim != 2 or faces.shape[1] < 3:
        return vertices, np.empty((0, 3), dtype=np.int64)
    faces = faces[:, :3]

    in_range = np.all((faces >= 0) & (faces < len(vertices)), axis=1)
    faces = faces[in_range]
    if len(faces) == 0:
        return vertices, faces.reshape((0, 3))

    keep = (
        (faces[:, 0] != faces[:, 1])
        & (faces[:, 1] != faces[:, 2])
        & (faces[:, 0] != faces[:, 2])
    )
    faces = faces[keep]
    if len(faces) == 0:
        return vertices, faces.reshape((0, 3))

    tri = vertices[faces]
    finite = np.all(np.isfinite(tri), axis=(1, 2))
    tri = tri[finite]
    faces = faces[finite]
    if len(faces) == 0:
        return vertices, faces.reshape((0, 3))

    areas = 0.5 * np.linalg.norm(
        np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]),
        axis=1,
    )
    faces = faces[areas > 1e-12]
    if len(faces) == 0:
        return vertices, faces.reshape((0, 3))

    keys = np.sort(faces, axis=1)
    _, unique_idx = np.unique(keys, axis=0, return_index=True)
    unique_idx.sort()
    return vertices, faces[unique_idx]


def _bbox_diag(vertices: np.ndarray) -> float:
    """Return the model bounding-box diagonal, clamped to a non-zero value."""
    vertices = np.asarray(vertices, dtype=float)
    if vertices.size == 0:
        return 1.0
    lo = np.nanmin(vertices[:, :3], axis=0)
    hi = np.nanmax(vertices[:, :3], axis=0)
    diag = float(np.linalg.norm(hi - lo))
    return diag if np.isfinite(diag) and diag > 1e-12 else 1.0


def _effective_tolerance(
    vertices: np.ndarray,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> float:
    """Combine user absolute tolerance with model-size-relative tolerance."""
    abs_tol = float(absolute_tolerance or 0.0)
    rel_tol = float(relative_tolerance or 0.0) * _bbox_diag(vertices)
    return max(abs_tol, rel_tol, 1e-9)


def _merge_nearby_vertices(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    tolerance: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Weld vertices that land in the same tolerance grid cell.

    This is deliberately dependency-free.  It is not a full geometric remesher,
    but it fixes the common recovered-mesh problem where adjacent triangles use
    numerically distinct vertices along an otherwise shared edge.
    """
    vertices = np.asarray(vertices, dtype=float)[:, :3]
    faces = np.asarray(faces, dtype=np.int64)[:, :3]
    tol = float(tolerance or 0.0)
    if tol <= 0.0 or len(vertices) == 0 or len(faces) == 0:
        return vertices, faces

    keys = np.round(vertices / tol).astype(np.int64)
    key_to_new: dict[tuple[int, int, int], int] = {}
    inverse = np.empty(len(vertices), dtype=np.int64)
    sums: list[np.ndarray] = []
    counts: list[int] = []

    for old_idx, key_arr in enumerate(keys):
        key = (int(key_arr[0]), int(key_arr[1]), int(key_arr[2]))
        new_idx = key_to_new.get(key)
        if new_idx is None:
            new_idx = len(sums)
            key_to_new[key] = new_idx
            sums.append(vertices[old_idx].copy())
            counts.append(1)
        else:
            sums[new_idx] += vertices[old_idx]
            counts[new_idx] += 1
        inverse[old_idx] = new_idx

    welded = np.vstack(
        [total / max(count, 1) for total, count in zip(sums, counts, strict=True)]
    ).astype(float)
    remapped_faces = inverse[faces]
    welded, remapped_faces = _drop_degenerate_faces(welded, remapped_faces)
    return welded, remapped_faces


def _edge_occurrences(
    faces: np.ndarray,
) -> dict[tuple[int, int], list[tuple[int, int, int]]]:
    """Map undirected edges to ``(face_index, oriented_start, oriented_end)``."""
    occ: dict[tuple[int, int], list[tuple[int, int, int]]] = defaultdict(list)
    for fi, tri in enumerate(np.asarray(faces, dtype=np.int64)[:, :3]):
        a, b, c = [int(v) for v in tri]
        for u, v in ((a, b), (b, c), (c, a)):
            occ[(min(u, v), max(u, v))].append((fi, u, v))
    return occ


def _find_boundary_edges(faces: np.ndarray) -> list[tuple[int, int]]:
    """Return mesh edges used by exactly one triangle."""
    return [edge for edge, items in _edge_occurrences(faces).items() if len(items) == 1]


def _find_nonmanifold_edges(faces: np.ndarray) -> list[tuple[int, int]]:
    """Return mesh edges used by more than two triangles."""
    return [edge for edge, items in _edge_occurrences(faces).items() if len(items) > 2]


def _connected_face_components(faces: np.ndarray) -> list[np.ndarray]:
    """Return connected components of faces using shared triangle edges."""
    faces = np.asarray(faces, dtype=np.int64)[:, :3]
    if len(faces) == 0:
        return []

    adjacency: list[set[int]] = [set() for _ in range(len(faces))]
    for items in _edge_occurrences(faces).values():
        if len(items) < 2:
            continue
        face_ids = [fi for fi, _, _ in items]
        for i, fi in enumerate(face_ids):
            adjacency[fi].update(face_ids[:i])
            adjacency[fi].update(face_ids[i + 1 :])

    seen = np.zeros(len(faces), dtype=bool)
    components: list[np.ndarray] = []
    for start in range(len(faces)):
        if seen[start]:
            continue
        queue: deque[int] = deque([start])
        seen[start] = True
        comp: list[int] = []
        while queue:
            fi = queue.popleft()
            comp.append(fi)
            for nxt in adjacency[fi]:
                if not seen[nxt]:
                    seen[nxt] = True
                    queue.append(nxt)
        components.append(np.asarray(comp, dtype=np.int64))
    return components


def _remove_tiny_components(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    min_faces: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    """Drop tiny floating triangle islands while preserving all meaningful bodies."""
    faces = np.asarray(faces, dtype=np.int64)[:, :3]
    components = _connected_face_components(faces)
    if len(components) <= 1:
        return vertices, faces

    min_faces = int(max(min_faces, 1))
    keep_components = [comp for comp in components if len(comp) >= min_faces]
    if not keep_components:
        keep_components = [max(components, key=len)]

    keep_face_ids = np.concatenate(keep_components)
    keep_face_ids.sort()
    logger.info(
        "Recovered Shape mesh repair: kept %d/%d connected component(s).",
        len(keep_components),
        len(components),
    )
    return vertices, faces[keep_face_ids]


def _orient_faces_consistently(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Orient adjacent triangles consistently and make closed volumes outward."""
    vertices = np.asarray(vertices, dtype=float)[:, :3]
    faces = np.asarray(faces, dtype=np.int64)[:, :3].copy()
    if len(faces) == 0:
        return faces

    adjacency: list[list[tuple[int, bool]]] = [[] for _ in range(len(faces))]
    for items in _edge_occurrences(faces).values():
        if len(items) != 2:
            continue
        (f0, a0, b0), (f1, a1, b1) = items
        # For a consistently oriented manifold, neighbouring faces traverse the
        # shared edge in opposite directions.  Same direction means one side has
        # to be flipped relative to the other.
        same_direction = a0 == a1 and b0 == b1
        adjacency[f0].append((f1, same_direction))
        adjacency[f1].append((f0, same_direction))

    visited = np.zeros(len(faces), dtype=bool)
    flip = np.zeros(len(faces), dtype=bool)
    for start in range(len(faces)):
        if visited[start]:
            continue
        visited[start] = True
        queue: deque[int] = deque([start])
        while queue:
            fi = queue.popleft()
            for nb, same_direction in adjacency[fi]:
                desired_flip = flip[fi] ^ bool(same_direction)
                if not visited[nb]:
                    flip[nb] = desired_flip
                    visited[nb] = True
                    queue.append(nb)

    faces[flip, 1:] = faces[flip, 1:][:, ::-1]

    # If closed, use signed volume to make winding outward.  Positive signed
    # volume corresponds to outward orientation for right-handed coordinates.
    tri = vertices[faces]
    signed_volume = float(
        np.sum(np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2]))) / 6.0
    )
    if np.isfinite(signed_volume) and signed_volume < 0.0:
        faces[:, [1, 2]] = faces[:, [2, 1]]
    return faces


def _mesh_repair_and_validate(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    weld_tolerance: float,
    min_component_faces: int = 8,
    validate_watertight: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Repair common recovered mesh issues before OCC sewing."""
    vertices, faces = _drop_degenerate_faces(vertices, faces)
    if len(vertices) < 4 or len(faces) < 4:
        raise RuntimeError(
            "Recovered Shape has too few valid triangles for STEP sewing."
        )

    vertices, faces = _merge_nearby_vertices(
        vertices, faces, tolerance=float(weld_tolerance) * 0.25
    )
    vertices, faces = _remove_tiny_components(
        vertices, faces, min_faces=min_component_faces
    )
    faces = _orient_faces_consistently(vertices, faces)
    vertices, faces = _drop_degenerate_faces(vertices, faces)

    boundary_edges = _find_boundary_edges(faces)
    nonmanifold_edges = _find_nonmanifold_edges(faces)
    if validate_watertight and (boundary_edges or nonmanifold_edges):
        raise RuntimeError(
            "Recovered Shape mesh is not a watertight 2-manifold "
            f"({len(boundary_edges)} boundary edge(s), "
            f"{len(nonmanifold_edges)} non-manifold edge(s))."
        )

    if boundary_edges or nonmanifold_edges:
        logger.warning(
            "Recovered Shape mesh has %d boundary edge(s) and %d non-manifold edge(s); "
            "continuing because validate_watertight=False.",
            len(boundary_edges),
            len(nonmanifold_edges),
        )
    return vertices, faces


def _simplify_mesh_for_step(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    max_faces: Optional[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Bound OCC sewing cost while preserving a closed surface and volume."""
    limit = int(max_faces or 0)
    if limit <= 0 or len(faces) <= limit:
        return vertices, faces
    try:
        import trimesh

        source = trimesh.Trimesh(
            vertices=np.asarray(vertices, dtype=float),
            faces=np.asarray(faces, dtype=np.int64),
            process=False,
        )
        source_volume = abs(float(source.volume)) if source.is_watertight else None
        simplified = source.simplify_quadric_decimation(face_count=limit)
        if simplified is None or len(simplified.faces) < 4:
            raise RuntimeError("quadric simplifier returned no usable surface")
        if source.is_watertight and not simplified.is_watertight:
            raise RuntimeError("quadric simplification opened the closed surface")
        if source_volume and source_volume > 1e-12:
            volume_error = (
                abs(abs(float(simplified.volume)) - source_volume) / source_volume
            )
            if not np.isfinite(volume_error) or volume_error > 0.05:
                raise RuntimeError(
                    f"simplification changed enclosed volume by {volume_error:.1%}"
                )
        logger.info(
            "Recovered Shape STEP: simplified %d to %d triangles before OCC sewing.",
            len(faces),
            len(simplified.faces),
        )
        return (
            np.asarray(simplified.vertices, dtype=float),
            np.asarray(simplified.faces, dtype=np.int64),
        )
    except Exception:
        logger.warning(
            "Recovered Shape STEP simplification was rejected; sewing the "
            "validated original surface.",
            exc_info=True,
        )
        return vertices, faces
