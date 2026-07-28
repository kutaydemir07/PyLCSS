# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.


"""Mesh-boundary queries used by Design Studio selection nodes."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .selection_payloads import _selection_payload


_MESH_FACE_DIRECTIONS = ("<X", ">X", "<Y", ">Y", "<Z", ">Z")
_MESH_COMPONENT_INDEX_BASE = 1000
_MESH_COMPONENT_INDEX_STRIDE = 1000
_MESH_FEATURE_INDEX_BASE = 100000
_MESH_FEATURE_ANGLE_DEG = 35.0


def _mesh_component_stored_index(
    direction_index: int,
    component_index: int,
) -> int:
    return (
        _MESH_COMPONENT_INDEX_BASE
        + int(direction_index) * _MESH_COMPONENT_INDEX_STRIDE
        + int(component_index)
    )


def _mesh_points(mesh_like: Any) -> np.ndarray | None:
    """Return mesh node coordinates as a (3, n) array for mesh-backed selection."""
    try:
        import numpy as np

        if isinstance(mesh_like, dict):
            if mesh_like.get("mesh") is not None:
                return _mesh_points(mesh_like.get("mesh"))
            if mesh_like.get("vertices") is not None:
                pts = np.asarray(mesh_like.get("vertices"), dtype=float)
                if pts.ndim == 2 and pts.shape[1] >= 3 and len(pts) > 0:
                    return pts[:, :3].T
            return None
        if hasattr(mesh_like, "p"):
            pts = np.asarray(mesh_like.p, dtype=float)
            if pts.ndim == 2 and pts.shape[0] >= 3 and pts.shape[1] > 0:
                return pts[:3, :]
    except Exception:
        return None
    return None


def _mesh_selection_payload(
    mesh_like: Any,
    node_ids: object,
    selector_type: str,
    label: str,
    surface_faces: object | None = None,
    entity_type: str = "Face",
) -> dict[str, Any] | None:
    """Build a SelectFace-compatible payload for selected mesh nodes."""
    import numpy as np

    p = _mesh_points(mesh_like)
    if p is None:
        return None
    ids = np.asarray(node_ids, dtype=int)
    ids = ids[(ids >= 0) & (ids < p.shape[1])]
    ids = np.unique(ids)
    if ids.size == 0:
        return None

    pts = p[:, ids].T
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    center = 0.5 * (mins + maxs)
    centroid = pts.mean(axis=0)
    selection = {
        "mesh_selection": True,
        "selector_type": selector_type,
        "label": label,
        "node_ids": [int(v) for v in ids.tolist()],
        "node_count": int(ids.size),
        "center": [float(v) for v in center.tolist()],
        "centroid": [float(v) for v in centroid.tolist()],
        "points": [[float(x) for x in center.tolist()]],
        "bbox": {
            "xmin": float(mins[0]),
            "xmax": float(maxs[0]),
            "ymin": float(mins[1]),
            "ymax": float(maxs[1]),
            "zmin": float(mins[2]),
            "zmax": float(maxs[2]),
        },
    }

    if surface_faces is not None:
        try:
            face_arr = np.asarray(surface_faces, dtype=int)
            if face_arr.ndim == 2 and face_arr.shape[1] >= 3 and face_arr.size:
                face_arr = face_arr[:, :3]
                valid = np.all((face_arr >= 0) & (face_arr < p.shape[1]), axis=1)
                face_arr = face_arr[valid]
                if face_arr.size:
                    tri_pts = p[:, face_arr].transpose(1, 2, 0)
                    tri_centers = tri_pts.mean(axis=1)
                    tri_areas = 0.5 * np.linalg.norm(
                        np.cross(
                            tri_pts[:, 1, :] - tri_pts[:, 0, :],
                            tri_pts[:, 2, :] - tri_pts[:, 0, :],
                        ),
                        axis=1,
                    )
                    order = np.argsort(tri_areas)[::-1]
                    diag = float(np.linalg.norm(maxs - mins))
                    min_spacing = max(0.25, 0.12 * diag)
                    sample_points = []
                    for tri_idx in order:
                        point = np.asarray(tri_centers[int(tri_idx)], dtype=float)
                        if all(
                            np.linalg.norm(point - prev) >= min_spacing
                            for prev in sample_points
                        ):
                            sample_points.append(point)
                        if len(sample_points) >= 6:
                            break
                    if sample_points:
                        selection["points"] = [
                            [float(v) for v in point.tolist()]
                            for point in sample_points
                        ]
                    unique_nodes, inverse = np.unique(
                        face_arr.reshape(-1), return_inverse=True
                    )
                    vertices = p[:, unique_nodes].T
                    triangles = inverse.reshape((-1, 3))
                    selection["surface_node_ids"] = [
                        int(v) for v in unique_nodes.tolist()
                    ]
                    selection["surface_vertices"] = [
                        [float(x), float(y), float(z)] for x, y, z in vertices.tolist()
                    ]
                    selection["surface_triangles"] = [
                        [int(a), int(b), int(c)] for a, b, c in triangles.tolist()
                    ]
        except Exception:
            pass
    selection["entity_type"] = str(entity_type or "Face").title()
    return _selection_payload(None, [selection], selector_type, entity_type=entity_type)


def _mesh_axis_tolerance(p: np.ndarray) -> float:
    import numpy as np

    spans = np.max(p, axis=1) - np.min(p, axis=1)
    positive = spans[spans > 1e-9]
    ref = float(np.max(positive)) if positive.size else 1.0
    return max(1e-6, 0.0025 * ref)


def _mesh_tetrahedra(mesh_like: Any) -> np.ndarray | None:
    """Return tetra connectivity as a (4, n) array when available."""
    try:
        import numpy as np

        if isinstance(mesh_like, dict):
            if mesh_like.get("mesh") is not None:
                return _mesh_tetrahedra(mesh_like.get("mesh"))
            for key in ("t", "elements", "tetrahedra", "cells"):
                if mesh_like.get(key) is None:
                    continue
                arr = np.asarray(mesh_like.get(key), dtype=int)
                if arr.ndim != 2:
                    continue
                if key == "t" and arr.shape[0] in (4, 10):
                    return arr[:4, :]
                if key != "t" and arr.shape[1] in (4, 10):
                    return arr[:, :4].T
                if key != "t" and arr.shape[0] in (4, 10):
                    return arr[:4, :]
        if hasattr(mesh_like, "t"):
            arr = np.asarray(mesh_like.t, dtype=int)
            # scikit-fem and PyLCSS mesh wrappers store connectivity as
            # (nodes_per_element, element_count). A 3 x N shell mesh must not
            # be reinterpreted as N x 3 tetrahedra merely because N >= 4.
            if arr.ndim == 2 and arr.shape[0] in (4, 10):
                return arr[:4, :]
    except Exception:
        return None
    return None


def _mesh_surface_triangles(mesh_like: Any) -> np.ndarray | None:
    """Return explicit shell/surface connectivity as an ``(n, 3)`` array."""
    try:
        import numpy as np

        if isinstance(mesh_like, dict):
            if mesh_like.get("mesh") is not None:
                return _mesh_surface_triangles(mesh_like.get("mesh"))
            for key in ("faces", "triangles"):
                value = mesh_like.get(key)
                if value is None:
                    continue
                arr = np.asarray(value, dtype=int)
                # Face dictionaries normally use row-major
                # ``(face_count, nodes_per_face)`` connectivity.  Accept the
                # transposed ``(3, face_count)`` form when unambiguous.
                if arr.ndim == 2 and arr.shape[0] == 3 and arr.shape[1] != 3:
                    return arr.T
                if arr.ndim == 2 and arr.shape[1] >= 3:
                    return arr[:, :3]
            value = mesh_like.get("t")
            if value is not None:
                arr = np.asarray(value, dtype=int)
                if arr.ndim == 2 and arr.shape[0] == 3:
                    return arr.T
        if hasattr(mesh_like, "t"):
            arr = np.asarray(mesh_like.t, dtype=int)
            if arr.ndim == 2 and arr.shape[0] == 3:
                return arr.T
    except Exception:
        return None
    return None


def _mesh_boundary_face_data(mesh_like: Any) -> dict[str, Any] | None:
    """Extract exterior triangle faces, centers, normals and an edge scale."""
    try:
        import numpy as np

        p = _mesh_points(mesh_like)
        t = _mesh_tetrahedra(mesh_like)
        surface = _mesh_surface_triangles(mesh_like)
        if p is None or (
            (t is None or t.size == 0) and (surface is None or surface.size == 0)
        ):
            return None

        pts = p.T
        n_nodes = p.shape[1]
        if t is not None and t.size:
            t = t[:, np.all((t >= 0) & (t < n_nodes), axis=0)]
            face_patterns = ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3))
            face_counts = {}
            face_nodes = {}
            for elem in t.T:
                for pattern in face_patterns:
                    face = tuple(int(elem[i]) for i in pattern)
                    key = tuple(sorted(face))
                    face_counts[key] = face_counts.get(key, 0) + 1
                    face_nodes[key] = face
            boundary = [
                face_nodes[key] for key, count in face_counts.items() if count == 1
            ]
        else:
            surface = np.asarray(surface, dtype=int)
            valid_surface = np.all((surface >= 0) & (surface < n_nodes), axis=1)
            boundary = surface[valid_surface].tolist()
        if not boundary:
            return None

        faces = np.asarray(boundary, dtype=int)
        tri = pts[faces]
        centers = tri.mean(axis=1)
        normals = np.cross(tri[:, 1, :] - tri[:, 0, :], tri[:, 2, :] - tri[:, 0, :])
        areas2 = np.linalg.norm(normals, axis=1)
        valid = areas2 > 1e-12
        if not np.any(valid):
            return None
        tri = tri[valid]
        faces = faces[valid]
        centers = centers[valid]
        normals = normals[valid]
        areas2 = areas2[valid]
        normals = normals / areas2[:, None]

        # Persisted virtual-face patch ids are created by the VTK viewer and
        # resolved again here during graph execution. Use the same bounds
        # centre in both places so non-uniform meshes do not flip a clicked
        # patch to the opposite direction between picking and solving.
        body_center = 0.5 * (pts.min(axis=0) + pts.max(axis=0))
        inward = np.einsum("ij,ij->i", normals, centers - body_center) < 0.0
        normals[inward] *= -1.0

        edge_lengths = np.concatenate(
            [
                np.linalg.norm(tri[:, 1, :] - tri[:, 0, :], axis=1),
                np.linalg.norm(tri[:, 2, :] - tri[:, 1, :], axis=1),
                np.linalg.norm(tri[:, 0, :] - tri[:, 2, :], axis=1),
            ]
        )
        edge_lengths = edge_lengths[edge_lengths > 1e-9]
        edge_scale = (
            float(np.median(edge_lengths))
            if edge_lengths.size
            else _mesh_axis_tolerance(p)
        )

        return {
            "faces": faces,
            "centers": centers,
            "normals": normals,
            "areas": 0.5 * areas2,
            "edge_scale": edge_scale,
        }
    except Exception:
        return None


def _mesh_direction_selection(
    mesh_like: Any,
    selector: str,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Select exterior nodes and surface triangles for a directional mesh face."""
    import numpy as np

    p = _mesh_points(mesh_like)
    if p is None:
        return np.array([], dtype=int), None

    axis = {"X": 0, "Y": 1, "Z": 2}.get(selector[-1], 2)
    sign = -1.0 if selector.startswith("<") else 1.0
    data = _mesh_boundary_face_data(mesh_like)
    if data is None:
        tol = _mesh_axis_tolerance(p)
        limit = float(np.min(p[axis]) if sign < 0.0 else np.max(p[axis]))
        if sign < 0.0:
            return np.where(p[axis] <= limit + tol)[0].astype(int), None
        return np.where(p[axis] >= limit - tol)[0].astype(int), None

    centers = data["centers"]
    normals = data["normals"]
    faces = data["faces"]
    axis_span = float(np.max(p[axis]) - np.min(p[axis]))
    edge_scale = float(data.get("edge_scale") or 0.0)
    tol = max(
        1e-6,
        0.005 * axis_span,
        min(3.0 * edge_scale, 0.08 * axis_span)
        if axis_span > 1e-9
        else 3.0 * edge_scale,
    )
    direction = np.zeros(3, dtype=float)
    direction[axis] = sign

    normal_mask = np.dot(normals, direction) >= 0.30
    if np.any(normal_mask):
        candidate_centers = centers[normal_mask]
        extreme = float(
            np.min(candidate_centers[:, axis])
            if sign < 0.0
            else np.max(candidate_centers[:, axis])
        )
        if sign < 0.0:
            near_mask = centers[:, axis] <= extreme + tol
        else:
            near_mask = centers[:, axis] >= extreme - tol
        mask = normal_mask & near_mask
    else:
        extreme = float(
            np.min(centers[:, axis]) if sign < 0.0 else np.max(centers[:, axis])
        )
        if sign < 0.0:
            mask = centers[:, axis] <= extreme + tol
        else:
            mask = centers[:, axis] >= extreme - tol

    surface_faces = faces[mask] if np.any(mask) else np.empty((0, 3), dtype=int)
    ids = (
        np.unique(surface_faces.reshape(-1))
        if surface_faces.size
        else np.array([], dtype=int)
    )
    if ids.size == 0:
        tol = _mesh_axis_tolerance(p)
        limit = float(np.min(p[axis]) if sign < 0.0 else np.max(p[axis]))
        if sign < 0.0:
            ids = np.where(p[axis] <= limit + tol)[0]
        else:
            ids = np.where(p[axis] >= limit - tol)[0]
        surface_faces = None
    return ids.astype(int), surface_faces


def _split_surface_face_components(
    surface_faces: object,
) -> list[np.ndarray]:
    """Split selected exterior triangles into connected surface patches."""
    import numpy as np

    face_arr = np.asarray(surface_faces, dtype=int)
    if face_arr.ndim != 2 or face_arr.shape[1] < 3 or face_arr.size == 0:
        return []
    face_arr = face_arr[:, :3]

    edge_to_faces = {}
    for face_idx, tri in enumerate(face_arr):
        a, b, c = [int(v) for v in tri[:3]]
        for edge in ((a, b), (b, c), (c, a)):
            key = tuple(sorted(edge))
            edge_to_faces.setdefault(key, []).append(face_idx)

    adjacency = [set() for _ in range(len(face_arr))]
    for owners in edge_to_faces.values():
        if len(owners) < 2:
            continue
        for owner in owners:
            adjacency[owner].update(v for v in owners if v != owner)

    components = []
    visited = np.zeros(len(face_arr), dtype=bool)
    for start in range(len(face_arr)):
        if visited[start]:
            continue
        stack = [start]
        visited[start] = True
        current = []
        while stack:
            idx = stack.pop()
            current.append(idx)
            for nxt in adjacency[idx]:
                if not visited[nxt]:
                    visited[nxt] = True
                    stack.append(nxt)
        components.append(face_arr[np.asarray(current, dtype=int)])
    return components


def _surface_component_metrics(
    mesh_like: Any,
    surface_faces: object,
) -> tuple[float, np.ndarray]:
    import numpy as np

    p = _mesh_points(mesh_like)
    face_arr = np.asarray(surface_faces, dtype=int)
    if p is None or face_arr.ndim != 2 or face_arr.shape[1] < 3 or face_arr.size == 0:
        return 0.0, np.zeros(3, dtype=float)
    face_arr = face_arr[:, :3]
    valid = np.all((face_arr >= 0) & (face_arr < p.shape[1]), axis=1)
    face_arr = face_arr[valid]
    if face_arr.size == 0:
        return 0.0, np.zeros(3, dtype=float)
    tri_pts = p[:, face_arr].transpose(1, 2, 0)
    areas = 0.5 * np.linalg.norm(
        np.cross(
            tri_pts[:, 1, :] - tri_pts[:, 0, :], tri_pts[:, 2, :] - tri_pts[:, 0, :]
        ),
        axis=1,
    )
    total_area = float(np.sum(areas))
    if total_area > 1e-12:
        center = np.sum(tri_pts.mean(axis=1) * areas[:, None], axis=0) / total_area
    else:
        center = tri_pts.reshape((-1, 3)).mean(axis=0)
    return total_area, center


def _feature_surface_component_indices(
    points: np.ndarray,
    faces: np.ndarray,
    normals: np.ndarray,
    centers: np.ndarray,
    areas: np.ndarray,
    feature_angle_deg: float = _MESH_FEATURE_ANGLE_DEG,
) -> list[np.ndarray]:
    """Split a triangulated surface into smooth, feature-bounded regions."""
    points = np.asarray(points, dtype=float)
    faces = np.asarray(faces, dtype=int)
    normals = np.asarray(normals, dtype=float)
    centers = np.asarray(centers, dtype=float)
    areas = np.asarray(areas, dtype=float)
    if (
        points.ndim != 2
        or points.shape[1] < 3
        or faces.ndim != 2
        or faces.shape[1] < 3
        or len(faces) != len(normals)
    ):
        return []
    faces = faces[:, :3]
    span = float(np.max(np.ptp(points[:, :3], axis=0))) if len(points) else 1.0
    weld_tol = max(1e-9, span * 1e-8)
    quantized = np.rint(points[:, :3] / weld_tol).astype(np.int64)

    # Coordinate-based keys also weld STL triangles that duplicate vertices.
    edge_to_faces = {}
    for face_idx, tri in enumerate(faces):
        vertices = [tuple(int(v) for v in quantized[int(pid)]) for pid in tri]
        for a, b in (
            (vertices[0], vertices[1]),
            (vertices[1], vertices[2]),
            (vertices[2], vertices[0]),
        ):
            edge_to_faces.setdefault(tuple(sorted((a, b))), []).append(face_idx)

    adjacency = [set() for _ in range(len(faces))]
    min_dot = math.cos(math.radians(float(feature_angle_deg)))
    for owners in edge_to_faces.values():
        for i, owner in enumerate(owners):
            for other in owners[i + 1 :]:
                if float(np.dot(normals[owner], normals[other])) >= min_dot:
                    adjacency[owner].add(other)
                    adjacency[other].add(owner)

    components = []
    visited = np.zeros(len(faces), dtype=bool)
    for start in range(len(faces)):
        if visited[start]:
            continue
        stack = [start]
        visited[start] = True
        current_component = []
        while stack:
            current = stack.pop()
            current_component.append(current)
            for nxt in adjacency[current]:
                if not visited[nxt]:
                    visited[nxt] = True
                    stack.append(nxt)
        component = np.asarray(current_component, dtype=int)
        area = float(np.sum(areas[component]))
        if area <= 1e-12:
            continue
        center = np.sum(centers[component] * areas[component, None], axis=0) / area
        components.append((area, center, component))

    components.sort(
        key=lambda item: (
            -float(item[0]),
            float(item[1][0]),
            float(item[1][1]),
            float(item[1][2]),
        )
    )
    return [component for _area, _center, component in components]


def _mesh_feature_components(mesh_like: Any) -> list[np.ndarray]:
    """Return cylindrical, conical, planar, and freeform surface regions."""
    data = _mesh_boundary_face_data(mesh_like)
    points = _mesh_points(mesh_like)
    if data is None or points is None:
        return []
    indices = _feature_surface_component_indices(
        points.T,
        data["faces"],
        data["normals"],
        data["centers"],
        data["areas"],
    )
    return [data["faces"][component] for component in indices]


def _mesh_direction_components(
    mesh_like: Any,
    selector: str,
) -> list[np.ndarray]:
    """Return connected patch components for one virtual mesh direction."""
    import numpy as np

    _ids, surface_faces = _mesh_direction_selection(mesh_like, selector)
    if surface_faces is None or len(surface_faces) == 0:
        return []

    components = _split_surface_face_components(surface_faces)
    decorated = []
    for component in components:
        area, center = _surface_component_metrics(mesh_like, component)
        if area <= 1e-12:
            continue
        decorated.append((area, center, np.asarray(component, dtype=int)))
    decorated.sort(
        key=lambda item: (
            -float(item[0]),
            float(item[1][0]),
            float(item[1][1]),
            float(item[1][2]),
        )
    )
    return [component for _area, _center, component in decorated]


def _mesh_component_selection(
    mesh_like: Any,
    stored_index: int,
) -> tuple[str | None, np.ndarray, np.ndarray | None]:
    """Resolve a stored connected-patch index to nodes and exterior triangles."""
    import numpy as np

    stored_index = int(stored_index)
    if stored_index >= _MESH_FEATURE_INDEX_BASE:
        component_index = stored_index - _MESH_FEATURE_INDEX_BASE
        components = _mesh_feature_components(mesh_like)
        if component_index < 0 or component_index >= len(components):
            return "Feature", np.array([], dtype=int), None
        surface_faces = np.asarray(components[component_index], dtype=int)
        ids = np.unique(surface_faces.reshape(-1))
        return "Feature", ids.astype(int), surface_faces

    local = stored_index - _MESH_COMPONENT_INDEX_BASE
    if local < 0:
        return None, np.array([], dtype=int), None
    direction_index = local // _MESH_COMPONENT_INDEX_STRIDE
    component_index = local % _MESH_COMPONENT_INDEX_STRIDE
    if direction_index < 0 or direction_index >= len(_MESH_FACE_DIRECTIONS):
        return None, np.array([], dtype=int), None

    selector = _MESH_FACE_DIRECTIONS[int(direction_index)]
    components = _mesh_direction_components(mesh_like, selector)
    if component_index < 0 or component_index >= len(components):
        return selector, np.array([], dtype=int), None

    surface_faces = components[int(component_index)]
    ids = (
        np.unique(surface_faces.reshape(-1))
        if surface_faces.size
        else np.array([], dtype=int)
    )
    return selector, ids.astype(int), surface_faces
