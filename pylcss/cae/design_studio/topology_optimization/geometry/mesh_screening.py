# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE.
"""Bounded geometric screening for intersections of non-adjacent triangles."""

from __future__ import annotations

import numpy as np


def _intersecting_triangles(first: np.ndarray, second: np.ndarray, tolerance: float) -> np.ndarray:
    """Separating-axis test, including axes needed for coplanar triangles."""
    edges_a = np.roll(first, -1, axis=1) - first
    edges_b = np.roll(second, -1, axis=1) - second
    normal_a = np.cross(edges_a[:, 0], edges_a[:, 1])
    normal_b = np.cross(edges_b[:, 0], edges_b[:, 1])
    axes = np.concatenate((
        normal_a[:, None], normal_b[:, None],
        np.cross(edges_a[:, :, None], edges_b[:, None, :]).reshape(-1, 9, 3),
        np.cross(normal_a[:, None], edges_a),
        np.cross(normal_b[:, None], edges_b),
    ), axis=1)
    lengths = np.linalg.norm(axes, axis=2)
    valid = lengths > np.finfo(float).eps * np.max(lengths, axis=1, keepdims=True)
    axes = np.divide(axes, lengths[:, :, None], out=np.zeros_like(axes), where=lengths[:, :, None] > 0)
    # Subtract a common origin to avoid cancellation for translated geometry.
    origin = first[:, :1]
    projection_a = np.einsum("nvc,nac->nav", first - origin, axes)
    projection_b = np.einsum("nvc,nac->nav", second - origin, axes)
    separated = ((projection_a.max(axis=2) < projection_b.min(axis=2) - tolerance)
                 | (projection_b.max(axis=2) < projection_a.min(axis=2) - tolerance)) & valid
    return ~np.any(separated, axis=1)


def screen_mesh_intersections(vertices: np.ndarray, faces: np.ndarray, *,
                              max_faces: int = 100_000, max_pairs: int = 1_000_000) -> dict:
    """Check all spatially overlapping, topologically non-adjacent faces.

    Shared-vertex neighbours are excluded; closure, winding and degenerate
    faces are checked separately. An exhausted budget reports an incomplete
    check rather than a false zero or a missing-dependency diagnosis.
    """
    report = {"self_intersecting_faces": None, "screening_scope": "not_run",
              "screening_method": "Non-adjacent triangle separating-axis test",
              "screened_candidate_pairs": 0}
    vertices, faces = np.asarray(vertices, dtype=float), np.asarray(faces, dtype=np.int64)
    if len(faces) > max_faces:
        report.update(screening_scope="face_budget", screening_reason=f"the surface exceeds the {max_faces:,}-face screening budget")
        return report
    if not len(faces):
        report.update(self_intersecting_faces=0, screening_scope="full")
        return report
    triangles = vertices[faces]
    tolerance = max(float(np.max(np.ptp(vertices, axis=0))) * 1e-10, 1e-12)
    lower, upper = triangles.min(axis=1) - tolerance, triangles.max(axis=1) + tolerance
    bounds = np.column_stack((lower, upper))
    try:
        from trimesh.util import bounds_tree
        tree = bounds_tree(bounds)
    except ImportError as exc:
        report.update(screening_scope="backend_unavailable", screening_reason=f"the spatial-index dependency could not be imported ({exc})")
        return report
    except Exception as exc:
        report.update(screening_scope="backend_error", screening_reason=f"the spatial index could not be built ({exc})")
        return report

    intersected = set()
    try:
        for index, bbox in enumerate(bounds):
            candidates = np.asarray([other for other in tree.intersection(bbox) if other > index], dtype=int)
            if not len(candidates):
                continue
            adjacent = np.any(faces[candidates, :, None] == faces[index, None, :], axis=(1, 2))
            candidates = candidates[~adjacent]
            if not len(candidates):
                continue
            if report["screened_candidate_pairs"] + len(candidates) > max_pairs:
                report.update(screening_scope="pair_budget", screening_reason=f"the check exceeds the {max_pairs:,}-pair screening budget")
                report["self_intersecting_faces"] = len(intersected) or None
                return report
            report["screened_candidate_pairs"] += len(candidates)
            first = np.broadcast_to(triangles[index], (len(candidates), 3, 3))
            hits = candidates[_intersecting_triangles(first, triangles[candidates], tolerance)]
            if len(hits):
                intersected.add(index)
                intersected.update(hits.tolist())
    except Exception as exc:
        report.update(screening_scope="screening_error", screening_reason=f"the geometric check failed ({exc})")
        report["self_intersecting_faces"] = len(intersected) or None
        return report
    report.update(self_intersecting_faces=len(intersected), screening_scope="full")
    return report
