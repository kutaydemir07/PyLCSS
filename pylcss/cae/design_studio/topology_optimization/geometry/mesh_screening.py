# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE.
"""Bounded geometric screening for intersections of non-adjacent triangles."""

from __future__ import annotations

import numpy as np
from ._compat import warn_retired_options


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


def screen_mesh_intersections(vertices: np.ndarray, faces: np.ndarray,
                              **legacy_limits) -> dict:
    """Check all spatially overlapping, topologically non-adjacent faces.

    Shared-vertex neighbours are excluded; closure, winding and degenerate
    faces are checked separately. Candidate pairs are streamed in batches;
    retired budget arguments warn and never skip a large mesh.
    """
    warn_retired_options(legacy_limits, operation='Mesh intersection screening')
    report = {"self_intersecting_faces": None, "screening_scope": "not_run",
              "screening_method": "Non-adjacent triangle separating-axis test",
              "screened_candidate_pairs": 0}
    vertices, faces = np.asarray(vertices, dtype=float), np.asarray(faces, dtype=np.int64)
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
    pending_first, pending_second = [], []

    def check_batch():
        if not pending_first:
            return
        left = np.asarray(pending_first, dtype=np.int64)
        right = np.asarray(pending_second, dtype=np.int64)
        pending_first.clear()
        pending_second.clear()
        adjacent = np.any(faces[right, :, None] == faces[left, None, :], axis=(1, 2))
        left, right = left[~adjacent], right[~adjacent]
        report['screened_candidate_pairs'] += len(left)
        for start in range(0, len(left), 4096):
            a, b = left[start:start+4096], right[start:start+4096]
            hits = _intersecting_triangles(triangles[a], triangles[b], tolerance)
            intersected.update(a[hits].tolist())
            intersected.update(b[hits].tolist())

    try:
        for index, bbox in enumerate(bounds):
            candidates = [other for other in tree.intersection(bbox) if other > index]
            if not len(candidates):
                continue
            pending_first.extend([index]*len(candidates))
            pending_second.extend(candidates)
            if len(pending_first) >= 4096:
                check_batch()
        check_batch()
    except Exception as exc:
        report.update(screening_scope="screening_error", screening_reason=f"the geometric check failed ({exc})")
        report["self_intersecting_faces"] = len(intersected) or None
        return report
    report.update(self_intersecting_faces=len(intersected), screening_scope="full")
    return report
