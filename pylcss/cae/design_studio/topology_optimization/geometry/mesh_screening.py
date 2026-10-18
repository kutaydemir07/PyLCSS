# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE.
"""Bounded geometric screening for intersections of non-adjacent triangles."""

from __future__ import annotations

import numpy as np
from ._compat import warn_retired_options


def _noncoplanar_interiors_cross(first: np.ndarray, second: np.ndarray, tolerance: float) -> bool:
    """Confirm a positive interior intersection without unstable skinny normals.

    SAT's tolerance deliberately admits nearby candidates. A nearly collinear
    triangle has an ill-conditioned plane normal, so plane-side tests alone can
    mistake a near-boundary miss for a crossing. Exact rational determinants
    on the input floating-point coordinates resolve only these positive
    candidates; no triangle is discarded because it is thin.
    """
    from fractions import Fraction
    from itertools import combinations

    triangles = [tuple(tuple(Fraction(float(value)) for value in point) for point in triangle)
                 for triangle in (first, second)]
    def subtract(a, b):
        return tuple(x-y for x, y in zip(a, b))
    def dot(a, b):
        return sum(x*y for x, y in zip(a, b))
    def cross(a, b):
        return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])
    points = set()
    for source, target in (triangles, triangles[::-1]):
        first_edge, second_edge = subtract(target[1], target[0]), subtract(target[2], target[0])
        for index in range(3):
            origin = source[index]
            direction = subtract(source[(index+1) % 3], origin)
            pvec = cross(direction, second_edge)
            determinant = dot(first_edge, pvec)
            if determinant == 0:
                continue
            relative = subtract(origin, target[0])
            u = dot(relative, pvec) / determinant
            qvec = cross(relative, first_edge)
            v = dot(direction, qvec) / determinant
            t = dot(second_edge, qvec) / determinant
            if 0 <= t <= 1 and u >= 0 and v >= 0 and u+v <= 1:
                points.add(tuple(a+t*b for a, b in zip(origin, direction)))

    def strictly_inside(point, triangle):
        a, b = subtract(triangle[1], triangle[0]), subtract(triangle[2], triangle[0])
        relative = subtract(point, triangle[0])
        aa, ab, bb = dot(a, a), dot(a, b), dot(b, b)
        denominator = aa*bb-ab*ab
        if denominator == 0:
            return False
        ar, br = dot(a, relative), dot(b, relative)
        u, v = (bb*ar-ab*br)/denominator, (aa*br-ab*ar)/denominator
        return u > 0 and v > 0 and u+v < 1

    minimum_length_squared = Fraction(float(tolerance)) ** 2
    for a, b in combinations(points, 2):
        delta = subtract(a, b)
        midpoint = tuple((x+y)/2 for x, y in zip(a, b))
        if (dot(delta, delta) > minimum_length_squared
                and all(strictly_inside(midpoint, triangle) for triangle in triangles)):
            return True
    return False


def _intersecting_triangles(first: np.ndarray, second: np.ndarray, tolerance: float,
                            *, include_boundary_contacts: bool = True) -> np.ndarray:
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
    hits = ~np.any(separated, axis=1)
    if include_boundary_contacts:
        return hits
    # Independently tessellated trimmed faces can meet at a T-junction without
    # sharing a mesh vertex. Merely touching there is not a surface crossing.
    unit_a = np.divide(normal_a, np.linalg.norm(normal_a, axis=1)[:, None],
                       out=np.zeros_like(normal_a), where=np.linalg.norm(normal_a, axis=1)[:, None] > 0)
    unit_b = np.divide(normal_b, np.linalg.norm(normal_b, axis=1)[:, None],
                       out=np.zeros_like(normal_b), where=np.linalg.norm(normal_b, axis=1)[:, None] > 0)
    distance_b = np.einsum('nvc,nc->nv', second - first[:, :1], unit_a)
    distance_a = np.einsum('nvc,nc->nv', first - second[:, :1], unit_b)
    coplanar = ((np.abs(distance_a).max(axis=1) <= tolerance)
                & (np.abs(distance_b).max(axis=1) <= tolerance))
    crosses_planes = ((distance_a.min(axis=1) < -tolerance)
                       & (distance_a.max(axis=1) > tolerance)
                       & (distance_b.min(axis=1) < -tolerance)
                       & (distance_b.max(axis=1) > tolerance))
    # Coplanar overlaps must have positive extent along every in-plane SAT
    # axis. Plane normals are intentionally excluded from this area test.
    in_plane_overlap = (np.minimum(projection_a[:, -6:].max(axis=2), projection_b[:, -6:].max(axis=2))
                        - np.maximum(projection_a[:, -6:].min(axis=2), projection_b[:, -6:].min(axis=2)))
    positive_coplanar_area = np.all((in_plane_overlap > tolerance) | ~valid[:, -6:], axis=1)
    result = hits & ((coplanar & positive_coplanar_area) | (~coplanar & crosses_planes))
    for index in np.flatnonzero(result & ~coplanar):
        result[index] = _noncoplanar_interiors_cross(first[index], second[index], tolerance)
    return result


def screen_mesh_intersections(vertices: np.ndarray, faces: np.ndarray,
                              *, include_boundary_contacts: bool = True,
                              **legacy_limits) -> dict:
    """Check all spatially overlapping, topologically non-adjacent faces.

    Shared-vertex neighbours are excluded; closure, winding and degenerate
    faces are checked separately. Candidate pairs are streamed in batches;
    retired budget arguments warn and never skip a large mesh.
    """
    warn_retired_options(legacy_limits, operation='Mesh intersection screening')
    report = {"self_intersecting_faces": None, "screening_scope": "not_run",
              "screening_method": "Non-adjacent triangle separating-axis test",
              "boundary_contacts_included": bool(include_boundary_contacts),
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
            hits = _intersecting_triangles(triangles[a], triangles[b], tolerance,
                                           include_boundary_contacts=include_boundary_contacts)
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
