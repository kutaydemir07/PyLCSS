# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE.
"""Keep STL facets out of original through-bores, including rotated bores."""

from __future__ import annotations

from typing import Any

import numpy as np


def source_bore_records(source_cad: Any, *, diagnostics: dict | None = None) -> list:
    """Return inward cylindrical through-bores in the original CAD coordinates.

    Half-circle or larger inward walls include open-sided drilled holes while
    excluding short concave blend arcs. Blind bores retain a material end cap;
    they are excluded because radial facet projection cannot preserve that cap.
    """
    from .cad_features import _coerce_to_solid, _preserved_bore_records

    shape = _coerce_to_solid(source_cad)
    if shape is None:
        return []
    from OCP.BRepClass3d import BRepClass3d_SolidClassifier
    from OCP.TopAbs import TopAbs_IN, TopAbs_ON
    from OCP.gp import gp_Pnt

    candidates = _preserved_bore_records(shape, minimum_angle=np.pi - 1e-6)
    classifiers = [BRepClass3d_SolidClassifier(solid.wrapped) for solid in shape.Solids()]

    def inside(point):
        for classifier in classifiers:
            classifier.Perform(gp_Pnt(*(float(value) for value in point)), 1e-8)
            if classifier.State() in (TopAbs_IN, TopAbs_ON):
                return True
        return False

    records = []
    for axis, origin, radius, lower, upper in candidates:
        margin = max(1e-5, float(radius) * 1e-5, float(upper - lower) * 1e-7)
        if inside(origin + axis * (lower - margin)) or inside(origin + axis * (upper + margin)):
            continue
        records.append((axis, origin, float(radius), float(lower), float(upper)))
    if diagnostics is not None:
        diagnostics.update(
            detected_cylindrical_cavity_count=len(candidates),
            through_bore_count=len(records),
            excluded_blind_bore_count=len(candidates) - len(records),
        )
    return records


def _clip_axial_polygon(points, axial, boundary, *, keep_above):
    """Clip one triangle/polygon against an axial plane without triangulating."""
    result, heights = [], []
    previous, previous_height = points[-1], axial[-1]
    previous_inside = previous_height >= boundary if keep_above else previous_height <= boundary
    for current, height in zip(points, axial):
        current_inside = height >= boundary if keep_above else height <= boundary
        if current_inside != previous_inside:
            fraction = (boundary - previous_height) / (height - previous_height)
            result.append(previous + fraction * (current - previous))
            heights.append(boundary)
        if current_inside:
            result.append(current)
            heights.append(height)
        previous, previous_height, previous_inside = current, height, current_inside
    return np.asarray(result), np.asarray(heights)


def _closest_radial_polygon(points, axis):
    """Closest point of a convex projected polygon to its cylinder axis."""
    following = np.roll(points, -1, axis=0)
    edges = following - points
    squared = np.einsum("ij,ij->i", edges, edges)
    parameter = np.clip(-np.einsum("ij,ij->i", points, edges) / np.maximum(squared, 1e-30), 0, 1)
    candidates = points + parameter[:, None] * edges
    cross = np.cross(points, following) @ axis
    scale = max(1.0, float(np.max(squared)))
    tolerance = scale * 1e-12
    if abs(float(np.sum(cross))) > tolerance and (
        np.min(cross) >= -tolerance or np.max(cross) <= tolerance
    ):
        return np.zeros(3)
    return candidates[np.argmin(np.einsum("ij,ij->i", candidates, candidates))]


def _bore_facet_supports(vertices, faces, record, tolerance):
    axis, origin, radius, lower, upper = record
    axial = (vertices - origin) @ axis
    radial = vertices - origin - axial[:, None] * axis
    triangles = radial[faces]
    edge_lengths = np.linalg.norm(triangles - np.roll(triangles, -1, axis=1), axis=2)
    # The distance-to-axis function is 1-Lipschitz, so a triangle farther than
    # its longest edge from the bore cannot intersect it. Keep all other faces.
    possible = (
        (np.min(np.linalg.norm(triangles, axis=2), axis=1) <= radius + np.max(edge_lengths, axis=1) + tolerance)
        & (np.max(axial[faces], axis=1) > lower + tolerance)
        & (np.min(axial[faces], axis=1) < upper - tolerance)
    )
    supports = []
    for face_index in np.flatnonzero(possible):
        face = faces[face_index]
        polygon, heights = radial[face], axial[face]
        if np.min(heights) < lower + tolerance:
            polygon, heights = _clip_axial_polygon(polygon, heights, lower + tolerance, keep_above=True)
        if len(polygon) and np.max(heights) > upper - tolerance:
            polygon, heights = _clip_axial_polygon(polygon, heights, upper - tolerance, keep_above=False)
        if len(polygon):
            closest = _closest_radial_polygon(polygon, axis)
            supports.append((int(face_index), closest, float(np.linalg.norm(closest))))
    return axial, radial, supports


def enforce_bore_clearance(vertices, faces, records, spacing):
    """Move wall vertices radially until whole STL facets clear every bore.

    Projecting vertices onto the nominal cylinder leaves straight triangle
    chords inside its bore. Each penetrating facet instead supplies a tangent
    supporting half-plane. Shared vertices satisfy every incident constraint,
    making the polygonal hole circumscribe the CAD bore without changing mesh
    connectivity. An unresolved cap is rejected instead of producing spikes.
    """
    original = np.asarray(vertices, dtype=float)
    faces = np.asarray(faces, dtype=np.int64)
    result = original.copy()
    spacing = np.asarray(spacing, dtype=float)
    tolerance = max(1e-9, float(np.linalg.norm(spacing)) * 1e-9)
    # Binary STL stores coordinates as float32. Reserve several coordinate
    # ulps so rounding the exported facets cannot put them inside the bore.
    coordinate_ulp = float(np.max(np.abs(np.spacing(original.astype(np.float32))), initial=0))
    safety = max(4 * tolerance, 4 * coordinate_ulp)
    if not records:
        return result, {"through_bore_count": 0, "corrected_vertex_count": 0}

    failure = (
        "The STL grid cannot preserve an original bore without closing or distorting it. "
        "Use a finer voxel mesh and regenerate the STL."
    )
    for _ in range(8):
        corrected = False
        for record in records:
            axis, origin, radius, _, _ = record
            axial, radial, supports = _bore_facet_supports(result, faces, record, tolerance)
            lengths = np.linalg.norm(radial, axis=1)
            units = radial / np.maximum(lengths[:, None], 1e-30)
            target = lengths.copy()
            for face_index, closest, distance in supports:
                if distance >= radius + safety - tolerance:
                    continue
                if distance <= tolerance:
                    raise ValueError(failure)
                face = faces[face_index]
                normal = closest / distance
                cosine = units[face] @ normal
                if np.any(cosine <= 1e-10):
                    raise ValueError(failure)
                required = (radius + safety) / cosine
                limit = max(3 * float(np.linalg.norm(spacing)), radius * 0.5)
                if np.any(required - lengths[face] > limit):
                    raise ValueError(failure)
                np.maximum.at(target, face, required)
            changed = target > lengths + tolerance
            if np.any(changed):
                result[changed] = origin + axial[changed, None] * axis + units[changed] * target[changed, None]
                corrected = True
        if not corrected:
            break

    clearances = []
    for record in records:
        _, _, radius, _, _ = record
        _, _, supports = _bore_facet_supports(result, faces, record, tolerance)
        minimum = min((distance for _, _, distance in supports), default=float(radius))
        if supports and minimum < radius + safety - tolerance:
            raise ValueError(failure)
        clearances.append({
            "radius_mm": float(radius),
            "minimum_facet_radius_mm": float(minimum),
            "checked_facet_count": len(supports),
        })

    original_cross = np.cross(original[faces[:, 1]] - original[faces[:, 0]], original[faces[:, 2]] - original[faces[:, 0]])
    final_cross = np.cross(result[faces[:, 1]] - result[faces[:, 0]], result[faces[:, 2]] - result[faces[:, 0]])
    if np.any(np.einsum("ij,ij->i", original_cross, final_cross) <= 0):
        raise ValueError(failure)
    displacement = np.linalg.norm(result - original, axis=1)
    return result, {
        "through_bore_count": len(records),
        "corrected_vertex_count": int(np.count_nonzero(displacement > tolerance)),
        "maximum_vertex_displacement_mm": float(np.max(displacement, initial=0)),
        "stl_coordinate_safety_mm": float(safety),
        "bore_clearances": clearances,
        "method": "Tangent facet constraints retain the original bore clearance without changing mesh connectivity",
    }
