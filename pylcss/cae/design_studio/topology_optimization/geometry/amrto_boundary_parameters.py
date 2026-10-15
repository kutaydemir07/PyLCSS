# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.
"""Align opposing C0 boundary clocks before a single surface fit.

The physical curves and their poles never change. Within intervals separated
by original C0 joins, affine knot reparameterization preserves the complete
B-spline curve. Smooth interior knots follow the same affine clock. Source
XYZs remain untouched while their UV chart follows the four boundary clocks.
This is an engineering parameterization extension, not an AMRTO paper step.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy.interpolate import BSpline


def _reverse(curve: BSpline) -> BSpline:
    return BSpline(1.0 - curve.t[::-1], curve.c[::-1].copy(), curve.k)


def _validate_curve(curve: Any) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(curve, BSpline):
        raise ValueError("Boundary parameter alignment requires SciPy B-spline curves")
    knots, controls = np.asarray(curve.t), np.asarray(curve.c)
    if (curve.k < 1 or controls.ndim != 2 or controls.shape[1] != 3
            or len(controls) != len(knots) - curve.k - 1 or curve.axis != 0
            or not np.isfinite(knots).all() or not np.isfinite(controls).all()
            or np.any(np.diff(knots) < 0) or knots[curve.k] != 0.0
            or knots[-curve.k - 1] != 1.0):
        raise ValueError("Require finite 3D boundary curves clamped to [0, 1]")
    values, counts = np.unique(knots, return_counts=True)
    if (counts[0] != curve.k + 1 or counts[-1] != curve.k + 1
            or values[0] != 0.0 or values[-1] != 1.0
            or np.any(counts[1:-1] > curve.k)):
        raise ValueError("Boundary curves must be continuous and clamped at both endpoints")
    anchors = values[(counts >= curve.k) | (values == 0.0) | (values == 1.0)]
    return anchors, counts


def _align_pair(first: BSpline, second: BSpline, threshold: float):
    curves = [first, second]
    anchors = [_validate_curve(curve)[0] for curve in curves]
    moved = [values.copy() for values in anchors]
    # Disjoint adjacent pairs cannot identify two distinct joins on one curve.
    ordered = sorted((float(value), side, index)
                     for side, values in enumerate(anchors)
                     for index, value in enumerate(values) if 0 < value < 1)
    pairs = []
    index = 0
    while index + 1 < len(ordered):
        a, b = ordered[index:index + 2]
        if a[1] != b[1] and b[0] - a[0] < threshold:
            value = 0.5 * (a[0] + b[0])
            moved[a[1]][a[2]], moved[b[1]][b[2]] = value, value
            pairs.append({"old_parameters": [a[0], b[0]], "new_parameter": value,
                          "parameter_gap": b[0] - a[0]})
            index += 2
        else:
            index += 1
    result, reports = [], []
    for curve, original, new in zip(curves, anchors, moved):
        if np.any(np.diff(new) <= 0):
            raise ValueError("Alignment would merge or reverse distinct joins on one boundary")
        knots = np.interp(curve.t, original, new)
        retimed = BSpline(knots, curve.c.copy(), curve.k)
        if (not np.array_equal(np.unique(curve.t, return_counts=True)[1],
                               np.unique(retimed.t, return_counts=True)[1])
                or not np.array_equal(curve.c, retimed.c)):
            raise ValueError("Parameter alignment changed boundary poles or knot multiplicities")
        breaks = np.unique(curve.t)
        # Independent arithmetic sanity check of the exact affine-span identity.
        sites = np.unique(np.concatenate([np.linspace(a, b, curve.k + 2)
                                          for a, b in zip(breaks[:-1], breaks[1:])]))
        error = float(np.linalg.norm(curve(sites) - retimed(np.interp(sites, original, new)),
                                     axis=1).max(initial=0.0))
        scale = max(float(np.max(np.abs(curve.c), initial=0.0)), 1.0)
        # A few ULPs of parameter arithmetic can become visible on a very
        # short physical span. Bound that evaluation effect using the convex
        # hull of derivative controls, while the geometric identity itself
        # rests on unchanged poles and affine knot intervals above.
        speeds = []
        for candidate in (curve, retimed):
            count = len(candidate.t) - candidate.k - 1
            denominators = (candidate.t[candidate.k + 1:candidate.k + count]
                            - candidate.t[1:count])
            derivatives = candidate.k * np.diff(candidate.c[:count], axis=0) / denominators[:, None]
            speeds.append(float(np.linalg.norm(derivatives, axis=1).max(initial=0.0)))
        crosscheck_tolerance = 1024 * np.finfo(float).eps * max(scale, *speeds)
        if error > crosscheck_tolerance:
            raise ValueError("Affine boundary-clock identity failed its arithmetic crosscheck")
        reports.append({"old_C0_parameters": original.tolist(), "new_C0_parameters": new.tolist(),
                        "controls_exactly_unchanged": True, "knot_multiplicities_unchanged": True,
                        "degree_unchanged": True, "added_boundary_controls": 0,
                        "clock_changes_only_at_original_C0_joins": True,
                        "maximum_geometry_crosscheck_error": error,
                        "arithmetic_geometry_crosscheck_tolerance": crosscheck_tolerance})
        result.append(retimed)
    return result, list(zip(anchors, moved)), {"aligned_pair_count": len(pairs),
                                              "pairs": pairs, "curves": reports}


def transport_boundary_uv(parameters: np.ndarray, maps) -> np.ndarray:
    """Evaluate the transfinite chart map; ``maps`` use sides bottom/right/top/left."""
    uv = np.asarray(parameters, dtype=float)
    u, v = uv.T
    def evaluate(values, mapping):
        return values if np.array_equal(*mapping) else np.interp(values, *mapping)
    bottom, right = evaluate(u, maps[0]), evaluate(v, maps[1])
    top, left = evaluate(u, maps[2]), evaluate(v, maps[3])
    # The displacement form leaves identity axes bitwise unchanged.
    return np.column_stack((u + (1 - v) * (bottom - u) + v * (top - u),
                            v + (1 - u) * (left - v) + u * (right - v)))


def _clock_slope(mapping, lo: float, hi: float) -> float:
    old, new = mapping
    index = int(np.clip(np.searchsorted(old, 0.5 * (lo + hi), side="right") - 1,
                        0, len(old) - 2))
    return float((new[index + 1] - new[index]) / (old[index + 1] - old[index]))


def _certify_uv_tiles(maps) -> dict:
    """The Jacobian determinant is bilinear on each clock rectangle."""
    us = np.unique(np.r_[maps[0][0], maps[2][0]])
    vs = np.unique(np.r_[maps[1][0], maps[3][0]])
    minimum, maximum_product = np.inf, 0.0
    for ua, ub in zip(us[:-1], us[1:]):
        bottom = _clock_slope(maps[0], ua, ub)
        top = _clock_slope(maps[2], ua, ub)
        for va, vb in zip(vs[:-1], vs[1:]):
            left, right = _clock_slope(maps[3], va, vb), _clock_slope(maps[1], va, vb)
            for u in (ua, ub):
                d_u_d_v = float(np.interp(u, *maps[2]) - np.interp(u, *maps[0]))
                for v in (va, vb):
                    d_v_d_u = float(np.interp(v, *maps[1]) - np.interp(v, *maps[3]))
                    diagonal = ((1 - v) * bottom + v * top) * ((1 - u) * left + u * right)
                    off_diagonal = d_u_d_v * d_v_d_u
                    minimum = min(minimum, diagonal - off_diagonal)
                    maximum_product = max(maximum_product, abs(diagonal) + abs(off_diagonal))
    rounding_allowance = 128 * np.finfo(float).eps * max(maximum_product, 1.0)
    if not np.isfinite(minimum) or minimum <= rounding_allowance:
        raise ValueError("Boundary-clock transport is not strictly orientation preserving over every UV tile")
    return {"method": "Bilinear Jacobian determinant bounded at all four corners of every exact clock tile",
            "tile_count": (len(us) - 1) * (len(vs) - 1),
            "minimum_jacobian_determinant": float(minimum),
            "arithmetic_rounding_allowance": rounding_allowance,
            "strictly_positive_complete_UV_tiles": True}


def align_layout_boundary_parameters(layout: Any, *, knot_alignment_tolerance: float = 1e-3) -> dict:
    """Atomically align boundary clocks and source UVs without changing XYZ.

    Only opposing original C0 joins closer than the fixed parameter tolerance
    are paired. Every individual curve retains strict join order, unchanged
    poles, unchanged multiplicities and the same physical geometry. The entire
    transfinite UV map and every actual source UV triangle must preserve
    orientation. Failure leaves the layout unchanged. This performs no fit.
    """
    if not np.isfinite(knot_alignment_tolerance) or not 0 < knot_alignment_tolerance < 0.25:
        raise ValueError("Require a finite boundary-clock alignment tolerance in (0, .25)")
    curves = getattr(layout, "boundary_splines", None)
    uv = np.asarray(layout.parameter_coords, dtype=float)
    vertices = np.asarray(layout.vertices)
    triangles = np.asarray(layout.triangles)
    if (curves is None or len(curves) != 4 or vertices.ndim != 2 or vertices.shape[1] != 3
            or not np.isfinite(vertices).all() or uv.shape != (len(vertices), 2)
            or not np.isfinite(uv).all() or np.any(uv < 0) or np.any(uv > 1)
            or triangles.ndim != 2 or triangles.shape[1] != 3 or not len(triangles)
            or not np.issubdtype(triangles.dtype, np.integer) or triangles.min() < 0
            or triangles.max() >= len(vertices)):
        raise ValueError("Require four boundary curves and finite indexed source geometry on [0, 1]^2")
    for curve in curves:
        _validate_curve(curve)
    forward = [curves[0], curves[1], _reverse(curves[2]), _reverse(curves[3])]
    retimed, maps, pair_reports = [None] * 4, [None] * 4, []
    for sides in ((0, 2), (1, 3)):
        pair, clocks, report = _align_pair(forward[sides[0]], forward[sides[1]], knot_alignment_tolerance)
        for index, side in enumerate(sides):
            retimed[side], maps[side] = pair[index], clocks[index]
        report["sides"] = list(sides)
        pair_reports.append(report)
    tile_proof = _certify_uv_tiles(maps)
    mapped = transport_boundary_uv(uv, maps)
    old_tri, new_tri = uv[triangles], mapped[triangles]
    def determinant(points):
        a, b = points[:, 1] - points[:, 0], points[:, 2] - points[:, 0]
        return a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
    old_det, new_det = determinant(old_tri), determinant(new_tri)
    if np.any(old_det == 0) or np.any(old_det * new_det <= 0):
        raise ValueError("Boundary-clock transport reverses or degenerates an actual source UV triangle")
    report = {"method": "Exact opposing C0 boundary-clock alignment before one surface fit", "accepted": True,
              "engineering_parameterization_extension": True, "paper_algorithm": False,
              "layout_id": int(layout.layout_id), "alignment_tolerance": float(knot_alignment_tolerance),
              "aligned_pair_count": sum(pair["aligned_pair_count"] for pair in pair_reports),
              "pairs": pair_reports, "complete_UV_map": tile_proof,
              "source_triangle_count": len(triangles), "source_triangle_orientation_preserved": True,
              "minimum_source_UV_area_ratio": float(np.min(new_det / old_det)),
              "maximum_UV_vertex_shift": float(np.linalg.norm(mapped - uv, axis=1).max(initial=0.0)),
              "source_XYZ_unchanged": True, "physical_boundary_geometry_unchanged": True,
              "boundary_controls_added": 0, "surface_fits": 0,
              "UV_maps": [{"old": old.tolist(), "new": new.tolist()} for old, new in maps]}
    result = [retimed[0], retimed[1], _reverse(retimed[2]), _reverse(retimed[3])]
    # Reversing a parameter twice can change a low-order floating-point bit.
    # Preserve the exact original curve object whenever its clock is identity.
    layout.boundary_splines = [curves[index] if np.array_equal(*maps[index]) else result[index]
                               for index in range(4)]
    layout.parameter_coords = mapped
    return report
