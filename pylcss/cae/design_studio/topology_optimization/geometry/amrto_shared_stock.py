# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE.
"""Certify canonical shared curves before composing incident patch sides."""
from __future__ import annotations

import numpy as np
from scipy.interpolate import BSpline


class CanonicalChainCertificationError(ValueError):
    """A rejected canonical proposal with a serializable strict certificate report."""
    def __init__(self, message, report):
        super().__init__(message)
        self.report = {'accepted': False, 'rejection_reason': str(message), **report}


def control_hull_in_box(controls, bounds, *, rounding_allowance=1e-9):
    """Bound a polynomial/positive-weight spline through its Euclidean poles."""
    points, limits = np.asarray(controls, dtype=float), np.asarray(bounds, dtype=float)
    if (points.ndim < 2 or points.shape[-1] != 3 or not np.isfinite(points).all()
            or limits.shape != (2, 3) or not np.isfinite(limits).all()
            or np.any(limits[0] >= limits[1])):
        raise ValueError('Stock hull certificate requires finite 3D poles and strict box limits')
    excess = float(np.maximum(np.maximum(limits[0] - points, points - limits[1]), 0).max(initial=0.0))
    return {'accepted': excess <= rounding_allowance,
            'maximum_control_hull_stock_excess': excess,
            'arithmetic_rounding_allowance': float(rounding_allowance),
            'method': 'Positive-weight spline convex hull in the supplied source stock box'}


def project_canonical_chain(curve, points, bounds, tolerance, *, maximum_subdivision_depth=16):
    """Project ONE shared-chain pole set, then certify its complete source error.

    Source and candidate curves are inserted into the same cubic Bezier basis.
    Their difference's control hull bounds its complete Euclidean error.
    Bounded de Casteljau subdivision tightens conservative hulls; exceeding
    the budget or exhausting the certificate depth rejects the curve.
    Incident sides must subsequently compose this same canonical result.
    """
    from .amrto_shared_curves import _polyline_curve
    points, limits = np.asarray(points, dtype=float), np.asarray(bounds, dtype=float)
    if (not isinstance(curve, BSpline) or curve.k != 3 or np.asarray(curve.c).ndim != 2
            or np.asarray(curve.c).shape[1] != 3 or points.ndim != 2 or points.shape[1] != 3
            or len(points) < 2 or not np.isfinite(points).all()
            or not np.isfinite(tolerance) or tolerance <= 0 or maximum_subdivision_depth < 0):
        raise ValueError('Canonical stock projection requires a finite cubic chain and positive source budget')
    control_hull_in_box(curve.c, limits)  # Validate the supplied stock before projection.
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    if np.any(lengths <= 0):
        raise ValueError('Canonical source chain has a zero-length physical segment')
    parameters = np.r_[0.0, np.cumsum(lengths)] / lengths.sum()
    source = _polyline_curve(points, parameters)
    projected = np.clip(curve.c, limits[0], limits[1])
    candidate = BSpline(curve.t.copy(), projected, curve.k)
    rejection_context = {'source_boundary_error_budget': float(tolerance),
                         'source_stock_bounds': limits.tolist(),
                         'projection_before_side_composition': True,
                         'outside_stock_curve_not_published': True}
    if candidate.t[3] != 0.0 or candidate.t[-4] != 1.0:
        raise ValueError('Canonical source certificate requires the unit parameter domain')
    breaks = np.unique(np.r_[candidate.t, source.t])
    refined = []
    for item in (candidate, source):
        for knot in breaks[1:-1]:
            count = int(np.count_nonzero(item.t == knot))
            if count < 3:
                item = item.insert_knot(float(knot), 3 - count)
        refined.append(item)
    if not np.array_equal(refined[0].t, refined[1].t):
        raise ValueError('Canonical source-error certificate lost its identical Bezier basis')
    delta = np.asarray(refined[0].c) - np.asarray(refined[1].c)
    pending = [(delta[3 * index:3 * index + 4], 0) for index in range(len(breaks) - 1)]
    allowance, bound, leaves = 1e-10, 0.0, 0
    while pending:
        poles, depth = pending.pop()
        upper = float(np.linalg.norm(poles, axis=1).max())
        middle = (poles[0] + 3 * poles[1] + 3 * poles[2] + poles[3]) / 8
        if max(np.linalg.norm(poles[0]), np.linalg.norm(poles[-1]), np.linalg.norm(middle)) > tolerance + allowance:
            raise CanonicalChainCertificationError(
                'Projected canonical chain exceeds its original-source boundary error budget',
                {**rejection_context, 'measured_source_error_lower_bound': float(max(
                    np.linalg.norm(poles[0]), np.linalg.norm(poles[-1]), np.linalg.norm(middle)))})
        if upper <= tolerance + allowance:
            bound, leaves = max(bound, upper), leaves + 1
            continue
        if depth >= maximum_subdivision_depth:
            raise CanonicalChainCertificationError(
                'Complete canonical-chain source error cannot be certified within the bounded subdivision budget',
                {**rejection_context, 'uncertified_error_upper_bound': upper,
                 'maximum_certificate_subdivision_depth': maximum_subdivision_depth})
        first = (poles[:-1] + poles[1:]) / 2
        second = (first[:-1] + first[1:]) / 2
        center = (second[0] + second[1]) / 2
        pending.extend(((np.array([poles[0], first[0], second[0], center]), depth + 1),
                        (np.array([center, second[1], first[2], poles[3]]), depth + 1)))
    report = {'accepted': True, 'projection_before_side_composition': True,
              'changed_pole_count': int(np.count_nonzero(np.any(projected != curve.c, axis=1))),
              'maximum_pole_projection': float(np.linalg.norm(projected - curve.c, axis=1).max(initial=0.0)),
              'control_count_unchanged': len(candidate.c) == len(curve.c),
              'complete_source_curve_error_bound': bound, 'source_boundary_error_budget': float(tolerance),
              'arithmetic_rounding_allowance': allowance, 'certificate_Bezier_leaves': leaves,
              'maximum_certificate_subdivision_depth': maximum_subdivision_depth,
              'source_error_method': 'Convex hull of exact common-basis cubic Bezier differences',
              'stock_hull': control_hull_in_box(candidate.c, limits)}
    return candidate, report
