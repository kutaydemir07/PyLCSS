# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE.
"""Measure CAD volume without integrating across unresolved spline knots.

For rectangular nonrational spline faces the divergence integral is a
polynomial on every knot rectangle. Gauss quadrature of order ceil(3*p/2)
integrates it exactly apart from floating-point roundoff. Other closed CAD
uses OCCT's span-aware Gauss-Kronrod integrator with its error estimate and
an independent tighter-tolerance convergence check.
"""
from __future__ import annotations

from collections import defaultdict
import math
from typing import Any

import numpy as np
from scipy.interpolate import BSpline


class UnsupportedVolumeGeometry(ValueError):
    """A face needs the general trimmed-surface volume integrator."""


def _closed_shape(shape):
    from OCP.BRep import BRep_Tool
    shape = shape.val() if hasattr(shape, 'val') else shape
    if (not shape.isValid() or not shape.Solids() or not shape.Shells()
            or not all(BRep_Tool.IsClosed_s(shell.wrapped) for shell in shape.Shells())):
        raise ValueError('Volume measurement requires valid closed solid CAD')
    return shape


def _rectangular_pcurves(face, bounds):
    """Prove that one wire covers the four UV-rectangle sides exactly once."""
    from OCP.BRepAdaptor import BRepAdaptor_Curve2d
    from OCP.BRepTools import BRepTools_WireExplorer
    from OCP.GeomAbs import GeomAbs_BezierCurve, GeomAbs_BSplineCurve, GeomAbs_Line

    wires = face.Wires()
    if len(wires) != 1:
        raise UnsupportedVolumeGeometry('The face has holes or multiple parameter wires')
    u0, u1, v0, v1 = bounds
    tolerance = max(1., *(abs(value) for value in bounds)) * 1e-11
    intervals = defaultdict(list)
    explorer = BRepTools_WireExplorer(wires[0].wrapped, face.wrapped)
    while explorer.More():
        curve = BRepAdaptor_Curve2d(explorer.Current(), face.wrapped)
        first, last = curve.FirstParameter(), curve.LastParameter()
        if not np.isfinite((first, last)).all() or last <= first:
            raise UnsupportedVolumeGeometry('A parameter boundary has no finite interval')
        a, b = curve.Value(first), curve.Value(last)
        endpoints = np.array([[a.X(), a.Y()], [b.X(), b.Y()]])
        kind = curve.GetType()
        if kind == GeomAbs_Line:
            direction = curve.Line().Direction()
            vectors = np.array([[direction.X(), direction.Y()]])
        elif kind in (GeomAbs_BezierCurve, GeomAbs_BSplineCurve):
            basis = curve.Bezier() if kind == GeomAbs_BezierCurve else curve.BSpline()
            if basis.Degree() != 1 or basis.IsRational():
                raise UnsupportedVolumeGeometry('The parameter boundary is not a straight polynomial segment')
            poles = np.array([[basis.Pole(i).X(), basis.Pole(i).Y()] for i in range(1, basis.NbPoles() + 1)])
            vectors = np.diff(poles, axis=0)
        else:
            raise UnsupportedVolumeGeometry('The parameter boundary is not axis aligned')
        axis = None
        for constant_axis in (0, 1):
            if (np.max(np.abs(vectors[:, constant_axis]), initial=0.) <= tolerance
                    and abs(endpoints[0, constant_axis] - endpoints[1, constant_axis]) <= tolerance):
                axis = constant_axis
                break
        if axis is None:
            raise UnsupportedVolumeGeometry('A parameter edge crosses the rectangle interior')
        variable = 1 - axis
        low, high = ((u0, u1) if axis == 0 else (v0, v1))
        constant = endpoints[:, axis].mean()
        side = 0 if abs(constant - low) <= tolerance else 1 if abs(constant - high) <= tolerance else None
        if side is None:
            raise UnsupportedVolumeGeometry('A parameter edge does not lie on a rectangle side')
        varying = vectors[:, variable]
        if (np.any(varying > tolerance) and np.any(varying < -tolerance)):
            raise UnsupportedVolumeGeometry('A parameter boundary doubles back')
        lower, upper = sorted(endpoints[:, variable])
        if upper - lower <= tolerance:
            raise UnsupportedVolumeGeometry('A parameter boundary is degenerate')
        intervals[axis, side].append((float(lower), float(upper)))
        explorer.Next()
    for axis in (0, 1):
        lower, upper = ((v0, v1) if axis == 0 else (u0, u1))
        for side in (0, 1):
            cursor = lower
            for first, last in sorted(intervals[axis, side]):
                if abs(first - cursor) > tolerance or last > upper + tolerance:
                    raise UnsupportedVolumeGeometry('The parameter wire has a gap or overlapping side intervals')
                cursor = last
            if abs(cursor - upper) > tolerance:
                raise UnsupportedVolumeGeometry('The parameter wire does not cover all four sides')
    return {'single_wire': True, 'axis_aligned_pcurves': True, 'rectangle_sides_covered_once': True}


def _rectangular_face_data(face):
    from OCP.BRepAdaptor import BRepAdaptor_Surface
    from OCP.GeomAbs import GeomAbs_BSplineSurface
    from OCP.TopAbs import TopAbs_FORWARD, TopAbs_REVERSED

    adaptor = BRepAdaptor_Surface(face.wrapped, True)
    if adaptor.GetType() != GeomAbs_BSplineSurface:
        raise UnsupportedVolumeGeometry('A face is not a polynomial B-spline surface')
    surface = adaptor.BSpline()  # OCCT applies the face location to this copy.
    if surface.IsURational() or surface.IsVRational() or surface.IsUPeriodic() or surface.IsVPeriodic():
        raise UnsupportedVolumeGeometry('Rational or periodic spline bases require general integration')
    bounds = tuple(map(float, face._uvBounds()))
    if not np.isfinite(bounds).all() or bounds[1] <= bounds[0] or bounds[3] <= bounds[2]:
        raise UnsupportedVolumeGeometry('A face has no finite parameter rectangle')
    boundary = _rectangular_pcurves(face, bounds)
    orientation = face.wrapped.Orientation()
    if orientation not in (TopAbs_FORWARD, TopAbs_REVERSED):
        raise UnsupportedVolumeGeometry('A face has an unsupported material orientation')
    poles = np.array([[surface.Pole(i, j).Coord() for j in range(1, surface.NbVPoles() + 1)]
                     for i in range(1, surface.NbUPoles() + 1)], dtype=float)
    knots_u = np.repeat([surface.UKnot(i) for i in range(1, surface.NbUKnots() + 1)],
                        [surface.UMultiplicity(i) for i in range(1, surface.NbUKnots() + 1)])
    knots_v = np.repeat([surface.VKnot(i) for i in range(1, surface.NbVKnots() + 1)],
                        [surface.VMultiplicity(i) for i in range(1, surface.NbVKnots() + 1)])
    return {'poles': poles, 'knots': (knots_u, knots_v),
            'degrees': (surface.UDegree(), surface.VDegree()), 'bounds': bounds,
            'sign': -1. if orientation == TopAbs_REVERSED else 1., 'boundary': boundary}


def _gauss_axis(knots, degree, lower, upper, extra_order):
    levels = np.unique(np.r_[lower, knots[(knots > lower) & (knots < upper)], upper])
    order = math.ceil(1.5 * degree) + extra_order
    nodes, weights = np.polynomial.legendre.leggauss(order)
    first, last = levels[:-1], levels[1:]
    locations = ((first[:, None] + last[:, None]) * .5
                 + (last - first)[:, None] * .5 * nodes).ravel()
    quadrature = ((last - first)[:, None] * .5 * weights).ravel()
    return locations, quadrature, order, len(first)


def _face_integral(data, origin, extra_order=0):
    poles = data['poles'] - origin
    axes = [_gauss_axis(knots, degree, *data['bounds'][2*axis:2*axis+2], extra_order)
            for axis, (knots, degree) in enumerate(zip(data['knots'], data['degrees']))]
    basis = [BSpline(knots, np.eye(poles.shape[axis]), degree)
             for axis, (knots, degree) in enumerate(zip(data['knots'], data['degrees']))]
    values = [spline(axis[0]) for spline, axis in zip(basis, axes)]
    derivatives = [spline(axis[0], nu=1) for spline, axis in zip(basis, axes)]
    position = np.einsum('ui,ijc,vj->uvc', values[0], poles, values[1], optimize=True)
    du = np.einsum('ui,ijc,vj->uvc', derivatives[0], poles, values[1], optimize=True)
    dv = np.einsum('ui,ijc,vj->uvc', values[0], poles, derivatives[1], optimize=True)
    integrand = np.einsum('uvc,uvc->uv', position, np.cross(du, dv)) / 3.
    terms = integrand * axes[0][1][:, None] * axes[1][1][None, :]
    return data['sign'] * float(terms.sum()), float(np.abs(terms).sum()), [axis[2] for axis in axes]


def _evaluate(data, u, v, origin):
    poles = data['poles'] - origin
    basis = [BSpline(knots, np.eye(poles.shape[index]), degree)(parameters)
             for index, (knots, degree, parameters) in enumerate(zip(data['knots'], data['degrees'], (u, v)))]
    return np.einsum('ui,ijc,vj->uvc', basis[0], poles, basis[1], optimize=True)


def _polynomial_difference_bound(parent, child):
    """Bound each polynomial cell by its Bernstein difference coefficients."""
    from math import comb
    axes = []
    inverses = []
    for index in (0, 1):
        lower, upper = child['bounds'][2*index:2*index+2]
        knots = np.r_[parent['knots'][index], child['knots'][index]]
        levels = np.unique(np.r_[lower, knots[(knots > lower) & (knots < upper)], upper])
        count = max(parent['degrees'][index], child['degrees'][index]) + 1
        nodes = (np.polynomial.legendre.leggauss(count)[0] + 1.) * .5
        locations = (levels[:-1, None] + np.diff(levels)[:, None] * nodes).ravel()
        degree = count - 1
        bernstein = np.array([[comb(degree, column) * x**column * (1-x)**(degree-column)
                               for column in range(count)] for x in nodes])
        axes.append((locations, len(levels)-1, count))
        inverses.append(np.linalg.inv(bernstein))
    origin = child['poles'].reshape(-1, 3).mean(axis=0)
    difference = (_evaluate(child, axes[0][0], axes[1][0], origin)
                  - _evaluate(parent, axes[0][0], axes[1][0], origin))
    cells = difference.reshape(axes[0][1], axes[0][2], axes[1][1], axes[1][2], 3).transpose(0, 2, 1, 3, 4)
    controls = np.einsum('ai,stijc,bj->stabc', inverses[0], cells, inverses[1], optimize=True)
    return float(np.linalg.norm(controls, axis=-1).max())


def assert_rectangular_spline_partition(before: Any, after: Any,
                                        parent_face_indices: list[int]) -> dict:
    """Verify exact UV coverage and polynomial surface identity of C1 children.

    The parent mapping comes from the continuity divider's checked face
    history. Every parent rectangle must be covered exactly once. A Bernstein
    coefficient bound checks the complete polynomial difference on the union
    of parent/child knot cells, rather than a few point samples.
    """
    before, after = _closed_shape(before), _closed_shape(after)
    parents = [_rectangular_face_data(face) for face in before.Faces()]
    children = [_rectangular_face_data(face) for face in after.Faces()]
    if len(parent_face_indices) != len(children):
        raise RuntimeError('Spline partition history does not cover all child faces')
    grouped = defaultdict(list)
    maximum = 0.
    for child, index in zip(children, parent_face_indices):
        if not isinstance(index, (int, np.integer)) or not 0 <= index < len(parents):
            raise RuntimeError('Spline partition history has an invalid parent')
        parent = parents[index]
        u0, u1, v0, v1 = parent['bounds']
        a, b, c, d = child['bounds']
        if not (u0 <= a < b <= u1 and v0 <= c < d <= v1):
            raise RuntimeError('Spline partition left its parent parameter rectangle')
        if parent['sign'] != child['sign']:
            raise RuntimeError('Spline partition changed a material-face orientation')
        bound = _polynomial_difference_bound(parent, child)
        scale = float(np.linalg.norm(np.ptp(parent['poles'].reshape(-1, 3), axis=0)))
        if bound > max(1e-8, scale * 1e-10):
            raise RuntimeError('Spline partition changed its source polynomial surface')
        maximum = max(maximum, bound)
        grouped[index].append(child['bounds'])
    for index, parent in enumerate(parents):
        rectangles = grouped[index]
        if not rectangles:
            raise RuntimeError('Spline partition omitted a source face')
        u0, u1, v0, v1 = parent['bounds']
        levels = np.unique([v0, v1, *(value for rectangle in rectangles for value in rectangle[2:])])
        events = {u0: [], u1: []}
        for a, b, c, d in rectangles:
            first, last = int(np.searchsorted(levels, c)), int(np.searchsorted(levels, d))
            events.setdefault(a, []).append((first, last, 1))
            events.setdefault(b, []).append((first, last, -1))
        coverage = np.zeros(len(levels), dtype=np.int64)
        for u, changes in sorted(events.items()):
            for first, last, sign in changes:
                coverage[first] += sign
                coverage[last] -= sign
            if u < u1 and not np.all(np.cumsum(coverage)[:-1] == 1):
                raise RuntimeError('Spline partition does not cover each parent UV interval exactly once')
        if np.any(coverage):
            raise RuntimeError('Spline partition left unmatched parameter intervals')
    return {'source_uv_coverage_exactly_once': True,
            'whole_polynomial_geometry_verified': True,
            'maximum_bernstein_difference_bound': maximum,
            'source_face_count': len(parents), 'child_face_count': len(children)}


def rectangular_spline_volume(shape: Any) -> tuple[float, dict]:
    """Integrate a closed body's proven rectangular nonrational spline faces."""
    shape = _closed_shape(shape)
    faces = [_rectangular_face_data(face) for face in shape.Faces()]
    minima = np.min([data['poles'].min(axis=(0, 1)) for data in faces], axis=0)
    maxima = np.max([data['poles'].max(axis=(0, 1)) for data in faces], axis=0)
    origin = (minima + maxima) * .5
    coarse = [_face_integral(data, origin) for data in faces]
    check = [_face_integral(data, origin, 1) for data in faces]
    volume = abs(math.fsum(row[0] for row in coarse))
    verification = abs(math.fsum(row[0] for row in check))
    absolute_sum = math.fsum(row[1] for row in coarse)
    numerical = max(1e-9, absolute_sum * np.finfo(float).eps * 4096)
    if not np.isfinite((volume, verification)).all() or volume <= 0 or abs(volume - verification) > numerical:
        raise RuntimeError('Polynomial spline-volume quadrature did not meet its roundoff check')
    return volume, {
        'method': 'Per-knot-span polynomial Gauss divergence integral',
        'polynomial_integral_exact': True, 'floating_point_roundoff_only': True,
        'face_count': len(faces), 'gauss_orders': [row[2] for row in coarse],
        'rectangular_pcurve_boundaries_verified': True,
        'higher_order_check_volume': verification,
        'higher_order_check_difference': abs(volume - verification),
        'numerical_tolerance': numerical, 'relative_error_estimate': numerical / volume,
    }


def measure_cad_volume(shape: Any, *, relative_tolerance: float = 1e-8) -> tuple[float, dict]:
    """Prefer exact polynomial integration; otherwise verify native convergence."""
    if not math.isfinite(relative_tolerance) or relative_tolerance <= 0:
        raise ValueError('Volume error tolerance must be finite and positive')
    shape = _closed_shape(shape)
    try:
        return rectangular_spline_volume(shape)
    except UnsupportedVolumeGeometry as error:
        reason = str(error)
    from OCP.BRepGProp import BRepGProp
    from OCP.GProp import GProp_GProps

    estimates = []
    for tolerance in (relative_tolerance * .1, relative_tolerance * .01):
        properties = GProp_GProps()
        achieved = float(BRepGProp.VolumePropertiesGK_s(
            shape.wrapped, properties, tolerance, True, True, False, False, False))
        volume = abs(float(properties.Mass()))
        if not np.isfinite((achieved, volume)).all() or achieved < 0 or volume <= 0:
            raise RuntimeError('Span-aware CAD volume integration failed')
        estimates.append({'volume': volume, 'requested_relative_error': tolerance,
                          'estimated_relative_error': achieved})
    difference = abs(estimates[0]['volume'] - estimates[1]['volume'])
    if (any(row['estimated_relative_error'] > relative_tolerance for row in estimates)
            or difference > max(1e-9, relative_tolerance * estimates[-1]['volume'])):
        raise RuntimeError('Span-aware CAD volume integration did not converge within its error allowance')
    return estimates[-1]['volume'], {
        'method': 'OpenCASCADE span-aware Gauss-Kronrod integration',
        'use_spline_knot_spans': True, 'polynomial_integral_exact': False,
        'polynomial_ineligibility_reason': reason, 'convergence': estimates,
        'convergence_difference': difference,
        'relative_error_estimate': max(row['estimated_relative_error'] for row in estimates),
    }


def assert_unchanged_cad_volume(before: Any, after: Any, *, relative_tolerance: float = 1e-6) -> dict:
    """Retain the material-volume invariant with resolved-knot integration."""
    first, first_report = measure_cad_volume(before, relative_tolerance=relative_tolerance * .01)
    second, second_report = measure_cad_volume(after, relative_tolerance=relative_tolerance * .01)
    difference = abs(first - second)
    if difference > max(1e-7, first * relative_tolerance):
        raise RuntimeError('Exact CAD subdivision changed the enclosed CAD volume')
    return {'volume_before': first, 'volume_after': second,
            'relative_volume_difference': difference / first,
            'allowed_relative_volume_difference': relative_tolerance,
            'before_integration': first_report, 'after_integration': second_report}


__all__ = ['UnsupportedVolumeGeometry', 'rectangular_spline_volume',
           'measure_cad_volume', 'assert_unchanged_cad_volume',
           'assert_rectangular_spline_partition']
