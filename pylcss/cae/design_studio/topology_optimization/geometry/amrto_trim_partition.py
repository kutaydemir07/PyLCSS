# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE.
"""Certify exact angular partitions of cylinders with arbitrary native trims.

Bounding UV rectangles cannot certify a cylinder with a wavy trimmed cap.
Instead, preserve every oriented original trim curve over its complete
parameter interval and cancel every newly introduced internal seam. On the
same cylindrical coordinate function, disjoint angular strips with these
boundaries have the same trimmed domain and surface orientation.
"""
from __future__ import annotations

import numpy as np

from OCP.BRep import BRep_Tool
from OCP.BRepAdaptor import BRepAdaptor_Surface
from OCP.Geom2d import Geom2d_TrimmedCurve
from OCP.TopAbs import TopAbs_EDGE, TopAbs_FORWARD, TopAbs_REVERSED
from OCP.TopExp import TopExp_Explorer
from OCP.TopoDS import TopoDS


_PARAMETER_TOLERANCE = 2e-11


def _edge_occurrences(face):
    # Keep both oriented occurrences of an original periodic seam. Shape.Edges
    # removes that multiplicity and cannot establish oriented trim coverage.
    explorer = TopExp_Explorer(face.wrapped, TopAbs_EDGE)
    result = []
    while explorer.More():
        edge = TopoDS.Edge_s(explorer.Current())
        explorer.Next()
        curve = BRep_Tool.Curve_s(edge, 0., 0.)
        pcurve = BRep_Tool.CurveOnSurface_s(edge, face.wrapped, 0., 0.)
        interval = BRep_Tool.Range_s(edge, face.wrapped)
        if (curve is None or pcurve is None or not np.isfinite(interval).all()
                or edge.Orientation() not in (TopAbs_FORWARD, TopAbs_REVERSED)):
            raise ValueError('Cylindrical trim proof requires finite nondegenerate native curves')
        result.append((edge, curve, pcurve, interval))
    if not result:
        raise ValueError('Cylindrical trim proof requires a bounded native face')
    return result


def _pcurve_signature(curve):
    # A trimmed-curve wrapper retains its basis parameterization.
    while isinstance(curve, Geom2d_TrimmedCurve):
        curve = curve.BasisCurve()
    kind = type(curve).__name__
    if kind == 'Geom2d_Line':
        point, direction = curve.Location(), curve.Direction()
        values = [point.X(), point.Y(), direction.X(), direction.Y()]
    elif kind == 'Geom2d_Circle':
        point, x, y = curve.Location(), curve.XAxis().Direction(), curve.YAxis().Direction()
        values = [point.X(), point.Y(), x.X(), x.Y(), y.X(), y.Y(), curve.Radius()]
    elif kind == 'Geom2d_BSplineCurve':
        values = [curve.Degree(), curve.NbPoles(), curve.NbKnots(), int(curve.IsPeriodic())]
        for index in range(1, curve.NbPoles() + 1):
            point = curve.Pole(index)
            values.extend((point.X(), point.Y(), curve.Weight(index)))
        for index in range(1, curve.NbKnots() + 1):
            values.extend((curve.Knot(index), curve.Multiplicity(index)))
    else:
        raise ValueError(f'Cylindrical trim proof does not support {kind}')
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError('Cylindrical trim proof requires finite curve coefficients')
    return kind, values


def _same_pcurve(left, right):
    a, x = _pcurve_signature(left)
    b, y = _pcurve_signature(right)
    # Equality of a few nearby coefficients is insufficient for a rational
    # pcurve or arbitrarily narrow knot spans. The native divider preserves
    # these complete curve definitions; reject reconstructed alternatives.
    return a == b and x.shape == y.shape and np.array_equal(x, y)


def _cylinder_coordinate_function(face):
    if face.geomType() != 'CYLINDER':
        raise ValueError('Trimmed angular partition proof requires an analytic cylinder')
    cylinder = BRepAdaptor_Surface(face.wrapped).Cylinder()
    position = cylinder.Position()
    values = [cylinder.Radius()]
    # Include both radial frame directions, not merely the axis and radius:
    # equal UV parameters must describe the same analytic point.
    for vector in (position.Location(), position.Direction(), position.XDirection(), position.YDirection()):
        values.extend((vector.X(), vector.Y(), vector.Z()))
    return np.asarray(values, dtype=float)


def _assert_interval_coverage(intervals, lower, upper):
    cursor = lower
    for first, last in sorted(intervals):
        if abs(first - cursor) > _PARAMETER_TOLERANCE or last <= first:
            raise ValueError('Cylindrical partition lost or duplicated a trim interval')
        cursor = last
    if abs(cursor - upper) > _PARAMETER_TOLERANCE:
        raise ValueError('Cylindrical partition did not cover an original trim interval')


def assert_trimmed_cylindrical_partition(original, divided) -> dict:
    """Require the same cylinder, oriented trim chains, and angular coverage.

    ``divided`` can be a native shape exposing Faces() or a list of child
    faces from the divider's history. Original 3D curve handles must survive;
    unsupported or reconstructed trim geometry fails explicitly. This proves
    a surface partition, not whole-body validity, global embedding or meshing.
    Those independent checks remain the caller's responsibility.
    """
    children = divided.Faces() if hasattr(divided, 'Faces') else list(divided)
    if len(children) < 2:
        raise ValueError('Cylindrical partition requires multiple child faces')
    reference = _cylinder_coordinate_function(original)
    original_edges = _edge_occurrences(original)
    coverage = [[] for _ in original_edges]
    internal = []
    u0, u1, v0, v1 = original._uvBounds()
    angular_intervals = []
    maximum_coordinate_error = 0.
    maximum_geometry_bound = 0.
    for child in children:
        if not child.isValid() or child.wrapped.Orientation() != original.wrapped.Orientation():
            raise ValueError('Cylindrical partition changed face validity or orientation')
        coordinates = _cylinder_coordinate_function(child)
        error = float(np.max(np.abs(reference - coordinates)))
        maximum_coordinate_error = max(maximum_coordinate_error, error)
        if not np.allclose(reference, coordinates, atol=_PARAMETER_TOLERANCE, rtol=1e-13):
            raise ValueError('Cylindrical partition changed its analytic coordinate function')
        difference = reference - coordinates
        geometry_bound = (np.linalg.norm(difference[1:4]) + abs(difference[0])
                          + abs(coordinates[0]) * (np.linalg.norm(difference[7:10])
                                                   + np.linalg.norm(difference[10:13]))
                          + max(abs(v0), abs(v1)) * np.linalg.norm(difference[4:7]))
        maximum_geometry_bound = max(maximum_geometry_bound, float(geometry_bound))
        if geometry_bound > max(1e-8, original.BoundingBox().DiagonalLength * 1e-10):
            raise ValueError('Cylindrical partition changed its same-parameter surface geometry')
        first_u, last_u, first_v, last_v = child._uvBounds()
        if not np.isfinite((first_u, last_u, first_v, last_v)).all() or (
                first_v < v0 - _PARAMETER_TOLERANCE or last_v > v1 + _PARAMETER_TOLERANCE):
            raise ValueError('Cylindrical partition left its original axial bounds')
        angular_intervals.append((first_u, last_u))
        for edge, curve, pcurve, interval in _edge_occurrences(child):
            matches = [index for index, (old_edge, old_curve, old_pcurve, old_interval)
                       in enumerate(original_edges)
                       if curve == old_curve and edge.Orientation() == old_edge.Orientation()
                       and _same_pcurve(pcurve, old_pcurve)
                       and interval[0] >= old_interval[0] - _PARAMETER_TOLERANCE
                       and interval[1] <= old_interval[1] + _PARAMETER_TOLERANCE]
            if len(matches) > 1:
                raise ValueError('Cylindrical partition has ambiguous original trim correspondence')
            if matches:
                coverage[matches[0]].append(interval)
            else:
                internal.append((edge, pcurve, interval))
    _assert_interval_coverage(angular_intervals, u0, u1)
    for (_, _, _, interval), pieces in zip(original_edges, coverage):
        _assert_interval_coverage(pieces, *interval)
    internal_pair_count = 0
    while internal:
        edge, pcurve, interval = internal.pop()
        kind, coefficients = _pcurve_signature(pcurve)
        if kind != 'Geom2d_Line' or abs(coefficients[2]) > 1e-12:
            raise ValueError('Cylindrical partition introduced a non-isoparametric boundary')
        wanted = TopAbs_REVERSED if edge.Orientation() == TopAbs_FORWARD else TopAbs_FORWARD
        matches = [index for index, (other, other_pcurve, other_range) in enumerate(internal)
                   if edge.IsSame(other) and other.Orientation() == wanted
                   and interval == other_range and _same_pcurve(pcurve, other_pcurve)]
        if len(matches) != 1:
            raise ValueError('Cylindrical partition internal seam does not cancel exactly')
        internal.pop(matches[0])
        internal_pair_count += 1
    return {'method': 'Same analytic cylinder and exact oriented native trim-chain subdivision',
            'source_trim_domain_covered_exactly_once': True,
            'original_trim_occurrence_count': len(original_edges),
            'subdivided_trim_interval_count': sum(map(len, coverage)),
            'new_internal_iso_u_edge_pair_count': internal_pair_count,
            'maximum_cylinder_coordinate_error': maximum_coordinate_error,
            'parameter_comparison_tolerance': _PARAMETER_TOLERANCE,
            'source_uv_coverage_exactly_once': True,
            'maximum_matching_uv_geometry_error': maximum_geometry_bound}
