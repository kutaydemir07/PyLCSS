# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.
"""Strict certificates for exact box domains with disjoint through bores.

The certificate enables ordinary set identities to be evaluated through
simpler exact primitives. It never declares a reconstructed body contained:
the caller still has to clip/check its box exterior and every bore overlap.
"""
from __future__ import annotations

import math
from typing import Any

import cadquery as cq
import numpy as np
from OCP.BRep import BRep_Tool
from OCP.BRepAdaptor import BRepAdaptor_Surface
from OCP.BRepAlgoAPI import BRepAlgoAPI_Common, BRepAlgoAPI_Cut
from OCP.BRepBndLib import BRepBndLib
from OCP.Bnd import Bnd_Box
from OCP.TopAbs import TopAbs_REVERSED
from OCP.TopTools import TopTools_ListOfShape
from OCP.gp import gp_Pnt, gp_Vec


class UnsupportedSourceDomain(ValueError):
    """The source has not met the complete primitive-domain eligibility rules."""


def _native_exact(first, second, operation):
    arguments, tools = TopTools_ListOfShape(), TopTools_ListOfShape()
    arguments.Append(first.wrapped)
    tools.Append(second.wrapped)
    builder = {'cut': BRepAlgoAPI_Cut, 'common': BRepAlgoAPI_Common}[operation]()
    builder.SetArguments(arguments)
    builder.SetTools(tools)
    builder.SetNonDestructive(True)
    builder.SetFuzzyValue(0.0)
    builder.SetRunParallel(False)
    builder.Build()
    if not builder.IsDone() or builder.Shape().IsNull():
        raise RuntimeError('Exact source-domain equivalence Boolean did not complete')
    return cq.Shape.cast(builder.Shape())


def _shape_report(shape) -> dict:
    return {'valid': bool(shape.isValid()), 'solid_count': len(shape.Solids()),
            'shell_count': len(shape.Shells()), 'face_count': len(shape.Faces()),
            'edge_count': len(shape.Edges()), 'vertex_count': len(shape.Vertices()),
            'all_shells_closed': all(BRep_Tool.IsClosed_s(shell.wrapped) for shell in shape.Shells()),
            'volume': abs(float(shape.Volume()))}


def _require_single_closed(shape) -> dict:
    report = _shape_report(shape)
    if (not report['valid'] or report['solid_count'] != 1 or report['shell_count'] != 1
            or not report['all_shells_closed'] or not math.isfinite(report['volume'])
            or report['volume'] <= 0):
        raise RuntimeError('Source-domain equivalence requires a valid single closed solid')
    solid = shape.Solids()[0]
    if (len(shape.Faces()) != len(solid.Faces()) or len(shape.Edges()) != len(solid.Edges())
            or len(shape.Vertices()) != len(solid.Vertices())):
        raise RuntimeError('Source-domain equivalence rejects loose topology outside the material solid')
    return report


def _require_topologically_empty(shape) -> dict:
    report = _shape_report(shape)
    if (not report['valid'] or report['volume'] != 0
            or any(report[name] for name in ('solid_count', 'shell_count', 'face_count',
                                             'edge_count', 'vertex_count'))):
        raise RuntimeError('Source/canonical difference is not valid and completely empty')
    return report


def _physical_normal(adaptor, bounds, orientation):
    point, du, dv = gp_Pnt(), gp_Vec(), gp_Vec()
    adaptor.D1(.5 * (bounds[0] + bounds[1]), .5 * (bounds[2] + bounds[3]), point, du, dv)
    normal = np.cross(np.asarray(du.Coord()), np.asarray(dv.Coord())) * orientation
    magnitude = np.linalg.norm(normal)
    if not np.isfinite(magnitude) or magnitude <= 0:
        raise UnsupportedSourceDomain('Source analytic face has undefined physical orientation')
    return np.asarray(point.Coord()), normal / magnitude


def _coordinate_axis(direction):
    direction = np.asarray(direction, dtype=float)
    axis = int(np.argmax(np.abs(direction)))
    if (np.max(np.abs(np.delete(direction, axis))) > 1e-12
            or abs(abs(direction[axis]) - 1) > 1e-12):
        raise UnsupportedSourceDomain('Source primitive axes are not aligned with the box coordinates')
    return axis


def certify_box_through_bores(source: Any, *, geometric_tolerance: float = 1e-7):
    """Return ``(box, bore_tools, report)`` after strict exact-set equivalence.

    Eligibility requires exactly six outward axis-aligned planar faces and
    complete inward cylinder faces for disjoint, internal, parallel through
    bores. Direct/indirect native frames are handled through physical D1
    normals. Every plane level, axial limit, face area, primitive separation,
    and material topology is checked. Native differences must be valid and
    completely empty in BOTH directions; both Commons must remain complete
    closed material with the analytic box-minus-bores volume. No fuzzy
    tolerance, approximation, or reconstructed-body acceptance occurs here.
    """
    if not math.isfinite(geometric_tolerance) or geometric_tolerance <= 0:
        raise ValueError('Source-domain geometric tolerance must be finite and positive')
    if hasattr(source, 'val'):
        source = source.val()
    if not hasattr(source, 'wrapped'):
        source = cq.Shape.cast(source)
    try:
        source_report = _require_single_closed(source)
    except RuntimeError as error:
        raise UnsupportedSourceDomain(str(error)) from error
    planes, bores = [], []
    for index, face in enumerate(source.Faces()):
        adaptor = BRepAdaptor_Surface(face.wrapped, True)
        orientation = -1 if face.wrapped.Orientation() == TopAbs_REVERSED else 1
        uv = face._uvBounds()
        point, normal = _physical_normal(adaptor, uv, orientation)
        if face.geomType() == 'PLANE':
            plane = adaptor.Plane()
            axis = _coordinate_axis(plane.Axis().Direction().Coord())
            location = np.asarray(plane.Location().Coord())
            planes.append({'face': index, 'axis': axis, 'level': float(location[axis]),
                           'outward_normal': normal.tolist(), 'area': float(face.Area()),
                           'wire_count': len(face.Wires())})
        elif face.geomType() == 'CYLINDER':
            cylinder = adaptor.Cylinder()
            direction = np.asarray(cylinder.Axis().Direction().Coord())
            axis = _coordinate_axis(direction)
            if abs(uv[1] - uv[0] - 2 * math.pi) > 1e-10:
                raise UnsupportedSourceDomain('Source bore does not span exactly one complete cylinder turn')
            location = np.asarray(cylinder.Location().Coord())
            radial = point - location
            radial -= direction * (radial @ direction)
            radial /= np.linalg.norm(radial)
            if float(normal @ radial) > -1 + 1e-12:
                raise UnsupportedSourceDomain('Source cylinder is not an inward-facing material void')
            axial = sorted(adaptor.Value(uv[0], value).Coord()[axis] for value in (uv[2], uv[3]))
            center = location.copy()
            center[axis] = 0.0
            bores.append({'face': index, 'axis': axis, 'axis_origin': center.tolist(),
                          'radius': float(cylinder.Radius()), 'axial_bounds': axial,
                          'area': float(face.Area()), 'wire_count': len(face.Wires()),
                          'native_frame_direct': bool(cylinder.Position().Direct())})
        else:
            raise UnsupportedSourceDomain('Source includes surfaces other than the six box planes and full cylindrical voids')
    if len(planes) != 6 or not bores or len(planes) + len(bores) != len(source.Faces()):
        raise UnsupportedSourceDomain('Source is not a six-plane box with complete through bores')
    lower, upper = np.empty(3), np.empty(3)
    for axis in range(3):
        pair = sorted((plane for plane in planes if plane['axis'] == axis), key=lambda plane: plane['level'])
        if len(pair) != 2 or pair[0]['level'] >= pair[1]['level']:
            raise UnsupportedSourceDomain('Source needs exactly two distinct bounding planes per coordinate axis')
        lower[axis], upper[axis] = pair[0]['level'], pair[1]['level']
        for plane, sign in zip(pair, (-1, 1)):
            if not np.allclose(plane['outward_normal'], np.eye(3)[axis] * sign, rtol=0, atol=1e-12):
                raise UnsupportedSourceDomain('Source plane orientation does not bound the declared material box')
    box_bounds = Bnd_Box()
    BRepBndLib.AddOptimal_s(source.wrapped, box_bounds, False, False)
    if not np.allclose(box_bounds.Get(), np.r_[lower, upper], rtol=0, atol=geometric_tolerance):
        raise UnsupportedSourceDomain('Analytic box planes and source bounds disagree')
    axes = {bore['axis'] for bore in bores}
    if len(axes) != 1:
        raise UnsupportedSourceDomain('Certificate requires parallel through bores')
    through_axis = next(iter(axes))
    transverse = [axis for axis in range(3) if axis != through_axis]
    height = float(upper[through_axis] - lower[through_axis])
    lengths = upper - lower
    for bore in bores:
        center, radius = np.asarray(bore['axis_origin']), bore['radius']
        if (radius <= 0 or np.any(center[transverse] - radius <= lower[transverse] + geometric_tolerance)
                or np.any(center[transverse] + radius >= upper[transverse] - geometric_tolerance)
                or not np.allclose(bore['axial_bounds'], [lower[through_axis], upper[through_axis]],
                                   rtol=0, atol=geometric_tolerance)):
            raise UnsupportedSourceDomain('Source cylinder is not a complete, strictly internal through bore')
        expected_area = 2 * math.pi * radius * height
        if abs(bore['area'] - expected_area) > max(1e-8, expected_area * 1e-10):
            raise UnsupportedSourceDomain('Source bore trim does not cover its entire cylindrical area')
    for index, first in enumerate(bores):
        for second in bores[index + 1:]:
            separation = np.linalg.norm((np.asarray(first['axis_origin']) - second['axis_origin'])[transverse])
            if separation <= first['radius'] + second['radius'] + geometric_tolerance:
                raise UnsupportedSourceDomain('Through bores overlap or touch')
    hole_area = sum(math.pi * bore['radius'] ** 2 for bore in bores)
    for plane in planes:
        expected = float(np.prod(np.delete(lengths, plane['axis'])))
        expected_wires = 1
        if plane['axis'] == through_axis:
            expected -= hole_area
            expected_wires += len(bores)
        if (abs(plane['area'] - expected) > max(1e-8, expected * 1e-10)
                or plane['wire_count'] != expected_wires):
            raise UnsupportedSourceDomain('Source plane trim is not the declared full box side with its through holes')
    box = cq.Solid.makeBox(*lengths, cq.Vector(*lower))
    canonical, tools = box, []
    for bore in bores:
        base = np.asarray(bore['axis_origin']).copy()
        base[through_axis] = lower[through_axis]
        tool = cq.Solid.makeCylinder(bore['radius'], height, cq.Vector(*base), cq.Vector(*np.eye(3)[through_axis]))
        tools.append(tool)
        canonical = _native_exact(canonical, tool, 'cut')
        _require_single_closed(canonical)
    canonical_report = _require_single_closed(canonical)
    analytic_volume = float(np.prod(lengths) - hole_area * height)
    volume_tolerance = max(1e-7, analytic_volume * 1e-10)
    for report in (source_report, canonical_report):
        if abs(report['volume'] - analytic_volume) > volume_tolerance:
            raise RuntimeError('Source/canonical material volume differs from exact analytic primitive volume')
    differences, intersections = [], []
    for label, first, second in (('source_minus_canonical', source, canonical),
                                  ('canonical_minus_source', canonical, source)):
        differences.append({'operation': label,
                            'result': _require_topologically_empty(_native_exact(first, second, 'cut'))})
    for label, first, second in (('source_common_canonical', source, canonical),
                                  ('canonical_common_source', canonical, source)):
        report = _require_single_closed(_native_exact(first, second, 'common'))
        if abs(report['volume'] - analytic_volume) > volume_tolerance:
            raise RuntimeError('Full closed source/canonical Common lost source material volume')
        intersections.append({'operation': label, 'result': report})
    report = {'method': 'Exact six-plane box minus disjoint full through-cylinder domain certificate',
              'accepted': True, 'source_domain_set_equivalence': True,
              'box_min': lower.tolist(), 'box_max': upper.tolist(), 'through_axis': through_axis,
              'plane_faces': planes, 'bores': bores, 'source': source_report,
              'canonical': canonical_report, 'analytic_volume': analytic_volume,
              'geometric_tolerance_mm': float(geometric_tolerance),
              'volume_tolerance': volume_tolerance, 'directed_empty_differences': differences,
              'directed_complete_closed_intersections': intersections,
              'native_fuzzy_tolerance': 0.0, 'body_containment_not_assessed': True,
              'set_identity': 'Source = regularized difference of Box and the union of the disjoint through-cylinder solids',
              'solid_set_semantics': 'Regularized CAD solid operations; boundary contact alone is not material-volume intrusion'}
    return box, tools, report
