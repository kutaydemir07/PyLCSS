# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.
"""Exact unit parameter domains for rectangular polynomial CAD faces."""
from __future__ import annotations

import cadquery as cq
import numpy as np
from OCP.BRepAdaptor import BRepAdaptor_Surface
from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace
from OCP.TColStd import TColStd_Array1OfReal
from OCP.TopAbs import TopAbs_REVERSED

from .amrto_volume import (
    UnsupportedVolumeGeometry, _polynomial_difference_bound,
    _rectangular_face_data, rectangular_spline_volume,
)


def normalize_paper_parameter_domains(shape, *, tolerance: float = 1e-5):
    """Rescale complete rectangular spline domains without changing geometry.

    Each original rectangle is proven from its trim wires. A Bernstein bound
    verifies the entire affine-matched polynomial surface, followed by strict
    re-sewing and independent resolved-knot volume integration. Mixed or
    arbitrarily trimmed shapes are ineligible and returned unchanged.
    """
    shape = shape.val() if hasattr(shape, 'val') else shape
    report = {'method': 'Exact affine unit parameter domains', 'applied': False,
              'surface_fits': 0, 'physical_geometry_changed': False}
    try:
        original_data = [_rectangular_face_data(face) for face in shape.Faces()]
        volume_before, _ = rectangular_spline_volume(shape)
    except UnsupportedVolumeGeometry as error:
        report.update(eligible=False, reason=str(error))
        return shape, report
    report['eligible'] = True
    if all(data['bounds'] == (0., 1., 0., 1.) for data in original_data):
        return shape, report
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError('Parameter normalization requires a positive sewing tolerance')
    faces, maximum = [], 0.
    for face, source in zip(shape.Faces(), original_data):
        u0, u1, v0, v1 = source['bounds']
        surface = BRepAdaptor_Surface(face.wrapped, True).BSpline().Copy()
        surface.Segment(u0, u1, v0, v1)
        for axis, lower, upper in (('U', u0, u1), ('V', v0, v1)):
            count = getattr(surface, 'Nb' + axis + 'Knots')()
            knots = TColStd_Array1OfReal(1, count)
            for index in range(1, count + 1):
                knots.SetValue(index, (getattr(surface, axis + 'Knot')(index) - lower) / (upper - lower))
            getattr(surface, 'Set' + axis + 'Knots')(knots)
        native = BRepBuilderAPI_MakeFace(surface, 0., 1., 0., 1., 1e-7).Face()
        if face.wrapped.Orientation() == TopAbs_REVERSED:
            native.Reverse()
        normalized = cq.Face(native)
        restored = _rectangular_face_data(normalized)
        restored['knots'] = tuple(lower + knots * (upper - lower)
                                  for knots, lower, upper in zip(restored['knots'], (u0, v0), (u1, v1)))
        restored['bounds'] = source['bounds']
        bound = _polynomial_difference_bound(source, restored)
        scale = float(np.linalg.norm(np.ptp(source['poles'].reshape(-1, 3), axis=0)))
        if source['sign'] != restored['sign'] or bound > max(1e-8, scale * 1e-10):
            raise RuntimeError('Affine parameter normalization changed a polynomial CAD face')
        maximum = max(maximum, bound)
        faces.append(normalized)
    points = np.concatenate([data['poles'].reshape(-1, 3) for data in original_data])
    scale = float(np.max(np.ptp(points, axis=0)))
    from .amrto_solid import stitch_paper_surfaces
    normalized, sewing = stitch_paper_surfaces(
        cq.Compound.makeCompound(faces), tolerance=tolerance, source_scale=scale,
        maximum_relative_tolerance=tolerance / scale, source_volume=volume_before)
    if not sewing['is_solid'] or len(normalized.Faces()) != len(faces):
        raise RuntimeError('Unit parameter domains did not retain the complete closed CAD body')
    volume_after, _ = rectangular_spline_volume(normalized)
    if abs(volume_after - volume_before) > max(1e-7, volume_before * 1e-9):
        raise RuntimeError('Affine parameter normalization changed the signed CAD boundary integral')
    report.update(applied=True, face_count=len(faces),
                  complete_rectangular_trim_domains_preserved=True,
                  whole_polynomial_geometry_verified=True,
                  maximum_bernstein_geometry_bound=maximum,
                  signed_integral_before=volume_before, signed_integral_after=volume_after,
                  relative_integral_difference=abs(volume_after - volume_before) / volume_before,
                  sewing=sewing)
    return normalized, report
