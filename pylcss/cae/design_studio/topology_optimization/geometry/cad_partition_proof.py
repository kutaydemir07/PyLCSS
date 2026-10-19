# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE.
"""Verify a contained CAD partition after removing redundant face seams.

OCCT's volume integral can count inconsistent pcurves on internal Boolean
seams even when their physical edges cancel.  This narrowly scoped fallback
removes only same-domain face partitions.  It proves that each oriented outer
edge is unchanged, keeps every supporting surface, and compares the resulting
closed material using both the usual volume and explicit common origins.

The caller must already have a valid, empty outside partition.  This module
does not heal exported CAD or relax a containment tolerance.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import io
import math

import numpy as np


def _brep_digest(shape):
    stream = io.BytesIO()
    shape.exportBrep(stream)
    return hashlib.sha256(stream.getvalue()).hexdigest()


def _edge_geometry(edge):
    """Snapshot the exact native 3-D curve, parameter interval, and vertices."""
    import cadquery as cq
    from OCP.BRep import BRep_Tool
    from OCP.BRepAdaptor import BRepAdaptor_Curve
    from OCP.GeomTools import GeomTools
    from OCP.TopoDS import TopoDS

    edge = TopoDS.Edge_s(edge)
    vertices = tuple(vertex.toTuple() for vertex in cq.Shape.cast(edge).Vertices())
    if BRep_Tool.Degenerated_s(edge):
        return ('degenerate', vertices)
    adaptor = BRepAdaptor_Curve(edge)
    stream = io.BytesIO()
    GeomTools.Write_s(adaptor.Curve().Curve(), stream)
    return (stream.getvalue(), adaptor.FirstParameter(), adaptor.LastParameter(), vertices)


def _same_signature(first, second):
    kind, values, tolerances = first
    other_kind, other_values, _ = second
    return (kind == other_kind and values.shape == other_values.shape
            and np.all(np.abs(values - other_values)
                       <= tolerances + 1e-12 * np.abs(values)))


def _grouped_boundary(shape, signatures, edges, *, existing_only):
    """Cancel only oppositely oriented identical native edges on one surface."""
    from OCP.TopAbs import TopAbs_EDGE, TopAbs_FORWARD, TopAbs_REVERSED
    from OCP.TopExp import TopExp_Explorer
    from .step_export import _geometry_signature

    boundaries, areas = defaultdict(Counter), defaultdict(float)
    for face in shape.Faces():
        signature = _geometry_signature(face, 1e-10)
        matches = [i for i, previous in enumerate(signatures)
                   if _same_signature(signature, previous)]
        if matches:
            group = matches[0]
        elif existing_only:
            raise RuntimeError('Partition normalization introduced a supporting surface')
        else:
            group = len(signatures)
            signatures.append(signature)
        areas[group] += face.Area()
        explorer = TopExp_Explorer(face.wrapped, TopAbs_EDGE)
        while explorer.More():
            edge = explorer.Current()
            orientation = edge.Orientation()
            if orientation not in (TopAbs_FORWARD, TopAbs_REVERSED):
                raise RuntimeError('Partition boundary has an unsupported edge orientation')
            # Hash only selects a bucket: IsSame proves identity even in the
            # event of a native hash collision. Orientation is counted below.
            bucket = edges[hash(edge)]
            index = next((index for candidate, index in bucket
                          if edge.IsSame(candidate)), None)
            if index is None:
                if existing_only:
                    raise RuntimeError('Partition normalization replaced an outer edge')
                index = sum(len(items) for items in edges.values())
                bucket.append((edge, index))
            boundaries[group][index] += 1 if orientation == TopAbs_FORWARD else -1
            explorer.Next()
    return ({group: {edge: count for edge, count in boundary.items() if count}
             for group, boundary in boundaries.items()}, dict(areas))


def _normalize_face_partitions(shape):
    import cadquery as cq
    from OCP.BRepBuilderAPI import BRepBuilderAPI_Copy
    from OCP.Precision import Precision
    from OCP.ShapeUpgrade import ShapeUpgrade_UnifySameDomain
    from .occ_shapes import _assess_brep_self_intersections
    from .step_export import _closed

    if not shape.isValid() or not _closed(shape):
        raise RuntimeError('Partition normalization requires valid closed solid CAD')
    original_digest = _brep_digest(shape)
    # Copy topology while retaining shared surface handles. Copying geometry
    # separately for each face loses the identity of a partitioned spline,
    # preventing exact same-domain cancellation. The geometry is audited
    # below, and the caller's complete BREP must remain unchanged.
    private = cq.Shape.cast(BRepBuilderAPI_Copy(shape.wrapped, False, False).Shape())
    signatures, edges = [], defaultdict(list)
    before, areas_before = _grouped_boundary(private, signatures, edges, existing_only=False)
    edge_geometry = {index: _edge_geometry(edge)
                     for bucket in edges.values() for edge, index in bucket}
    unifier = ShapeUpgrade_UnifySameDomain(private.wrapped, False, True, False)
    unifier.SetSafeInputMode(True)
    unifier.SetLinearTolerance(Precision.Confusion_s())
    unifier.SetAngularTolerance(1e-10)
    unifier.AllowInternalEdges(False)
    unifier.Build()
    result = cq.Shape.cast(unifier.Shape())
    if _brep_digest(shape) != original_digest:
        raise RuntimeError('Partition normalization mutated its input CAD')
    # The unifier adds pcurves for its merged face to edges of the private
    # copy, even in safe-input mode. Their physical curves/vertices must stay
    # byte-for-byte unchanged; the caller's original BREP remains untouched.
    if any(_edge_geometry(edge) != edge_geometry[index]
           for bucket in edges.values() for edge, index in bucket):
        raise RuntimeError('Partition normalization changed a physical edge')
    if (not result.isValid() or not _closed(result)
            or len(result.Solids()) != len(shape.Solids())):
        raise RuntimeError('Partition normalization did not retain valid closed material')
    after, areas_after = _grouped_boundary(result, signatures, edges, existing_only=True)
    if before != after:
        raise RuntimeError('Partition normalization changed an oriented outer boundary')
    if set(areas_before) != set(areas_after) or any(
            not np.isfinite(area) or area <= 0 for area in areas_after.values()):
        raise RuntimeError('Partition normalization lost a supporting-surface group')
    intersections = _assess_brep_self_intersections(result)
    if not intersections.get('passed', False):
        raise RuntimeError('Normalized partition failed native self-intersection analysis')
    return result, {
        'original_faces': len(shape.Faces()), 'normalized_faces': len(result.Faces()),
        'supporting_surface_groups': len(signatures),
        'oriented_outer_edges_unchanged': True,
        'inputs_unchanged': True, 'native_self_intersections': intersections,
    }


def _matching_normalized_surfaces(reference, candidate):
    """Compare physical area on each unchanged, normalized supporting face."""
    from .step_export import _geometry_signature

    unmatched = [(_geometry_signature(face, 1e-10), face.Area())
                 for face in candidate.Faces()]
    if len(unmatched) != len(reference.Faces()):
        raise RuntimeError('Normalized Common has a different supporting-surface multiset')
    maximum_area_delta = 0.
    for face in reference.Faces():
        signature, area = _geometry_signature(face, 1e-10), face.Area()
        match = next((i for i, (other, other_area) in enumerate(unmatched)
                      if _same_signature(signature, other)
                      and abs(other_area - area) <= max(1e-9, abs(area) * 1e-8)), None)
        if match is None:
            raise RuntimeError('Normalized Common changed a supporting surface or its area')
        _, other_area = unmatched.pop(match)
        maximum_area_delta = max(maximum_area_delta, abs(other_area - area))
    return maximum_area_delta


def _fixed_origin_volume(shape, origin, epsilon):
    from OCP.BRepGProp import BRepGProp_Domain, BRepGProp_Face, BRepGProp_Vinert
    from OCP.gp import gp_Pnt

    reference = gp_Pnt(*map(float, origin))
    values = []
    for face in shape.Faces():
        properties = BRepGProp_Vinert(
            BRepGProp_Face(face.wrapped, True), BRepGProp_Domain(face.wrapped),
            reference, reference, epsilon)
        values.append(properties.Mass())
    volume = math.fsum(values)
    if not np.isfinite(volume):
        raise RuntimeError('Common-origin volume is not finite')
    return volume


def normalize_contained_partition(material, inside, *, volume_tolerance, linear_tolerance):
    """Return an independently verified Common; the caller must prove empty Cut.

    This is only for an otherwise valid closed Common whose ordinary volume
    disagrees with the requested material.  It does not modify either input or
    change the requested material and requires all original volume/bounds
    tolerances, including positive oriented volume, to remain satisfied.
    """
    from .cad_features import _material_bounds, _material_bounds_delta

    if (not np.isfinite((volume_tolerance, linear_tolerance)).all()
            or volume_tolerance <= 0 or linear_tolerance <= 0):
        raise RuntimeError('Partition proof needs finite positive tolerances')
    # A local preservation witness must not turn into a second reconstruction
    # or a long-running full-design healing pass.
    if len(material.Faces()) > 64 or len(inside.Faces()) > 512:
        raise RuntimeError('Partition proof exceeds the local material face limit')
    reference, reference_proof = _normalize_face_partitions(material)
    candidate, candidate_proof = _normalize_face_partitions(inside)
    maximum_area_delta = _matching_normalized_surfaces(reference, candidate)
    volume = float(material.Volume(tol=1e-7))
    if not np.isfinite(volume) or volume <= 0 or volume_tolerance <= 0 or linear_tolerance <= 0:
        raise RuntimeError('Partition proof needs positive finite material and tolerances')
    normal_volumes = [float(part.Volume(tol=1e-7)) for part in (reference, candidate)]
    if any(not np.isfinite(value) or value <= 0 or abs(value - volume) > volume_tolerance
           for value in normal_volumes):
        raise RuntimeError('Normalized inside partition did not retain complete material volume')
    for part in (reference, candidate):
        if np.max(np.abs(_material_bounds_delta(material, part, linear_tolerance))) > linear_tolerance:
            raise RuntimeError('Normalized partition changed the material bounds')

    bounds = _material_bounds(material)
    center = (bounds[::2] + bounds[1::2]) * .5
    scale = max(float(np.linalg.norm(bounds[1::2] - bounds[::2])), linear_tolerance)
    origins = [center, *(center + scale * axis for axis in np.eye(3))]
    comparisons = []
    for origin in origins:
        values = []
        for part in (material, reference, candidate):
            coarse = _fixed_origin_volume(part, origin, 1e-9)
            fine = _fixed_origin_volume(part, origin, 1e-11)
            if abs(fine - coarse) > volume_tolerance * .01:
                raise RuntimeError('Common-origin volume did not converge')
            values.append(fine)
        if (any(value <= 0 for value in values)
                or max(abs(value - values[0]) for value in values[1:]) > volume_tolerance):
            raise RuntimeError('Common-origin integrals disagree on complete material volume')
        comparisons.append({'origin': origin.tolist(), 'original': values[0],
                            'normalized_original': values[1], 'normalized_common': values[2]})
    return candidate, {
        'method': 'Exact same-domain face partition cancellation with common-origin integrals',
        'material': reference_proof, 'common': candidate_proof,
        'original_volume': volume, 'normalized_volumes': normal_volumes,
        'volume_tolerance': float(volume_tolerance),
        'linear_tolerance': float(linear_tolerance),
        'maximum_normalized_face_area_delta': maximum_area_delta,
        'common_origin_comparisons': comparisons,
    }
