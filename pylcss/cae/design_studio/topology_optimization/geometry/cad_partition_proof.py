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


def _grouped_boundary(shape, signatures, edges, *, existing_only, ignored_internal_edges=()):
    """Cancel only oppositely oriented identical native edges on one surface."""
    from OCP.TopAbs import TopAbs_EDGE, TopAbs_FORWARD, TopAbs_REVERSED, TopAbs_INTERNAL
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
            if (orientation == TopAbs_INTERNAL
                    and any(edge.IsSame(trace) for trace in ignored_internal_edges)):
                explorer.Next()
                continue
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


def _internal_trace_wires(shape):
    """Find isolated open INTERNAL traces that cannot be directed boundaries.

    Only a wire containing one nondegenerate INTERNAL edge is eligible. Its
    distinct endpoints must be separated beyond their native tolerances, and
    the edge must occur exactly once in the complete shape. Mixed wires,
    closed loops and edges reused by a directed boundary are never removed.
    """
    import cadquery as cq
    from OCP.BRep import BRep_Tool
    from OCP.TopAbs import TopAbs_EDGE, TopAbs_WIRE, TopAbs_INTERNAL
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS, TopoDS_Iterator

    occurrences = defaultdict(list)
    # Count every occurrence in the complete shape, including standalone
    # wires/edges outside its faces. ReShape removal acts on those too.
    explorer = TopExp_Explorer(shape.wrapped, TopAbs_EDGE)
    while explorer.More():
        edge = explorer.Current()
        occurrences[hash(edge)].append(edge)
        explorer.Next()
    internal = []
    for face in shape.Faces():
        explorer = TopExp_Explorer(face.wrapped, TopAbs_EDGE)
        while explorer.More():
            edge = explorer.Current()
            if edge.Orientation() == TopAbs_INTERNAL:
                internal.append(edge)
            explorer.Next()
    if not internal:
        return [], []
    if len(internal) > 32:
        raise RuntimeError('Partition proof exceeds the local INTERNAL trace limit')
    wires, edges = [], []
    for face in shape.Faces():
        children = TopoDS_Iterator(face.wrapped)
        while children.More():
            wire = children.Value()
            children.Next()
            if wire.ShapeType() != TopAbs_WIRE:
                continue
            members, iterator = [], TopoDS_Iterator(wire)
            while iterator.More():
                members.append(iterator.Value())
                iterator.Next()
            if not any(member.Orientation() == TopAbs_INTERNAL for member in members):
                continue
            if (len(members) != 1 or members[0].ShapeType() != TopAbs_EDGE
                    or members[0].Orientation() != TopAbs_INTERNAL or wire.Closed()):
                raise RuntimeError('Partition trace must be a single open INTERNAL edge wire')
            edge = TopoDS.Edge_s(members[0])
            vertices = cq.Shape.cast(edge).Vertices()
            if BRep_Tool.Degenerated_s(edge) or len(vertices) != 2:
                raise RuntimeError('Partition trace must be a single open INTERNAL edge wire')
            first, last = (BRep_Tool.Pnt_s(vertex.wrapped) for vertex in vertices)
            tolerance = sum(BRep_Tool.Tolerance_s(vertex.wrapped) for vertex in vertices)
            separation = first.Distance(last)
            if (not np.isfinite((separation, tolerance)).all()
                    or tolerance < 0 or separation <= tolerance):
                raise RuntimeError('Partition trace must be a single open INTERNAL edge wire')
            uses = [item for item in occurrences[hash(edge)] if item.IsSame(edge)]
            if len(uses) != 1 or any(item.Orientation() != TopAbs_INTERNAL for item in uses):
                raise RuntimeError('INTERNAL trace is reused by a directed boundary or multiple occurrences')
            wires.append(wire)
            edges.append(edge)
    if len(edges) != len(internal) or any(
            not any(edge.IsSame(candidate) for candidate in edges) for edge in internal):
        raise RuntimeError('INTERNAL occurrence is not an isolated open trace wire')
    return wires, edges


def _normalize_face_partitions(shape):
    import cadquery as cq
    from OCP.BRepBuilderAPI import BRepBuilderAPI_Copy
    from OCP.Precision import Precision
    from OCP.ShapeBuild import ShapeBuild_ReShape
    from OCP.ShapeUpgrade import ShapeUpgrade_UnifySameDomain
    from .cad_features import _material_bounds_delta
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
    trace_wires, trace_edges = _internal_trace_wires(private)
    signatures, edges = [], defaultdict(list)
    before, areas_before = _grouped_boundary(
        private, signatures, edges, existing_only=False, ignored_internal_edges=trace_edges)
    edge_geometry = {index: _edge_geometry(edge)
                     for bucket in edges.values() for edge, index in bucket}
    if trace_wires:
        # These audited INTERNAL wires have no directed physical boundary.
        # Remove them only from the private containment witness, preserving
        # every actual outer/hole edge and all supporting surfaces and areas.
        reshaper = ShapeBuild_ReShape()
        for wire in trace_wires:
            reshaper.Remove(wire)
        pruned = cq.Shape.cast(reshaper.Apply(private.wrapped))
        if (not pruned.isValid() or not _closed(pruned)
                or len(pruned.Faces()) != len(private.Faces())
                or len(pruned.Solids()) != len(private.Solids())):
            raise RuntimeError('INTERNAL trace removal did not retain valid closed material')
        boundary, areas = _grouped_boundary(pruned, signatures, edges, existing_only=True)
        if boundary != before or areas != areas_before:
            raise RuntimeError('INTERNAL trace removal changed a directed boundary or surface area')
        linear_tolerance = 2.0 * Precision.Confusion_s()
        if np.max(np.abs(_material_bounds_delta(private, pruned, linear_tolerance))) > linear_tolerance:
            raise RuntimeError('INTERNAL trace removal changed the material bounds')
        private = pruned
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
        'internal_trace_wires_removed': len(trace_wires),
        'internal_trace_edges_removed': len(trace_edges),
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


def _matching_oriented_curve_boundaries(reference, candidate):
    """Prove complete boundary equality despite exact edge subdivisions.

    A supporting curve's full native serialization includes its poles, knots,
    weights and parameterization. Together with the placement, this identifies
    the same physical curve without sampling. Signed interval endpoints
    describe its oriented coverage: adjacent subdivisions cancel exactly at
    their common parameter. No parameter or geometry tolerance is introduced.
    """
    from OCP.BRep import BRep_Tool
    from OCP.BRepAdaptor import BRepAdaptor_Curve
    from OCP.GeomTools import GeomTools
    from OCP.TopAbs import TopAbs_EDGE, TopAbs_FORWARD, TopAbs_REVERSED
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS
    from .step_export import _geometry_signature

    signatures = []

    def coverage(shape, *, existing_only):
        boundaries = defaultdict(Counter)
        edge_count = 0
        for face in shape.Faces():
            signature = _geometry_signature(face, 1e-10)
            matches = [i for i, previous in enumerate(signatures)
                       if _same_signature(signature, previous)]
            if len(matches) > 1:
                raise RuntimeError('Curve-boundary proof has ambiguous supporting surfaces')
            if matches:
                group = matches[0]
            elif existing_only:
                raise RuntimeError('Curve-boundary proof changed a supporting surface')
            else:
                group = len(signatures)
                signatures.append(signature)
            explorer = TopExp_Explorer(face.wrapped, TopAbs_EDGE)
            while explorer.More():
                edge = TopoDS.Edge_s(explorer.Current())
                explorer.Next()
                orientation = edge.Orientation()
                if orientation not in (TopAbs_FORWARD, TopAbs_REVERSED):
                    raise RuntimeError('Curve-boundary proof has an unsupported edge orientation')
                # A pole edge has no regular 3-D parameterized curve. This
                # narrow fallback cannot prove its boundary correspondence.
                if BRep_Tool.Degenerated_s(edge):
                    raise RuntimeError('Curve-boundary proof does not support degenerate edges')
                adaptor = BRepAdaptor_Curve(edge)
                first, last = adaptor.FirstParameter(), adaptor.LastParameter()
                if not np.isfinite((first, last)).all() or last <= first:
                    raise RuntimeError('Curve-boundary proof requires finite ordered intervals')
                # The curve representation can have its own location in
                # addition to the TopoDS edge placement. The adaptor exposes
                # the aggregate transform used by its physical 3-D curve.
                transform = adaptor.Trsf()
                location = tuple(transform.Value(i, j)
                                 for i in range(1, 4) for j in range(1, 5))
                if not np.isfinite(location).all():
                    raise RuntimeError('Curve-boundary proof has a nonfinite placement')
                stream = io.BytesIO()
                GeomTools.Write_s(adaptor.Curve().Curve(), stream)
                key = (group, stream.getvalue(), location)
                sign = 1 if orientation == TopAbs_FORWARD else -1
                boundaries[key][first] += sign
                boundaries[key][last] -= sign
                edge_count += 1
        # Ignore only exactly cancelled, opposite coverage on one surface.
        result = {}
        for key, endpoints in boundaries.items():
            remaining = {parameter: count for parameter, count in endpoints.items() if count}
            if remaining:
                result[key] = remaining
        return result, edge_count

    before, original_count = coverage(reference, existing_only=False)
    after, candidate_count = coverage(candidate, existing_only=True)
    if before != after:
        raise RuntimeError('Normalized Common changed an oriented curve interval boundary')
    return {
        'method': 'Exact serialized curves, placements and signed parameter interval coverage',
        'supporting_surface_groups': len(signatures),
        'curve_boundary_groups': len(before),
        'reference_oriented_edges': original_count,
        'candidate_oriented_edges': candidate_count,
        'complete_oriented_boundaries_identical': True,
        'parameter_comparison_tolerance': 0.0,
    }


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
    change the requested material. It retains the original bounds and
    same-origin volume tolerances, including positive oriented volume. An
    automatic-origin volume discrepancy additionally requires exact complete
    boundary correspondence.
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
    if any(not np.isfinite(value) or value <= 0 for value in normal_volumes):
        raise RuntimeError('Normalized partition has a nonpositive or nonfinite material volume')
    ordinary_volume_disputed = any(abs(value - volume) > volume_tolerance
                                   for value in normal_volumes)
    boundary_proof = None
    exact_boundary_required = (ordinary_volume_disputed
                               or reference_proof['internal_trace_wires_removed']
                               or candidate_proof['internal_trace_wires_removed'])
    if exact_boundary_required:
        # Even after removing face seams, splitting an unchanged outer curve
        # changes OCCT's automatically selected integration reference. Some
        # source collars already have tolerance-level pcurve inconsistencies,
        # making that integral depend on the reference. Accept this case only
        # with complete geometric boundary identity AND all converged,
        # same-origin comparisons below. The original material and volume
        # tolerance remain unchanged; no exported geometry is modified.
        boundary_proof = _matching_oriented_curve_boundaries(reference, candidate)
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
        'ordinary_volume_discrepancies': [value - volume for value in normal_volumes],
        'ordinary_volume_disputed': ordinary_volume_disputed,
        'complete_boundary_equivalence': boundary_proof,
        'volume_verification': ('Exact oriented boundaries and converged same-origin integrals'
                                if exact_boundary_required else
                                'Ordinary volume and converged same-origin integrals'),
        'volume_tolerance': float(volume_tolerance),
        'linear_tolerance': float(linear_tolerance),
        'maximum_normalized_face_area_delta': maximum_area_delta,
        'common_origin_comparisons': comparisons,
    }
