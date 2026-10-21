# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE.
"""Remove bounded, fault-mapped coplanar partitions without moving geometry.

This is a topology normalization, not an acceptance certificate. The caller
must retain all material, source, void, complete native and STEP checks.
"""
from __future__ import annotations

from collections import defaultdict
import io

import numpy as np

from .cad_partition_proof import (
    _brep_digest, _edge_geometry, _grouped_boundary, _same_signature,
)


def _placement(adaptor):
    transform = adaptor.Trsf()
    return tuple(transform.Value(row, column)
                 for row in range(1, 4) for column in range(1, 5))


def _face_geometry(face):
    from OCP.BRepAdaptor import BRepAdaptor_Surface
    from OCP.GeomTools import GeomTools

    adaptor = BRepAdaptor_Surface(face.wrapped, False)
    stream = io.BytesIO()
    GeomTools.Write_s(adaptor.Surface().Surface(), stream)
    return stream.getvalue(), _placement(adaptor)


def _located_edge_geometry(edge):
    from OCP.BRep import BRep_Tool
    from OCP.BRepAdaptor import BRepAdaptor_Curve

    placement = None if BRep_Tool.Degenerated_s(edge.wrapped) else _placement(
        BRepAdaptor_Curve(edge.wrapped))
    return _edge_geometry(edge.wrapped), placement


def _contact_plane_groups(shape, quality):
    """Select only connected planes beside mapped analytic vertex contacts."""
    from .step_export import _geometry_signature

    faults = quality.get('self_intersection_faults', [])
    if (not faults or quality.get('unmapped_fault_count', 0)
            or quality.get('other_fault_count', 0)):
        return []
    faces, vertices = shape.Faces(), shape.Vertices()
    groups = []
    analytic = {'PLANE', 'CYLINDER', 'CONE', 'SPHERE', 'TORUS'}
    for fault in faults:
        kinds = fault.get('shape_types')
        vi, fi = fault.get('vertex_indices', []), fault.get('face_indices', [])
        if (kinds not in (['VERTEX'], ['EDGE', 'VERTEX'])
                or len(vi) != (2 if kinds == ['VERTEX'] else 1)
                or not fi or len(fi) > 8
                or any(not isinstance(i, int) or i < 0 or i >= len(vertices) for i in vi)
                or any(not isinstance(i, int) or i < 0 or i >= len(faces) for i in fi)
                or any(faces[i].geomType() not in analytic for i in fi)):
            return []
        planes = [i for i in fi if faces[i].geomType() == 'PLANE']
        adjacency = {i: set() for i in planes}
        signatures = {i: _geometry_signature(faces[i], 1e-10) for i in planes}
        for position, first in enumerate(planes):
            for second in planes[position + 1:]:
                if not _same_signature(signatures[first], signatures[second]):
                    continue
                shared = [edge for edge in faces[first].Edges()
                          if any(edge.isSame(other) for other in faces[second].Edges())]
                if any(any(vertex.isSame(vertices[i]) for i in vi)
                       for edge in shared for vertex in edge.Vertices()):
                    adjacency[first].add(second)
                    adjacency[second].add(first)
        remaining = set(planes)
        found = False
        while remaining:
            component, pending = set(), [min(remaining)]
            while pending:
                index = pending.pop()
                if index not in component:
                    component.add(index)
                    pending.extend(adjacency[index] - component)
            remaining -= component
            if len(component) >= 2:
                groups.append(component)
                found = True
        # Do not repair one contact while silently discarding another type.
        if not found:
            return []
    # A face may occur in several fault reports. Normalize its connected
    # component once, retaining the original fault-to-face mapping.
    merged = []
    for group in groups:
        overlapping = [item for item in merged if item & group]
        for item in overlapping:
            group |= item
            merged.remove(item)
        merged.append(group)
    return sorted((sorted(group) for group in merged), key=lambda group: group[0]) \
        if len(set().union(*merged)) <= 8 else []


def normalize_coplanar_contact_partitions(shape, quality):
    """Return a strictly proved local normalization, or the untouched input.

    Only mapped VERTEX or EDGE/VERTEX contacts with wholly analytic incident
    faces are eligible. At most eight original planar faces are considered.
    Native faults involving a fitted spline or a FACE remain ineligible.
    """
    import cadquery as cq
    from OCP.BRep import BRep_Builder, BRep_Tool
    from OCP.BRepBuilderAPI import BRepBuilderAPI_Copy
    from OCP.BRepTools import BRepTools_ReShape
    from OCP.Precision import Precision
    from OCP.ShapeUpgrade import ShapeUpgrade_UnifySameDomain
    from OCP.TopoDS import TopoDS_Shell
    from .cad_features import _material_bounds, _material_bounds_delta
    from .occ_shapes import _assess_brep_self_intersections
    from .step_export import _closed

    proof = {'method': 'Bounded fault-mapped coplanar partition cancellation',
             'applied': False, 'physical_geometry_changed': False,
             'closed_body_and_final_step_checks_required': True}
    if quality.get('passed', False):
        return shape, quality, proof
    groups = _contact_plane_groups(shape, quality)
    if not groups:
        proof['reason'] = 'No bounded coplanar component of eligible analytic contacts'
        return shape, quality, proof
    if not shape.isValid() or not _closed(shape):
        raise RuntimeError('Contact partition normalization requires valid closed solid CAD')
    original_digest = _brep_digest(shape)
    # Geometry is private too: even an unexpected kernel-side modification
    # cannot alter a shared fitted or source CAD surface owned by the caller.
    copier = BRepBuilderAPI_Copy(shape.wrapped, True, False)
    private = cq.Shape.cast(copier.Shape())
    original_faces = shape.Faces()
    private_faces = [cq.Face(copier.ModifiedShape(face.wrapped).Oriented(face.wrapped.Orientation()))
                     for face in original_faces]
    face_snapshots = [(face, _face_geometry(face), BRep_Tool.Tolerance_s(face.wrapped))
                      for face in private_faces]
    edge_snapshots = [(edge, _located_edge_geometry(edge),
                       BRep_Tool.Tolerance_s(edge.wrapped)) for edge in private.Edges()]
    vertex_snapshots = [(vertex, vertex.toTuple(), BRep_Tool.Tolerance_s(vertex.wrapped))
                        for vertex in private.Vertices()]
    selected = set(index for group in groups for index in group)
    replacement = BRepTools_ReShape()
    components = []
    for group in groups:
        parts = [private_faces[index] for index in group]
        shell = TopoDS_Shell()
        builder = BRep_Builder()
        builder.MakeShell(shell)
        for part in parts:
            builder.Add(shell, part.wrapped)
        component = cq.Shape.cast(shell)
        signatures, edges = [], defaultdict(list)
        before, areas_before = _grouped_boundary(component, signatures, edges, existing_only=False)
        if len(signatures) != 1:
            raise RuntimeError('A contact component has different supporting planes')
        unifier = ShapeUpgrade_UnifySameDomain(shell, False, True, False)
        unifier.SetSafeInputMode(True)
        unifier.SetLinearTolerance(Precision.Confusion_s())
        unifier.SetAngularTolerance(1e-10)
        unifier.AllowInternalEdges(False)
        unifier.Build()
        unified = cq.Shape.cast(unifier.Shape())
        if not unified.isValid() or len(unified.Faces()) != 1:
            raise RuntimeError('A bounded planar contact component did not unify to one valid face: '
                               f'{group}, faces={len(unified.Faces())}, valid={unified.isValid()}')
        after, areas_after = _grouped_boundary(unified, signatures, edges, existing_only=True)
        if before != after:
            raise RuntimeError('Contact partition normalization changed an oriented outer boundary')
        # Identical oriented physical edges on the identical plane prove the
        # same planar region. Default pcurve area quadrature depends on the
        # redundant internal partition; keep it diagnostic, not a substitute
        # for that exact boundary proof or the strict whole-material gates.
        if set(areas_before) != set(areas_after) or any(
                not np.isfinite(area) or area <= 0 for area in areas_after.values()):
            raise RuntimeError('Contact partition normalization lost its planar region')
        merged_face = unified.Faces()[0]
        if BRep_Tool.Tolerance_s(merged_face.wrapped) > max(
                BRep_Tool.Tolerance_s(part.wrapped) for part in parts):
            raise RuntimeError('Contact partition normalization enlarged a face tolerance')
        replacement.Replace(parts[0].wrapped, merged_face.wrapped)
        for part in parts[1:]:
            replacement.Remove(part.wrapped)
        components.append({'original_face_indices': group,
                           'oriented_outer_edges_unchanged': True,
                           'supporting_plane_unchanged': True,
                           'original_area': sum(areas_before.values()),
                           'normalized_area': sum(areas_after.values())})
    result = cq.Shape.cast(replacement.Apply(private.wrapped))
    if _brep_digest(shape) != original_digest:
        raise RuntimeError('Contact partition normalization mutated its input CAD')
    for face, geometry, tolerance in face_snapshots:
        if (_face_geometry(face) != geometry
                or BRep_Tool.Tolerance_s(face.wrapped) != tolerance):
            raise RuntimeError('Contact partition normalization changed an original supporting surface')
    for edge, geometry, tolerance in edge_snapshots:
        if (_located_edge_geometry(edge) != geometry
                or BRep_Tool.Tolerance_s(edge.wrapped) != tolerance):
            raise RuntimeError('Contact partition normalization changed an existing physical edge')
    for vertex, point, tolerance in vertex_snapshots:
        if (vertex.toTuple() != point or BRep_Tool.Tolerance_s(vertex.wrapped) != tolerance):
            raise RuntimeError('Contact partition normalization changed an existing vertex')
    result_faces = result.Faces()
    expected_count = len(original_faces) - len(selected) + len(groups)
    if (not result.isValid() or not _closed(result)
            or len(result.Solids()) != len(shape.Solids())
            or len(result_faces) != expected_count):
        raise RuntimeError('Contact partition normalization did not preserve closed material')
    for index, face in enumerate(private_faces):
        if index not in selected and not any(
                candidate.wrapped.IsEqual(face.wrapped) for candidate in result_faces):
            raise RuntimeError('Contact partition normalization changed an unrelated face')
    volumes = [abs(float(body.Volume(tol=1e-7))) for body in (shape, result)]
    volume_tolerance = max(1e-9, volumes[0] * 1e-6)
    if (not np.isfinite(volumes).all() or min(volumes) <= 0
            or abs(volumes[1] - volumes[0]) > volume_tolerance):
        raise RuntimeError('Contact partition normalization changed the enclosed material')
    bounds = _material_bounds(shape)
    linear_tolerance = max(2.0 * Precision.Confusion_s(),
                          float(np.linalg.norm(bounds[1::2] - bounds[::2])) * 1e-9)
    if np.max(np.abs(_material_bounds_delta(shape, result, linear_tolerance))) > linear_tolerance:
        raise RuntimeError('Contact partition normalization changed material bounds')
    updated = _assess_brep_self_intersections(result)
    proof.update(applied=True, original_faces=len(original_faces), normalized_faces=len(result_faces),
                 selected_face_limit=8, components=components, inputs_unchanged=True,
                 original_supporting_surfaces_and_placements_unchanged=True,
                 physical_edges_and_vertices_unchanged=True, unrelated_faces_unchanged=True,
                 volumes=volumes, volume_tolerance=volume_tolerance,
                 linear_tolerance=linear_tolerance, native_self_intersections=updated)
    return result, updated, proof
