# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE.
"""Retry a local containment Boolean that incorrectly reuses shared topology."""
from __future__ import annotations

from collections import defaultdict


def _has_shared_edge(first, second):
    buckets = defaultdict(list)
    for edge in second.Edges():
        buckets[hash(edge.wrapped)].append(edge.wrapped)
    return any(
        any(edge.wrapped.IsSame(candidate)
            for candidate in buckets.get(hash(edge.wrapped), ()))
        for edge in first.Edges()
    )


def _empty_partition(shape):
    return (shape.isValid()
            and not any(getattr(shape, kind)()
                        for kind in ('Solids', 'Shells', 'Faces', 'Wires', 'Edges', 'Vertices')))


def _copy_material_topology(material):
    """Copy topology, retaining the very same native curves and surfaces."""
    import cadquery as cq
    from OCP.BRep import BRep_Tool
    from OCP.BRepBuilderAPI import BRepBuilderAPI_Copy
    from OCP.TopLoc import TopLoc_Location
    from OCP.TopoDS import TopoDS

    copier = BRepBuilderAPI_Copy(material.wrapped, False, False)
    copied = cq.Shape.cast(copier.Shape())
    for kind in ('Solids', 'Shells', 'Faces', 'Wires', 'Edges', 'Vertices'):
        parts = getattr(material, kind)()
        copied_parts = getattr(copied, kind)()
        if len(parts) != len(copied_parts):
            raise RuntimeError('Containment topology copy changed the material partition')
        for part in parts:
            replacement = copier.ModifiedShape(part.wrapped)
            # ModifiedShape returns the canonical forward-oriented image.
            # Check orientation on its occurrence in the copied solid.
            occurrences = [candidate.wrapped for candidate in copied_parts
                           if candidate.wrapped.IsSame(replacement)]
            if (part.wrapped.IsSame(replacement)
                    or len(occurrences) != 1
                    or part.wrapped.Orientation() != occurrences[0].Orientation()
                    or not part.wrapped.Location().IsEqual(replacement.Location())):
                raise RuntimeError('Containment topology copy did not retain shape placement')
    for face in material.Faces():
        original = face.wrapped
        replacement = TopoDS.Face_s(copier.ModifiedShape(original))
        location, copied_location = TopLoc_Location(), TopLoc_Location()
        surface = BRep_Tool.Surface_s(original, location)
        copied_surface = BRep_Tool.Surface_s(replacement, copied_location)
        if (surface is not copied_surface or not location.IsEqual(copied_location)
                or BRep_Tool.Tolerance_s(original) != BRep_Tool.Tolerance_s(replacement)):
            raise RuntimeError('Containment topology copy replaced a supporting surface')
    for edge in material.Edges():
        original = edge.wrapped
        replacement = TopoDS.Edge_s(copier.ModifiedShape(original))
        location, copied_location = TopLoc_Location(), TopLoc_Location()
        curve = BRep_Tool.Curve_s(original, location, 0.0, 0.0)
        copied_curve = BRep_Tool.Curve_s(replacement, copied_location, 0.0, 0.0)
        if (curve is not copied_curve or not location.IsEqual(copied_location)
                or BRep_Tool.Range_s(original) != BRep_Tool.Range_s(replacement)
                or BRep_Tool.Degenerated_s(original) != BRep_Tool.Degenerated_s(replacement)
                or BRep_Tool.Tolerance_s(original) != BRep_Tool.Tolerance_s(replacement)):
            raise RuntimeError('Containment topology copy replaced a boundary curve')
    for vertex in material.Vertices():
        original = vertex.wrapped
        replacement = TopoDS.Vertex_s(copier.ModifiedShape(original))
        if (BRep_Tool.Pnt_s(original).Coord() != BRep_Tool.Pnt_s(replacement).Coord()
                or BRep_Tool.Tolerance_s(original) != BRep_Tool.Tolerance_s(replacement)):
            raise RuntimeError('Containment topology copy moved a boundary vertex')
    return copied


def retry_shared_material_partition(material, domain, outside, inside, *, boolean, fuzzy):
    """Return a valid witness after isolating a small shared material operand.

    No geometry is fitted, healed, or transferred. The original empty Cut must
    be paired with an invalid nonempty Common and actual shared native edges.
    The copied operand must produce another empty Cut and valid closed Common
    at exactly the same tolerance. The caller still checks the original
    material's full volume and bounds.
    """
    from .cad_partition_proof import _brep_digest
    from .step_export import _closed

    if (not 1 <= len(material.Faces()) <= 64
            or not material.isValid() or not _closed(material)
            or not _empty_partition(outside)
            or not inside.Faces()
            or (inside.isValid() and _closed(inside))
            or not _has_shared_edge(material, domain)):
        raise RuntimeError('Invalid inside partition is not eligible for shared-topology retry')
    before = tuple(_brep_digest(shape) for shape in (material, domain))
    try:
        copied = _copy_material_topology(material)
        if not copied.isValid() or not _closed(copied) or _has_shared_edge(copied, domain):
            raise RuntimeError('Containment topology copy did not isolate valid material')
        retried_outside = boolean(copied, domain, 'cut', fuzzy)
        if not _empty_partition(retried_outside):
            raise RuntimeError('Isolated containment retry did not retain an empty outside partition')
        retried_inside = boolean(copied, domain, 'common', fuzzy)
        if not retried_inside.isValid() or not _closed(retried_inside):
            raise RuntimeError('Isolated containment retry did not produce a valid closed inside partition')
    finally:
        if tuple(_brep_digest(shape) for shape in (material, domain)) != before:
            raise RuntimeError('Shared-topology containment retry changed an original operand')
    return retried_inside
