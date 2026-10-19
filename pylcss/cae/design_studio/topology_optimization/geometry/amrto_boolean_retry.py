# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE.
"""Bounded exact spline subdivision for failed analytic stock-pad trims."""
from __future__ import annotations

def boolean_with_history(body, tools, operation):
    """Keep OCCT provenance for a non-destructive, zero-fuzzy Boolean."""
    import cadquery as cq
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Common, BRepAlgoAPI_Fuse
    from OCP.TopTools import TopTools_ListOfShape

    arguments, native_tools = TopTools_ListOfShape(), TopTools_ListOfShape()
    arguments.Append(body.wrapped)
    for tool in tools:
        native_tools.Append(tool.wrapped)
    builder = {'fuse': BRepAlgoAPI_Fuse, 'common': BRepAlgoAPI_Common}[operation]()
    builder.SetArguments(arguments)
    builder.SetTools(native_tools)
    builder.SetNonDestructive(True)
    builder.SetRunParallel(True)
    builder.Build()
    if not builder.IsDone():
        raise RuntimeError(f'Stock-boundary pad {operation} failed')
    return cq.Shape.cast(builder.Shape()), builder


def incident_original_spline_faces(body, failed, builders):
    """Trace spline edge-neighbors of invalid result faces through history.

    Unmapped, ambiguous or extensive faults are deliberately unsupported.
    Proximity and face indices are never used as a substitute for provenance.
    """
    import cadquery as cq

    faces = failed.Faces()
    invalid = [face for face in faces if not face.isValid()]
    if not invalid or len(invalid) > 8:
        raise RuntimeError('Pad trim retry requires a bounded set of invalid faces')
    bad_edges = [edge for face in invalid for edge in face.Edges()]
    adjacent = [face for face in faces if face.geomType() == 'BSPLINE'
                and any(edge.isSame(bad) for edge in face.Edges() for bad in bad_edges)]
    if not adjacent:
        raise RuntimeError('Invalid pad trim has no incident original spline')
    origins = [[] for _ in adjacent]
    for index, face in enumerate(body.Faces()):
        if face.geomType() != 'BSPLINE':
            continue
        descendants = [face]
        for builder in builders:
            next_faces = []
            for descendant in descendants:
                # Unmodified faces retain their native identity. Modified and
                # generated faces carry the explicit Boolean history.
                next_faces.append(descendant)
                for raw in (*builder.Modified(descendant.wrapped),
                            *builder.Generated(descendant.wrapped)):
                    next_faces.extend(cq.Shape.cast(raw).Faces())
            descendants = next_faces
        for position, result_face in enumerate(adjacent):
            if any(result_face.isSame(child) for child in descendants):
                origins[position].append(index)
    if any(len(indices) != 1 for indices in origins):
        raise RuntimeError('Invalid pad trim has ambiguous or missing spline provenance')
    selected = sorted({indices[0] for indices in origins})
    if len(selected) > 8:
        raise RuntimeError('Pad trim retry would subdivide too many original faces')
    return selected


def exact_bezier_retry_body(body, selected):
    """Replace only implicated rectangles with their exact polynomial spans.

    No fitted geometry, control points or knot locations are optimized. The
    final sewn body must pass a complete polynomial/UV partition proof and
    an independent span-integrated material-volume check.
    """
    import cadquery as cq
    from OCP.BRep import BRep_Tool
    from OCP.BRepBuilderAPI import BRepBuilderAPI_Copy
    from OCP.GeomAbs import GeomAbs_CN
    from OCP.ShapeFix import ShapeFix_Shape
    from OCP.ShapeUpgrade import ShapeUpgrade_ShapeDivideContinuity
    from .amrto_intrinsic import _validate_patch_partition
    from .amrto_solid import stitch_paper_surfaces
    from .amrto_volume import assert_rectangular_spline_partition, assert_unchanged_cad_volume
    from .step_export import _closed

    faces = body.Faces()
    if (not body.isValid() or not _closed(body) or not body.Solids()
            or not selected or len(selected) > 8
            or any(index < 0 or index >= len(faces) for index in selected)):
        raise RuntimeError('Exact pad retry requires a valid closed original spline body')
    replacements, patch_reports, additional_faces = [], [], 0
    selected = set(selected)
    for index, face in enumerate(faces):
        if index not in selected:
            replacements.append(face)
            continue
        divider = ShapeUpgrade_ShapeDivideContinuity(
            BRepBuilderAPI_Copy(face.wrapped, False, False).Shape())
        divider.SetSurfaceCriterion(GeomAbs_CN)
        divider.SetBoundaryCriterion(GeomAbs_CN)
        divider.SetPCurveCriterion(GeomAbs_CN)
        divider.SetPrecision(1e-7)
        if not divider.Perform():
            raise RuntimeError('Exact Bezier subdivision did not complete')
        fixer = ShapeFix_Shape(divider.Result())
        fixer.SetPrecision(1e-7)
        fixer.SetMaxTolerance(1e-5)
        fixer.Perform()
        divided = cq.Shape.cast(fixer.Shape())
        children = divided.Faces()
        additional_faces += len(children) - 1
        if not 1 < len(children) <= 512 or additional_faces > 2048:
            raise RuntimeError('Exact pad retry exceeded its subdivision bound')
        proof = _validate_patch_partition(face, divided)
        patch_reports.append({'source_face_index': index, 'child_count': len(children), **proof})
        replacements.extend(children)
    bounds = body.BoundingBox()
    scale = max(bounds.xlen, bounds.ylen, bounds.zlen)
    # This is a single fixed sewing tolerance, not an escalating repair loop.
    candidate, sewing = stitch_paper_surfaces(cq.Compound.makeCompound(replacements),
        tolerance=1e-4, source_scale=scale, maximum_relative_tolerance=1e-4 / scale,
        _allow_partial_seam_subdivision=False)
    if (not sewing['is_solid'] or not candidate.isValid() or not _closed(candidate)
            or len(candidate.Solids()) != len(body.Solids())):
        raise RuntimeError('Exact pad retry did not retain connected closed material')
    parents = [(BRep_Tool.Surface_s(face.wrapped), face._uvBounds()) for face in faces]
    indices = []
    for face in candidate.Faces():
        a, b, c, d = face._uvBounds()
        u, v = (a + b) * .5, (c + d) * .5
        point = BRep_Tool.Surface_s(face.wrapped).Value(u, v)
        matches = [(point.Distance(surface.Value(u, v)), index)
                   for index, (surface, (u0, u1, v0, v1)) in enumerate(parents)
                   if u0 <= a < b <= u1 and v0 <= c < d <= v1]
        if not matches:
            raise RuntimeError('Exact pad retry lost original parameter correspondence')
        indices.append(min(matches)[1])
    partition = assert_rectangular_spline_partition(body, candidate, indices)
    volume = assert_unchanged_cad_volume(body, candidate, relative_tolerance=1e-6)
    return candidate, {'method': 'One exact Bezier-span retry of incident original spline faces',
        'patches_refitted': 0, 'holes_capped': 0, 'patches': patch_reports,
        'sewing': sewing, 'partition': partition, 'volume': volume}
