# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Bounded stitching of independently fitted AMRTO surfaces.

Section 2.7's fitted surfaces need stitching before volumetric consumers can
use them. Sewing changes edge/vertex topology and tolerances, not the fitted
surface geometry. This module never fits patches, subdivides charts, caps
holes, or turns an open shell into an alleged solid.
"""

from __future__ import annotations

import math
from typing import Any

import cadquery as cq
from OCP.BRep import BRep_Tool
from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeSolid, BRepBuilderAPI_Sewing
from OCP.BRepBndLib import BRepBndLib
from OCP.Bnd import Bnd_Box
from OCP.ShapeFix import ShapeFix_Shell, ShapeFix_Solid
from OCP.TopLoc import TopLoc_Location

from .amrto_nurbs_fitting import _assemble_nested_solids
from .occ_shapes import _solid_volume


def _surface_location(face: Any) -> tuple[Any, Any]:
    location = TopLoc_Location()
    surface = BRep_Tool.Surface_s(face.wrapped, location)
    return surface, location


def _same_surfaces(before: list[Any], shape: Any) -> bool:
    """Match every face to the same underlying surface, including multiplicity."""
    remaining = list(before)
    for face in shape.Faces():
        surface, location = _surface_location(face)
        index = next((i for i, (original, original_location) in enumerate(remaining)
                      if original == surface and original_location.IsEqual(location)), None)
        if index is None:
            return False
        remaining.pop(index)
    return not remaining


def split_closed_paper_faces(
    shape: Any, *, reference_volume: float | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Open periodic face seams by exact parameter subdivision, without a fit.

    OpenCASCADE's Boolean analyzer can reject an otherwise regular fitted
    face that spans a whole closed surface direction. Two trimmed spans of
    the very same surface remove this seam ambiguity. The recorded source
    face history must cover every output face once; every changed source UV
    rectangle must be partitioned once with unchanged evaluated geometry.
    This operation never performs global C1 subdivision or patch refitting.
    """
    from OCP.ShapeUpgrade import ShapeUpgrade_ShapeDivideClosed
    from .amrto_intrinsic import _validate_patch_partition

    if hasattr(shape, 'val'):
        shape = shape.val()
    if not hasattr(shape, 'wrapped'):
        shape = cq.Shape.cast(shape)
    source_faces = shape.Faces()
    report = {'method': 'Exact periodic face parameter subdivision',
              'input_face_count': len(source_faces), 'output_face_count': len(source_faces),
              'split_source_face_indices': [], 'patches_refitted': 0,
              'source_uv_coverage_exactly_once': True,
              'maximum_matching_uv_geometry_error': 0.0, 'parent_face_indices': []}
    divider = ShapeUpgrade_ShapeDivideClosed(shape.wrapped)
    divider.SetNbSplitPoints(1)
    divider.Perform()
    native = divider.Result()
    if native.IsNull():
        raise RuntimeError('Periodic face subdivision returned no CAD geometry')
    divided = cq.Shape.cast(native)
    output_faces = divided.Faces()
    context = divider.GetContext()
    matched = set()
    parents = [None] * len(output_faces)
    for parent_index, original in enumerate(source_faces):
        children = cq.Shape.cast(context.Apply(original.wrapped))
        indices = []
        for child in children.Faces():
            index = next((i for i, output in enumerate(output_faces)
                          if output.wrapped.IsSame(child.wrapped)), None)
            if index is None or index in matched:
                raise RuntimeError('Periodic face subdivision lost source face correspondence')
            matched.add(index)
            indices.append(index)
            parents[index] = parent_index
        if not indices:
            raise RuntimeError('Periodic face subdivision removed a source face')
        if len(indices) > 1:
            if original.geomType() == 'CYLINDER':
                from .amrto_trim_partition import assert_trimmed_cylindrical_partition
                proof = assert_trimmed_cylindrical_partition(original, children)
            else:
                proof = _validate_patch_partition(original, children)
            report['maximum_matching_uv_geometry_error'] = max(
                report['maximum_matching_uv_geometry_error'],
                proof['maximum_matching_uv_geometry_error'])
            report['split_source_face_indices'].append(parent_index)
    if len(matched) != len(output_faces):
        raise RuntimeError('Periodic face subdivision created an untracked CAD face')
    if not divided.isValid():
        raise RuntimeError('Periodic face subdivision did not preserve native CAD validity')
    if shape.Solids():
        if (len(divided.Solids()) != len(shape.Solids())
                or not all(BRep_Tool.IsClosed_s(shell.wrapped) for shell in divided.Shells())):
            raise RuntimeError('Periodic face subdivision did not preserve closed material solids')
        from .amrto_volume import assert_unchanged_cad_volume
        volume_proof = assert_unchanged_cad_volume(shape, divided)
        report.update(cad_volume=volume_proof['volume_after'],
                      volume_relative_difference=volume_proof['relative_volume_difference'],
                      volume_invariant=volume_proof)
    report.update(output_face_count=len(output_faces), parent_face_indices=parents)
    return divided, report


def stitch_paper_surfaces(
    shape: Any,
    *,
    tolerance: float = 1e-3,
    source_scale: float | None = None,
    maximum_relative_tolerance: float = 0.003,
    source_volume: float | None = None,
    maximum_volume_relative_error: float = 0.05,
) -> tuple[Any, dict[str, Any]]:
    """Stitch once at each of at most nine bounded tolerances; keep every fit.

    A successful result has no free/multiple edges, contains every original
    face on its unchanged surface, and consists entirely of closed, valid
    solids. Nested closed shells become cavities. Global self-intersection,
    exact-feature restoration, and STEP verification remain downstream
    checks: native topological validity alone is not their substitute.

    Failure returns the complete best sewn surface model and a report with
    ``is_solid=False``. This keeps a useful CAD result without falsely making
    it available to solid-only analysis or lattice workflows.
    """
    if hasattr(shape, 'val'):
        shape = shape.val()
    if not hasattr(shape, 'wrapped'):
        shape = cq.Shape.cast(shape)
    faces = shape.Faces()
    if not faces:
        raise ValueError('There are no fitted CAD faces to stitch')
    if source_scale is None:
        box = Bnd_Box()
        BRepBndLib.Add_s(shape.wrapped, box, False)
        bounds = box.Get()
        source_scale = max(bounds[i + 3] - bounds[i] for i in range(3))
    for value, label in ((source_scale, 'source scale'), (tolerance, 'stitching tolerance'),
                         (maximum_relative_tolerance, 'maximum relative tolerance')):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f'The {label} must be finite and positive')
    if not math.isfinite(maximum_volume_relative_error) or maximum_volume_relative_error < 0:
        raise ValueError('The maximum volume error must be finite and nonnegative')
    if source_volume is not None and (not math.isfinite(source_volume) or source_volume <= 0):
        raise ValueError('The reference source volume must be finite and positive')

    maximum = float(source_scale) * float(maximum_relative_tolerance)
    initial = min(float(tolerance), maximum)
    # Fixed finite stitching schedule. Higher tolerance is not necessarily
    # better: it can collapse a short edge, so stop at the first valid closure.
    schedule = sorted({initial, maximum, *(
        float(source_scale) * relative for relative in
        (1e-5, 1e-4, 5e-4, 1e-3, 1.5e-3, 2e-3, 2.5e-3)
        if initial < float(source_scale) * relative < maximum)})
    source_surfaces = [_surface_location(face) for face in faces]
    report = {
        'method': 'Bounded OpenCASCADE stitching of AMRTO independent surfaces',
        'input_face_count': len(faces), 'source_scale': float(source_scale),
        'maximum_tolerance': maximum, 'maximum_relative_tolerance': maximum_relative_tolerance,
        'fitting_attempts': 0, 'patches_refitted': 0, 'holes_capped': 0,
        'is_solid': False, 'is_valid': bool(shape.isValid()), 'free_edge_count': None,
        'surface_geometry_unchanged': True, 'trials': [],
        'global_intersection_validation': 'not_run', 'cad_volume': None,
    }
    best_shape, best_free = shape, math.inf
    for current_tolerance in schedule:
        sewing = BRepBuilderAPI_Sewing(current_tolerance)
        sewing.SetMaxTolerance(current_tolerance)
        for face in faces:
            sewing.Add(face.wrapped)
        sewing.Perform()
        sewn_native = sewing.SewedShape()
        trial = {'tolerance': current_tolerance, 'relative_tolerance': current_tolerance / source_scale}
        report['trials'].append(trial)
        if sewn_native.IsNull():
            trial['failure'] = 'null_sewn_shape'
            continue
        sewn = cq.Shape.cast(sewn_native)
        free = int(sewing.NbFreeEdges())
        unchanged = len(sewn.Faces()) == len(faces) and _same_surfaces(source_surfaces, sewn)
        trial.update(free_edge_count=free, multiple_edge_count=int(sewing.NbMultipleEdges()),
                     degenerated_shape_count=int(sewing.NbDegeneratedShapes()),
                     face_count=len(sewn.Faces()), surface_geometry_unchanged=unchanged)
        if unchanged and free < best_free:
            best_shape, best_free = sewn, free
            report.update(free_edge_count=free, is_valid=bool(sewn.isValid()),
                          sew_tolerance=current_tolerance)
        if free or sewing.NbMultipleEdges() or sewing.NbDegeneratedShapes() or not unchanged:
            continue
        shells = sewn.Shells()
        if not shells or sum(len(shell.Faces()) for shell in shells) != len(faces):
            trial['failure'] = 'faces_outside_shells'
            continue
        solids = []
        try:
            for shell in shells:
                fixer = ShapeFix_Shell(shell.wrapped)
                fixer.SetPrecision(current_tolerance)
                fixer.SetMaxTolerance(current_tolerance)
                fixer.Perform()
                fixed_shell = fixer.Shell()
                if not BRep_Tool.IsClosed_s(fixed_shell):
                    raise ValueError('open_shell')
                maker = BRepBuilderAPI_MakeSolid(fixed_shell)
                if not maker.IsDone():
                    raise ValueError('solid_builder_failed')
                solid_fixer = ShapeFix_Solid(maker.Solid())
                solid_fixer.SetPrecision(current_tolerance)
                solid_fixer.SetMaxTolerance(current_tolerance)
                solid_fixer.Perform()
                solid = cq.Solid(solid_fixer.Solid())
                if not solid.isValid() or not all(BRep_Tool.IsClosed_s(s.wrapped) for s in solid.Shells()):
                    raise ValueError('invalid_or_open_solid')
                solids.append(solid)
            solids, cavity_count = _assemble_nested_solids(solids, current_tolerance)
            candidate = solids[0] if len(solids) == 1 else cq.Compound.makeCompound(solids)
            if len(candidate.Faces()) != len(faces) or not _same_surfaces(source_surfaces, candidate):
                raise ValueError('surface_geometry_or_coverage_changed')
            volume = _solid_volume(candidate, relative_tolerance=1e-4)
            relative_error = abs(volume - source_volume) / source_volume if source_volume is not None else None
            trial.update(cad_volume=volume, source_volume_relative_error=relative_error)
            if volume <= 0 or (relative_error is not None and relative_error > maximum_volume_relative_error):
                raise ValueError('enclosed_volume_outside_source_limit')
        except Exception as error:
            trial['failure'] = str(error)
            continue
        report.update(is_solid=True, is_valid=True, free_edge_count=0, cad_volume=volume,
                      solid_count=len(solids), shell_count=len(shells), cavity_shell_count=cavity_count,
                      sew_tolerance=current_tolerance, source_volume_relative_error=relative_error,
                      status='closed_solid')
        return candidate, report
    report.update(status='open_or_incomplete_brep', solid_count=0,
                  shell_count=len(best_shape.Shells()), cavity_shell_count=0)
    return best_shape, report


__all__ = ['stitch_paper_surfaces', 'split_closed_paper_faces']
