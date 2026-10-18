# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE.
"""Cache strict intrinsic patch checks before expensive closed-body checks."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import time

import numpy as np

from .occ_shapes import _assess_brep_self_intersections


def _surface_fingerprint(surface, tolerance):
    digest = hashlib.sha256(b"AMRTO intrinsic C1 check v1")
    digest.update(np.asarray((surface.degree_u, surface.degree_v,
                              surface.ctrlpts_size_u, surface.ctrlpts_size_v,
                              bool(getattr(surface, "segment_boundary_edges", False)),
                              bool(getattr(surface, "rational", False))), dtype=np.int64).tobytes())
    for values in (surface.ctrlpts, getattr(surface, "weights", ()),
                   surface.knotvector_u, surface.knotvector_v,
                   getattr(surface, "boundary_split_parameters_u", ()),
                   getattr(surface, "boundary_split_parameters_v", ()), (tolerance,)):
        array = np.asarray(values, dtype=np.float64)
        digest.update(np.asarray((array.ndim, *array.shape), dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _validate_patch_partition(original, divided):
    """Check exact surface correspondence and a complete rectangular UV tiling."""
    from OCP.BRep import BRep_Tool

    faces = divided.Faces()
    if not divided.isValid() or not faces:
        raise ValueError("Local C1 subdivision is not a valid patch")
    parent = BRep_Tool.Surface_s(original.wrapped)
    u0, u1, v0, v1 = original._uvBounds()
    bounds = original.BoundingBox()
    tolerance = max(1e-8, float(bounds.DiagonalLength) * 1e-10)
    rectangles, maximum_error = [], 0.0
    for face in faces:
        first_u, last_u, first_v, last_v = face._uvBounds()
        if (not np.isfinite((first_u, last_u, first_v, last_v)).all()
                or not u0 - 1e-12 <= first_u < last_u <= u1 + 1e-12
                or not v0 - 1e-12 <= first_v < last_v <= v1 + 1e-12):
            raise ValueError("Local C1 subdivision left its source UV domain")
        underlying = BRep_Tool.Surface_s(face.wrapped)
        for a in (.125, .5, .875):
            for b in (.125, .5, .875):
                u, v = first_u + a * (last_u - first_u), first_v + b * (last_v - first_v)
                maximum_error = max(maximum_error, parent.Value(u, v).Distance(underlying.Value(u, v)))
        rectangles.append((max(u0, first_u), min(u1, last_u), max(v0, first_v), min(v1, last_v)))
    if maximum_error > tolerance:
        raise ValueError("Local C1 subdivision changed its exact source surface")
    # Sweep UV rectangles using one-dimensional coverage differences. This
    # verifies every interval once without allocating a dense 2D sample grid.
    levels = np.unique([v0, v1, *(value for rectangle in rectangles for value in rectangle[2:])])
    events = {u0: [], u1: []}
    for first_u, last_u, first_v, last_v in rectangles:
        interval = (int(np.searchsorted(levels, first_v)), int(np.searchsorted(levels, last_v)))
        events.setdefault(first_u, []).append((*interval, 1))
        events.setdefault(last_u, []).append((*interval, -1))
    coverage = np.zeros(len(levels), dtype=np.int64)
    for u, changes in sorted(events.items()):
        for first, last, sign in changes:
            coverage[first] += sign
            coverage[last] -= sign
        if u < u1 and not np.all(np.cumsum(coverage)[:-1] == 1):
            raise ValueError("Local C1 subdivision did not cover its source UV domain exactly once")
    if np.any(coverage):
        raise ValueError("Local C1 subdivision left unmatched source UV intervals")
    return {"maximum_matching_uv_geometry_error": maximum_error,
            "geometry_tolerance": tolerance, "source_uv_coverage_exactly_once": True}


def preflight_layout_intrinsic_fold(layout, *, tolerance=1e-4):
    """Find a definite fold in an isolated exact C1 child without compound BOP.

    Call only for source-rejected fits before potentially expensive Boolean
    integration. A pure FACE self-intersection on a single child establishes
    an intrinsic fold; edge/vertex contacts, unknown faults and unsupported
    native checks defer to the required complete-body checker. A clean result
    is never an acceptance certificate for either the patch or the final CAD.
    """
    started = time.perf_counter()
    report = {'method': 'Individual exact C1 child native FACE self-intersection preflight',
              'layout_id': int(layout.layout_id), 'definitive_fold': False,
              'deferred_to_global': True, 'closed_body_and_final_step_checks_required': True,
              'compound_BOP_executed': False, 'child_reports': [], 'cache_hit': False}
    fingerprint = None
    try:
        fingerprint = _surface_fingerprint(layout.nurbs_surface, tolerance)
        if (getattr(layout, '_intrinsic_preflight_fingerprint', None) == fingerprint
                and isinstance(getattr(layout, '_intrinsic_preflight_report', None), dict)):
            cached = deepcopy(layout._intrinsic_preflight_report)
            cached.update(layout_id=int(layout.layout_id), cache_hit=True)
            return cached
        import cadquery as cq
        from OCP.BRepAdaptor import BRepAdaptor_Surface
        from OCP.BRepBuilderAPI import BRepBuilderAPI_Copy
        from OCP.GeomAbs import GeomAbs_C0, GeomAbs_C1
        from OCP.ShapeFix import ShapeFix_Shape
        from OCP.ShapeUpgrade import ShapeUpgrade_ShapeDivideContinuity
        from .amrto_nurbs_fitting import nurbs_surface_to_occ_face

        face = cq.Shape.cast(nurbs_surface_to_occ_face(layout.nurbs_surface, tol=tolerance))
        if not face.isValid():
            raise ValueError('Original isolated face requires complete-body topology normalization')
        candidate = face
        adaptor = BRepAdaptor_Surface(face.wrapped, True)
        if adaptor.UContinuity() == GeomAbs_C0 or adaptor.VContinuity() == GeomAbs_C0:
            divider = ShapeUpgrade_ShapeDivideContinuity(BRepBuilderAPI_Copy(face.wrapped, False, False).Shape())
            divider.SetSurfaceCriterion(GeomAbs_C1)
            divider.SetBoundaryCriterion(GeomAbs_C1)
            divider.SetPCurveCriterion(GeomAbs_C1)
            divider.SetPrecision(1e-7)
            if not divider.Perform():
                raise ValueError('The isolated patch could not be divided into exact C1 spans')
            fixer = ShapeFix_Shape(divider.Result())
            fixer.SetPrecision(1e-7)
            fixer.SetMaxTolerance(1e-5)
            fixer.Perform()
            candidate = cq.Shape.cast(fixer.Shape())
        report['surface_geometry_validation'] = _validate_patch_partition(face, candidate)
        children = candidate.Faces()
        report['normalized_face_count'] = len(children)
        for index, child in enumerate(children):
            row = {'child_index': index, 'definitive_fold': False}
            try:
                adaptor = BRepAdaptor_Surface(child.wrapped, True)
                if adaptor.UContinuity() == GeomAbs_C0 or adaptor.VContinuity() == GeomAbs_C0:
                    raise ValueError('Isolated child retained an internal C0 span')
                quality = _assess_brep_self_intersections(child)
                row['native_quality'] = quality
                faults = quality.get('self_intersection_faults', [])
                definite = bool(not quality.get('passed', False) and faults
                    and quality.get('failed_face_indices') == [0]
                    and quality.get('unmapped_fault_count') == 0
                    and quality.get('other_fault_count') == 0
                    and all(fault.get('shape_types') == ['FACE'] and fault.get('face_indices') == [0]
                            for fault in faults))
                row['definitive_fold'] = definite
                report['child_reports'].append(row)
                if definite:
                    report.update(definitive_fold=True, deferred_to_global=False,
                                  failed_child_index=index,
                                  reason='A single exact C1 child has a mapped pure FACE self-intersection')
                    break
            except MemoryError:
                raise
            except Exception as error:
                row.update(deferred_to_global=True, reason=f'{type(error).__name__}: {error}')
                report['child_reports'].append(row)
        report['checked_child_count'] = len(report['child_reports'])
        if not report['definitive_fold']:
            report['reason'] = 'No definitive isolated-child fold; complete-body and STEP checks remain required'
    except MemoryError:
        raise
    except Exception as error:
        report['reason'] = f'{type(error).__name__}: {error}'
    report['elapsed_seconds'] = time.perf_counter() - started
    if fingerprint is not None:
        layout._intrinsic_preflight_fingerprint = fingerprint
        layout._intrinsic_preflight_report = deepcopy(report)
    return report


def assess_layout_intrinsic_intersections(layout, *, tolerance=1e-4):
    """Return explicit intrinsic faults; defer unsupported open checks globally.

    This is a preliminary rejection screen only. Every successful or deferred
    patch must still pass the mandatory closed-body and final STEP checks.
    """
    started = time.perf_counter()
    try:
        fingerprint = _surface_fingerprint(layout.nurbs_surface, tolerance)
    except (AttributeError, ValueError, TypeError) as error:
        return {"method": "Strict native BOP on exact C1 children of one fitted patch",
                "layout_id": int(layout.layout_id), "passed": None, "cache_hit": False,
                "deferred_to_global": True, "closed_body_and_final_step_checks_required": True,
                "reason": f"{type(error).__name__}: {error}",
                "elapsed_seconds": time.perf_counter() - started}
    if (getattr(layout, "_intrinsic_surface_fingerprint", None) == fingerprint
            and isinstance(getattr(layout, "_intrinsic_surface_report", None), dict)):
        report = deepcopy(layout._intrinsic_surface_report)
        report.update(layout_id=int(layout.layout_id), cache_hit=True)
        return report
    report = {"method": "Strict native BOP on exact C1 children of one fitted patch",
              "layout_id": int(layout.layout_id), "cache_hit": False,
              "closed_body_and_final_step_checks_required": True}
    try:
        import cadquery as cq
        from OCP.BRepAdaptor import BRepAdaptor_Surface
        from OCP.BRepBuilderAPI import BRepBuilderAPI_Copy
        from OCP.GeomAbs import GeomAbs_C0, GeomAbs_C1
        from OCP.ShapeFix import ShapeFix_Shape
        from OCP.ShapeUpgrade import ShapeUpgrade_ShapeDivideContinuity
        from .amrto_nurbs_fitting import nurbs_surface_to_occ_face

        face = cq.Shape.cast(nurbs_surface_to_occ_face(layout.nurbs_surface, tol=tolerance))
        if not face.isValid():
            raise ValueError("The isolated fitted patch requires closed-body topology healing")
        adaptor = BRepAdaptor_Surface(face.wrapped, True)
        candidate = face
        if adaptor.UContinuity() == GeomAbs_C0 or adaptor.VContinuity() == GeomAbs_C0:
            divider = ShapeUpgrade_ShapeDivideContinuity(BRepBuilderAPI_Copy(face.wrapped, False, False).Shape())
            divider.SetSurfaceCriterion(GeomAbs_C1)
            divider.SetBoundaryCriterion(GeomAbs_C1)
            divider.SetPCurveCriterion(GeomAbs_C1)
            divider.SetPrecision(1e-7)
            if not divider.Perform():
                raise ValueError("The isolated patch could not be divided into exact C1 spans")
            fixer = ShapeFix_Shape(divider.Result())
            fixer.SetPrecision(1e-7)
            fixer.SetMaxTolerance(1e-5)
            fixer.Perform()
            candidate = cq.Shape.cast(fixer.Shape())
        report["surface_geometry_validation"] = _validate_patch_partition(face, candidate)
        if any(BRepAdaptor_Surface(child.wrapped, True).UContinuity() == GeomAbs_C0
               or BRepAdaptor_Surface(child.wrapped, True).VContinuity() == GeomAbs_C0
               for child in candidate.Faces()):
            raise ValueError("The isolated patch retained an internal C0 span")
        report["normalized_face_count"] = len(candidate.Faces())
        quality = _assess_brep_self_intersections(candidate)
        report["native_quality"] = quality
        if quality["passed"]:
            report.update(passed=True, deferred_to_global=False)
        elif (quality.get("self_intersection_faults") and quality["failed_face_indices"]
              and not quality["unmapped_fault_count"] and not quality.get("other_fault_count", 0)
              and all(fault.get("shape_types") == ["FACE"]
                      for fault in quality["self_intersection_faults"])):
            report.update(passed=False, deferred_to_global=False)
        else:
            report.update(passed=None, deferred_to_global=True,
                          reason="Open-patch edge/vertex contacts or unsupported faults require the closed-body checker")
    except MemoryError:
        raise
    except Exception as error:
        # OCP binding exceptions such as StdFail_NotDone derive directly from
        # Exception. An unsupported open check must not preempt the required
        # closed-body check, whose result still controls CAD acceptance.
        report.update(passed=None, deferred_to_global=True, reason=f"{type(error).__name__}: {error}")
    report["elapsed_seconds"] = time.perf_counter() - started
    layout._intrinsic_surface_fingerprint = fingerprint
    layout._intrinsic_surface_report = deepcopy(report)
    return report
