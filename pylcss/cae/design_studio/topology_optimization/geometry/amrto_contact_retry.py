# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE.
"""Map failed final CAD supports back to current source-fitting charts.

This module supplies evidence only. It neither retries a fit nor changes a
source-fit acceptance result, and it does not authorize exporting a candidate.
"""
from __future__ import annotations

import numpy as np

from .cad_partition_proof import _same_signature
from .step_export import _geometry_signature


class ContactIntegrationError(RuntimeError):
    """Retain an explicitly failed native contact candidate inside this process.

    Shapes never enter the serialized reconstruction report. The integration
    boundary maps their supporting surfaces and optionally writes a BREP
    checkpoint before retaining only JSON-safe evidence.
    """

    def __init__(self, message, *, shape, quality, stage, evidence=None):
        super().__init__(message)
        self.shape = shape
        self.quality = quality
        self.stage = str(stage)
        self.evidence = dict(evidence or {})


def _normalized_c1_supports(face):
    """Yield exact affine-normalized C1 children of a rectangular spline."""
    import cadquery as cq
    from OCP.BRep import BRep_Tool
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace
    from OCP.TColStd import TColStd_Array1OfReal

    if face.geomType() != 'BSPLINE':
        return
    parent = BRep_Tool.Surface_s(face.wrapped)
    u0, u1, v0, v1 = face._uvBounds()
    axes = []
    for axis, lower, upper in (('U', u0, u1), ('V', v0, v1)):
        degree = getattr(parent, axis + 'Degree')()
        intervals = [lower]
        for index in range(1, getattr(parent, 'Nb' + axis + 'Knots')() + 1):
            knot = getattr(parent, axis + 'Knot')(index)
            if lower < knot < upper and getattr(parent, axis + 'Multiplicity')(index) >= degree:
                intervals.append(knot)
        axes.append([*intervals, upper])
    for first_u, last_u in zip(axes[0], axes[0][1:]):
        for first_v, last_v in zip(axes[1], axes[1][1:]):
            # Segment and change only a private surface. Original poles,
            # knots, parameter domains, topology and tolerances are untouched.
            child = parent.Copy()
            child.Segment(first_u, last_u, first_v, last_v)
            for axis, lower, upper in (('U', first_u, last_u), ('V', first_v, last_v)):
                count = getattr(child, 'Nb' + axis + 'Knots')()
                knots = TColStd_Array1OfReal(1, count)
                for index in range(1, count + 1):
                    knots.SetValue(index, (getattr(child, axis + 'Knot')(index) - lower) / (upper - lower))
                getattr(child, 'Set' + axis + 'Knots')(knots)
            native = BRepBuilderAPI_MakeFace(child, 0., 1., 0., 1., 1e-7).Face()
            yield cq.Face(native), [first_u, last_u, first_v, last_v]


def map_fitted_contact_layouts(fitted_shape, layout_ids, failed_shape, quality, *,
                               protected_shapes=(), geometry_tolerance=1e-8):
    """Return JSON-safe support provenance for mapped native contact faces.

    ``layout_ids`` must follow the original fitted face order, before sewing.
    ``protected_shapes`` is a sequence of ``(label, shape)`` source operands.
    A face is labeled original CAD only after its complete supporting-spline
    signature matches one of those operands. Unmatched faces remain unknown.
    Matching geometry identifies possible retry charts; all original source,
    material, void, native and STEP gates remain required after any retry.
    """
    if not np.isfinite(geometry_tolerance) or geometry_tolerance <= 0:
        raise ValueError('Contact support mapping requires a finite positive geometry tolerance')
    faces = fitted_shape.Faces()
    layout_ids = [int(value) for value in layout_ids]
    if len(faces) != len(layout_ids) or len(set(layout_ids)) != len(layout_ids):
        raise ValueError('Contact support mapping requires one unique layout ID per fitted face')
    result = {
        'method': 'Complete exact C1 supporting-spline signature correspondence',
        'evidence_only': True, 'source_fit_acceptance_changed': False,
        'closed_body_and_final_step_checks_required': True,
        'geometry_linear_tolerance': float(geometry_tolerance),
        'current_fitted_face_count': len(faces), 'normalized_c1_support_count': 0,
        'native_quality': quality, 'face_mappings': [], 'retry_layout_ids': [],
        'unknown_spline_face_indices': [], 'ambiguous_spline_face_indices': [],
        'retry_mapping_complete': False,
    }
    final_faces = failed_shape.Faces()
    indices = sorted({int(index) for fault in quality.get('self_intersection_faults', [])
                      for index in fault.get('face_indices', [])})
    if any(index < 0 or index >= len(final_faces) for index in indices):
        raise ValueError('Native contact face index is outside its failed candidate')
    targets = {}
    for index in indices:
        face = final_faces[index]
        row = {'failed_face_index': index, 'surface_type': face.geomType(),
               'fitted_matches': [], 'protected_operand_matches': []}
        result['face_mappings'].append(row)
        if face.geomType() == 'BSPLINE':
            try:
                targets[index] = (_geometry_signature(face, geometry_tolerance), row)
            except Exception as error:
                row['unsupported_signature_reason'] = f'{type(error).__name__}: {error}'
                row['provenance'] = 'unknown'
                result['unknown_spline_face_indices'].append(index)
        else:
            row['provenance'] = 'analytic support; not a fitted spline retry target'
    if not targets:
        return result
    for parent_index, face in enumerate(faces):
        for child, bounds in _normalized_c1_supports(face):
            signature = _geometry_signature(child, geometry_tolerance)
            result['normalized_c1_support_count'] += 1
            for target_signature, row in targets.values():
                if _same_signature(signature, target_signature):
                    row['fitted_matches'].append({
                        'layout_id': layout_ids[parent_index], 'fitted_face_index': parent_index,
                        'parent_parameter_bounds': bounds,
                        'complete_degree_control_weight_knot_signature_matches': True,
                    })
    for label, shape in protected_shapes:
        for operand_index, face in enumerate(shape.Faces()):
            if face.geomType() != 'BSPLINE':
                continue
            signature = _geometry_signature(face, geometry_tolerance)
            for target_signature, row in targets.values():
                if _same_signature(signature, target_signature):
                    row['protected_operand_matches'].append({
                        'operand_label': str(label), 'operand_face_index': operand_index,
                        'complete_degree_control_weight_knot_signature_matches': True,
                    })
    retry_ids = set()
    for index, (_, row) in targets.items():
        matches = row['fitted_matches']
        if matches and not row['protected_operand_matches']:
            ids = {match['layout_id'] for match in matches}
            row['provenance'] = 'current fitted source chart'
            if len(ids) != 1:
                result['ambiguous_spline_face_indices'].append(index)
            else:
                retry_ids.update(ids)
        elif row['protected_operand_matches'] and not matches:
            row['provenance'] = 'original CAD operand'
        elif matches:
            row['provenance'] = 'ambiguous fitted and original CAD support'
            result['ambiguous_spline_face_indices'].append(index)
        else:
            row['provenance'] = 'unknown'
            result['unknown_spline_face_indices'].append(index)
    result['retry_mapping_complete'] = not (
        result['unknown_spline_face_indices'] or result['ambiguous_spline_face_indices']
        or quality.get('unmapped_fault_count', 0) or quality.get('other_fault_count', 0))
    if result['retry_mapping_complete']:
        result['retry_layout_ids'] = sorted(retry_ids)
    return result
