# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE.
"""Retain joined CAD boundaries while applying the paper's optional MRA.

This is an explicit engineering extension to Algorithms 2/3. It does not
refit a patch, alter its partition, or retry reconstruction. Exact knot
insertion aligns the two bases before restoring the original boundary rows.
A convex-hull bound admits only bounded changes to the original fitted patch.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import math
from typing import Any, Callable

import numpy as np
from geomdl import operations

from .amrto_multiresolution import MultiresolutionResult, apply_paper_multiresolution_strategy


def restore_joined_boundaries(reference: Any, candidate: Any) -> tuple[Any, dict]:
    """Restore four exact clamped boundary curves in a common spline basis.

    Both inputs remain unchanged. A knot removed by Algorithm 3 is reinserted
    exactly when the engineering join needs the original boundary basis.
    The reported displacement bounds the complete surface, not just samples.
    Rational surfaces require identical positive weights after insertion.
    """
    for axis in ('u', 'v'):
        if getattr(reference, 'degree_' + axis) != getattr(candidate, 'degree_' + axis):
            raise ValueError('Boundary restoration requires matching surface degrees')
        degree = getattr(reference, 'degree_' + axis)
        a, b = getattr(reference, 'knotvector_' + axis), getattr(candidate, 'knotvector_' + axis)
        if (a[degree], a[-degree - 1]) != (b[degree], b[-degree - 1]):
            raise ValueError('Boundary restoration requires matching parameter domains')
        if (a.count(a[0]) != degree + 1 or a.count(a[-1]) != degree + 1
                or b.count(b[0]) != degree + 1 or b.count(b[-1]) != degree + 1):
            raise ValueError('Boundary restoration requires clamped surface boundaries')
    if bool(reference.rational) != bool(candidate.rational):
        raise ValueError('Boundary restoration requires matching rational types')

    before, after = deepcopy(reference), deepcopy(candidate)
    insertions = []
    for index, axis in enumerate(('u', 'v')):
        counts_a = Counter(getattr(before, 'knotvector_' + axis))
        counts_b = Counter(getattr(after, 'knotvector_' + axis))
        for knot in sorted(counts_a.keys() | counts_b.keys()):
            target = max(counts_a[knot], counts_b[knot])
            for label, surface, counts in (('reference', before, counts_a), ('candidate', after, counts_b)):
                missing = target - counts[knot]
                if missing:
                    values, numbers = [None, None], [0, 0]
                    values[index], numbers[index] = knot, missing
                    operations.insert_knot(surface, values, numbers)
                    insertions.append({'surface': label, 'axis': axis, 'knot': knot, 'count': missing})
    if any(getattr(before, 'knotvector_' + axis) != getattr(after, 'knotvector_' + axis)
           for axis in ('u', 'v')):
        raise ValueError('Boundary restoration could not establish an identical spline basis')

    shape = (after.ctrlpts_size_u, after.ctrlpts_size_v, -1)
    values = np.asarray(after.ctrlptsw if after.rational else after.ctrlpts, dtype=float).reshape(shape).copy()
    border = np.asarray(before.ctrlptsw if before.rational else before.ctrlpts, dtype=float).reshape(shape)
    if not np.isfinite(values).all() or not np.isfinite(border).all():
        raise ValueError('Boundary restoration requires finite control points')
    values[[0, -1], :, :] = border[[0, -1], :, :]
    values[:, [0, -1], :] = border[:, [0, -1], :]
    after.set_ctrlpts(values.reshape(-1, values.shape[-1]).tolist(), shape[0], shape[1])

    weights_match = not before.rational or (
        np.array_equal(before.weights, after.weights)
        and np.isfinite(before.weights).all() and np.all(np.asarray(before.weights) > 0))
    bound = (float(np.linalg.norm(np.asarray(after.ctrlpts) - np.asarray(before.ctrlpts), axis=1).max())
             if weights_match else None)
    report = {
        'method': 'Exact common-basis knot insertion and original homogeneous boundary rows',
        'engineering_join_extension': True,
        'reference_control_shape': [reference.ctrlpts_size_u, reference.ctrlpts_size_v],
        'candidate_control_shape': [candidate.ctrlpts_size_u, candidate.ctrlpts_size_v],
        'joined_control_shape': [after.ctrlpts_size_u, after.ctrlpts_size_v],
        'knot_insertions': insertions,
        'boundary_control_rows_identical': True,
        'surface_displacement_bound': bound,
        'bound_certified': bool(weights_match),
        'bound_method': 'Convex hull of control-point differences in the identical positive-weight spline basis',
    }
    if not weights_match:
        report['bound_failure'] = 'Rational weights differ or are not finite and positive'
    return after, report


def apply_joined_multiresolution(
    surface: Any, *, maximum_displacement: float,
    curvature: float | None = None, curvature_ck: float = .1, curvature_cp: float = 10.,
    compression_ratio: float = .5, layers: int = 1,
    maximum_knot_removal_attempts: int = 64,
    orientation_evaluator: Callable[[Any], dict] | None = None,
) -> MultiresolutionResult:
    """Run Algorithms 2/3 once, preserving joins and bounded surface fidelity.

    An inadmissible optional MRA candidate leaves the initial fitted patch
    unchanged. The report retains its executed algorithm and rejection reason.
    An optional independent source-orientation evaluator must return a report
    containing ``accepted``. A displacement bound alone cannot exclude a fold
    in a narrow knot span. Failed or unavailable orientation validation keeps
    the initial fit; retaining it does not establish its own validity. The
    evaluator can close over the layout's actual source geometry and UV map.
    Its finite checks do not establish a continuous injectivity bound.
    No additional fit or partition pass is performed.
    """
    if not math.isfinite(maximum_displacement) or maximum_displacement < 0:
        raise ValueError('Maximum multiresolution displacement must be finite and nonnegative')
    result = apply_paper_multiresolution_strategy(
        surface, curvature=curvature, curvature_ck=curvature_ck, curvature_cp=curvature_cp,
        compression_ratio=compression_ratio, layers=layers,
        maximum_knot_removal_attempts=maximum_knot_removal_attempts)
    candidate, join = restore_joined_boundaries(surface, result.candidate)
    bound = join['surface_displacement_bound']
    accepted = bool(join['bound_certified'] and bound <= maximum_displacement)
    rejection = None
    if not accepted:
        rejection = (join.get('bound_failure') if not join['bound_certified'] else
                     f'Multiresolution surface displacement bound {bound:.9g} exceeds {maximum_displacement:.9g}')
    orientation = None
    if accepted and orientation_evaluator is not None:
        try:
            orientation = orientation_evaluator(candidate)
            if not isinstance(orientation, dict) or not isinstance(orientation.get('accepted'), (bool, np.bool_)):
                raise ValueError('Orientation evaluator must return a report with boolean accepted')
            if not orientation['accepted']:
                accepted = False
                rejection = 'Multiresolution candidate fails independent source orientation validation'
        except (ValueError, ArithmeticError, np.linalg.LinAlgError) as error:
            accepted = False
            orientation = {'accepted': False, 'error': str(error)}
            rejection = 'Multiresolution candidate orientation validation could not be completed'
    result.candidate = candidate if accepted else deepcopy(surface)
    result.report.update(
        paper_algorithms_executed=list(result.report.get('paper_algorithms', [])),
        boundary_join=join, maximum_displacement=float(maximum_displacement),
        candidate_accepted=accepted, candidate_rejected=not accepted,
        accepted=accepted, boundary_restoration_extension=True,
        effective_editable_compression_ratio=(result.report.get('actual_compression_ratio', 1.)
                                              if accepted else 1.),
        fidelity_admission_method='Certified control-net convex-hull displacement bound; no refit',
        candidate_source_orientation=orientation,
        independent_orientation_validation_requested=orientation_evaluator is not None,
        output_control_shape=[result.candidate.ctrlpts_size_u, result.candidate.ctrlpts_size_v],
    )
    if not accepted:
        result.report['candidate_rejection_reason'] = rejection
        result.report['output'] = 'Original joined fit retained without another fitting pass'
    else:
        result.report['output'] = 'Boundary-preserving multiresolution candidate'
    return result
