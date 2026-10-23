# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE.
"""One private, whole-network interpolation after a failed native proposal.

No chart is rolled back individually and no fitting runs. Fixed positive
weights and an identical basis make pole interpolation an exact convex
interpolation of the two surface geometries. Existing common boundaries stay
exact. The caller must still repeat native solid, Boolean and STEP checks.
"""
from __future__ import annotations

from copy import deepcopy
from time import perf_counter

import numpy as np
from scipy.interpolate import BSpline


def _surface_signature(surface):
    poles = np.asarray(surface.ctrlpts, dtype=float)
    shape = (int(surface.ctrlpts_size_u), int(surface.ctrlpts_size_v))
    weights = getattr(surface, 'weights', None)
    weights = np.ones(len(poles)) if weights is None else np.asarray(weights, dtype=float)
    if (poles.shape != (shape[0]*shape[1], 3) or not np.isfinite(poles).all()
            or weights.shape != (len(poles),) or not np.isfinite(weights).all() or np.any(weights <= 0)):
        raise ValueError('Backoff requires finite Euclidean poles and positive fixed weights')
    return poles, weights, shape


def _same_source(first, second):
    return (np.array_equal(first.vertices, second.vertices)
            and np.array_equal(first.triangles, second.triangles)
            and np.array_equal(first.parameter_coords, second.parameter_coords)
            and np.array_equal(np.asarray(getattr(first, 'protected_source_crease_edges', ())),
                               np.asarray(getattr(second, 'protected_source_crease_edges', ()))))


def _source_deviation(layout, surface, *, grid_size=73):
    """Independent rational-aware source map check, including every source node."""
    from .amrto_nurbs_fitting import _knot_span_samples
    from .amrto_parameterization import sample_parameter_grid
    poles, weights, shape = _surface_signature(surface)
    controls = poles.reshape(*shape, 3)
    weights = weights.reshape(shape)
    uniform = (np.arange(grid_size)+.5)/grid_size
    u = np.unique(np.r_[uniform, _knot_span_samples(surface.knotvector_u)])
    v = np.unique(np.r_[uniform, _knot_span_samples(surface.knotvector_v)])
    if len(u)*len(v) > 1_000_000:
        raise ValueError('Backoff source validation exceeds its explicit sampling budget')
    axes = [BSpline(surface.knotvector_u, np.eye(shape[0]), surface.degree_u),
            BSpline(surface.knotvector_v, np.eye(shape[1]), surface.degree_v)]
    def evaluate(au, av, paired=False):
        specification = 'ni,ijc,nj->nc' if paired else 'ui,ijc,vj->uvc'
        denominator_specification = 'ni,ij,nj->n' if paired else 'ui,ij,vj->uv'
        numerator = np.einsum(specification, au, controls*weights[:, :, None], av, optimize=True)
        denominator = np.einsum(denominator_specification, au, weights, av, optimize=True)
        if not np.isfinite(denominator).all() or np.any(denominator <= 0):
            raise ValueError('Backoff rational validation has a nonpositive denominator')
        return numerator/denominator[..., None]
    source = sample_parameter_grid(layout, u, v)
    evaluated = evaluate(axes[0](u), axes[1](v))
    error = float(np.linalg.norm(evaluated-source, axis=2).max(initial=0.))
    ids = np.unique(np.asarray(layout.triangles))
    uv = np.asarray(layout.parameter_coords)[ids]
    nodes = evaluate(axes[0](uv[:, 0]), axes[1](uv[:, 1]), paired=True)
    error = max(error, float(np.linalg.norm(nodes-np.asarray(layout.vertices)[ids], axis=1).max(initial=0.)))
    return error


def backoff_surface_network(layouts, candidate, factor=.5):
    """Return private scaled charts and an independently admitted proposal.

    Both inputs remain untouched. Same source ownership, source UVs, basis,
    fixed positive weights and exact boundary poles are mandatory. Unchanged
    pre-existing failed source charts are explicitly retained as warnings;
    changing such a chart or introducing a new failure rejects the proposal.
    Exactly one requested factor is evaluated, without retries or refitting.
    """
    from .amrto_fit_orientation import assess_fit_orientation
    from .amrto_network_quality import inspect_network_quality
    from .amrto_network_reconstruction import compare_network_quality
    from .amrto_shared_stock import control_hull_in_box

    started = perf_counter()
    if not np.isfinite(factor) or not 0 < factor < 1:
        raise ValueError('Whole-network backoff requires a finite factor strictly between zero and one')
    private = deepcopy(list(layouts))
    report = {'accepted': False, 'method': 'One common-factor interpolation of the entire surface network',
        'factor': float(factor), 'source_XYZ_and_UV_unchanged': True,
        'shared_boundary_geometry_unchanged': True, 'individual_patch_rollback': False,
        'fitting_batches': 0, 'whole_proposal_native_admission_required': True,
        'global_G1_or_G2_certified': False, 'patches': []}
    original_ids = [int(item.layout_id) for item in layouts]
    candidate_ids = [int(item.layout_id) for item in candidate]
    report['initial_source_fit_warning_layout_ids'] = sorted(int(item.layout_id) for item in layouts
        if not item.fitting_report.get('source_fit_acceptance', {}).get('accepted', False))
    report['retained_source_fit_warning_layout_ids'] = []
    try:
        if (len(set(original_ids)) != len(original_ids) or len(set(candidate_ids)) != len(candidate_ids)
                or set(original_ids) != set(candidate_ids)):
            raise ValueError('Backoff requires identical unique chart ownership')
        by_id = {int(item.layout_id): item for item in candidate}
        plans = []
        for base, target in zip(layouts, private):
            proposed = by_id[int(base.layout_id)]
            if not _same_source(base, proposed):
                raise ValueError(f'Chart {base.layout_id} changed its original source or UV ownership')
            a, wa, shape = _surface_signature(base.nurbs_surface)
            b, wb, other_shape = _surface_signature(proposed.nurbs_surface)
            if (shape != other_shape or not np.array_equal(wa, wb)
                    or base.nurbs_surface.degree_u != proposed.nurbs_surface.degree_u
                    or base.nurbs_surface.degree_v != proposed.nurbs_surface.degree_v
                    or not np.array_equal(base.nurbs_surface.knotvector_u, proposed.nurbs_surface.knotvector_u)
                    or not np.array_equal(base.nurbs_surface.knotvector_v, proposed.nurbs_surface.knotvector_v)):
                raise ValueError(f'Chart {base.layout_id} changed its spline basis or fixed weights')
            boundary = np.ones(shape, bool); boundary[1:-1, 1:-1] = False
            if not np.array_equal(a[boundary.ravel()], b[boundary.ravel()]):
                raise ValueError(f'Chart {base.layout_id} moved an exact common boundary pole')
            initially_accepted = bool(base.fitting_report.get('source_fit_acceptance', {}).get('accepted', False))
            if not initially_accepted and not np.array_equal(a, b):
                raise ValueError(f'Chart {base.layout_id} modified a pre-existing rejected source fit')
            poles = a+factor*(b-a)
            # The Euclidean setter preserves positive NURBS weights as well
            # as ordinary polynomial B-spline control-net dimensions.
            target.nurbs_surface.ctrlpts = poles.tolist()
            plans.append((base, target, a, poles, initially_accepted))
        for base, target, initial_poles, poles, initially_accepted in plans:
            orientation = assess_fit_orientation(base, target.nurbs_surface)
            deviation = _source_deviation(base, target.nurbs_surface)
            acceptance = base.fitting_report.get('source_fit_acceptance', {})
            limit = float(getattr(base, 'solid_source_fit_limit', acceptance.get('maximum_deviation_mm', 0.)))
            bounds = getattr(base, 'source_box_bounds', None)
            stock = (control_hull_in_box(poles, bounds) if bounds is not None
                     else {'accepted': False, 'reason': 'No source stock bounds were supplied'})
            unchanged = bool(np.array_equal(initial_poles, poles))
            retained_warning = bool(not initially_accepted and unchanged)
            good = bool(orientation['accepted'] and deviation <= limit and stock['accepted'])
            row = {'layout_id': int(base.layout_id), 'accepted': bool(good or retained_warning),
                'independent_source_checks_accepted': good, 'source_fit_warning_retained_unchanged': retained_warning,
                'source_deviation_mm': deviation, 'source_limit_mm': limit,
                'orientation': orientation, 'stock': stock, 'changed': not unchanged,
                'maximum_pole_displacement_mm': float(np.linalg.norm(poles-initial_poles, axis=1).max(initial=0.)),
                'boundary_poles_unchanged': True, 'common_component_factor': float(factor)}
            report['patches'].append(row)
            if retained_warning:
                report['retained_source_fit_warning_layout_ids'].append(int(base.layout_id))
            target.fitting_report['source_fit_acceptance'].update(
                accepted=bool(good and initially_accepted), measured_deviation_mm=deviation,
                surface_orientation=orientation, stock_hull=stock)
            target.fitting_report['maximum_source_map_residual'] = deviation
            target.fitting_report['source_orientation'] = orientation
            target.fitting_report['network_backoff'] = deepcopy(row)
        report['new_source_fit_failure_layout_ids'] = [item['layout_id'] for item in report['patches'] if not item['accepted']]
        if report['new_source_fit_failure_layout_ids']:
            report['reason'] = 'Whole-network backoff introduced a source, orientation or stock failure'
            return private, report
        report['before'] = inspect_network_quality(layouts)
        report['after'] = inspect_network_quality(private)
        report['surface_quality_admission'] = compare_network_quality(report['before'], report['after'])
        report['accepted'] = bool(report['surface_quality_admission']['accepted'])
        if not report['accepted']:
            report['reason'] = 'The scaled network did not improve measured joins without worsening network quality'
        return private, report
    except (ValueError, FloatingPointError) as error:
        report['reason'] = str(error)
        return private, report
    finally:
        report['elapsed_seconds'] = perf_counter()-started
