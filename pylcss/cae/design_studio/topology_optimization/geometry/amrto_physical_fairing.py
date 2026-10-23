# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE.
"""Bounded physical-curvature fairing with exact boundary first derivatives.

The frozen metric/normal linear model integrates the squared second fundamental
form in an orthonormal physical tangent frame. It is not a UV second-difference
penalty. Both outer pole rings stay fixed, so clamped shared curves and their
one-sided first derivatives are identical. Actual nonlinear physical curvature,
source deviation, source orientation and stock bounds independently admit a
proposal. Sampled checks do not certify global injectivity or curvature bounds.
"""
from __future__ import annotations

from copy import deepcopy
from collections.abc import Mapping

import numpy as np
from scipy import sparse
from scipy.interpolate import BSpline
from scipy.sparse.linalg import spsolve

from .amrto_network_fitting import _tensor_rows
from .amrto_surface_fairing import _pole_normals


def _quadrature(knots, degree):
    spans = np.unique(knots[degree:-degree])
    order = 3 if len(spans) - 1 <= 12 else 2 if len(spans) - 1 <= 32 else 1
    nodes, weights = np.polynomial.legendre.leggauss(order)
    sites = np.concatenate([.5 * (a + b) + .5 * (b - a) * nodes
                            for a, b in zip(spans[:-1], spans[1:])])
    factors = np.concatenate([.5 * (b - a) * weights for a, b in zip(spans[:-1], spans[1:])])
    return sites, factors


def _operators(surface, *, maximum_samples):
    axes = [_quadrature(np.asarray(getattr(surface, 'knotvector_' + axis)),
                        getattr(surface, 'degree_' + axis)) for axis in ('u', 'v')]
    if len(axes[0][0]) * len(axes[1][0]) + 17 * 17 > maximum_samples:
        raise ValueError('All-span physical-curvature quadrature exceeds its explicit sample budget')
    uv = np.array([(u, v) for u in axes[0][0] for v in axes[1][0]])
    weights = .5 * np.outer(axes[0][1], axes[1][1]).ravel()
    # A single midpoint on every tiny span and a uniform quadrature catch
    # different geometric features. Average the two physical integral rules;
    # the independent curvature admission uses a distinct 15x15 grid.
    uniform = (np.arange(17) + .5) / 17
    uv = np.vstack((uv, np.array([(u, v) for u in uniform for v in uniform])))
    weights = np.r_[weights, np.full(17 * 17, .5 / (17 * 17))]
    basis = []
    for index, axis in enumerate(('u', 'v')):
        count = getattr(surface, 'ctrlpts_size_' + axis)
        function = BSpline(getattr(surface, 'knotvector_' + axis), np.eye(count),
                           getattr(surface, 'degree_' + axis))
        basis.append([sparse.csr_matrix(function(uv[:, index], nu=derivative)) for derivative in (0, 1, 2)])
    operators = {(a, b): _tensor_rows(basis[0][a], basis[1][b])
                 for a, b in ((0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2))}
    return operators, weights


def _physical_curvature(operators, controls, weights):
    du, dv, duu, duv, dvv = [operators[order] @ controls
                            for order in ((1, 0), (0, 1), (2, 0), (1, 1), (0, 2))]
    e, f, g = (np.einsum('ij,ij->i', a, b) for a, b in ((du, du), (du, dv), (dv, dv)))
    cross = np.cross(du, dv)
    area = np.linalg.norm(cross, axis=1)
    determinant = e * g - f * f
    if np.any(area <= 0) or np.any(determinant <= 1e-12 * np.maximum(e * g, np.finfo(float).tiny)):
        raise ValueError('Physical-curvature quadrature found a singular surface metric')
    normal = cross / area[:, None]
    second_u, mixed, second_v = [np.einsum('ij,ij->i', normal, value) for value in (duu, duv, dvv)]
    ratio = f / e
    k11 = second_u / e
    k12 = (mixed - ratio * second_u) / np.sqrt(determinant)
    k22 = (second_v - 2 * ratio * mixed + ratio ** 2 * second_u) * e / determinant
    components = np.stack((k11, k12, k22), axis=1)
    physical_weights = weights * area
    curvature_energy = float(physical_weights @ (k11 ** 2 + 2 * k12 ** 2 + k22 ** 2))
    mean = .5 * (k11 + k22)
    separation = .5 * np.sqrt((k11 - k22) ** 2 + 4 * k12 ** 2)
    principal = np.maximum(np.abs(mean + separation), np.abs(mean - separation))
    if not np.isfinite(components).all() or not np.isfinite(curvature_energy):
        raise ValueError('Physical-curvature quadrature is nonfinite')
    return {'e': e, 'f': f, 'g': g, 'determinant': determinant, 'normal': normal,
            'components': components, 'physical_weights': physical_weights,
            'energy': curvature_energy, 'maximum': float(principal.max()),
            'p95': float(np.percentile(principal, 95)),
            'rms': float(np.sqrt(physical_weights @ (principal ** 2) / physical_weights.sum()))}


def fair_physical_curvature(layout, maximum_displacement, *, maximum_controls=8192,
                             maximum_samples=8192, strengths=(.1, 1., 100.)):
    """Try at most three fixed-metric normal-only proposals, retaining the best.

    Return ``(surface, report)`` without mutating the layout. Exact shared
    positions and first derivatives are frozen. Final connected native-solid
    admission remains the caller's responsibility.
    """
    from .amrto_fit_orientation import assess_fit_orientation
    from .amrto_continuity import inspect_internal_spline_continuity
    from .amrto_network_quality import _evaluate, inspect_patch_quality
    from .amrto_nurbs_fitting import _surface_map_residual
    from .amrto_shared_stock import control_hull_in_box

    if (not np.isfinite(maximum_displacement) or maximum_displacement <= 0
            or not 1 <= len(strengths) <= 3 or not all(np.isfinite(s) and s > 0 for s in strengths)):
        raise ValueError('Physical fairing requires positive displacement and one to three positive strengths')
    original = layout.nurbs_surface
    report = {'layout_id': int(layout.layout_id), 'accepted': False,
              'method': 'Frozen physical-metric squared second-fundamental-form normal pole fit',
              'shared_boundary_positions_exactly_unchanged': True,
              'shared_boundary_first_derivatives_exactly_unchanged': True,
              'maximum_proposal_count': len(strengths), 'proposals': [],
              'maximum_surface_displacement_budget_mm': float(maximum_displacement),
              'global_curvature_or_injectivity_certificate': False,
              'connected_native_solid_admission_required': True}
    controls = np.asarray(original.ctrlpts).reshape(original.ctrlpts_size_u, original.ctrlpts_size_v, 3)
    if controls.shape[0] * controls.shape[1] > maximum_controls:
        return original, {**report, 'reason': 'Physical fairing control-count budget exceeded'}
    if min(controls.shape[:2]) <= 4:
        return original, {**report, 'reason': 'No poles remain after freezing boundary first derivatives'}
    weights = getattr(original, 'weights', None)
    if weights is not None and not np.all(np.asarray(weights) == 1.):
        return original, {**report, 'reason': 'Physical fairing currently requires unit rational weights'}
    for axis in ('u', 'v'):
        knots, degree = np.asarray(getattr(original, 'knotvector_' + axis)), getattr(original, 'degree_' + axis)
        if degree != 3 or np.count_nonzero(knots == 0.) != degree + 1 or np.count_nonzero(knots == 1.) != degree + 1:
            return original, {**report, 'reason': 'Exact boundary-derivative preservation requires clamped unit-domain cubics'}
    try:
        operators, quadrature = _operators(original, maximum_samples=maximum_samples)
        before = _physical_curvature(operators, controls.reshape(-1, 3), quadrature)
    except ValueError as error:
        return original, {**report, 'reason': str(error)}
    free_mask = np.zeros(controls.shape[:2], dtype=bool)
    free_mask[2:-2, 2:-2] = True
    free = np.flatnonzero(free_mask.ravel())
    normals = _pole_normals(original, controls).reshape(-1, 3)[free]
    flat = controls.reshape(-1, 3)
    e, f, determinant = before['e'], before['f'], before['determinant']
    ratio = f / e
    hessian = [operators[(2, 0)].multiply(1 / e[:, None]),
        (operators[(1, 1)] - operators[(2, 0)].multiply(ratio[:, None])).multiply(1 / np.sqrt(determinant)[:, None]),
        (operators[(0, 2)] - operators[(1, 1)].multiply(2 * ratio[:, None])
            + operators[(2, 0)].multiply(ratio[:, None] ** 2)).multiply((e / determinant)[:, None])]
    rows, targets = [], []
    for index, operator in enumerate(hessian):
        local = operator.tocsr()[:, free]
        projection = sum((local.multiply(before['normal'][:, axis, None]).multiply(normals[None, :, axis])
                          for axis in range(3)), sparse.csr_matrix(local.shape))
        multiplier = np.sqrt(before['physical_weights'] * (2. if index == 1 else 1.))
        rows.append(projection.multiply(multiplier[:, None]))
        targets.append(before['components'][:, index] * multiplier)
    matrix = sparse.vstack(rows, format='csr')
    target = np.concatenate(targets)
    normal = (matrix.T @ matrix).tocsc()
    diagonal = normal.diagonal()
    positive = diagonal[diagonal > 0]
    if not len(positive) or before['energy'] <= 1e-20:
        return original, {**report, 'reason': 'No physical curvature mode requires correction'}
    normalization = float(np.median(positive))
    normal /= normalization
    right = -(matrix.T @ target) / normalization
    bounds = getattr(layout, 'source_box_bounds', None)
    lower, upper = np.full(len(free), -maximum_displacement), np.full(len(free), maximum_displacement)
    if bounds is not None:
        for axis in range(3):
            keep = np.abs(normals[:, axis]) > 1e-14
            a = (np.asarray(bounds)[0, axis] - flat[free[keep], axis]) / normals[keep, axis]
            b = (np.asarray(bounds)[1, axis] - flat[free[keep], axis]) / normals[keep, axis]
            lower[keep] = np.maximum(lower[keep], np.minimum(a, b))
            upper[keep] = np.minimum(upper[keep], np.maximum(a, b))
    if np.any(lower > upper):
        return original, {**report, 'reason': 'Original free poles lie outside the stock movement intervals'}
    before_quality = inspect_patch_quality(layout, maximum_controls=maximum_controls)['physical_curvature']
    continuity_before = inspect_internal_spline_continuity(original)
    def continuity_rms(row):
        angles = [line['maximum_sampled_tangent_plane_jump_degrees'] for line in row['lines']
                  if line['maximum_sampled_tangent_plane_jump_degrees'] is not None]
        return float(np.sqrt(np.mean(np.square(angles)))) if angles else 0.
    continuity_rms_before = continuity_rms(continuity_before)
    edge_t = (np.arange(9) + .5) / 9
    edge_sites = np.array([(t, 0.) for t in edge_t] + [(1., t) for t in edge_t]
                          + [(t, 1.) for t in edge_t] + [(0., t) for t in edge_t])
    edge_before = _evaluate(original, edge_sites, second=False)
    best, best_energy = original, before['energy']
    for strength in strengths:
        movement = spsolve(sparse.eye(len(free), format='csc') + strength * normal, strength * right)
        movement = np.clip(movement, lower, upper)
        poles = flat.copy()
        poles[free] += movement[:, None] * normals
        row = {'strength': float(strength), 'accepted': False,
               'maximum_surface_displacement_bound_mm': float(np.linalg.norm(poles - flat, axis=1).max())}
        report['proposals'].append(row)
        if not np.isfinite(poles).all():
            row['reason'] = 'Nonfinite physical fairing solve'
            continue
        try:
            actual = _physical_curvature(operators, poles, quadrature)
        except ValueError as error:
            row['reason'] = str(error)
            continue
        row.update(physical_curvature_energy=actual['energy'], quadrature_curvature_maximum_per_mm=actual['maximum'])
        if actual['energy'] >= best_energy * (1 - 1e-4):
            row['reason'] = 'Actual physical curvature integral did not improve'
            continue
        candidate = deepcopy(original)
        candidate.set_ctrlpts(poles.tolist(), *controls.shape[:2])
        continuity_after = inspect_internal_spline_continuity(candidate)
        continuity_rms_after = continuity_rms(continuity_after)
        row.update(internal_C0_angle_rms_before_degrees=continuity_rms_before,
                   internal_C0_angle_rms_after_degrees=continuity_rms_after,
                   internal_C0_angle_maximum_before_degrees=continuity_before['maximum_sampled_tangent_plane_jump_degrees'],
                   internal_C0_angle_maximum_after_degrees=continuity_after['maximum_sampled_tangent_plane_jump_degrees'])
        if (continuity_rms_after > continuity_rms_before + 1e-5
                or continuity_after['maximum_sampled_tangent_plane_jump_degrees'] >
                   continuity_before['maximum_sampled_tangent_plane_jump_degrees'] + 1e-5
                or sum(line['undefined_one_sided_normal_count'] for line in continuity_after['lines']) >
                   sum(line['undefined_one_sided_normal_count'] for line in continuity_before['lines'])):
            row['reason'] = 'Internal C0 tangent-plane jumps or undefined normals worsened'
            continue
        edge_after = _evaluate(candidate, edge_sites, second=False)
        edge_change = max(float(np.linalg.norm(a - b, axis=1).max()) for a, b in zip(edge_before, edge_after))
        row['boundary_position_and_first_derivative_maximum_change'] = edge_change
        if edge_change > 1e-12:
            row['reason'] = 'Exact frozen boundary positions or first derivatives changed'
            continue
        working = deepcopy(layout)
        working.nurbs_surface = candidate
        quality = inspect_patch_quality(working, maximum_controls=maximum_controls)['physical_curvature']
        row['independent_physical_curvature'] = quality
        field = 'maximum_absolute_principal_curvature_per_mm'
        gradient = 'absolute_mean_curvature_neighbor_gradient_per_mm_squared'
        if (quality[field]['maximum'] > before_quality[field]['maximum'] * (1 + 1e-6)
                or quality[field]['p95'] > before_quality[field]['p95'] * (1 + 1e-6)
                or quality[gradient]['rms'] > before_quality[gradient]['rms'] * (1 + 1e-6)):
            row['reason'] = 'Independent physical curvature or its variation worsened'
            continue
        orientation = assess_fit_orientation(layout, candidate)
        deviation = _surface_map_residual(layout, candidate, grid_size=73)
        limit = float(getattr(layout, 'solid_source_fit_limit', maximum_displacement))
        stock = control_hull_in_box(poles, bounds) if bounds is not None else {'accepted': True}
        row.update(source_orientation=orientation, source_deviation_mm=float(deviation), source_limit_mm=limit, stock=stock)
        if not orientation['accepted'] or deviation > limit or not stock['accepted']:
            row['reason'] = 'Independent source orientation, source deviation or stock admission failed'
            continue
        best, best_energy = candidate, actual['energy']
        row.update(accepted=True, reason='Physical curvature, variation and source admission improved')
    report.update(accepted=best is not original, physical_curvature_energy_before=before['energy'],
                  physical_curvature_energy_after=best_energy, quadrature_site_count=len(quadrature),
                  free_normal_pole_count=len(free), control_count_unchanged=True,
                  independent_physical_curvature_before=before_quality,
                  maximum_surface_displacement_bound_mm=float(np.linalg.norm(np.asarray(best.ctrlpts) - flat, axis=1).max()),
                  displacement_reference='The surface entering this helper; independent strengths are never accumulated',
                  displacement_certificate='Unit-weight nonnegative B-spline partition of unity bounds the whole unchanged-basis surface by its largest pole movement')
    return best, report


def fair_network_physical_curvature(layouts, budgets, *, maximum_charts=12,
                                     minimum_dimensionless_roughness=1e-3):
    """Return ``(private_layout_copies, report)`` after at most 12 chart fits.

    ``budgets`` maps integer layout IDs to physical movement limits, or is a
    sequence aligned with ``layouts``. Each limit is measured from that chart
    on entry, not from successive strengths. Rank by sampled dimensionless
    physical mean-curvature gradient; exact/protected source charts, rejected
    source fits and charts without movement allowance are not proposed.
    Original layouts are never modified. The caller must independently accept
    the complete connected native solid before publishing the returned copy.
    """
    from .amrto_network_quality import inspect_patch_quality

    if (isinstance(maximum_charts, bool) or not isinstance(maximum_charts, int)
            or not 1 <= maximum_charts <= 12):
        raise ValueError('Physical network fairing permits one to twelve bounded chart attempts')
    if not np.isfinite(minimum_dimensionless_roughness) or minimum_dimensionless_roughness < 0:
        raise ValueError('Physical network fairing needs a nonnegative finite roughness threshold')
    if not isinstance(budgets, Mapping) and len(budgets) != len(layouts):
        raise ValueError('Physical fairing budgets must map layout IDs or match the layout sequence')
    candidates = deepcopy(layouts)
    ranked, skipped = [], []
    for index, layout in enumerate(candidates):
        identifier = int(layout.layout_id)
        budget = float(budgets.get(identifier, 0.) if isinstance(budgets, Mapping) else budgets[index])
        reason = None
        if not np.isfinite(budget) or budget <= 0:
            reason = 'No positive reserved physical movement budget'
        elif getattr(layout, 'require_exact_boundaries', False) or len(getattr(layout, 'protected_source_crease_edges', ())):
            reason = 'Explicitly exact or protected source chart'
        elif not getattr(layout, 'fitting_report', {}).get('source_fit_acceptance', {}).get('accepted', True):
            reason = 'Original source fitting admission is rejected'
        if reason is not None:
            skipped.append({'layout_id': identifier, 'reason': reason})
            continue
        try:
            quality = inspect_patch_quality(layout)['physical_curvature']
        except ValueError as error:
            skipped.append({'layout_id': identifier, 'reason': str(error)})
            continue
        roughness = quality['dimensionless_mean_curvature_gradient_rms']
        if roughness is None or not np.isfinite(roughness) or roughness < minimum_dimensionless_roughness:
            skipped.append({'layout_id': identifier, 'reason': 'No significant sampled physical curvature variation'})
            continue
        ranked.append((roughness, identifier, index, budget))
    ranked.sort(key=lambda value: (-value[0], value[1]))
    reports = []
    for roughness, identifier, index, budget in ranked[:maximum_charts]:
        candidate, report = fair_physical_curvature(candidates[index], budget)
        report['selection_dimensionless_curvature_gradient_rms'] = float(roughness)
        reports.append(report)
        if not report['accepted']:
            continue
        candidates[index].nurbs_surface = candidate
        candidates[index].physical_curvature_fairing_report = report
        best = next(row for row in reversed(report['proposals']) if row['accepted'])
        if hasattr(candidates[index], 'fitting_report'):
            fit = candidates[index].fitting_report.setdefault('source_fit_acceptance', {})
            fit.update(measured_deviation_mm=best['source_deviation_mm'], surface_orientation=best['source_orientation'],
                       stock_hull=best['stock'])
    return candidates, {'method': 'Bounded physical-curvature fairing ranked by dimensionless physical roughness',
        'maximum_chart_attempts': maximum_charts, 'maximum_linear_solves': 3 * maximum_charts,
        'eligible_chart_count': len(ranked), 'attempted_chart_count': len(reports),
        'accepted_chart_count': sum(row['accepted'] for row in reports),
        'accepted_layout_ids': [row['layout_id'] for row in reports if row['accepted']],
        'selection_threshold': minimum_dimensionless_roughness,
        'source_layouts_unchanged': True, 'shared_curves_and_first_derivatives_exactly_unchanged': True,
        'connected_native_solid_admission_required': True, 'patches': reports, 'skipped': skipped}


def remaining_physical_fairing_budgets(layouts, total_budgets, reference_surfaces, *, reserve_fraction=.25):
    """Return ``(per_layout_budget_map, report)`` for a bounded second stage.

    ``reference_surfaces`` contains the surfaces immediately before the joint
    normal fit, either keyed by layout ID or in layout order. Entries may also
    be layouts holding those surfaces. ``total_budgets`` follows the same map
    or aligned-sequence convention as ``fair_network_physical_curvature``.

    For an identical spline basis and unchanged positive weights, nonnegative
    basis functions bound the first stage's whole-surface displacement by
    ``D = max_i ||P_current_i - P_reference_i||``. The second stage may use
    ``min(reserve_fraction * B, max(0, B-D))``. The triangle inequality then
    bounds the combined displacement by ``D + second_budget <= B``. The
    second stage must retain that same basis, as the physical fairer does.
    Different bases, changed weights or missing references receive zero
    allowance; pole subtraction across incompatible bases proves no bound.

    This is a displacement budget, not a source-distance, topology or native
    solid certificate. Source and connected-solid admission remain mandatory.
    No input surface, chart or budget is modified.
    """
    if not np.isfinite(reserve_fraction) or not 0 <= reserve_fraction <= 1:
        raise ValueError('The physical fairing reserve fraction must be finite and in [0, 1]')
    for name, values in (('total budgets', total_budgets), ('reference surfaces', reference_surfaces)):
        if not isinstance(values, Mapping) and len(values) != len(layouts):
            raise ValueError(f'Physical fairing {name} must map layout IDs or match the layout sequence')
    budgets, rows = {}, []
    for index, layout in enumerate(layouts):
        identifier = int(layout.layout_id)
        budget = float(total_budgets.get(identifier, 0.) if isinstance(total_budgets, Mapping) else total_budgets[index])
        reference = (reference_surfaces.get(identifier) if isinstance(reference_surfaces, Mapping)
                     else reference_surfaces[index])
        row = {'layout_id': identifier, 'total_movement_budget_mm': budget,
               'physical_fairing_budget_mm': 0., 'same_basis_and_weights': False,
               'reserve_fraction': float(reserve_fraction), 'complete_surface_displacement_bound_available': False}
        rows.append(row)
        budgets[identifier] = 0.
        if not np.isfinite(budget) or budget <= 0:
            row['reason'] = 'No positive finite total movement budget'
            continue
        if reference is None:
            row['reason'] = 'No pre-joint surface reference; combined displacement cannot be bounded'
            continue
        if hasattr(reference, 'nurbs_surface'):
            if int(reference.layout_id) != identifier:
                row['reason'] = 'Aligned reference layout ID does not match the current chart'
                continue
            reference = reference.nurbs_surface
        current = layout.nurbs_surface
        same_basis = all(getattr(current, name) == getattr(reference, name) for name in
                         ('degree_u', 'degree_v', 'ctrlpts_size_u', 'ctrlpts_size_v'))
        same_basis = same_basis and all(np.array_equal(getattr(current, name), getattr(reference, name))
                                       for name in ('knotvector_u', 'knotvector_v'))
        if not same_basis:
            row['reason'] = 'Pre-joint and current surface bases differ; pole movements are not comparable'
            continue
        controls, initial = np.asarray(current.ctrlpts, dtype=float), np.asarray(reference.ctrlpts, dtype=float)
        if (controls.shape != initial.shape or controls.ndim != 2 or controls.shape[1] != 3
                or not np.isfinite(controls).all() or not np.isfinite(initial).all()):
            row['reason'] = 'Surface controls must be finite matching 3D arrays'
            continue
        weights = []
        for surface in (current, reference):
            value = getattr(surface, 'weights', None)
            weights.append(np.ones(len(controls)) if value is None else np.asarray(value, dtype=float))
        if (any(value.shape != (len(controls),) or not np.isfinite(value).all() or np.any(value <= 0)
                for value in weights) or not np.array_equal(weights[0], weights[1])):
            row['reason'] = 'Surface weights changed or are not finite positive matching weights'
            continue
        used = float(np.linalg.norm(controls - initial, axis=1).max(initial=0.))
        remaining = max(0., budget - used)
        allowance = min(reserve_fraction * budget, remaining)
        arithmetic_allowance = 64 * np.finfo(float).eps * max(float(np.abs(controls).max(initial=0.)), budget, 1.)
        budgets[identifier] = allowance
        row.update(same_basis_and_weights=True, complete_surface_displacement_bound_available=True,
            largest_pole_movement_since_pre_joint_mm=used,
            remaining_total_allowance_mm=remaining,
            reserved_fraction_maximum_mm=float(reserve_fraction * budget),
            physical_fairing_budget_mm=allowance,
            maximum_combined_surface_displacement_bound_mm=used + allowance,
            arithmetic_rounding_allowance_mm=arithmetic_allowance,
            first_stage_exceeds_total_budget=bool(used > budget + arithmetic_allowance),
            reason='Triangle-inequality budget for unchanged positive-weight spline bases')
    return budgets, {'method': 'Conservative remaining movement bound from pre-joint spline poles',
        'reserve_fraction': float(reserve_fraction), 'charts': rows,
        'positive_budget_chart_count': sum(value > 0 for value in budgets.values()),
        'maximum_second_stage_fraction_of_original_budget': float(reserve_fraction),
        'source_layouts_and_references_unchanged': True,
        'second_stage_must_preserve_basis_and_weights': True,
        'source_and_connected_native_solid_admission_required': True}
