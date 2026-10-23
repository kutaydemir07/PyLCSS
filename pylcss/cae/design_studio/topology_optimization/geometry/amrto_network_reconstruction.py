# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE.
"""One private shared-curve refit and a coupled surface proposal.

This module never publishes a CAD body. The caller must repeat native solid,
prescribed material/void, volume and STEP checks before replacing its body.
Changing common curves requires whole-proposal rollback, not isolated faces.
"""
from copy import deepcopy
from contextlib import nullcontext
from time import perf_counter

import numpy as np


def _artificial_seam_summary(quality):
    category = quality.get('shared_seams', {}).get('categories', {}).get('artificial_seam', {})
    return category, category.get('sampled_tangent_plane_jump_degrees', {})


def compare_network_quality(before, after):
    """Compare the same source segments without dropping difficult samples."""
    checks = {}
    checks['all_charts_measured'] = bool(
        after.get('measured_patch_count') == before.get('measured_patch_count')
        and not after.get('invalid_or_unsupported_patches'))
    first, old = _artificial_seam_summary(before)
    second, new = _artificial_seam_summary(after)
    checks['same_artificial_seam_coverage'] = bool(
        first.get('segment_count') == second.get('segment_count')
        and old.get('sample_count', 0) == new.get('sample_count', 0)
        and first.get('noncoincident_samples', 0) >= second.get('noncoincident_samples', 0)
        and first.get('undefined_normal_samples', 0) >= second.get('undefined_normal_samples', 0))
    for key in ('rms', 'p95', 'maximum'):
        a, b = old.get(key), new.get(key)
        checks[f'artificial_seam_{key}_not_worse'] = bool(
            (old.get('sample_count') == new.get('sample_count') == 0)
            or (a is not None and b is not None and np.isfinite([a, b]).all() and b <= a + 1e-5))
    old_internal = before.get('internal_C0_representation_lines', {})
    new_internal = after.get('internal_C0_representation_lines', {})
    checks['no_more_internal_lines_over_5_degrees'] = bool(
        new_internal.get('lines_over_5_degrees', np.inf)
        <= old_internal.get('lines_over_5_degrees', -np.inf))
    checks['physical_improvement_measured'] = bool(
        (old.get('rms') is not None and new.get('rms') is not None
         and new['rms'] < old['rms'] - 1e-5)
        or new_internal.get('lines_over_5_degrees', np.inf)
        < old_internal.get('lines_over_5_degrees', -np.inf))
    return {'accepted': all(checks.values()), 'checks': checks,
            'scope': 'Same bounded source-segment samples; no global continuity certificate'}


def prepare_surface_network_proposal(layouts, payload, config, *, boundary_tolerance,
                                     fit_layouts, progress_callback=None,
                                     refit_shared_boundaries=None, fixed_layout_ids=()):
    """Return private charts plus an explicit source/quality admission report.

    ``fit_layouts`` performs one fitting batch only when curves/UVs change. Its
    worker policy is supplied by reconstruction, keeping this helper independent
    of processes.
    """
    from .amrto_network_fitting import fair_surface_network
    from .amrto_network_quality import inspect_network_quality, compare_network_ripple_quality
    from .amrto_shared_curves import prepare_shared_boundary_curves
    from .amrto_surface_fairing import fairing_displacement_budget

    started = perf_counter()
    refit = (getattr(config, 'fair_network_boundary_curves', True)
             if refit_shared_boundaries is None else bool(refit_shared_boundaries))
    candidate = deepcopy(layouts)
    report = {'accepted': False, 'engineering_extension': True,
              'method': 'Source-bounded fair shared curves and coupled point/normal/bending surfaces',
              'original_source_XYZ_unchanged': True, 'patch_partition_unchanged': True,
              'shared_boundary_geometry_may_change': refit,
              'proposal_basis': 'fair_shared_curves_and_repaired_UV' if refit else 'existing_common_curves',
              'whole_proposal_native_admission_required': True,
              'independent_contact_patch_rollback_allowed': False,
              'global_G1_or_G2_certified': False, 'fitting_batches': 0}
    report['native_contact_fixed_layout_ids'] = sorted(map(int, fixed_layout_ids))

    def progress(message):
        if callable(progress_callback):
            progress_callback(message)

    progress('Measuring CAD seams and curvature before the connected fit')
    report['before'] = inspect_network_quality(layouts)
    try:
        bounds = getattr(config, 'source_box_bounds', None)
        if bounds is None:
            bounds = getattr(layouts[0], 'source_box_bounds', None) if layouts else None
        if refit:
            progress('Fairing common CAD curves and aligning source boundary parameters')
            report['shared_curves'] = prepare_shared_boundary_curves(
                candidate, boundary_tolerance, bounds=bounds,
                certify_stock_hull=bounds is not None, blocked_junction_positions=(),
                smooth_native_boundary_clocks=True,
                fair_shared_curves=getattr(config, 'fair_network_boundary_curves', True))
        if refit and getattr(config, 'repair_network_uv', True):
            from .amrto_uv_quality_repair import repair_network_uv_distortion
            progress('Correcting genuinely distorted source parameterization within fixed shared boundaries')
            report['parameterization_correction'] = repair_network_uv_distortion(
                candidate, maximum_charts=3, maximum_iterations=32)
        if refit:
            candidate = list(fit_layouts(candidate))
            report['fitting_batches'] = 1
        original_failures = {int(item.layout_id) for item in layouts
            if not item.fitting_report.get('source_fit_acceptance', {}).get('accepted', False)}
        current_failures = {int(item.layout_id) for item in candidate
            if not item.fitting_report.get('source_fit_acceptance', {}).get('accepted', False)}
        report['initial_source_fit_warning_layout_ids'] = sorted(original_failures)
        report['proposed_source_fit_warning_layout_ids'] = sorted(current_failures)
        if not current_failures.issubset(original_failures):
            report['reason'] = 'Connected refitting introduced new source-fit or orientation failures'
            return candidate, report
        original_by_id = {item.layout_id: item for item in layouts}
        if (len(candidate) != len(layouts) or {item.layout_id for item in candidate} != set(original_by_id)
                or any(not np.array_equal(item.vertices, original_by_id[item.layout_id].vertices)
                       or not np.array_equal(item.triangles, original_by_id[item.layout_id].triangles)
                       for item in candidate)):
            raise ValueError('Connected refitting changed the original source geometry or chart coverage')
        budgets = {item.layout_id: fairing_displacement_budget(item, payload)[0] for item in candidate}
        for identifier in fixed_layout_ids:
            budgets[int(identifier)] = 0.
        reference_surfaces = {item.layout_id: deepcopy(item.nurbs_surface) for item in candidate}
        report['coupled_fitting'] = fair_surface_network(
            candidate, budgets,
            normal_weight=getattr(config, 'network_normal_weight', 100.),
            bending_weight=getattr(config, 'network_bending_weight', 5.),
            interior_normal_weight=getattr(config, 'network_interior_normal_weight', 25.),
            source_crease_angle=config.seam_source_crease_angle,
            maximum_controls=getattr(config, 'network_maximum_controls', 65536),
            maximum_iterations=getattr(config, 'network_maximum_iterations', 3),
            fixed_layout_ids=fixed_layout_ids,
            progress_callback=progress)
        if getattr(config, 'fair_network_physical_curvature', True):
            from .amrto_physical_fairing import (fair_network_physical_curvature,
                                                remaining_physical_fairing_budgets)
            remaining, certificate = remaining_physical_fairing_budgets(
                candidate, budgets, reference_surfaces, reserve_fraction=.25)
            if any(row.get('first_stage_exceeds_total_budget', False) for row in certificate['charts']):
                report['reason'] = 'The joint fit exceeded its complete surface movement allowance'
                report['combined_movement_budget'] = certificate
                return candidate, report
            progress('Fairing physical curvature within the remaining source movement allowance')
            candidate, report['physical_curvature_fairing'] = fair_network_physical_curvature(
                candidate, remaining, maximum_charts=12)
            report['physical_curvature_fairing']['combined_movement_budget'] = certificate
        progress('Checking common-curve positions, normal jumps and curvature after the connected fit')
        report['after'] = inspect_network_quality(candidate)
        report['surface_quality_admission'] = compare_network_quality(report['before'], report['after'])
        report['accepted'] = bool(report['surface_quality_admission']['accepted'])
        if not report['accepted']:
            report['reason'] = 'The connected proposal did not improve the measured network without worsening joins'
        else:
            progress('Comparing interior ripples at matching physical source regions')
            report['ripple_quality_admission'] = compare_network_ripple_quality(layouts, candidate)
        return candidate, report
    finally:
        report['elapsed_seconds'] = perf_counter() - started


def complete_surface_network(layouts, payload, config, *, boundary_tolerance,
                             fit_layouts, complete_native, progress_callback=None):
    """Try at most three private proposals with native and ripple admission.

    Existing common curves are tried first: no new fitting batch or enlarged
    boundary-knot unions are needed. A failed fit, source/quality check, native
    reconstruction or visible-ripple check can trigger one shared-curve/UV
    refit. One complete native contact mapping can instead constrain a single
    repeat joint fit against the original contact charts. It is not isolated
    patch rollback: all neighbors are refitted and every check is repeated.
    Each attempt starts from the same input; movements never accumulate.
    ``complete_native`` returns a shape and its independent solid report.
    """
    from .amrto_network_quality import confirm_network_ripple_visibility

    started, attempts = perf_counter(), []
    try:
        from threadpoolctl import threadpool_limits
        limiter = threadpool_limits(limits=1, user_api='blas')
    except ImportError:
        limiter = nullcontext()
    modes = [(False, ())]
    if getattr(config, 'fair_network_boundary_curves', True) or getattr(config, 'repair_network_uv', True):
        modes.append((True, ()))
    contact_retry_scheduled = False
    with limiter:
        for refit, fixed_ids in modes:
            report = {'accepted': False, 'proposal_basis':
                      'fair_shared_curves_and_repaired_UV' if refit else 'existing_common_curves'}
            try:
                candidate, report = prepare_surface_network_proposal(
                    layouts, payload, config, boundary_tolerance=boundary_tolerance,
                    fit_layouts=fit_layouts, progress_callback=progress_callback,
                    refit_shared_boundaries=refit, fixed_layout_ids=fixed_ids)
                if report['accepted']:
                    shape, native = complete_native(candidate)
                    report['solid_integration'] = native
                    if native.get('is_solid'):
                        report['ripple_quality_admission'] = confirm_network_ripple_visibility(
                            report['ripple_quality_admission'], shape)
                        report['accepted'] = report['ripple_quality_admission'].get('accepted') is True
                        if report['accepted']:
                            report.update(attempts=attempts, maximum_proposals=len(modes),
                                          total_elapsed_seconds=perf_counter()-started,
                                          independent_contact_patch_rollback_allowed=False)
                            return candidate, shape, native, report
                        report['reason'] = 'Interior ripple comparison rejected the native proposal or remained unresolved'
                    else:
                        report['accepted'] = False
                        report['reason'] = native.get('reason', 'Native CAD admission rejected this proposal')
                        mapping = native.get('integration_contact_mapping') or {}
                        identifiers = tuple(sorted(set(map(int, mapping.get('retry_layout_ids', ())))))
                        if (not refit and not contact_retry_scheduled and identifiers
                                and mapping.get('retry_mapping_complete') is True):
                            # Insert immediately after this original-basis try;
                            # the shared-curve refit remains the final fallback.
                            modes.insert(1, (False, identifiers))
                            contact_retry_scheduled = True
            except MemoryError:
                raise
            except Exception as error:
                report.update(accepted=False, reason=str(error))
            attempts.append(report)
    return layouts, None, {'is_solid': False}, {
        'accepted': False, 'attempts': attempts, 'maximum_proposals': len(modes),
        'total_elapsed_seconds': perf_counter()-started,
        'independent_contact_patch_rollback_allowed': False,
        'previous_accepted_solid_retained': True,
        'reason': 'No complete surface proposal passed source, seam, native and visible-ripple checks'}
