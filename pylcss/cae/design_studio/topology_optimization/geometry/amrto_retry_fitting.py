# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE.
"""Bounded source-chart fitting with unchanged directed source coverage."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import numpy as np

from .amrto_chart_retry import refine_failed_chart_retry
from .amrto_chart_subdivision import _quad_coverage
from .amrto_shared_curves import prepare_shared_boundary_curves


@dataclass
class ChartRetryFits:
    fitted: list
    pristine: list
    tangent_edges: set
    report: dict


def fit_bounded_chart_retry(layouts, cfg, *, tangent_edges, bounds,
                            boundary_tolerance, source_fit_limit,
                            intrinsic_tolerance, report=None,
                            progress_callback=None):
    """Fit at most three times, refining only rejected original source charts.

    The source proposal precedes this function. Each failed fit may cause one
    of two further source refinements. A contact-driven proposal therefore
    has its own explicitly reported three-fit budget; it is not a relaxation
    of any source, intrinsic, material, native or STEP acceptance test.
    """
    from .amrto_reconstruction import _fit_paper_layouts

    report = report if report is not None else {}
    report.setdefault('refinement_rounds', [])
    report.setdefault('patch_partition_unchanged', True)
    report.update(maximum_fitting_attempts=3, maximum_fit_rejection_refinements=2)
    tangent_edges = set(tangent_edges)
    layouts = deepcopy(layouts)
    baseline = _quad_coverage(layouts)

    def progress(message, completed=0, total=0):
        if callable(progress_callback):
            progress_callback(message, completed, total)

    for refinement_round in range(3):
        # Seam preparation changes parameter clocks. The next proposal must
        # use the original source mapping, never a fitted sample grid.
        pristine = deepcopy(layouts)
        report['boundary_curves'] = prepare_shared_boundary_curves(
            layouts, boundary_tolerance, bounds=bounds,
            certify_stock_hull=bounds is not None,
            blocked_junction_positions=(() if cfg.couple_shared_boundary_tangents else
                {tuple(map(float, point)) for layout in layouts for point in layout.vertices}),
            tangent_preserving_edges=tangent_edges)
        for layout in layouts:
            layout.source_box_bounds = bounds
            layout.solid_fit_tolerance = boundary_tolerance
            layout.solid_source_fit_limit = source_fit_limit
        fitted = []
        for index, layout in enumerate(_fit_paper_layouts(
                layouts, cfg, intrinsic_tolerance=intrinsic_tolerance), 1):
            fitted.append(layout)
            progress(f'Refitting CAD charts {index}/{len(layouts)}', index, len(layouts))
        if ([int(layout.layout_id) for layout in fitted]
                != [int(layout.layout_id) for layout in pristine]):
            raise ValueError('Chart fitting changed the ordered source layout correspondence')
        for original, current in zip(pristine, fitted):
            for name in ('vertices', 'triangles', 'quads', 'source_vertex_indices'):
                if (hasattr(original, name) != hasattr(current, name)
                        or (hasattr(original, name)
                            and not np.array_equal(getattr(original, name), getattr(current, name)))):
                    raise ValueError(f'Chart fitting changed original source {name}')
        if _quad_coverage(fitted) != baseline:
            raise ValueError('Chart fitting changed directed original source quad coverage')
        rejected = [int(layout.layout_id) for layout in fitted
                    if not layout.fitting_report.get('source_fit_acceptance', {}).get('accepted', False)]
        report.update(
            attempts=refinement_round + 1, rejected_layout_ids=rejected,
            source_quad_coverage_exactly_once=True,
            patch_reports=[{'layout_id': int(layout.layout_id), **layout.fitting_report,
                            'multiresolution': layout.multiresolution_report} for layout in fitted])
        if not rejected or refinement_round == 2:
            break
        progress('Subdividing rejected CAD charts while retaining every source quad')
        layouts, added_edges, refinement = refine_failed_chart_retry(
            pristine, rejected, cfg, round_index=refinement_round + 1)
        rejected_ids = set(rejected)
        refinement['rejected_fit_reports'] = [
            deepcopy(row) for row in report['patch_reports'] if int(row['layout_id']) in rejected_ids]
        report['refinement_rounds'].append(refinement)
        if _quad_coverage(layouts) != baseline:
            raise ValueError('Chart refinement changed directed original source quad coverage')
        if not refinement['applied']:
            break
        tangent_edges.update(added_edges)
        if refinement['subdivision']['applied']:
            report['patch_partition_unchanged'] = False
    return ChartRetryFits(fitted, pristine, tangent_edges, report)
