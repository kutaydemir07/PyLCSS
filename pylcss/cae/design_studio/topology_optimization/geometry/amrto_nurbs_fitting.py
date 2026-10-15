# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Step 5: NURBS fitting and B-rep CAD model generation.

``fit_paper_nurbs_surface_for_layout`` implements the paper's independent
chord-length fit and subsequent control-net hierarchy/curvature classification.
The older constrained-fitting helpers remain separate for existing callers.

Reference:
Section 2.7: NURBS fitting and the generation of CAD model (step 5)
Section 3.1: Multi-resolution control strategy & Algorithms 2, 3
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Any, List, Tuple

import cadquery as cq
import numpy as np
from pylcss.process_utils import headless_subprocess_kwargs
try:
    from geomdl import BSpline, exchange, fitting, multi
except ImportError:
    BSpline = None
    exchange = None
    fitting = None
    multi = None
from OCP.BRepBuilderAPI import (
    BRepBuilderAPI_MakeEdge,
    BRepBuilderAPI_MakeFace,
    BRepBuilderAPI_MakeSolid,
    BRepBuilderAPI_MakeWire,
    BRepBuilderAPI_Sewing,
)
from OCP.Geom import Geom_BSplineSurface
from OCP.IGESControl import IGESControl_Writer
from OCP.ShapeFix import ShapeFix_Shell, ShapeFix_Solid
from OCP.TColStd import TColStd_Array1OfInteger, TColStd_Array1OfReal
from OCP.TColgp import TColgp_Array2OfPnt
from OCP.gp import gp_Pnt

from .amrto_common import QuadLayout, get_json2on_executable
from .occ_shapes import _solid_volume, _unify_same_domain_shape

logger = logging.getLogger(__name__)


def fit_paper_nurbs_surface_for_layout(
    layout: QuadLayout,
    degree_u: int = 3,
    degree_v: int = 3,
    k_type: float = 0.5,
    min_ctrlpts: int = 6,
    curvature_ck: float = 0.1,
    curvature_cp: float = 10.0,
    multires_enabled: bool = True,
    compression_ratio: float = 0.5,
    multires_layers: int = 1,
    maximum_knot_removal_attempts: int = 64,
    join_boundaries: bool = False,
    maximum_multires_displacement: float | None = None,
    maximum_source_deviation: float | None = None,
) -> Any:
    """AMRTO step 5: one chord-length fit followed by Algorithms 2 and 3.

    Equations (16)--(18) use unit rational weights, as in the authors'
    ``PYTOCAD_simple.py``. Each sampled chart is fitted independently with
    fixed control counts. The application's optional ``join_boundaries``
    extension constrains common CAD seams in that single least-squares solve.
    Supplying ``maximum_source_deviation`` enables a separately reported
    engineering extension: exact boundary-clock alignment and one bounded
    fixed-basis signed-area constrained optimization within this fitting
    stage. It does not add control points or repeat the global fitting stage.

    The paper leaves the precise initial control-count rule unspecified. Here
    ``k_type`` is the fraction of step 4 samples in each direction, subject to
    the requested minimum and the least-squares data rank. This choice is
    reported, rather than presented as an additional equation of the paper.
    """
    from .amrto_multiresolution import apply_paper_multiresolution_strategy

    if fitting is None:
        raise RuntimeError("NURBS surface fitting requires the 'geomdl' package")
    if layout.sampled_grid_3d is None:
        raise ValueError(f"Layout {layout.layout_id} has no sampling points")
    if not np.isfinite(k_type) or not 0 < k_type <= 1:
        raise ValueError("The initial control-count fraction must be in (0, 1]")
    n_u, n_v = layout.sampled_shape
    if n_u <= degree_u or n_v <= degree_v:
        raise ValueError("The sampling grid must exceed the spline degree in each direction")
    points = np.asarray(layout.sampled_grid_3d, dtype=float).reshape(n_u, n_v, 3)
    if not np.isfinite(points).all():
        raise ValueError("The sampling grid contains nonfinite points")
    # Chord-length parameterization is undefined on an entirely collapsed
    # data row or column. Reject malformed data without perturbing geometry.
    if any((np.linalg.norm(np.diff(points, axis=axis), axis=2).sum(axis=axis) <= 0).any()
           for axis in (0, 1)):
        raise ValueError("The sampling grid contains a collapsed row or column")
    counts = [min(count, max(degree + 1, min_ctrlpts, round(k_type * count)))
              for count, degree in ((n_u, degree_u), (n_v, degree_v))]
    source_guard = maximum_source_deviation is not None
    source_fit_report = None
    boundary_parameter_report = None
    target_transport_report = None
    fitting_layout = layout
    source_box_bounds = getattr(layout, 'source_box_bounds', None)
    if source_guard:
        if (not join_boundaries or not np.isfinite(maximum_source_deviation)
                or maximum_source_deviation <= 0):
            raise ValueError('Source-constrained solid fitting requires a finite positive source-deviation limit')
        from .amrto_boundary_parameters import align_layout_boundary_parameters
        from .amrto_fitting_targets import harmonic_fitting_targets
        from .amrto_parameterization import sample_parameter_grid
        try:
            boundary_parameter_report = align_layout_boundary_parameters(layout)
        except ValueError as error:
            # Alignment is optional and atomic. A proposed clock map that
            # would flip an actual triangle leaves the original chart intact;
            # the same one-fit source admission below still remains mandatory.
            boundary_parameter_report = {
                'accepted': False, 'engineering_parameterization_extension': True,
                'rejection_reason': str(error), 'output': 'Original source UV chart retained',
                'source_XYZ_unchanged': True, 'source_UV_unchanged': True,
                'physical_boundary_geometry_unchanged': True, 'surface_fits': 0,
            }
        # The XYZ mesh is unchanged, but samples evaluated in the old UV
        # chart are stale. Keep the exact Stage4 sample counts and refresh
        # their source inverse-map values in the new chart before this fit.
        points = sample_parameter_grid(layout, np.linspace(0, 1, n_u), np.linspace(0, 1, n_v))
        layout.sampled_grid_3d = points.reshape(-1, 3)
        transport_budget = (maximum_multires_displacement if maximum_multires_displacement is not None
                            else getattr(layout, 'solid_fit_tolerance', maximum_source_deviation))
        try:
            fitting_layout, target_transport_report = harmonic_fitting_targets(
                layout, maximum_displacement=transport_budget)
        except ValueError as error:
            # Keep a complete reviewable fit, but never advertise a rejected
            # target-compatibility proposal as an accepted solid input.
            target_transport_report = {
                'accepted': False, 'engineering_extension': True,
                'rejection_reason': str(error), 'public_source_XYZ_unchanged': True,
                'output': 'Original source targets retained for the reviewable rejected surface',
            }
    if join_boundaries:
        if not getattr(layout, 'boundary_splines', None):
            raise ValueError('Solid CAD fitting requires prepared common boundary curves')
        # One least-squares solve per chart. This does not call the legacy
        # fit/refinement loop. Matching curves implement the CAD stitching
        # integration needed by the application's volumetric consumers.
        surface = _fit_with_exact_polyline_boundaries(
            fitting_layout, degree_u, degree_v, counts[0], counts[1], dense_fitting=True)
    else:
        surface = fitting.approximate_surface(
            points.reshape(-1, 3).tolist(), size_u=n_u, size_v=n_v,
            degree_u=degree_u, degree_v=degree_v, centripetal=False,
            ctrlpts_size_u=counts[0], ctrlpts_size_v=counts[1])
    if source_guard:
        from .amrto_fit_orientation import assess_fit_orientation
        from .amrto_fit_repair import repair_surface_orientation
        from .amrto_shared_stock import control_hull_in_box
        def stock_admission(candidate):
            if source_box_bounds is None:
                return {'accepted': True, 'assessed': False, 'reason': 'No source stock box was supplied'}
            weights = getattr(candidate, 'weights', None)
            if weights is not None and (not np.isfinite(weights).all() or np.any(np.asarray(weights) <= 0)):
                return {'accepted': False, 'assessed': True, 'reason': 'Stock hull requires positive finite weights'}
            poles = np.asarray(candidate.ctrlpts).reshape(candidate.ctrlpts_size_u, candidate.ctrlpts_size_v, 3)
            return {'assessed': True, **control_hull_in_box(poles, source_box_bounds)}
        orientation = assess_fit_orientation(layout, surface)
        source_error = _surface_map_residual(layout, surface, grid_size=73)
        stock = stock_admission(surface)
        source_fit_report = {
            'method': 'One initial LS solve and at most one bounded fixed-basis signed-area constrained optimization',
            'engineering_extension': True, 'literal_paper_algorithm': False,
            'maximum_source_deviation': float(maximum_source_deviation),
            'initial_source_map_residual': source_error, 'initial_source_orientation': orientation,
            'initial_control_shape': [surface.ctrlpts_size_u, surface.ctrlpts_size_v],
            'optimization_calls': 0, 'control_points_added': 0, 'repartition_loops': 0,
            'fitting_targets': 'Separate bounded harmonic copy; public source XYZ unchanged',
            'initial_stock_hull': stock,
        }
        if not orientation['accepted'] or source_error > maximum_source_deviation or not stock['accepted']:
            optimized, optimization = repair_surface_orientation(
                fitting_layout, surface, reference_layout=layout, box_bounds=source_box_bounds,
                position_limit=maximum_source_deviation,
                maximum_seconds=45., maximum_iterations=8, maximum_control_points=4096,
                maximum_training_sites=100_000, maximum_constraint_sites=50_000,
                maximum_verification_sites=150_000)
            source_fit_report.update(optimization_calls=1, constrained_optimization=optimization)
            if optimized is not None:
                surface = optimized
                orientation = assess_fit_orientation(layout, surface)
                source_error = _surface_map_residual(layout, surface, grid_size=73)
                stock = stock_admission(surface)
        source_fit_report.update(
            accepted=bool(orientation['accepted'] and source_error <= maximum_source_deviation
                          and stock['accepted'] and target_transport_report['accepted']),
            source_map_residual=source_error, source_orientation=orientation,
            stock_hull=stock,
            output_control_shape=[surface.ctrlpts_size_u, surface.ctrlpts_size_v])
    surface.delta = .05
    layout.fitting_report = {
        "method": "AMRTO Section 2.7: independent chord-length least-squares approximation",
        "paper_equations": [16, 17, 18], "rational_weights": "unit",
        "parameterization": "chord_length", "preserve_boundary": bool(join_boundaries),
        "solid_boundary_join_extension": bool(join_boundaries),
        "initial_control_shape": counts, "sample_shape": [n_u, n_v],
        "initial_control_count_policy": "Sample fraction per axis, bounded by minimum and data count",
        "initial_control_count_fraction": k_type, "initial_fitting_batches": 1,
        "paper_multiresolution": bool(multires_enabled),
    }
    if source_guard:
        layout.fitting_report.update(boundary_parameterization=boundary_parameter_report,
                                     fitting_target_transport=target_transport_report,
                                     constrained_source_fit=source_fit_report,
                                     maximum_source_deviation=float(maximum_source_deviation))
    if join_boundaries:
        layout.fitting_report['method'] = 'Single least-squares fit with common G0 CAD stitching curves'
        layout.fitting_report['parameterization'] = 'Shared-boundary source UV basis'
    if multires_enabled:
        strategy = apply_paper_multiresolution_strategy
        extension = {}
        if join_boundaries:
            from .amrto_boundary_join import apply_joined_multiresolution
            strategy = apply_joined_multiresolution
            extension['maximum_displacement'] = maximum_multires_displacement
            if source_guard:
                def source_admission(candidate):
                    assessed = assess_fit_orientation(layout, candidate)
                    error = _surface_map_residual(layout, candidate, grid_size=73)
                    stock = stock_admission(candidate)
                    return {**assessed, 'orientation_accepted': assessed['accepted'],
                            'source_map_residual': error,
                            'stock_hull': stock,
                            'maximum_source_deviation': float(maximum_source_deviation),
                            'accepted': bool(assessed['accepted'] and error <= maximum_source_deviation
                                             and stock['accepted'] and target_transport_report['accepted'])}
                extension['orientation_evaluator'] = source_admission
        result = strategy(
            surface, curvature_ck=curvature_ck, curvature_cp=curvature_cp,
            compression_ratio=compression_ratio, layers=multires_layers,
            maximum_knot_removal_attempts=maximum_knot_removal_attempts, **extension)
        surface = result.candidate
        layout.principal_curvature = result.report["principal_curvature"]
        layout.multiresolution_report = {"enabled": True, "paper_algorithm_executed": True,
                                       **result.report}
    else:
        layout.multiresolution_report = {"enabled": False, "paper_algorithm_executed": False}
    if join_boundaries:
        # Sewing locates real T-junctions geometrically. Spline basis knots
        # are not CAD edge endpoints; splitting at all of them creates tiny
        # trim edges and unstable local pcurves without changing the surface.
        surface.segment_boundary_edges = False
        layout.fitting_report['cad_boundary_topology'] = 'Natural surface edges; sewing resolves shared T-junctions'
    if source_guard:
        # Optional MRA must not turn an accepted initial fit into an invalid
        # source map. Retain the complete surface for review if it fails;
        # volumetric callers must honor this explicit final acceptance flag.
        final_orientation = assess_fit_orientation(layout, surface)
        final_error = _surface_map_residual(layout, surface, grid_size=73)
        final_stock = stock_admission(surface)
        layout.fitting_report.update(
            source_fit_acceptance={
                'accepted': bool(final_orientation['accepted'] and final_error <= maximum_source_deviation
                                 and final_stock['accepted'] and target_transport_report['accepted']),
                'maximum_deviation_mm': float(maximum_source_deviation),
                'measured_deviation_mm': final_error,
                'surface_orientation': final_orientation,
                'stock_hull': final_stock,
                'harmonic_fitting_target_transport_accepted': bool(target_transport_report['accepted']),
                'engineering_constrained_fit_extension': True,
                'certified_global_injectivity': False,
            },
            maximum_source_map_residual=final_error, source_orientation=final_orientation)
    layout.nurbs_surface = surface
    layout.fitting_report.update(ctrlpts_u=surface.ctrlpts_size_u,
                                 ctrlpts_v=surface.ctrlpts_size_v,
                                 multiresolution=layout.multiresolution_report)
    return surface


def fit_nurbs_surface_for_layout(
    layout: QuadLayout,
    degree_u: int = 3,
    degree_v: int = 3,
    k_type: float = 0.5,
    min_ctrlpts: int = 6,
    curvature_ck: float = 0.1,
    curvature_cp: float = 10.0,
    preserve_boundary: bool = False,
    fit_tolerance: float = 0.0,
    max_refinement_controls: int = 64,
    multires_enabled: bool = True,
    compression_ratio: float = 0.5,
    multires_layers: int = 1,
    multires_regeneration: str = "author_uniform",
    maximum_knot_removal_attempts: int = 64,
    orientation_repair_seconds: float = 60.0,
    orientation_repair_iterations: int = 8,
    curvature_reference_length: float = 1.0,
) -> Any:
    """Fit a bicubic B-spline surface to the layout's 3D sampling grid (Eq. 16, 17)."""
    if fitting is None:
        raise RuntimeError(
            "NURBS surface fitting requires the 'geomdl' package. "
            "Please install it with: pip install geomdl"
        )
    if layout.sampled_grid_3d is None:
        raise ValueError(f"Layout {layout.layout_id} has no sampling points.")

    n_u, n_v = layout.sampled_shape
    if not np.isfinite(curvature_reference_length) or curvature_reference_length <= 0:
        raise ValueError('Curvature reference length must be finite and positive')
    layout.curvature_reference_length = curvature_reference_length
    pts_3d = layout.sampled_grid_3d

    if n_u <= degree_u or n_v <= degree_v:
        raise ValueError("The sampling grid must have more points than the spline degree in each direction")
    grid_3d = np.asarray(pts_3d, dtype=float).reshape(n_u, n_v, 3)
    if not np.isfinite(grid_3d).all():
        raise ValueError("The sampling grid contains nonfinite points")
    curvature = _principal_curvature(grid_3d)
    layout.principal_curvature = curvature
    # This mesh estimate is diagnostic. Algorithm 3 classifies the fitted
    # surface using its exact derivatives, after the initial fit is complete.
    ratio = k_type
    num_ctrl_u = min(n_u, max(degree_u + 1, min_ctrlpts, int(round(ratio * n_u))))
    num_ctrl_v = min(n_v, max(degree_v + 1, min_ctrlpts, int(round(ratio * n_v))))
    if preserve_boundary:
        refinement = []
        fitted_bases, skipped_targets, sample_cache = set(), [], {}
        best_surface, best_error, best_knots, best_refinement = None, float("inf"), ([], []), None
        local_knots_u, local_knots_v = [], []
        # Length-based sampling can reach 96 nodes even for a nearly planar
        # patch. Do not immediately allocate a 48-by-48 control net when a
        # compact net passes independent accuracy checks. The requested ratio
        # remains a refinement target; fixed boundary knots are always retained.
        initial_u = min(num_ctrl_u, max(min_ctrlpts, 12)) if fit_tolerance > 0 else num_ctrl_u
        initial_v = min(num_ctrl_v, max(min_ctrlpts, 12)) if fit_tolerance > 0 else num_ctrl_v
        targets = [(initial_u, initial_v)]
        if fit_tolerance > 0:
            targets.extend((max(initial_u, count), max(initial_v, count))
                           for count in (10, 16, 24, 36, 48, max_refinement_controls)
                           if count <= max_refinement_controls)
            targets.append((num_ctrl_u, num_ctrl_v))
            targets.sort(key=lambda sizes: (sizes[0] * sizes[1], sizes))
        for target_u, target_v in dict.fromkeys(targets):
            used_local_knots = (list(local_knots_u), list(local_knots_v))
            surf = _fit_with_exact_polyline_boundaries(layout, degree_u, degree_v, target_u, target_v,
                                                      dense_fitting=fit_tolerance > 0,
                                                      local_knots=(local_knots_u, local_knots_v),
                                                      fitted_bases=fitted_bases, sample_cache=sample_cache)
            if surf is None:
                skipped_targets.append([target_u, target_v])
                continue
            if fit_tolerance <= 0:
                break
            error, error_uv, site_errors = _surface_map_validation(
                layout, surf, grid_size=35, sample_cache=sample_cache)
            refinement.append({"ctrlpts_u": surf.ctrlpts_size_u, "ctrlpts_v": surf.ctrlpts_size_v,
                               "maximum_dense_map_residual": error})
            if error < best_error:
                best_surface, best_error, best_knots = surf, error, used_local_knots
                best_refinement = len(refinement) - 1
            if error <= 3 * fit_tolerance:
                break
            # Add resolution where the harmonic map compresses a feature.
            # Raising a uniform control count alone misses narrow UV strips.
            if layout.parameter_coords is not None:
                # A patch can match every mesh vertex while missing the
                # inverse map between vertices. Refine the failing validation
                # sites too, using the same independent residual check.
                worst = np.argsort(site_errors)[::-1]
                for axis, knots in ((0, local_knots_u), (1, local_knots_v)):
                    added = 0
                    for index in worst:
                        value = float(round(error_uv[index, axis], 8))
                        if site_errors[index] <= 3 * fit_tolerance:
                            break
                        if 1e-6 < value < 1 - 1e-6 and all(abs(value - knot) > 1e-5 for knot in knots):
                            knots.append(value)
                            added += 1
                            if added == 6:
                                break
        # A larger least-squares basis need not improve the independent maximum
        # residual. Keep the best tested fit, including its actual local knots.
        if best_surface is not None:
            surf, used_local_knots = best_surface, best_knots
        surf.delta = 0.05
        layout.nurbs_surface = surf
        layout.fitting_report = {
            "method": "Least squares of inverse-map samples and original mesh vertices with shared boundaries",
            "preserve_boundary": True, "principal_curvature": curvature,
            "paper_multiresolution": False,
            "requested_ratio_control_counts": [num_ctrl_u, num_ctrl_v],
            "adaptive_initial_control_counts": [initial_u, initial_v],
            "ctrlpts_u": surf.ctrlpts_size_u, "ctrlpts_v": surf.ctrlpts_size_v,
            "boundary_method": "Shared cubic curves" if hasattr(layout, "boundary_splines") else "Shared polylines",
            "boundary_basis": surf.boundary_basis_report,
            "adaptive_refinements": refinement, "requested_map_residual": 3 * fit_tolerance,
            "selected_refinement_index": best_refinement,
            "duplicate_basis_targets_skipped": skipped_targets,
            "maximum_dense_map_residual": best_error if refinement else 0.0,
            "localized_knots_u": used_local_knots[0], "localized_knots_v": used_local_knots[1],
            "dense_map_residual_satisfied": not refinement or best_error <= 3 * fit_tolerance,
        }
        return _finalize_multiresolution_fit(layout, surf, multires_enabled=multires_enabled,
            curvature_ck=curvature_ck, curvature_cp=curvature_cp, compression_ratio=compression_ratio,
            layers=multires_layers, regeneration=multires_regeneration, fit_tolerance=fit_tolerance,
            preserve_boundary=preserve_boundary, maximum_knot_removal_attempts=maximum_knot_removal_attempts,
            orientation_repair_seconds=orientation_repair_seconds, orientation_repair_iterations=orientation_repair_iterations,
            residual_evaluator=lambda source, candidate: _surface_map_residual(source, candidate, sample_cache=sample_cache))

    # Degenerate data indicates a failed map; changing the geometry by epsilon
    # perturbations would conceal that failure.
    if (np.linalg.norm(np.diff(grid_3d, axis=0), axis=2).sum(axis=0) <= 0).any() or \
       (np.linalg.norm(np.diff(grid_3d, axis=1), axis=2).sum(axis=1) <= 0).any():
        raise ValueError("The sampling grid contains a collapsed row or column")
    grid_reg = grid_3d

    # Least squares approximation fitting
    surf = fitting.approximate_surface(
        grid_reg.reshape(-1, 3).tolist(),
        size_u=n_u,
        size_v=n_v,
        degree_u=degree_u,
        degree_v=degree_v,
        centripetal=False,
        ctrlpts_size_u=num_ctrl_u,
        ctrlpts_size_v=num_ctrl_v,
    )
    surf.delta = 0.05
    layout.nurbs_surface = surf
    layout.fitting_report = {"method": "geomdl least-squares approximation",
                             "preserve_boundary": False, "paper_multiresolution": False,
                             "principal_curvature": curvature}
    return _finalize_multiresolution_fit(layout, surf, multires_enabled=multires_enabled,
        curvature_ck=curvature_ck, curvature_cp=curvature_cp, compression_ratio=compression_ratio,
        layers=multires_layers, regeneration=multires_regeneration, fit_tolerance=fit_tolerance,
        preserve_boundary=preserve_boundary, maximum_knot_removal_attempts=maximum_knot_removal_attempts,
        orientation_repair_seconds=orientation_repair_seconds, orientation_repair_iterations=orientation_repair_iterations)


def _finalize_multiresolution_fit(layout, surface, *, multires_enabled, curvature_ck, curvature_cp,
                                  compression_ratio, layers, regeneration, fit_tolerance,
                                  preserve_boundary, maximum_knot_removal_attempts,
                                  orientation_repair_seconds=60.0, orientation_repair_iterations=8,
                                  residual_evaluator=None):
    from .amrto_fit_validation import apply_validated_multiresolution
    from .amrto_fit_orientation import assess_fit_orientation

    if preserve_boundary and layout.parameter_coords is not None:
        try:
            orientation = assess_fit_orientation(layout, surface)
        except (ValueError, RuntimeError) as error:
            orientation = {"accepted": False, "layout_id": int(layout.layout_id),
                           "method": "Exact spline derivative/source UV orientation check",
                           "error": f"{type(error).__name__}: {error}"}
        layout.fitting_report["surface_orientation"] = orientation
        if (not orientation["accepted"] and fit_tolerance > 0
                and orientation_repair_seconds > 0 and orientation_repair_iterations > 0):
            from .amrto_fit_repair import repair_surface_orientation
            logger.info("AMRTO patch %s: repairing folded interior with fixed shared borders", layout.layout_id)
            repaired, repair = repair_surface_orientation(layout, surface, position_limit=3 * fit_tolerance,
                maximum_seconds=orientation_repair_seconds, maximum_iterations=orientation_repair_iterations)
            layout.fitting_report["interior_orientation_repair"] = repair
            if repaired is not None:
                surface = repaired
                layout.nurbs_surface = repaired
                layout.fitting_report.update(
                    maximum_dense_map_residual=repair["final_position_maximum_error"],
                    dense_map_residual_satisfied=repair["final_position_maximum_error"] <= 3 * fit_tolerance,
                    surface_orientation=_combined_orientation_checks(repair["iterations"][-1]["orientation_checks"]))
            logger.info("AMRTO patch %s interior repair: %s", layout.layout_id, repair.get("status", "rejected"))

    if not multires_enabled:
        layout.multiresolution_report = {"enabled": False, "paper_algorithm_executed": False}
        return surface
    candidate = apply_validated_multiresolution(layout, surface, curvature_ck=curvature_ck,
        curvature_reference_length=getattr(layout, 'curvature_reference_length', 1.),
        curvature_cp=curvature_cp, compression_ratio=compression_ratio, layers=layers,
        regeneration=regeneration, fit_tolerance=fit_tolerance, preserve_boundary=preserve_boundary,
        residual_evaluator=residual_evaluator or _surface_map_residual,
        orientation_evaluator=_validate_final_orientation if preserve_boundary and layout.parameter_coords is not None else None,
        original_residual=layout.fitting_report.get("maximum_dense_map_residual"),
        original_orientation=layout.fitting_report.get("surface_orientation"),
        maximum_knot_removal_attempts=maximum_knot_removal_attempts)
    # Removing a knot can preserve geometry while deleting a shared edge split.
    # Keep the original segmentation independently of the compressed basis.
    for axis in ("u", "v"):
        setattr(candidate, f"boundary_split_parameters_{axis}",
                np.unique(getattr(surface, f"knotvector_{axis}")).tolist())
    layout.nurbs_surface = candidate
    report = layout.multiresolution_report
    if candidate is not surface and report.get("candidate_source_orientation") is not None:
        layout.fitting_report["surface_orientation"] = report["candidate_source_orientation"]
    layout.fitting_report.update(paper_multiresolution=not bool(report.get("error")),
        multiresolution=report, ctrlpts_u=candidate.ctrlpts_size_u, ctrlpts_v=candidate.ctrlpts_size_v)
    if report.get("accepted") and report.get("candidate_dense_map_residual") is not None:
        layout.fitting_report["maximum_dense_map_residual"] = report["candidate_dense_map_residual"]
        layout.fitting_report["dense_map_residual_satisfied"] = report["candidate_dense_map_residual"] <= 3 * fit_tolerance
    return candidate


def _combined_orientation_checks(checks):
    reports = [check["report"] for check in checks]
    worst = min(reports, key=lambda report: (report["accepted"], report.get("minimum_source_normal_dot")
                                            if report.get("minimum_source_normal_dot") is not None else -1.))
    return {**worst, "accepted": all(report["accepted"] for report in reports),
            "method": "Exact source orientation at 129 uniform nodes plus knot sites and five one-sided offsets",
            "sample_count": sum(report["sample_count"] for report in reports),
            "inverted_site_count": sum(report["inverted_site_count"] for report in reports),
            "degenerate_site_count": sum(report["degenerate_site_count"] for report in reports),
            "verification_checks": checks}


def _validate_final_orientation(layout, surface):
    from .amrto_fit_orientation import assess_fit_orientation
    if not layout.fitting_report.get("interior_orientation_repair", {}).get("accepted"):
        return assess_fit_orientation(layout, surface)
    return _combined_orientation_checks([
        {"uniform_samples": 129, "knot_probe_fraction": fraction,
         "report": assess_fit_orientation(layout, surface, uniform_samples=129,
                                           knot_probe_fraction=fraction, maximum_sites=150_000)}
        for fraction in (1e-6, 1e-4, .005, .01, .02)])



def _principal_curvature(grid: np.ndarray) -> float:
    """Estimate maximum absolute principal curvature using both fundamental forms."""
    nu, nv, _ = grid.shape
    order = 2 if min(nu, nv) >= 3 else 1
    du, dv = np.gradient(grid, 1 / (nu - 1), 1 / (nv - 1), axis=(0, 1), edge_order=order)
    duu = np.gradient(du, 1 / (nu - 1), axis=0, edge_order=order)
    dvv = np.gradient(dv, 1 / (nv - 1), axis=1, edge_order=order)
    duv = np.gradient(du, 1 / (nv - 1), axis=1, edge_order=order)
    cross = np.cross(du, dv)
    magnitude = np.linalg.norm(cross, axis=2)
    normal = np.divide(cross, magnitude[..., None], out=np.zeros_like(cross), where=magnitude[..., None] > 0)
    e = np.einsum("ijk,ijk->ij", du, du)
    f = np.einsum("ijk,ijk->ij", du, dv)
    g = np.einsum("ijk,ijk->ij", dv, dv)
    second_u = np.einsum("ijk,ijk->ij", duu, normal)
    m = np.einsum("ijk,ijk->ij", duv, normal)
    n = np.einsum("ijk,ijk->ij", dvv, normal)
    determinant = e * g - f * f
    valid = determinant > np.finfo(float).eps * np.maximum(e * g, np.finfo(float).tiny)
    if not valid.any():
        raise ValueError("Cannot fit a surface with a collapsed tangent plane")
    mean = (e[valid] * n[valid] + g[valid] * second_u[valid] - 2 * f[valid] * m[valid]) / (2 * determinant[valid])
    gaussian = (second_u[valid] * n[valid] - m[valid] ** 2) / determinant[valid]
    separation = np.sqrt(np.maximum(mean ** 2 - gaussian, 0))
    return float(np.max(np.maximum(np.abs(mean + separation), np.abs(mean - separation))))


def _remove_collinear_boundary_vertices(points: np.ndarray) -> np.ndarray:
    """Remove redundant straight-segment samples only within floating-point noise."""
    if len(points) < 3:
        return points
    tolerance = 128 * np.finfo(float).eps * max(float(np.max(np.ptp(points, axis=0))), 1.0)
    kept = [points[0]]
    for index in range(1, len(points) - 1):
        start, middle, end = kept[-1], points[index], points[index + 1]
        direction = end - start
        squared = float(np.dot(direction, direction))
        if squared > 0:
            parameter = float(np.dot(middle - start, direction) / squared)
            distance = float(np.linalg.norm(middle - (start + parameter * direction)))
            if 0 < parameter < 1 and distance <= tolerance:
                continue
        kept.append(middle)
    kept.append(points[-1])
    return np.asarray(kept)


def prepare_shared_fit_boundaries(layouts: List[QuadLayout], tolerance: float) -> dict:
    """Simplify common boundary chains once, keeping every patch junction fixed.

    RDP bounds each retained polyline segment's deviation from its source chain.
    Shared chains use identical points in both directions, including T-junctions.
    Harmonic parameterization retains the original boundary; only fitting uses
    these common curves. This is a controlled approximation, not knot removal.
    """
    if tolerance <= 0:
        return {"tolerance": 0.0, "removed_boundary_vertices": 0}
    anchors = set()
    adjacency = {}
    edge_owners = {}
    def key(point):
        return tuple(float(x) for x in point)
    for owner, layout in enumerate(layouts):
        for path in layout.boundary_paths:
            anchors.add(key(layout.vertices[path[0]]))
            anchors.add(key(layout.vertices[path[-1]]))
            for first, second in zip(path, path[1:]):
                a, b = key(layout.vertices[first]), key(layout.vertices[second])
                adjacency.setdefault(a, set()).add(b)
                adjacency.setdefault(b, set()).add(a)
                edge_owners.setdefault(tuple(sorted((a, b))), set()).add(owner)
    # A junction need not be a square-map corner: repaired patches can cross
    # it inside all their boundary paths. Splitting only at path endpoints
    # lets each incident patch replace a different chain by a different chord.
    for vertex, neighbors in adjacency.items():
        owners = {frozenset(edge_owners[tuple(sorted((vertex, neighbor)))])
                  for neighbor in neighbors}
        if len(neighbors) != 2 or len(owners) != 1:
            anchors.add(vertex)
    cache = {}
    removed = 0
    def simplify(points):
        first, last = key(points[0]), key(points[-1])
        reversed_order = last < first
        canonical = points[::-1] if reversed_order else points
        identifier = tuple(map(key, canonical))
        if identifier not in cache:
            keep = {0, len(canonical) - 1}
            pending = [(0, len(canonical) - 1)]
            while pending:
                left, right = pending.pop()
                if right - left < 2:
                    continue
                start, end = canonical[left], canonical[right]
                direction = end - start
                squared = np.dot(direction, direction)
                interior = canonical[left + 1:right]
                parameters = np.clip((interior - start) @ direction / squared, 0, 1) if squared > 0 else np.zeros(len(interior))
                distances = np.linalg.norm(interior - (start + parameters[:, None] * direction), axis=1)
                offset = int(np.argmax(distances))
                if distances[offset] > tolerance:
                    middle = left + offset + 1
                    keep.add(middle)
                    pending.extend(((left, middle), (middle, right)))
            cache[identifier] = sorted(keep)
        indices = cache[identifier]
        return sorted(len(points) - 1 - i for i in indices) if reversed_order else indices
    for layout in layouts:
        simplified = []
        for path in layout.boundary_paths:
            cuts = [i for i, vertex in enumerate(path) if key(layout.vertices[vertex]) in anchors]
            retained = []
            for left, right in zip(cuts, cuts[1:]):
                local = simplify(layout.vertices[path[left:right + 1]])
                segment = [path[left + i] for i in local]
                retained.extend(segment[:-1])
            retained.append(path[-1])
            removed += len(path) - len(retained)
            simplified.append(retained)
        layout.fit_boundary_paths = simplified
    return {"tolerance": float(tolerance), "removed_boundary_vertices": removed,
            "method": "Shared-chain polyline approximation with fixed patch junctions",
            "paper_multiresolution": False}


def _bounded_interior_fit(initial, multiply_normal, target, bounds):
    """Minimize the existing fitting objective inside the physical CAD box."""
    from scipy.optimize import minimize

    shape = initial.shape
    lower = np.broadcast_to(bounds[0], shape).reshape(-1)
    upper = np.broadcast_to(bounds[1], shape).reshape(-1)

    def objective(values):
        controls = values.reshape(shape)
        normal_controls = multiply_normal(controls)
        value = .5 * np.sum(controls * normal_controls) - np.sum(target * controls)
        return float(value), (normal_controls - target).reshape(-1)

    solution = minimize(objective, np.clip(initial.reshape(-1), lower, upper),
        jac=True, method="L-BFGS-B", bounds=list(zip(lower, upper)),
        options={"maxiter": 200, "ftol": 1e-12, "gtol": 1e-8})
    return np.clip(solution.x, lower, upper).reshape(shape)


def _complete_shared_curve_basis(first, second, target):
    """Keep every seam knot and distribute additional controls across [0, 1]."""
    multiplicities = {}
    for curve in (first, second):
        for value, count in zip(*np.unique(curve.t, return_counts=True)):
            value = float(value)
            multiplicities[value] = max(int(count), multiplicities.get(value, 0))
    multiplicities[0.0] = multiplicities[1.0] = 4
    knots = np.concatenate([np.repeat(value, count) for value, count in sorted(multiplicities.items())])
    needed = max(0, int(target) - (len(knots) - 4))
    if not needed:
        return knots
    # Selecting the first N members of a denser grid puts every free knot in
    # the first half of a chart. A steep feature near u=1 then has no support.
    additional = [value for value in np.linspace(0, 1, needed + 2)[1:-1]
                  if np.min(np.abs(knots - value)) > 1e-8]
    while len(additional) < needed:
        breaks = np.unique(np.r_[knots, additional])
        index = int(np.argmax(np.diff(breaks)))
        value = .5 * (breaks[index] + breaks[index + 1])
        if min(value - breaks[index], breaks[index + 1] - value) <= 1e-8:
            raise ValueError('The requested shared boundary basis has no distinct interior knot sites')
        additional.append(value)
    return np.sort(np.r_[knots, additional])


def _fit_with_exact_polyline_boundaries(layout: QuadLayout, degree_u: int, degree_v: int,
                                        target_u: int, target_v: int, dense_fitting: bool = False,
                                        local_knots: tuple = ((), ()), *, fitted_bases=None,
                                        sample_cache=None) -> Any:
    """Constrain each edge to the stage-3 polyline, including all its kinks.

    Repeated internal knots retain C0 continuity on boundary kinks. Opposing
    edges share a union knot vector; differing patch sampling densities and
    T-junctions therefore cannot move the common physical boundary.
    """
    from scipy.interpolate import BSpline as ScipyBSpline
    from scipy.linalg import lstsq

    boundary_paths = getattr(layout, "fit_boundary_paths", layout.boundary_paths)
    paths = [boundary_paths[0], boundary_paths[1],
             list(reversed(boundary_paths[2])), list(reversed(boundary_paths[3]))]
    # Shared fitting paths retain every junction, even on a straight segment.
    # These knots later split the CAD boundary into matching topological edges.
    points = [layout.vertices[path] if hasattr(layout, "fit_boundary_paths") else
              _remove_collinear_boundary_vertices(layout.vertices[path]) for path in paths]
    parameters = []
    for polyline in points:
        lengths = np.linalg.norm(np.diff(polyline, axis=0), axis=1)
        if not len(lengths) or (lengths <= 0).any():
            raise ValueError("Exact spline boundaries require nonzero polyline segments")
        cumulative = np.r_[0.0, np.cumsum(lengths)]
        parameters.append(cumulative / cumulative[-1])

    shared_curves = getattr(layout, "boundary_splines", None)
    knot_reconciliation_reports = []
    if shared_curves is not None:
        shared_curves = [shared_curves[0], shared_curves[1],
                         ScipyBSpline(1 - shared_curves[2].t[::-1], shared_curves[2].c[::-1], 3),
                         ScipyBSpline(1 - shared_curves[3].t[::-1], shared_curves[3].c[::-1], 3)]
        from .amrto_shared_curves import reconcile_roundoff_boundary_knots
        for first, second in ((0, 2), (1, 3)):
            reconciled, report = reconcile_roundoff_boundary_knots([shared_curves[first], shared_curves[second]])
            shared_curves[first], shared_curves[second] = reconciled
            knot_reconciliation_reports.append(report)

    def knots_for(first: np.ndarray, second: np.ndarray, degree: int, target: int) -> np.ndarray:
        breaks = np.unique(np.r_[first[1:-1], second[1:-1]])
        knots = np.r_[np.zeros(degree + 1), np.repeat(breaks, degree), np.ones(degree + 1)]
        count = len(knots) - degree - 1
        if count < target:
            additional = np.linspace(0, 1, target - count + 2)[1:-1]
            knots = np.sort(np.r_[knots, additional])
        return knots

    ku = knots_for(parameters[0], parameters[2], degree_u, target_u)
    kv = knots_for(parameters[1], parameters[3], degree_v, target_v)
    if shared_curves is not None:
        ku = _complete_shared_curve_basis(shared_curves[0], shared_curves[2], target_u)
        kv = _complete_shared_curve_basis(shared_curves[1], shared_curves[3], target_v)
    for axis, knots in enumerate((ku, kv)):
        additions = []
        for value in local_knots[axis]:
            if np.min(np.abs(np.r_[knots, additions] - value)) > 1e-8:
                additions.append(value)
        if axis == 0:
            ku = np.sort(np.r_[knots, additions])
        else:
            kv = np.sort(np.r_[knots, additions])
    count_u, count_v = len(ku) - degree_u - 1, len(kv) - degree_v - 1
    if fitted_bases is not None:
        signature = (ku.tobytes(), kv.tobytes())
        if signature in fitted_bases:
            return None
        fitted_bases.add(signature)
    greville_u = np.array([np.mean(ku[i + 1:i + degree_u + 1]) for i in range(count_u)])
    greville_v = np.array([np.mean(kv[i + 1:i + degree_v + 1]) for i in range(count_v)])

    def edge_controls(index: int, greville: np.ndarray) -> np.ndarray:
        if shared_curves is not None:
            knots, degree = (ku, degree_u) if index in (0, 2) else (kv, degree_v)
            curve = shared_curves[index]
            if curve.k != degree:
                raise ValueError("Shared boundary and fitted surface must have the same spline degree")
            # The union basis contains the exact original knot vector. Knot
            # insertion changes only representation, avoiding both rounded
            # breakpoints and an unnecessary boundary collocation solve.
            for value, count in zip(*np.unique(knots, return_counts=True)):
                difference = int(count) - int(np.count_nonzero(curve.t == value))
                if difference > 0:
                    curve = curve.insert_knot(float(value), difference)
            if len(curve.t) != len(knots) or not np.array_equal(curve.t, knots):
                raise ValueError("The fitted basis must contain every exact shared boundary knot")
            return np.asarray(curve.c[:len(knots) - degree - 1]).copy()
        return np.column_stack([np.interp(greville, parameters[index], points[index][:, axis]) for axis in range(3)])

    controls = np.zeros((count_u, count_v, 3))
    controls[:, 0] = edge_controls(0, greville_u)
    controls[-1, :] = edge_controls(1, greville_v)
    controls[:, -1] = edge_controls(2, greville_u)
    controls[0, :] = edge_controls(3, greville_v)
    # Add Greville nodes when the exact boundary requires more controls than
    # the adaptive sampling grid. Do not extrapolate an underdetermined net.
    grid_u = np.unique(np.r_[np.linspace(0, 1, layout.sampled_shape[0]), greville_u])
    grid_v = np.unique(np.r_[np.linspace(0, 1, layout.sampled_shape[1]), greville_v])
    if dense_fitting:
        grid_u = np.unique(np.r_[grid_u, np.linspace(0, 1, max(24, 2 * count_u)),
                                _knot_span_samples(ku, (.5,))])
        grid_v = np.unique(np.r_[grid_v, np.linspace(0, 1, max(24, 2 * count_v)),
                                _knot_span_samples(kv, (.5,))])
    sampled = _sample_fit_grid(layout, grid_u, grid_v, sample_cache)
    au = ScipyBSpline.design_matrix(grid_u, ku, degree_u).toarray()
    av = ScipyBSpline.design_matrix(grid_v, kv, degree_v).toarray()
    residual = sampled - np.einsum("ui,ijc,vj->uvc", au, controls, av, optimize=True)
    # All coordinates share the same two basis matrices. Factor each matrix
    # once for the three coordinate right-hand sides.
    along_u = lstsq(au[:, 1:-1], residual.reshape(len(grid_u), -1), lapack_driver="gelsy")[0]
    along_u = along_u.reshape(count_u - 2, len(grid_v), 3)
    along_v = lstsq(av[:, 1:-1], along_u.transpose(1, 0, 2).reshape(len(grid_v), -1),
                    lapack_driver="gelsy")[0]
    controls[1:-1, 1:-1] = along_v.reshape(count_v - 2, count_u - 2, 3).transpose(1, 0, 2)
    box_bounds = getattr(layout, "source_box_bounds", None)
    bounded_dense_fit = False
    if dense_fitting and layout.parameter_coords is not None:
        # A uniform UV grid can miss physical features compressed by the
        # harmonic map. Include the original mesh vertices in the objective.
        # The sparse tensor basis has at most 16 entries per bicubic sample.
        from scipy.sparse import csr_matrix, kron
        from scipy.sparse.linalg import spsolve
        uv = np.asarray(layout.parameter_coords, dtype=float)
        ids = np.unique(layout.triangles)
        ids = ids[np.all((uv[ids] > 1e-10) & (uv[ids] < 1 - 1e-10), axis=1)]
        if len(ids):
            node_u = ScipyBSpline.design_matrix(uv[ids, 0], ku, degree_u).tocsr()
            node_v = ScipyBSpline.design_matrix(uv[ids, 1], kv, degree_v).tocsr()
            row_ids, column_ids, values = [], [], []
            for row in range(len(ids)):
                u_indices = node_u.indices[node_u.indptr[row]:node_u.indptr[row + 1]]
                u_values = node_u.data[node_u.indptr[row]:node_u.indptr[row + 1]]
                v_indices = node_v.indices[node_v.indptr[row]:node_v.indptr[row + 1]]
                v_values = node_v.data[node_v.indptr[row]:node_v.indptr[row + 1]]
                for iu, bu in zip(u_indices, u_values):
                    for iv, bv in zip(v_indices, v_values):
                        if 0 < iu < count_u - 1 and 0 < iv < count_v - 1:
                            row_ids.append(row)
                            column_ids.append((iu - 1) * (count_v - 2) + iv - 1)
                            values.append(bu * bv)
            node_basis = csr_matrix((values, (row_ids, column_ids)),
                                    shape=(len(ids), (count_u - 2) * (count_v - 2)))
            grid_basis = kron(csr_matrix(au[:, 1:-1]), csr_matrix(av[:, 1:-1]), format="csr")
            evaluated_nodes = np.einsum("ni,ijc,nj->nc", node_u.toarray(), controls,
                                        node_v.toarray(), optimize=True)
            evaluated_grid = np.einsum("ui,ijc,vj->uvc", au, controls, av, optimize=True)
            weight_squared = len(grid_u) * len(grid_v) / len(ids)
            normal = (grid_basis.T @ grid_basis + weight_squared * node_basis.T @ node_basis).tocsc()
            right = grid_basis.T @ (sampled - evaluated_grid).reshape(-1, 3)
            right += weight_squared * node_basis.T @ (layout.vertices[ids] - evaluated_nodes)
            # The training grid includes Greville nodes, giving a full-rank
            # basis. Tiny diagonal regularization guards numerical roundoff.
            normal.setdiag(normal.diagonal() + max(float(normal.diagonal().max()), 1.) * 1e-12)
            correction = spsolve(normal, right)
            constrained_target = normal @ controls[1:-1, 1:-1].reshape(-1, 3) + right
            controls[1:-1, 1:-1] += correction.reshape(count_u - 2, count_v - 2, 3)
            if box_bounds is not None:
                controls[1:-1, 1:-1] = _bounded_interior_fit(
                    controls[1:-1, 1:-1],
                    lambda values: (normal @ values.reshape(-1, 3)).reshape(values.shape),
                    constrained_target.reshape(count_u - 2, count_v - 2, 3), box_bounds)
                bounded_dense_fit = True
    if box_bounds is not None and not bounded_dense_fit:
        inner_u, inner_v = au[:, 1:-1], av[:, 1:-1]
        normal_u, normal_v = inner_u.T @ inner_u, inner_v.T @ inner_v
        target = np.einsum("ui,uvc,vj->ijc", inner_u, residual, inner_v, optimize=True)
        controls[1:-1, 1:-1] = _bounded_interior_fit(controls[1:-1, 1:-1],
            lambda values: np.einsum("ij,jkc,kl->ilc", normal_u, values, normal_v, optimize=True),
            target, box_bounds)
    if not np.isfinite(controls).all():
        raise ValueError("Boundary-constrained fitting produced nonfinite control points")
    surface = BSpline.Surface()
    surface.degree_u, surface.degree_v = degree_u, degree_v
    surface.set_ctrlpts(controls.reshape(-1, 3).tolist(), count_u, count_v)
    surface.knotvector_u, surface.knotvector_v = ku.tolist(), kv.tolist()
    surface.segment_boundary_edges = True
    surface.boundary_basis_report = {
        "method": "Exact shared-curve knot insertion" if shared_curves is not None else "Exact polyline degree elevation",
        "rounded_boundary_knots": False,
        "roundoff_knot_reconciliation": knot_reconciliation_reports,
        "minimum_positive_knot_span_u": float(np.diff(np.unique(ku)).min()),
        "minimum_positive_knot_span_v": float(np.diff(np.unique(kv)).min()),
    }
    return surface


def _knot_span_samples(knots, fractions=(.25, .75)) -> np.ndarray:
    unique = np.unique(knots)
    return np.concatenate([unique[:-1] + value * np.diff(unique) for value in fractions])


def _sample_fit_grid(layout, grid_u, grid_v, cache=None):
    """Reuse inverse-map samples only within one fit of unchanged source UVs."""
    from .amrto_parameterization import sample_parameter_grid

    if cache is None:
        return sample_parameter_grid(layout, grid_u, grid_v)
    key = (grid_u.tobytes(), grid_v.tobytes())
    if key not in cache:
        sampled = sample_parameter_grid(layout, grid_u, grid_v)
        # Retain only the recent training and validation grids. No source or
        # parameterization cache persists on a mutable layout between fits.
        if len(cache) >= 2:
            del cache[next(iter(cache))]
        cache[key] = sampled
    return cache[key]


def _surface_map_residual(layout: QuadLayout, surface: Any, grid_size: int = 35, *, sample_cache=None) -> float:
    """Validate each knot span and mesh vertices, including very narrow UV strips."""
    return _surface_map_validation(layout, surface, grid_size, sample_cache=sample_cache)[0]


def _surface_map_validation(layout: QuadLayout, surface: Any, grid_size: int = 35, *, sample_cache=None):
    """Return the maximum residual and sites that can guide local refinement.

    Every grid site and source vertex contributes to the maximum. The 256
    largest grid errors supplement all source vertices when choosing knots;
    retaining these sites does not reduce the independent validation grid.
    """
    from scipy.interpolate import BSpline as ScipyBSpline
    coordinates = (np.arange(grid_size) + .5) / grid_size
    test_u = np.unique(np.r_[coordinates, _knot_span_samples(surface.knotvector_u)])
    test_v = np.unique(np.r_[coordinates, _knot_span_samples(surface.knotvector_v)])
    source = _sample_fit_grid(layout, test_u, test_v, sample_cache)
    au = ScipyBSpline.design_matrix(test_u, surface.knotvector_u, surface.degree_u).toarray()
    av = ScipyBSpline.design_matrix(test_v, surface.knotvector_v, surface.degree_v).toarray()
    evaluated = np.einsum("ui,ijc,vj->uvc", au, np.asarray(surface.ctrlpts2d), av, optimize=True)
    grid_errors = np.linalg.norm(evaluated - source, axis=2)
    maximum = float(grid_errors.max())
    worst = np.argsort(grid_errors.ravel())[-256:]
    u_indices, v_indices = np.unravel_index(worst, grid_errors.shape)
    error_uv = np.column_stack((test_u[u_indices], test_v[v_indices]))
    site_errors = grid_errors.ravel()[worst]
    if layout.parameter_coords is not None:
        ids = np.unique(layout.triangles)
        uv = np.asarray(layout.parameter_coords)[ids]
        node_u = ScipyBSpline.design_matrix(uv[:, 0], surface.knotvector_u, surface.degree_u).toarray()
        node_v = ScipyBSpline.design_matrix(uv[:, 1], surface.knotvector_v, surface.degree_v).toarray()
        nodes = np.einsum("ni,ijc,nj->nc", node_u, np.asarray(surface.ctrlpts2d), node_v, optimize=True)
        node_errors = np.linalg.norm(nodes - layout.vertices[ids], axis=1)
        maximum = max(maximum, float(node_errors.max()))
        error_uv = np.vstack((uv, error_uv))
        site_errors = np.r_[node_errors, site_errors]
    return maximum, error_uv, site_errors


def nurbs_surface_to_occ_face(surf: Any, tol: float = 1e-4) -> Any:
    """Convert a geomdl BSpline surface to an OpenCASCADE TopoDS_Face."""
    deg_u = int(surf.degree_u)
    deg_v = int(surf.degree_v)
    ctrlpts = np.asarray(surf.ctrlpts2d, dtype=float)  # shape (size_u, size_v, 3)
    size_u, size_v = ctrlpts.shape[0], ctrlpts.shape[1]

    poles = TColgp_Array2OfPnt(1, size_u, 1, size_v)
    for i in range(size_u):
        for j in range(size_v):
            p = ctrlpts[i, j]
            poles.SetValue(i + 1, j + 1, gp_Pnt(float(p[0]), float(p[1]), float(p[2])))

    # Knots & multiplicities for u
    kv_u = surf.knotvector_u
    u_uniq, u_mults = np.unique(kv_u, return_counts=True)
    knots_u = TColStd_Array1OfReal(1, len(u_uniq))
    mults_u = TColStd_Array1OfInteger(1, len(u_uniq))
    for i, (k, m) in enumerate(zip(u_uniq, u_mults)):
        knots_u.SetValue(i + 1, float(k))
        mults_u.SetValue(i + 1, int(m))

    # Knots & multiplicities for v
    kv_v = surf.knotvector_v
    v_uniq, v_mults = np.unique(kv_v, return_counts=True)
    knots_v = TColStd_Array1OfReal(1, len(v_uniq))
    mults_v = TColStd_Array1OfInteger(1, len(v_uniq))
    for i, (k, m) in enumerate(zip(v_uniq, v_mults)):
        knots_v.SetValue(i + 1, float(k))
        mults_v.SetValue(i + 1, int(m))

    occ_surf = Geom_BSplineSurface(poles, knots_u, knots_v, mults_u, mults_v, deg_u, deg_v)
    surf.occ_face_construction_report = {
        "segmented_boundary_requested": bool(getattr(surf, "segment_boundary_edges", False)),
        "segmented_boundary_succeeded": False,
        "boundary_curve_method": "Natural surface boundary",
        "fallback_reason": None,
        "normalized_exact_edge_count": 0,
        "global_parameter_roundoff_fallback_count": 0,
    }
    if getattr(surf, "segment_boundary_edges", False):
        try:
            from OCP.BRep import BRep_Builder
            from OCP.Geom import Geom_TrimmedCurve
            from OCP.Geom2d import Geom2d_BezierCurve, Geom2d_Line
            from OCP.gp import gp_Pnt2d, gp_Dir2d
            from OCP.TColgp import TColgp_Array1OfPnt2d
            from OCP.BRepCheck import BRepCheck_Analyzer
            from OCP.TopLoc import TopLoc_Location

            wire_maker = BRepBuilderAPI_MakeWire()
            builder = BRep_Builder()
            location = TopLoc_Location()
            # Knot removal changes the spline basis, but patch junctions must
            # remain topological edge endpoints even on an unchanged curve.
            # Keep these split values independently of control-net complexity.
            split_u = np.unique(np.r_[u_uniq, getattr(surf, "boundary_split_parameters_u", ())])
            split_v = np.unique(np.r_[v_uniq, getattr(surf, "boundary_split_parameters_v", ())])
            if (not np.isfinite(split_u).all() or not np.isfinite(split_v).all()
                    or split_u[0] < u_uniq[0] or split_u[-1] > u_uniq[-1]
                    or split_v[0] < v_uniq[0] or split_v[-1] > v_uniq[-1]):
                raise ValueError("Spline boundary split parameters must lie in the surface domain")
            # Isocurves already contain the exact boundary control points and
            # knots. Lifting a local-parameter pcurve with BuildCurves3d would
            # approximate them and can fail on compressed knot intervals.
            # Keep each exact segment in a local unit interval. Long global
            # curve parameters paired with very short edge intervals can lose
            # trim correspondence during repeated STEP transfers. An affine
            # knot change and a linear UV Bezier preserve the exact geometry.
            sides = [(split_u, occ_surf.VIso(0.0), (0.0, 0.0), (1.0, 0.0), False),
                     (split_v, occ_surf.UIso(1.0), (1.0, 0.0), (0.0, 1.0), False),
                     (split_u, occ_surf.VIso(1.0), (0.0, 1.0), (1.0, 0.0), True),
                     (split_v, occ_surf.UIso(0.0), (0.0, 0.0), (0.0, 1.0), True)]
            for parameters, iso_curve, origin, direction, reverse in sides:
                intervals = list(zip(parameters, parameters[1:]))
                if reverse:
                    intervals.reverse()
                for first, last in intervals:
                    first, last = float(first), float(last)
                    if last - first <= np.finfo(float).eps:
                        continue
                    curve = iso_curve.Copy()
                    curve.Segment(first, last)
                    normalized_knots = TColStd_Array1OfReal(1, curve.NbKnots())
                    for knot_index in range(1, curve.NbKnots() + 1):
                        normalized_knots.SetValue(knot_index, (curve.Knot(knot_index) - first) / (last - first))
                    curve.SetKnots(normalized_knots)
                    uv_poles = TColgp_Array1OfPnt2d(1, 2)
                    for pole_index, parameter in enumerate((first, last), 1):
                        uv_poles.SetValue(pole_index, gp_Pnt2d(origin[0] + direction[0] * parameter,
                                                           origin[1] + direction[1] * parameter))
                    pcurve = Geom2d_BezierCurve(uv_poles)
                    curve_first, curve_last = 0., 1.
                    # Extremely compressed UV spans can amplify roundoff in
                    # the affine UV evaluation. Retain exact global parameters
                    # for that edge rather than changing its OCC tolerance.
                    correspondence_error = 0.
                    for sample_parameter in np.linspace(0., 1., 9):
                        uv_point = pcurve.Value(float(sample_parameter))
                        correspondence_error = max(correspondence_error,
                            curve.Value(float(sample_parameter)).Distance(occ_surf.Value(uv_point.X(), uv_point.Y())))
                    if correspondence_error > 1e-7:
                        curve = Geom_TrimmedCurve(iso_curve, first, last)
                        pcurve = Geom2d_Line(gp_Pnt2d(*origin), gp_Dir2d(*direction))
                        curve_first, curve_last = first, last
                        surf.occ_face_construction_report["global_parameter_roundoff_fallback_count"] += 1
                    else:
                        surf.occ_face_construction_report["normalized_exact_edge_count"] += 1
                    edge_maker = BRepBuilderAPI_MakeEdge(curve, curve_first, curve_last)
                    if not edge_maker.IsDone():
                        raise RuntimeError("Failed to create a segmented spline boundary edge")
                    edge = edge_maker.Edge()
                    builder.UpdateEdge(edge, pcurve, occ_surf, location, 1e-7)
                    builder.Range(edge, occ_surf, location, curve_first, curve_last)
                    if reverse:
                        edge.Reverse()
                    wire_maker.Add(edge)
            if not wire_maker.IsDone():
                raise RuntimeError("Failed to create the spline boundary wire")
            face_maker = BRepBuilderAPI_MakeFace(occ_surf, wire_maker.Wire(), True)
            if not face_maker.IsDone() or not BRepCheck_Analyzer(face_maker.Face()).IsValid():
                surf.occ_face_construction_report["fallback_reason"] = "Exact segmented boundary did not form a valid OCC face"
                face_maker = BRepBuilderAPI_MakeFace(occ_surf, float(tol))
            else:
                surf.occ_face_construction_report.update(
                    segmented_boundary_succeeded=True,
                    boundary_curve_method="Exact surface isocurve segments with affine UV pcurves and checked unit-interval correspondence",
                    boundary_split_count_u=int(len(split_u)), boundary_split_count_v=int(len(split_v)),
                )
        except Exception as error:
            surf.occ_face_construction_report["fallback_reason"] = str(error)
            face_maker = BRepBuilderAPI_MakeFace(occ_surf, float(tol))
    else:
        face_maker = BRepBuilderAPI_MakeFace(occ_surf, float(tol))
    if not face_maker.IsDone():
        raise RuntimeError("Failed to build TopoDS_Face from NURBS surface")
    surf.occ_face_construction_report["edge_count"] = len(cq.Shape.cast(face_maker.Face()).Edges())
    return face_maker.Face()


def _assemble_nested_solids(solids: list[cq.Solid], tolerance: float) -> tuple[list[cq.Solid], int]:
    """Classify nested shells by containment and attach odd-depth shells as voids."""
    from OCP.BRepClass3d import BRepClass3d_SolidClassifier
    from OCP.TopAbs import TopAbs_IN
    from OCP.TopoDS import TopoDS

    if len(solids) < 2:
        return solids, 0
    from OCP.Bnd import Bnd_Box
    from OCP.BRepBndLib import BRepBndLib
    # Conservative spline bounds suffice for rejecting distant point queries;
    # optimizing every face bound and integrating volumes is unnecessary here.
    bounds = []
    for solid in solids:
        box = Bnd_Box()
        BRepBndLib.Add_s(solid.wrapped, box, False)
        bounds.append(box.Get())
    inside = np.zeros((len(solids), len(solids)), dtype=bool)
    parents = [None] * len(solids)
    classifiers = {}
    for i, solid in enumerate(solids):
        point = solid.Vertices()[0].Center()
        coordinates = point.toTuple()
        query = gp_Pnt(*coordinates)
        for j, outer in enumerate(solids):
            if i == j:
                continue
            box = bounds[j]
            if not all(box[axis] - tolerance <= coordinates[axis] <= box[axis + 3] + tolerance
                       for axis in range(3)):
                continue
            # Reuse the loaded body's geometry for all candidate cavity points.
            # Constructing its explorer repeatedly is costly on thousands of
            # spline faces; Perform resets the per-query classification state.
            if j not in classifiers:
                classifiers[j] = BRepClass3d_SolidClassifier(outer.wrapped)
            classifier = classifiers[j]
            classifier.Perform(query, tolerance)
            if classifier.State() == TopAbs_IN:
                inside[j, i] = True
    for i in range(len(solids)):
        containers = np.flatnonzero(inside[:, i]).tolist()
        if containers:
            # The immediate parent contains no other container of this shell.
            closest = [j for j in containers if not any(inside[j, k] for k in containers if k != j)]
            if len(closest) != 1:
                raise ValueError("Shell containment is ambiguous or cyclic")
            parents[i] = closest[0]
    depths = []
    for i in range(len(solids)):
        depth, parent = 0, parents[i]
        visited = {i}
        while parent is not None:
            if parent in visited:
                raise ValueError("Cyclic shell containment")
            visited.add(parent)
            depth += 1
            parent = parents[parent]
        depths.append(depth)
    output = []
    for i, solid in enumerate(solids):
        if depths[i] % 2:
            continue
        maker = BRepBuilderAPI_MakeSolid(solid.Shells()[0].wrapped)
        for j, cavity in enumerate(solids):
            if parents[j] == i:
                maker.Add(TopoDS.Shell_s(cavity.Shells()[0].wrapped.Reversed()))
        candidate = cq.Solid(maker.Solid())
        if not candidate.isValid():
            raise ValueError("Nested shell assembly did not form a valid solid")
        output.append(candidate)
    return output, sum(depth % 2 for depth in depths)


def assemble_brep_cad_model(
    layouts: List[QuadLayout],
    sew_tolerance: float = 1e-3,
    merge_angle_deg: float = 0.0,
) -> Tuple[cq.Workplane, dict]:
    """Assemble all fitted NURBS patches into a sewn, solid OpenCASCADE B-rep model."""
    sewing = BRepBuilderAPI_Sewing(float(sew_tolerance))
    valid_faces = 0
    failed_layout_ids = []
    from OCP.BRep import BRep_Tool
    layout_surfaces = []

    for lay in layouts:
        if lay.nurbs_surface is None:
            failed_layout_ids.append(lay.layout_id)
            continue
        try:
            face = nurbs_surface_to_occ_face(lay.nurbs_surface, tol=sew_tolerance)
            layout_surfaces.append((BRep_Tool.Surface_s(face), int(lay.layout_id)))
            sewing.Add(face)
            valid_faces += 1
        except Exception as exc:
            failed_layout_ids.append(lay.layout_id)
            logger.warning("Could not build face for layout %d: %s", lay.layout_id, exc)

    if valid_faces == 0:
        raise RuntimeError("No valid NURBS faces could be constructed for CAD assembly.")

    sewing.Perform()
    sewed_shape = sewing.SewedShape()

    if sewed_shape.IsNull():
        raise RuntimeError("NURBS sewing produced a null shape")
    from OCP.BRep import BRep_Tool

    sewed = cq.Shape.cast(sewed_shape)
    shells = sewed.Shells()
    solids = []
    all_shells_closed = bool(shells)
    for shell in shells:
        try:
            shell_fixer = ShapeFix_Shell(shell.wrapped)
            shell_fixer.Perform()
            fixed_shell = shell_fixer.Shell()
            # MakeSolid accepts an open shell too; it does not certify closure.
            if not BRep_Tool.IsClosed_s(fixed_shell):
                all_shells_closed = False
                continue
            maker = BRepBuilderAPI_MakeSolid(fixed_shell)
            if not maker.IsDone():
                all_shells_closed = False
                continue
            solid_fixer = ShapeFix_Solid(maker.Solid())
            solid_fixer.Perform()
            candidate = cq.Solid(solid_fixer.Solid())
            if not candidate.isValid() or not all(BRep_Tool.IsClosed_s(s.wrapped) for s in candidate.Shells()):
                all_shells_closed = False
                continue
            solids.append(candidate)
        except Exception as exc:
            all_shells_closed = False
            logger.debug("Solid closure exception: %s; retaining complete sewn model", exc)
    free_edge_count = int(sewing.NbFreeEdges())
    all_faces_in_shells = sum(len(shell.Faces()) for shell in shells) == len(sewed.Faces())
    is_solid = (not failed_layout_ids and all_shells_closed and len(solids) == len(shells)
                and all_faces_in_shells and free_edge_count == 0)
    cavity_shell_count = 0
    if is_solid:
        try:
            solids, cavity_shell_count = _assemble_nested_solids(solids, sew_tolerance)
        except Exception as exc:
            logger.warning("Nested shell assembly failed: %s", exc)
            is_solid = False
    if is_solid:
        final_shape = solids[0] if len(solids) == 1 else cq.Compound.makeCompound(solids)
    else:
        # An open shell or failed component must not remove other parts.
        final_shape = sewed
    if merge_angle_deg > 0.0:
        final_shape = _unify_same_domain_shape(final_shape, merge_angle_deg=merge_angle_deg)
    workplane = cq.Workplane().newObject([final_shape])
    report = {
        "step": 5, "nurbs_patch_count": valid_faces,
        "face_layout_ids": [
            [layout_id for surface, layout_id in layout_surfaces
             if surface == BRep_Tool.Surface_s(face.wrapped)]
            for face in final_shape.Faces()
        ],
        "input_layout_count": len(layouts), "failed_layout_ids": failed_layout_ids,
        "cad_volume": _solid_volume(final_shape, relative_tolerance=1e-4) if is_solid else None,
        "volume_relative_integration_tolerance": 1e-4,
        "is_solid": bool(is_solid), "is_valid": bool(final_shape.isValid()),
        "shell_count": len(shells), "solid_count": len(solids),
        "cavity_shell_count": cavity_shell_count,
        "free_edge_count": free_edge_count, "sew_tolerance": sew_tolerance,
        "status": "solid" if is_solid else "open_or_incomplete_brep",
        "paper_multiresolution": False,
        "boundary_preserving_patch_count": sum(bool(getattr(layout, "fitting_report", {}).get("preserve_boundary"))
                                               for layout in layouts),
        "boundary_construction": {
            "segmented_patch_count": sum(bool(getattr(layout.nurbs_surface, "occ_face_construction_report", {})
                                              .get("segmented_boundary_succeeded")) for layout in layouts),
            "natural_boundary_fallback_patch_ids": [layout.layout_id for layout in layouts
                if getattr(layout.nurbs_surface, "occ_face_construction_report", {}).get("fallback_reason")],
            "patch_reports": [{"layout_id": layout.layout_id,
                               **getattr(layout.nurbs_surface, "occ_face_construction_report", {})}
                              for layout in layouts],
        },
    }

    return workplane, report


def export_cad_model(
    workplane_or_shape: Any,
    layouts: List[QuadLayout],
    export_path: str | Path,
    export_format: str = "STEP",
) -> Path:
    """Export the reconstructed CAD model to STEP, IGES, JSON, or Rhino 3DM."""
    p = Path(export_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fmt = export_format.upper()

    shape = workplane_or_shape
    if hasattr(shape, "objects") and shape.objects:
        cand = shape.objects[0]
        if hasattr(cand, "wrapped") and not isinstance(cand, cq.Vector):
            shape = cand
        elif hasattr(shape, "val"):
            val = shape.val()
            if not isinstance(val, cq.Vector):
                shape = val
    elif hasattr(shape, "val"):
        val = shape.val()
        if not isinstance(val, cq.Vector):
            shape = val
    occ_shape = shape.wrapped if hasattr(shape, "wrapped") else shape

    if fmt in {"STEP", "STP"}:
        from .step_export import export_cad_model as export_step_model
        # Final AMRTO transfer needs the same scoped precision policy as the
        # passive-feature transfers. Its caller verifies the round trip.
        return export_step_model(cq.Shape.cast(occ_shape), export_path=p, verify=False)

    elif fmt in {"IGES", "IGS"}:
        writer = IGESControl_Writer()
        if not writer.AddShape(occ_shape) or not writer.Write(str(p)):
            raise RuntimeError(f"IGES export failed: {p}")
        return p

    elif fmt == "JSON":
        surfs = [lay.nurbs_surface for lay in layouts if lay.nurbs_surface is not None]
        container = multi.SurfaceContainer(surfs)
        container.delta = 0.04
        exchange.export_json(container, str(p))
        return p

    elif fmt in {"3DM", "RHINO"}:
        # Export json first, then invoke json2on.exe
        json_path = p.with_suffix(".json")
        surfs = [lay.nurbs_surface for lay in layouts if lay.nurbs_surface is not None]
        container = multi.SurfaceContainer(surfs)
        container.delta = 0.04
        exchange.export_json(container, str(json_path))

        j2on = get_json2on_executable()
        if j2on and j2on.is_file():
            try:
                subprocess.run([str(j2on), str(json_path)], check=True, timeout=60,
                               capture_output=True, **headless_subprocess_kwargs())
            except Exception as exc:
                logger.warning("json2on failed to generate 3DM: %s", exc)
        if not p.is_file():
            raise RuntimeError(f"Rhino export did not produce {p}")
        return p

    else:
        raise ValueError(f"Unsupported CAD export format: {export_format}")
