# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.
"""Validate Algorithms 2/3 candidates before replacing a fitted CAD patch."""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable

import numpy as np
from scipy.interpolate import BSpline

from .amrto_multiresolution import apply_multiresolution_strategy, sampled_surface_difference


def _sites(surface: Any, axis: str) -> np.ndarray:
    knots = np.unique(getattr(surface, f"knotvector_{axis}"))
    fractions = [a + (b-a)*fraction for a, b in zip(knots[:-1], knots[1:])
                 for fraction in (.25, .5, .75)]
    return np.unique(np.r_[np.linspace(0., 1., 35), knots, fractions])


def _boundary_points(surface: Any, axis: str, parameter: np.ndarray, end: int) -> np.ndarray:
    controls = np.asarray(surface.ctrlpts).reshape(surface.ctrlpts_size_u, surface.ctrlpts_size_v, 3)
    row = controls[:, end] if axis == "u" else controls[end, :]
    basis = BSpline.design_matrix(parameter, getattr(surface, f"knotvector_{axis}"),
                                 getattr(surface, f"degree_{axis}")).toarray()
    if not surface.rational:
        return basis @ row
    weights = np.asarray(surface.weights).reshape(surface.ctrlpts_size_u, surface.ctrlpts_size_v)
    weights = weights[:, end] if axis == "u" else weights[end, :]
    return (basis @ (row * weights[:, None])) / (basis @ weights)[:, None]


def surface_boundary_difference(original: Any, candidate: Any) -> dict:
    """Check all four clamped spline boundaries independently of fit samples."""
    maxima, samples = [], 0
    for axis in ("u", "v"):
        sites = np.unique(np.r_[_sites(original, axis), _sites(candidate, axis)])
        for end in (0, -1):
            distances = np.linalg.norm(_boundary_points(original, axis, sites, end)
                                       - _boundary_points(candidate, axis, sites, end), axis=1)
            if not np.isfinite(distances).all():
                raise ValueError("Multiresolution boundary contains nonfinite points")
            maxima.append(float(distances.max()))
            samples += len(distances)
    return {"maximum_distance": max(maxima), "side_maximum_distances": maxima,
            "sample_count": samples, "certified_bound": False,
            "method": "Clamped boundary evaluation at uniform and knot-span quarter sites"}


def restore_original_control_border(original: Any, candidate: Any) -> Any:
    """Production G0 extension: freeze the original lower control-net border.

    This is deliberately separate from the authors' unmodified Algorithm 2.
    It preserves all four clamped boundary curves exactly in the original basis.
    """
    if (original.ctrlpts_size_u, original.ctrlpts_size_v) != (candidate.ctrlpts_size_u, candidate.ctrlpts_size_v):
        raise ValueError("Border restoration requires the regenerated original control shape")
    if any(getattr(original, key) != getattr(candidate, key)
           for key in ("degree_u", "degree_v", "knotvector_u", "knotvector_v")):
        raise ValueError("Border restoration requires the original spline basis")
    nu, nv = original.ctrlpts_size_u, original.ctrlpts_size_v
    points = np.asarray(candidate.ctrlpts).reshape(nu, nv, 3).copy()
    original_points = np.asarray(original.ctrlpts).reshape(nu, nv, 3)
    points[[0, -1], :, :] = original_points[[0, -1], :, :]
    points[:, [0, -1], :] = original_points[:, [0, -1], :]
    result = deepcopy(candidate)
    flat = points.reshape(-1, 3)
    if original.rational:
        weights = np.asarray(original.weights)
        flat = np.c_[flat * weights[:, None], weights]
    result.set_ctrlpts(flat.tolist(), nu, nv)
    return result


def apply_validated_multiresolution(
    layout: Any, original: Any, *, curvature_ck: float, curvature_cp: float,
    compression_ratio: float, layers: int, regeneration: str,
    fit_tolerance: float, preserve_boundary: bool,
    residual_evaluator: Callable[[Any, Any], float], maximum_knot_removal_attempts: int = 64,
    orientation_evaluator: Callable[[Any, Any], dict] | None = None,
    original_residual: float | None = None, original_orientation: dict | None = None,
    curvature_reference_length: float = 1.0,
) -> Any:
    """Execute classification/hierarchy/removal and accept only validated geometry.

    Independent geometry and shared-boundary guards extend the paper's short
    algorithms. A rejected compression keeps the fitted original and its report;
    it is never reported as an achieved control reduction.
    """
    extent = float(np.linalg.norm(np.ptp(np.asarray(original.ctrlpts), axis=0)))
    budget = fit_tolerance if fit_tolerance > 0 else max(1e-9, extent * 1e-12)
    boundary_limit = max(1e-9, extent * 1e-12)
    try:
        result = apply_multiresolution_strategy(original, curvature_ck=curvature_ck,
            curvature_cp=curvature_cp, compression_ratio=compression_ratio, layers=layers,
            regeneration=regeneration, maximum_error=budget,
            curvature_reference_length=curvature_reference_length,
            maximum_knot_removal_attempts=maximum_knot_removal_attempts,
            knot_removal_acceptance_predicate=(
                lambda base, trial: surface_boundary_difference(base, trial)["maximum_distance"] <= boundary_limit
            ) if preserve_boundary else None)
        candidate = result.candidate
        report = result.report
        report["raw_candidate_geometry_validation"] = report.get("geometry_validation")
        if preserve_boundary and result.classification == "hierarchy":
            candidate = restore_original_control_border(original, candidate)
            report["boundary_restoration_extension"] = True
            report["boundary_restoration_method"] = "Original lower-net border frozen in original spline basis"
        # The preserve branch often returns an unchanged deep copy. Its
        # exact basis, poles and weights prove identical geometry; reuse the
        # caller's checks of that same source map and orientation checker.
        identical = (
            all(getattr(original, name) == getattr(candidate, name)
                for name in ("degree_u", "degree_v", "knotvector_u", "knotvector_v",
                             "ctrlpts_size_u", "ctrlpts_size_v", "rational"))
            and np.array_equal(original.ctrlpts, candidate.ctrlpts)
            and np.array_equal(getattr(original, "weights", None), getattr(candidate, "weights", None)))
        if identical:
            geometry = {"maximum_distance": 0.0, "sample_count": 0, "certified_bound": True,
                        "method": "Bitwise identical spline basis, controls and weights"}
            boundary = {**geometry, "side_maximum_distances": [0.0] * 4}
        else:
            geometry = sampled_surface_difference(original, candidate)
            boundary = surface_boundary_difference(original, candidate)
        residual = ((original_residual if identical and original_residual is not None
                     else residual_evaluator(layout, candidate)) if fit_tolerance > 0 else None)
        accepted = geometry["maximum_distance"] <= budget
        reasons = []
        if not accepted:
            reasons.append("Candidate exceeds cumulative geometry budget")
        if preserve_boundary and boundary["maximum_distance"] > boundary_limit:
            accepted = False
            reasons.append("Candidate moves a shared boundary")
        if residual is not None and residual > 3 * fit_tolerance:
            accepted = False
            reasons.append("Candidate exceeds source-map residual limit")
        orientation = None
        if accepted and orientation_evaluator is not None:
            orientation = (original_orientation if identical and original_orientation is not None
                           else orientation_evaluator(layout, candidate))
            if not orientation["accepted"]:
                accepted = False
                reasons.append("Candidate fails source orientation validation")
        original_count = original.ctrlpts_size_u * original.ctrlpts_size_v
        candidate_count = candidate.ctrlpts_size_u * candidate.ctrlpts_size_v
        report.update(accepted=accepted, acceptance_reasons=reasons,
            geometry_validation=geometry, boundary_validation=boundary,
            boundary_maximum_error=boundary_limit if preserve_boundary else None,
            candidate_dense_map_residual=residual,
            candidate_source_orientation=orientation,
            source_map_residual_limit=3 * fit_tolerance if fit_tolerance > 0 else None,
            exported_control_count=candidate_count if accepted else original_count,
            original_control_count=original_count,
            achieved_exported_compression_ratio=(candidate_count if accepted else original_count)/original_count,
            accepted_editable_control_count=(report.get("highest_layer_control_count", candidate_count)
                                              if accepted else original_count),
            paper_algorithm_executed=True,
            identical_surface_validation_reused=identical and (
                (fit_tolerance > 0 and original_residual is not None)
                or (orientation_evaluator is not None and original_orientation is not None)),
            production_validation_extension=True)
        layout.multiresolution_layers = result.layers
        layout.multiresolution_report = report
        return candidate if accepted else original
    except (ValueError, ZeroDivisionError, IndexError, np.linalg.LinAlgError) as error:
        # A numerically failed optional compression does not discard a valid fit.
        layout.multiresolution_layers = []
        layout.multiresolution_report = {"accepted": False, "error": str(error),
            "paper_algorithm_executed": True, "production_validation_extension": True,
            "exported_control_count": original.ctrlpts_size_u * original.ctrlpts_size_v}
        return original
