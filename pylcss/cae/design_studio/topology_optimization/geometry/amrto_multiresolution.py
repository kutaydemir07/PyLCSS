# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.
"""AMRTO Algorithms 2/3: control-net hierarchy and curvature classification.

Higher layers approximate the *control points* of the preceding surface, not
its tessellation. Regeneration evaluates a higher layer to replace the lower
layer's controls while retaining the lower degree, knots and rational weights.
The author-uniform sampling mode follows ``Figure_multi_resolution.py`` with
the display translations removed. Greville sampling is an explicit alternative
interpretation of the paper's reference to the lower layer's knot vectors.

This module never edits its input or restores CAD boundaries. A production
caller must validate any boundary-preserving extension and the final CAD.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import math
from typing import Any, Callable, Sequence

import numpy as np
from geomdl import BSpline, fitting, operations
from scipy.interpolate import BSpline as ScipyBSpline


@dataclass
class MultiresolutionResult:
    candidate: Any
    classification: str
    layers: list[Any] = field(default_factory=list)
    regenerated_layers: list[Any] = field(default_factory=list)
    report: dict[str, Any] = field(default_factory=dict)


def classify_patch_curvature(curvature: float, curvature_ck: float = .1,
                             curvature_cp: float = 10.) -> str:
    """Apply Algorithm 3's strict inequalities; threshold equality is middle."""
    if not all(math.isfinite(x) for x in (curvature, curvature_ck, curvature_cp)):
        raise ValueError("Curvature and classification thresholds must be finite")
    if curvature < 0 or not 0 <= curvature_ck <= curvature_cp:
        raise ValueError("Require nonnegative curvature and 0 <= Ck <= Cp")
    if curvature < curvature_ck:
        return "knot_removal"
    if curvature > curvature_cp:
        return "preserve"
    return "hierarchy"


def principal_curvature(surface: Any, samples: int = 9) -> float:
    """Estimate maximum absolute principal curvature from exact derivatives.

    The paper does not prescribe a scalar reduction or sample grid. This finite
    maximum is recorded as the classification policy, not a curvature bound.
    """
    if samples < 3:
        raise ValueError("Curvature sampling requires at least three sites per axis")
    u, v = [_validation_sites(surface, axis, samples) for axis in ("u", "v")]
    # Exact basis derivatives in bounded tensor tiles include ALL cross-pairs
    # of knot-span quarter sites, including features narrow in both axes.
    controls = _controls(surface)
    if surface.rational:
        weights = np.asarray(surface.weights).reshape(*_shape(surface))
        tensor = np.concatenate((controls * weights[..., None], weights[..., None]), axis=2)
    else:
        tensor = controls
    bases = []
    for axis, sites in (("u", u), ("v", v)):
        count = getattr(surface, f"ctrlpts_size_{axis}")
        spline = ScipyBSpline(getattr(surface, f"knotvector_{axis}"), np.eye(count),
                             getattr(surface, f"degree_{axis}"))
        bases.append([spline(sites, nu=order) for order in (0, 1, 2)])
    maximum, regular_count = 0., 0
    tile_size = max(1, 65536 // len(v))
    for first in range(0, len(u), tile_size):
        bu = [basis[first:first + tile_size] for basis in bases[0]]
        bv = bases[1]
        d = {}
        denominator_derivatives = {}
        for i, j in ((0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2)):
            value = np.einsum("ui,ijc,vj->uvc", bu[i], tensor, bv[j], optimize=True)
            if not surface.rational:
                d[i, j] = value
                continue
            numerator, denominator_derivative = value[..., :3], value[..., 3:]
            if (i, j) == (0, 0):
                denominator = denominator_derivative
                d[0, 0] = numerator / denominator
            elif i + j == 1:
                d[i, j] = (numerator - denominator_derivative * d[0, 0]) / denominator
            elif (i, j) == (1, 1):
                d[i, j] = (numerator - denominator_derivatives[1, 0] * d[0, 1]
                           - denominator_derivatives[0, 1] * d[1, 0]
                           - denominator_derivative * d[0, 0]) / denominator
            else:
                first_order = (i // 2, j // 2)
                d[i, j] = (numerator - 2 * denominator_derivatives[first_order] * d[first_order]
                           - denominator_derivative * d[0, 0]) / denominator
            denominator_derivatives[i, j] = denominator_derivative
        du, dv = d[1, 0], d[0, 1]
        cross = np.cross(du, dv)
        length = np.linalg.norm(cross, axis=2)
        e, f, g = [np.einsum("uvc,uvc->uv", a, b) for a, b in ((du, du), (du, dv), (dv, dv))]
        determinant = e * g - f * f
        valid = (length > 0) & (determinant > np.finfo(float).eps * np.maximum(e * g, np.finfo(float).tiny))
        if not valid.any():
            continue
        normal = cross[valid] / length[valid, None]
        second_u, mixed, second_v = [np.einsum("nc,nc->n", d[ij][valid], normal)
                                    for ij in ((2, 0), (1, 1), (0, 2))]
        mean = (e[valid] * second_v + g[valid] * second_u - 2 * f[valid] * mixed) / (2 * determinant[valid])
        gaussian = (second_u * second_v - mixed * mixed) / determinant[valid]
        separation = np.sqrt(np.maximum(0., mean * mean - gaussian))
        values = np.maximum(np.abs(mean + separation), np.abs(mean - separation))
        if not np.isfinite(values).all():
            raise ValueError("Surface has nonfinite regular tangent-plane curvature samples")
        maximum = max(maximum, float(values.max()))
        regular_count += int(valid.sum())
    if not regular_count:
        raise ValueError("Surface has no finite regular tangent-plane curvature samples")
    return maximum


def _domain(surface: Any, axis: str) -> tuple[float, float]:
    degree = getattr(surface, f"degree_{axis}")
    knots = getattr(surface, f"knotvector_{axis}")
    return float(knots[degree]), float(knots[-degree - 1])


def _shape(surface: Any) -> tuple[int, int]:
    return surface.ctrlpts_size_u, surface.ctrlpts_size_v


def _controls(surface: Any) -> np.ndarray:
    points = np.asarray(surface.ctrlpts, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("Require a finite three-dimensional control net")
    return points.reshape(*_shape(surface), 3)


def _with_controls(template: Any, points: np.ndarray) -> Any:
    result = deepcopy(template)
    nu, nv = _shape(template)
    flat = np.asarray(points).reshape(nu * nv, 3)
    if template.rational:
        weights = np.asarray(template.weights)
        flat = np.c_[flat * weights[:, None], weights]
    result.set_ctrlpts(flat.tolist(), nu, nv)
    # set_ctrlpts invalidates evaluation caches but retains the original knots.
    return result


def _regeneration_sites(template: Any, axis: str, method: str) -> np.ndarray:
    count = getattr(template, f"ctrlpts_size_{axis}")
    if method == "author_uniform":
        return np.linspace(0., 1., count)
    if method != "greville":
        raise ValueError("Regeneration must be 'author_uniform' or 'greville'")
    degree = getattr(template, f"degree_{axis}")
    knots = np.asarray(getattr(template, f"knotvector_{axis}"))
    lo, hi = _domain(template, axis)
    return np.array([(knots[i + 1:i + degree + 1].mean() - lo) / (hi - lo)
                     for i in range(count)])


def _choose_control_shape(previous: tuple[int, int], minimum: tuple[int, int],
                          original: tuple[int, int], budget: int) -> tuple[int, int]:
    """Allocate a total-count budget, preserving the control-net aspect ratio."""
    if budget < minimum[0] * minimum[1]:
        return minimum
    factor = math.sqrt(budget / (original[0] * original[1]))
    shape = [min(previous[i], max(minimum[i], round(original[i] * factor))) for i in (0, 1)]
    aspect = original[0] / original[1]
    while shape[0] * shape[1] > budget:
        choices = [tuple(shape[j] - int(j == axis) for j in (0, 1))
                   for axis in (0, 1) if shape[axis] > minimum[axis]]
        if not choices:
            break
        shape = list(min(choices, key=lambda p: abs(math.log((p[0] / p[1]) / aspect))))
    while True:
        choices = [tuple(shape[j] + int(j == axis) for j in (0, 1))
                   for axis in (0, 1) if shape[axis] < previous[axis]
                   and (shape[axis] + 1) * shape[1 - axis] <= budget]
        if not choices:
            return tuple(shape)
        shape = list(min(choices, key=lambda p: abs(math.log((p[0] / p[1]) / aspect))))


def build_control_hierarchy(surface: Any, *, layers: int = 1,
                            compression_ratio: float = .5,
                            layer_specs: Sequence[tuple[int, int, int, int]] | None = None,
                            regeneration: str = "author_uniform") -> MultiresolutionResult:
    """Construct Algorithm 2's layers and regenerate the ORIGINAL control net.

    ``layer_specs`` contains (controls_u, controls_v, degree_u, degree_v).
    Without explicit specs, ``compression_ratio`` is the highest layer's total
    nu*nv budget divided by the original nu*nv count. Intermediate layers use
    geometric fractions of this overall budget. Integer/minimum-degree limits
    are reported; this ratio is never applied independently to both axes.
    """
    if not isinstance(layers, int) or layers < 1 or not 0 < compression_ratio <= 1:
        raise ValueError("Require at least one layer and 0 < compression_ratio <= 1")
    if regeneration not in ("author_uniform", "greville"):
        raise ValueError("Regeneration must be 'author_uniform' or 'greville'")
    original_shape = _shape(surface)
    _controls(surface)
    current = surface
    fitted, specifications = [], []
    explicit = layer_specs is not None
    if explicit and not layer_specs:
        raise ValueError("Explicit layer specifications cannot be empty")
    count = len(layer_specs) if explicit else layers
    for index in range(count):
        previous = _shape(current)
        if explicit:
            nu, nv, pu, pv = layer_specs[index]
        else:
            pu, pv = surface.degree_u, surface.degree_v
            budget = math.floor(math.prod(original_shape) * compression_ratio ** ((index + 1) / count))
            nu, nv = _choose_control_shape(previous, (pu + 1, pv + 1), original_shape, budget)
            if (nu, nv) == previous:
                continue
        if not all(isinstance(x, (int, np.integer)) for x in (nu, nv, pu, pv)):
            raise ValueError("Layer control counts and degrees must be integers")
        if pu < 1 or pv < 1 or not pu < nu <= previous[0] or not pv < nv <= previous[1]:
            raise ValueError("Layer counts must exceed degrees and not exceed the preceding net")
        driver = fitting.approximate_surface(_controls(current).reshape(-1, 3).tolist(),
            size_u=previous[0], size_v=previous[1], degree_u=pu, degree_v=pv,
            centripetal=False, ctrlpts_size_u=nu, ctrlpts_size_v=nv)
        fitted.append(driver)
        specifications.append({"control_shape": [nu, nv], "degrees": [pu, pv]})
        current = driver
    templates = [surface] + fitted
    regenerated = []
    driver = current
    for template in reversed(templates[:-1]):
        u = _regeneration_sites(template, "u", regeneration)
        v = _regeneration_sites(template, "v", regeneration)
        ulo, uhi = _domain(driver, "u")
        vlo, vhi = _domain(driver, "v")
        params = [[ulo + x * (uhi - ulo), vlo + y * (vhi - vlo)] for x in u for y in v]
        driver = _with_controls(template, np.asarray(driver.evaluate_list(params)))
        regenerated.append(driver)
    candidate = driver if fitted else deepcopy(surface)
    original_count = math.prod(original_shape)
    driver_count = math.prod(_shape(fitted[-1])) if fitted else original_count
    return MultiresolutionResult(candidate, "hierarchy", fitted, regenerated, {
        "algorithm": 2, "method": "Fit control nets to higher layers and regenerate original controls",
        "regeneration": regeneration, "regeneration_preserves_original_degree_and_knots": True,
        "regeneration_preserves_original_weights": True,
        "original_control_shape": list(original_shape), "original_control_count": original_count,
        "layer_specs": specifications, "requested_layers": count, "constructed_layers": len(fitted),
        "requested_compression_ratio": compression_ratio, "explicit_layer_specs": explicit,
        "compression_definition": "Highest editable layer total controls / original total controls",
        "highest_layer_control_count": driver_count,
        "regenerated_original_control_count": math.prod(_shape(candidate)),
        "actual_compression_ratio": driver_count / original_count,
        "compression_target_met": driver_count <= compression_ratio * original_count,
        "accepted": True, "boundary_restoration_extension": False,
    })


def _validation_sites(surface: Any, axis: str, samples: int) -> np.ndarray:
    lo, hi = _domain(surface, axis)
    knots = np.unique(getattr(surface, f"knotvector_{axis}"))
    knots = knots[(knots >= lo) & (knots <= hi)]
    spans = [left + (right - left) * fraction for left, right in zip(knots[:-1], knots[1:])
             for fraction in (.25, .5, .75)]
    return np.unique(np.r_[np.linspace(lo, hi, samples), knots, spans])


def _evaluate_tensor(surface: Any, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    bu = ScipyBSpline.design_matrix(u, surface.knotvector_u, surface.degree_u).toarray()
    bv = ScipyBSpline.design_matrix(v, surface.knotvector_v, surface.degree_v).toarray()
    controls = _controls(surface)
    if surface.rational:
        weights = np.asarray(surface.weights).reshape(*_shape(surface))
        points = np.einsum("ui,ijc,vj->uvc", bu, controls * weights[..., None], bv, optimize=True)
        denominator = np.einsum("ui,ij,vj->uv", bu, weights, bv, optimize=True)
        return points / denominator[..., None]
    return np.einsum("ui,ijc,vj->uvc", bu, controls, bv, optimize=True)


def sampled_surface_difference(original: Any, candidate: Any, *, samples: int = 17) -> dict[str, Any]:
    """Independent same-parameter check including quarter points of every span."""
    if samples < 3:
        raise ValueError("Surface validation requires at least three sites per axis")
    if any(_domain(original, a) != _domain(candidate, a) for a in ("u", "v")):
        raise ValueError("Comparison requires identical parameter domains")
    u, v = [np.unique(np.r_[_validation_sites(original, a, samples),
                             _validation_sites(candidate, a, samples)]) for a in ("u", "v")]
    distances = np.linalg.norm(_evaluate_tensor(original, u, v) - _evaluate_tensor(candidate, u, v), axis=2)
    if not np.isfinite(distances).all():
        raise ValueError("Candidate contains nonfinite geometric evaluations")
    return {"method": "Independent tensor evaluation at uniform sites and knot-span quarter points",
            "maximum_distance": float(distances.max()), "sample_count": int(distances.size),
            "certified_bound": False, "same_parameter_comparison": True}


def remove_low_curvature_knots(surface: Any, *, maximum_error: float,
                               validation_samples: int = 17,
                               maximum_attempts: int = 256,
                               acceptance_predicate: Callable[[Any, Any], bool] | None = None) -> MultiresolutionResult:
    """Remove only interior knots passing a cumulative geometric error guard.

    geomdl's operation mutates its argument and can remove a knot even when its
    internal tolerance test fails. Each trial therefore uses a fresh copy and
    is checked independently against the immutable ORIGINAL surface.
    """
    if not math.isfinite(maximum_error) or maximum_error < 0 or maximum_attempts < 0:
        raise ValueError("Require a nonnegative finite error budget and attempt limit")
    current = deepcopy(surface)
    attempts, accepted_count = [], 0
    while len(attempts) < maximum_attempts:
        changed = False
        for axis, name in enumerate(("u", "v")):
            degree = getattr(current, f"degree_{name}")
            knots = getattr(current, f"knotvector_{name}")
            interior = np.unique(knots[degree + 1:-degree - 1])
            for knot in interior:
                if len(attempts) >= maximum_attempts:
                    break
                trial = deepcopy(current)
                parameter, number = [None, None], [0, 0]
                parameter[axis], number[axis] = float(knot), 1
                record = {"axis": name, "knot": float(knot), "accepted": False}
                try:
                    operations.remove_knot(trial, parameter, number)
                    if getattr(trial, f"ctrlpts_size_{name}") != getattr(current, f"ctrlpts_size_{name}") - 1:
                        raise ValueError("Knot removal did not remove exactly one control row")
                    # geomdl 5.4.0's removal helper incorrectly retains a
                    # center pole even on an exactly inserted affine knot.
                    # Invert its independently constructed insertion operator
                    # instead of trusting those trial poles. This is inverse
                    # knot insertion in control space, not a surface refit.
                    count = getattr(trial, f"ctrlpts_size_{name}")
                    curve = BSpline.Curve()
                    curve.degree = degree
                    curve.ctrlpts = np.eye(count).tolist()
                    curve.knotvector = getattr(trial, f"knotvector_{name}")
                    operations.insert_knot(curve, [float(knot)], [1])
                    insertion = np.asarray(curve.ctrlpts)
                    if not np.allclose(curve.knotvector, getattr(current, f"knotvector_{name}"),
                                       rtol=0, atol=1e-14):
                        raise ValueError("Inverse insertion did not reproduce the original knots")
                    homogeneous = np.asarray(current.ctrlptsw if current.rational else current.ctrlpts)
                    homogeneous = homogeneous.reshape(*_shape(current), -1)
                    ordered = np.moveaxis(homogeneous, axis, 0)
                    solution, _, rank, _ = np.linalg.lstsq(insertion, ordered.reshape(len(ordered), -1), rcond=None)
                    if rank != count:
                        raise ValueError("Knot insertion operator is rank deficient")
                    solution = solution.reshape((count,) + ordered.shape[1:])
                    repaired = np.moveaxis(solution, 0, axis)
                    raw = np.asarray(trial.ctrlptsw if trial.rational else trial.ctrlpts)
                    record["inverse_insertion_pole_correction"] = float(np.max(np.abs(raw - repaired.reshape(raw.shape))))
                    trial.set_ctrlpts(repaired.reshape(-1, repaired.shape[-1]).tolist(), *_shape(trial))
                    error = sampled_surface_difference(surface, trial, samples=validation_samples)
                    record.update(error)
                    if error["maximum_distance"] <= maximum_error:
                        predicate_passed = acceptance_predicate is None or bool(acceptance_predicate(surface, trial))
                        record["acceptance_predicate_passed"] = predicate_passed
                        if predicate_passed:
                            current = trial
                            accepted_count += 1
                            changed = record["accepted"] = True
                except (ValueError, ZeroDivisionError, IndexError) as exc:
                    record["error"] = str(exc)
                attempts.append(record)
        if not changed:
            break
    original_count, final_count = math.prod(_shape(surface)), math.prod(_shape(current))
    return MultiresolutionResult(current, "knot_removal", report={
        "algorithm": 3, "method": "Interior knot removal via inverse insertion with independent cumulative geometry guard",
        "original_control_count": original_count, "candidate_control_count": final_count,
        "actual_compression_ratio": final_count / original_count,
        "removed_knot_count": accepted_count, "attempts": attempts,
        "maximum_error": maximum_error, "accepted": True,
        "geometry_validation": sampled_surface_difference(surface, current, samples=validation_samples),
        "attempt_budget_exhausted": len(attempts) >= maximum_attempts,
        "acceptance_predicate_extension": acceptance_predicate is not None,
        "boundary_restoration_extension": False,
    })


def apply_multiresolution_strategy(surface: Any, *, curvature: float | None = None,
                                  curvature_ck: float = .1, curvature_cp: float = 10.,
                                  compression_ratio: float = .5, layers: int = 1,
                                  layer_specs: Sequence[tuple[int, int, int, int]] | None = None,
                                  regeneration: str = "author_uniform",
                                  maximum_error: float | None = None,
                                  validation_samples: int = 17,
                                  maximum_knot_removal_attempts: int = 256,
                                  knot_removal_acceptance_predicate: Callable[[Any, Any], bool] | None = None) -> MultiresolutionResult:
    """Apply Algorithm 3; return an uncommitted candidate and explicit evidence.

    ``maximum_error`` is a model-unit geometric budget, an additional acceptance
    guard absent from the short reference demonstration. The low-curvature
    branch always has a guard (numerical exactness if no budget is supplied).
    A rejected hierarchy remains available as ``candidate``; check ``accepted``
    before applying it. Boundary restoration is deliberately left to the caller.
    """
    measured = principal_curvature(surface) if curvature is None else float(curvature)
    classification = classify_patch_curvature(measured, curvature_ck, curvature_cp)
    if maximum_error is not None and (not math.isfinite(maximum_error) or maximum_error < 0):
        raise ValueError("The geometric error budget must be finite and nonnegative")
    if classification == "preserve":
        result = MultiresolutionResult(deepcopy(surface), classification, report={
            "algorithm": 3, "method": "Preserve high-curvature original patch",
            "accepted": True, "actual_compression_ratio": 1., "boundary_restoration_extension": False})
    elif classification == "knot_removal":
        budget = maximum_error if maximum_error is not None else 1e-9
        result = remove_low_curvature_knots(surface, maximum_error=budget,
            validation_samples=validation_samples, maximum_attempts=maximum_knot_removal_attempts,
            acceptance_predicate=knot_removal_acceptance_predicate)
    else:
        result = build_control_hierarchy(surface, layers=layers, compression_ratio=compression_ratio,
            layer_specs=layer_specs, regeneration=regeneration)
        result.report["geometry_validation"] = sampled_surface_difference(surface, result.candidate,
                                                                           samples=validation_samples)
        if maximum_error is not None:
            result.report["accepted"] = result.report["geometry_validation"]["maximum_distance"] <= maximum_error
        result.report["maximum_error"] = maximum_error
    result.report.update(classification=classification, principal_curvature=measured,
        principal_curvature_supplied=curvature is not None,
        curvature_policy="Maximum absolute principal curvature at uniform and all knot-span quarter tensor sites",
        curvature_ck=curvature_ck, curvature_cp=curvature_cp)
    return result
