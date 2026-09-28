# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Product-family solution spaces and platform commonality metrics."""

from __future__ import annotations

import copy
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Mapping, Optional

import numpy as np

from .contracts import EvaluatableProblem, FloatArray, StopCallback
from .solver import SolutionSpaceSolver

logger = logging.getLogger(__name__)

FamilyProgressCallback = Callable[[str, int, int, str], None]
FamilyResults = dict[str, object]


def compute_product_family_solutions(
    problem: EvaluatableProblem,
    weight: FloatArray,
    design_lower: FloatArray,
    design_upper: FloatArray,
    search_lower: FloatArray,
    search_upper: FloatArray,
    requirement_upper: FloatArray,
    requirement_lower: FloatArray,
    parameters: Optional[FloatArray],
    solver_type: str,
    progress_callback: Optional[FamilyProgressCallback] = None,
    stop_callback: Optional[StopCallback] = None,
) -> FamilyResults:
    """Compute variant boxes and their common platform intersection."""
    variants = getattr(problem, "requirement_sets", {})
    quantities = getattr(problem, "quantities_of_interest", ())
    tasks = [
        (
            name,
            *_requirements_for_variant(quantities, overrides),
        )
        for name, overrides in variants.items()
    ]
    total_steps = len(tasks) + 1
    results: FamilyResults = {}

    def solve_variant(name: str, lower: FloatArray, upper: FloatArray) -> FloatArray:
        variant_problem = copy.deepcopy(problem)
        solver = SolutionSpaceSolver(
            variant_problem,
            weight,
            design_lower,
            design_upper,
            search_lower,
            search_upper,
            upper,
            lower,
            parameters,
            solver_type=solver_type,
        )
        box, success, _elapsed, _samples = solver.solve()
        if not success:
            raise RuntimeError(f"No feasible solution space found for variant {name!r}")
        return box

    if stop_callback is not None:
        for completed, (name, lower, upper) in enumerate(tasks, start=1):
            if stop_callback():
                return results
            if progress_callback:
                progress_callback(name, completed, total_steps, f"Starting {name}")
            results[name] = solve_variant(name, lower, upper)
    elif tasks:
        workers = min(len(tasks), max(1, (os.cpu_count() or 1) // 2))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(solve_variant, name, lower, upper): name
                for name, lower, upper in tasks
            }
            for completed, future in enumerate(as_completed(futures), start=1):
                name = futures[future]
                try:
                    results[name] = future.result()
                except Exception as exc:
                    for pending in futures:
                        pending.cancel()
                    raise RuntimeError(
                        f"Product-family solve failed for variant {name!r}"
                    ) from exc
                if progress_callback:
                    progress_callback(name, completed, total_steps, f"Completed {name}")

    if stop_callback and stop_callback():
        return results
    if progress_callback:
        progress_callback("Platform", total_steps, total_steps, "Calculating platform")
    _add_platform_results(results)
    return results


def calculate_variable_communality(
    variant_boxes: list[FloatArray],
    platform_box: Optional[FloatArray],
) -> Optional[FloatArray]:
    """Return common-intersection width divided by total-family width per variable."""
    if not variant_boxes or platform_box is None:
        return None

    platform = np.asarray(platform_box, dtype=float)
    variants = np.asarray(variant_boxes, dtype=float)
    if platform.ndim != 2 or platform.shape[1] != 2:
        raise ValueError("platform_box must have shape (n_variables, 2)")
    if variants.ndim != 3 or variants.shape[1:] != platform.shape:
        raise ValueError("all variant boxes must have the same shape as platform_box")

    variant_lower = variants[:, :, 0]
    variant_upper = variants[:, :, 1]
    total_width = np.max(variant_upper, axis=0) - np.min(variant_lower, axis=0)
    platform_width = platform[:, 1] - platform[:, 0]
    values = np.zeros(platform.shape[0], dtype=float)

    positive_platform = platform_width > 0.0
    values[positive_platform] = np.divide(
        platform_width[positive_platform],
        total_width[positive_platform],
        out=np.ones(np.count_nonzero(positive_platform), dtype=float),
        where=total_width[positive_platform] > 0.0,
    )

    fixed_platform = np.isclose(platform_width, 0.0)
    if np.any(fixed_platform):
        target = platform[fixed_platform, 0][None, :]
        all_agree = np.all(
            np.isclose(variant_lower[:, fixed_platform], target, atol=1e-10)
            & np.isclose(variant_upper[:, fixed_platform], target, atol=1e-10),
            axis=0,
        )
        values[fixed_platform] = all_agree.astype(float)

    return np.clip(values, 0.0, 1.0)


def _requirements_for_variant(
    quantities: object,
    overrides: Mapping[str, Mapping[str, float]],
) -> tuple[FloatArray, FloatArray]:
    lower: list[float] = []
    upper: list[float] = []
    for quantity in quantities:
        name = quantity["name"]
        override = overrides.get(name, {})
        lower.append(float(override.get("req_min", quantity["min"])))
        upper.append(float(override.get("req_max", quantity["max"])))
    return np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)


def _add_platform_results(results: FamilyResults) -> None:
    boxes = [
        np.asarray(box, dtype=float)
        for box in results.values()
        if isinstance(box, np.ndarray)
    ]
    if not boxes:
        results["Platform"] = None
        results["Platform_Infeasible"] = True
        results["Communality"] = None
        return

    first_shape = boxes[0].shape
    if first_shape[1:] != (2,) or any(box.shape != first_shape for box in boxes):
        raise ValueError("variant solution boxes must all have shape (n_variables, 2)")

    platform = np.column_stack(
        (
            np.maximum.reduce([box[:, 0] for box in boxes]),
            np.minimum.reduce([box[:, 1] for box in boxes]),
        )
    )
    results["Platform"] = platform
    results["Platform_Infeasible"] = bool(np.any(platform[:, 0] > platform[:, 1]))
    results["Communality"] = calculate_variable_communality(boxes, platform)


__all__ = [
    "FamilyProgressCallback",
    "FamilyResults",
    "calculate_variable_communality",
    "compute_product_family_solutions",
]
