# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Sampling helpers used to redraw single- and multi-box visualizations."""

from __future__ import annotations

from typing import Iterable, Optional, Sequence

import numpy as np

from .contracts import EvaluatableProblem, FloatArray, SampleBatch
from .models import BoxSolutionSpace
from .sampling import sample_box

PlotPair = tuple[int, int]


def resample_solution_space(
    problem: EvaluatableProblem,
    bounds: FloatArray,
    design_lower: FloatArray,
    design_upper: FloatArray,
    requirement_upper: FloatArray,
    requirement_lower: FloatArray,
    parameters: Optional[FloatArray],
    sample_size: int = 1000,
    active_plots: Optional[Iterable[PlotPair]] = None,
    center_slice: bool = False,
) -> list[SampleBatch]:
    """Sample one solution box independently for each requested plot."""
    bounds = np.asarray(bounds, dtype=float)
    design_lower = np.asarray(design_lower, dtype=float)
    design_upper = np.asarray(design_upper, dtype=float)
    if bounds.ndim != 2 or bounds.shape[1] != 2:
        raise ValueError("bounds must have shape (n_design_variables, 2)")
    if (
        design_lower.shape != (bounds.shape[0],)
        or design_upper.shape != design_lower.shape
    ):
        raise ValueError("design bounds must match the solution-box dimensions")
    if np.any(design_lower > design_upper):
        raise ValueError("design bounds contain an inverted interval")
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")

    if parameters is None or np.asarray(parameters).size == 0:
        dimension_count = bounds.shape[0]
        parameter_indices = np.array([], dtype=int)
        parameter_bounds = np.full((2, dimension_count), np.nan)
    else:
        parameter_bounds = np.asarray(parameters, dtype=float)
        if parameter_bounds.ndim != 2 or parameter_bounds.shape[0] != 2:
            raise ValueError("parameters must have shape (2, n_total_variables)")
        is_design_variable = np.isnan(parameter_bounds[0])
        active_dimension_count = int(np.sum(is_design_variable))
        total_dimension_count = parameter_bounds.shape[1]
        parameter_indices = np.flatnonzero(~is_design_variable)
        if bounds.shape[0] not in {active_dimension_count, total_dimension_count}:
            raise ValueError(
                "bounds rows must match either the active design variables "
                f"({active_dimension_count}) or all variables ({total_dimension_count})"
            )
        dimension_count = bounds.shape[0]

    normalization = np.ones(dimension_count)
    offset = np.zeros(dimension_count)
    samples_by_plot: list[SampleBatch] = []

    for first_axis, second_axis in active_plots or ():
        if first_axis < 0 or second_axis < 0:
            raise ValueError("plot indices must not be negative")

        sample_bounds = bounds.copy()
        if first_axis < dimension_count and second_axis < dimension_count:
            if center_slice:
                _fix_other_dimensions_at_center(
                    sample_bounds, excluded={first_axis, second_axis}
                )
            sample_bounds[first_axis] = (
                design_lower[first_axis],
                design_upper[first_axis],
            )
            sample_bounds[second_axis] = (
                design_lower[second_axis],
                design_upper[second_axis],
            )
        else:
            design_axis = first_axis if first_axis < dimension_count else second_axis
            if design_axis >= dimension_count:
                raise ValueError("each plot must include at least one design variable")
            if center_slice:
                _fix_other_dimensions_at_center(sample_bounds, excluded={design_axis})

        good, _count, bad, points, violations, qoi = sample_box(
            problem,
            sample_bounds,
            parameter_bounds,
            requirement_lower,
            requirement_upper,
            normalization,
            offset,
            parameter_indices,
            sample_size,
            dimension_count,
        )
        assert qoi is not None
        samples_by_plot.append(
            {
                "points": points,
                "is_good": good,
                "is_bad": bad,
                "violation_idx": violations,
                "qoi_values": qoi,
            }
        )

    return samples_by_plot


def resample_multimodal_solution_spaces(
    problem: EvaluatableProblem,
    boxes: Sequence[BoxSolutionSpace | FloatArray],
    design_lower: FloatArray,
    design_upper: FloatArray,
    requirement_upper: FloatArray,
    requirement_lower: FloatArray,
    parameters: Optional[FloatArray],
    sample_size: int = 1000,
    active_plots: Optional[Iterable[PlotPair]] = None,
    center_slice: bool = False,
) -> list[SampleBatch]:
    """Resample several modes and merge their samples plot by plot."""
    modes = list(boxes)
    plots = list(active_plots or ())
    if not modes or not plots:
        return []
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")

    samples_per_box = max(1, int(np.ceil(sample_size / len(modes))))
    samples_by_plot: list[list[SampleBatch]] = [[] for _ in plots]

    for mode in modes:
        mode_bounds = mode.bounds if isinstance(mode, BoxSolutionSpace) else mode
        sampled = resample_solution_space(
            problem,
            np.asarray(mode_bounds, dtype=float),
            design_lower,
            design_upper,
            requirement_upper,
            requirement_lower,
            parameters,
            samples_per_box,
            active_plots=plots,
            center_slice=center_slice,
        )
        for plot_index, samples in enumerate(sampled):
            samples_by_plot[plot_index].append(samples)

    return [_merge_plot_samples(group) for group in samples_by_plot]


def _fix_other_dimensions_at_center(
    bounds: FloatArray,
    *,
    excluded: set[int],
) -> None:
    for dimension in range(bounds.shape[0]):
        if dimension not in excluded:
            center = float(np.mean(bounds[dimension]))
            bounds[dimension] = center


def _merge_plot_samples(samples: Sequence[SampleBatch]) -> SampleBatch:
    is_good = np.concatenate([batch["is_good"] for batch in samples])
    order = np.argsort(is_good.astype(np.int8), kind="stable")
    return {
        "points": np.hstack([batch["points"] for batch in samples])[:, order],
        "is_good": is_good[order],
        "is_bad": np.concatenate([batch["is_bad"] for batch in samples])[order],
        "violation_idx": np.concatenate([batch["violation_idx"] for batch in samples])[
            order
        ],
        "qoi_values": np.hstack([batch["qoi_values"] for batch in samples])[:, order],
    }


__all__ = ["resample_multimodal_solution_spaces", "resample_solution_space"]
