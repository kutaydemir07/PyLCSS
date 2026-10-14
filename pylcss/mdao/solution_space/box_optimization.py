# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.
# Markus Zimmermann, Johannes Edler von Hoessle
# Computing solution spaces for robust design
# https://doi.org/10.1002/nme.4450

"""Phase-I/Phase-II optimization of one axis-aligned solution-space box."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .contracts import (
    EvaluatableProblem,
    FloatArray,
    IntArray,
    ProgressCallback,
    SampleBatch,
    StopCallback,
)
from .coordinates import VariableLayout
from .phase1 import make_box_state, make_point_box, run_phase_one
from .phase2 import run_phase_two
from .validation import (
    minimum_all_success_sample_size,
    validate_box_optimization_inputs,
)


@dataclass(frozen=True)
class BoxOptimizationResult:
    """Result of optimizing one axis-aligned solution-space box."""

    bounds: FloatArray
    good_fraction: float
    good_fraction_lower_bound: float
    successes: int
    sample_count: int
    samples: SampleBatch
    phase1_iters: int
    phase2_iters: int
    target_reached: bool
    cancelled: bool


def optimize_box(
    problem: EvaluatableProblem,
    x0: FloatArray,
    init_bounds: Optional[FloatArray],
    dsl: FloatArray,
    dsu: FloatArray,
    reqL: FloatArray,
    reqU: FloatArray,
    parameters: Optional[FloatArray],
    ind_parameters: IntArray,
    sample_size: int,
    growth_rate: float,
    target_good_fraction: float,
    confidence: float,
    phase1_max_iterations: int,
    phase2_max_iterations: int,
    phase1_convergence_tol: float,
    weight: Optional[FloatArray] = None,
    callback: Optional[ProgressCallback] = None,
    label: str = "",
    stop_callback: Optional[StopCallback] = None,
) -> BoxOptimizationResult:
    """Optimize active variables and restore fixed/uncertain coordinates in the result."""
    dsl = np.asarray(dsl, dtype=float)
    dsu = np.asarray(dsu, dtype=float)
    reqL = np.asarray(reqL, dtype=float)
    reqU = np.asarray(reqU, dtype=float)
    validate_box_optimization_inputs(
        x0=x0,
        init_bounds=init_bounds,
        dsl=dsl,
        dsu=dsu,
        reqL=reqL,
        reqU=reqU,
        sample_size=sample_size,
        growth_rate=growth_rate,
        target_good_fraction=target_good_fraction,
        confidence=confidence,
        phase1_max_iterations=phase1_max_iterations,
        phase2_max_iterations=phase2_max_iterations,
        phase1_convergence_tol=phase1_convergence_tol,
    )
    input_count = dsl.size
    if parameters is None or np.asarray(parameters).size == 0:
        layout = VariableLayout.from_bounds(dsl, dsu, None)
        input_indices = np.arange(input_count)
        if np.asarray(ind_parameters).size:
            raise ValueError("parameter indices require parameter bounds")
    else:
        layout = VariableLayout.from_parameters(parameters)
        declared = np.asarray(ind_parameters, dtype=int)
        if not np.array_equal(np.sort(declared), layout.parameter_indices):
            raise ValueError(
                "ind_parameters must identify every parameter exactly once"
            )
        if layout.parameters.shape[1] == input_count:
            input_indices = np.arange(input_count)
        elif layout.active_indices.size == input_count:
            input_indices = layout.active_indices.copy()
        else:
            raise ValueError(
                "design bounds must match either active or all model variables"
            )
        fixed = dsl == dsu
        layout.parameters[:, input_indices[fixed]] = np.vstack((dsl[fixed], dsu[fixed]))
    active = layout.active_mask[input_indices]
    lower = dsl[active]
    widths = dsu[active] - lower
    physical = (
        make_point_box(x0, dsl, dsu)
        if init_bounds is None
        else np.asarray(init_bounds, dtype=float).copy()
    )
    normalized = (physical[active] - lower[:, None]) / widths[:, None]
    if weight is None:
        weights = np.ones(lower.size)
    else:
        weights = np.asarray(weight, dtype=float)
        if weights.shape == (input_count,):
            weights = weights[active]
        elif weights.shape != (lower.size,):
            raise ValueError("weight must match active or supplied design variables")
        if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
            raise ValueError("weight values must be finite and non-negative")
    state = make_box_state(normalized, layout.parameters.shape[1], reqU.size)
    run_phase_one(
        state,
        problem,
        widths,
        lower,
        reqL,
        reqU,
        layout.parameters,
        layout.parameter_indices,
        sample_size,
        growth_rate,
        weights,
        phase1_convergence_tol,
        confidence=confidence,
        phase1_max_iterations=phase1_max_iterations,
        stop_callback=stop_callback,
        callback=callback,
        label=label,
    )
    validation_size = max(
        sample_size, minimum_all_success_sample_size(target_good_fraction, confidence)
    )
    run_phase_two(
        state,
        problem,
        widths,
        lower,
        reqL,
        reqU,
        layout.parameters,
        layout.parameter_indices,
        validation_size,
        target_good_fraction,
        confidence,
        weights,
        phase2_max_iterations=phase2_max_iterations,
        stop_callback=stop_callback,
        callback=callback,
        label=label,
    )
    active_bounds = state.bounds * widths[:, None] + lower[:, None]
    full_bounds = layout.expand_bounds(active_bounds)
    return BoxOptimizationResult(
        bounds=full_bounds[input_indices],
        good_fraction=state.good_fraction,
        good_fraction_lower_bound=state.good_fraction_lower_bound,
        successes=state.m,
        sample_count=state.N,
        samples=state.samples,
        phase1_iters=state.phase1_iter_count,
        phase2_iters=state.phase2_iter_count,
        target_reached=state.phase2_target_reached,
        cancelled=bool(stop_callback and stop_callback()),
    )


__all__ = ["BoxOptimizationResult", "optimize_box"]
