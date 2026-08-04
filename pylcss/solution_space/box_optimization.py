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

    @property
    def m(self) -> int:
        """Compatibility alias for the number of successful samples."""
        return self.successes

    @property
    def N(self) -> int:
        """Compatibility alias for the total number of samples."""
        return self.sample_count


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
    """Compute a robust solution box from one feasible anchor point."""
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

    dimension_count = dsl.size
    design_widths = np.where(dsu > dsl, dsu - dsl, 1.0)
    if init_bounds is None:
        physical_bounds = make_point_box(x0, dsl, dsu)
    else:
        physical_bounds = np.asarray(init_bounds, dtype=float).copy()
    normalized_bounds = (physical_bounds - dsl[:, None]) / design_widths[:, None]

    if weight is None:
        weight_array = np.ones(dimension_count)
    else:
        weight_array = np.asarray(weight, dtype=float)
        if weight_array.shape != (dimension_count,):
            raise ValueError(f"weight must contain {dimension_count} values")
        if not np.all(np.isfinite(weight_array)) or np.any(weight_array < 0.0):
            raise ValueError("weight values must be finite and non-negative")

    state = make_box_state(
        bounds=normalized_bounds,
        n_dims=dimension_count,
        n_qoi=reqU.size,
    )
    run_phase_one(
        state,
        problem,
        design_widths,
        dsl,
        reqL,
        reqU,
        parameters,
        ind_parameters,
        sample_size,
        growth_rate,
        weight_array,
        phase1_convergence_tol,
        confidence=confidence,
        phase1_max_iterations=phase1_max_iterations,
        stop_callback=stop_callback,
        callback=callback,
        label=label,
    )

    phase2_sample_size = max(
        sample_size,
        minimum_all_success_sample_size(target_good_fraction, confidence),
    )
    run_phase_two(
        state,
        problem,
        design_widths,
        dsl,
        reqL,
        reqU,
        parameters,
        ind_parameters,
        phase2_sample_size,
        target_good_fraction,
        confidence,
        weight_array,
        phase2_max_iterations=phase2_max_iterations,
        stop_callback=stop_callback,
        callback=callback,
        label=label,
    )

    return BoxOptimizationResult(
        bounds=state.bounds * design_widths[:, None] + dsl[:, None],
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
