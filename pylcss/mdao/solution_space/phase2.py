# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.
# Markus Zimmermann, Johannes Edler von Hoessle
# Computing solution spaces for robust design
# https://doi.org/10.1002/nme.4450

from __future__ import annotations

from typing import Optional

from .contracts import (
    EvaluatableProblem,
    FloatArray,
    IntArray,
    ProgressCallback,
    StopCallback,
)
from .phase1 import BoxState
from .sampling import sample_box_batch
from .step_a import trim_box


def run_phase_two(
    state: BoxState,
    problem: EvaluatableProblem,
    dv_norm: FloatArray,
    dv_norm_l: FloatArray,
    reqL: FloatArray,
    reqU: FloatArray,
    parameters: Optional[FloatArray],
    ind_parameters: IntArray,
    sample_size: int,
    target_good_fraction: float,
    confidence: float,
    weight: FloatArray,
    phase2_max_iterations: int = 100,
    stop_callback: Optional[StopCallback] = None,
    callback: Optional[ProgressCallback] = None,
    label: str = "",
) -> None:
    """Validate the current box, trimming only when another validation is available."""
    state.phase2_target_reached = False
    for iteration in range(phase2_max_iterations):
        if stop_callback and stop_callback():
            return
        points, samples = sample_box_batch(
            problem,
            state.bounds,
            parameters,
            reqL,
            reqU,
            dv_norm,
            dv_norm_l,
            ind_parameters,
            sample_size,
        )
        state.record_samples(samples, confidence)
        state.phase2_iter_count += 1
        count = state.phase2_iter_count
        if callback and (count == 1 or count % 5 == 0):
            callback(
                None,
                None,
                f"  {label} Phase 2 iter {count}: "
                f"a={state.good_fraction:.4f} a_l={state.good_fraction_lower_bound:.4f}",
            )
        if state.good_fraction_lower_bound >= target_good_fraction:
            state.phase2_target_reached = True
            if callback:
                callback(
                    None,
                    None,
                    f"  {label} Phase 2 target met at iter {count}: "
                    f"a_l={state.good_fraction_lower_bound:.4f}",
                )
            return
        if not samples["is_good"].any():
            if callback:
                callback(None, None, f"  {label} Phase 2 stopped: no feasible samples.")
            return
        if samples["is_bad"].any() and iteration + 1 < phase2_max_iterations:
            state.bounds, _ = trim_box(
                state.bounds, points, samples["is_good"], samples["is_bad"], weight
            )
