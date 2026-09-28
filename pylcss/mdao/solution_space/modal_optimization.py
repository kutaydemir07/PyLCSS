# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Parallel Stage-3 optimization of independently discovered modal boxes."""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import numpy as np

from .box_optimization import optimize_box
from .contracts import (
    EvaluatableProblem,
    FloatArray,
    IntArray,
    ProgressCallback,
    StopCallback,
)
from .models import BoxSolutionSpace, MMSSParameters


def optimize_modal_boxes(
    problem: EvaluatableProblem,
    boxes: list[BoxSolutionSpace],
    dsl: FloatArray,
    dsu: FloatArray,
    reqL: FloatArray,
    reqU: FloatArray,
    parameters: Optional[FloatArray],
    ind_parameters: IntArray,
    params: MMSSParameters,
    weight: Optional[FloatArray] = None,
    callback: Optional[ProgressCallback] = None,
    stop_callback: Optional[StopCallback] = None,
) -> list[BoxSolutionSpace]:
    """Optimize each mode and return only boxes that meet the reliability target."""
    if not boxes:
        return []

    sample_size = max(50, int(params.optimization_sample_size))
    phase2_max = int(params.phase2_max_iterations or params.max_iterations)
    worker_count = params.n_workers or max(1, (os.cpu_count() or 1) // 2)
    if callback:
        callback(
            None,
            None,
            "Stage 3 - Computation: grow and trim "
            f"{len(boxes)} modal solution spaces in parallel "
            f"(workers={min(worker_count, len(boxes))}).",
        )

    def optimize_one(index: int) -> tuple[int, bool]:
        if stop_callback and stop_callback():
            return index, False
        box = boxes[index]
        result = optimize_box(
            problem,
            x0=np.clip(box.bounds[:, 0], dsl, dsu),
            init_bounds=box.bounds,
            dsl=dsl,
            dsu=dsu,
            reqL=reqL,
            reqU=reqU,
            parameters=parameters,
            ind_parameters=ind_parameters,
            sample_size=sample_size,
            growth_rate=params.phase1_growth_rate,
            target_good_fraction=params.target_good_fraction,
            confidence=params.good_fraction_confidence,
            phase1_max_iterations=params.max_iterations,
            phase2_max_iterations=phase2_max,
            phase1_convergence_tol=params.phase1_convergence_tol,
            weight=weight,
            stop_callback=stop_callback,
            label=f"Mode {index + 1}",
        )
        box.bounds = result.bounds
        box.samples = result.samples
        box.good_fraction = result.good_fraction
        box.good_fraction_lower_bound = result.good_fraction_lower_bound
        box.validation_successes = result.successes
        box.validation_samples = result.sample_count
        return index, result.target_reached and not result.cancelled

    successful: set[int] = set()
    if worker_count <= 1 or len(boxes) == 1:
        for index in range(len(boxes)):
            completed_index, target_reached = optimize_one(index)
            if target_reached:
                successful.add(completed_index)
            _report_completion(callback, boxes, completed_index, target_reached)
    else:
        with ThreadPoolExecutor(max_workers=min(worker_count, len(boxes))) as executor:
            futures = {
                executor.submit(optimize_one, index): index
                for index in range(len(boxes))
            }
            for future in as_completed(futures):
                try:
                    completed_index, target_reached = future.result()
                except Exception as exc:
                    for pending in futures:
                        pending.cancel()
                    raise RuntimeError(
                        "Solution-space optimization failed for mode "
                        f"{futures[future] + 1}"
                    ) from exc
                if target_reached:
                    successful.add(completed_index)
                _report_completion(
                    callback,
                    boxes,
                    completed_index,
                    target_reached,
                )

    return [box for index, box in enumerate(boxes) if index in successful]


def _report_completion(
    callback: Optional[ProgressCallback],
    boxes: list[BoxSolutionSpace],
    index: int,
    target_reached: bool,
) -> None:
    if not callback:
        return
    outcome = "complete" if target_reached else "discarded (target not reached)"
    callback(
        None,
        None,
        f"  Mode {index + 1} {outcome}: "
        f"a_l={boxes[index].good_fraction_lower_bound:.4f}",
    )


__all__ = ["optimize_modal_boxes"]
