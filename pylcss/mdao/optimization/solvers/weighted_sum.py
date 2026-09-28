# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Weighted-sum Pareto approximation using repeated local solves."""

from __future__ import annotations

from collections.abc import Mapping
from itertools import product
import logging
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from ..evaluator import ModelEvaluator
from ..models import OptimizationResult
from .base import BaseSolver, StepCallback
from .scipy_solver import ScipySolver

logger = logging.getLogger(__name__)


class WeightedSumSolver(BaseSolver):
    """Explore scalarization weights and select a normalized compromise."""

    def __init__(self, settings: Mapping[str, Any]) -> None:
        super().__init__(settings)
        self._active_solver: BaseSolver | None = None

    def stop(self) -> None:
        super().stop()
        if self._active_solver is not None:
            self._active_solver.stop()

    def solve(
        self,
        evaluator: ModelEvaluator,
        x0: ArrayLike,
        callback: StepCallback | None = None,
    ) -> OptimizationResult:
        self._prepare_evaluator(evaluator)
        point_count = int(self.settings.get("pareto_points", 11))
        objective_count = len(evaluator.objs)
        weight_sets = (
            [np.array([1.0])]
            if objective_count < 2
            else _generate_weight_sets(objective_count, point_count)
        )
        results: list[OptimizationResult] = []
        original_weights = [objective.weight for objective in evaluator.objs]

        try:
            for run_index, weights in enumerate(weight_sets):
                if self.stop_requested:
                    break
                local_settings = dict(self.settings)
                local_settings["method"] = local_settings.get(
                    "ms_local_solver",
                    "SLSQP",
                )
                self._active_solver = ScipySolver(local_settings)
                for objective, weight in zip(evaluator.objs, weights):
                    objective.weight = float(weight)
                evaluator.clear_cache()
                try:
                    result = self._active_solver.solve(
                        evaluator,
                        x0,
                        callback,
                    )
                    if result.objectives:
                        results.append(result)
                except Exception:
                    logger.exception(
                        "Weighted-sum run %d failed.",
                        run_index + 1,
                    )
                finally:
                    self._active_solver = None
        finally:
            self._active_solver = None
            for objective, weight in zip(evaluator.objs, original_weights):
                objective.weight = weight
            evaluator.clear_cache()

        if not results:
            return OptimizationResult(
                x=np.asarray(x0, dtype=float),
                cost=float("inf"),
                objectives={},
                constraints={},
                max_violation=float("inf"),
                message="All weighted-sum runs failed.",
                success=False,
                feasibility_tolerance=evaluator.feasibility_tolerance,
                converged=False,
            )

        feasible = [
            result
            for result in results
            if result.max_violation <= result.feasibility_tolerance
        ]
        candidates = feasible or results
        signed_objectives = np.asarray(
            [
                [
                    (1.0 if objective.minimize else -1.0)
                    * float(result.objectives[objective.name])
                    for objective in evaluator.objs
                ]
                for result in candidates
            ],
            dtype=float,
        )
        ideal = np.min(signed_objectives, axis=0)
        spans = np.ptp(signed_objectives, axis=0)
        spans[spans < 1e-15] = 1.0
        compromise_index = int(
            np.argmin(
                np.linalg.norm(
                    (signed_objectives - ideal) / spans,
                    axis=1,
                )
            )
        )
        best = candidates[compromise_index]
        best.pareto_front = [
            {
                "x": np.asarray(result.x, dtype=float).tolist(),
                "objectives": dict(result.objectives),
                "constraints": dict(result.constraints),
                "max_violation": float(result.max_violation),
            }
            for result in feasible
        ]
        state = "stopped" if self.stop_requested else "completed"
        best.success = bool(best.success and not self.stop_requested)
        best.converged = bool(best.converged and not self.stop_requested)
        best.message += f" ({len(results)} weighted points explored; {state})"
        return best


def _generate_weight_sets(
    objective_count: int,
    point_count: int,
) -> list[np.ndarray]:
    if point_count < 1:
        raise ValueError("Pareto point count must be at least 1.")
    if objective_count == 2:
        return [
            np.array(
                [
                    index / (point_count - 1) if point_count > 1 else 0.5,
                    1.0 - (index / (point_count - 1) if point_count > 1 else 0.5),
                ]
            )
            for index in range(point_count)
        ]

    levels = np.linspace(0.0, 1.0, point_count)
    weights: list[np.ndarray] = []
    for combination in product(levels, repeat=objective_count - 1):
        if sum(combination) <= 1.0:
            weights.append(
                np.asarray(
                    [*combination, 1.0 - sum(combination)],
                    dtype=float,
                )
            )
        if len(weights) == point_count:
            break
    return weights


__all__ = ["WeightedSumSolver"]
