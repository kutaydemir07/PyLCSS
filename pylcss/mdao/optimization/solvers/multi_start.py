# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Multi-start orchestration for local optimization methods."""

from __future__ import annotations

import logging

import numpy as np
from numpy.typing import ArrayLike
from scipy.stats.qmc import LatinHypercube

from ..evaluator import ModelEvaluator
from ..models import OptimizationResult
from .base import BaseSolver, StepCallback

logger = logging.getLogger(__name__)


class MultiStartSolver(BaseSolver):
    """Run a local solver from Latin-hypercube starting designs."""

    def __init__(self, settings: dict[str, object]) -> None:
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
        # Local import avoids a factory -> multi-start -> factory import cycle.
        from .factory import SCIPY_METHODS, get_solver

        self._prepare_evaluator(evaluator)
        n_starts = int(self.settings.get("ms_n_starts", 10))
        if n_starts < 1:
            raise ValueError("Multi-start optimization needs at least one start.")

        local_method = str(self.settings.get("ms_local_solver", "SLSQP"))
        if local_method not in SCIPY_METHODS:
            raise ValueError(
                "Multi-start local solver must be one of: "
                + ", ".join(SCIPY_METHODS)
                + "."
            )

        initial = np.asarray(x0, dtype=float)
        n_variables = len(evaluator.vars)
        if initial.shape != (n_variables,):
            raise ValueError(
                f"Initial design must contain {n_variables} scalar values."
            )

        lower = np.asarray([item.min_val for item in evaluator.vars], dtype=float)
        upper = np.asarray([item.max_val for item in evaluator.vars], dtype=float)
        finite_box = np.isfinite(lower) & np.isfinite(upper)
        starts = [initial.copy()]
        if n_starts > 1:
            samples = LatinHypercube(
                d=n_variables,
                seed=self.settings.get("seed", 42),
            ).random(n=n_starts - 1)
            physical_samples = (
                evaluator.to_physical(samples)
                if evaluator.scaling and np.all(finite_box)
                else None
            )
            for sample_index, sample in enumerate(samples):
                point = initial.copy()
                if physical_samples is not None:
                    point[finite_box] = physical_samples[sample_index, finite_box]
                else:
                    point[finite_box] = lower[finite_box] + sample[finite_box] * (
                        upper[finite_box] - lower[finite_box]
                    )
                # Give unbounded dimensions distinct, deterministic nearby starts.
                point[~finite_box] += sample[~finite_box] - 0.5
                starts.append(point)

        local_settings = dict(self.settings)
        local_settings["method"] = local_method
        best_result: OptimizationResult | None = None
        completed_starts = 0

        try:
            for index, start in enumerate(starts):
                if self.stop_requested:
                    break

                try:
                    self._active_solver = get_solver(local_method, local_settings)
                    result = self._active_solver.solve(evaluator, start, callback)
                    completed_starts += 1
                    if best_result is None or _result_rank(result) < _result_rank(
                        best_result
                    ):
                        best_result = result
                except Exception:
                    logger.exception("Multi-start run %d failed.", index + 1)
                finally:
                    self._active_solver = None
        finally:
            self._active_solver = None

        if best_result is None:
            state = "stopped" if self.stop_requested else "failed"
            return OptimizationResult(
                x=initial,
                cost=float("inf"),
                objectives={},
                constraints={},
                max_violation=float("inf"),
                message=f"Multi-start {state} before any local run completed.",
                success=False,
                feasibility_tolerance=evaluator.feasibility_tolerance,
                converged=False,
            )

        state = "stopped" if self.stop_requested else "completed"
        best_result.success = bool(best_result.success and not self.stop_requested)
        best_result.converged = bool(
            best_result.converged and not self.stop_requested
        )
        best_result.message += (
            f" (best of {completed_starts} completed starts; {state})"
        )
        return best_result


def _result_rank(result: OptimizationResult) -> tuple[int, float, float]:
    """Rank feasible results first, then violation and displayed objective."""
    feasible = result.max_violation <= result.feasibility_tolerance
    return (
        0 if feasible else 1,
        0.0 if feasible else result.max_violation,
        result.cost,
    )


__all__ = ["MultiStartSolver"]
