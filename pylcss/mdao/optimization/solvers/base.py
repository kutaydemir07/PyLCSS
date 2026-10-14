# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from threading import Event
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import ArrayLike

from ..configuration import parse_feasibility_tolerance
from ..evaluator import EVALUATION_FAILURE, ModelEvaluator
from ..models import FloatArray, OptimizationResult

StepCallback: TypeAlias = Callable[
    [FloatArray, float, Mapping[str, float], float],
    None,
]


@dataclass
class Candidate:
    """Evaluated scalar-solver design, ranked against active constraints."""

    solver_x: FloatArray
    objective: float
    outputs: dict[str, Any]
    violation: float
    feasibility_tolerance: float

    @property
    def feasible(self) -> bool:
        return self.violation <= self.feasibility_tolerance

    @property
    def rank(self) -> tuple[int, float, float]:
        return (
            0 if self.feasible else 1,
            0.0 if self.feasible else self.violation,
            self.objective,
        )


def build_constraints(evaluator: ModelEvaluator) -> list[dict[str, Any]]:
    """Build scaled inequality residuals shared by the numerical solvers."""
    bounds = [
        (constraint.name, lower, upper, evaluator.constraint_solver_scale(index))
        for index, constraint in enumerate(evaluator.constraints)
        for lower, upper in [evaluator.constraint_solve_bounds(index)]
    ]
    residual_count = sum(
        int(np.isfinite(lower)) + int(np.isfinite(upper))
        for _, lower, upper, _ in bounds
    )
    if not residual_count:
        return []

    def residuals(x: ArrayLike) -> FloatArray:
        _, outputs, _ = evaluator.evaluate(x)
        if not evaluator.is_valid_result(outputs):
            return np.full(residual_count, -EVALUATION_FAILURE, dtype=float)
        result = []
        for name, lower, upper, scale in bounds:
            value = float(outputs[name])
            if np.isfinite(lower):
                result.append((value - lower) / scale)
            if np.isfinite(upper):
                result.append((upper - value) / scale)
        return np.asarray(result, dtype=float)

    return [{"type": "ineq", "fun": residuals}]


class BaseSolver(ABC):
    """Common cancellation and configuration interface for all solvers."""

    def __init__(self, settings: Mapping[str, Any]) -> None:
        self.settings = dict(settings)
        self._stop_event = Event()

    @property
    def stop_requested(self) -> bool:
        return self._stop_event.is_set()

    def stop(self) -> None:
        """Request cooperative cancellation at the next solver checkpoint."""
        self._stop_event.set()

    def _prepare_evaluator(self, evaluator: ModelEvaluator) -> float:
        """Synchronize the solver and evaluator feasibility contracts."""
        evaluator.set_feasibility_tolerance(
            parse_feasibility_tolerance(
                self.settings, default=evaluator.feasibility_tolerance
            )
        )
        return evaluator.feasibility_tolerance

    @abstractmethod
    def solve(
        self,
        evaluator: ModelEvaluator,
        x0: ArrayLike,
        callback: StepCallback | None = None,
    ) -> OptimizationResult:
        """Solve a configured problem from a physical-coordinate initial point."""
        raise NotImplementedError


__all__ = ["BaseSolver", "StepCallback"]
