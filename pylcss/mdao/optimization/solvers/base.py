# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from threading import Event
from typing import Any, TypeAlias

from numpy.typing import ArrayLike

from ..evaluator import ModelEvaluator
from ..models import FloatArray, OptimizationResult

StepCallback: TypeAlias = Callable[
    [FloatArray, float, Mapping[str, float], float],
    None,
]


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
        tolerance = self.settings.get(
            "feasibility_tol",
            self.settings.get("tol", evaluator.feasibility_tolerance),
        )
        evaluator.set_feasibility_tolerance(float(tolerance))
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
