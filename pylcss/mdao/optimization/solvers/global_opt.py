# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Global black-box optimization orchestration."""

from __future__ import annotations

import logging
import warnings
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from ...numerics import solve_with_nevergrad
from ..evaluator import ModelEvaluator
from ..models import FloatArray, OptimizationResult
from ..parsing import parse_boolean
from .base import BaseSolver, Candidate, StepCallback, build_constraints

logger = logging.getLogger(__name__)


class GlobalSolver(BaseSolver):
    """Run Nevergrad black-box global optimization."""

    def solve(
        self,
        evaluator: ModelEvaluator,
        x0: ArrayLike,
        callback: StepCallback | None = None,
    ) -> OptimizationResult:
        self._prepare_evaluator(evaluator)
        method = "Nevergrad"
        max_iterations = int(self.settings.get("maxiter", 1000))
        initial_physical = np.asarray(x0, dtype=float)
        lower = np.asarray(
            [variable.min_val for variable in evaluator.variables],
            dtype=float,
        )
        upper = np.asarray(
            [variable.max_val for variable in evaluator.variables],
            dtype=float,
        )
        if not (np.all(np.isfinite(lower)) and np.all(np.isfinite(upper))):
            return _failed_result(
                initial_physical,
                f"{method} requires finite lower and upper bounds on every variable.",
                evaluator.feasibility_tolerance,
            )

        original_scaling = evaluator.scaling
        use_scaling = parse_boolean(
            self.settings.get("scaling", evaluator.scaling),
            "Variable scaling",
        )
        evaluator.scaling = use_scaling
        try:
            bounds = (
                [
                    (0.0, 0.0) if variable.min_val == variable.max_val else (0.0, 1.0)
                    for variable in evaluator.variables
                ]
                if use_scaling
                else [
                    (variable.min_val, variable.max_val)
                    for variable in evaluator.variables
                ]
            )
            initial_solver = (
                evaluator.to_normalized(initial_physical)
                if use_scaling
                else initial_physical.copy()
            )
            evaluator.evaluate(initial_solver)
            return self._solve_in_current_coordinates(
                evaluator=evaluator,
                initial_physical=initial_physical,
                initial_solver=initial_solver,
                bounds=bounds,
                method=method,
                max_iterations=max_iterations,
                callback=callback,
            )
        finally:
            evaluator.scaling = original_scaling

    def _solve_in_current_coordinates(
        self,
        *,
        evaluator: ModelEvaluator,
        initial_physical: FloatArray,
        initial_solver: FloatArray,
        bounds: list[tuple[float, float]],
        method: str,
        max_iterations: int,
        callback: StepCallback | None,
    ) -> OptimizationResult:
        constraints = build_constraints(evaluator)
        best_feasible: Candidate | None = None
        best_any: Candidate | None = None
        best_displayed = float("inf")

        def remember(x: ArrayLike) -> Candidate | None:
            nonlocal best_any, best_feasible
            _, raw_results, violation = evaluator.evaluate(x)
            if not evaluator.is_valid_result(raw_results):
                return None
            candidate = Candidate(
                objective=evaluator.normalized_objective(raw_results),
                solver_x=np.asarray(x, dtype=float).copy(),
                outputs=dict(raw_results),
                violation=float(violation),
                feasibility_tolerance=evaluator.feasibility_tolerance,
            )
            if best_any is None or candidate.rank < best_any.rank:
                best_any = candidate
            if candidate.feasible and (
                best_feasible is None or candidate.objective < best_feasible.objective
            ):
                best_feasible = candidate
            return candidate

        def objective_wrapper(x: ArrayLike) -> float:
            if self.stop_requested:
                raise StopIteration
            candidate = remember(x)
            return (
                candidate.objective
                if candidate is not None
                else float(evaluator.evaluate(x)[0])
            )

        def callback_wrapper(x: ArrayLike) -> None:
            nonlocal best_displayed
            if self.stop_requested:
                raise StopIteration
            candidate = remember(x)
            if candidate is None or callback is None:
                return
            displayed = evaluator.displayed_objective(candidate.outputs)
            if candidate.feasible and displayed < best_displayed:
                best_displayed = displayed
                callback(
                    candidate.solver_x,
                    displayed,
                    candidate.outputs,
                    candidate.violation,
                )
            elif np.isinf(best_displayed):
                best_displayed = displayed
                callback(
                    candidate.solver_x,
                    displayed,
                    candidate.outputs,
                    candidate.violation,
                )

        # Freeze automatic objective scales at the documented initial design
        # and retain it as a safe fallback before invoking a backend.
        remember(initial_solver)

        backend_result: Any = None
        backend_error: Exception | None = None
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="overflow encountered in scalar multiply",
                    category=RuntimeWarning,
                )
                warnings.filterwarnings(
                    "ignore",
                    message=r"Bounds are .* sigma away from each other.*",
                    category=RuntimeWarning,
                )
                backend_result = solve_with_nevergrad(
                    objective_wrapper,
                    initial_solver,
                    bounds,
                    maxiter=max_iterations,
                    constraints=constraints,
                    callback=callback_wrapper,
                    optimizer_name=self.settings.get("optimizer_name", "NGOpt"),
                    seed=self.settings.get("seed"),
                )
        except Exception as exc:
            backend_error = exc
            logger.exception("%s backend failed.", method)

        final_candidate: Candidate | None = None
        backend_x = getattr(backend_result, "x", None)
        if backend_x is not None:
            final_candidate = remember(backend_x)

        chosen = final_candidate
        if best_feasible is not None and (
            chosen is None
            or not chosen.feasible
            or best_feasible.objective < chosen.objective - 1e-12
        ):
            chosen = best_feasible
        elif chosen is None and best_any is not None:
            chosen = best_any

        backend_message = str(
            getattr(backend_result, "message", f"{method} did not return a result.")
        )
        if backend_error is not None:
            backend_message = (
                f"{method} failed with {type(backend_error).__name__}: {backend_error}"
            )
        if chosen is None:
            return _failed_result(
                initial_physical,
                backend_message,
                evaluator.feasibility_tolerance,
            )

        if chosen is best_feasible and chosen is not final_candidate:
            backend_message += " Returned the best feasible evaluated design."
        elif chosen is best_any and chosen is not final_candidate:
            backend_message += " Returned the least-violating evaluated design."
        if self.stop_requested:
            backend_message += " Stopped by user."

        outputs = chosen.outputs
        backend_converged = bool(getattr(backend_result, "success", False))
        return OptimizationResult(
            x=evaluator.to_physical(chosen.solver_x),
            cost=evaluator.displayed_objective(outputs),
            objectives={
                objective.name: outputs[objective.name]
                for objective in evaluator.objectives
            },
            constraints={
                constraint.name: outputs[constraint.name]
                for constraint in evaluator.constraints
            },
            max_violation=chosen.violation,
            message=backend_message,
            success=(not self.stop_requested and chosen.feasible),
            feasibility_tolerance=evaluator.feasibility_tolerance,
            converged=backend_converged,
        )


def _failed_result(
    x: ArrayLike,
    message: str,
    feasibility_tolerance: float = 1e-6,
) -> OptimizationResult:
    return OptimizationResult(
        x=np.asarray(x, dtype=float),
        cost=float("inf"),
        objectives={},
        constraints={},
        max_violation=float("inf"),
        message=message,
        success=False,
        feasibility_tolerance=feasibility_tolerance,
        converged=False,
    )


__all__ = ["GlobalSolver"]
