# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Local constrained optimization through :func:`scipy.optimize.minimize`."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import minimize

from ..evaluator import ModelEvaluator
from ..methods import SCIPY_METHODS
from ..models import FloatArray, OptimizationResult
from .base import BaseSolver, Candidate, StepCallback, build_constraints


class ScipySolver(BaseSolver):
    """Solve a scalar problem with a supported local SciPy method."""

    def solve(
        self,
        evaluator: ModelEvaluator,
        x0: ArrayLike,
        callback: StepCallback | None = None,
    ) -> OptimizationResult:
        method = str(self.settings.get("method", "SLSQP"))
        if method not in SCIPY_METHODS:
            raise ValueError(
                f"Unsupported local solver {method!r}. Supported methods: "
                + ", ".join(SCIPY_METHODS)
                + "."
            )

        feasibility_tolerance = self._prepare_evaluator(evaluator)
        initial_physical = np.asarray(x0, dtype=float)
        solver_x0, bounds = _solver_coordinates(evaluator, initial_physical)
        # Freeze automatic objective scales at the documented initial design
        # before a numerical backend requests derivatives.
        evaluator.evaluate(solver_x0)
        constraints = build_constraints(evaluator)
        if all(lower == upper for lower, upper in bounds):
            _, raw_results, violation = evaluator.evaluate(solver_x0)
            if not evaluator.is_valid_result(raw_results):
                return _invalid_result(
                    evaluator,
                    solver_x0,
                    "The fixed design could not be evaluated.",
                )
            candidate = Candidate(
                solver_x=solver_x0,
                objective=evaluator.normalized_objective(raw_results),
                outputs=dict(raw_results),
                violation=float(violation),
                feasibility_tolerance=feasibility_tolerance,
            )
            return _candidate_result(
                evaluator,
                candidate,
                message="All design variables are fixed.",
                success=(not self.stop_requested and candidate.feasible),
                converged=not self.stop_requested,
            )

        best: Candidate | None = None

        def objective_function(x: ArrayLike) -> float:
            nonlocal best
            if self.stop_requested:
                raise StopIteration

            penalized_cost, raw_results, violation = evaluator.evaluate(x)
            if not evaluator.is_valid_result(raw_results):
                return penalized_cost

            objective = evaluator.normalized_objective(raw_results)
            candidate = Candidate(
                solver_x=np.asarray(x, dtype=float).copy(),
                objective=objective,
                outputs=dict(raw_results),
                violation=float(violation),
                feasibility_tolerance=feasibility_tolerance,
            )
            if best is None or candidate.rank < best.rank:
                best = candidate
            if callback is not None:
                callback(
                    candidate.solver_x,
                    evaluator.displayed_objective(raw_results),
                    raw_results,
                    candidate.violation,
                )
            return objective

        options: dict[str, Any] = {"maxiter": int(self.settings.get("maxiter", 1000))}
        tolerance = float(self.settings.get("tol", 1e-6))
        if method == "SLSQP":
            options["ftol"] = tolerance
            if "eps" in self.settings:
                options["eps"] = float(self.settings["eps"])

        backend_result: Any = None
        failure_message: str | None = None
        try:
            backend_result = minimize(
                objective_function,
                solver_x0,
                method=method,
                bounds=bounds,
                constraints=constraints,
                tol=tolerance,
                options=options,
            )
        except StopIteration:
            failure_message = "Stopped by user."
        except Exception as exc:
            failure_message = f"Solver failed with {type(exc).__name__}: {exc}"

        final: Candidate | None = None
        backend_x = getattr(backend_result, "x", None)
        if backend_x is not None:
            _, raw_results, violation = evaluator.evaluate(backend_x)
            if evaluator.is_valid_result(raw_results):
                final = Candidate(
                    solver_x=np.asarray(backend_x, dtype=float).copy(),
                    objective=evaluator.normalized_objective(raw_results),
                    outputs=dict(raw_results),
                    violation=float(violation),
                    feasibility_tolerance=feasibility_tolerance,
                )

        chosen = final
        if best is not None and (chosen is None or best.rank < chosen.rank):
            chosen = best
        if chosen is None:
            return _invalid_result(
                evaluator,
                solver_x0,
                failure_message
                or _backend_message(backend_result, "No valid design was evaluated."),
            )

        message = failure_message or _backend_message(backend_result, "Done.")
        if chosen is best and chosen is not final:
            if message and message[-1] not in ".!?":
                message += "."
            message += " Returned the best evaluated design."
        backend_converged = bool(getattr(backend_result, "success", False))
        return _candidate_result(
            evaluator,
            chosen,
            message=message,
            success=(not self.stop_requested and chosen.feasible),
            converged=backend_converged,
        )


def _solver_coordinates(
    evaluator: ModelEvaluator,
    initial_physical: FloatArray,
) -> tuple[FloatArray, list[tuple[float, float]]]:
    if evaluator.scaling:
        initial = evaluator.to_normalized(initial_physical)
        bounds = [
            (0.0, 0.0) if variable.min_val == variable.max_val else (0.0, 1.0)
            for variable in evaluator.variables
        ]
    else:
        initial = initial_physical.copy()
        bounds = [
            (variable.min_val, variable.max_val) for variable in evaluator.variables
        ]
    for index, (lower, upper) in enumerate(bounds):
        initial[index] = np.clip(initial[index], lower, upper)
    return initial, bounds


def _candidate_result(
    evaluator: ModelEvaluator,
    candidate: Candidate,
    *,
    message: str,
    success: bool,
    converged: bool,
) -> OptimizationResult:
    return OptimizationResult(
        x=evaluator.to_physical(candidate.solver_x),
        cost=evaluator.displayed_objective(candidate.outputs),
        objectives={
            objective.name: candidate.outputs[objective.name]
            for objective in evaluator.objectives
        },
        constraints={
            constraint.name: candidate.outputs[constraint.name]
            for constraint in evaluator.constraints
        },
        max_violation=candidate.violation,
        message=message,
        success=success,
        feasibility_tolerance=evaluator.feasibility_tolerance,
        converged=converged,
    )


def _invalid_result(
    evaluator: ModelEvaluator,
    solver_x: FloatArray,
    message: str,
) -> OptimizationResult:
    _, raw_results, violation = evaluator.evaluate(solver_x)
    error = evaluator.evaluation_error(raw_results)
    if error:
        message += f" {error}"
    return OptimizationResult(
        x=evaluator.to_physical(solver_x),
        cost=float("inf"),
        objectives={},
        constraints={},
        max_violation=violation,
        message=message,
        success=False,
        feasibility_tolerance=evaluator.feasibility_tolerance,
        converged=False,
    )


def _backend_message(result: Any, fallback: str) -> str:
    return str(getattr(result, "message", fallback))


__all__ = ["ScipySolver"]
