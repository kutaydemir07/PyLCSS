# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Global black-box optimization orchestration."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any
import warnings

import numpy as np
from numpy.typing import ArrayLike

from ..evaluator import ModelEvaluator
from ..models import FloatArray, OptimizationResult
from ..parsing import parse_boolean
from .backends import solve_with_differential_evolution, solve_with_nevergrad
from .base import BaseSolver, StepCallback

logger = logging.getLogger(__name__)


@dataclass
class _Candidate:
    objective: float
    solver_x: FloatArray
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


class GlobalSolver(BaseSolver):
    """Run Nevergrad or SciPy Differential Evolution."""

    def solve(
        self,
        evaluator: ModelEvaluator,
        x0: ArrayLike,
        callback: StepCallback | None = None,
    ) -> OptimizationResult:
        self._prepare_evaluator(evaluator)
        method = str(self.settings.get("method", "Nevergrad"))
        max_iterations = int(self.settings.get("maxiter", 1000))
        initial_physical = np.asarray(x0, dtype=float)
        if (
            method == "Differential Evolution"
            and int(self.settings.get("num_workers", self.settings.get("workers", 1)))
            != 1
        ):
            return _failed_result(
                initial_physical,
                "Differential Evolution requires one worker because in-process "
                "engineering model callbacks are not safely picklable.",
                evaluator.feasibility_tolerance,
            )
        lower = np.asarray(
            [variable.min_val for variable in evaluator.vars],
            dtype=float,
        )
        upper = np.asarray(
            [variable.max_val for variable in evaluator.vars],
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
            self.settings.get("scaling", False),
            "Variable scaling",
        )
        evaluator.scaling = use_scaling
        try:
            bounds = (
                [
                    (0.0, 0.0)
                    if abs(variable.max_val - variable.min_val) <= 1e-15
                    else (0.0, 1.0)
                    for variable in evaluator.vars
                ]
                if use_scaling
                else [
                    (variable.min_val, variable.max_val) for variable in evaluator.vars
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
        constraints = _build_constraints(evaluator)
        best_feasible: _Candidate | None = None
        best_any: _Candidate | None = None
        best_displayed = float("inf")

        def remember(x: ArrayLike) -> _Candidate | None:
            nonlocal best_any, best_feasible
            _, raw_results, violation = evaluator.evaluate(x)
            if not evaluator.is_valid_result(raw_results):
                return None
            candidate = _Candidate(
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
            if method == "Nevergrad":
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
                        num_workers=int(self.settings.get("num_workers", 1)),
                        seed=self.settings.get("seed"),
                    )
            elif method == "Differential Evolution":
                backend_result = solve_with_differential_evolution(
                    objective_wrapper,
                    bounds,
                    constraints=constraints,
                    maxiter=max_iterations,
                    x0=initial_solver,
                    callback=callback_wrapper,
                    **self._differential_evolution_options(),
                )
            else:
                raise ValueError(f"Unknown global method: {method}")
        except Exception as exc:
            backend_error = exc
            logger.exception("%s backend failed.", method)

        final_candidate: _Candidate | None = None
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
                objective.name: outputs[objective.name] for objective in evaluator.objs
            },
            constraints={
                constraint.name: outputs[constraint.name]
                for constraint in evaluator.cons
            },
            max_violation=chosen.violation,
            message=backend_message,
            success=(not self.stop_requested and chosen.feasible),
            feasibility_tolerance=evaluator.feasibility_tolerance,
            converged=backend_converged,
        )

    def _differential_evolution_options(self) -> dict[str, Any]:
        accepted = (
            "strategy",
            "popsize",
            "tol",
            "mutation",
            "recombination",
            "seed",
            "polish",
            "atol",
            "workers",
            "updating",
        )
        options = {key: self.settings[key] for key in accepted if key in self.settings}
        if "num_workers" in self.settings:
            options["workers"] = int(self.settings["num_workers"])
        if int(options.get("workers", 1)) != 1:
            options["updating"] = "deferred"
        return options


def _build_constraints(
    evaluator: ModelEvaluator,
) -> list[dict[str, Any]]:
    constraints: list[dict[str, Any]] = []
    for index, constraint in enumerate(evaluator.cons):
        lower, upper = evaluator.constraint_solve_bounds(index)

        def value(
            x: ArrayLike,
            output_name: str = constraint.name,
        ) -> float:
            _, raw_results, _ = evaluator.evaluate(x)
            if not evaluator.is_valid_result(raw_results):
                return -1e15
            return float(raw_results[output_name])

        scale = evaluator.constraint_solver_scale(index)
        if np.isfinite(lower):
            constraints.append(
                {
                    "type": "ineq",
                    "fun": (
                        lambda x, fun=value, bound=lower, divisor=scale:
                        (fun(x) - bound) / divisor
                    ),
                }
            )
        if np.isfinite(upper):
            constraints.append(
                {
                    "type": "ineq",
                    "fun": (
                        lambda x, fun=value, bound=upper, divisor=scale:
                        (bound - fun(x)) / divisor
                    ),
                }
            )
    return constraints


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
