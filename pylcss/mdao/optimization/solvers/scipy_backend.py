# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Thin adapters for SciPy global and feasibility optimizers."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import logging
from typing import Any
import warnings

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import NonlinearConstraint, differential_evolution, minimize

from ..models import FloatArray
from .backend_types import SolverResult

logger = logging.getLogger(__name__)


def solve_with_differential_evolution(
    objective_func: Callable[[FloatArray], float],
    bounds: Sequence[tuple[float, float]],
    constraints: Sequence[Mapping[str, Any] | NonlinearConstraint] | None = None,
    maxiter: int = 5000,
    x0: ArrayLike | None = None,
    callback: Callable[[FloatArray], None] | None = None,
    **kwargs: Any,
) -> SolverResult:
    """Run SciPy Differential Evolution with normalized constraints."""
    scipy_constraints = tuple(_convert_constraints(constraints))

    def native_callback(
        x: FloatArray,
        convergence: float | None = None,
    ) -> bool:
        del convergence
        if callback is None:
            return False
        try:
            callback(np.asarray(x, dtype=float))
        except StopIteration:
            return True
        return False

    workers = int(kwargs.get("workers", 1))
    updating = kwargs.get(
        "updating",
        "deferred" if workers != 1 else "immediate",
    )
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="delta_grad == 0.0.*",
                category=UserWarning,
            )
            result = differential_evolution(
                objective_func,
                bounds,
                constraints=scipy_constraints,
                maxiter=maxiter,
                popsize=int(kwargs.get("popsize", 15)),
                mutation=kwargs.get("mutation", (0.5, 1.0)),
                recombination=float(kwargs.get("recombination", 0.7)),
                strategy=str(kwargs.get("strategy", "best1bin")),
                tol=float(kwargs.get("tol", 0.01)),
                atol=float(kwargs.get("atol", 0.0)),
                seed=kwargs.get("seed"),
                x0=None if x0 is None else np.asarray(x0, dtype=float),
                callback=native_callback,
                disp=bool(kwargs.get("disp", False)),
                polish=bool(kwargs.get("polish", True)),
                workers=workers,
                updating=updating,
            )
    except Exception as exc:
        logger.exception("Differential Evolution failed.")
        return SolverResult(
            x=None,
            fun=float("inf"),
            message=(f"Differential Evolution failed with {type(exc).__name__}: {exc}"),
            success=False,
        )
    return SolverResult(
        x=np.asarray(result.x, dtype=float),
        fun=float(result.fun),
        message=str(result.message),
        success=bool(result.success),
    )


def run_goal_attainment_slsqp(
    objective_func: Callable[[FloatArray], float],
    constraints_func: Callable[[FloatArray], ArrayLike],
    x0: ArrayLike,
    bounds: Sequence[tuple[float, float]],
    maxiter: int = 5000,
) -> FloatArray:
    """Find a feasible anchor for the solution-space workflow."""
    initial = np.asarray(x0, dtype=float)
    try:
        result = minimize(
            objective_func,
            initial,
            method="SLSQP",
            bounds=bounds,
            constraints=[{"type": "ineq", "fun": constraints_func}],
            options={"maxiter": maxiter, "disp": False},
        )
    except Exception:
        logger.exception("SLSQP goal-attainment search failed.")
        return initial
    return np.asarray(result.x, dtype=float)


def _convert_constraints(
    constraints: Sequence[Mapping[str, Any] | NonlinearConstraint] | None,
) -> list[NonlinearConstraint]:
    converted: list[NonlinearConstraint] = []
    for constraint in constraints or ():
        if isinstance(constraint, NonlinearConstraint):
            converted.append(constraint)
            continue
        function = constraint.get("fun")
        if not callable(function):
            continue
        if constraint.get("type") == "ineq":
            converted.append(NonlinearConstraint(function, 0.0, np.inf))
        elif constraint.get("type") == "eq":
            converted.append(NonlinearConstraint(function, 0.0, 0.0))
    return converted


__all__ = [
    "run_goal_attainment_slsqp",
    "solve_with_differential_evolution",
]
