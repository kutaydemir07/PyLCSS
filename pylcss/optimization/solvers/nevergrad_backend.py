# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Nevergrad ask/tell adapter with fixed-variable and constraint support."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import logging
import os
from typing import Any
import warnings

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import NonlinearConstraint

from ..models import FloatArray
from .backend_types import SolverResult

logger = logging.getLogger(__name__)

PENALTY_VALUE = 1e9
_FINITE_BOUND_LIMIT = 1e20
_METAMODEL_ERRORS = (
    "only 0-dimensional arrays can be converted to Python scalars",
    "only size-1 arrays can be converted to Python scalars",
)


def solve_with_nevergrad(
    objective_func: Callable[[FloatArray], float],
    x0: ArrayLike,
    bounds: Sequence[tuple[float, float]],
    maxiter: int = 5000,
    constraints: Sequence[Mapping[str, Any] | NonlinearConstraint] | None = None,
    callback: Callable[[FloatArray], None] | None = None,
    **kwargs: Any,
) -> SolverResult:
    """Minimize an objective through Nevergrad's ask/tell interface."""
    try:
        import nevergrad as ng
    except ImportError:
        return SolverResult(
            x=np.asarray(x0, dtype=float),
            fun=float("inf"),
            message="Nevergrad is not installed.",
            success=False,
        )

    initial = np.asarray(x0, dtype=float)
    fixed_indices, active_indices, active_bounds = _partition_bounds(bounds)
    if not active_indices:
        try:
            value = float(objective_func(initial))
        except Exception as exc:
            return SolverResult(
                x=initial,
                fun=float("inf"),
                message=f"Fixed design evaluation failed: {exc}",
                success=False,
            )
        return SolverResult(
            x=initial,
            fun=value,
            message="All variables are fixed.",
            success=True,
        )

    lower = np.asarray([item[0] for item in active_bounds], dtype=float)
    upper = np.asarray([item[1] for item in active_bounds], dtype=float)
    active_initial = np.clip(initial[active_indices], lower, upper)
    requested_optimizer = str(kwargs.get("optimizer_name", "NGOpt"))
    worker_count = int(kwargs.get("num_workers", max(1, os.cpu_count() or 1)))
    seed = kwargs.get("seed")
    optimizer_names = [requested_optimizer]
    if requested_optimizer != "TwoPointsDE":
        optimizer_names.append("TwoPointsDE")

    last_error: Exception | None = None
    for optimizer_name in optimizer_names:
        try:
            parametrization = ng.p.Array(init=active_initial)
            parametrization.set_bounds(lower, upper)
            if seed is not None:
                parametrization.random_state.seed(int(seed))

            if optimizer_name not in ng.optimizers.registry:
                logger.warning(
                    "Nevergrad optimizer %s is unknown; using NGOpt.",
                    optimizer_name,
                )
                optimizer_name = "NGOpt"
            optimizer_class = ng.optimizers.registry[optimizer_name]
            optimizer = optimizer_class(
                parametrization=parametrization,
                budget=maxiter,
                num_workers=worker_count,
            )
            stopped = False

            for _ in range(maxiter):
                if optimizer.num_ask >= maxiter:
                    break
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    candidate = optimizer.ask()
                active_x = np.asarray(candidate.value, dtype=float)
                full_x = _expand_vector(
                    active_x,
                    bounds,
                    fixed_indices,
                    active_indices,
                )
                try:
                    loss = float(objective_func(full_x))
                    violations = _constraint_violations(full_x, constraints)
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", RuntimeWarning)
                        if violations:
                            optimizer.tell(candidate, loss, violations)
                        else:
                            optimizer.tell(candidate, loss)
                    if callback is not None:
                        callback(full_x)
                except StopIteration:
                    stopped = True
                    break
                except Exception as exc:
                    if _is_metamodel_error(exc):
                        raise
                    logger.exception(
                        "Nevergrad candidate evaluation failed at %s.",
                        full_x,
                    )
                    _tell_failed_candidate(optimizer, candidate, constraints)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                recommendation = optimizer.recommend()
            final_x = _expand_vector(
                np.asarray(recommendation.value, dtype=float),
                bounds,
                fixed_indices,
                active_indices,
            )
            recommendation_loss = getattr(recommendation, "loss", None)
            return SolverResult(
                x=final_x,
                fun=(
                    float(recommendation_loss)
                    if recommendation_loss is not None
                    else float("nan")
                ),
                message=(
                    f"{optimizer_name} {'stopped' if stopped else 'completed'} "
                    f"after {optimizer.num_ask} evaluations."
                ),
                success=not stopped,
            )
        except Exception as exc:
            last_error = exc
            if _is_metamodel_error(exc) and optimizer_name != "TwoPointsDE":
                logger.warning(
                    "Nevergrad %s hit a metamodel incompatibility; retrying "
                    "with TwoPointsDE.",
                    optimizer_name,
                )
                continue
            break

    return SolverResult(
        x=initial,
        fun=float("inf"),
        message=f"Nevergrad failed: {last_error}",
        success=False,
    )


def _partition_bounds(
    bounds: Sequence[tuple[float, float]],
) -> tuple[list[int], list[int], list[tuple[float, float]]]:
    fixed: list[int] = []
    active: list[int] = []
    active_bounds: list[tuple[float, float]] = []
    for index, (lower, upper) in enumerate(bounds):
        if np.isclose(lower, upper, atol=1e-9):
            fixed.append(index)
            continue
        active.append(index)
        active_bounds.append(
            (
                max(float(lower), -_FINITE_BOUND_LIMIT)
                if np.isfinite(lower)
                else -_FINITE_BOUND_LIMIT,
                min(float(upper), _FINITE_BOUND_LIMIT)
                if np.isfinite(upper)
                else _FINITE_BOUND_LIMIT,
            )
        )
    return fixed, active, active_bounds


def _expand_vector(
    active_x: FloatArray,
    bounds: Sequence[tuple[float, float]],
    fixed_indices: Sequence[int],
    active_indices: Sequence[int],
) -> FloatArray:
    full: FloatArray = np.zeros(len(bounds), dtype=float)
    full[list(active_indices)] = active_x
    for index in fixed_indices:
        full[index] = bounds[index][0]
    return full


def _constraint_violations(
    x: FloatArray,
    constraints: Sequence[Mapping[str, Any] | NonlinearConstraint] | None,
) -> list[float]:
    violations: list[float] = []
    for constraint in constraints or ():
        if isinstance(constraint, NonlinearConstraint):
            try:
                values = np.atleast_1d(np.asarray(constraint.fun(x), dtype=float))
                lower = np.broadcast_to(
                    np.asarray(constraint.lb, dtype=float),
                    values.shape,
                )
                upper = np.broadcast_to(
                    np.asarray(constraint.ub, dtype=float),
                    values.shape,
                )
            except Exception:
                violations.append(PENALTY_VALUE)
                continue
            for value, minimum, maximum in zip(values, lower, upper):
                if not np.isfinite(value):
                    violations.append(PENALTY_VALUE)
                elif value < minimum:
                    violations.append(float(minimum - value))
                elif value > maximum:
                    violations.append(float(value - maximum))
                else:
                    violations.append(0.0)
            continue

        function = constraint.get("fun")
        if not callable(function):
            continue
        try:
            values = np.atleast_1d(np.asarray(function(x), dtype=float))
        except Exception:
            violations.append(PENALTY_VALUE)
            continue
        constraint_type = constraint.get("type")
        for value in values:
            if not np.isfinite(value):
                violations.append(PENALTY_VALUE)
            elif constraint_type == "ineq":
                violations.append(max(0.0, -float(value)))
            elif constraint_type == "eq":
                violations.append(abs(float(value)))
    return violations


def _tell_failed_candidate(
    optimizer: Any,
    candidate: Any,
    constraints: Sequence[Mapping[str, Any] | NonlinearConstraint] | None,
) -> None:
    violation_count = max(1, len(constraints or ()))
    try:
        optimizer.tell(
            candidate,
            PENALTY_VALUE,
            [PENALTY_VALUE] * violation_count,
        )
    except Exception:
        optimizer.tell(candidate, PENALTY_VALUE)


def _is_metamodel_error(exc: Exception) -> bool:
    return any(message in str(exc) for message in _METAMODEL_ERRORS)


__all__ = ["solve_with_nevergrad"]
