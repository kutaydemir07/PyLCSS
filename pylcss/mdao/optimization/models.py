# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Domain models shared by optimization evaluators, solvers, and the UI."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

FloatArray: TypeAlias = NDArray[np.float64]
ParetoPoint: TypeAlias = dict[str, Any]


def _validate_name(name: str, kind: str) -> None:
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"{kind} name must be a non-empty string.")


@dataclass
class Variable:
    """A scalar design variable and its physical bounds."""

    name: str
    min_val: float
    max_val: float
    value: float = 0.0

    def __post_init__(self) -> None:
        _validate_name(self.name, "Variable")
        self.min_val = float(self.min_val)
        self.max_val = float(self.max_val)
        self.value = float(self.value)
        if np.isnan(self.min_val) or np.isnan(self.max_val):
            raise ValueError(f"Bounds for variable {self.name!r} cannot be NaN.")
        if self.min_val > self.max_val:
            raise ValueError(
                f"Variable {self.name!r} has lower bound greater than upper bound."
            )
        if not np.isfinite(self.value):
            raise ValueError(f"Value for variable {self.name!r} must be finite.")


@dataclass
class Objective:
    """A named model output selected as an optimization objective."""

    name: str
    weight: float = 1.0
    minimize: bool = True
    # Scalar solvers divide by this characteristic magnitude. ``None`` freezes
    # an automatic reference from the initial design.
    scale: float | None = None

    def __post_init__(self) -> None:
        _validate_name(self.name, "Objective")
        self.weight = float(self.weight)
        self.minimize = bool(self.minimize)
        if not np.isfinite(self.weight) or self.weight < 0.0:
            raise ValueError(
                f"Weight for objective {self.name!r} must be finite and non-negative."
            )
        if self.scale is not None:
            self.scale = float(self.scale)
            if not np.isfinite(self.scale) or self.scale <= 0.0:
                raise ValueError(
                    f"Scale for objective {self.name!r} must be finite and positive."
                )


@dataclass
class Constraint:
    """Admissible lower and upper bounds for a named model output."""

    name: str
    min_val: float = float("-inf")
    max_val: float = float("inf")

    def __post_init__(self) -> None:
        _validate_name(self.name, "Constraint")
        self.min_val = float(self.min_val)
        self.max_val = float(self.max_val)
        if np.isnan(self.min_val) or np.isnan(self.max_val):
            raise ValueError(f"Bounds for constraint {self.name!r} cannot be NaN.")
        if self.min_val > self.max_val:
            raise ValueError(
                f"Constraint {self.name!r} has minimum greater than maximum."
            )


@dataclass
class OptimizationResult:
    """Solver-independent optimization result in physical coordinates."""

    x: FloatArray
    cost: float
    objectives: dict[str, float]
    constraints: dict[str, float]
    max_violation: float
    message: str
    success: bool
    pareto_front: list[ParetoPoint] | None = None
    feasibility_tolerance: float = 1e-6
    converged: bool | None = None

    def __post_init__(self) -> None:
        self.x = np.asarray(self.x, dtype=float)
        self.cost = float(self.cost)
        self.objectives = {
            str(name): float(value) for name, value in self.objectives.items()
        }
        self.constraints = {
            str(name): float(value) for name, value in self.constraints.items()
        }
        self.max_violation = float(self.max_violation)
        self.message = str(self.message)
        self.success = bool(self.success)
        self.converged = (
            self.success if self.converged is None else bool(self.converged)
        )
        self.feasibility_tolerance = float(self.feasibility_tolerance)
        if (
            not np.isfinite(self.feasibility_tolerance)
            or self.feasibility_tolerance <= 0.0
        ):
            raise ValueError("Result feasibility tolerance must be positive.")


__all__ = [
    "Constraint",
    "FloatArray",
    "Objective",
    "OptimizationResult",
    "ParetoPoint",
    "Variable",
]
