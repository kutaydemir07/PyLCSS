# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

from collections.abc import Mapping
from typing import Any

from .base import BaseSolver
from .global_opt import GlobalSolver
from .multi_start import MultiStartSolver
from .pareto import ParetoSolver
from .scipy_solver import ScipySolver
from ..methods import GLOBAL_METHODS, SCIPY_METHODS, SUPPORTED_METHODS


def get_solver(method: str, settings: Mapping[str, Any]) -> BaseSolver:
    """Create the solver registered for a user-facing method name."""
    solver_settings = dict(settings)

    if method in SCIPY_METHODS:
        return ScipySolver(solver_settings)

    if method in GLOBAL_METHODS:
        return GlobalSolver(solver_settings)

    if method == "NSGA-II":
        return ParetoSolver(solver_settings)

    if method == "Multi-Start":
        return MultiStartSolver(solver_settings)

    raise ValueError(
        f"Unknown optimization method {method!r}. "
        f"Supported methods: {', '.join(SUPPORTED_METHODS)}."
    )


__all__ = [
    "GLOBAL_METHODS",
    "SCIPY_METHODS",
    "SUPPORTED_METHODS",
    "get_solver",
]
