# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Public facade for third-party numerical backend adapters."""

from .backend_types import SolverResult
from .nevergrad_backend import solve_with_nevergrad
from .scipy_backend import (
    run_goal_attainment_slsqp,
    solve_with_differential_evolution,
)

__all__ = [
    "SolverResult",
    "run_goal_attainment_slsqp",
    "solve_with_differential_evolution",
    "solve_with_nevergrad",
]
