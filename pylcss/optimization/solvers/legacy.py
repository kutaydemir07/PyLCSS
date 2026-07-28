# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Compatibility imports for the former solver backend module."""

from .backends import (
    SolverResult,
    run_goal_attainment_slsqp,
    solve_with_differential_evolution,
    solve_with_nevergrad,
)

__all__ = [
    "SolverResult",
    "run_goal_attainment_slsqp",
    "solve_with_differential_evolution",
    "solve_with_nevergrad",
]
