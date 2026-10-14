# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Numerical adapters shared by optimization and solution-space analysis."""

from .nevergrad_backend import solve_with_nevergrad
from .results import SolverResult
from .scipy_backend import run_constrained_slsqp

__all__ = ["SolverResult", "run_constrained_slsqp", "solve_with_nevergrad"]
