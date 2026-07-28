# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Small public entry points for solution-space analysis."""

from __future__ import annotations

from typing import Optional

from .contracts import EvaluatableProblem, FloatArray, ProgressCallback
from .solver import SolutionSpaceSolver, SolverResult


def compute_solution_space(
    problem: EvaluatableProblem,
    weight: FloatArray,
    design_lower: FloatArray,
    design_upper: FloatArray,
    search_lower: FloatArray,
    search_upper: FloatArray,
    requirement_upper: FloatArray,
    requirement_lower: FloatArray,
    parameters: Optional[FloatArray],
    sample_size: int = 1000,
    callback: Optional[ProgressCallback] = None,
    solver_type: str = "goal_attainment",
) -> SolverResult:
    """Compute one box-shaped solution space and its validation samples."""
    solver = SolutionSpaceSolver(
        problem,
        weight,
        design_lower,
        design_upper,
        search_lower,
        search_upper,
        requirement_upper,
        requirement_lower,
        parameters,
        solver_type=solver_type,
    )
    solver.final_sample_size = sample_size
    return solver.solve(callback=callback)


__all__ = ["compute_solution_space"]
