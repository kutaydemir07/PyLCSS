# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Type contracts for the sensitivity-analysis package."""

from collections.abc import Sequence
from typing import Literal, TypeAlias, TypedDict

import numpy as np
from numpy.typing import NDArray

FloatArray: TypeAlias = NDArray[np.float64]


class ProblemDefinition(TypedDict):
    """Independent, uniformly distributed variables accepted by the public API."""

    names: Sequence[str]
    bounds: Sequence[Sequence[float]]


class SalibProblem(TypedDict):
    """Normalized problem mapping passed to SALib."""

    num_vars: int
    names: list[str]
    bounds: list[list[float]]


class SobolResult(TypedDict):
    """Sobol first-, total-, and optional second-order estimates."""

    variable_names: list[str]
    first_order: FloatArray
    total_order: FloatArray
    second_order: FloatArray | None
    confidence_first: FloatArray
    confidence_total: FloatArray
    confidence_second: FloatArray | None
    calc_second_order: bool
    method: Literal["Sobol"]


class ConvergenceResult(TypedDict):
    """Sobol-index traces over increasing base sample sizes."""

    sample_sizes: list[int]
    variable_names: list[str]
    S1_traces: FloatArray
    ST_traces: FloatArray
    calc_second_order: bool
    seed: int
    method: Literal["Sobol"]


class AnalysisError(TypedDict):
    """Per-output failure returned by non-strict batch analysis."""

    error: str
    error_type: str
    method: Literal["Sobol"]


BatchResult: TypeAlias = dict[str, SobolResult | AnalysisError]


__all__ = [
    "AnalysisError",
    "BatchResult",
    "ConvergenceResult",
    "FloatArray",
    "ProblemDefinition",
    "SalibProblem",
    "SobolResult",
]
