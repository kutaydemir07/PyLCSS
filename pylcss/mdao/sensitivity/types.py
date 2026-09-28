# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Type contracts for the sensitivity-analysis package."""

from collections.abc import Sequence
from typing import Literal, TypeAlias, TypedDict

import numpy as np
from numpy.typing import NDArray

FloatArray: TypeAlias = NDArray[np.float64]
SensitivityMethod: TypeAlias = Literal["Sobol", "Morris", "FAST", "Delta"]


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
    s2_matrix: FloatArray
    calc_second_order: bool
    method: Literal["Sobol"]


class MorrisResult(TypedDict):
    """Morris elementary-effects estimates and ranking."""

    variable_names: list[str]
    mu: list[float]
    mu_star: list[float]
    sigma: list[float]
    mu_star_conf: list[float]
    important_variables: list[str]
    ranked_variables: list[str]
    selection_threshold: float | None
    method: Literal["Morris"]


class FastResult(TypedDict):
    """Extended FAST estimates and bootstrap confidence intervals."""

    variable_names: list[str]
    first_order: FloatArray
    total_order: FloatArray
    confidence_first: FloatArray
    confidence_total: FloatArray
    method: Literal["FAST"]


class DeltaResult(TypedDict):
    """Delta moment-independent and first-order estimates."""

    variable_names: list[str]
    delta: FloatArray
    delta_conf: FloatArray
    S1: FloatArray
    S1_conf: FloatArray
    method: Literal["Delta"]


class ConvergenceResult(TypedDict):
    """Sobol-index traces over increasing base sample sizes."""

    sample_sizes: list[int]
    variable_names: list[str]
    S1_traces: FloatArray
    ST_traces: FloatArray
    calc_second_order: bool
    seed: int
    method: Literal["Sobol"]


class PlotData(TypedDict):
    """Method-neutral values consumed by chart adapters."""

    variables: list[str]
    first_order: object
    total_order: object
    confidence: object
    output_name: str
    method: object


class AnalysisError(TypedDict):
    """Per-output failure returned by non-strict batch analysis."""

    error: str
    error_type: str
    method: SensitivityMethod


SensitivityResult: TypeAlias = SobolResult | MorrisResult | FastResult | DeltaResult
BatchResult: TypeAlias = dict[str, SensitivityResult | AnalysisError]


__all__ = [
    "AnalysisError",
    "BatchResult",
    "ConvergenceResult",
    "DeltaResult",
    "FastResult",
    "FloatArray",
    "MorrisResult",
    "PlotData",
    "ProblemDefinition",
    "SalibProblem",
    "SensitivityMethod",
    "SensitivityResult",
    "SobolResult",
]
