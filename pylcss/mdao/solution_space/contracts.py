# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Shared typing contracts for the solution-space package."""

from __future__ import annotations

from typing import Callable, Protocol, TypedDict

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
IntArray = NDArray[np.int_]


class EvaluatableProblem(Protocol):
    """Minimal model interface required by the numerical algorithms."""

    def evaluate_matrix(self, values: FloatArray) -> FloatArray:
        """Evaluate all quantities of interest for column-oriented samples."""


class SampleBatch(TypedDict):
    """Evaluated samples shared by solvers and UI consumers."""

    points: FloatArray
    is_good: BoolArray
    is_bad: BoolArray
    qoi_values: FloatArray
    violation_idx: IntArray


ProgressCallback = Callable[[object | None, object | None, str], None]
StopCallback = Callable[[], bool]


__all__ = [
    "BoolArray",
    "EvaluatableProblem",
    "FloatArray",
    "IntArray",
    "ProgressCallback",
    "SampleBatch",
    "StopCallback",
]
