# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Public API for global sensitivity analysis."""

from .analyzer import SensitivityAnalyzer
from .types import (
    BatchResult,
    ConvergenceResult,
    DeltaResult,
    FastResult,
    MorrisResult,
    PlotData,
    ProblemDefinition,
    SensitivityMethod,
    SensitivityResult,
    SobolResult,
)

__all__ = [
    "BatchResult",
    "ConvergenceResult",
    "DeltaResult",
    "FastResult",
    "MorrisResult",
    "PlotData",
    "ProblemDefinition",
    "SensitivityAnalyzer",
    "SensitivityMethod",
    "SensitivityResult",
    "SobolResult",
]
