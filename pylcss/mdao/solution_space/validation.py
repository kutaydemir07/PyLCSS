# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Validation shared by solution-space numerical entry points."""

from __future__ import annotations

from numbers import Integral
from typing import Optional

import numpy as np

from .contracts import FloatArray


def minimum_all_success_sample_size(
    target_good_fraction: float,
    confidence: float,
) -> int:
    """Return the smallest all-success batch able to meet a Bayesian target."""
    if not 0.0 < target_good_fraction < 1.0:
        raise ValueError("target_good_fraction must be between 0 and 1")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1")
    return max(
        1,
        int(np.ceil(np.log1p(-confidence) / np.log(target_good_fraction))) - 1,
    )


def validate_box_optimization_inputs(
    *,
    x0: FloatArray,
    init_bounds: Optional[FloatArray],
    dsl: FloatArray,
    dsu: FloatArray,
    reqL: FloatArray,
    reqU: FloatArray,
    sample_size: int,
    growth_rate: float,
    target_good_fraction: float,
    confidence: float,
    phase1_max_iterations: int,
    phase2_max_iterations: int,
    phase1_convergence_tol: float,
) -> None:
    """Validate public numerical inputs before starting expensive work."""
    if dsl.ndim != 1 or dsu.shape != dsl.shape or dsl.size == 0:
        raise ValueError("dsl and dsu must be non-empty one-dimensional arrays")
    if not np.all(np.isfinite(dsl)) or not np.all(np.isfinite(dsu)):
        raise ValueError("design-space bounds must be finite")
    if np.any(dsl > dsu):
        raise ValueError(
            "each design-space lower bound must not exceed its upper bound"
        )

    anchor = np.asarray(x0, dtype=float)
    if anchor.shape != dsl.shape or not np.all(np.isfinite(anchor)):
        raise ValueError(f"x0 must contain {dsl.size} finite values")
    if np.any(anchor < dsl) or np.any(anchor > dsu):
        raise ValueError("x0 must lie inside the design space")

    if init_bounds is not None:
        bounds = np.asarray(init_bounds, dtype=float)
        if bounds.shape != (dsl.size, 2) or not np.all(np.isfinite(bounds)):
            raise ValueError(f"init_bounds must have shape ({dsl.size}, 2)")
        if np.any(bounds[:, 0] > bounds[:, 1]):
            raise ValueError("init_bounds contains an inverted interval")
        if np.any(bounds[:, 0] < dsl) or np.any(bounds[:, 1] > dsu):
            raise ValueError("init_bounds must lie inside the design space")

    if reqL.ndim != 1 or reqU.shape != reqL.shape or reqL.size == 0:
        raise ValueError("reqL and reqU must be equally sized, non-empty vectors")
    if np.any(np.isnan(reqL)) or np.any(np.isnan(reqU)) or np.any(reqL > reqU):
        raise ValueError("requirement bounds must be ordered and must not contain NaN")
    if not isinstance(sample_size, Integral) or isinstance(sample_size, bool):
        raise TypeError("sample_size must be an integer")
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    if not 0.0 < growth_rate <= 1.0:
        raise ValueError("growth_rate must be in (0, 1]")
    minimum_all_success_sample_size(target_good_fraction, confidence)
    if (
        not isinstance(phase1_max_iterations, Integral)
        or isinstance(phase1_max_iterations, bool)
        or not isinstance(phase2_max_iterations, Integral)
        or isinstance(phase2_max_iterations, bool)
    ):
        raise TypeError("phase iteration limits must be integers")
    if phase1_max_iterations < 0 or phase2_max_iterations < 0:
        raise ValueError("phase iteration limits must be non-negative")
    if phase1_convergence_tol < 0.0:
        raise ValueError("phase1_convergence_tol must be non-negative")


__all__ = [
    "minimum_all_success_sample_size",
    "validate_box_optimization_inputs",
]
