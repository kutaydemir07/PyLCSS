# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Shared adapter for constrained SciPy searches."""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import minimize

from .results import FloatArray

logger = logging.getLogger(__name__)


def run_constrained_slsqp(
    objective_func: Callable[[FloatArray], float],
    constraints_func: Callable[[FloatArray], ArrayLike],
    x0: ArrayLike,
    bounds: Sequence[tuple[float, float]],
    maxiter: int = 5000,
) -> FloatArray:
    """Run a caller-supplied objective and inequality residuals with SLSQP."""
    initial = np.asarray(x0, dtype=float)
    try:
        result = minimize(
            objective_func,
            initial,
            method="SLSQP",
            bounds=bounds,
            constraints=[{"type": "ineq", "fun": constraints_func}],
            options={"maxiter": maxiter, "disp": False},
        )
    except Exception:
        logger.exception("Constrained SLSQP search failed.")
        return initial
    return np.asarray(result.x, dtype=float)


__all__ = [
    "run_constrained_slsqp",
]
