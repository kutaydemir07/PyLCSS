# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Presentation helpers for Sobol sensitivity results."""

import numpy as np
from numpy.typing import ArrayLike, NDArray


def second_order_matrix(second_order: ArrayLike) -> NDArray[np.float64]:
    """Mirror pairwise estimates, retaining NaN for unavailable/self effects."""
    raw = np.asarray(second_order, dtype=float)
    if raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
        raise ValueError("Second-order sensitivity estimates must be a square matrix.")

    first, second = np.triu_indices(raw.shape[0], k=1)
    upper = raw[first, second]
    lower = raw[second, first]
    pairs = np.where(np.isfinite(upper), upper, lower)
    pairs = np.where(np.isfinite(pairs), pairs, np.nan)

    matrix = np.full(raw.shape, np.nan, dtype=float)
    matrix[first, second] = pairs
    matrix[second, first] = pairs
    return matrix
