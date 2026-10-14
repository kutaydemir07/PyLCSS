# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.
# Markus Zimmermann, Johannes Edler von Hoessle
# Computing solution spaces for robust design
# https://doi.org/10.1002/nme.4450

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.stats import qmc

from .contracts import BoolArray, EvaluatableProblem, FloatArray, IntArray, SampleBatch
from .coordinates import VariableLayout
from .samples import empty_samples


def draw_samples(
    bounds: FloatArray,
    n: int,
    *,
    seed: int | np.random.Generator | None = None,
) -> FloatArray:
    """Latin Hypercube samples uniformly inside an axis-aligned box.

    Args:
        bounds: ``(dim, 2)`` array, columns are ``[lower, upper]``.
        n:      number of samples to draw.

    Returns:
        ``(dim, n)`` array of design points. Empty array if ``n == 0``.
    """
    bounds = np.asarray(bounds, dtype=float)
    if bounds.ndim != 2 or bounds.shape[1] != 2:
        raise ValueError("bounds must have shape (n_dimensions, 2)")
    if not isinstance(n, (int, np.integer)) or isinstance(n, (bool, np.bool_)):
        raise TypeError("n must be an integer")
    if n < 0:
        raise ValueError("n must not be negative")
    if not np.all(np.isfinite(bounds)):
        raise ValueError("bounds must contain only finite values")
    if np.any(bounds[:, 0] > bounds[:, 1]):
        raise ValueError("bounds contains an inverted interval")

    dim = bounds.shape[0]
    if n == 0 or dim == 0:
        return np.zeros((dim, n))
    sampler = qmc.LatinHypercube(d=dim, seed=seed)
    u = sampler.random(n=n).T  # (dim, n)
    lo = bounds[:, 0:1]
    up = bounds[:, 1:2]
    return lo + (up - lo) * u


def evaluate_designs(
    problem: EvaluatableProblem,
    x_design: FloatArray,
    parameters: Optional[FloatArray],
    ind_parameters: IntArray,
    reqL: FloatArray,
    reqU: FloatArray,
) -> tuple[SampleBatch, FloatArray]:
    """Evaluate designs and retain the actual parameter realization per column."""
    x_design = np.asarray(x_design, dtype=float)
    reqL = np.asarray(reqL, dtype=float)
    reqU = np.asarray(reqU, dtype=float)
    if x_design.ndim != 2 or not np.all(np.isfinite(x_design)):
        raise ValueError("x_design must be a finite, column-oriented matrix")
    if reqL.ndim != 1 or reqU.shape != reqL.shape or reqL.size == 0:
        raise ValueError("reqL and reqU must be equally sized, non-empty vectors")
    if np.any(np.isnan(reqL)) or np.any(np.isnan(reqU)) or np.any(reqL > reqU):
        raise ValueError("requirement bounds must be ordered and must not contain NaN")
    layout = VariableLayout.from_parameters(
        np.full((2, x_design.shape[0]), np.nan) if parameters is None else parameters
    )
    indices = np.asarray(ind_parameters, dtype=int)
    if indices.ndim != 1 or not np.array_equal(
        np.sort(indices), layout.parameter_indices
    ):
        raise ValueError("ind_parameters must identify every parameter exactly once")
    active = layout.active_points(x_design)
    count = x_design.shape[1]
    if count == 0:
        return empty_samples(layout.parameters.shape[1], reqL.size), np.zeros(0)
    parameter_points = draw_samples(layout.parameters[:, indices].T, count)
    full = layout.expand_points(active, parameter_points)
    y = np.asarray(problem.evaluate_matrix(full), dtype=float)
    if y.ndim == 1:
        if reqL.size == 1 and y.size == count:
            y = y.reshape(1, count)
        elif count == 1 and y.size == reqL.size:
            y = y.reshape(reqL.size, 1)
    if y.shape != (reqL.size, count):
        raise ValueError(
            f"problem.evaluate_matrix returned shape {y.shape}; "
            f"expected ({reqL.size}, {count})"
        )
    slack = np.vstack((reqU[:, None] - y, y - reqL[:, None]))
    finite = np.all(np.isfinite(y), axis=0)
    margin = np.where(finite, np.min(slack, axis=0), -np.inf)
    good = finite & (margin >= -1e-12)
    return {
        "points": full,
        "qoi_values": y,
        "is_good": good,
        "is_bad": ~good,
        "violation_idx": np.argmin(slack, axis=0),
    }, margin


def classify_good_bad(
    problem: EvaluatableProblem,
    x_design: FloatArray,
    parameters: Optional[FloatArray],
    ind_parameters: IntArray,
    reqL: FloatArray,
    reqU: FloatArray,
) -> tuple[FloatArray, BoolArray, BoolArray, FloatArray, IntArray]:
    """Return QoIs, good/bad masks, minimum slack, and critical constraint indices."""
    batch, margin = evaluate_designs(
        problem, x_design, parameters, ind_parameters, reqL, reqU
    )
    return (
        batch["qoi_values"],
        batch["is_good"],
        batch["is_bad"],
        margin,
        batch["violation_idx"],
    )


def sample_box_batch(
    problem: EvaluatableProblem,
    dvbox: FloatArray,
    parameters: Optional[FloatArray],
    reqL: FloatArray,
    reqU: FloatArray,
    dv_norm: FloatArray,
    dv_norm_l: FloatArray,
    ind_parameters: IntArray,
    N: int,
) -> tuple[FloatArray, SampleBatch]:
    """Sample a normalized box; return normalized designs and full evaluated inputs.

    Both arrays are sorted by feasibility margin, worst first. Fixed-only
    problems are supported by a box with zero active rows.
    """
    dvbox = np.asarray(dvbox, dtype=float)
    if dvbox.ndim != 2 or dvbox.shape[1] != 2:
        raise ValueError("dvbox must have shape (n_dimensions, 2)")
    if N <= 0:
        raise ValueError("number of samples must be positive")
    dim = dvbox.shape[0]
    scale = np.asarray(dv_norm, dtype=float)
    offset = np.asarray(dv_norm_l, dtype=float)
    if scale.shape != (dim,) or offset.shape != (dim,):
        raise ValueError("normalization vectors must match the box dimensions")
    if not np.all(np.isfinite(scale)) or not np.all(np.isfinite(offset)):
        raise ValueError("normalization vectors must contain only finite values")
    normalized = draw_samples(dvbox, N)
    physical = normalized * scale[:, None] + offset[:, None]
    batch, margin = evaluate_designs(
        problem, physical, parameters, ind_parameters, reqL, reqU
    )
    order = np.argsort(margin)
    sorted_batch = {
        key: values[:, order] if values.ndim == 2 else values[order]
        for key, values in batch.items()
    }
    return normalized[:, order], sorted_batch


def sample_and_classify(
    problem: EvaluatableProblem,
    dvbox: FloatArray,
    parameters: Optional[FloatArray],
    reqL: FloatArray,
    reqU: FloatArray,
    dv_norm: FloatArray,
    dv_norm_l: FloatArray,
    ind_parameters: IntArray,
    N: int,
    dim: int,
    return_qoi: bool = True,
) -> tuple[BoolArray, int, BoolArray, FloatArray, IntArray, Optional[FloatArray]]:
    """Sample and classify a normalized box, sorted from worst to best margin."""
    if np.asarray(dvbox).shape != (dim, 2):
        raise ValueError(f"dvbox must have shape ({dim}, 2)")
    points, batch = sample_box_batch(
        problem, dvbox, parameters, reqL, reqU, dv_norm, dv_norm_l, ind_parameters, N
    )
    return (
        batch["is_good"],
        int(np.count_nonzero(batch["is_good"])),
        batch["is_bad"],
        points,
        batch["violation_idx"],
        batch["qoi_values"] if return_qoi else None,
    )


__all__ = [
    "classify_good_bad",
    "draw_samples",
    "evaluate_designs",
    "sample_and_classify",
    "sample_box_batch",
]
