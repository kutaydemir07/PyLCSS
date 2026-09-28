# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.
# Markus Zimmermann, Johannes Edler von Hoessle
# Computing solution spaces for robust design
# https://doi.org/10.1002/nme.4450

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.stats import qmc

from .contracts import BoolArray, EvaluatableProblem, FloatArray, IntArray


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
        ``(dim, n)`` array of design points. Empty array if ``n <= 0``.
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


def classify_good_bad(
    problem: EvaluatableProblem,
    x_design: FloatArray,
    parameters: Optional[FloatArray],
    ind_parameters: IntArray,
    reqL: FloatArray,
    reqU: FloatArray,
) -> tuple[FloatArray, BoolArray, BoolArray, FloatArray, IntArray]:
    """Evaluate the problem at design points and classify good vs. bad.

    Each sample is "good" iff every QoI satisfies ``[reqL, reqU]``. Columns
    identified by ``ind_parameters`` are sampled from their finite parameter
    bounds once per design column; NaN-bounded columns are design variables.

    Returns:
        y_qoi:           (num_qoi, N) QoI values.
        good_mask:       (N,) boolean, True iff all QoIs satisfy req bounds.
        bad_mask:        (N,) boolean, complement of good_mask.
        c_min:           (N,) worst constraint slack (negative -> infeasible).
        violation_idx:   (N,) index of the worst-failing constraint per sample.
    """
    x_design = np.asarray(x_design, dtype=float)
    reqL = np.asarray(reqL, dtype=float)
    reqU = np.asarray(reqU, dtype=float)
    if x_design.ndim != 2:
        raise ValueError("x_design must be a two-dimensional, column-oriented array")
    if not np.all(np.isfinite(x_design)):
        raise ValueError("x_design must contain only finite values")
    if reqL.ndim != 1 or reqU.shape != reqL.shape or reqL.size == 0:
        raise ValueError("reqL and reqU must be equally sized, non-empty vectors")
    if np.any(np.isnan(reqL)) or np.any(np.isnan(reqU)) or np.any(reqL > reqU):
        raise ValueError("requirement bounds must be ordered and must not contain NaN")

    n_design_rows, N = x_design.shape
    if N == 0:
        n_qoi = reqU.shape[0]
        return (
            np.zeros((n_qoi, 0)),
            np.zeros(0, dtype=bool),
            np.zeros(0, dtype=bool),
            np.zeros(0),
            np.zeros(0, dtype=int),
        )

    if parameters is None:
        if np.asarray(ind_parameters).size:
            raise ValueError("ind_parameters must be empty when parameters is None")
        total_vars = n_design_rows
        ind_p = np.array([], dtype=int)
        active_design = x_design
    else:
        parameters = np.asarray(parameters, dtype=float)
        if parameters.ndim != 2 or parameters.shape[0] != 2:
            raise ValueError("parameters must have shape (2, n_total_variables)")
        total_vars = parameters.shape[1]
        ind_p = np.asarray(ind_parameters, dtype=int)
        if ind_p.ndim != 1 or np.unique(ind_p).size != ind_p.size:
            raise ValueError(
                "ind_parameters must contain unique one-dimensional indices"
            )
        if np.any(ind_p < 0) or np.any(ind_p >= total_vars):
            raise ValueError("ind_parameters contains an out-of-range index")
        partial_nan = np.isnan(parameters[0]) != np.isnan(parameters[1])
        if np.any(partial_nan):
            raise ValueError(
                "each variable must have either two parameter bounds or two NaN values"
            )
        declared_parameters = np.flatnonzero(~np.isnan(parameters[0]))
        if not np.array_equal(np.sort(ind_p), declared_parameters):
            raise ValueError(
                "ind_parameters must identify every finite-bounded parameter column"
            )

        parameter_bounds = parameters[:, ind_p]
        if parameter_bounds.size and (
            not np.all(np.isfinite(parameter_bounds))
            or np.any(parameter_bounds[0] > parameter_bounds[1])
        ):
            raise ValueError("sampled parameter bounds must be finite and ordered")

    ind_dvs = np.setdiff1d(np.arange(total_vars), ind_p)
    if parameters is not None:
        if n_design_rows == ind_dvs.size:
            active_design = x_design
        elif n_design_rows == total_vars:
            # Multimodal boxes retain fixed coordinates for display. Ignore
            # those rows here because their authoritative values are in
            # ``parameters``.
            active_design = x_design[ind_dvs]
        else:
            raise ValueError(
                "x_design row count must match either the active design "
                f"variables ({ind_dvs.size}) or all variables ({total_vars})"
            )

    if len(ind_p) > 0:
        p_sampler = qmc.LatinHypercube(d=len(ind_p), seed=None)
        p_unit = p_sampler.random(n=N).T
        p_min = parameters[0, ind_p].reshape(-1, 1)
        p_max = parameters[1, ind_p].reshape(-1, 1)
        p_samp = p_min + (p_max - p_min) * p_unit
    else:
        p_samp = None

    x_full = np.zeros((total_vars, N))
    x_full[ind_dvs] = active_design
    if p_samp is not None:
        x_full[ind_p] = p_samp

    y = np.asarray(problem.evaluate_matrix(x_full), dtype=float)
    n_qoi = reqU.size
    if y.ndim == 1:
        if n_qoi == 1 and y.size == N:
            y = y.reshape(1, N)
        elif N == 1 and y.size == n_qoi:
            y = y.reshape(n_qoi, 1)
    if y.shape != (n_qoi, N):
        raise ValueError(
            f"problem.evaluate_matrix returned shape {y.shape}; expected ({n_qoi}, {N})"
        )

    ct = reqU.reshape(-1, 1) - y
    cd = y - reqL.reshape(-1, 1)
    c = np.vstack((ct, cd))
    finite_outputs = np.all(np.isfinite(y), axis=0)
    c_min = np.where(finite_outputs, np.min(c, axis=0), -np.inf)
    violation_idx = np.argmin(c, axis=0)

    good_mask = finite_outputs & (c_min >= -1e-12)
    bad_mask = ~good_mask
    return y, good_mask, bad_mask, c_min, violation_idx


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
    """Draw N LHS samples from a normalized box, classify into sets A and B.

    N samples are drawn from Omega, yielding set A
    (good designs, |A| = m) and set B (bad designs). Handles normalization
    between DV space and physical space internally.

    Args:
        dvbox:          (dim, 2) bounds in normalized DV space.
        dv_norm:        (dim,) scaling factors back to physical space.
        dv_norm_l:      (dim,) lower offsets for normalization.
        N:              number of samples.

    Returns:
        A, m, B, dv_sample (normalized), violation_idx, y_sample.
        Sorted by ascending feasibility margin (worst first).
    """
    if N <= 0:
        raise ValueError(f"Number of samples N must be positive, got {N}")
    if dim <= 0:
        raise ValueError(f"Dimension dim must be positive, got {dim}")
    if dvbox.shape != (dim, 2):
        raise ValueError(
            f"dvbox shape mismatch: expected ({dim}, 2), got {dvbox.shape}"
        )
    dv_norm = np.asarray(dv_norm, dtype=float)
    dv_norm_l = np.asarray(dv_norm_l, dtype=float)
    if dv_norm.shape != (dim,) or dv_norm_l.shape != (dim,):
        raise ValueError(f"normalization vectors must each contain {dim} values")
    if not np.all(np.isfinite(dv_norm)) or not np.all(np.isfinite(dv_norm_l)):
        raise ValueError("normalization vectors must contain only finite values")

    dv_sample_norm = draw_samples(dvbox, N)
    dv_sample_phys = dv_sample_norm * dv_norm.reshape(-1, 1) + dv_norm_l.reshape(-1, 1)

    y, good, _bad, c_min, viol = classify_good_bad(
        problem, dv_sample_phys, parameters, ind_parameters, reqL, reqU
    )

    # MMSS intentionally sorts by feasibility margin. Bad points are processed
    # worst-first in Step A, and the last feasible point is the most robust
    # available start for the PyLCSS compatibility solver.
    order = np.argsort(c_min)
    dv_sample_sorted = dv_sample_norm[:, order]
    viol_sorted = viol[order]
    y_sorted = y[:, order] if return_qoi else None
    good_sorted = good[order]
    bad_sorted = ~good_sorted
    m = int(np.sum(good_sorted))

    return good_sorted, m, bad_sorted, dv_sample_sorted, viol_sorted, y_sorted


def sample_box(
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
) -> tuple[BoolArray, int, BoolArray, FloatArray, IntArray, Optional[FloatArray]]:
    """Backward-compatible sampling API used by the existing PyLCSS solver."""
    return sample_and_classify(
        problem,
        dvbox,
        parameters,
        reqL,
        reqU,
        dv_norm,
        dv_norm_l,
        ind_parameters,
        N,
        dim,
        return_qoi=True,
    )


# Compatibility with the name used by the original implementation.
monte_carlo = sample_box


__all__ = [
    "classify_good_bad",
    "draw_samples",
    "monte_carlo",
    "sample_and_classify",
    "sample_box",
]
