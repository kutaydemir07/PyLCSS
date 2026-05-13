# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.
# Markus Zimmermann, Johannes Edler von Hoessle
# Computing solution spaces for robust design
# https://doi.org/10.1002/nme.4450

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from scipy.stats import qmc


def draw_samples(bounds: np.ndarray, n: int) -> np.ndarray:
    """Latin Hypercube samples uniformly inside an axis-aligned box.

    Args:
        bounds: ``(dim, 2)`` array, columns are ``[lower, upper]``.
        n:      number of samples to draw.

    Returns:
        ``(dim, n)`` array of design points. Empty array if ``n <= 0``.
    """
    dim = bounds.shape[0]
    if n <= 0 or dim == 0:
        return np.zeros((dim, max(0, n)))
    sampler = qmc.LatinHypercube(d=dim, seed=None)
    u = sampler.random(n=n).T  # (dim, n)
    lo = bounds[:, 0:1]
    up = bounds[:, 1:2]
    return lo + (up - lo) * u


def classify_good_bad(
    problem,
    x_design: np.ndarray,
    parameters: Optional[np.ndarray],
    ind_parameters: np.ndarray,
    reqL: np.ndarray,
    reqU: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate the problem at design points and classify good vs. bad.

    Each sample is "good" iff every QoI satisfies ``[reqL, reqU]``. Uncertain
    parameters (NaN entries in ``parameters[0]``) are sampled once per design
    column.

    Returns:
        y_qoi:           (num_qoi, N) QoI values.
        good_mask:       (N,) boolean, True iff all QoIs satisfy req bounds.
        bad_mask:        (N,) boolean, complement of good_mask.
        c_min:           (N,) worst constraint slack (negative -> infeasible).
        violation_idx:   (N,) index of the worst-failing constraint per sample.
    """
    n_dims, N = x_design.shape
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
        total_vars = n_dims
        ind_p = np.array([], dtype=int)
    else:
        total_vars = parameters.shape[1]
        ind_p = ind_parameters

    ind_dvs = np.setdiff1d(np.arange(total_vars), ind_p)

    if len(ind_p) > 0:
        p_sampler = qmc.LatinHypercube(d=len(ind_p), seed=None)
        p_unit = p_sampler.random(n=N).T
        p_min = parameters[0, ind_p].reshape(-1, 1)
        p_max = parameters[1, ind_p].reshape(-1, 1)
        p_samp = p_min + (p_max - p_min) * p_unit
    else:
        p_samp = None

    x_full = np.zeros((total_vars, N))
    x_full[ind_dvs] = x_design
    if p_samp is not None:
        x_full[ind_p] = p_samp

    y = problem.evaluate_matrix(x_full)  # (num_qoi, N)

    ct = reqU.reshape(-1, 1) - y
    cd = y - reqL.reshape(-1, 1)
    c = np.vstack((ct, cd))
    c_min = np.min(c, axis=0)
    violation_idx = np.argmin(c, axis=0)

    good_mask = c_min >= -1e-12
    bad_mask = ~good_mask
    return y, good_mask, bad_mask, c_min, violation_idx


def sample_and_classify(
    problem,
    dvbox: np.ndarray,
    parameters: Optional[np.ndarray],
    reqL: np.ndarray,
    reqU: np.ndarray,
    dv_norm: np.ndarray,
    dv_norm_l: np.ndarray,
    ind_parameters: np.ndarray,
    N: int,
    dim: int,
    return_qoi: bool = True,
) -> Tuple[np.ndarray, int, np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
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
        raise ValueError(f"dvbox shape mismatch: expected ({dim}, 2), got {dvbox.shape}")

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


def monte_carlo(
    problem,
    dvbox: np.ndarray,
    parameters: Optional[np.ndarray],
    reqL: np.ndarray,
    reqU: np.ndarray,
    dv_norm: np.ndarray,
    dv_norm_l: np.ndarray,
    ind_parameters: np.ndarray,
    N: int,
    dim: int,
) -> Tuple[np.ndarray, int, np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
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
