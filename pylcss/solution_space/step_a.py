# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.
# Markus Zimmermann, Johannes Edler von Hoessle
# Computing solution spaces for robust design
# https://doi.org/10.1002/nme.4450

from __future__ import annotations

from typing import Tuple

import numpy as np


def modification_step_a(
    candidate_box: np.ndarray,
    design_points: np.ndarray,
    good_designs: np.ndarray,
    bad_designs: np.ndarray,
    weight: np.ndarray,
) -> Tuple[np.ndarray, float]:
    """Vectorized step A.

    Args:
        candidate_box: (dim, 2) current box.
        design_points: (dim, N) MC sample points.
        good_designs:  (N,) boolean mask, True for samples satisfying QoIs.
        bad_designs:   (N,) boolean mask, complement of ``good_designs``.
        weight: (dim,) per-dim weight in the volume measure.

    Returns:
        ``(best_box, mu)`` - the trimmed box with the largest weighted log
        volume across all anchor candidates, and that volume (linear scale).
        Falls back to the input box with ``mu = 0.0`` when there are no good
        samples.
    """
    weight = np.asarray(weight, dtype=float)
    dim = candidate_box.shape[0]

    ind_a = np.where(good_designs)[0]
    ind_b = np.where(bad_designs)[0]
    num_a = ind_a.size
    num_b = ind_b.size

    if num_a == 0:
        return candidate_box, 0.0

    if num_b == 0:
        widths = candidate_box[:, 1] - candidate_box[:, 0]
        log_mu = float(np.sum(np.log(np.maximum(weight * widths, 1e-20))))
        return candidate_box, float(np.exp(log_mu)) if log_mu > -700 else 0.0

    samples_a = design_points[:, ind_a]  # (dim, num_a)
    samples_b = design_points[:, ind_b]  # (dim, num_b)

    # ----- Per-dim cost of "keep left of B" / "keep right of B" -----
    # cost_keep_left[d, j]  = #good points strictly > B_dj (lost if we cut to keep A < B)
    # cost_keep_right[d, j] = #good points strictly < B_dj (lost if we cut to keep A > B)
    cost_keep_left = np.empty((dim, num_b))
    cost_keep_right = np.empty((dim, num_b))
    for d in range(dim):
        sorted_a_d = np.sort(samples_a[d, :])
        # searchsorted side="right": index of insertion that places equal items LEFT
        idx_right = np.searchsorted(sorted_a_d, samples_b[d, :], side="right")
        idx_left = np.searchsorted(sorted_a_d, samples_b[d, :], side="left")
        cost_keep_left[d, :] = num_a - idx_right  # how many A > B
        cost_keep_right[d, :] = idx_left           # how many A < B

    weighted_cost_left = cost_keep_left * weight[:, np.newaxis]   # (dim, num_b)
    weighted_cost_right = cost_keep_right * weight[:, np.newaxis]  # (dim, num_b)

    # Relative epsilon so we don't include B itself on the kept side.
    range_vec = candidate_box[:, 1] - candidate_box[:, 0]
    eps_vec = np.maximum(range_vec, 1e-12) * 1e-7  # (dim,)

    active_dims = weight > 0
    if not np.any(active_dims):
        active_dims = np.ones(dim, dtype=bool)

    best_log_mu = -np.inf
    best_box = candidate_box.copy()

    # For each good sample (anchor), compute the tightest box that keeps the
    # anchor on the feasible side of every bad point, then pick max mu.
    a_exp = samples_a[:, :, np.newaxis]   # (dim, num_a, 1)
    b_exp = samples_b[:, np.newaxis, :]   # (dim, 1, num_b)
    mask_keep_left = a_exp < b_exp        # (dim, num_a, num_b)

    costs = np.where(
        mask_keep_left,
        weighted_cost_left[:, np.newaxis, :],
        weighted_cost_right[:, np.newaxis, :],
    )

    best_dims = np.argmin(costs, axis=0)  # (num_a, num_b)

    dim_idx = np.arange(dim)[:, np.newaxis, np.newaxis]
    is_best = dim_idx == best_dims[np.newaxis, :, :]  # (dim, num_a, num_b)

    keep_left_best  = is_best &  mask_keep_left
    keep_right_best = is_best & ~mask_keep_left

    cut_upper = b_exp - eps_vec[:, np.newaxis, np.newaxis]
    cut_lower = b_exp + eps_vec[:, np.newaxis, np.newaxis]

    min_cut_upper = np.min(np.where(keep_left_best,  cut_upper,  np.inf), axis=2)  # (dim, num_a)
    max_cut_lower = np.max(np.where(keep_right_best, cut_lower, -np.inf), axis=2)  # (dim, num_a)

    new_lb = np.maximum(candidate_box[:, 0:1], max_cut_lower)  # (dim, num_a)
    new_ub = np.minimum(candidate_box[:, 1:2], min_cut_upper)  # (dim, num_a)

    diffs = np.maximum(new_ub - new_lb, 1e-20)
    log_vols = np.sum(np.log(diffs[active_dims, :]), axis=0)   # (num_a,)

    best_anchor = int(np.argmax(log_vols))
    best_log_mu = float(log_vols[best_anchor])
    best_box = np.column_stack(
        (new_lb[:, best_anchor].copy(), new_ub[:, best_anchor].copy())
    )

    best_mu = float(np.exp(best_log_mu)) if best_log_mu > -700 else 0.0
    return best_box, best_mu


def step_a_vectorized(
    candidate_box: np.ndarray,
    design_points: np.ndarray,
    good_designs: np.ndarray,
    bad_designs: np.ndarray,
    dim: int,
    weight: np.ndarray,
) -> Tuple[np.ndarray, float]:
    """Backward-compatible Step-A API used by the existing PyLCSS solver."""
    _ = dim
    return modification_step_a(
        candidate_box,
        design_points,
        good_designs,
        bad_designs,
        weight,
    )
