# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.
# Markus Zimmermann, Johannes Edler von Hoessle
# Computing solution spaces for robust design
# https://doi.org/10.1002/nme.4450

from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from .bayesian import good_fraction_lower_bound
from .monte_carlo import sample_and_classify
from .phase1 import BoxState
from .step_a import modification_step_a


def phase2_iter(
    state: BoxState,
    problem,
    dv_norm: np.ndarray,
    dv_norm_l: np.ndarray,
    reqL: np.ndarray,
    reqU: np.ndarray,
    parameters: Optional[np.ndarray],
    ind_parameters: np.ndarray,
    sample_size: int,
    target_good_fraction: float,
    confidence: float,
    weight: np.ndarray,
    phase2_max_iterations: int = 100,
    stop_callback: Optional[Callable] = None,
    callback: Optional[Callable] = None,
    label: str = "",
) -> None:
    """Phase-II loop. Updates ``state`` in place.

    Each iteration: sample inside current bounds -> classify -> Bayesian lower
    bound a_l -> step A if target not yet met. Sets
    ``state.phase2_target_reached = True`` when ``a_l >= target_good_fraction``.
    """
    n_dims = len(dv_norm)

    for _ in range(phase2_max_iterations):
        if stop_callback and stop_callback():
            return

        good, m, bad, x_samp, viol, y = sample_and_classify(
            problem, state.bounds, parameters, reqL, reqU,
            dv_norm, dv_norm_l, ind_parameters, sample_size, n_dims,
        )
        N = good.size
        good_frac = (m / N) if N else 0.0

        lower = good_fraction_lower_bound(m, N, confidence)

        state.samples = {
            "points": x_samp * dv_norm.reshape(-1, 1) + dv_norm_l.reshape(-1, 1),
            "is_good": good,
            "is_bad": bad,
            "qoi_values": y,
            "violation_idx": viol,
        }
        state.good_fraction = good_frac
        state.good_fraction_lower_bound = lower
        state.m = m
        state.N = N
        state.phase2_iter_count += 1
        it = state.phase2_iter_count

        if callback and (it == 1 or it % 5 == 0):
            callback(
                None, None,
                f"  {label} Phase 2 iter {it}: a={good_frac:.4f} a_l={lower:.4f}",
            )

        if lower >= target_good_fraction:
            if callback:
                callback(
                    None, None,
                    f"  {label} Phase 2 target met at iter {it}: a_l={lower:.4f}",
                )
            state.phase2_target_reached = True
            return

        if good.any() and bad.any():
            state.bounds, _ = modification_step_a(
                state.bounds, x_samp, good, bad, weight
            )
