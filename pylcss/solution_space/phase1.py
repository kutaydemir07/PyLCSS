# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.
# Markus Zimmermann, Johannes Edler von Hoessle
# Computing solution spaces for robust design
# https://doi.org/10.1002/nme.4450

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

import numpy as np

from .bayesian import good_fraction_lower_bound
from .monte_carlo import sample_and_classify
from .step_a import modification_step_a
from .step_b import modification_step_b


# ---------------------------------------------------------------------------
# Shared geometry helpers
# ---------------------------------------------------------------------------


def make_point_box(anchor: np.ndarray, dsl: np.ndarray, dsu: np.ndarray) -> np.ndarray:
    """Build a degenerate ``(n_dims, 2)`` box at ``anchor``, clipped to DS."""
    a = np.asarray(anchor, dtype=float).flatten()
    a = np.clip(a, dsl, dsu)
    return np.column_stack((a, a))


# ---------------------------------------------------------------------------
# Branch state
# ---------------------------------------------------------------------------


@dataclass
class BoxState:
    """Current candidate box Omega^z and sampling statistics for one branch.

    Passed into phase1_iter and phase2_iter, which update it in place each
    iteration. When phase1_done is True the box has converged in Phase I;
    when phase2_target_reached is True the Bayesian stopping criterion a_l >= a*
    has been met.
    """

    bounds: np.ndarray          # (n_dims, 2) current candidate box Omega^z
    samples: Dict[str, Any]     # last sample: points, is_good, is_bad, qoi_values, violation_idx

    # Last sample statistics
    good_fraction: float = 0.0             # empirical a_hat = m/N from last sample
    good_fraction_lower_bound: float = 0.0  # Bayesian lower bound a_l on true a
    m: int = 0                             # good designs in last sample
    N: int = 0                             # total designs in last sample

    # Termination flags
    phase1_done: bool = False
    phase2_target_reached: bool = False

    # Iteration counters
    phase1_iter_count: int = 0
    phase2_iter_count: int = 0


def _empty_samples(n_dims: int, n_qoi: int) -> Dict[str, Any]:
    return {
        "points": np.zeros((n_dims, 0)),
        "is_good": np.zeros(0, dtype=bool),
        "is_bad": np.zeros(0, dtype=bool),
        "qoi_values": np.zeros((n_qoi, 0)),
        "violation_idx": np.zeros(0, dtype=int),
    }


def make_box_state(
    bounds: np.ndarray,
    n_dims: int,
    n_qoi: int,
) -> BoxState:
    """Create an initial BoxState at the given bounds with empty samples."""
    return BoxState(
        bounds=np.asarray(bounds, dtype=float).copy(),
        samples=_empty_samples(n_dims, n_qoi),
    )


# ---------------------------------------------------------------------------
# Phase I kernel
# ---------------------------------------------------------------------------


def phase1_iter(
    state: BoxState,
    problem,
    dv_norm: np.ndarray,
    dv_norm_l: np.ndarray,
    reqL: np.ndarray,
    reqU: np.ndarray,
    parameters: Optional[np.ndarray],
    ind_parameters: np.ndarray,
    sample_size: int,
    growth_rate: float,
    weight: np.ndarray,
    phase1_convergence_tol: float,
    confidence: float = 0.95,
    phase1_max_iterations: int = 100,
    stop_callback: Optional[Callable] = None,
    callback: Optional[Callable] = None,
    label: str = "",
) -> None:
    """Phase-I expansion loop. Updates ``state`` in place.

    Each iteration:
        1. Step B - expand every boundary by ``g * DS-width``.
        2. sample_and_classify the expanded box.
        3. Adaptive g: if good_frac < 0.10, halve g and retry;
           if good_frac >= 0.80, grow g for the next iteration.
        4. Step A - trim to keep only good designs, maximize mu.
        5. Convergence: sets ``state.phase1_done = True`` when mu stagnates.
    """
    _FRAC_LOW  = 0.10
    _FRAC_HIGH = 0.80
    _MAX_RETRIES = 5

    n_dims = len(dv_norm)
    g = float(growth_rate)
    tol = float(phase1_convergence_tol)
    mu_vec: list = []

    dvbox = state.bounds.copy()

    for i in range(phase1_max_iterations):
        if stop_callback and stop_callback():
            break

        # Step B + adaptive retry
        m = 0
        x_samp_n = y = good = bad = viol = None
        dvbox_accepted = None

        for _attempt in range(_MAX_RETRIES):
            dvbox_new = modification_step_b(dvbox, np.zeros(n_dims), np.ones(n_dims), g)
            good, m, bad, x_samp_n, viol, y = sample_and_classify(
                problem, dvbox_new, parameters, reqL, reqU,
                dv_norm, dv_norm_l, ind_parameters, sample_size, n_dims,
            )
            good_frac = m / sample_size

            if good_frac >= _FRAC_LOW:
                dvbox_accepted = dvbox_new
                if good_frac >= _FRAC_HIGH:
                    g = min(g * 1.5, 0.3)
                break
            else:
                g = max(g * 0.5, 0.005)

        if m == 0 or dvbox_accepted is None:
            break

        dvbox = dvbox_accepted

        # Step A: trim - find box containing only good samples with max mu
        dvbox, mu = modification_step_a(dvbox, x_samp_n, good, bad, weight)
        mu_vec.append(mu)

        # The Step-A sample belongs to the expanded box. Evaluate the trimmed
        # candidate box with a fresh uniform sample.
        good_eval, m_eval, bad_eval, x_eval_n, viol_eval, y_eval = sample_and_classify(
            problem, dvbox, parameters, reqL, reqU,
            dv_norm, dv_norm_l, ind_parameters, sample_size, n_dims,
        )
        N_eval = int(good_eval.size)
        lower = good_fraction_lower_bound(m_eval, N_eval, confidence)

        # Update state
        state.bounds = dvbox
        state.samples = {
            "points": x_eval_n * dv_norm.reshape(-1, 1) + dv_norm_l.reshape(-1, 1),
            "is_good": good_eval,
            "is_bad": bad_eval,
            "qoi_values": y_eval,
            "violation_idx": viol_eval,
        }
        state.good_fraction = float(m_eval / max(N_eval, 1))
        state.good_fraction_lower_bound = lower
        state.m = m_eval
        state.N = N_eval
        state.phase1_iter_count += 1
        it = state.phase1_iter_count

        if callback and (it == 1 or it % 5 == 0):
            callback(
                None, None,
                f"  {label} Phase 1 iter {it}: "
                f"a={state.good_fraction:.3f} "
                f"a_l={state.good_fraction_lower_bound:.3f} "
                f"mu={mu:.6f} g={g:.4f}",
            )

        # Convergence: mu stagnated?
        if len(mu_vec) > 3:
            rel_change = (
                abs((mu_vec[-1] - mu_vec[-2]) / mu_vec[-1])
                if mu_vec[-1] != 0 else 0.0
            )
            if rel_change < tol:
                state.phase1_done = True
                if callback:
                    callback(
                        None, None,
                        f"  {label} Phase 1 converged ({it} iters).",
                    )
                return

    state.phase1_done = True
    if callback:
        callback(
            None, None,
            f"  {label} Phase 1 done ({state.phase1_iter_count} iters).",
        )
