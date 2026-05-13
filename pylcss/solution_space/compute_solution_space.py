# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.
# Markus Zimmermann, Johannes Edler von Hoessle
# Computing solution spaces for robust design
# https://doi.org/10.1002/nme.4450

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

import numpy as np

from .phase1 import make_box_state, make_point_box, phase1_iter
from .phase2 import phase2_iter


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass
class SolutionSpaceResult:
    """Output of compute_solution_space."""

    bounds: np.ndarray                # (n_dims, 2) final candidate box
    good_fraction: float              # m/N from the last MC sample
    good_fraction_lower_bound: float  # a_l
    m: int                            # good designs on the last MC sample
    N: int                            # total designs on the last MC sample
    samples: Dict[str, Any]           # last MC: points, is_good, is_bad, qoi_values, violation_idx
    phase1_iters: int
    phase2_iters: int


# ---------------------------------------------------------------------------
# Algorithm entry point
# ---------------------------------------------------------------------------


def compute_solution_space(
    problem,
    x0: np.ndarray,
    init_bounds: Optional[np.ndarray],
    dsl: np.ndarray,
    dsu: np.ndarray,
    reqL: np.ndarray,
    reqU: np.ndarray,
    parameters: Optional[np.ndarray],
    ind_parameters: np.ndarray,
    sample_size: int,
    growth_rate: float,
    target_good_fraction: float,
    confidence: float,
    phase1_max_iterations: int,
    phase2_max_iterations: int,
    phase1_convergence_tol: float,
    weight: Optional[np.ndarray] = None,
    callback: Optional[Callable] = None,
    label: str = "",
    stop_callback: Optional[Callable] = None,
) -> SolutionSpaceResult:
    """Compute a solution box from one anchor point.

    Phase I: alternate step A (trim) and step B (grow) until the log-volume
    measure mu stagnates. Phase II: apply only step A until the Bayesian lower
    bound on the fraction of good designs a_l reaches the target a*.
    """
    n_dims = len(dsl)
    ds_widths = np.where(dsu - dsl > 0, dsu - dsl, 1.0)
    dv_norm = ds_widths
    dv_norm_l = dsl
    if init_bounds is None:
        bounds = make_point_box(x0, dsl, dsu)
    else:
        bounds = np.asarray(init_bounds, dtype=float).copy()
    # Normalize to [0, 1]^dim; phase1 and phase2 operate in normalized space.
    bounds = (bounds - dsl[:, None]) / ds_widths[:, None]

    if weight is None:
        weight = np.ones(n_dims)

    state = make_box_state(
        bounds=bounds,
        n_dims=n_dims,
        n_qoi=int(reqU.shape[0]),
    )

    phase1_iter(
        state, problem, dv_norm, dv_norm_l, reqL, reqU, parameters, ind_parameters,
        sample_size, growth_rate, weight, phase1_convergence_tol,
        confidence=confidence,
        phase1_max_iterations=phase1_max_iterations,
        stop_callback=stop_callback,
        callback=callback,
        label=label,
    )

    phase2_iter(
        state, problem, dv_norm, dv_norm_l, reqL, reqU, parameters, ind_parameters,
        sample_size, target_good_fraction, confidence, weight,
        phase2_max_iterations=phase2_max_iterations,
        stop_callback=stop_callback,
        callback=callback,
        label=label,
    )

    return SolutionSpaceResult(
        bounds=state.bounds * ds_widths[:, None] + dsl[:, None],
        good_fraction=state.good_fraction,
        good_fraction_lower_bound=state.good_fraction_lower_bound,
        m=state.m,
        N=state.N,
        samples=state.samples,
        phase1_iters=state.phase1_iter_count,
        phase2_iters=state.phase2_iter_count,
    )


__all__ = ["SolutionSpaceResult", "compute_solution_space"]
