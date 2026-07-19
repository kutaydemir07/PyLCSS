# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.
# Markus Zimmermann, Johannes Edler von Hoessle
# Computing solution spaces for robust design
# https://doi.org/10.1002/nme.4450

from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from .phase1 import make_box_state, make_point_box, phase1_iter
from .phase2 import phase2_iter

logger = logging.getLogger(__name__)


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


# ---------------------------------------------------------------------------
# Parallel modal solution spaces (Stage 3)
# ---------------------------------------------------------------------------


def _run_branches_parallel(
    problem,
    boxes: List[Any],
    branch_indices: List[int],
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
    n_workers: int = 1,
    callback: Optional[Callable] = None,
    stop_callback: Optional[Callable] = None,
    label: str = "",
) -> None:
    """Run Phase I + Phase II for each mode independently, in parallel."""
    K = len(branch_indices)
    if K == 0:
        return

    def _do_one(k_idx: int) -> int:
        if stop_callback and stop_callback():
            return k_idx
        box = boxes[k_idx]
        anchor = np.clip(box.bounds[:, 0], dsl, dsu)
        res = compute_solution_space(
            problem,
            x0=anchor,
            init_bounds=box.bounds,
            dsl=dsl,
            dsu=dsu,
            reqL=reqL,
            reqU=reqU,
            parameters=parameters,
            ind_parameters=ind_parameters,
            sample_size=sample_size,
            growth_rate=growth_rate,
            target_good_fraction=target_good_fraction,
            confidence=confidence,
            phase1_max_iterations=phase1_max_iterations,
            phase2_max_iterations=phase2_max_iterations,
            phase1_convergence_tol=phase1_convergence_tol,
            weight=weight,
            callback=None,
            stop_callback=stop_callback,
            label=f"Mode {k_idx + 1}",
        )
        box.bounds = res.bounds
        box.samples = res.samples
        box.good_fraction = res.good_fraction
        box.good_fraction_lower_bound = res.good_fraction_lower_bound
        box.validation_successes = res.m
        box.validation_samples = res.N
        return k_idx

    if n_workers <= 1 or K <= 1:
        for k in branch_indices:
            _do_one(k)
            if callback:
                callback(
                    None, None,
                    f"  Mode {k + 1} solution space complete: "
                    f"a_l={boxes[k].good_fraction_lower_bound:.4f}",
                )
        return

    with ThreadPoolExecutor(max_workers=min(n_workers, K)) as pool:
        futures = {pool.submit(_do_one, k): k for k in branch_indices}
        n_done = 0
        for fut in as_completed(futures):
            try:
                k = fut.result()
            except Exception:
                logger.exception("Modal solution-space task failed")
                continue
            n_done += 1
            if callback:
                callback(
                    None, None,
                    f"  Mode {k + 1} solution space complete ({n_done}/{K}): "
                    f"a_l={boxes[k].good_fraction_lower_bound:.4f}",
                )


# ---------------------------------------------------------------------------
# Multi-Modal entry point
# ---------------------------------------------------------------------------


def run_multimodal(
    problem,
    boxes: List[Any],
    dsl: np.ndarray,
    dsu: np.ndarray,
    reqL: np.ndarray,
    reqU: np.ndarray,
    parameters: Optional[np.ndarray],
    ind_parameters: np.ndarray,
    params,
    weight: Optional[np.ndarray] = None,
    callback: Optional[Callable] = None,
    stop_callback: Optional[Callable] = None,
) -> List[Any]:
    """Compute one box-shaped solution space per mode in parallel."""

    sample_size = max(50, int(params.optimization_sample_size))
    growth_rate = float(getattr(params, "phase1_growth_rate", 0.2))
    target_a = float(params.target_good_fraction)
    confidence = float(params.good_fraction_confidence)
    phase1_max = int(params.max_iterations)
    phase2_max = int(getattr(params, "phase2_max_iterations", 0) or phase1_max)
    phase1_tol = float(getattr(params, "phase1_convergence_tol", 1e-4))
    n_workers = int(getattr(params, "n_workers", 0)) or max(1, (os.cpu_count() or 1) // 2)

    if callback:
        callback(
            None, None,
            "Stage 3 - Computation: grow and trim "
            f"{len(boxes)} modal solution spaces in parallel "
            f"(workers={min(n_workers, len(boxes))}).",
        )

    _run_branches_parallel(
        problem,
        boxes=boxes,
        branch_indices=list(range(len(boxes))),
        dsl=dsl,
        dsu=dsu,
        reqL=reqL,
        reqU=reqU,
        parameters=parameters,
        ind_parameters=ind_parameters,
        sample_size=sample_size,
        growth_rate=growth_rate,
        target_good_fraction=target_a,
        confidence=confidence,
        phase1_max_iterations=phase1_max,
        phase2_max_iterations=phase2_max,
        phase1_convergence_tol=phase1_tol,
        weight=weight,
        n_workers=n_workers,
        callback=callback,
        stop_callback=stop_callback,
        label="Stage 3 - Computation",
    )

    return boxes

__all__ = ["SolutionSpaceResult", "compute_solution_space", "run_multimodal"]
