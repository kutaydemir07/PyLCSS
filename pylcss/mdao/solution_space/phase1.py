# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.
# Markus Zimmermann, Johannes Edler von Hoessle
# Computing solution spaces for robust design
# https://doi.org/10.1002/nme.4450

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .bayesian import good_fraction_lower_bound
from .contracts import (
    EvaluatableProblem,
    FloatArray,
    IntArray,
    ProgressCallback,
    SampleBatch,
    StopCallback,
)
from .samples import empty_samples
from .sampling import sample_box_batch
from .step_a import trim_box
from .step_b import expand_box

# ---------------------------------------------------------------------------
# Shared geometry helpers
# ---------------------------------------------------------------------------


def make_point_box(
    anchor: FloatArray,
    dsl: FloatArray,
    dsu: FloatArray,
) -> FloatArray:
    """Build a degenerate ``(n_dims, 2)`` box at ``anchor``, clipped to DS."""
    a = np.asarray(anchor, dtype=float).flatten()
    a = np.clip(a, dsl, dsu)
    return np.column_stack((a, a))


# ---------------------------------------------------------------------------
# Modal solution-space state
# ---------------------------------------------------------------------------


@dataclass
class BoxState:
    """Current candidate box Omega^z and statistics for one MMSS mode.

    Passed into the Phase I and Phase II loops, which update it in place each
    iteration. When phase2_target_reached is True the stopping criterion a_l >= a*
    has been met.
    """

    bounds: FloatArray
    samples: SampleBatch

    # Last sample statistics
    good_fraction: float = 0.0  # empirical a_hat = m/N from last sample
    good_fraction_lower_bound: float = 0.0  # Bayesian lower bound a_l on true a
    m: int = 0  # good designs in last sample
    N: int = 0  # total designs in last sample

    # Termination flags
    phase2_target_reached: bool = False

    # Iteration counters
    phase1_iter_count: int = 0
    phase2_iter_count: int = 0

    def record_samples(self, samples: SampleBatch, confidence: float) -> None:
        """Store statistics for the current, freshly sampled bounds."""
        self.samples = samples
        self.m = int(np.count_nonzero(samples["is_good"]))
        self.N = int(samples["is_good"].size)
        self.good_fraction = self.m / self.N if self.N else 0.0
        self.good_fraction_lower_bound = good_fraction_lower_bound(
            self.m, self.N, confidence
        )


def make_box_state(
    bounds: FloatArray,
    n_dims: int,
    n_qoi: int,
) -> BoxState:
    """Create an initial BoxState at the given bounds with empty samples."""
    return BoxState(
        bounds=np.asarray(bounds, dtype=float).copy(),
        samples=empty_samples(n_dims, n_qoi),
    )


# ---------------------------------------------------------------------------
# Phase I kernel
# ---------------------------------------------------------------------------


def run_phase_one(
    state: BoxState,
    problem: EvaluatableProblem,
    dv_norm: FloatArray,
    dv_norm_l: FloatArray,
    reqL: FloatArray,
    reqU: FloatArray,
    parameters: Optional[FloatArray],
    ind_parameters: IntArray,
    sample_size: int,
    growth_rate: float,
    weight: FloatArray,
    phase1_convergence_tol: float,
    confidence: float = 0.95,
    phase1_max_iterations: int = 100,
    stop_callback: Optional[StopCallback] = None,
    callback: Optional[ProgressCallback] = None,
    label: str = "",
) -> None:
    """Expand and trim, adapting growth until a usable sample can be drawn."""
    n_dims = len(dv_norm)
    if n_dims == 0:
        return
    g = float(growth_rate)
    previous_mu = None
    dvbox = state.bounds.copy()
    for _ in range(phase1_max_iterations):
        if stop_callback and stop_callback():
            return
        while True:
            if stop_callback and stop_callback():
                return
            candidate = expand_box(dvbox, np.zeros(n_dims), np.ones(n_dims), g)
            if np.array_equal(candidate, dvbox):
                return
            x_sample, batch = sample_box_batch(
                problem,
                candidate,
                parameters,
                reqL,
                reqU,
                dv_norm,
                dv_norm_l,
                ind_parameters,
                sample_size,
            )
            fraction = float(np.mean(batch["is_good"]))
            if fraction >= 0.10:
                if fraction >= 0.80:
                    g = min(g * 1.5, 0.3)
                break
            # No fixed floor or retry cap: stop when floating-point boundaries
            # cannot change, allowing thin feasible regions to be expanded.
            g *= 0.5
        dvbox, mu = trim_box(
            candidate, x_sample, batch["is_good"], batch["is_bad"], weight
        )
        _, evaluated = sample_box_batch(
            problem,
            dvbox,
            parameters,
            reqL,
            reqU,
            dv_norm,
            dv_norm_l,
            ind_parameters,
            sample_size,
        )
        state.bounds = dvbox
        state.record_samples(evaluated, confidence)
        state.phase1_iter_count += 1
        iteration = state.phase1_iter_count
        if callback and (iteration == 1 or iteration % 5 == 0):
            callback(
                None,
                None,
                f"  {label} Phase 1 iter {iteration}: "
                f"a={state.good_fraction:.3f} "
                f"a_l={state.good_fraction_lower_bound:.3f} mu={mu:.6f} g={g:.4g}",
            )
        if previous_mu is not None and iteration >= 4:
            relative_change = abs(mu - previous_mu) / max(abs(mu), np.finfo(float).tiny)
            if relative_change < phase1_convergence_tol:
                if callback:
                    callback(
                        None, None, f"  {label} Phase 1 converged ({iteration} iters)."
                    )
                return
        previous_mu = mu
    if callback:
        callback(
            None, None, f"  {label} Phase 1 done ({state.phase1_iter_count} iters)."
        )
