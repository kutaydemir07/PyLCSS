# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.
# WCCM-ECCOMAS 2026 — Computing Multi-Modal Solution Spaces for Non-Convex Feasible Regions in Robust Design
# Authors: Kutay Demir, Detlef Gerhard, Ruhr-Universität Bochum

from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Integral
from typing import Any, Optional

import numpy as np

from .bayesian import good_fraction_lower_bound
from .contracts import FloatArray, SampleBatch


@dataclass
class MMSSParameters:
    # --- Discovery & clustering ---
    lhs_restart_count: int = 0  # 0 -> auto, scaled with dimension/capacity
    max_modes: int = 0  # 0 -> unknown K, inferred by HDBSCAN
    min_cluster_size: int = 2
    solver_type: str = "goal_attainment"
    deflation_gamma: float = 0.5
    deflation_sigma: float = 0.15
    discovery_solver_maxiter: int = 1000

    # --- Monte Carlo ---
    optimization_sample_size: int = 200

    # --- Bayesian stopping (Phase II) ---
    target_good_fraction: float = 0.99  # a* — required lower bound on a
    good_fraction_confidence: float = 0.95  # 1 - α_c

    # --- Phase I ---
    max_iterations: int = 100
    phase1_growth_rate: float = 0.05  # g
    phase1_convergence_tol: float = 1e-4  # relative change in μ to declare convergence

    # --- Phase II ---
    phase2_max_iterations: int = 0  # 0 → falls back to max_iterations

    # --- Stage 5: decoupling over the retained modes ---
    decoupling_enabled: bool = True
    min_common_width_ratio: float = 0.0

    # --- Parallelism ---
    n_workers: int = 0  # 0 → auto (cpu_count() // 2)

    def __post_init__(self) -> None:
        """Reject invalid settings before starting an expensive solve."""
        non_negative = {
            "lhs_restart_count": self.lhs_restart_count,
            "max_modes": self.max_modes,
            "phase2_max_iterations": self.phase2_max_iterations,
            "n_workers": self.n_workers,
        }
        for name, value in non_negative.items():
            if not isinstance(value, Integral) or isinstance(value, bool):
                raise TypeError(f"{name} must be an integer")
            if value < 0:
                raise ValueError(f"{name} must not be negative")

        positive = {
            "min_cluster_size": self.min_cluster_size,
            "discovery_solver_maxiter": self.discovery_solver_maxiter,
            "optimization_sample_size": self.optimization_sample_size,
            "max_iterations": self.max_iterations,
        }
        for name, value in positive.items():
            if not isinstance(value, Integral) or isinstance(value, bool):
                raise TypeError(f"{name} must be an integer")
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.min_cluster_size < 2:
            raise ValueError("min_cluster_size must be at least 2 for HDBSCAN")
        if str(self.solver_type).lower() not in {
            "goal_attainment",
            "slsqp",
            "nevergrad",
        }:
            raise ValueError(
                "solver_type must be 'goal_attainment', 'slsqp', or 'nevergrad'"
            )

        for name, value in {
            "target_good_fraction": self.target_good_fraction,
            "good_fraction_confidence": self.good_fraction_confidence,
        }.items():
            if not 0.0 < value < 1.0:
                raise ValueError(f"{name} must be between 0 and 1")

        if not 0.0 < self.phase1_growth_rate <= 1.0:
            raise ValueError("phase1_growth_rate must be in (0, 1]")
        if (
            not np.isfinite(self.phase1_convergence_tol)
            or self.phase1_convergence_tol < 0.0
        ):
            raise ValueError("phase1_convergence_tol must not be negative")
        if not np.isfinite(self.deflation_gamma) or self.deflation_gamma < 0.0:
            raise ValueError("deflation_gamma must not be negative")
        if not np.isfinite(self.deflation_sigma) or self.deflation_sigma <= 0.0:
            raise ValueError("deflation_sigma must be positive")
        if not 0.0 <= self.min_common_width_ratio <= 1.0:
            raise ValueError("min_common_width_ratio must be in [0, 1]")


@dataclass
class BoxSolutionSpace:
    box_id: int
    bounds: FloatArray
    good_fraction: float = 0.0
    good_fraction_lower_bound: float = 0.0
    good_fraction_confidence: float = 0.95
    validation_successes: int = 0
    validation_samples: int = 0
    cluster_size: int = 0
    samples: Optional[SampleBatch] = None
    label: str = ""
    volume: float = 0.0

    def record_samples(self, samples: SampleBatch, confidence: float) -> None:
        self.samples = samples
        self.validation_successes = int(np.count_nonzero(samples["is_good"]))
        self.validation_samples = int(samples["is_good"].size)
        self.good_fraction = (
            self.validation_successes / self.validation_samples
            if self.validation_samples
            else 0.0
        )
        self.good_fraction_confidence = confidence
        self.good_fraction_lower_bound = good_fraction_lower_bound(
            self.validation_successes, self.validation_samples, confidence
        )

    def compute_volume(self, dv_norm: FloatArray) -> float:
        """Compute and store volume relative to the design-space widths."""
        dv_norm = np.asarray(dv_norm, dtype=float)
        if self.bounds.ndim != 2 or self.bounds.shape[1] != 2:
            raise ValueError("bounds must have shape (n_dimensions, 2)")
        if dv_norm.shape != (self.bounds.shape[0],):
            raise ValueError("dv_norm must match the number of box dimensions")
        active = dv_norm > 0.0
        widths = self.bounds[:, 1] - self.bounds[:, 0]
        widths = np.maximum(widths, 0.0)
        self.volume = float(np.prod(widths[active] / dv_norm[active]))
        return self.volume


@dataclass
class DecoupledMultiModalForm:
    """Paper notation ``D_MMSS`` for a decoupled union of modal boxes.

    Common variables share intervals across modes; separating variables
    retain mode-specific intervals.
    """

    common_variable_indices: list[int] = field(default_factory=list)
    separating_variable_indices: list[int] = field(default_factory=list)
    mode_boxes: list[BoxSolutionSpace] = field(default_factory=list)
    label: str = "Decoupled multi-modal form"
    message: str = ""
    volume: float = 0.0
    common_groups_per_dim: dict[int, list[dict[str, Any]]] = field(default_factory=dict)
    interval_unions: list[list[tuple[float, float]]] = field(default_factory=list)

    def is_valid(self) -> bool:
        return bool(
            self.mode_boxes
            and (
                self.common_variable_indices
                or self.separating_variable_indices
                or self.interval_unions
            )
        )


@dataclass
class MultiModalResult:
    boxes: list[BoxSolutionSpace] = field(default_factory=list)
    decoupled_form: Optional[DecoupledMultiModalForm] = None
    computation_time: float = 0.0
    total_volume: float = 0.0
    n_boxes_valid: int = 0
    n_clusters_found: int = 0
    clustering_method: str = ""
    samples_all: Optional[SampleBatch] = None
