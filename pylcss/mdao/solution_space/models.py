# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.
# WCCM-ECCOMAS 2026 — Computing Multi-Modal Solution Spaces for Non-Convex Feasible Regions in Robust Design
# Authors: Kutay Demir, Detlef Gerhard, Ruhr-Universität Bochum

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from .contracts import FloatArray, SampleBatch


@dataclass
class MMSSParameters:
    # --- Discovery & clustering ---
    lhs_restart_count: int = 0  # 0 -> auto, scaled with dimension/capacity
    initial_sample_size: int = 0  # deprecated pre-deflation setting
    max_modes: int = 0  # 0 -> unknown K, inferred by HDBSCAN
    max_clusters: int = 0  # deprecated optional safety limit
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

    # Compatibility with configurations written before the five-stage
    # terminology was aligned with the MMSS paper.  Decoupling is not a third
    # optimization phase: Phase I and Phase II belong to Stage 3 (Computation).
    phase3_decoupling_enabled: Optional[bool] = None
    phase3_find_commonality: Optional[bool] = None
    phase3_min_common_width_ratio: Optional[float] = None

    # --- Parallelism ---
    n_workers: int = 0  # 0 → auto (cpu_count() // 2)

    def __post_init__(self) -> None:
        """Reject invalid settings before starting an expensive solve."""
        non_negative = {
            "lhs_restart_count": self.lhs_restart_count,
            "initial_sample_size": self.initial_sample_size,
            "max_modes": self.max_modes,
            "max_clusters": self.max_clusters,
            "phase2_max_iterations": self.phase2_max_iterations,
            "n_workers": self.n_workers,
        }
        for name, value in non_negative.items():
            if value < 0:
                raise ValueError(f"{name} must not be negative")

        positive = {
            "min_cluster_size": self.min_cluster_size,
            "discovery_solver_maxiter": self.discovery_solver_maxiter,
            "optimization_sample_size": self.optimization_sample_size,
            "max_iterations": self.max_iterations,
        }
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")

        for name, value in {
            "target_good_fraction": self.target_good_fraction,
            "good_fraction_confidence": self.good_fraction_confidence,
        }.items():
            if not 0.0 < value < 1.0:
                raise ValueError(f"{name} must be between 0 and 1")

        if not 0.0 < self.phase1_growth_rate <= 1.0:
            raise ValueError("phase1_growth_rate must be in (0, 1]")
        if self.phase1_convergence_tol < 0.0:
            raise ValueError("phase1_convergence_tol must not be negative")
        if self.deflation_gamma < 0.0:
            raise ValueError("deflation_gamma must not be negative")
        if self.deflation_sigma <= 0.0:
            raise ValueError("deflation_sigma must be positive")
        if not 0.0 <= self.min_common_width_ratio <= 1.0:
            raise ValueError("min_common_width_ratio must be in [0, 1]")
        if (
            self.phase3_min_common_width_ratio is not None
            and not 0.0 <= self.phase3_min_common_width_ratio <= 1.0
        ):
            raise ValueError("phase3_min_common_width_ratio must be in [0, 1]")


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

    def compute_volume(self, dv_norm: FloatArray) -> float:
        """Compute and store volume relative to the design-space widths."""
        dv_norm = np.asarray(dv_norm, dtype=float)
        if self.bounds.ndim != 2 or self.bounds.shape[1] != 2:
            raise ValueError("bounds must have shape (n_dimensions, 2)")
        if dv_norm.shape != (self.bounds.shape[0],):
            raise ValueError("dv_norm must match the number of box dimensions")
        active = dv_norm > 1e-20
        widths = self.bounds[:, 1] - self.bounds[:, 0]
        widths = np.maximum(widths, 0.0)
        self.volume = float(np.prod(widths[active] / dv_norm[active]))
        return self.volume


@dataclass
class DecoupledMultiModalForm:
    """Paper notation ``D_MMSS`` for a decoupled union of modal boxes.

    Canonical terminology follows the paper: common variables, separating
    variables, and modes. Compatibility properties expose names used by the
    initial PyLCSS port.
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

    @property
    def shared_variable_indices(self) -> list[int]:
        return self.common_variable_indices

    @property
    def branch_variable_indices(self) -> list[int]:
        return self.separating_variable_indices

    @property
    def branch_boxes(self) -> list[BoxSolutionSpace]:
        return self.mode_boxes

    @property
    def shared_groups_per_dim(self) -> dict[int, list[dict[str, Any]]]:
        return self.common_groups_per_dim


# Backward-compatible name used by the first PyLCSS port.
SharedIntervalFamily = DecoupledMultiModalForm


@dataclass
class MultiModalResult:
    boxes: list[BoxSolutionSpace] = field(default_factory=list)
    decoupled_form: Optional[DecoupledMultiModalForm] = None
    decoupled_forms: list[DecoupledMultiModalForm] = field(default_factory=list)
    computation_time: float = 0.0
    total_volume: float = 0.0
    n_boxes_valid: int = 0
    n_clusters_found: int = 0
    clustering_method: str = ""
    samples_all: Optional[SampleBatch] = None

    @property
    def shared_family(self) -> Optional[DecoupledMultiModalForm]:
        return self.decoupled_form

    @shared_family.setter
    def shared_family(self, value: Optional[DecoupledMultiModalForm]) -> None:
        self.decoupled_form = value

    @property
    def shared_families(self) -> list[DecoupledMultiModalForm]:
        return self.decoupled_forms
