# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.
# WCCM-ECCOMAS 2026 — Computing Multi-Modal Solution Spaces for Non-Convex Feasible Regions in Robust Design
# Authors: Kutay Demir, Detlef Gerhard, Ruhr-Universität Bochum

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple


@dataclass
class MMSSParameters:
    # --- Discovery & clustering ---
    lhs_restart_count: int = 0            # 0 -> auto, scaled with dimension/capacity
    initial_sample_size: int = 0          # deprecated pre-deflation setting
    max_modes: int = 0                    # 0 -> unknown K, inferred by HDBSCAN
    max_clusters: int = 0                 # deprecated optional safety limit
    min_cluster_size: int = 2
    solver_type: str = "goal_attainment"
    deflation_gamma: float = 0.5
    deflation_sigma: float = 0.15
    discovery_solver_maxiter: int = 1000

    # --- Monte Carlo ---
    optimization_sample_size: int = 200

    # --- Bayesian stopping (Phase II) ---
    target_good_fraction: float = 0.99    # a* — required lower bound on a
    good_fraction_confidence: float = 0.95  # 1 - α_c

    # --- Phase I ---
    max_iterations: int = 100
    phase1_growth_rate: float = 0.05      # g
    phase1_convergence_tol: float = 1e-4  # relative change in μ to declare convergence

    # --- Phase II ---
    phase2_max_iterations: int = 0        # 0 → falls back to max_iterations

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
    n_workers: int = 0                    # 0 → auto (cpu_count() // 2)


@dataclass
class BoxSolutionSpace:
    box_id: int
    bounds: np.ndarray
    good_fraction: float = 0.0
    good_fraction_lower_bound: float = 0.0
    good_fraction_confidence: float = 0.95
    validation_successes: int = 0
    validation_samples: int = 0
    cluster_size: int = 0
    samples: Optional[Dict[str, Any]] = None
    label: str = ""
    volume: float = 0.0

    def compute_volume(self, dv_norm):
        active = dv_norm > 1e-20
        widths = self.bounds[:, 1] - self.bounds[:, 0]
        widths = np.maximum(widths, 0.0)
        self.volume = float(np.prod(widths[active] / dv_norm[active]))


@dataclass
class DecoupledMultiModalForm:
    """Paper notation ``D_MMSS`` for a decoupled union of modal boxes.

    Canonical terminology follows the paper: common variables, separating
    variables, and modes. Compatibility properties expose names used by the
    initial PyLCSS port.
    """

    common_variable_indices: List[int] = field(default_factory=list)
    separating_variable_indices: List[int] = field(default_factory=list)
    mode_boxes: List[BoxSolutionSpace] = field(default_factory=list)
    label: str = "Decoupled multi-modal form"
    message: str = ""
    volume: float = 0.0
    common_groups_per_dim: Dict[int, List[Dict[str, Any]]] = field(default_factory=dict)
    interval_unions: List[List[Tuple[float, float]]] = field(default_factory=list)

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
    def shared_variable_indices(self) -> List[int]:
        return self.common_variable_indices

    @property
    def branch_variable_indices(self) -> List[int]:
        return self.separating_variable_indices

    @property
    def branch_boxes(self) -> List[BoxSolutionSpace]:
        return self.mode_boxes

    @property
    def shared_groups_per_dim(self) -> Dict[int, List[Dict[str, Any]]]:
        return self.common_groups_per_dim


# Backward-compatible name used by the first PyLCSS port.
SharedIntervalFamily = DecoupledMultiModalForm


@dataclass
class MultiModalResult:
    boxes: List[BoxSolutionSpace] = field(default_factory=list)
    decoupled_form: Optional[DecoupledMultiModalForm] = None
    decoupled_forms: List[DecoupledMultiModalForm] = field(default_factory=list)
    computation_time: float = 0.0
    total_volume: float = 0.0
    n_boxes_valid: int = 0
    n_clusters_found: int = 0
    clustering_method: str = ""
    samples_all: Optional[Dict[str, Any]] = None

    @property
    def shared_family(self) -> Optional[DecoupledMultiModalForm]:
        return self.decoupled_form

    @shared_family.setter
    def shared_family(self, value: Optional[DecoupledMultiModalForm]) -> None:
        self.decoupled_form = value

    @property
    def shared_families(self) -> List[DecoupledMultiModalForm]:
        return self.decoupled_forms
