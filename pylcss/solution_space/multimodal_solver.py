# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.
# WCCM-ECCOMAS 2026 — Computing Multi-Modal Solution Spaces for Non-Convex Feasible Regions in Robust Design
# Authors: Kutay Demir, Detlef Gerhard, Ruhr-Universität Bochum

from __future__ import annotations

import logging
import time
from typing import Callable, List, Optional

import numpy as np

from .multimodal_models import (
    BoxSolutionSpace,
    DecoupledMultiModalForm,
    MMSSParameters,
    MultiModalResult,
)

from .discovery import (
    DeflationOptimizationMixin,
    LHSInitializationMixin,
    SeedClusteringMixin,
)

# Stage 3 box-shaped solution-space computation
from .compute_solution_space import run_multimodal
from .phase1 import make_point_box
from .decoupling import DecouplingResolutionMixin
from .redundancy import RedundancyResolutionMixin

logger = logging.getLogger(__name__)


class MultiModalSolutionSpaceSolver(
    LHSInitializationMixin,
    DeflationOptimizationMixin,
    SeedClusteringMixin,
    RedundancyResolutionMixin,
    DecouplingResolutionMixin,
):
    """Compute a finite union of distinct box-shaped solution spaces.

    The implementation follows the paper's five stages at method level:
    (1) deflated searches from Latin-hypercube starts, (2) HDBSCAN clustering
    with singleton noise finds, (3) one Phase-I/Phase-II box-shaped solution
    space per mode, (4) largest-space retention for each all-variable overlap
    set, and (5) extended-problem decoupling into common and separating
    variables.
    """

    def __init__(
        self,
        problem,
        dsl: np.ndarray,
        dsu: np.ndarray,
        reqL: np.ndarray,
        reqU: np.ndarray,
        parameters: Optional[np.ndarray] = None,
        params: Optional[MMSSParameters] = None,
        weight: Optional[np.ndarray] = None,
    ):
        self.problem = problem
        self.dsl = np.asarray(dsl, dtype=float)
        self.dsu = np.asarray(dsu, dtype=float)
        self.reqL = np.asarray(reqL, dtype=float)
        self.reqU = np.asarray(reqU, dtype=float)
        self.parameters = parameters
        self.params = params if params is not None else MMSSParameters()

        self.original_dim = len(dsl)
        self.dv_norm = self.dsu - self.dsl

        if self.parameters is None or np.asarray(self.parameters).size == 0:
            self.parameters = np.full((2, self.original_dim), np.nan)
        else:
            self.parameters = np.asarray(self.parameters, dtype=float).copy()
            if self.parameters.shape != (2, self.original_dim):
                raise ValueError(
                    "parameters must have shape "
                    f"(2, {self.original_dim}), got {self.parameters.shape}"
                )

        # Fixed design coordinates behave like parameters during discovery.
        self.fixed_mask = np.isclose(self.dv_norm, 0.0)
        self.parameters[:, self.fixed_mask] = np.vstack(
            (self.dsl[self.fixed_mask], self.dsu[self.fixed_mask])
        )
        self.ind_parameters = np.where(~np.isnan(self.parameters[0]))[0]
        self.active_mask = np.isnan(self.parameters[0])

        self.active_dsl = self.dsl[self.active_mask]
        self.active_dsu = self.dsu[self.active_mask]
        self.active_dv_norm = self.dv_norm[self.active_mask].copy()
        self.active_dv_norm[self.active_dv_norm == 0] = 1.0
        self.dim = len(self.active_dsl)
        if self.dim == 0:
            raise ValueError("Multi-Modal discovery needs at least one active design variable")

        weight_array = np.asarray(
            weight if weight is not None else np.ones(self.original_dim),
            dtype=float,
        )
        if weight_array.size == self.original_dim:
            self.weight = weight_array
        elif weight_array.size == self.dim:
            self.weight = np.ones(self.original_dim, dtype=float)
            self.weight[self.active_mask] = weight_array
        else:
            raise ValueError(
                "weight must match either all or active design variables; "
                f"got {weight_array.size} values"
            )

        self._stop = False

    def stop(self) -> None:
        self._stop = True

    # ---- Discovery dispatch ------------------------------------------------

    def _expand_active_cluster(self, cluster_active: np.ndarray) -> np.ndarray:
        """Expand a (active_dim, N) cluster back to (original_dim, N).

        Discovery works in the active (non-fixed) sub-space; downstream code
        expects full-dimension physical vectors.
        """
        if cluster_active is None or cluster_active.size == 0:
            return np.empty((self.original_dim, 0))
        n_cols = cluster_active.shape[1]
        full = np.zeros((self.original_dim, n_cols), dtype=float)
        inactive = ~self.active_mask
        full[inactive, :] = self.parameters[0, inactive][:, np.newaxis]
        full[self.active_mask, :] = cluster_active
        return full

    def _run_discovery(self, callback: Optional[Callable]) -> List[np.ndarray]:
        """Run the configured discovery method and return clusters in
        original-dim physical space."""
        requested_restarts = int(getattr(self.params, "lhs_restart_count", 0))
        if requested_restarts <= 0:
            # The paper requires the LHS budget to scale with design-space
            # size. Search is performed on [0,1]^n, so active dimension and
            # requested mode capacity are the unit-invariant size measures.
            requested_capacity = int(getattr(self.params, "max_modes", 0))
            if requested_capacity <= 0:
                requested_capacity = int(getattr(self.params, "max_clusters", 0))
            n_starts = max(25, 10 * self.dim, 5 * max(requested_capacity, 5))
        else:
            n_starts = requested_restarts
        if callback:
            callback(
                None,
                None,
                "Stage 1 - Discovery: deflated searches from "
                f"{n_starts} Latin-hypercube starts...",
            )
        found_regions = self._find_feasible_regions_via_optimization(
            n_starts=n_starts, callback=callback
        )
        if found_regions is None or found_regions.shape[1] == 0:
            return []
        seeds_active = found_regions

        if callback:
            callback(
                None,
                None,
                "Stage 2 - Clustering: HDBSCAN on "
                f"{seeds_active.shape[1]} feasible finds...",
            )
        try:
            clusters_active = self._cluster_seeds_hdbscan(seeds_active)
        except Exception as e:
            logger.warning("Seed clustering failed (%s). Using individual seeds.", e)
            clusters_active = [seeds_active[:, i : i + 1] for i in range(seeds_active.shape[1])]

        clusters_full = [
            self._expand_active_cluster(c)
            for c in clusters_active
            if c is not None and c.size > 0
        ]

        max_k = int(getattr(self.params, "max_modes", 0))
        if max_k <= 0:
            max_k = int(getattr(self.params, "max_clusters", 0))
        if max_k > 0 and len(clusters_full) > max_k:
            clusters_full = clusters_full[:max_k]
        return clusters_full or []

    # ---- Solve -------------------------------------------------------------

    def solve(
        self,
        callback: Optional[Callable] = None,
        stop_callback: Optional[Callable] = None,
    ) -> MultiModalResult:
        start_time = time.time()
        result = MultiModalResult()

        # Stages 1-2: Discovery and clustering -------------------------------
        clusters = self._run_discovery(callback)
        if not clusters:
            logger.error("Discovery found no feasible points.")
            result.computation_time = time.time() - start_time
            return result

        solver_name = (
            "Nevergrad" if str(self.params.solver_type).lower() == "nevergrad"
            else "SLSQP"
        )
        result.clustering_method = (
            f"{solver_name} deflated LHS searches + HDBSCAN"
        )
        result.n_clusters_found = len(clusters)

        if self._stop or (stop_callback and stop_callback()):
            return result

        # Stage 3: one box-shaped solution space per cluster ----------------
        boxes: List[BoxSolutionSpace] = []
        for k, cluster in enumerate(clusters):
            if cluster is None or cluster.size == 0:
                continue
            if cluster.shape[1] == 1:
                anchor = cluster[:, 0]
            else:
                centroid = np.mean(cluster, axis=1)
                d2 = np.sum((cluster - centroid.reshape(-1, 1)) ** 2, axis=0)
                anchor = cluster[:, int(np.argmin(d2))]
            boxes.append(
                BoxSolutionSpace(
                    box_id=k,
                    bounds=make_point_box(anchor, self.dsl, self.dsu),
                    cluster_size=int(cluster.shape[1]),
                    label=f"Mode {k + 1}",
                )
            )

        if not boxes:
            logger.error("No usable cluster anchors after filtering.")
            result.computation_time = time.time() - start_time
            return result

        # Phase I (grow/trim) and Phase II (good-fraction target) are the two
        # internal phases of Stage 3 - Computation.
        boxes = run_multimodal(
            self.problem,
            boxes=boxes,
            dsl=self.dsl,
            dsu=self.dsu,
            reqL=self.reqL,
            reqU=self.reqU,
            parameters=self.parameters,
            ind_parameters=self.ind_parameters,
            params=self.params,
            weight=self.weight,
            callback=callback,
            stop_callback=lambda: self._stop or (stop_callback and stop_callback()),
        )

        # Stage 4: all-variable overlap defines redundant sets; retain only
        # the largest solution space from each connected set.
        n_before = len(boxes)
        if callback:
            callback(
                None,
                None,
                f"Stage 4 - Redundancy: checking {n_before} solution spaces...",
            )
        boxes = self._resolve_redundant_solution_spaces(boxes, callback=callback)
        if callback:
            callback(
                None,
                None,
                "Stage 4 - Redundancy: retained "
                f"{len(boxes)} of {n_before} solution spaces.",
            )

        # 4. Build the result ----------------------------------------------
        valid_boxes: List[BoxSolutionSpace] = []
        for i, box in enumerate(boxes):
            box.box_id = i
            box.label = box.label or f"Mode {i + 1}"
            box.compute_volume(self.dv_norm)
            widths = box.bounds[:, 1] - box.bounds[:, 0]
            active = self.dv_norm > 1e-20
            n_zero = int((widths[active] <= 0.0).sum())
            logger.debug(
                "Mode %d: a_l=%.4f min_width=%.4e n_zero_active_dims=%d",
                i, box.good_fraction_lower_bound, float(widths.min()), n_zero,
            )
            if np.all(widths[active] > 0.0):
                valid_boxes.append(box)

        result.boxes = valid_boxes
        result.n_boxes_valid = len(valid_boxes)
        result.total_volume = sum(b.volume for b in valid_boxes)
        result.decoupled_form = DecoupledMultiModalForm(mode_boxes=valid_boxes)

        decoupling_enabled = bool(getattr(self.params, "decoupling_enabled", True))
        legacy_enabled = getattr(self.params, "phase3_decoupling_enabled", None)
        legacy_commonality = getattr(self.params, "phase3_find_commonality", None)
        if legacy_enabled is not None:
            decoupling_enabled = bool(legacy_enabled)
        elif legacy_commonality is not None:
            decoupling_enabled = bool(legacy_commonality)

        # Stage 5: duplicate separating variables and QoIs per mode, merge
        # common variables, then re-run the box-shaped solution-space method.
        if decoupling_enabled and len(valid_boxes) >= 2:
            if callback:
                callback(
                    None,
                    None,
                    "Stage 5 - Decoupling: finding common and separating "
                    "variables and refining the extended problem...",
                )
            result.decoupled_form = self._compute_decoupled_form(
                valid_boxes,
                callback=callback,
                stop_callback=lambda: self._stop or (stop_callback and stop_callback()),
            )

        # Aggregate per-mode samples for downstream UI consumers.
        all_points, all_good, all_bad, all_viol, all_qoi = [], [], [], [], []
        for box in valid_boxes:
            if box.samples:
                all_points.append(box.samples["points"])
                all_good.append(box.samples["is_good"])
                all_bad.append(box.samples["is_bad"])
                all_viol.append(box.samples.get("violation_idx", np.zeros(0, dtype=int)))
                all_qoi.append(box.samples.get("qoi_values", np.zeros((self.reqU.shape[0], 0))))
        if all_points:
            result.samples_all = {
                "points": np.hstack(all_points),
                "is_good": np.concatenate(all_good),
                "is_bad": np.concatenate(all_bad),
                "violation_idx": np.concatenate(all_viol),
                "qoi_values": np.hstack(all_qoi),
            }

        result.computation_time = time.time() - start_time
        return result
