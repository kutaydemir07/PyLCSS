# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.
# WCCM-ECCOMAS 2026 — Computing Multi-Modal Solution Spaces for Non-Convex Feasible Regions in Robust Design
# Authors: Kutay Demir, Detlef Gerhard, Ruhr-Universität Bochum

from __future__ import annotations

import logging
import time
from typing import Optional

import numpy as np

from .contracts import (
    EvaluatableProblem,
    FloatArray,
    ProgressCallback,
    StopCallback,
)
from .coordinates import VariableLayout
from .decoupling import DecouplingResolutionMixin
from .discovery import (
    DeflationOptimizationMixin,
    LHSInitializationMixin,
    SeedClusteringMixin,
)
from .geometry import relative_union_volume
from .modal_optimization import optimize_modal_boxes
from .models import (
    BoxSolutionSpace,
    DecoupledMultiModalForm,
    MMSSParameters,
    MultiModalResult,
)
from .phase1 import make_point_box
from .redundancy import RedundancyResolutionMixin
from .samples import merge_samples

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
    space per mode, (4) largest-box retention per connected overlap group,
    and (5) extended-problem decoupling into common and separating
    variables.
    """

    def __init__(
        self,
        problem: EvaluatableProblem,
        dsl: FloatArray,
        dsu: FloatArray,
        reqL: FloatArray,
        reqU: FloatArray,
        parameters: Optional[FloatArray] = None,
        params: Optional[MMSSParameters] = None,
        weight: Optional[FloatArray] = None,
    ) -> None:
        self.problem = problem
        self.dsl = np.asarray(dsl, dtype=float)
        self.dsu = np.asarray(dsu, dtype=float)
        self.reqL = np.asarray(reqL, dtype=float)
        self.reqU = np.asarray(reqU, dtype=float)
        self.parameters = parameters
        self.params = params if params is not None else MMSSParameters()
        self._validate_bounds()

        self.original_dim = self.dsl.size
        self.dv_norm = self.dsu - self.dsl

        self.layout = VariableLayout.from_bounds(self.dsl, self.dsu, self.parameters)
        self.parameters = self.layout.parameters
        self.ind_parameters = self.layout.parameter_indices
        self.active_mask = self.layout.active_mask
        self.dv_norm[~self.active_mask] = 0.0

        self.active_dsl = self.dsl[self.active_mask]
        self.active_dsu = self.dsu[self.active_mask]
        self.active_dv_norm = self.dv_norm[self.active_mask].copy()
        self.dim = len(self.active_dsl)
        if self.dim == 0:
            raise ValueError(
                "Multi-Modal discovery needs at least one active design variable"
            )

        weight_array = np.asarray(
            weight if weight is not None else np.ones(self.original_dim),
            dtype=float,
        )
        if weight_array.ndim != 1 or not np.all(np.isfinite(weight_array)):
            raise ValueError("weight must be a finite one-dimensional array")
        if np.any(weight_array < 0.0):
            raise ValueError("weight values must not be negative")
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

    def _validate_bounds(self) -> None:
        if self.dsl.ndim != 1 or self.dsl.size == 0:
            raise ValueError("dsl must be a non-empty one-dimensional array")
        if self.dsu.shape != self.dsl.shape:
            raise ValueError("dsl and dsu must have equal shape")
        if not np.all(np.isfinite(self.dsl)) or not np.all(np.isfinite(self.dsu)):
            raise ValueError("design-space bounds must be finite")
        if np.any(self.dsl > self.dsu):
            raise ValueError("design-space bounds contain an inverted interval")
        if (
            self.reqL.ndim != 1
            or self.reqU.shape != self.reqL.shape
            or self.reqL.size == 0
        ):
            raise ValueError(
                "requirement bounds must be equally sized, non-empty vectors"
            )
        if np.any(np.isnan(self.reqL)) or np.any(np.isnan(self.reqU)):
            raise ValueError("requirement bounds must not contain NaN")
        if np.any(self.reqL > self.reqU):
            raise ValueError("requirement bounds contain an inverted interval")

    def stop(self) -> None:
        self._stop = True

    # ---- Discovery dispatch ------------------------------------------------

    def _expand_active_cluster(self, cluster_active: np.ndarray) -> np.ndarray:
        """Restore full physical coordinates for a cluster of active designs."""
        if cluster_active is None or cluster_active.size == 0:
            return np.empty((self.original_dim, 0))
        return self.layout.expand_points(cluster_active)

    def _run_discovery(
        self,
        callback: Optional[ProgressCallback],
        stop_callback: Optional[StopCallback],
    ) -> list[FloatArray]:
        """Run the configured discovery method and return clusters in
        original-dim physical space."""
        requested_restarts = self.params.lhs_restart_count
        if requested_restarts <= 0:
            # The paper requires the LHS budget to scale with design-space
            # size. Search is performed on [0,1]^n, so active dimension and
            # requested mode capacity are the unit-invariant size measures.
            requested_capacity = self.params.max_modes
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
            n_starts=n_starts,
            callback=callback,
            stop_callback=stop_callback,
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
        if self._stop or (stop_callback and stop_callback()):
            return []
        clusters_active = self._cluster_seeds_hdbscan(seeds_active)

        clusters_full = [
            self._expand_active_cluster(c)
            for c in clusters_active
            if c is not None and c.size > 0
        ]

        max_k = self.params.max_modes
        if max_k > 0 and len(clusters_full) > max_k:
            clusters_full.sort(key=lambda cluster: cluster.shape[1], reverse=True)
            clusters_full = clusters_full[:max_k]
        return clusters_full or []

    # ---- Solve -------------------------------------------------------------

    def solve(
        self,
        callback: Optional[ProgressCallback] = None,
        stop_callback: Optional[StopCallback] = None,
    ) -> MultiModalResult:
        start_time = time.time()
        self._stop = False
        result = MultiModalResult()

        def should_stop() -> bool:
            return self._stop or bool(stop_callback and stop_callback())

        # Stages 1-2: Discovery and clustering -------------------------------
        clusters = self._run_discovery(callback, should_stop)
        if not clusters:
            if not should_stop():
                logger.error("Discovery found no feasible points.")
            result.computation_time = time.time() - start_time
            return result

        solver_name = (
            "Nevergrad"
            if str(self.params.solver_type).lower() == "nevergrad"
            else "SLSQP"
        )
        result.clustering_method = f"{solver_name} deflated LHS searches + HDBSCAN"
        result.n_clusters_found = len(clusters)

        if should_stop():
            result.computation_time = time.time() - start_time
            return result

        # Stage 3: one box-shaped solution space per cluster ----------------
        boxes: list[BoxSolutionSpace] = []
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
        boxes = optimize_modal_boxes(
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
            stop_callback=should_stop,
        )
        if should_stop():
            result.computation_time = time.time() - start_time
            return result
        if not boxes:
            logger.warning("No modal box reached the requested reliability target.")
            result.computation_time = time.time() - start_time
            return result

        # Stage 4: retain one representative per connected overlap group.
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
        valid_boxes: list[BoxSolutionSpace] = []
        for i, box in enumerate(boxes):
            box.box_id = i
            box.label = box.label or f"Mode {i + 1}"
            box.compute_volume(self.dv_norm)
            widths = box.bounds[:, 1] - box.bounds[:, 0]
            active = self.active_mask
            n_zero = int((widths[active] <= 0.0).sum())
            logger.debug(
                "Mode %d: a_l=%.4f min_width=%.4e n_zero_active_dims=%d",
                i,
                box.good_fraction_lower_bound,
                float(widths.min()),
                n_zero,
            )
            if np.all(widths[active] > 0.0):
                valid_boxes.append(box)

        result.boxes = valid_boxes
        result.n_boxes_valid = len(valid_boxes)
        result.total_volume = relative_union_volume(
            [box.bounds for box in valid_boxes], self.dv_norm
        )
        result.decoupled_form = DecoupledMultiModalForm(mode_boxes=valid_boxes)

        # Stage 5: duplicate separating variables and QoIs per mode, merge
        # common variables, then re-run the box-shaped solution-space method.
        if self.params.decoupling_enabled and len(valid_boxes) >= 2:
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
                stop_callback=should_stop,
            )

        batches = [box.samples for box in valid_boxes if box.samples is not None]
        if batches:
            result.samples_all = merge_samples(batches)

        result.computation_time = time.time() - start_time
        return result
