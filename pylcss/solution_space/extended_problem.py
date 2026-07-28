# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.
# WCCM-ECCOMAS 2026 - Computing Multi-Modal Solution Spaces for Non-Convex Feasible Regions in Robust Design
# Authors: Kutay Demir, Detlef Gerhard, Ruhr-Universitaet Bochum

"""Extended-problem refinement for Stage 5 (Decoupling).

The extended representation assigns one design coordinate to every common
``(dimension, mode-subset)`` group and one separating coordinate for modes not
covered by such a group. Phase I and Phase II can then refine a single extended
box whose samples are expanded back to all retained modes and
tested jointly.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from .box_optimization import BoxOptimizationResult, optimize_box
from .contracts import (
    EvaluatableProblem,
    FloatArray,
    IntArray,
    ProgressCallback,
    StopCallback,
)
from .models import BoxSolutionSpace

logger = logging.getLogger(__name__)

SharedGroups = dict[int, list[dict[str, Any]]]


@dataclass
class ExtendedLayout:
    """Mapping between extended coordinates and original mode/dimension pairs."""

    mode_count: int
    original_dimension_count: int
    coordinate_count: int
    coord_to_branches: list[list[int]] = field(default_factory=list)
    coord_to_dim: list[int] = field(default_factory=list)
    branch_dim_to_coord: IntArray = field(
        default_factory=lambda: np.zeros((0, 0), dtype=int)
    )

    def is_trivial(self) -> bool:
        return all(len(branches) == 1 for branches in self.coord_to_branches)


def build_extended_layout(
    mode_count: int,
    original_dimension_count: int,
    shared_groups_per_dim: SharedGroups,
) -> ExtendedLayout:
    """Assign one coordinate per common group plus separating coordinates."""
    coord_to_branches: list[list[int]] = []
    coord_to_dim: list[int] = []
    branch_dim_to_coord = -np.ones(
        (mode_count, original_dimension_count),
        dtype=int,
    )

    for j in range(original_dimension_count):
        covered: set[int] = set()
        for group in shared_groups_per_dim.get(j, []):
            branches = list(group["branches"])
            cidx = len(coord_to_branches)
            coord_to_branches.append(branches)
            coord_to_dim.append(j)
            for k in branches:
                branch_dim_to_coord[k, j] = cidx
                covered.add(k)

        for k in range(mode_count):
            if k in covered:
                continue
            cidx = len(coord_to_branches)
            coord_to_branches.append([k])
            coord_to_dim.append(j)
            branch_dim_to_coord[k, j] = cidx

    return ExtendedLayout(
        mode_count=mode_count,
        original_dimension_count=original_dimension_count,
        coordinate_count=len(coord_to_branches),
        coord_to_branches=coord_to_branches,
        coord_to_dim=coord_to_dim,
        branch_dim_to_coord=branch_dim_to_coord,
    )


def extended_initial_bounds(
    layout: ExtendedLayout,
    boxes: list[BoxSolutionSpace],
    shared_groups_per_dim: SharedGroups,
) -> FloatArray:
    """Build the initial extended box in physical design coordinates."""
    bounds = np.zeros((layout.coordinate_count, 2), dtype=float)
    shared_lookup: dict[tuple[int, frozenset[int]], FloatArray] = {}
    for j, groups in shared_groups_per_dim.items():
        for group in groups:
            key = (j, frozenset(group["branches"]))
            shared_lookup[key] = np.asarray(group["bounds"], dtype=float)

    for cidx in range(layout.coordinate_count):
        j = layout.coord_to_dim[cidx]
        branches = layout.coord_to_branches[cidx]
        if len(branches) >= 2:
            bounds[cidx] = shared_lookup[(j, frozenset(branches))]
        else:
            bounds[cidx] = boxes[branches[0]].bounds[j]
    return bounds


class ExtendedProblemAdapter:
    """Adapter that lets ``optimize_box`` refine an extended box.

    For one extended sample column, every retained mode receives its physical
    design vector. The original problem is evaluated once per branch, and the
    QoI blocks are stacked vertically. Tiled requirement vectors therefore
    implement AND-feasibility across all branches.
    """

    def __init__(
        self,
        base_problem: EvaluatableProblem,
        layout: ExtendedLayout,
        original_ind_parameters: IntArray,
        n_total_orig: int,
    ) -> None:
        self.base = base_problem
        self.layout = layout
        self.orig_ind_p = np.asarray(original_ind_parameters, dtype=int)
        self.n_total_orig = int(n_total_orig)
        self.orig_ind_dvs = np.setdiff1d(np.arange(self.n_total_orig), self.orig_ind_p)

    def evaluate_matrix(self, x_full_extended: FloatArray) -> FloatArray:
        coordinate_count = self.layout.coordinate_count
        sample_count = x_full_extended.shape[1]
        extended_designs = x_full_extended[:coordinate_count, :]
        parameter_samples = x_full_extended[coordinate_count:, :]

        y_blocks: list[FloatArray] = []
        for mode_index in range(self.layout.mode_count):
            mode_inputs = np.zeros((self.n_total_orig, sample_count), dtype=float)
            coordinate_indices = self.layout.branch_dim_to_coord[mode_index]
            mode_inputs[self.orig_ind_dvs, :] = extended_designs[
                coordinate_indices,
                :,
            ]
            if self.orig_ind_p.size:
                mode_inputs[self.orig_ind_p, :] = parameter_samples
            y_blocks.append(self.base.evaluate_matrix(mode_inputs))

        return np.vstack(y_blocks)


def run_extended_refinement(
    base_problem: EvaluatableProblem,
    layout: ExtendedLayout,
    initial_extended_bounds: FloatArray,
    dsl_orig: FloatArray,
    dsu_orig: FloatArray,
    reqL: FloatArray,
    reqU: FloatArray,
    parameters_orig: Optional[FloatArray],
    ind_parameters_orig: IntArray,
    sample_size: int,
    growth_rate: float,
    target_good_fraction: float,
    confidence: float,
    phase1_max_iterations: int,
    phase2_max_iterations: int,
    phase1_convergence_tol: float,
    callback: Optional[ProgressCallback] = None,
    stop_callback: Optional[StopCallback] = None,
) -> Optional[BoxOptimizationResult]:
    """Run Phase 1 + Phase 2 on the extended box."""
    mode_count = layout.mode_count
    dsl_extended = np.asarray([dsl_orig[j] for j in layout.coord_to_dim], dtype=float)
    dsu_extended = np.asarray([dsu_orig[j] for j in layout.coord_to_dim], dtype=float)

    reqL_extended = np.tile(
        np.asarray(reqL, dtype=float).flatten(),
        mode_count,
    )
    reqU_extended = np.tile(
        np.asarray(reqU, dtype=float).flatten(),
        mode_count,
    )

    params_arr = (
        None if parameters_orig is None else np.asarray(parameters_orig, dtype=float)
    )
    ind_p_orig = np.asarray(ind_parameters_orig, dtype=int)
    n_p = int(ind_p_orig.size)

    if params_arr is None or params_arr.size == 0 or n_p == 0:
        parameters_extended = np.full((2, layout.coordinate_count), np.nan)
        ind_parameters_extended = np.array([], dtype=int)
        n_total_orig = int(dsl_orig.shape[0])
    else:
        parameters_extended = np.full(
            (2, layout.coordinate_count + n_p),
            np.nan,
        )
        parameters_extended[:, layout.coordinate_count :] = params_arr[:, ind_p_orig]
        ind_parameters_extended = np.arange(
            layout.coordinate_count,
            layout.coordinate_count + n_p,
            dtype=int,
        )
        n_total_orig = int(params_arr.shape[1])

    adapter = ExtendedProblemAdapter(
        base_problem=base_problem,
        layout=layout,
        original_ind_parameters=ind_p_orig,
        n_total_orig=n_total_orig,
    )

    # ``optimize_box`` requires an x0. The actual Phase 1/2 state is
    # ``initial_extended_bounds`` below: intersections for common coordinates and
    # original branch intervals for non-common coordinates.
    anchor = np.clip(initial_extended_bounds[:, 0], dsl_extended, dsu_extended)
    try:
        return optimize_box(
            problem=adapter,
            x0=anchor,
            init_bounds=initial_extended_bounds,
            dsl=dsl_extended,
            dsu=dsu_extended,
            reqL=reqL_extended,
            reqU=reqU_extended,
            parameters=parameters_extended,
            ind_parameters=ind_parameters_extended,
            sample_size=sample_size,
            growth_rate=growth_rate,
            target_good_fraction=target_good_fraction,
            confidence=confidence,
            phase1_max_iterations=phase1_max_iterations,
            phase2_max_iterations=phase2_max_iterations,
            phase1_convergence_tol=phase1_convergence_tol,
            weight=None,
            callback=callback,
            stop_callback=stop_callback,
            label="extended",
        )
    except Exception:
        logger.exception("Extended-problem refinement failed")
        return None


def project_extended_to_modes(
    extended_bounds: FloatArray,
    layout: ExtendedLayout,
    source_boxes: list[BoxSolutionSpace],
) -> list[BoxSolutionSpace]:
    """Project refined extended bounds back to one box per retained mode."""
    out: list[BoxSolutionSpace] = []
    for mode_index in range(layout.mode_count):
        bounds = np.zeros((layout.original_dimension_count, 2), dtype=float)
        for dimension in range(layout.original_dimension_count):
            bounds[dimension] = extended_bounds[
                int(layout.branch_dim_to_coord[mode_index, dimension])
            ]

        box = copy.copy(source_boxes[mode_index])
        box.bounds = bounds
        box.samples = None
        box.volume = 0.0
        out.append(box)
    return out
