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
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from .multimodal_models import BoxSolutionSpace
from .compute_solution_space import compute_solution_space

logger = logging.getLogger(__name__)


@dataclass
class ExtendedLayout:
    """Mapping between extended coordinates and original mode/dimension pairs."""

    K: int
    n_dims_orig: int
    n_z: int
    coord_to_branches: List[List[int]] = field(default_factory=list)
    coord_to_dim: List[int] = field(default_factory=list)
    branch_dim_to_coord: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 0), dtype=int)
    )

    def is_trivial(self) -> bool:
        return all(len(branches) == 1 for branches in self.coord_to_branches)


def build_extended_layout(
    K: int,
    n_dims_orig: int,
    shared_groups_per_dim: Dict[int, List[Dict[str, Any]]],
) -> ExtendedLayout:
    """Assign one coordinate per common group plus separating coordinates."""
    coord_to_branches: List[List[int]] = []
    coord_to_dim: List[int] = []
    branch_dim_to_coord = -np.ones((K, n_dims_orig), dtype=int)

    for j in range(n_dims_orig):
        covered: set[int] = set()
        for group in shared_groups_per_dim.get(j, []):
            branches = list(group["branches"])
            cidx = len(coord_to_branches)
            coord_to_branches.append(branches)
            coord_to_dim.append(j)
            for k in branches:
                branch_dim_to_coord[k, j] = cidx
                covered.add(k)

        for k in range(K):
            if k in covered:
                continue
            cidx = len(coord_to_branches)
            coord_to_branches.append([k])
            coord_to_dim.append(j)
            branch_dim_to_coord[k, j] = cidx

    return ExtendedLayout(
        K=K,
        n_dims_orig=n_dims_orig,
        n_z=len(coord_to_branches),
        coord_to_branches=coord_to_branches,
        coord_to_dim=coord_to_dim,
        branch_dim_to_coord=branch_dim_to_coord,
    )


def extended_initial_bounds(
    layout: ExtendedLayout,
    boxes: List[BoxSolutionSpace],
    shared_groups_per_dim: Dict[int, List[Dict[str, Any]]],
) -> np.ndarray:
    """Build the initial extended box in physical design coordinates."""
    bounds = np.zeros((layout.n_z, 2), dtype=float)
    shared_lookup: Dict[tuple, np.ndarray] = {}
    for j, groups in shared_groups_per_dim.items():
        for group in groups:
            key = (j, frozenset(group["branches"]))
            shared_lookup[key] = np.asarray(group["bounds"], dtype=float)

    for cidx in range(layout.n_z):
        j = layout.coord_to_dim[cidx]
        branches = layout.coord_to_branches[cidx]
        if len(branches) >= 2:
            bounds[cidx] = shared_lookup[(j, frozenset(branches))]
        else:
            bounds[cidx] = boxes[branches[0]].bounds[j]
    return bounds


class ExtendedProblemAdapter:
    """Adapter that lets ``compute_solution_space`` refine an extended box.

    For one extended sample column, every retained mode receives its physical
    design vector. The original problem is evaluated once per branch, and the
    QoI blocks are stacked vertically. Tiled requirement vectors therefore
    implement AND-feasibility across all branches.
    """

    def __init__(
        self,
        base_problem,
        layout: ExtendedLayout,
        original_ind_parameters: np.ndarray,
        n_total_orig: int,
    ):
        self.base = base_problem
        self.layout = layout
        self.orig_ind_p = np.asarray(original_ind_parameters, dtype=int)
        self.n_total_orig = int(n_total_orig)
        self.orig_ind_dvs = np.setdiff1d(np.arange(self.n_total_orig), self.orig_ind_p)

    def evaluate_matrix(self, x_full_extended: np.ndarray) -> np.ndarray:
        n_z = self.layout.n_z
        N = x_full_extended.shape[1]
        z = x_full_extended[:n_z, :]
        p = x_full_extended[n_z:, :]

        y_blocks: List[np.ndarray] = []
        for k in range(self.layout.K):
            x_full_k = np.zeros((self.n_total_orig, N), dtype=float)
            coord_idx = self.layout.branch_dim_to_coord[k]
            x_full_k[self.orig_ind_dvs, :] = z[coord_idx, :]
            if self.orig_ind_p.size:
                x_full_k[self.orig_ind_p, :] = p
            y_blocks.append(self.base.evaluate_matrix(x_full_k))

        return np.vstack(y_blocks)


def run_extended_refinement(
    base_problem,
    layout: ExtendedLayout,
    initial_extended_bounds: np.ndarray,
    dsl_orig: np.ndarray,
    dsu_orig: np.ndarray,
    reqL: np.ndarray,
    reqU: np.ndarray,
    parameters_orig: Optional[np.ndarray],
    ind_parameters_orig: np.ndarray,
    sample_size: int,
    growth_rate: float,
    target_good_fraction: float,
    confidence: float,
    phase1_max_iterations: int,
    phase2_max_iterations: int,
    phase1_convergence_tol: float,
    callback: Optional[Callable] = None,
    stop_callback: Optional[Callable] = None,
):
    """Run Phase 1 + Phase 2 on the extended box."""
    K = layout.K
    dsl_extended = np.asarray([dsl_orig[j] for j in layout.coord_to_dim], dtype=float)
    dsu_extended = np.asarray([dsu_orig[j] for j in layout.coord_to_dim], dtype=float)

    reqL_extended = np.tile(np.asarray(reqL, dtype=float).flatten(), K)
    reqU_extended = np.tile(np.asarray(reqU, dtype=float).flatten(), K)

    params_arr = None if parameters_orig is None else np.asarray(parameters_orig, dtype=float)
    ind_p_orig = np.asarray(ind_parameters_orig, dtype=int)
    n_p = int(ind_p_orig.size)

    if params_arr is None or params_arr.size == 0 or n_p == 0:
        parameters_extended = np.full((2, layout.n_z), np.nan)
        ind_parameters_extended = np.array([], dtype=int)
        n_total_orig = int(dsl_orig.shape[0])
    else:
        parameters_extended = np.full((2, layout.n_z + n_p), np.nan)
        parameters_extended[:, layout.n_z:] = params_arr[:, ind_p_orig]
        ind_parameters_extended = np.arange(
            layout.n_z, layout.n_z + n_p, dtype=int
        )
        n_total_orig = int(params_arr.shape[1])

    adapter = ExtendedProblemAdapter(
        base_problem=base_problem,
        layout=layout,
        original_ind_parameters=ind_p_orig,
        n_total_orig=n_total_orig,
    )

    # ``compute_solution_space`` requires an x0. The actual Phase 1/2 state is
    # ``initial_extended_bounds`` below: intersections for common coordinates and
    # original branch intervals for non-common coordinates.
    anchor = np.clip(
        initial_extended_bounds[:, 0], dsl_extended, dsu_extended
    )
    try:
        return compute_solution_space(
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
    extended_bounds: np.ndarray,
    layout: ExtendedLayout,
    source_boxes: List[BoxSolutionSpace],
) -> List[BoxSolutionSpace]:
    """Project refined extended bounds back to one box per retained mode."""
    out: List[BoxSolutionSpace] = []
    for k in range(layout.K):
        bounds = np.zeros((layout.n_dims_orig, 2), dtype=float)
        for j in range(layout.n_dims_orig):
            bounds[j] = extended_bounds[int(layout.branch_dim_to_coord[k, j])]

        box = copy.copy(source_boxes[k])
        box.bounds = bounds
        box.samples = None
        box.volume = 0.0
        out.append(box)
    return out
