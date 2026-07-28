# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Execution workflow for the topology optimization graph node."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import numpy as np

from ..optimization.results import TopologyOptVoxelResult
from ..optimization.voxel_solver import TopologyOptVoxelSolver
from .boundary_mapping import _bounds_payload
from .execution_output import build_topology_output, finalize_topology_output
from .execution_setup import PreparedTopologyStudy, prepare_topology_study
from .voxelization import (
    _effective_density_cutoff,
    _initial_design_density,
    _source_material_fraction,
)

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[dict[str, object], np.ndarray, int, int], None]
CancelCallback = Callable[[], bool]


def _preview_payload(
    node: Any,
    study: PreparedTopologyStudy,
    density: np.ndarray,
    stage: str | None = None,
) -> dict[str, Any]:
    """Build the lightweight payload emitted during solver progress."""
    density_cutoff = _effective_density_cutoff(
        node.get_property("density_cutoff") or 0.45
    )
    payload = {
        "type": "topopt_voxel",
        "density": density,
        "design_domain": study.design_domain,
        "grid_shape": density.shape,
        "bounds": _bounds_payload(study.bounds),
        "density_cutoff": density_cutoff,
        "target_vol_frac": study.problem.volfrac,
        "final_vol_frac": _source_material_fraction(
            density,
            study.design_domain,
        ),
        "bounding_vol_frac": float(np.mean(density)) if density.size else 0.0,
        "_preview": True,
    }
    if stage:
        payload["stage"] = stage
    return payload


def _emit_preview(
    node: Any,
    study: PreparedTopologyStudy,
    progress_callback: ProgressCallback | None,
    density: np.ndarray,
    step: int,
    total: int,
    stage: str | None = None,
) -> None:
    """Notify a UI callback without allowing preview failures to stop a solve."""
    if progress_callback is None:
        return
    try:
        progress_callback(
            _preview_payload(node, study, density, stage=stage),
            density,
            max(0, int(step)),
            max(1, int(total)),
        )
    except Exception:
        logger.debug("Topology preview callback failed.", exc_info=True)


def _solve_study(
    node: Any,
    study: PreparedTopologyStudy,
    progress_callback: ProgressCallback | None,
    cancel_callback: CancelCallback | None,
) -> TopologyOptVoxelResult | None:
    """Execute one prepared topology study and maintain cancellation state."""
    problem = study.problem
    _emit_preview(
        node,
        study,
        progress_callback,
        _initial_design_density(
            study.nelx,
            study.nely,
            study.nelz,
            problem.volfrac,
            study.design_domain,
        ),
        0,
        problem.max_iter,
        stage="Design domain preview",
    )

    solver = TopologyOptVoxelSolver(problem)
    node._active_solver = solver

    def _callback(
        iteration: int,
        _compliance: float,
        _change: float,
        density: np.ndarray,
    ) -> None:
        if cancel_callback is not None and cancel_callback():
            solver.stop()
        _emit_preview(
            node,
            study,
            progress_callback,
            density,
            max(0, iteration - 1),
            problem.max_iter,
        )

    try:
        return solver.run(callback=_callback)
    except Exception as exc:
        logger.exception("TopologyOptVoxelNode: solver error")
        node.set_error(str(exc))
        return None
    finally:
        if getattr(node, "_active_solver", None) is solver:
            node._active_solver = None


class _TopologyExecutionMixin:
    """Run the validated topology workflow for a graph node."""

    def run(
        self,
        progress_callback: ProgressCallback | None = None,
        cancel_callback: CancelCallback | None = None,
    ) -> dict[str, Any] | None:
        self.clear_error()
        study = prepare_topology_study(self)
        if study is None:
            return None

        result = _solve_study(
            self,
            study,
            progress_callback,
            cancel_callback,
        )
        if result is None:
            return None

        logger.info("TopologyOptVoxelNode: %s", result.message)
        output_context = build_topology_output(self, result, study)
        if output_context is None:
            return None
        return finalize_topology_output(self, output_context, cancel_callback)
