# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Execution workflow for the topology optimization graph node."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import replace
from typing import Any

import numpy as np

from ..optimization.results import TopologyOptVoxelResult
from ..optimization.voxel_solver import TopologyOptVoxelSolver
from .boundary_mapping import _bounds_payload
from .execution_output import build_topology_output, finalize_topology_output
from .execution_setup import PreparedTopologyStudy, prepare_topology_study
from .topology_convergence import (
    refine_prepared_study,
    resample_density_field,
    topology_convergence_report,
)
from .voxelization import (
    _effective_density_cutoff,
    _source_material_fraction,
)

logger = logging.getLogger(__name__)

ProgressCallback = Callable[
    [dict[str, object], np.ndarray | None, int, int],
    None,
]
CancelCallback = Callable[[], bool]


def _preview_payload(
    node: Any,
    study: PreparedTopologyStudy,
    density: np.ndarray,
    stage: str | None = None,
) -> dict[str, Any]:
    """Build the lightweight payload emitted during solver progress."""
    density_cutoff = _effective_density_cutoff(
        node.get_property("density_cutoff") or 0.30
    )
    payload = {
        "type": "topopt_voxel",
        "density": density,
        "design_domain": study.design_domain,
        "grid_shape": density.shape,
        "bounds": _bounds_payload(study.bounds),
        "density_cutoff": density_cutoff,
        "visualization_mode": "Density",
        "target_vol_frac": study.problem.volfrac,
        "final_vol_frac": _source_material_fraction(
            density,
            study.design_domain,
        ),
        "bounding_vol_frac": float(np.mean(density)) if density.size else 0.0,
        "_preview": True,
        "preview_kind": "design_domain" if stage else "density",
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
            None,
            max(0, int(step)),
            max(1, int(total)),
        )
    except Exception:
        logger.debug("Topology preview callback failed.", exc_info=True)


def _emit_recovery_preview(
    study: PreparedTopologyStudy,
    output_context: Any,
    progress_callback: ProgressCallback | None,
) -> None:
    """Publish lattice geometry, but keep solid runs on the voxel actor.

    A recovered solid surface is only a transient triangulation on the way to
    the final CAD body. Rendering it is expensive and replaces the responsive
    voxel view with a mesh that is discarded moments later. A lattice has no
    automatic CAD replacement, so its recovered/manufactured representation
    remains useful and must still be published.
    """
    if progress_callback is None:
        return
    try:
        if study.problem.lattice_cell_type:
            surface_preview = dict(output_context.payload)
            surface_preview["_preview"] = True
            surface_preview["stage"] = "Manufactured lattice ready"
            progress_callback(surface_preview, None, 1, 1)
            return
        progress_callback(
            {
                "type": "topopt_progress",
                "status_only": True,
                "stage": "Surface recovered — reconstructing CAD in background",
            },
            None,
            1,
            1,
        )
    except Exception:
        logger.debug(
            "Recovered topology preview notification failed.",
            exc_info=True,
        )


def _solve_study(
    node: Any,
    study: PreparedTopologyStudy,
    progress_callback: ProgressCallback | None,
    cancel_callback: CancelCallback | None,
    *,
    progress_offset: int = 0,
    progress_total: int | None = None,
    preview_stage: str = "Design domain preview",
) -> TopologyOptVoxelResult | None:
    """Execute one prepared topology study and maintain cancellation state."""
    problem = study.problem
    _emit_preview(
        node,
        study,
        progress_callback,
        np.asarray(study.design_domain, dtype=float),
        progress_offset,
        progress_total or problem.max_iter,
        stage=preview_stage,
    )

    solver = TopologyOptVoxelSolver(problem)
    node._active_solver = solver

    def _callback(
        iteration: int,
        compliance: float,
        change: float,
        _density: np.ndarray,
    ) -> None:
        if cancel_callback is not None and cancel_callback():
            solver.stop()
        # Do not send the 3-D field to VTK on every iteration. Queued density
        # arrays, contour extraction, smoothing, and rendering compete with the
        # optimizer and can make the main window appear locked. A tiny status
        # payload preserves progress feedback; the viewer renders once before
        # the solve and again from the final result.
        if progress_callback is not None:
            try:
                progress_callback(
                    {
                        "type": "topopt_progress",
                        "status_only": True,
                        "objective": float(compliance),
                        "design_change": float(change),
                    },
                    None,
                    max(0, int(progress_offset) + int(iteration) - 1),
                    max(1, int(progress_total or problem.max_iter)),
                )
            except Exception:
                logger.debug(
                    "Topology progress-status callback failed.",
                    exc_info=True,
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


def _use_progressive_resolution(
    node: Any,
    study: PreparedTopologyStudy,
) -> bool:
    """Return whether a guided compliance study benefits from two grids.

    The target grid is always solved.  A homogenized lattice is especially
    well suited to this warm start: the analysis mesh represents the smooth
    macro density field, not the explicit unit-cell geometry, so developing
    that field first on a smaller grid does not remove lattice detail from the
    final result.
    """
    problem = study.problem
    extrusion_axis = {
        "x": 0,
        "y": 1,
        "z": 2,
    }.get(str(problem.mc.extrusion or "none").lower())
    resolution_axes = [
        axis for axis in range(3) if axis != extrusion_axis
    ]
    if not resolution_axes:
        resolution_axes = [0, 1, 2]
    dimensions = (problem.nelx, problem.nely, problem.nelz)
    profile_is_resolved = min(
        dimensions[axis] for axis in resolution_axes
    ) >= 8
    inactive_axis_is_usable = (
        extrusion_axis is None or dimensions[extrusion_axis] >= 2
    )
    return bool(
        str(node.get_property("workflow_mode") or "Guided").strip().lower()
        == "guided"
        and problem.objective_mode == "compliance"
        and problem.physics_mode == "structural"
        and not problem.stress_constraint_enabled
        and problem.nelx * problem.nely * problem.nelz >= 8_000
        and profile_is_resolved
        and inactive_axis_is_usable
    )


def _solve_progressive_study(
    node: Any,
    study: PreparedTopologyStudy,
    progress_callback: ProgressCallback | None,
    cancel_callback: CancelCallback | None,
) -> TopologyOptVoxelResult | None:
    """Solve coarse first, then warm-start the target guided resolution."""
    target_problem = study.problem
    coarse_study = refine_prepared_study(
        study,
        0.5,
        max_voxels=max(1, target_problem.nelx * target_problem.nely * target_problem.nelz),
    )
    coarse_problem = replace(
        coarse_study.problem,
        initial_density=None,
        max_iter=min(20, target_problem.max_iter),
        tol=max(float(target_problem.tol), 0.015),
        patience=min(int(target_problem.patience), 3),
        heaviside_beta_max=min(
            float(target_problem.heaviside_beta_max),
            4.0,
        ),
    )
    coarse_study = replace(coarse_study, problem=coarse_problem)
    coarse_started = time.perf_counter()
    progressive_total = (
        int(coarse_problem.max_iter) + int(target_problem.max_iter)
    )
    coarse_result = _solve_study(
        node,
        coarse_study,
        progress_callback,
        cancel_callback,
        progress_offset=0,
        progress_total=progressive_total,
        preview_stage="Fast initialization pass",
    )
    coarse_time = time.perf_counter() - coarse_started
    if coarse_result is None or (
        cancel_callback is not None and cancel_callback()
    ):
        return None

    seed = coarse_result.design_density
    if seed is None:
        seed = coarse_result.density
    target_shape = (
        target_problem.nelx,
        target_problem.nely,
        target_problem.nelz,
    )
    initial_density = resample_density_field(seed, target_shape)
    fine_problem = replace(
        target_problem,
        initial_density=initial_density,
        # The coarse solve is only a topology warm start. The target grid must
        # solve the engineering problem the user requested: selected fidelity
        # tolerance and its full convergence budget. The former 24-iteration
        # cap made Guided results materially different from their reported
        # settings.
        max_iter=int(target_problem.max_iter),
        tol=float(target_problem.tol),
        heaviside_beta_init=min(
            max(float(target_problem.heaviside_beta_init), 4.0),
            float(target_problem.heaviside_beta_max),
        ),
    )
    fine_study = replace(study, problem=fine_problem)
    fine_started = time.perf_counter()
    fine_result = _solve_study(
        node,
        fine_study,
        progress_callback,
        cancel_callback,
        progress_offset=int(coarse_problem.max_iter),
        progress_total=progressive_total,
        preview_stage="Refining topology",
    )
    fine_time = time.perf_counter() - fine_started
    if fine_result is None:
        return None
    fine_result.progressive_resolution = {
        "method": "progressive full-resolution warm start",
        "levels": [
            {
                "grid": [
                    coarse_problem.nelx,
                    coarse_problem.nely,
                    coarse_problem.nelz,
                ],
                "voxels": int(
                    coarse_problem.nelx
                    * coarse_problem.nely
                    * coarse_problem.nelz
                ),
                "iterations": int(coarse_result.n_iter),
                "time_s": float(coarse_time),
            },
            {
                "grid": list(target_shape),
                "voxels": int(np.prod(target_shape)),
                "iterations": int(fine_result.n_iter),
                "time_s": float(fine_time),
            },
        ],
    }
    return fine_result


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

        solve_started = time.perf_counter()
        if _use_progressive_resolution(self, study):
            result = _solve_progressive_study(
                self,
                study,
                progress_callback,
                cancel_callback,
            )
        else:
            result = _solve_study(
                self,
                study,
                progress_callback,
                cancel_callback,
            )
        if result is None:
            return None
        result.solve_time_s = time.perf_counter() - solve_started

        # Publish the useful engineering result before geometry recovery. The
        # worker remains in the background and replaces this density preview
        # with either the explicit lattice or the completed blue CAD solid.
        if progress_callback is not None:
            try:
                density_preview = _preview_payload(
                    self,
                    study,
                    np.asarray(result.density, dtype=float),
                )
                density_preview["stage"] = (
                    "Density solved — building lattice in background"
                    if study.problem.lattice_cell_type
                    else "Density solved — recovering CAD in background"
                )
                progress_callback(
                    density_preview,
                    None,
                    1,
                    1,
                )
            except Exception:
                logger.debug(
                    "Final topology density preview failed.",
                    exc_info=True,
                )

        convergence_report: dict[str, object] | None = None
        if (
            str(self.get_property("workflow_mode") or "Guided").strip().lower()
            == "expert"
            and bool(self.get_property("topology_convergence_enabled"))
        ):
            levels = int(self.get_property("topology_convergence_levels") or 3)
            levels = max(2, min(levels, 3))
            studies = [study]
            results = [result]
            for factor in ((1.5, 2.0)[: levels - 1]):
                if cancel_callback is not None and cancel_callback():
                    return None
                refined_study = refine_prepared_study(
                    study,
                    factor,
                    max_voxels=500_000,
                )
                level_started = time.perf_counter()
                refined_result = _solve_study(
                    self,
                    refined_study,
                    progress_callback,
                    cancel_callback,
                )
                if refined_result is None:
                    return None
                refined_result.solve_time_s = (
                    time.perf_counter() - level_started
                )
                studies.append(refined_study)
                results.append(refined_result)
            convergence_report = topology_convergence_report(studies, results)
            study = studies[-1]
            result = results[-1]
            result.solve_time_s = float(
                sum(level_result.solve_time_s for level_result in results)
            )
            result.topology_convergence = convergence_report

        logger.info("TopologyOptVoxelNode: %s", result.message)
        recovery_started = time.perf_counter()
        output_context = build_topology_output(self, result, study)
        if output_context is None:
            return None
        recovery_time_s = time.perf_counter() - recovery_started
        _emit_recovery_preview(study, output_context, progress_callback)
        finalization_started = time.perf_counter()
        output = finalize_topology_output(self, output_context, cancel_callback)
        if convergence_report is not None:
            output["topology_convergence"] = convergence_report
        if result.progressive_resolution is not None:
            output["progressive_resolution"] = result.progressive_resolution
        output["timing"] = {
            "optimization_s": float(result.solve_time_s),
            "recovery_s": float(recovery_time_s),
            "validation_and_cad_s": float(
                time.perf_counter() - finalization_started
            ),
        }
        return output
