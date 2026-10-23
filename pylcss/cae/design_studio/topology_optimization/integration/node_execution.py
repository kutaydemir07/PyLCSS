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
from pylcss.input_values import as_bool

from ..configuration.voxel_meshing import (
    REFINEMENT_MAX_ITERATIONS,
    REFINEMENT_MIN_ITERATIONS,
    REFINEMENT_STABILITY_TOLERANCE,
    estimated_voxel_memory_bytes,
    progressive_voxel_shapes,
)

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
    """Publish the actual recovered geometry without starting CAD conversion."""
    if progress_callback is None:
        return
    try:
        surface_preview = dict(output_context.payload)
        surface_preview["_preview"] = True
        recovered = surface_preview.get("recovered_shape")
        if recovered is None:
            surface_preview["stage"] = "Surface recovery failed — showing solved density"
            surface_preview["visualization_mode"] = "Density"
        else:
            surface_preview["stage"] = "Surface ready"
            surface_preview["visualization_mode"] = "Surface"
        progress_callback(surface_preview, None, 1, 1)
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
    refinement_stage: bool = False,
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
    previous_objective = None
    stable_steps = 0
    refinement_ready = False

    def _callback(
        iteration: int,
        compliance: float,
        change: float,
        _density: np.ndarray,
    ) -> None:
        nonlocal previous_objective, stable_steps, refinement_ready
        if cancel_callback is not None and cancel_callback():
            solver.stop()
        elif refinement_stage:
            # Adapt early, once a load path settles, with a hard ten-step
            # ceiling. Never freeze the coarse material boundary as a mask.
            relative_change = (
                abs(compliance - previous_objective) / max(abs(compliance), 1e-12)
                if previous_objective is not None
                else float("inf")
            )
            stable_steps = (
                stable_steps + 1
                if (relative_change < REFINEMENT_STABILITY_TOLERANCE and change < 0.05)
                else 0
            )
            previous_objective = compliance
            if iteration >= REFINEMENT_MIN_ITERATIONS and stable_steps >= 2:
                refinement_ready = True
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
        result = solver.run(callback=_callback)
        if refinement_ready:
            result.message = "Warm-start stage stabilized; advancing to a finer grid."
        return result
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
    """Use coarse-to-fine continuation only when explicitly enabled."""
    return (
        as_bool(node.get_property("automatic_mesh_refinement"))
        and len(
            progressive_voxel_shapes(
                (study.nelx, study.nely, study.nelz),
                isotropic=study.voxel_pitch is not None,
            )
        )
        > 1
    )


def _solve_progressive_study(
    node: Any,
    study: PreparedTopologyStudy,
    progress_callback: ProgressCallback | None,
    cancel_callback: CancelCallback | None,
) -> TopologyOptVoxelResult | None:
    """Run bounded coarse passes, then solve the exact final grid fully."""
    target_problem = study.problem
    target_shape = (study.nelx, study.nely, study.nelz)
    shapes = progressive_voxel_shapes(
        target_shape, isotropic=study.voxel_pitch is not None
    )
    warm_iterations = min(REFINEMENT_MAX_ITERATIONS, target_problem.max_iter)
    progressive_total = warm_iterations * (len(shapes) - 1) + target_problem.max_iter
    records = []
    seed = target_problem.initial_density
    offset = 0
    for index, shape in enumerate(shapes):
        if cancel_callback is not None and cancel_callback():
            return None
        final_level = index == len(shapes) - 1
        # Always rebuild from the original geometry and physical settings,
        # never from a coarse mask or its enlarged regularization radius.
        level = (
            study
            if final_level
            else refine_prepared_study(
                study,
                1.0,
                max_voxels=int(np.prod(target_shape)),
                target_shape=shape,
            )
        )
        initial_density = (
            resample_density_field(seed, shape) if seed is not None else None
        )
        problem = replace(level.problem, initial_density=initial_density)
        if not final_level:
            beta_max = min(float(problem.heaviside_beta_max), 2.0)
            problem = replace(
                problem,
                max_iter=warm_iterations,
                tol=max(float(problem.tol), REFINEMENT_STABILITY_TOLERANCE),
                patience=min(int(problem.patience), 3),
                heaviside_beta_init=min(float(problem.heaviside_beta_init), beta_max),
                heaviside_beta_max=beta_max,
            )
        level = replace(level, problem=problem)
        started = time.perf_counter()
        result = _solve_study(
            node,
            level,
            progress_callback,
            cancel_callback,
            progress_offset=offset,
            progress_total=progressive_total,
            preview_stage=f"Voxel refinement {index + 1}/{len(shapes)} · "
            + " × ".join(map(str, shape)),
            refinement_stage=not final_level,
        )
        if (
            result is None
            or result.message == "Stopped by user"
            or (cancel_callback is not None and cancel_callback())
        ):
            return None
        records.append(
            {
                "grid": list(shape),
                "voxels": int(np.prod(shape)),
                "iterations": int(result.n_iter),
                "time_s": time.perf_counter() - started,
                "performance": getattr(result, 'performance', {}),
                "final_target": final_level,
                "transition": "final solve"
                if final_level
                else "stabilized"
                if result.n_iter < warm_iterations
                else "iteration limit",
            }
        )
        seed = (
            result.design_density
            if result.design_density is not None
            else result.density
        )
        offset += problem.max_iter
    result.progressive_resolution = {
        "method": "optional global coarse-to-fine voxel continuation",
        "reference": "https://arxiv.org/abs/1009.4975",
        "reference_scope": "Update timing only; global continuation has no local AMR or derefinement.",
        "target_grid": list(target_shape),
        "levels": records,
    }
    return result


def _convergence_settings(node):
    if not as_bool(node.get_property("topology_convergence_enabled")):
        return None
    raw = node.get_property("topology_convergence_levels")
    levels = float(3 if raw is None else raw)
    if not np.isfinite(levels) or not levels.is_integer() or levels < 2:
        raise ValueError(
            "Topology convergence requires an integer count of at least two resolutions."
        )
    raw_objective = node.get_property("topology_objective_tolerance")
    raw_overlap = node.get_property("topology_overlap_tolerance")
    objective = float(0.05 if raw_objective is None else raw_objective)
    overlap = float(0.90 if raw_overlap is None else raw_overlap)
    if not np.isfinite([objective, overlap]).all() or not (
        0 <= objective <= 1 and 0 <= overlap <= 1
    ):
        raise ValueError(
            "Topology comparison tolerances must be finite fractions between 0 and 1."
        )
    return int(levels), objective, overlap


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
        try:
            comparison = _convergence_settings(self)
        except (TypeError, ValueError) as exc:
            self.set_error(str(exc))
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
        progressive_report = result.progressive_resolution

        # Publish the useful engineering result before geometry recovery. The
        # worker remains in the background and replaces this density preview
        # with the recovered surface.
        if progress_callback is not None:
            try:
                density_preview = _preview_payload(
                    self,
                    study,
                    np.asarray(result.density, dtype=float),
                )
                density_preview["stage"] = (
                    "Density solved — recovering surface in background"
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
        if comparison is not None:
            levels, objective_tolerance, overlap_tolerance = comparison
            studies = [study]
            results = [result]
            for level in range(1, levels):
                factor = 1.0 + 0.5 * level
                if cancel_callback is not None and cancel_callback():
                    return None
                try:
                    refined_study = refine_prepared_study(
                        study,
                        factor,
                        max_voxels=study.max_voxel_elements or None,
                    )
                    new_shape = (
                        refined_study.nelx,
                        refined_study.nely,
                        refined_study.nelz,
                    )
                    previous_shape = (
                        studies[-1].nelx,
                        studies[-1].nely,
                        studies[-1].nelz,
                    )
                    if (
                        not all(a >= b for a, b in zip(new_shape, previous_shape))
                        or new_shape == previous_shape
                    ):
                        raise ValueError(
                            "The element limit prevents another finer comparison grid."
                        )
                except ValueError as exc:
                    self.set_error(
                        f"Topology convergence mesh could not be prepared: {exc} "
                        "Increase Maximum Elements to allow the requested refinement."
                    )
                    return None
                level_started = time.perf_counter()
                refined_result = _solve_study(
                    self,
                    refined_study,
                    progress_callback,
                    cancel_callback,
                )
                if refined_result is None:
                    return None
                refined_result.solve_time_s = time.perf_counter() - level_started
                studies.append(refined_study)
                results.append(refined_result)
            convergence_report = topology_convergence_report(
                studies,
                results,
                maximum_relative_objective_change=objective_tolerance,
                minimum_topology_jaccard=overlap_tolerance,
            )
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
        output = finalize_topology_output(self, output_context)
        if convergence_report is not None:
            output["topology_convergence"] = convergence_report
        if progressive_report is not None:
            output["progressive_resolution"] = progressive_report
        output["voxel_meshing"] = dict(getattr(self, "_resolved_voxel_meshing", {}))
        output["voxel_meshing"].update(
            target_grid=output["voxel_meshing"].get("grid"),
            grid=list(result.density.shape),
            elements=int(np.prod(result.density.shape)),
            voxel_size=study.voxel_pitch,
            estimated_memory_bytes=estimated_voxel_memory_bytes(result.density.shape),
            voxel_edges=(
                (study.bounds[1] - study.bounds[0]) / np.asarray(result.density.shape)
            ).tolist(),
        )
        output["timing"] = {
            "optimization_s": float(result.solve_time_s),
            "recovery_s": float(recovery_time_s),
            "validation_and_cad_s": float(time.perf_counter() - finalization_started),
        }
        return output
