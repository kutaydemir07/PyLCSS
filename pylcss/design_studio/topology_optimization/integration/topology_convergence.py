# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Mesh-refinement helpers for topology-resolution convergence studies."""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import Sequence

import numpy as np
from scipy import ndimage as ndi

from ..optimization.results import TopologyOptVoxelResult
from .execution_setup import PreparedTopologyStudy
from .voxelization import _mesh_design_domain_grid, _non_design_region_masks

logger = logging.getLogger(__name__)


def _nearest_resample(values: np.ndarray, shape: Sequence[int]) -> np.ndarray:
    """Resample a structured grid by nearest-neighbour index selection."""
    source = np.asarray(values)
    target_shape = tuple(max(1, int(value)) for value in tuple(shape)[:3])
    indices = [
        np.clip(
            np.rint(np.linspace(0, source.shape[axis] - 1, target_shape[axis])).astype(
                np.int64
            ),
            0,
            source.shape[axis] - 1,
        )
        for axis in range(3)
    ]
    return source[np.ix_(*indices)]


def resample_density_field(
    values: np.ndarray,
    shape: Sequence[int],
) -> np.ndarray:
    """Trilinearly transfer a density design to another structured grid."""
    source = np.asarray(values, dtype=float)
    if source.ndim != 3:
        raise ValueError("A topology warm start must be a three-dimensional field.")
    target_shape = tuple(max(1, int(value)) for value in tuple(shape)[:3])
    if source.shape == target_shape:
        return source.copy()
    coordinates = np.meshgrid(
        *(
            np.linspace(0.0, source.shape[axis] - 1.0, target_shape[axis])
            for axis in range(3)
        ),
        indexing="ij",
    )
    return np.asarray(
        ndi.map_coordinates(
            source,
            coordinates,
            order=1,
            mode="nearest",
            prefilter=False,
        ),
        dtype=float,
    )


def _refined_shape(
    base_shape: Sequence[int],
    factor: float,
    *,
    max_voxels: int = 500_000,
) -> tuple[int, int, int]:
    """Scale all axes together, keep them even, and respect the solver cap."""
    base = np.maximum(np.asarray(tuple(base_shape)[:3], dtype=float), 1.0)
    dims = np.maximum(np.rint(base * float(factor)).astype(int), 2)
    for axis, value in enumerate(dims):
        if value >= 4:
            dims[axis] = max(4, int(round(float(value) / 4.0)) * 4)
        elif value > 1:
            dims[axis] = value + value % 2
    total = int(np.prod(dims))
    if total > int(max_voxels):
        shrink = (float(max_voxels) / float(total)) ** (1.0 / 3.0)
        dims = np.maximum(np.floor(dims * shrink).astype(int), 2)
        dims -= dims % 2
        dims = np.maximum(dims, 2)
    return tuple(int(value) for value in dims)


def refine_prepared_study(
    study: PreparedTopologyStudy,
    factor: float,
    *,
    max_voxels: int = 500_000,
) -> PreparedTopologyStudy:
    """Clone a study at a finer grid while preserving physical parameters."""
    old_shape = (study.nelx, study.nely, study.nelz)
    new_shape = _refined_shape(old_shape, factor, max_voxels=max_voxels)
    span = np.maximum(study.bounds[1][:3] - study.bounds[0][:3], 1e-12)

    problem = study.problem
    if problem.filter_radius_is_physical:
        physical_radius = float(problem.rmin)
    else:
        # Legacy expert studies expressed rmin in cells. Freeze its current
        # physical meaning before changing the grid.
        physical_radius = float(problem.rmin) * float(
            np.max((problem.unitx, problem.unity, problem.unitz))
        )

    # Coarsening keeps the physical radius but grows the element it is measured
    # against, so a radius that spans two elements on the target grid spans one
    # on a half-scale one and less than one below that. A cone filter narrower
    # than an element averages each element with itself, which is no
    # regularization at all, and the problem rejects it outright.
    #
    # A coarse level is a topology warm start, not the study: it develops the
    # shape and hands it up. Holding it at one element regularizes it at the
    # only scale its own grid can express, and the target level — which is the
    # study — still carries the requested physical radius exactly.
    coarsest_new_edge = float(
        np.max(np.asarray(span, dtype=float) / np.asarray(new_shape, dtype=float))
    )
    # A level that cannot carry the requested radius cannot enforce the length
    # scale that radius encodes either: its thresholds would still be the ones
    # derived for the requested member size while the radius under them had
    # changed, so it would enforce some other size without saying so. Such a
    # level is a warm start, so it regularizes at the scale its own grid can
    # express and leaves the requirement to the target level.
    robust_length_scale = bool(problem.robust_length_scale)
    if physical_radius < coarsest_new_edge:
        logger.info(
            "Coarse level holds the density filter at one element "
            "(%.4g requested, %.4g on this grid); the minimum member size is "
            "enforced on the target level.",
            physical_radius,
            coarsest_new_edge,
        )
        physical_radius = coarsest_new_edge
        robust_length_scale = False

    design_domain = None
    if study.source_design_domain is not None:
        design_domain = _mesh_design_domain_grid(
            study.source_design_domain,
            study.bounds,
            *new_shape,
        )
    if design_domain is None:
        design_domain = _nearest_resample(study.design_domain, new_shape).astype(bool)
    try:
        if not study.non_design_regions:
            raise ValueError("No source non-design regions to re-voxelize.")
        passive_solid, passive_void = _non_design_region_masks(
            list(study.non_design_regions), study.bounds, *new_shape
        )
    except (TypeError, ValueError):
        passive_solid = _nearest_resample(study.passive_solid_mask, new_shape).astype(
            bool
        )
        passive_void = _nearest_resample(study.passive_void_mask, new_shape).astype(
            bool
        )
    refined_problem = replace(
        problem,
        nelx=new_shape[0],
        nely=new_shape[1],
        nelz=new_shape[2],
        unitx=float(span[0] / new_shape[0]),
        unity=float(span[1] / new_shape[1]),
        unitz=float(span[2] / new_shape[2]),
        rmin=physical_radius,
        filter_radius_is_physical=True,
        robust_length_scale=robust_length_scale,
        design_domain=design_domain,
        passive_solid_mask=passive_solid,
        passive_void_mask=passive_void,
    )
    return replace(
        study,
        problem=refined_problem,
        design_domain=design_domain,
        passive_solid_mask=passive_solid,
        passive_void_mask=passive_void,
        nelx=new_shape[0],
        nely=new_shape[1],
        nelz=new_shape[2],
    )


def topology_convergence_report(
    studies: Sequence[PreparedTopologyStudy],
    results: Sequence[TopologyOptVoxelResult],
) -> dict[str, object]:
    """Compare objectives, binary topology overlap, and connectivity by level."""
    if not studies or len(studies) != len(results):
        raise ValueError("Topology convergence needs matching study/result levels.")
    finest_shape = results[-1].density.shape
    records: list[dict[str, object]] = []
    previous_objective: float | None = None
    previous_binary: np.ndarray | None = None
    previous_components: int | None = None

    for index, (study, result) in enumerate(zip(studies, results, strict=True)):
        objective_history = result.objective_history or result.compliance_history
        objective = float(objective_history[-1]) if objective_history else float("nan")
        binary = _nearest_resample(
            np.asarray(result.density, dtype=float) >= 0.5,
            finest_shape,
        ).astype(bool)
        components = int(
            ndi.label(
                binary,
                structure=ndi.generate_binary_structure(3, 1),
            )[1]
        )
        relative_change = None
        overlap = None
        component_stable = None
        if previous_objective is not None and np.isfinite(objective):
            relative_change = abs(objective - previous_objective) / max(
                abs(objective), 1e-12
            )
        if previous_binary is not None:
            union = int(np.count_nonzero(binary | previous_binary))
            overlap = (
                float(np.count_nonzero(binary & previous_binary)) / float(union)
                if union
                else 1.0
            )
            component_stable = components == previous_components
        records.append(
            {
                "level": index + 1,
                "grid": [study.nelx, study.nely, study.nelz],
                "voxels": int(study.nelx * study.nely * study.nelz),
                "objective": objective,
                "relative_objective_change": relative_change,
                "topology_jaccard": overlap,
                "connected_components": components,
                "component_count_stable": component_stable,
                "iterations": int(result.n_iter),
            }
        )
        previous_objective = objective
        previous_binary = binary
        previous_components = components

    last = records[-1]
    converged = len(records) > 1 and bool(
        last["relative_objective_change"] is not None
        and float(last["relative_objective_change"]) <= 0.05
        and last["topology_jaccard"] is not None
        and float(last["topology_jaccard"]) >= 0.90
        and bool(last["component_count_stable"])
    )
    return {
        "performed": True,
        "converged": converged,
        "criteria": {
            "maximum_relative_objective_change": 0.05,
            "minimum_topology_jaccard": 0.90,
            "require_stable_component_count": True,
        },
        "levels": records,
        "message": (
            "Topology is stable across the final two resolutions."
            if converged
            else "Topology changed materially on refinement; refine again or "
            "revisit the physical length scale before release."
        ),
    }
