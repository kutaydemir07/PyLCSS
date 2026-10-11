# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Surface recovery orchestration powered by AMRTO Step 1 (Ren et al., 2025)."""

from __future__ import annotations

import logging
from dataclasses import asdict
from typing import Any, Optional, Sequence

import numpy as np
import trimesh
from scipy import ndimage as ndi

from ..manufacturing.structures import (
    ManufacturingStructureOptions,
    LAST_STRUCTURE_DIAGNOSTICS,
    achieved_relative_density,
    build_manufacturing_field,
    resolve_target_relative_density,
)
from ..manufacturing.member_sizing import OptimizedMemberPlan
from .analytic_shapes import (
    BoxRegion,
    CylinderRegion,
    _voxel_origin_cell,
    _convert_legacy_to_physical_shapes,
    _project_passive_shapes_surfaces,
)
from .passive_regions import (
    _apply_passive_density_regions,
    _resample_source_mask,
)
from .amrto_common import AMRTOConfig
from .amrto_tri_mesh import generate_topopt_triangular_mesh

logger = logging.getLogger(__name__)

# Diagnostics from the most recent lattice sizing, for callers to report.
LAST_LATTICE_SIZING: dict[str, float] = {}


def _recover_voxel_shape(
    density: np.ndarray,
    bounds: Optional[tuple[np.ndarray, np.ndarray]],
    cutoff: float,
    solid_boxes: Sequence[BoxRegion] = (),
    void_boxes: Sequence[BoxRegion] = (),
    solid_cylinders: Sequence[CylinderRegion] = (),
    void_cylinders: Sequence[CylinderRegion] = (),
    joint_pin_cylinders: Sequence[CylinderRegion] = (),
    extrusion_axis: str = "none",
    source_mask: Optional[np.ndarray] = None,
    passive_solid_mask: Optional[np.ndarray] = None,
    passive_void_mask: Optional[np.ndarray] = None,
    blend_radius: float = 0.0,
    structure_options: Optional[ManufacturingStructureOptions] = None,
    member_plan: Optional[OptimizedMemberPlan] = None,
    **kwargs: Any,
) -> Optional[dict[str, np.ndarray]]:
    """Extract a recovered surface from a topology optimization density field.

    Powered by Step 1 of the AMRTO pipeline (Automated CAD Model Reconstruction
    of Topology Optimization Result, Ren et al., Comput. Methods Appl. Mech.
    Engrg. 435 (2025) 117673):
    - Smooths the pseudo-density field
    - Extracts the isosurface using Marching Cubes mapped to physical coordinates
    - Repairs the triangular mesh via Laplacian smoothing, normal correction,
      gap filling, and degenerate patch elimination.
    """
    try:
        LAST_LATTICE_SIZING.clear()
        grid = np.asarray(density, dtype=float)
        if grid.ndim != 3 or min(grid.shape) < 1:
            return None

        grid = np.nan_to_num(grid, nan=0.0, posinf=1.0, neginf=0.0)
        if float(np.max(grid)) <= 0.0:
            return None

        eff_cutoff = float(np.clip(cutoff if cutoff is not None else 0.5, 1e-4, 0.9999))

        analysis_shape = np.asarray(grid.shape, dtype=float)
        _, analysis_cell = _voxel_origin_cell(tuple(grid.shape), bounds)
        source_solid = grid >= eff_cutoff
        if source_mask is not None and np.shape(source_mask) == grid.shape:
            source_solid &= np.asarray(source_mask, dtype=bool)
        if passive_solid_mask is not None and np.shape(passive_solid_mask) == grid.shape:
            source_solid |= np.asarray(passive_solid_mask, dtype=bool)
        if passive_void_mask is not None and np.shape(passive_void_mask) == grid.shape:
            source_solid &= ~np.asarray(passive_void_mask, dtype=bool)
        analyzed_reference_volume = float(np.count_nonzero(source_solid) * np.prod(analysis_cell))

        # Resolve periodic cells on a manufacturing grid, retaining the physical
        # cell pitch. Refinement is a geometry operation, independent of FE size.
        quality_key = str(kwargs.get("surface_quality") or "Professional").strip().lower()
        min_samples, point_budget = {"standard": (24, 1_000_000), "high": (36, 2_000_000)}.get(
            quality_key, (48, 3_500_000)
        )
        is_lattice = structure_options is not None and structure_options.mode != "solid"
        if not is_lattice:
            # Solid continuum topology density is already computed on the FE grid.
            # Avoid isotropic over-refinement driven by thin structural dimensions
            # (e.g., thickness=8 driving 6x zoom and 8M voxels).
            refine = 1.0
            if max(grid.shape) < min_samples:
                refine = max(1.0, float(np.ceil(min_samples / max(grid.shape))))
            refine = min(refine, 2.0)
        else:
            refine = max(1.0, float(np.ceil(min_samples / min(grid.shape))))
            if structure_options.family is not None:
                refine = max(refine, float(np.ceil(
                    structure_options.family.minimum_cell_voxels / structure_options.cell_size_voxels
                )))
            refine = min(refine, max(1.0, (point_budget / float(grid.size)) ** (1.0 / 3.0)), 12.0)
        zoom = np.full(3, refine, dtype=float)
        extrusion_index = {"x": 0, "y": 1, "z": 2}.get(str(extrusion_axis or "").lower())
        if extrusion_index is not None and not is_lattice:
            grid = np.broadcast_to(np.mean(grid, axis=extrusion_index, keepdims=True), grid.shape).copy()
            retained_count = int(np.count_nonzero(source_solid))
            if 0 < retained_count < grid.size:
                ordered = np.partition(grid.ravel(), grid.size - retained_count)
                eff_cutoff = float(np.clip(ordered[grid.size - retained_count], 1e-4, 0.9999))
            zoom[extrusion_index] = max(1.0, 4.0 / grid.shape[extrusion_index])
        if np.any(zoom > 1.0):
            grid = ndi.zoom(grid, zoom=tuple(zoom), order=1, mode="nearest", grid_mode=True)
        # scipy rounds each resulting dimension. Use the realized dimensions
        # for physical spacing and volume rather than the requested zoom.
        zoom = np.asarray(grid.shape, dtype=float) / analysis_shape
        resolution_scale = float(np.max(zoom))
        spacing = analysis_cell / zoom
        source_field = _resample_source_mask(source_mask, tuple(grid.shape))
        solid_field = _resample_source_mask(passive_solid_mask, tuple(grid.shape))
        void_field = _resample_source_mask(passive_void_mask, tuple(grid.shape))
        if source_field is not None:
            grid = np.where(source_field, grid, 0.0)
        if solid_field is not None:
            grid = np.where(solid_field, 1.0, grid)
        if void_field is not None:
            grid = np.where(void_field, 0.0, grid)
        grid = _apply_passive_density_regions(
            grid, solid_boxes=solid_boxes, void_boxes=void_boxes,
            solid_cylinders=solid_cylinders, void_cylinders=void_cylinders,
            bounds=bounds, cell=analysis_cell, spacing=spacing,
        )
        if void_field is not None:
            grid = np.where(void_field, 0.0, grid)
        build_options = structure_options
        lattice_sizing: dict[str, Any] = {}
        lattice_connectivity: dict[str, Any] = {}

        # Handle lattice manufacturing structures if requested
        if is_lattice:
            if structure_options.target_relative_density > 0.0:
                build_options, reached = resolve_target_relative_density(
                    grid, eff_cutoff, structure_options, resolution_scale=resolution_scale,
                    passive_solid_mask=solid_field, passive_void_mask=void_field, member_plan=member_plan,
                )
                lattice_sizing = {
                    "target_relative_density": float(structure_options.target_relative_density),
                    "achieved_relative_density": float(reached),
                    "member_thickness_voxels": float(build_options.member_thickness_voxels),
                    "member_thickness_model_units": float(build_options.member_thickness_voxels * np.mean(analysis_cell)),
                    "member_scale": float(build_options.member_scale),
                }
            envelope = grid.copy()
            grid = build_manufacturing_field(
                grid, eff_cutoff, build_options, resolution_scale=resolution_scale,
                passive_solid_mask=solid_field, passive_void_mask=void_field, member_plan=member_plan,
            )
            lattice_connectivity = dict(LAST_STRUCTURE_DIAGNOSTICS)
            lattice_sizing.setdefault("achieved_relative_density", achieved_relative_density(grid, envelope, eff_cutoff))
            LAST_LATTICE_SIZING.update(lattice_sizing)
            eff_cutoff = 0.5

        # Apply passive solid / void masks if provided
        if solid_field is not None:
            grid = np.maximum(grid, np.asarray(solid_field, dtype=float))
        if source_field is not None:
            grid = np.where(source_field, grid, 0.0)

        # Apply passive analytic shapes to density field
        passive_shapes_present = bool(
            solid_cylinders
            or void_cylinders
            or solid_boxes
            or void_boxes
            or joint_pin_cylinders
        )
        if passive_shapes_present:
            grid = _apply_passive_density_regions(
                grid,
                solid_boxes=solid_boxes,
                void_boxes=void_boxes,
                solid_cylinders=solid_cylinders,
                void_cylinders=void_cylinders,
                joint_pin_cylinders=joint_pin_cylinders,
                bounds=bounds, cell=analysis_cell, spacing=spacing,
            )
        if void_field is not None:
            grid = np.where(void_field, 0.0, grid)
        if joint_pin_cylinders:
            grid = _apply_passive_density_regions(
                grid, solid_cylinders=joint_pin_cylinders,
                bounds=bounds, cell=analysis_cell, spacing=spacing,
            )

        manufacturing_count = int(np.count_nonzero(grid >= eff_cutoff))
        manufacturing_reference_volume = float(manufacturing_count * np.prod(spacing))

        cfg = AMRTOConfig(
            density_cutoff=eff_cutoff,
        )

        verts, faces, step1_report = generate_topopt_triangular_mesh(
            grid,
            cutoff=eff_cutoff,
            bounds=bounds,
            config=cfg,
        )

        if len(verts) == 0 or len(faces) == 0:
            return None

        # Reproject analytic interface boundaries if passive shapes are present
        if passive_shapes_present:
            try:
                shapes = _convert_legacy_to_physical_shapes(
                    bounds,
                    solid_boxes=solid_boxes,
                    void_boxes=void_boxes,
                    solid_cylinders=solid_cylinders,
                    void_cylinders=void_cylinders,
                    joint_pin_cylinders=joint_pin_cylinders,
                )
                if shapes:
                    verts = _project_passive_shapes_surfaces(
                        verts, shapes, tolerance=float(np.max(spacing)) * 0.5
                    )
            except Exception as exc:
                logger.debug("Passive shapes surface projection skipped: %s", exc)

        # Remove only sub-resolution contour specks, retaining every member-sized
        # body. This is recorded explicitly rather than hidden in a volume proxy.
        tm = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        removed_components = 0
        removed_volume = 0.0
        if is_lattice:
            bodies = tm.split(only_watertight=False)
            def subresolution(body: trimesh.Trimesh) -> bool:
                # Clipping a sheet to a curved envelope can leave long boundary
                # whiskers with many triangles but no voxel-wide cross-section.
                return len(body.faces) <= 8 or np.count_nonzero(
                    np.asarray(body.extents) < spacing
                ) >= 2

            # An inward shell bounds a cavity, even when it is very small.
            # Removing it would add material rather than remove contour dust.
            specks = [body for body in bodies if body.volume > 0 and subresolution(body)]
            kept = [body for body in bodies if body.volume <= 0 or not subresolution(body)]
            candidate_volume = float(sum(abs(body.volume) for body in specks if body.is_watertight))
            # A missing load-bearing member must never be disguised as cleanup.
            # Cull only negligible contour artifacts, below 0.1% of total mass.
            if tm.is_watertight and candidate_volume > abs(float(tm.volume)) * 1e-3:
                kept, specks = list(bodies), []
            if kept and specks:
                removed_components = len(specks)
                removed_volume = candidate_volume
                tm = trimesh.util.concatenate(kept)
                verts, faces = np.asarray(tm.vertices), np.asarray(tm.faces)
        mesh_volume = abs(float(tm.volume)) if tm.is_watertight else None
        is_watertight = bool(tm.is_watertight)
        boundary_shells = tm.split(only_watertight=False)
        component_count = len(boundary_shells)
        if is_lattice and analyzed_reference_volume > 0.0 and mesh_volume is not None:
            lattice_sizing["mesh_relative_density"] = mesh_volume / analyzed_reference_volume
            LAST_LATTICE_SIZING.update(lattice_sizing)

        recovered_shape = {
            "vertices": np.asarray(verts, dtype=float),
            "faces": np.asarray(faces, dtype=np.int64),
            "surface_backend": "AMRTO Step 1 (Marching Cubes + Laplacian Smoothing)",
            "surface_quality_preset": "AMRTO",
            "surface_quality": {
                "connected_components": int(component_count),
                "boundary_shell_count": int(component_count),
                "positive_outer_shell_count": sum(body.volume > 0 for body in boundary_shells),
                "negative_cavity_shell_count": sum(body.volume < 0 for body in boundary_shells),
                "component_semantics": "Connected surface shells; inward shells bound cavities",
                "removed_subresolution_components": int(removed_components),
                "removed_subresolution_volume": float(removed_volume),
                "is_watertight": is_watertight,
            },
            "manufacturing_structure": (
                structure_options.display_name
                if structure_options is not None
                else "Solid Envelope"
            ),
            "manufacturing_reference_volume": manufacturing_reference_volume if is_lattice else None,
            "manufacturing_reference_sample_count": manufacturing_count,
            "analyzed_reference_volume": analyzed_reference_volume,
            "recovery_target_volume": manufacturing_reference_volume if is_lattice else analyzed_reference_volume,
            "volume": mesh_volume,
            "lattice_sizing": dict(lattice_sizing),
            "lattice_connectivity": dict(lattice_connectivity),
            "resolved_structure_options": asdict(build_options) if is_lattice else None,
            "effective_recovery_cutoff": float(eff_cutoff),
            "protected_profile_points": np.empty((0, 3), dtype=float),
            "structure_resolution_scale": resolution_scale,
            "recovery_grid_shape": [int(value) for value in grid.shape],
            "stage1_report": step1_report,
            "is_watertight": is_watertight,
        }
        return recovered_shape
    except Exception:
        logger.exception("Voxel shape recovery failed via AMRTO")
        return None


__all__ = ["_recover_voxel_shape", "LAST_LATTICE_SIZING"]
