# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Recover physical continuum and lattice boundaries from voxel fields."""

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
from .cad_config import SurfaceRecoveryConfig
from .mesh_surface import generate_topopt_triangular_mesh
from .mesh_screening import screen_mesh_intersections

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
    structure_options: Optional[ManufacturingStructureOptions] = None,
    member_plan: Optional[OptimizedMemberPlan] = None,
    **kwargs: Any,
) -> Optional[dict[str, np.ndarray]]:
    """Extract a recovered surface from a topology optimization density field.

    Marching Cubes uses physical voxel spacing. Feature-constrained smoothing
    retains domain boundaries and per-shell signed volume, including cavities.
    """
    if 'blend_radius' in kwargs:
        from ._compat import warn_retired_options
        warn_retired_options({'blend_radius': kwargs.pop('blend_radius')}, operation='Surface recovery')
    def stage(label):
        callback = kwargs.get("progress_stage_callback")
        if callable(callback):
            callback(label)

    try:
        LAST_LATTICE_SIZING.clear()
        # Graph-joint pins are separate assembly hardware. Recover the design
        # body and its bores first, then add the pins after all void constraints.
        hardware_cylinders = joint_pin_cylinders
        joint_pin_cylinders = ()
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
        quality_scale = {"standard": 1., "high": 1.5}.get(quality_key, 2.)
        is_lattice = structure_options is not None and structure_options.mode != "solid"
        refine = 1.
        if is_lattice:
            # Resolve the actual unit cell and wall, rather than upsampling
            # every axis because the part has one thin overall dimension.
            family = structure_options.family
            cell_samples = family.minimum_cell_voxels if family is not None else 12
            refine = max(1., quality_scale*cell_samples/structure_options.cell_size_voxels,
                         3./structure_options.member_thickness_voxels)
        zoom = np.full(3, refine, dtype=float)
        cubic_grid = bool(is_lattice and kwargs.get("cubic_lattice_grid"))
        if cubic_grid:
            # Keep cells physically cubic even when the saved analysis voxels
            # have different X/Y/Z edge lengths.
            nominal_edge = float(np.mean(analysis_cell))
            build_edge = min(float(np.min(analysis_cell)), nominal_edge / refine)
            target_counts = np.ceil(analysis_shape * analysis_cell / build_edge)
            zoom = target_counts / analysis_shape
        point_budget = kwargs.get("max_recovery_voxels")
        if point_budget and float(np.prod(np.round(analysis_shape * zoom))) > int(point_budget):
            raise ValueError(
                "The requested lattice needs more than the build-grid voxel budget. "
                "Increase cell size or minimum wall/strut thickness."
            )
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
        if cubic_grid:
            resolution_scale = float(np.mean(analysis_cell) / np.mean(spacing))
        source_field = _resample_source_mask(source_mask, tuple(grid.shape))
        solid_field = _resample_source_mask(passive_solid_mask, tuple(grid.shape))
        void_field = _resample_source_mask(passive_void_mask, tuple(grid.shape))
        cad_domain = None
        preservation_report = {}
        explicit_solids = kwargs.get("explicit_solid_shapes") or ()
        explicit_voids = kwargs.get("explicit_void_shapes") or ()
        source_cad = kwargs.get("source_cad_shape")
        from .bore_constraints import source_bore_records
        bore_records = source_bore_records(source_cad)
        if source_cad is None:
            # Older saved outputs contain the exact collars but no source CAD.
            # Their inward walls still define the openings to preserve.
            for shape in explicit_solids:
                bore_records.extend(source_bore_records(shape))
        if explicit_solids or explicit_voids or source_cad is not None:
            stage("Preserving original non-design geometry")
            from .preserved_mesh import refined_preserved_masks
            solid_field, void_field, cad_domain, coarse_solids, preservation_report = refined_preserved_masks(
                bounds, tuple(analysis_shape.astype(int)), tuple(grid.shape), solid_field, void_field,
                solid_shapes=explicit_solids, void_shapes=explicit_voids, source_shape=source_cad,
            )
            # Interpolation must not leave the old coarse layer as solid material
            # outside the newly sampled original face layer.
            if is_lattice:
                grid = np.where(coarse_solids & ~solid_field,
                                np.minimum(grid, structure_options.maximum_relative_density), grid)
            if source_field is not None:
                source_field |= solid_field
            if cad_domain is not None:
                source_field = cad_domain if source_field is None else source_field & cad_domain
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
            stage("Generating lattice cells")
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
        if cad_domain is not None:
            grid = np.where(cad_domain, grid, 0.0)

        manufacturing_count = int(np.count_nonzero(grid >= eff_cutoff))
        manufacturing_reference_volume = float(manufacturing_count * np.prod(spacing))

        cfg = SurfaceRecoveryConfig(
            density_cutoff=eff_cutoff,
        )

        def constrain_field(field):
            # Gaussian filtering is part of mesh extraction. Reapply prescribed
            # geometry afterwards so it cannot blur material back into voids.
            field = np.asarray(field, dtype=float).copy()
            if solid_field is not None:
                field[solid_field] = 1.0
            if source_field is not None:
                field[~source_field] = 0.0
            if void_field is not None:
                field[void_field] = 0.0
            if passive_shapes_present:
                field = _apply_passive_density_regions(
                    field, solid_boxes=solid_boxes, void_boxes=void_boxes,
                    solid_cylinders=solid_cylinders, void_cylinders=void_cylinders,
                    joint_pin_cylinders=joint_pin_cylinders,
                    bounds=bounds, cell=analysis_cell, spacing=spacing,
                )
            if cad_domain is not None:
                field[~cad_domain] = 0.0
            if void_field is not None:
                field[void_field] = 0.0
            if bore_records:
                origin, _ = _voxel_origin_cell(tuple(field.shape), bounds)
                coordinates = [
                    (origin[index] + (np.arange(size) + 0.5) * spacing[index]).reshape(
                        tuple(size if axis == index else 1 for axis in range(3))
                    )
                    for index, size in enumerate(field.shape)
                ]
                for axis, center, radius, start, end in bore_records:
                    relative = [coordinates[index] - center[index] for index in range(3)]
                    axial = sum(relative[index] * axis[index] for index in range(3))
                    radial = np.sqrt(np.maximum(
                        sum(value * value for value in relative) - axial * axial, 0.0
                    ))
                    axial += float(np.dot(center, axis))
                    distance = np.maximum.reduce([
                        radial - radius,
                        np.broadcast_to(start - axial, field.shape),
                        np.broadcast_to(axial - end, field.shape),
                    ])
                    # Zero distance must contour at the requested level, not
                    # at 0.5: recovery can use an inverse-projected cutoff.
                    field = np.minimum(field, np.clip(
                        eff_cutoff + distance / float(np.min(spacing)), 0.0, 1.0
                    ))
            return field

        has_constraints = bool(
            source_field is not None or solid_field is not None or void_field is not None
            or passive_shapes_present or bore_records
        )

        stage("Meshing the lattice surface" if is_lattice else "Meshing the solid surface")
        verts, faces, recovery_report = generate_topopt_triangular_mesh(
            grid,
            cutoff=eff_cutoff,
            bounds=bounds,
            config=cfg,
            field_constraint=constrain_field if has_constraints else None,
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
        from .mesh_reduction import compact_mesh
        stage("Reducing mesh triangles")
        verts, faces, reduction_report = compact_mesh(
            verts, faces, tolerance=float(np.min(spacing))*0.2)
        from .bore_constraints import enforce_bore_clearance
        verts, bore_report = enforce_bore_clearance(verts, faces, bore_records, spacing)
        recovery_report["bore_clearance"] = bore_report
        if hardware_cylinders:
            from .cad_features import _payload_region_cylinders, _region_cylinder_to_solid
            hardware_bounds = bounds
            if hardware_bounds is None:
                lower, pitch = _voxel_origin_cell(tuple(grid.shape), None)
                hardware_bounds = (lower, lower + np.asarray(grid.shape) * pitch)
            pins = _payload_region_cylinders(
                {"passive_regions": {"joint_pin_cylinders": list(hardware_cylinders)}},
                "joint_pin_cylinders",
            )
            bodies = [trimesh.Trimesh(verts, faces, process=False)]
            for pin in pins:
                shape = _region_cylinder_to_solid(pin, hardware_bounds)
                if shape is None:
                    continue
                points, triangles = shape.tessellate(float(np.min(spacing)) * 0.1, 0.2)
                bodies.append(trimesh.Trimesh(
                    [point.toTuple() for point in points], triangles, process=True,
                ))
            combined = trimesh.util.concatenate(bodies)
            verts, faces = np.asarray(combined.vertices), np.asarray(combined.faces)
            recovery_report["joint_hardware_body_count"] = len(bodies) - 1
        tm = trimesh.Trimesh(verts, faces, process=False)
        mesh_volume = abs(float(tm.volume)) if tm.is_watertight else None
        is_watertight = bool(tm.is_watertight)
        boundary_shells = tm.split(only_watertight=False)
        component_count = len(boundary_shells)
        if is_lattice and analyzed_reference_volume > 0.0 and mesh_volume is not None:
            lattice_sizing["mesh_relative_density"] = mesh_volume / analyzed_reference_volume
            LAST_LATTICE_SIZING.update(lattice_sizing)

        stage("Checking mesh intersections")
        recovered_shape = {
            "vertices": np.asarray(verts, dtype=float),
            "faces": np.asarray(faces, dtype=np.int64),
            "surface_backend": "Marching Cubes with feature-constrained smoothing",
            "surface_quality_preset": "Feature preserving",
            "surface_quality": {
                **screen_mesh_intersections(verts, faces),
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
            "structure_resolution_scale": resolution_scale,
            "recovery_grid_shape": [int(value) for value in grid.shape],
            "surface_recovery_report": recovery_report,
            "preserved_geometry": preservation_report,
            "mesh_reduction": reduction_report,
            "is_watertight": is_watertight,
        }
        if is_lattice and kwargs.get("include_manufacturing_field"):
            # Preserve explicit resolved cells for downstream voxel FEA; the
            # graded macro field describes their thickness, not solid occupancy.
            recovered_shape["manufactured_density"] = np.asarray(grid >= eff_cutoff, dtype=np.uint8)
        return recovered_shape
    except Exception:
        logger.exception("Voxel boundary recovery failed")
        if kwargs.get("raise_errors"):
            raise
        return None


__all__ = ["_recover_voxel_shape", "LAST_LATTICE_SIZING"]
