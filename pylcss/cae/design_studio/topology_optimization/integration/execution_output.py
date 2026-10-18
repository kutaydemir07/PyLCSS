# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Recover geometry and assemble the final topology node payload."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..geometry.surface_recovery import _recover_voxel_shape
from ..manufacturing.structures import (
    ManufacturingStructureOptions,
    build_manufacturing_field,
    cell_fit_warning,
    cell_resolution_warning,
    passive_region_masks,
)
from ..optimization.results import TopologyOptVoxelResult
from ..optimization.voxel_solver import _projection_matched_level
from .boundary_mapping import _bounds_payload
from .execution_setup import PreparedTopologyStudy
from .lattice_settings import resolve_lattice_structure_options
from .voxelization import (
    _effective_density_cutoff,
    _fractional_cylinder_volume,
    _meaningful_material_components,
    _recovered_shape_volume,
    _source_material_fraction,
    _source_volume_fraction,
    voxel_size_from_bounds,
)

logger = logging.getLogger(__name__)

# Invalidate surfaces recovered before certified final CAD mesh restoration.
_SURFACE_RECOVERY_CACHE_REVISION = 6


def _cad_shape_cache_identity(shape: Any) -> Any:
    """Identify current CAD contents, including edits to an existing TShape."""
    if shape is None:
        return None
    if isinstance(shape, dict):
        for key in ("shape", "cad", "geometry", "components", "mesh", "recovered_shape"):
            nested = shape.get(key)
            if nested is not None and nested is not shape:
                return _cad_shape_cache_identity(nested)
    if isinstance(shape, (list, tuple)):
        return tuple(_cad_shape_cache_identity(item) for item in shape)
    if hasattr(shape, "val"):
        shape = shape.val()
    from ..geometry.cad_features import _coerce_to_solid
    from ..geometry.cad_fingerprint import cad_shape_fingerprint

    cad_shape = _coerce_to_solid(shape)
    if cad_shape is not None:
        return cad_shape_fingerprint(cad_shape)
    from .geometry_mapping import _surface_mesh_arrays

    arrays = _surface_mesh_arrays(shape, allow_tessellation=False)
    if arrays is not None:
        return tuple((array.shape, hash(np.ascontiguousarray(array).tobytes())) for array in arrays)
    hash_code = getattr(shape, "hashCode", None)
    return hash_code() if callable(hash_code) else id(shape)


@dataclass(frozen=True)
class TopologyOutputContext:
    """Recovered output and state needed for geometry and optimizer checks."""

    payload: dict[str, Any]
    manufactured_density: np.ndarray | None
    structure_options: ManufacturingStructureOptions
    result: TopologyOptVoxelResult
    study: PreparedTopologyStudy


def _select_recovery_field(
    result: TopologyOptVoxelResult,
    density: np.ndarray,
    density_cutoff: float,
) -> tuple[np.ndarray, float]:
    """Select the smooth field used by surface recovery, and its matching level.

    Recovery contours the filtered field because the projected one is nearly
    binary and has no usable sub-voxel gradient. The level has to describe the
    same boundary the user's ``density_cutoff`` describes on the projected
    field, and because the projection is strictly increasing that level is its
    exact inverse image -- recomputed here rather than reused from the solve, so
    that changing the cutoff without re-solving moves the recovered boundary
    with it.
    """
    filtered = (
        np.asarray(result.recovery_density, dtype=float)
        if result.recovery_density is not None
        else None
    )
    if filtered is not None and filtered.shape == density.shape:
        if result.projection_beta is None:
            # No projection in the network: filtered is the physical field.
            return filtered, float(density_cutoff)
        return filtered, _projection_matched_level(
            float(density_cutoff),
            beta=float(result.projection_beta),
            eta=float(
                result.projection_eta if result.projection_eta is not None else 0.5
            ),
        )
    return np.asarray(density, dtype=float), float(density_cutoff)


def _objective_reduction_metrics(
    result: TopologyOptVoxelResult,
) -> tuple[float | None, float | None, int]:
    """Compare objectives only within a common continuation stage.

    Penalization continuation deliberately changes the optimization problem,
    so comparing the first low-beta objective with the final high-beta value
    can report an apparent deterioration even when every final-stage step
    improved the design. Return the comparable final-stage reduction, the
    full-run change for diagnostics, and the baseline history index.
    """
    history = [float(value) for value in result.objective_history]
    if len(history) < 2:
        return None, None, 0

    baseline_index = 0
    beta_history = [float(value) for value in result.beta_history]
    if len(beta_history) == len(history) and beta_history:
        final_beta = beta_history[-1]
        baseline_index = next(
            (
                index
                for index, value in enumerate(beta_history)
                if np.isclose(value, final_beta, rtol=1e-9, atol=1e-12)
            ),
            0,
        )

    def reduction(start: float, finish: float) -> float:
        return 100.0 * (1.0 - finish / max(abs(start), 1.0e-30))

    comparable = (
        reduction(history[baseline_index], history[-1])
        if baseline_index < len(history) - 1
        else 0.0
    )
    full_run = reduction(history[0], history[-1])
    return comparable, full_run, baseline_index


def _loads_stranded_in_void(result: Any, problem: Any) -> int:
    """Count point loads sitting on near-void material in the final design.

    A manufacturing projection such as pull-out or overhang can delete the
    material under a load. The solve still returns a number, but it is the
    compliance of a structure that is not actually loaded, so it has to be
    reported rather than silently believed.
    """
    try:
        density = np.asarray(result.density, dtype=float)
        if density.ndim != 3:
            return 0
        shape = np.asarray(density.shape, dtype=float) - 1.0
    except Exception:
        return 0

    load_cases = list(getattr(problem.bc, "load_cases", []) or [])
    if not load_cases:
        load_cases = [problem.bc]
    stranded = 0
    for case in load_cases:
        for point in list(getattr(case, "point_forces", []) or []):
            if len(point) < 6:
                continue
            try:
                fractions = np.clip(np.asarray(point[:3], dtype=float), 0.0, 1.0)
                magnitude = float(np.linalg.norm(np.asarray(point[3:6], dtype=float)))
            except Exception:
                continue
            if magnitude <= 1e-12:
                continue
            index = np.rint(fractions * shape).astype(int)
            index = np.clip(index, 0, np.asarray(density.shape) - 1)
            if float(density[tuple(index)]) < 0.1:
                stranded += 1
    return stranded


def build_topology_output(
    node: Any,
    result: TopologyOptVoxelResult,
    study: PreparedTopologyStudy,
) -> TopologyOutputContext | None:
    """Recover the optimized surface and construct the base result payload."""
    bounds = study.bounds
    material = study.material
    bc = study.bc
    design_domain = study.design_domain
    design_goal = study.design_goal
    explicit_passive_solid_mask = study.passive_solid_mask
    explicit_passive_void_mask = study.passive_void_mask
    problem = study.problem

    def _selection_type_counts(entries: list[Any]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            kind = str(entry.get("selection_entity_type") or "Unknown").title()
            counts[kind] = counts.get(kind, 0) + 1
        return counts

    support_selection_types = _selection_type_counts(study.constraints)
    load_selection_types = _selection_type_counts(study.loads)
    logger.info("TopologyOptVoxelNode: %s", result.message)
    density = np.asarray(result.density, dtype=float)
    density_cutoff = _effective_density_cutoff(
        node.get_property("density_cutoff"),
        density=density,
        target_volfrac=problem.volfrac,
    )
    visualization_mode = (
        "Density"
        if str(node.get_property("visualization") or "STL Shape") == "Density"
        else "STL Shape"
    )
    # Lattice dimensions given in model units win over the voxel-denominated
    # ones, because a voxel is a different physical length on every part.
    voxel_size = voxel_size_from_bounds(tuple(int(v) for v in density.shape), bounds)
    try:
        structure_options = resolve_lattice_structure_options(
            node,
            tuple(int(v) for v in density.shape),
            bounds,
        )
    except (TypeError, ValueError) as exc:
        node.set_error(f"Invalid lattice geometry: {exc}")
        return None
    # The CAD stage needs to know which structure was built: a strut lattice is
    # reconstructed from its centrelines rather than from the isosurface.
    setattr(node, "_last_structure_options", structure_options)
    passive_solid_mask, passive_void_mask = passive_region_masks(
        tuple(int(v) for v in density.shape),
        solid_boxes=bc.solid_boxes,
        void_boxes=bc.void_boxes,
        solid_cylinders=bc.solid_cylinders,
        void_cylinders=bc.void_cylinders,
    )
    passive_solid_mask |= explicit_passive_solid_mask
    passive_void_mask |= explicit_passive_void_mask
    passive_solid_mask &= ~passive_void_mask
    member_plan = None
    setattr(node, "_last_member_plan", None)
    is_infill_study = callable(getattr(node, "resolve_infill_cell_size", None))
    manufactured_density = None
    if structure_options.mode != "solid":
        manufactured_density = build_manufacturing_field(
            density,
            density_cutoff,
            structure_options,
            passive_solid_mask=passive_solid_mask,
            passive_void_mask=passive_void_mask,
            member_plan=member_plan,
        )
    connectivity_density = (
        manufactured_density if manufactured_density is not None else density
    )
    component_count, component_voxels = _meaningful_material_components(
        connectivity_density,
        0.5 if manufactured_density is not None else density_cutoff,
        design_domain,
    )
    source_component_count, _ = _meaningful_material_components(
        np.asarray(design_domain, dtype=float),
        0.5,
        design_domain,
    )
    # Recover the delivered blueprint on its smoother filtered field. The
    # inverse-projected cutoff describes the corresponding nominal boundary;
    # robust studies analyze the separate eroded density field.
    recovery_field, recovery_cutoff = _select_recovery_field(
        result,
        density,
        density_cutoff,
    )

    # Surface recovery needs the original boundaries as well as their coarse
    # analysis masks: interpolating those masks can intrude into a CAD bore.
    explicit_solid_shapes: list[Any] = []
    explicit_void_shapes: list[Any] = []
    for region in getattr(study, "non_design_regions", ()):
        geom = region.get("geometry") if isinstance(region, dict) else region
        rtype = (
            str(region.get("region_type") or "solid").lower()
            if isinstance(region, dict)
            else "solid"
        )
        if geom is not None:
            if rtype == "void":
                explicit_void_shapes.append(geom)
            else:
                explicit_solid_shapes.append(geom)

    source_cad_shape = None
    try:
        from ..geometry.non_design import design_domain_shape

        source_cad_shape = design_domain_shape(study.source_design_domain)
    except (ValueError, TypeError):
        pass

    def recover_surface(
        field: np.ndarray,
        cutoff: float,
    ) -> dict[str, Any] | None:
        return _recover_voxel_shape(
            field,
            bounds,
            cutoff,
            print_ready=True,
            decimate_ratio=1.0,
            solid_boxes=bc.solid_boxes,
            void_boxes=bc.void_boxes,
            solid_cylinders=bc.solid_cylinders,
            void_cylinders=bc.void_cylinders,
            joint_pin_cylinders=bc.joint_pin_cylinders,
            source_mask=design_domain,
            passive_solid_mask=passive_solid_mask,
            passive_void_mask=passive_void_mask,
            explicit_solid_shapes=explicit_solid_shapes,
            explicit_void_shapes=explicit_void_shapes,
            source_cad_shape=source_cad_shape,
            source_lattice_geometry=(source_cad_shape if source_cad_shape is not None else study.source_design_domain) if is_infill_study else None,
            include_manufacturing_field=is_infill_study,
            raise_errors=is_infill_study,
            structure_options=structure_options,
            member_plan=member_plan,
        )

    # Key on the actual density bytes — not id(result.density), which Python's
    # allocator can reuse across runs and silently return a stale recovered
    # shape for a different solve.
    density_view = np.ascontiguousarray(recovery_field)
    cache_key = (
        hash(density_view.tobytes()),
        density_view.shape,
        hash(np.ascontiguousarray(density).tobytes()),
        tuple(np.asarray(bounds, dtype=float).ravel()),
        hash(np.ascontiguousarray(design_domain, dtype=bool).tobytes()),
        recovery_cutoff,
        structure_options,
        str(bc.solid_boxes),
        str(bc.void_boxes),
        str(bc.solid_cylinders),
        str(bc.void_cylinders),
        str(bc.joint_pin_cylinders),
        hash(np.ascontiguousarray(passive_solid_mask).tobytes()),
        hash(np.ascontiguousarray(passive_void_mask).tobytes()),
        _SURFACE_RECOVERY_CACHE_REVISION,
        _cad_shape_cache_identity(source_cad_shape),
        _cad_shape_cache_identity(study.source_design_domain) if is_infill_study else None,
        tuple(_cad_shape_cache_identity(shape) for shape in explicit_solid_shapes),
        tuple(_cad_shape_cache_identity(shape) for shape in explicit_void_shapes),
    )
    if getattr(node, "_last_recovery_key", None) == cache_key:
        recovered = node._last_recovered_shape
    else:
        recovered = recover_surface(recovery_field, recovery_cutoff)

        # A volume-matched filtered field normally gives the smoothest surface,
        # but a very grey coupled solution can lose a narrow load path during
        # resampling even though the analyzed projected density is connected.
        # Never publish that topological contradiction. Retry the projected
        # field and use it only when it restores the component count actually
        # analyzed by the solver.
        recovered_components = int(
            (recovered or {}).get("surface_quality", {}).get("connected_components", 0)
            or 0
        )
        recovered_coverage_raw = (recovered or {}).get(
            "protected_voxel_center_coverage"
        )
        recovered_coverage = (
            float(recovered_coverage_raw)
            if recovered_coverage_raw is not None
            else None
        )
        filtered_recovery_used = (
            result.recovery_density is not None and recovery_field is not density
        )
        if (
            structure_options.mode == "solid"
            and filtered_recovery_used
            and component_count > 0
            and (
                recovered_components != component_count
                or (
                    recovered_coverage is not None
                    and recovered_coverage < 1.0 - 1.0e-12
                )
            )
        ):
            projected_recovery = recover_surface(density, density_cutoff)
            projected_components = int(
                (projected_recovery or {})
                .get("surface_quality", {})
                .get("connected_components", 0)
                or 0
            )
            projected_coverage_raw = (projected_recovery or {}).get(
                "protected_voxel_center_coverage"
            )
            projected_coverage = (
                float(projected_coverage_raw)
                if projected_coverage_raw is not None
                else None
            )
            filtered_component_error = abs(recovered_components - component_count)
            projected_component_error = abs(projected_components - component_count)
            coverage_improved = bool(
                projected_coverage is not None
                and (
                    recovered_coverage is None
                    or projected_coverage > recovered_coverage + 1.0e-12
                )
            )
            if (
                projected_recovery is not None
                and projected_components > 0
                and (
                    projected_component_error < filtered_component_error
                    or coverage_improved
                )
            ):
                recovered = projected_recovery
                recovered["recovery_field_fallback"] = (
                    "projected density better retained analyzed connectivity "
                    "and above-cutoff voxel coverage"
                )
        node._last_recovery_key = cache_key
        node._last_recovered_shape = recovered
    requested_structure_options = structure_options
    resolved_options = (
        recovered.get("resolved_structure_options")
        if isinstance(recovered, dict)
        else None
    )
    if isinstance(resolved_options, dict):
        resolved_options = ManufacturingStructureOptions(**resolved_options)
    if (
        isinstance(resolved_options, ManufacturingStructureOptions)
        and resolved_options.mode != "solid"
    ):
        # Recovery sizes actual walls/struts to the requested mass. Export and
        # analysis-grid previews must use those same dimensions, rather than
        # the unsized initial one-voxel member thickness.
        structure_options = resolved_options
        setattr(node, "_last_structure_options", structure_options)
        manufactured_density = build_manufacturing_field(
            density,
            density_cutoff,
            structure_options,
            passive_solid_mask=passive_solid_mask,
            passive_void_mask=passive_void_mask,
            member_plan=member_plan,
        )
    if isinstance(recovered, dict) and recovered.get("recovery_field_fallback"):
        recovery_field = density
        recovery_cutoff = density_cutoff
    if is_infill_study and isinstance(recovered, dict):
        geometric_density = recovered.get("manufactured_density")
        if geometric_density is not None:
            manufactured_density = geometric_density
    if isinstance(recovered, dict):
        try:
            effective_cutoff = float(recovered.get("effective_recovery_cutoff"))
            if np.isfinite(effective_cutoff):
                recovery_cutoff = effective_cutoff
        except (TypeError, ValueError):
            pass
    if bounds is not None:
        mins, maxs = bounds
        total_volume = float(np.prod(np.maximum(maxs[:3] - mins[:3], 0.0)))
    else:
        total_volume = float(np.prod(density.shape))
    source_volume_fraction = _source_volume_fraction(density, design_domain)
    source_volume = float(total_volume * source_volume_fraction)
    source_material_fraction = _source_material_fraction(density, design_domain)
    source_mask = np.asarray(design_domain, dtype=bool)
    source_values = (
        density[source_mask]
        if source_mask.shape == density.shape and np.any(source_mask)
        else density.ravel()
    )
    intermediate_density_fraction = (
        float(np.mean((source_values > 0.10) & (source_values < 0.90)))
        if source_values.size
        else 0.0
    )
    final_volume = float(source_material_fraction * source_volume)
    material_density = float(material.get("rho", material.get("density", 0.0)))
    analysis_grid_manufactured_material_fraction = (
        _source_material_fraction(manufactured_density, design_domain)
        if manufactured_density is not None
        else source_material_fraction
    )
    analysis_grid_manufactured_volume = float(
        analysis_grid_manufactured_material_fraction * source_volume
    )
    recovered_assembly_volume = _recovered_shape_volume(recovered)
    assembly_hardware_volume = float(
        sum(_fractional_cylinder_volume(pin, bounds) for pin in bc.joint_pin_cylinders)
    )
    recovered_design_volume = (
        max(0.0, recovered_assembly_volume - assembly_hardware_volume)
        if recovered_assembly_volume is not None
        else None
    )
    recovery_grid_manufactured_volume = None
    if manufactured_density is not None and isinstance(recovered, dict):
        try:
            candidate = float(recovered.get("manufacturing_reference_volume"))
            if np.isfinite(candidate) and candidate > 0.0:
                recovery_grid_manufactured_volume = candidate
        except (TypeError, ValueError):
            pass
    recovery_reference_volume = (
        recovery_grid_manufactured_volume
        if recovery_grid_manufactured_volume is not None
        else analysis_grid_manufactured_volume
        if manufactured_density is not None
        else final_volume
    )
    manufactured_volume = recovery_reference_volume
    manufactured_material_fraction = (
        manufactured_volume / source_volume if source_volume > 1e-12 else 0.0
    )
    (
        objective_reduction_pct,
        objective_full_run_change_pct,
        objective_baseline_index,
    ) = _objective_reduction_metrics(result)
    recovery_volume_delta_pct = (
        100.0 * (recovered_design_volume / recovery_reference_volume - 1.0)
        if (recovered_design_volume is not None and recovery_reference_volume > 1e-12)
        else None
    )
    voxel_edge_lengths = np.asarray(
        (problem.unitx, problem.unity, problem.unitz),
        dtype=float,
    )
    length_scale_active_axes = [0, 1, 2]
    limiting_voxel_edge = float(np.max(voxel_edge_lengths[length_scale_active_axes]))

    def _feature_elements(value: float) -> float | None:
        physical = float(value or 0.0)
        if physical <= 0.0 or limiting_voxel_edge <= 0.0:
            return None
        return physical / limiting_voxel_edge

    output: dict[str, Any] = {
        "type": "topopt_voxel",
        "source_cad_shape": source_cad_shape,
        "source_lattice_geometry": recovered.get("source_lattice_geometry") if is_infill_study and isinstance(recovered, dict) else None,
        "design_goal": design_goal,
        "objective_mode": problem.objective_mode,
        "physics_mode": problem.physics_mode,
        "physics_model": "Linear-elastic, small-displacement structural response.",
        "optimizer_used": result.optimizer_used,
        "formulation_used": result.formulation_used,
        "physical_length_scale": {
            "requested_minimum_solid_size": float(
                node.get_property("topology_minimum_solid_size_mm") or 0.0
            ),
            "requested_minimum_void_size": float(
                node.get_property("topology_minimum_void_size_mm") or 0.0
            ),
            "filter_radius": float(problem.rmin),
            "filter_radius_units": (
                "model units" if problem.filter_radius_is_physical else "voxel units"
            ),
            "minimum_solid_size": float(problem.minimum_solid_size),
            "minimum_void_size": float(problem.minimum_void_size),
            "minimum_solid_size_elements": _feature_elements(
                problem.minimum_solid_size
            ),
            "minimum_void_size_elements": _feature_elements(problem.minimum_void_size),
            "voxel_edge_lengths": [float(value) for value in voxel_edge_lengths],
            "active_topology_axes": ["XYZ"[axis] for axis in length_scale_active_axes],
        },
        "load_aggregation": problem.load_aggregation,
        "stress_aggregation": (
            {
                "method": "adaptive-scaled p-norm of von Mises stress",
                "pnorm_exponent": float(problem.stress_pnorm_p),
                "qp_relaxation_exponent": float(problem.stress_penalty),
                "scaling_target": "maximum",
                "scaling_damping": float(problem.stress_scaling_damping),
                "allowable_stress": float(problem.yield_stress),
            }
            if problem.stress_constraint_enabled
            else None
        ),
        "density": density,
        "manufactured_density": manufactured_density,
        "design_domain": design_domain,
        "passive_solid_mask": passive_solid_mask,
        "passive_void_mask": passive_void_mask,
        "design_density": (
            np.asarray(result.design_density, dtype=float)
            if result.design_density is not None
            else None
        ),
        "grid_shape": density.shape,
        "bounds": _bounds_payload(bounds),
        # What was actually built. The CAD stage needs it to know whether this
        # result has an exact analytic body available, and carrying it in the
        # payload means the viewer preview and the export worker route the same
        # way as the solve without each holding its own copy.
        "structure_options": structure_options,
        "requested_structure_options": requested_structure_options,
        "member_plan": member_plan,
        "density_cutoff": density_cutoff,
        # The smooth-boundary viewer contours a field the same way recovery
        # does, so it has to be handed the same field. Contouring the projected
        # `density` instead showed the viewer a terraced boundary that did not
        # match the recovered shape exported from the very same solve.
        "recovery_density": recovery_field,
        "recovery_cutoff": recovery_cutoff,
        "recovery_field_fallback": (
            recovered.get("recovery_field_fallback")
            if isinstance(recovered, dict)
            else None
        ),
        "beta_history": list(result.beta_history),
        "projection_eta_history": list(result.projection_eta_history),
        "recovered_shape": recovered,
        "surface_recovery_requested": True,
        "visualization_mode": visualization_mode,
        "target_vol_frac": problem.volfrac,
        "final_vol_frac": source_material_fraction,
        "intermediate_density_fraction": intermediate_density_fraction,
        "active_target_vol_frac": float(result.active_target_volfrac or 0.0),
        "passive_source_vol_frac": float(result.passive_source_volfrac or 0.0),
        "minimum_source_vol_frac": float(result.min_source_volfrac or 0.0),
        "bounding_vol_frac": float(np.mean(density)) if density.size else 0.0,
        "source_volume_fraction": source_volume_fraction,
        "volume": final_volume,
        "density_equivalent_volume": final_volume,
        "total_volume": total_volume,
        "source_volume": source_volume,
        "mass": final_volume * material_density,
        "density_equivalent_mass": final_volume * material_density,
        "recovered_design_volume": recovered_design_volume,
        "recovered_design_mass": (
            recovered_design_volume * material_density
            if recovered_design_volume is not None
            else None
        ),
        "recovered_assembly_volume": recovered_assembly_volume,
        "recovered_assembly_mass": (
            recovered_assembly_volume * material_density
            if recovered_assembly_volume is not None
            else None
        ),
        "assembly_hardware_volume": assembly_hardware_volume,
        "recovery_volume_delta_pct": recovery_volume_delta_pct,
        "recovery_volume_reference": (
            "manufactured structure"
            if manufactured_density is not None
            else "density-equivalent design"
        ),
        "manufacturing": {
            "structure": structure_options.display_name,
            "process": "Unrestricted",
            "boundary_interfaces": {
                "support_selection_types": support_selection_types,
                "load_selection_types": load_selection_types,
                "recommendation": (
                    "Use physical CAD faces for production load/support "
                    "interfaces. Edge and vertex selections are zero-area "
                    "academic idealizations and are mesh-regularized."
                ),
            },
            "lattice_settings_mode": (
                "Direct Infill"
                if structure_options.family is not None
                else "Not Applicable"
            ),
            "cell_size_voxels": structure_options.cell_size_voxels,
            # The length one analysis voxel measures, so every voxel-denominated
            # dimension above can be checked against a process capability
            # without re-deriving it from the grid and the bounds.
            "voxel_size": voxel_size,
            "cell_size_physical": (
                structure_options.cell_size_voxels * voxel_size
                if voxel_size is not None
                else None
            ),
            "member_thickness_physical": (
                structure_options.member_thickness_voxels * voxel_size
                if voxel_size is not None
                else None
            ),
            "skin_thickness_physical": (
                structure_options.skin_thickness_voxels * voxel_size
                if voxel_size is not None
                else None
            ),
            "lattice_dimension_unit": "model units",
            "member_thickness_voxels": (structure_options.member_thickness_voxels),
            "skin_thickness_voxels": structure_options.skin_thickness_voxels,
            "surface_backend": (
                recovered.get("surface_backend")
                if isinstance(recovered, dict)
                else None
            ),
            "surface_quality": (
                recovered.get("surface_quality")
                if isinstance(recovered, dict)
                else None
            ),
            "surface_quality_preset": (
                recovered.get("surface_quality_preset")
                if isinstance(recovered, dict)
                else "Compact cubic spline"
            ),
            "material_fraction": manufactured_material_fraction,
            "volume": manufactured_volume,
            "mass": manufactured_volume * material_density,
            "analysis_grid_material_fraction": (
                analysis_grid_manufactured_material_fraction
            ),
            "analysis_grid_volume": analysis_grid_manufactured_volume,
            "recovery_grid_reference_used": (
                recovery_grid_manufactured_volume is not None
            ),
            "lattice_sizing": (
                recovered.get("lattice_sizing") if isinstance(recovered, dict) else None
            ),
            # How much of the optimized envelope the built lattice reached, and
            # how many ligaments it took to keep it in one piece. A lattice can
            # hit its relative-density target while occupying only a corner of
            # the envelope, so the mass number alone does not say whether the
            # manufactured geometry matches the Density view.
            "lattice_connectivity": (
                recovered.get("lattice_connectivity")
                if isinstance(recovered, dict)
                else None
            ),
            "requires_independent_reanalysis": (structure_options.mode != "solid"),
            "meaningful_component_count": component_count,
            "source_body_count": source_component_count,
            "component_voxels": component_voxels,
        },
        "lattice_generation": (
            {
                "method": "Direct geometric lattice infill",
                "cell_family": structure_options.display_name,
                "independent_validation_required": True,
            }
            if structure_options.family is not None
            else None
        ),
        "compliance": (
            float(result.compliance_history[-1]) if result.compliance_history else None
        ),
        "stress_pnorm": (
            float(result.stress_history[-1]) if result.stress_history else None
        ),
        "structural_cases": [
            {
                "name": name,
                "compliance": float(value),
                "maximum_displacement": result.case_max_displacements.get(name),
            }
            for name, value in result.case_compliances.items()
        ],
        "iterations": result.n_iter,
        "max_iterations": problem.max_iter,
        "converged": result.converged,
        "message": result.message,
        "compliance_history": result.compliance_history,
        "objective_history": result.objective_history,
        "objective_reduction_pct": objective_reduction_pct,
        "objective_reduction_scope": "final continuation stage",
        "objective_reduction_baseline_iteration": objective_baseline_index + 1,
        "objective_full_run_change_pct": objective_full_run_change_pct,
        "change_history": result.change_history,
        "stress_history": result.stress_history,
        "case_compliances": getattr(result, "case_compliances", {}),
        "case_max_displacements": getattr(result, "case_max_displacements", {}),
        "multibody": {
            "global_joints": [
                {
                    "name": joint.name,
                    "type": joint.kind,
                    "anchor_a": list(joint.anchor_a),
                    "anchor_b": list(joint.anchor_b),
                    "axis": joint.axis,
                    "relative_stiffness": joint.relative_stiffness,
                }
                for joint in bc.joints
            ],
            "pose_count": len(bc.load_cases) if bc.load_cases else 1,
            "assembly_hardware": [
                {
                    "type": "pin",
                    "axis": str(pin[0]),
                    "fractional_region": list(pin[1:]),
                    "included_in_topology_material_budget": False,
                }
                for pin in bc.joint_pin_cylinders
            ],
            "poses": [
                {
                    "name": case.name,
                    "weight": case.weight,
                    "replace_supports": case.replace_supports,
                    "support_region_count": len(case.fixed_boxes),
                    "replace_joints": case.replace_joints,
                    "joint_count": len(case.joints),
                }
                for case in bc.load_cases
            ],
        },
        "passive_regions": {
            "solid_boxes": list(bc.solid_boxes),
            "void_boxes": list(bc.void_boxes),
            "solid_cylinders": list(bc.solid_cylinders),
            "void_cylinders": list(bc.void_cylinders),
            "joint_pin_cylinders": list(bc.joint_pin_cylinders),
            "explicit_solid_voxels": int(np.count_nonzero(explicit_passive_solid_mask)),
            "explicit_void_voxels": int(np.count_nonzero(explicit_passive_void_mask)),
            "explicit_solid_shapes": explicit_solid_shapes,
            "explicit_void_shapes": explicit_void_shapes,
        },
    }

    if is_infill_study:
        if isinstance(recovered, dict):
            output["passive_regions"].update(recovered.get("preserved_lattice_geometry") or {})
        _apply_lattice_infill_reporting(
            node, output, recovered, requested_structure_options
        )

    return TopologyOutputContext(
        payload=output,
        manufactured_density=manufactured_density,
        structure_options=structure_options,
        result=result,
        study=study,
    )


def _apply_lattice_infill_reporting(
    node: Any,
    output: dict[str, Any],
    recovered: Any,
    structure_options: ManufacturingStructureOptions,
) -> None:
    """Report a lattice infill on the geometry it delivers.

    Two corrections, both about *which field was measured*.

    The payload above is the topology optimizer's report, and most of it is
    undefined for a node that runs no solve: there is no material budget, no
    compliance, no objective and no iteration count. ``study_kind`` lets the
    result panel show the lattice numbers instead of a page of zeros and
    inapplicable rows.

    The rest is a grid mismatch. ``final_vol_frac`` measures the *design field*,
    which for an infill is the constant 1.0 meaning "this is all part", so it
    reported 100% material for a 25% lattice. ``meaningful_component_count``
    measures the analysis-grid rebuild, which is sized from a nominal member
    rather than from the one the density search settled on. Both are replaced
    here by the corresponding measurement on the build that was delivered —
    the same field the recovered surface was contoured from.
    """
    sizing = recovered.get("lattice_sizing") if isinstance(recovered, dict) else None
    connectivity = (
        recovered.get("lattice_connectivity") if isinstance(recovered, dict) else None
    )
    surface = recovered.get("surface_quality") if isinstance(recovered, dict) else None
    output["study_kind"] = "lattice_infill"

    achieved = None
    if isinstance(sizing, dict) and sizing.get("achieved_relative_density"):
        achieved = float(sizing["achieved_relative_density"])
    elif isinstance(connectivity, dict) and connectivity.get("relative_density"):
        achieved = float(connectivity["relative_density"])
    if achieved is not None:
        output["final_vol_frac"] = achieved
    output["target_vol_frac"] = (
        float(structure_options.target_relative_density or 0.0) or None
    )

    # The delivered body count. The mesh has already had sub-resolution
    # contouring specks removed, so this is the number of bodies a slicer will
    # actually see, not a voxel-connectivity proxy for it.
    if isinstance(surface, dict) and surface.get("connected_components"):
        manufacturing = output.get("manufacturing")
        if isinstance(manufacturing, dict):
            manufacturing["meaningful_component_count"] = int(
                surface["connected_components"]
            )

    output["lattice_infill"] = {
        "pattern": structure_options.display_name,
        "requested_relative_density": float(
            structure_options.target_relative_density or 0.0
        ),
        "achieved_relative_density": achieved,
        "cell_size_mode": str(
            node.get_property("infill_cell_size_mode") or "Automatic"
        ),
        "cell_fineness": str(node.get_property("infill_fineness") or ""),
        "build_quality": str(node.get_property("infill_build_quality") or ""),
        "skin_thickness": float(
            structure_options.skin_thickness_voxels
            * (output.get("manufacturing", {}).get("voxel_size") or 0.0)
        ),
        "skin_mode": str(node.get_property("infill_skin_mode") or "None"),
        "cell_size_voxels": float(structure_options.cell_size_voxels),
        "member_thickness_model_units": (
            float(sizing.get("member_thickness_model_units"))
            if isinstance(sizing, dict)
            and sizing.get("member_thickness_model_units") is not None
            else None
        ),
        "envelope_reach": (
            float(connectivity.get("envelope_reach"))
            if isinstance(connectivity, dict)
            and connectivity.get("envelope_reach") is not None
            else None
        ),
        "build_grid": [
            int(node.get_property("nelx") or 0),
            int(node.get_property("nely") or 0),
            int(node.get_property("nelz") or 0),
        ],
        # Set when the build-quality budget could not carry the requested
        # pitch and it was grown to keep the delivered density honest. The
        # user asked for a cell size and did not get it, so this travels with
        # the result rather than living only in the log.
        "pitch_coarsened": bool(getattr(node, "_infill_pitch_coarsened", False)),
        "requested_cell_size": (
            float(getattr(node, "_requested_infill_cell_size", 0.0)) or None
        ),
        "cell_size": float(getattr(node, "_resolved_infill_cell_size", 0.0) or 0.0)
        or None,
    }


def finalize_topology_output(
    node: Any,
    context: TopologyOutputContext,
) -> dict[str, Any]:
    """Attach geometry/optimization checks; CAD and FEA are explicit actions."""
    output = context.payload
    structure_options = context.structure_options
    result = context.result
    prepared = context.study
    problem = prepared.problem
    material = prepared.material
    is_infill_study = callable(getattr(node, "resolve_infill_cell_size", None))
    warnings_out: list[str] = []
    failed_checks: list[str] = []
    advisory_checks: list[str] = []

    def block(check: str, message: str) -> None:
        failed_checks.append(check)
        warnings_out.append(message)

    def advise(check: str, message: str) -> None:
        advisory_checks.append(check)
        warnings_out.append(message)

    manufacturing = output.get("manufacturing") or {}
    boundary_interfaces = manufacturing.get("boundary_interfaces") or {}
    singular_interface_count = sum(
        int(count)
        for key in ("support_selection_types", "load_selection_types")
        for kind, count in (boundary_interfaces.get(key, {}) or {}).items()
        if str(kind).title() in {"Edge", "Vertex"}
    )
    if singular_interface_count:
        advise(
            "zero_area_boundary_interface",
            f"{singular_interface_count} topology load/support interface(s) "
            "use an edge or vertex. These line/point conditions have no added "
            "material pad; local stresses are sensitive to mesh resolution. "
            "Select a face if the physical interface acts over an area.",
        )
    manufactured_components = int(manufacturing.get("meaningful_component_count") or 0)
    source_bodies = int(manufacturing.get("source_body_count") or 0)
    expected_components = max(1, source_bodies)
    if output.get("recovery_field_fallback"):
        advise(
            "projected_density_recovery",
            "The smooth filtered recovery field lost an analyzed load path, "
            "so geometry recovery used the connected projected density. "
            "Review surface finish and refine the grid before release.",
        )
    if manufactured_components == 0:
        block(
            "no_load_bearing_component",
            "The recovered result contains no meaningful load-bearing material "
            "component. Re-check the density cutoff, material budget, preserved "
            "interfaces, and load path.",
        )
    elif manufactured_components > expected_components:
        block(
            "disconnected_load_path",
            f"The manufactured {structure_options.display_name.lower()} has "
            f"{manufactured_components} disconnected load-bearing components "
            f"inside {expected_components} source body/bodies. This result is "
            "not verification-ready; enlarge the member/skin, refine the topology "
            "grid, or revise the preserved interfaces.",
        )
    # A lattice that reached its mass target while filling only part of its
    # envelope is the failure a relative density cannot express: the Density
    # view shows the whole load path and the manufactured geometry does not
    # occupy it. Name the gap and the pitch that caused it.
    connectivity = manufacturing.get("lattice_connectivity")
    if isinstance(connectivity, dict) and connectivity:
        reach = float(connectivity.get("envelope_reach") or 0.0)
        ligaments = int(connectivity.get("connecting_ligaments") or 0)
        pitch = float(connectivity.get("resolved_cell_voxels") or 0.0)
        if 0.0 < reach < 0.90:
            block(
                "lattice_does_not_fill_envelope",
                f"The manufactured {structure_options.display_name.lower()} "
                f"reaches only {reach:.0%} of the optimized envelope at a "
                f"resolved cell pitch of {pitch:.1f} voxels. The Density view "
                "shows material this lattice does not fill. Reduce the cell "
                "pitch, refine the topology grid or the surface quality, and "
                "re-check the manufactured geometry against the design field.",
            )
        if ligaments:
            advise(
                "lattice_connected_by_ligaments",
                f"Trimming the cell family to the envelope left the lattice in "
                f"pieces; {ligaments} connecting ligament(s) one member thick "
                "were added inside the envelope to restore a single load path. "
                "They are real geometry and carry load, but they are not part "
                "of the periodic cell: confirm them in re-analysis, or reduce "
                "the cell pitch so the network closes on its own.",
            )
    intermediate_fraction = float(output.get("intermediate_density_fraction") or 0.0)
    if structure_options.mode == "solid" and intermediate_fraction > 0.10:
        advise(
            "intermediate_density",
            f"{intermediate_fraction:.1%} of the design domain remains at an "
            "intermediate physical density (0.1 < rho < 0.9). The recovered "
            "boundary is usable for concept review, but the optimization "
            "should be sharpened or continued before geometry release.",
        )
    recovery_delta = output.get("recovery_volume_delta_pct")
    if recovery_delta is not None:
        absolute_delta = abs(float(recovery_delta))
        if absolute_delta > 10.0:
            reference = str(
                output.get("recovery_volume_reference") or "density-equivalent design"
            )
            block(
                "recovery_volume_drift",
                "Recovered-geometry volume differs from the "
                f"{reference} volume by {float(recovery_delta):+.1f}%. "
                "Refine the topology/recovery grid or revise the cutoff.",
            )
        elif absolute_delta > 5.0:
            advise(
                "recovery_volume_drift",
                "Recovered-geometry volume differs from the optimized "
                f"density-equivalent volume by {float(recovery_delta):+.1f}%.",
            )
    surface_quality = manufacturing.get("surface_quality") or {}
    if isinstance(surface_quality, dict):
        if (
            surface_quality.get("is_watertight", surface_quality.get("watertight"))
            is False
        ):
            # Watertightness fails on either defect, so name whichever is
            # actually present. A mesh can have no open boundary at all and
            # still be unusable because three faces share an edge.
            open_edges = int(surface_quality.get("open_boundary_edges") or 0)
            nonmanifold = int(surface_quality.get("nonmanifold_edges") or 0)
            defects = []
            if open_edges:
                defects.append(f"{open_edges} open boundary edges")
            if nonmanifold:
                defects.append(f"{nonmanifold} non-manifold edges")
            block(
                "recovered_surface_not_watertight",
                "The recovered topology surface is not watertight"
                + (f" ({', '.join(defects)})" if defects else "")
                + ". Repair or refine it before meshing/export.",
            )
        recovered_components = int(surface_quality.get("connected_components") or 0)
        hardware_components = len(
            list((output.get("multibody") or {}).get("assembly_hardware") or [])
        )
        expected_surface_components = expected_components + hardware_components
        if recovered_components > expected_surface_components:
            block(
                "recovered_surface_disconnected",
                f"The recovered surface contains {recovered_components} "
                "components; no more than "
                f"{expected_surface_components} are expected from the design "
                "bodies and separate joint hardware.",
            )
        # A closed, two-manifold surface can still pass through itself, and the
        # fairing and feature-snapping stages are what put it there. Every
        # slicer and tetrahedral mesher downstream refuses such a body, so this
        # blocks rather than advises.
        self_intersections = surface_quality.get("self_intersecting_faces")
        if self_intersections:
            block(
                "recovered_surface_self_intersecting",
                f"{int(self_intersections)} faces of the recovered surface "
                "pass through another part of the same surface. Refine the "
                "topology/recovery grid or lower the surface-quality "
                "smoothing before meshing or printing this body.",
            )
        elif self_intersections is None:
            # Two different reasons produce `None`, and telling the user the
            # wrong one sends them to install a package they already have. The
            # screening report names which it was.
            scope = str(surface_quality.get("screening_scope") or "").strip()
            reason = str(
                surface_quality.get("screening_reason")
                or (
                    f"screening was incomplete ({scope})"
                    if scope and scope != "full"
                    else "the recovery report did not provide a screening result"
                )
            )
            advise(
                "self_intersection_not_screened",
                "The recovered surface was not screened for self-intersection: "
                f"{reason}. A watertight mesh can still pass through itself; "
                "check it in the slicer or mesher before release.",
            )
    global_joints = list((output.get("multibody") or {}).get("global_joints") or [])
    jointed_multibody = bool(source_bodies > 1 and global_joints)
    if jointed_multibody:
        block(
            "multibody_joint_reanalysis_required",
            "Idealized joints need an external multibody validation model.",
        )
    if not result.converged:
        block(
            "optimizer_not_converged",
            f"Stopped after {result.n_iter} iterations without convergence.",
        )
    if not is_infill_study:
        final_fraction = float(output.get("final_vol_frac") or 0.0)
        volume_tolerance = max(1e-4, 0.01 * float(problem.volfrac))
        if final_fraction > float(problem.volfrac) + volume_tolerance:
            block(
                "topology_volume_constraint",
                f"Target volume is not satisfied ({final_fraction:.1%} material "
                f"> {float(problem.volfrac):.1%} budget). The result is not "
                "feasible under the requested material and stress constraints.",
            )
    if (
        result.stress_history
        and float(result.stress_history[-1]) > float(problem.yield_stress) * 1.01
    ):
        block(
            "topology_stress_constraint",
            f"The stress P-norm constraint is not satisfied "
            f"({float(result.stress_history[-1]):.3g} MPa > "
            f"{float(problem.yield_stress):.3g} MPa allowable). The "
            "minimum-mass result is not feasible.",
        )
    recovered_payload = output.get("recovered_shape")
    resolution_warning = cell_resolution_warning(
        structure_options,
        float(
            (recovered_payload or {}).get("structure_resolution_scale") or 1.0
            if isinstance(recovered_payload, dict)
            else 1.0
        ),
    )
    if resolution_warning:
        # Under-resolving a periodic surface does not thin it, it breaks it
        # into pieces, and the connectivity cull then removes most of them. The
        # component count catches the result; this names the cause.
        advise("lattice_cell_under_resolved", resolution_warning)
    # The grid can resolve a cell that the part still cannot hold. A rib
    # thinner than the cell fragments the network however finely it is sampled,
    # and the repair then rebuilds the load path out of off-pattern ligaments,
    # so this is asked separately from the resolution check above.
    fit_warning = cell_fit_warning(
        structure_options,
        (recovered_payload or {}).get("lattice_connectivity")
        if isinstance(recovered_payload, dict)
        else None,
    )
    if fit_warning:
        advise("lattice_cell_does_not_fit_envelope", fit_warning)
    infill_summary = output.get("lattice_infill")
    if isinstance(infill_summary, dict) and infill_summary.get("pitch_coarsened"):
        # The user asked for a cell size and did not get it. That travels with
        # the result, not only in the log: the pitch is half the specification
        # of an infill and a silent substitution is indistinguishable from the
        # setting not working, which is how this node read before.
        advise(
            "infill_pitch_coarsened_to_budget",
            f"The requested {float(infill_summary.get('requested_cell_size') or 0.0):.3g} "
            "cell needs a finer grid than the "
            f"{infill_summary.get('build_quality') or 'current'} build quality "
            f"allows, so a {float(infill_summary.get('cell_size') or 0.0):.3g} "
            "cell was built instead. The relative density is still the one "
            "requested — the cells are larger and there are fewer of them. "
            "Raise Build Quality for the pitch you asked for.",
        )
    stranded = _loads_stranded_in_void(result, problem)
    if stranded:
        block(
            "load_in_void",
            f"{stranded} load point(s) ended up in near-void material. A "
            "material removal at a load interface invalidates its load path. "
            "Re-check the selected geometry and preserved interfaces.",
        )
    # FEA is an explicit action on the reconstructed CAD body. A legacy flag
    # must not silently analyse a voxel surrogate after the topology solve.
    if float(result.min_source_volfrac or 0.0) > float(problem.volfrac) + 1e-6:
        block(
            "passive_material_exceeds_budget",
            "Target volume is below the fixed passive material; final "
            "volume cannot reach the requested fraction unless passive "
            "solid regions are reduced.",
        )
    if output.get("recovered_shape") is None and output.get(
        "surface_recovery_requested"
    ):
        block(
            "surface_recovery_missing",
            "No recovered surface was produced. Refine the voxel grid or "
            "revise the density cutoff before downstream meshing.",
        )

    # Recovery supplies the manufacturing mesh immediately. CAD is an
    # explicit preview/export operation. Legacy automatic-CAD properties
    # must not start a reconstruction after every solve.
    output.pop("cad_shape", None)
    output.pop("shape", None)
    if output.get("recovered_shape") is not None:
        output["cad_reconstruction"] = {
            "method": "On demand",
            "skipped": True,
            "editable": None,
        }
    from ..geometry.lattice_cad import topopt_export_formats

    output["export_formats"] = list(topopt_export_formats(output))

    output["quality_gate"] = {
        "base_failed_checks": failed_checks,
        "base_advisory_checks": advisory_checks,
        "manufactured_component_count": manufactured_components,
        "expected_component_count": expected_components,
    }
    output["warnings"] = list(output.get("warnings") or []) + warnings_out
    from ..verification.quality_gate import update_verification_status

    return update_verification_status(
        output,
        material=material,
        safety_factor=node.get_property("validation_yield_safety_factor"),
        displacement_limit=node.get_property("validation_displacement_limit_mm"),
    )
