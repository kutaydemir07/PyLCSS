# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Recover geometry and assemble the final topology node payload."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from pylcss.input_values import as_bool

from ..configuration.length_scale import (
    THIN_MATERIAL_REPORTING_FRACTION,
    thin_material_fraction,
)
from ..geometry.surface_recovery import _recover_voxel_shape
from ..manufacturing.structures import (
    ManufacturingStructureOptions,
    build_manufacturing_field,
    cell_fit_warning,
    cell_resolution_warning,
    passive_region_masks,
)
from ..manufacturing.member_sizing import (
    OptimizedMemberPlan,
)
from ..optimization.results import TopologyOptVoxelResult
from ..optimization.voxel_solver import _projection_matched_level
from ..geometry.lattice_cad import lattice_cad_strategy
from .boundary_mapping import _bounds_payload
from .execution_setup import PreparedTopologyStudy
from .lattice_settings import (
    lattice_dimension_range_text,
    lattice_setup_is_guided,
    resolve_lattice_structure_options,
)
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
CancelCallback = Callable[[], bool]


@dataclass(frozen=True)
class TopologyOutputContext:
    """Recovered output plus state required for optional validation and CAD."""

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
                result.projection_eta
                if result.projection_eta is not None
                else 0.5
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
        return 100.0 * (
            1.0 - finish / max(abs(start), 1.0e-30)
        )

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
    mc = study.manufacturing
    problem = study.problem

    def _selection_type_counts(entries: list[Any]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            kind = str(
                entry.get("selection_entity_type") or "Unknown"
            ).title()
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
    print_ready = True
    guided_lattice_density = False
    visualization_mode = "Density" if str(node.get_property("visualization") or "CAD") == "Density" else "CAD"
    decimate = 1.0
    # Lattice dimensions given in model units win over the voxel-denominated
    # ones, because a voxel is a different physical length on every part.
    voxel_size = voxel_size_from_bounds(
        tuple(int(v) for v in density.shape), bounds
    )
    try:
        structure_options = resolve_lattice_structure_options(
            node,
            tuple(int(v) for v in density.shape),
            bounds,
        )
    except (TypeError, ValueError) as exc:
        hint = ""
        if not lattice_setup_is_guided(
            node.get_property("lattice_settings_mode")
        ):
            hint = " " + lattice_dimension_range_text(
                node.get_property("structure_mode"),
                voxel_size,
                cell_size=node.get_property("lattice_cell_size_mm"),
            )
        node.set_error(f"Invalid lattice geometry: {exc}{hint}")
        return None
    # The CAD stage needs to know which structure was built: a strut lattice is
    # reconstructed from its centrelines rather than from the isosurface.
    setattr(node, "_last_structure_options", structure_options)
    if structure_options.mode != "solid":
        print_ready = True
    # Guided presents the recovered surface while the final solid CAD body is
    # being reconstructed. Lattice studies keep this manufactured-mesh view.
    surface_requested = True
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
    member_plan: OptimizedMemberPlan | None = None
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
    # Recover the surface from the filtered density, not the projected one.
    #
    # `density` is what the FE solve used and what every reported quantity is
    # measured on, but the completed Heaviside continuation leaves it nearly
    # binary. An isosurface through a binary field can only pass between voxel
    # centres, so marching cubes reproduces the grid terraces — the stair-step
    # ridges and voxel-scale ripples seen on recovered members. The filtered
    # field varies smoothly over ~2*rmin voxels and locates the same interface
    # to sub-voxel accuracy. `recovery_cutoff` is the level on it that encloses
    # the analyzed material volume, so this changes surface quality, not mass.
    recovery_field, recovery_cutoff = _select_recovery_field(
        result,
        density,
        density_cutoff,
    )

    def recover_surface(
        field: np.ndarray,
        cutoff: float,
    ) -> dict[str, Any] | None:
        return _recover_voxel_shape(
            field,
            bounds,
            cutoff,
            print_ready=print_ready,
            decimate_ratio=decimate,
            solid_boxes=bc.solid_boxes,
            void_boxes=bc.void_boxes,
            solid_cylinders=bc.solid_cylinders,
            void_cylinders=bc.void_cylinders,
            joint_pin_cylinders=bc.joint_pin_cylinders,
            extrusion_axis=str(getattr(problem.mc, "extrusion", "none") or "none"),
            source_mask=design_domain,
            passive_solid_mask=passive_solid_mask,
            passive_void_mask=passive_void_mask,
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
        recovery_cutoff,
        structure_options,
        "none",
        str(bc.solid_boxes),
        str(bc.void_boxes),
        str(bc.solid_cylinders),
        str(bc.void_cylinders),
        str(bc.joint_pin_cylinders),
        hash(np.ascontiguousarray(passive_solid_mask).tobytes()),
        hash(np.ascontiguousarray(passive_void_mask).tobytes()),
        None,
    )
    if not surface_requested:
        recovered = None
    elif getattr(node, "_last_recovery_key", None) == cache_key:
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
            (recovered or {}).get("surface_quality", {}).get(
                "connected_components", 0
            )
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
            result.recovery_density is not None
            and recovery_field is not density
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
                (projected_recovery or {}).get("surface_quality", {}).get(
                    "connected_components", 0
                )
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
            filtered_component_error = abs(
                recovered_components - component_count
            )
            projected_component_error = abs(
                projected_components - component_count
            )
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
        if isinstance(recovered, dict) else None
    )
    if isinstance(resolved_options, dict):
        resolved_options = ManufacturingStructureOptions(**resolved_options)
    if isinstance(resolved_options, ManufacturingStructureOptions) and resolved_options.mode != "solid":
        # Recovery sizes actual walls/struts to the requested mass. Export and
        # analysis-grid previews must use those same dimensions, rather than
        # the unsized initial one-voxel member thickness.
        structure_options = resolved_options
        setattr(node, "_last_structure_options", structure_options)
        manufactured_density = build_manufacturing_field(
            density, density_cutoff, structure_options,
            passive_solid_mask=passive_solid_mask,
            passive_void_mask=passive_void_mask, member_plan=member_plan,
        )
    if (
        isinstance(recovered, dict)
        and recovered.get("recovery_field_fallback")
    ):
        recovery_field = density
        recovery_cutoff = density_cutoff
    if isinstance(recovered, dict):
        try:
            effective_cutoff = float(
                recovered.get("effective_recovery_cutoff")
            )
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
            candidate = float(
                recovered.get("manufacturing_reference_volume")
            )
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
        manufactured_volume / source_volume
        if source_volume > 1e-12
        else 0.0
    )
    (
        objective_reduction_pct,
        objective_full_run_change_pct,
        objective_baseline_index,
    ) = _objective_reduction_metrics(result)
    recovery_volume_delta_pct = (
        100.0 * (recovered_design_volume / recovery_reference_volume - 1.0)
        if (
            recovered_design_volume is not None
            and recovery_reference_volume > 1e-12
        )
        else None
    )
    voxel_edge_lengths = np.asarray(
        (problem.unitx, problem.unity, problem.unitz),
        dtype=float,
    )
    extrusion_axis = {
        "x": 0,
        "y": 1,
        "z": 2,
    }.get(str(problem.mc.extrusion or "none").lower())
    length_scale_active_axes = [
        axis for axis in range(3) if axis != extrusion_axis
    ]
    if not length_scale_active_axes:
        length_scale_active_axes = [0, 1, 2]
    limiting_voxel_edge = float(
        np.max(voxel_edge_lengths[length_scale_active_axes])
    )

    def _feature_elements(value: float) -> float | None:
        physical = float(value or 0.0)
        if physical <= 0.0 or limiting_voxel_edge <= 0.0:
            return None
        return physical / limiting_voxel_edge

    explicit_solid_shapes: list[Any] = []
    explicit_void_shapes: list[Any] = []
    for region in getattr(study, "non_design_regions", ()):
        geom = region.get("geometry") if isinstance(region, dict) else region
        rtype = str(region.get("region_type") or "solid").lower() if isinstance(region, dict) else "solid"
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
    output: dict[str, Any] = {
        "type": "topopt_voxel",
        "source_cad_shape": source_cad_shape,
        "design_goal": design_goal,
        "objective_mode": problem.objective_mode,
        "physics_mode": problem.physics_mode,
        "physics_model": "Linear-elastic, small-displacement structural response.",
        "optimizer_used": result.optimizer_used,
        "formulation_used": result.formulation_used,
        "physical_length_scale": {
            "requested_minimum_solid_size": 0.0,
            "requested_minimum_void_size": 0.0,
            "filter_radius": float(problem.rmin),
            "filter_radius_units": (
                "model units"
                if problem.filter_radius_is_physical
                else "voxel units"
            ),
            "minimum_solid_size": float(problem.minimum_solid_size),
            "minimum_void_size": float(problem.minimum_void_size),
            "minimum_solid_size_elements": _feature_elements(
                problem.minimum_solid_size
            ),
            "minimum_void_size_elements": _feature_elements(
                problem.minimum_void_size
            ),
            "voxel_edge_lengths": [
                float(value) for value in voxel_edge_lengths
            ],
            "active_topology_axes": [
                "XYZ"[axis] for axis in length_scale_active_axes
            ],
            "planar_extrusion_reduction": (
                {
                    "axis": "XYZ"[
                        int(getattr(node, "_guided_planar_extrusion_axis"))
                    ],
                    "retained_3d_layers": int(
                        density.shape[
                            int(getattr(node, "_guided_planar_extrusion_axis"))
                        ]
                    ),
                    "reason": (
                        "Exact extrusion with through-thickness-invariant "
                        "supports and in-plane loads"
                    ),
                }
                if getattr(node, "_guided_planar_extrusion_axis", None)
                is not None
                else None
            ),
            "maximum_member_size": float(
                problem.mc.max_member_size_physical
            ),
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
        "surface_recovery_requested": surface_requested,
        "extrusion_axis": mc.extrusion,
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
            "process": str(
                "Unrestricted"
            ),
            "constraints": {
                "symmetry_planes": str(problem.mc.symmetry or "none"),
                "extrusion_axis": str(problem.mc.extrusion or "none"),
                "additive_build_direction": str(
                    problem.mc.overhang_build_axis or "none"
                ),
                "additive_overhang_angle_deg": float(
                    problem.mc.overhang_angle_deg
                ),
                "pull_out_direction": str(
                    problem.mc.pull_out_axis or "none"
                ),
                "maximum_member_size": float(
                    problem.mc.max_member_size_physical
                ),
                "pattern_repeat": int(problem.mc.pattern_repeat),
                "pattern_axis": str(problem.mc.pattern_axis),
            },
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
                ("Guided" if guided_lattice_density else "Manual")
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
                recovered.get("lattice_sizing")
                if isinstance(recovered, dict)
                else None
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
            {"method": "Direct geometric lattice infill", "cell_family": structure_options.display_name,
             "independent_validation_required": True}
            if structure_options.family is not None else None
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
        _apply_lattice_infill_reporting(node, output, recovered, requested_structure_options)

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
    sizing = (
        recovered.get("lattice_sizing") if isinstance(recovered, dict) else None
    )
    connectivity = (
        recovered.get("lattice_connectivity")
        if isinstance(recovered, dict)
        else None
    )
    surface = (
        recovered.get("surface_quality") if isinstance(recovered, dict) else None
    )
    output["study_kind"] = "lattice_infill"

    achieved = None
    if isinstance(sizing, dict) and sizing.get("achieved_relative_density"):
        achieved = float(sizing["achieved_relative_density"])
    elif isinstance(connectivity, dict) and connectivity.get("relative_density"):
        achieved = float(connectivity["relative_density"])
    if achieved is not None:
        output["final_vol_frac"] = achieved
    output["target_vol_frac"] = float(
        structure_options.target_relative_density or 0.0
    ) or None

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
            node.get_property("infill_skin_thickness_mm") or 0.0
        ),
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
        "pitch_coarsened": bool(
            getattr(node, "_infill_pitch_coarsened", False)
        ),
        "requested_cell_size": (
            float(getattr(node, "_requested_infill_cell_size", 0.0)) or None
        ),
        "cell_size": float(
            getattr(node, "_resolved_infill_cell_size", 0.0) or 0.0
        )
        or None,
    }


def finalize_topology_output(
    node: Any,
    context: TopologyOutputContext,
    cancel_callback: CancelCallback | None,
    progress_callback: Any = None,
) -> dict[str, Any]:
    """Run optional validation/CAD conversion and attach release warnings."""
    output = context.payload
    manufactured_density = context.manufactured_density
    structure_options = context.structure_options
    result = context.result
    prepared = context.study
    problem = prepared.problem
    material = prepared.material
    constraint_list = prepared.constraints
    load_list = prepared.loads
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
        for kind, count in (
            boundary_interfaces.get(key, {}) or {}
        ).items()
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
    manufactured_components = int(
        manufacturing.get("meaningful_component_count") or 0
    )
    source_bodies = int(manufacturing.get("source_body_count") or 0)
    expected_components = max(1, source_bodies)
    if output.get("recovery_field_fallback"):
        advise(
            "projected_density_recovery",
            "The smooth filtered recovery field lost an analyzed load path, "
            "so geometry recovery used the connected projected density. "
            "Review surface finish and refine the grid before release.",
        )
    if structure_options.mode != "solid" and not is_infill_study:
        block(
            "explicit_structure_external_reanalysis_required",
            "The built-in voxel check does not resolve local lattice/rib "
            "member bending, joint stress, fatigue, or process defects. "
            "Validate the explicit manufactured geometry with an appropriate "
            "beam/solid model before engineering release.",
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
            "grid, or revise the preserved interfaces."
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
    intermediate_fraction = float(
        output.get("intermediate_density_fraction") or 0.0
    )
    if (
        structure_options.mode == "solid"
        and intermediate_fraction > 0.10
    ):
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
                output.get("recovery_volume_reference")
                or "density-equivalent design"
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
        if surface_quality.get("is_watertight", surface_quality.get("watertight")) is False:
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
        recovered_components = int(
            surface_quality.get("connected_components") or 0
        )
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
            reason = str(surface_quality.get("screening_reason") or (
                f"screening was incomplete ({scope})"
                if scope and scope != "full"
                else "the recovery report did not provide a screening result"
            ))
            advise(
                "self_intersection_not_screened",
                "The recovered surface was not screened for self-intersection: "
                f"{reason}. A watertight mesh can still pass through itself; "
                "check it in the slicer or mesher before release.",
            )
    # The requested minimum member size and the delivered geometry were both
    # being computed and neither was ever compared to the other, so a study
    # could be handed a 4 mm minimum member size, produce a 2 mm web, and pass
    # every gate. A density filter is a regularization, not a length-scale
    # constraint, so the only way to know is to measure what came out.
    #
    # Reported rather than blocked, deliberately. The probe resolves a member
    # size to about one voxel and always picks up a small residue from the
    # rounding of free edges, and with no formulation imposing the length scale
    # there is nothing for a hard gate to hand back to the user beyond this
    # number. Stating it with its uncertainty is the honest form of the check.
    requested_member = 0.0
    if structure_options.mode == "solid" and requested_member > 0.0:
        density_field = (
            manufactured_density
            if manufactured_density is not None
            else result.recovery_density
        )
        thin = None
        if density_field is not None:
            thin = thin_material_fraction(
                np.asarray(density_field, dtype=float)
                >= float(result.recovery_cutoff or 0.5),
                requested_member,
                (problem.unitx, problem.unity, problem.unitz),
            )
        output["requested_minimum_member_size"] = requested_member
        output["thin_material_fraction"] = (
            thin.get("fraction") if thin else None
        )
        output["thin_material_probe_size"] = (
            thin.get("probe_size") if thin else None
        )
        if thin is None:
            advise(
                "minimum_member_size_not_measurable",
                f"The {requested_member:.3g} mm minimum member size could not "
                "be checked: it is smaller than this voxel grid can probe. "
                "Refine the topology grid if the requirement has to be "
                "verified on the result.",
            )
        else:
            probe = float(thin["probe_size"])
            fraction = float(thin["fraction"])
            # A discrete ball only has odd voxel diameters, so an even request
            # always loses a voxel. That is worth saying when the lost voxel is
            # a large share of the request -- on a three-voxel feature it is a
            # quarter of it -- and is noise when the grid is fine enough that
            # one voxel does not matter.
            probe_note = (
                ""
                if probe >= 0.85 * requested_member
                else (
                    f" The grid could only probe this at {probe:.3g} mm, below "
                    f"the {requested_member:.3g} mm requested, so members "
                    "between the two sizes are not visible to the check; "
                    "refine the grid to close that gap."
                )
            )
            if fraction > THIN_MATERIAL_REPORTING_FRACTION:
                advise(
                    "minimum_member_size_not_held",
                    f"{fraction:.1%} of the material sits in members thinner "
                    f"than {probe:.3g} mm — a ball of that diameter cannot "
                    "reach it — against a requested minimum member size of "
                    f"{requested_member:.3g} mm. A density filter regularizes "
                    "the design but does not impose a minimum length scale, so "
                    "this measures the result rather than reporting a solver "
                    "failure. Enlarge the filter radius or the requested size, "
                    "refine the grid, or accept the thin sections against the "
                    "process capability." + probe_note,
                )
            elif probe_note:
                advise(
                    "minimum_member_size_probe_coarse",
                    f"The delivered geometry holds {probe:.3g} mm."
                    + probe_note,
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
            f"Stopped after {result.n_iter} iterations without convergence."
        )
    if not is_infill_study:
        final_fraction = float(output.get("final_vol_frac") or 0.0)
        volume_tolerance = max(1e-4, 0.01 * float(problem.volfrac))
        if final_fraction > float(problem.volfrac) + volume_tolerance:
            block(
                "topology_volume_constraint",
                f"Target volume is not satisfied ({final_fraction:.1%} material "
                f"> {float(problem.volfrac):.1%} budget). The result is not "
                "feasible under the requested material and stress constraints."
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
            "minimum-mass result is not feasible."
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
            "pull-out or overhang direction that disagrees with where the "
            "loads act will remove the material under them, and the reported "
            "compliance is then meaningless. Re-check the manufacturing "
            "direction against the load positions."
        )
    # FEA is an explicit action on the reconstructed CAD body. A legacy flag
    # must not silently analyse a voxel surrogate after the topology solve.
    validation_requested = False
    if float(result.min_source_volfrac or 0.0) > float(problem.volfrac) + 1e-6:
        block(
            "passive_material_exceeds_budget",
            "Target volume is below the fixed passive material; final "
            "volume cannot reach the requested fraction unless passive "
            "solid regions are reduced."
        )
    validation_completed = False
    if validation_requested and jointed_multibody:
        output["validation_summary"] = {
            "status": "not applicable",
            "reason": (
                "The built-in voxel validator does not transfer idealized "
                "topology joints to connector/contact definitions."
            ),
        }
    elif validation_requested:
        try:
            validation_payload = output
            if manufactured_density is not None:
                validation_payload = dict(output)
                validation_payload["density"] = manufactured_density
                validation_payload["density_cutoff"] = 0.5
            validation = node._run_embedded_validation(
                validation_payload,
                material,
                constraint_list,
                load_list,
                cancel_callback=cancel_callback,
            )
            if validation is not None:
                validation_completed = True
                output["validation"] = validation
                study = (
                    validation.get("convergence_study")
                    if isinstance(validation, dict)
                    else None
                )
                output["validation_summary"] = {
                    "max_stress": validation.get(
                        "peak_stress_nodal", validation.get("max_stress_gauss")
                    ),
                    "max_displacement": validation.get("peak_displacement"),
                    "compliance": validation.get("compliance"),
                    "converged": (
                        study.get("converged") if isinstance(study, dict) else None
                    ),
                }
                if output["validation_summary"].get("converged") is False:
                    block(
                        "mesh_convergence_not_demonstrated",
                        "The independent CalculiX mesh-convergence study did "
                        "not converge. Refine the verification model before "
                        "using the result for release.",
                    )
                validated_stress = output["validation_summary"].get("max_stress")
                allowable = float(
                    material.get(
                        "yield_strength",
                        material.get("yield", 0.0),
                    )
                    or 0.0
                )
                safety_factor = max(
                    1.0,
                    float(
                        node.get_property("validation_yield_safety_factor")
                        or 1.0
                    ),
                )
                if allowable > 0.0:
                    allowable /= safety_factor
                if (
                    validated_stress is not None
                    and allowable > 0.0
                    and float(validated_stress) > allowable
                ):
                    block(
                        "validated_stress_exceeds_yield",
                        f"Independent CalculiX validation exceeds the material "
                        f"allowable ({float(validated_stress):.3g} MPa > "
                        f"{allowable:.3g} MPa after safety factor "
                        f"{safety_factor:g}). The manufactured result fails "
                        "the allowable-stress check."
                    )
                displacement_limit = float(
                    node.get_property("validation_displacement_limit_mm") or 0.0
                )
                validated_displacement = output["validation_summary"].get(
                    "max_displacement"
                )
                if (
                    displacement_limit > 0.0
                    and validated_displacement is not None
                    and float(validated_displacement) > displacement_limit
                ):
                    block(
                        "validated_displacement_exceeds_limit",
                        "Independent CalculiX validation exceeds the specified "
                        f"displacement limit ({float(validated_displacement):.3g} "
                        f"mm > {displacement_limit:.3g} mm)."
                    )
            else:
                block(
                    "validation_missing",
                    "Validation was requested but did not produce a CalculiX result.",
                )
        except Exception as exc:
            msg = f"Validation skipped/failed: {exc}"
            logger.warning("TopologyOptVoxelNode: %s", msg)
            output["validation_error"] = str(exc)
            block("validation_failed", msg)
    elif is_infill_study:
        # Not a failed check. A lattice infill has no solve to validate and
        # never claimed one, so listing this as a blocking gate would mark
        # every correct result as failed and leave the user with nothing to
        # act on. It replaces the reanalysis block above, which says the same
        # thing in optimizer terms.
        advise(
            "infill_is_geometry_only",
            "This is generated geometry, not an analyzed result: nothing here "
            "resolves member bending, joint stress, fatigue or process "
            "defects. Run it through an analysis before using it for anything "
            "load bearing.",
        )
    else:
        block(
            "independent_validation_not_run",
            "Independent analysis of the manufactured geometry was not run. "
            "The topology result is suitable for concept screening, not "
            "engineering release.",
        )

    if (
        output.get("recovered_shape") is None
        and output.get("surface_recovery_requested")
    ):
        block(
            "surface_recovery_missing",
            "No recovered surface was produced. Refine the voxel grid or "
            "revise the density cutoff before downstream meshing.",
        )

    # Recovery supplies the manufacturing mesh immediately. CAD is an
    # explicit preview/export operation. Legacy automatic-CAD properties
    # must not start a reconstruction after every solve.
    mesh_only = lattice_cad_strategy(structure_options) == "isosurface"
    if mesh_only:
        output.pop("cad_shape", None)
        output.pop("shape", None)
    eager_cad = False
    if eager_cad:
        try:
            cad_payload = dict(output)
            cad_shape = node._run_embedded_cad_reconstruction(
                cad_payload, progress_callback=progress_callback, cancel_callback=cancel_callback
            )
            if cad_payload.get("cad_preview_mesh") is not None:
                output["cad_preview_mesh"] = cad_payload["cad_preview_mesh"]
            output["cad_shape"] = cad_shape
            output["shape"] = cad_shape
            output["visualization_mode"] = "CAD"
            reconstruction_report = dict(
                getattr(node, "_last_cad_reconstruction_report", {}) or {}
            )
            # Validation and mass properties were already evaluated in the
            # CAD child. Repeating OCC calls here can stall the Qt interpreter.
            output["cad_reconstruction"] = reconstruction_report | {
                "method": reconstruction_report.get("method", "Compact cubic spline"),
                "volume": reconstruction_report.get("cad_volume"),
            }
            performance = reconstruction_report.get("mechanical_validation") or {}
            if performance.get("passed") is False:
                block("cad_performance_changed", performance.get("reason") or
                      "Reconstructed CAD failed mechanical performance comparison.")
            if output["cad_reconstruction"].get("valid") is False:
                block(
                    "cad_reconstruction_invalid",
                    "The reconstructed CAD solid is invalid and must be repaired "
                    "before export or downstream meshing.",
                )
        except Exception as exc:
            from pylcss.cae.user_interface.cad.cad_background import CadJobCancelled
            if isinstance(exc, CadJobCancelled):
                output["cad_reconstruction"] = {"method": "Cancelled", "cancelled": True}
            else:
                msg = f"CAD reconstruction failed: {exc}"
                logger.warning("TopologyOptVoxelNode: %s", msg)
                output["cad_error"] = str(exc)
                block("cad_reconstruction_failed", msg)
            if output.get("recovered_shape") is not None:
                output["visualization_mode"] = (
                    "Manufactured Mesh"
                    if structure_options.mode != "solid"
                    else "Surface"
                )
    elif output.get("recovered_shape") is not None:
        output["cad_reconstruction"] = {
            "method": "STL surface mesh" if mesh_only else "On demand",
            "skipped": True, "editable": False if mesh_only else None,
        }
    from ..geometry.lattice_cad import topopt_export_formats
    output["export_formats"] = list(topopt_export_formats(structure_options))

    verification_passed = validation_completed and not failed_checks
    # Compatibility alias for existing graph consumers. This means the
    # checks configured in this study passed; it is not product certification
    # or an automatic release decision.
    output["release_ready"] = verification_passed
    output["verification_passed"] = verification_passed
    # An infill has no solve, so it can never reach "verification passed" and
    # calling every correct result "concept only" says nothing. Report whether
    # the geometry checks passed, which is the question this node can answer.
    infill_status = (
        "geometry built"
        if not failed_checks
        else "geometry checks failed"
    )
    output["quality_gate"] = {
        "status": (
            infill_status
            if is_infill_study
            else "verification passed"
            if verification_passed
            else "concept only"
        ),
        "release_ready": verification_passed,
        "verification_passed": verification_passed,
        "failed_checks": failed_checks,
        "advisory_checks": advisory_checks,
        "manufactured_component_count": manufactured_components,
        "expected_component_count": expected_components,
        "independent_validation_completed": validation_completed,
    }
    prior_warnings = list(output.get("warnings") or [])
    if prior_warnings or warnings_out:
        output["warnings"] = prior_warnings + warnings_out
    return output
