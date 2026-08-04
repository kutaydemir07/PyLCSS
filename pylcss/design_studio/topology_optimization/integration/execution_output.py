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
from ..manufacturing.printability import (
    SELF_SUPPORTING_ANGLE_DEG,
    lattice_feature_size_report,
    lattice_overhang_report,
    overhang_advice,
)
from ..manufacturing.member_sizing import (
    OptimizedMemberPlan,
    optimize_lattice_members,
)
from ..optimization.results import TopologyOptVoxelResult
from ..optimization.voxel_solver import _projection_matched_level
from ..geometry.lattice_cad import lattice_cad_strategy
from .boundary_mapping import _bounds_payload
from .execution_setup import PreparedTopologyStudy
from .lattice_settings import (
    lattice_dimension_range_text,
    lattice_setup_is_guided,
    node_minimum_feature_size,
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
    # The same lookup the solver made. Resolved here rather than carried on the
    # result so the report cannot claim a two-scale solve the solver did not
    # actually run; both calls hit one registry with one set of rules.
    cell_law = problem.homogenized_cell_law(allow_build=False)
    logger.info("TopologyOptVoxelNode: %s", result.message)
    density = np.asarray(result.density, dtype=float)
    density_cutoff = _effective_density_cutoff(
        node.get_property("density_cutoff"),
        density=density,
        target_volfrac=problem.volfrac,
    )
    print_ready = as_bool(node.get_property("print_ready_mesh"))
    guided_mode = (
        str(node.get_property("workflow_mode") or "Guided").strip().lower()
        == "guided"
    )
    # A guided lattice study is the professional variable-density method, not
    # a fixed infill pattern applied after a different analysis.  Older example
    # projects stored ``False`` before the guided workflow made that distinction
    # explicit; honoring that hidden legacy value would make the manufactured
    # lattice disagree with the homogenized density field used by the solver.
    guided_lattice_density = bool(
        lattice_setup_is_guided(
            node.get_property("lattice_settings_mode")
        )
        and problem.lattice_cell_type
    )
    visualization_mode = (
        "Density"
        if guided_mode
        else str(node.get_property("visualization") or "Density")
    )
    decimate = float(node.get_property("mesh_decimate_ratio") or 1.0)
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
        node.set_error(f"Invalid lattice manufacturing settings: {exc}{hint}")
        return None
    # The CAD stage needs to know which structure was built: a strut lattice is
    # reconstructed from its centrelines rather than from the isosurface.
    setattr(node, "_last_structure_options", structure_options)
    if structure_options.mode != "solid":
        print_ready = True
    # Guided presents the recovered surface while the final solid CAD body is
    # being reconstructed. Lattice studies keep this manufactured-mesh view.
    if guided_mode and structure_options.mode == "solid":
        visualization_mode = "Surface"
    surface_requested = True
    surface_backend = (
        "legacy"
        if str(node.get_property("surface_recovery_method") or "")
        .strip()
        .lower()
        .startswith("legacy")
        else "vtk_sdf"
    )
    surface_quality = str(
        node.get_property("surface_quality") or "Professional"
    )
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
    member_sizing_error: str | None = None
    setattr(node, "_last_member_plan", None)
    is_infill_study = callable(getattr(node, "resolve_infill_cell_size", None))
    guided_member_sizing = bool(
        not is_infill_study
        and guided_mode
        and structure_options.family is not None
        and structure_options.family.maxwell == "stretch"
    )
    if (
        lattice_cad_strategy(structure_options) == "beam"
        and (
            guided_member_sizing
            or (
                not guided_mode
                and as_bool(node.get_property("optimize_lattice_members"))
            )
        )
        and problem.physics_mode != "thermal"
    ):
        try:
            safety_factor = max(
                1.0,
                float(
                    node.get_property("validation_yield_safety_factor")
                    or 1.0
                ),
            )
            material_yield = float(
                material.get(
                    "yield_strength",
                    material.get(
                        "yield",
                        node.get_property("yield_stress") or 0.0,
                    ),
                )
                or 0.0
            )
            maximum_member = max(
                structure_options.member_thickness_voxels,
                float(
                    node.get_property(
                        "lattice_max_member_thickness_voxels"
                    )
                    or 3.0
                ),
            )
            span = tuple(
                float(value)
                for value in np.maximum(bounds[1][:3] - bounds[0][:3], 0.0)
            )
            member_plan = optimize_lattice_members(
                (
                    (density >= density_cutoff) | passive_solid_mask
                )
                & ~passive_void_mask,
                mode=structure_options.mode,
                cell_size_voxels=structure_options.cell_size_voxels,
                minimum_diameter_voxels=(
                    structure_options.member_thickness_voxels
                ),
                maximum_diameter_voxels=maximum_member,
                bc=bc,
                span=span,
                youngs_modulus=float(problem.E0),
                allowable_stress=material_yield / safety_factor,
                displacement_limit=float(
                    node.get_property("validation_displacement_limit_mm")
                    or 0.0
                ),
                buckling_length_factor=float(
                    node.get_property("lattice_buckling_length_factor")
                    or 1.0
                ),
                maximum_iterations=int(
                    node.get_property("lattice_member_sizing_iterations")
                    or 35
                ),
            )
            setattr(node, "_last_member_plan", member_plan)
        except (TypeError, ValueError, ArithmeticError) as exc:
            member_sizing_error = str(exc)
            setattr(node, "_last_member_plan", None)
            logger.warning("Lattice member sizing was not completed: %s", exc)
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
            extrusion_axis=mc.extrusion,
            source_mask=design_domain,
            passive_solid_mask=passive_solid_mask,
            passive_void_mask=passive_void_mask,
            structure_options=structure_options,
            member_plan=member_plan,
            surface_backend=surface_backend,
            surface_quality=surface_quality,
        )

    # Key on the actual density bytes — not id(result.density), which Python's
    # allocator can reuse across runs and silently return a stale recovered
    # shape for a different solve.
    density_view = np.ascontiguousarray(recovery_field)
    cache_key = (
        hash(density_view.tobytes()),
        density_view.shape,
        recovery_cutoff,
        print_ready,
        decimate,
        surface_backend,
        surface_quality,
        structure_options,
        str(mc.extrusion),
        str(bc.solid_boxes),
        str(bc.void_boxes),
        str(bc.solid_cylinders),
        str(bc.void_cylinders),
        str(bc.joint_pin_cylinders),
        hash(np.ascontiguousarray(passive_solid_mask).tobytes()),
        hash(np.ascontiguousarray(passive_void_mask).tobytes()),
        member_plan.signature() if member_plan is not None else member_sizing_error,
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

    output: dict[str, Any] = {
        "type": "topopt_voxel",
        "design_goal": design_goal,
        "objective_mode": problem.objective_mode,
        "physics_mode": problem.physics_mode,
        "physics_model": (
            "Weighted structural compliance plus steady-state thermal "
            "compliance; thermal expansion and thermal stress are not modeled."
            if problem.physics_mode == "thermo_mechanical"
            else "Steady-state conduction with reduced-order volumetric heat rejection."
            if problem.physics_mode == "thermal"
            else "Linear-elastic, small-displacement structural response."
        ),
        "optimizer_used": result.optimizer_used,
        "formulation_used": result.formulation_used,
        "physical_length_scale": {
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
                node.get_property("manufacturing_process") or "None"
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
                "automatic_contact_thickness": (
                    study.automatic_exclusion_thickness
                ),
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
                else surface_quality
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
        "lattice_optimization": (
            {
                "method": (
                    (
                        "homogenized anisotropic cell law driving the macro "
                        "sensitivity loop, followed by "
                        if cell_law is not None
                        else "isotropic continuum surrogate followed by "
                    )
                    + (
                        "field-graded explicit lattice reconstruction"
                        if structure_options.variable_density
                        else "fixed-cell explicit lattice reconstruction"
                    )
                    + " and optional "
                    "load-aware member sizing"
                ),
                "phase": (
                    "phase 2 / sized explicit members"
                    if member_plan is not None
                    else (
                        "phase 1 / continuum-density interpretation"
                        if structure_options.variable_density
                        else "phase 1 / topology envelope with fixed cells"
                    )
                ),
                "cell_family": structure_options.display_name,
                "stiffness_model": study.lattice_porosity,
                "continuum_assumption": study.lattice_porosity,
                "stiffness_interpolation": (
                    (
                        "tabulated C11(rho), C12(rho), C44(rho) with "
                        "differentiable monotone interpolation"
                    )
                    if cell_law is not None
                    else f"E/E0 = relative_density^{float(problem.penal):g}"
                ),
                "continuum_surrogate_penalty": float(problem.penal),
                "variable_density": bool(structure_options.variable_density),
                "density_grading_control": (
                    "Program Controlled (Guided)"
                    if guided_lattice_density
                    else "User Controlled (Expert)"
                ),
                "minimum_relative_density": (
                    structure_options.minimum_relative_density
                ),
                "maximum_relative_density": (
                    structure_options.maximum_relative_density
                ),
                "solid_transition_density": (
                    structure_options.solid_transition_density
                ),
                "minimum_member_thickness_voxels": (
                    structure_options.member_thickness_voxels
                ),
                "maximum_member_thickness_voxels": float(
                    node.get_property(
                        "lattice_max_member_thickness_voxels"
                    )
                    or 3.0
                ),
                "member_size_optimization": member_plan is not None,
                "member_sizing": (
                    member_plan.diagnostics()
                    if member_plan is not None
                    else {
                        "completed": False,
                        "reason": (
                            member_sizing_error
                            or "This cell family uses continuum-density sizing."
                        ),
                    }
                ),
                "cell_material_law": (
                    cell_law.diagnostics() if cell_law is not None else None
                ),
                "macro_material_model": (
                    "homogenized anisotropic cell tensor"
                    if cell_law is not None
                    else "isotropic continuum power-law surrogate"
                ),
                "limitation": (
                    (
                        "The macro solve uses this cell family's homogenized "
                        "anisotropic tensor, so density and stiffness are "
                        "coupled through the real cell. Orientation is fixed "
                        "to the global axes and cell size is uniform, so a "
                        "load path oblique to those axes cannot yet rotate its "
                        "cells to meet it. A homogenized macro stress would "
                        "not be strut stress, so explicit-geometry validation "
                        "is still required. "
                        if cell_law is not None
                        else "This cell family has no homogenized law, so the "
                        "macro solve uses an isotropic continuum power-law "
                        "surrogate rather than a cell material card. "
                    )
                    + "The stretch-dominated Octet can use second-phase axial-"
                    "truss sizing for stress, Euler buckling, and displacement. "
                    "BCC is bending-dominated, so axial sizing is only an Expert "
                    "diagnostic and may expose a mechanism. The beam surrogate "
                    "does not resolve joint bending or local solid stress "
                    "concentrations. Honeycomb remains continuum-density sized."
                ),
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
        "thermal_cases": [
            {
                "name": name,
                "thermal_compliance": float(value),
                "maximum_temperature_rise": (
                    result.thermal_case_max_temperature_rises.get(name)
                ),
            }
            for name, value in result.thermal_case_compliances.items()
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
        "thermal_compliance": (
            float(result.thermal_compliance_history[-1])
            if result.thermal_compliance_history
            else None
        ),
        "thermal_compliance_history": result.thermal_compliance_history,
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
            "automatic_exclusion_scope": study.automatic_exclusion_scope,
            "automatic_exclusion_thickness": (
                study.automatic_exclusion_thickness
            ),
            "automatic_exclusion_unit": "mm",
        },
    }

    return TopologyOutputContext(
        payload=output,
        manufactured_density=manufactured_density,
        structure_options=structure_options,
        result=result,
        study=study,
    )


def _report_lattice_printability(
    node: Any,
    output: dict[str, Any],
    structure_options: Any,
    problem: Any,
    block: Callable[[str, str], None],
    advise: Callable[[str, str], None],
) -> None:
    """Check the built cells against the process, not only the envelope.

    The optimizer's manufacturing constraints act on the macro density field.
    The cells are built afterwards, from the family's own geometry, so neither
    the printable length scale nor the overhang direction reaches them. Both
    are measured here on what was actually built and attached to the report.
    """
    # The built cell's dimensions are published under `manufacturing`, beside
    # the voxel length they are denominated in.
    manufacturing_report = output.get("manufacturing") or {}
    voxel_size = manufacturing_report.get("voxel_size")
    minimum_feature = node_minimum_feature_size(node)

    feature_report = lattice_feature_size_report(
        structure_options,
        voxel_size,
        minimum_feature,
    )
    if feature_report is not None:
        output.setdefault("lattice_optimization", {})
        if isinstance(output["lattice_optimization"], dict):
            output["lattice_optimization"]["feature_size_check"] = feature_report
        for violation in feature_report["violations"]:
            block(
                "lattice_below_minimum_feature_size",
                f"The manufactured structure is below the stated printable "
                f"minimum: {violation} (model units). Raise the minimum member "
                "size, coarsen the cell, or state a capability the process can "
                "actually build.",
            )
    elif minimum_feature <= 0.0:
        advise(
            "lattice_feature_size_unchecked",
            "No minimum member size is stated, so the manufactured "
            f"{structure_options.display_name.lower()} wall/strut was not "
            "checked against a process capability. Set the minimum member "
            "size under Design Intent to have it enforced.",
        )

    overhang_report = lattice_overhang_report(
        structure_options,
        getattr(problem.mc, "overhang_build_axis", "none"),
        getattr(problem.mc, "overhang_angle_deg", SELF_SUPPORTING_ANGLE_DEG),
        voxel_scale=(
            problem.unitx,
            problem.unity,
            problem.unitz,
        ),
    )
    if overhang_report is not None:
        if isinstance(output.get("lattice_optimization"), dict):
            output["lattice_optimization"]["overhang_check"] = overhang_report
        message = overhang_advice(overhang_report)
        if message:
            advise("lattice_member_overhang", message)


def finalize_topology_output(
    node: Any,
    context: TopologyOutputContext,
    cancel_callback: CancelCallback | None,
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
    guided_mode = (
        str(node.get_property("workflow_mode") or "Guided").strip().lower()
        == "guided"
    )
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
            "use an edge or vertex. They were regularized into finite voxel "
            "contact pads, but the resulting branches remain sensitive to "
            "that artificial pad. Select the physical CAD interface face for "
            "a production study.",
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
    if structure_options.mode != "solid":
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
        if surface_quality.get("watertight") is False:
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
            advise(
                "self_intersection_not_screened",
                "The recovered surface was not screened for self-intersection "
                "because the spatial-index backend is unavailable. A watertight "
                "mesh can still pass through itself; check it in the slicer or "
                "mesher before release.",
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
    requested_member = float(getattr(problem, "minimum_solid_size", 0.0) or 0.0)
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
    lattice_optimization = output.get("lattice_optimization") or {}
    member_sizing = lattice_optimization.get("member_sizing") or {}
    guided_member_sizing = bool(
        guided_mode
        and structure_options.family is not None
        and structure_options.family.maxwell == "stretch"
    )
    sizing_requested = (
        lattice_cad_strategy(structure_options) == "beam"
        and (
            guided_member_sizing
            or (
                not guided_mode
                and as_bool(node.get_property("optimize_lattice_members"))
            )
        )
    )
    if sizing_requested and not lattice_optimization.get(
        "member_size_optimization"
    ):
        block(
            "lattice_member_sizing_failed",
            "Load-aware lattice member sizing was requested but did not "
            f"complete: {member_sizing.get('reason') or 'unknown error'}.",
        )
    elif sizing_requested:
        if not bool(member_sizing.get("converged")):
            block(
                "lattice_member_sizing_not_converged",
                "The second-phase member-sizing iteration did not converge.",
            )
        stress_utilization = float(
            member_sizing.get("maximum_stress_utilization") or 0.0
        )
        buckling_utilization = float(
            member_sizing.get("maximum_buckling_utilization") or 0.0
        )
        if stress_utilization > 1.005:
            block(
                "lattice_member_stress_exceeded",
                "The axial-truss sizing stage exceeds its allowable-stress "
                f"limit (utilization {stress_utilization:.3g}).",
            )
        if buckling_utilization > 1.005:
            block(
                "lattice_member_buckling_exceeded",
                "The axial-truss sizing stage exceeds Euler buckling "
                f"(utilization {buckling_utilization:.3g}).",
            )
        sizing_displacement_limit = float(
            node.get_property("validation_displacement_limit_mm") or 0.0
        )
        sized_displacement = float(
            member_sizing.get("maximum_displacement") or 0.0
        )
        if (
            sizing_displacement_limit > 0.0
            and sized_displacement > 1.005 * sizing_displacement_limit
        ):
            block(
                "lattice_member_displacement_exceeded",
                "The axial-truss sizing stage exceeds the displacement limit "
                f"({sized_displacement:.3g} mm > "
                f"{sizing_displacement_limit:.3g} mm).",
            )
        if any(
            "mechanism" in str(message).lower()
            for message in member_sizing.get("warnings", ())
        ):
            block(
                "lattice_axial_mechanism",
                "The lattice beam surrogate contains an axial mechanism or "
                "near-mechanism; use the stretch-dominated Octet Truss family "
                "or revise the restraints.",
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
    if structure_options.mode != "solid":
        _report_lattice_printability(
            node,
            output,
            structure_options,
            problem,
            block,
            advise,
        )
    if (
        structure_options.mode != "solid"
        and structure_options.skin_thickness_voxels > 0.0
    ):
        advise(
            "sealed_lattice_skin",
            f"{structure_options.display_name} is enclosed by a solid skin "
            f"({structure_options.skin_thickness_voxels:g} voxels), so the "
            "lattice is sealed inside and not visible on the recovered "
            "surface. On a powder-bed process a closed skin also traps "
            "unfused powder. Set the skin thickness to 0 to expose the "
            "lattice and allow powder removal."
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
    validation_requested = as_bool(node.get_property("validate_after_optimize"))
    if float(result.min_source_volfrac or 0.0) > float(problem.volfrac) + 1e-6:
        block(
            "passive_material_exceeds_budget",
            "Target volume is below the fixed passive material; final "
            "volume cannot reach the requested fraction unless passive "
            "solid regions are reduced."
        )
    validation_completed = False
    if validation_requested and problem.physics_mode == "thermal":
        block(
            "thermal_validation_unavailable",
            "Structural CalculiX validation is not applicable to a "
            "thermal-only topology study. The result remains concept-level "
            "until it is checked in an independent thermal model."
        )
    elif validation_requested and jointed_multibody:
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

    # An extrusion-constrained solid envelope is a swept profile, so it has an
    # exact editable B-rep and always receives one. A general 3-D solid does
    # not: freeform patch fitting used to supply one, and what it delivered was
    # a smooth body that no longer described the load path.
    #
    # A strut lattice does have an exact B-rep, and it does not come from the
    # isosurface: the sized centrelines already describe the body, so it is one
    # quadric per member and one ball per joint, with a face count that tracks
    # the member count instead of the voxel count (measured: 0.1 s and 488
    # faces for a 140-member BCC envelope, 1.2 s and 703 faces for a 200-member
    # octet). That path is built and validated in :mod:`..geometry.lattice_cad`
    # and it is reached here rather than being left for an export button.
    #
    # The minimal-surface and prismatic families are the ones with no compact
    # boundary representation -- a gyroid is not a union of quadrics -- so they
    # keep the mesh route and are released as STL.
    has_extrusion_axis = str(output.get("extrusion_axis") or "none").strip().lower() in {
        "x",
        "y",
        "z",
    }
    analytic_lattice_cad = lattice_cad_strategy(structure_options) == "beam"
    eager_cad = bool(
        analytic_lattice_cad
        or (
            structure_options.mode == "solid"
            and has_extrusion_axis
            and output.get("recovered_shape") is not None
        )
    )
    if eager_cad:
        try:
            cad_shape = node._run_embedded_cad_reconstruction(output)
            output["cad_shape"] = cad_shape
            output["shape"] = cad_shape
            output["visualization_mode"] = "CAD"
            reconstruction_report = dict(
                getattr(node, "_last_cad_reconstruction_report", {}) or {}
            )
            try:
                solid = cad_shape.val() if hasattr(cad_shape, "val") else cad_shape
                output["cad_reconstruction"] = reconstruction_report | {
                    "method": reconstruction_report.get(
                        "method", "Recovered Shape"
                    ),
                    "valid": bool(solid.isValid())
                    if hasattr(solid, "isValid")
                    else None,
                    "volume": float(solid.Volume())
                    if hasattr(solid, "Volume")
                    else None,
                }
            except Exception:
                output["cad_reconstruction"] = reconstruction_report or {
                    "method": "Recovered Shape",
                }
            if output["cad_reconstruction"].get("valid") is False:
                block(
                    "cad_reconstruction_invalid",
                    "The reconstructed CAD solid is invalid and must be repaired "
                    "before export or downstream meshing.",
                )
        except Exception as exc:
            msg = f"CAD reconstruction failed: {exc}"
            logger.warning("TopologyOptVoxelNode: %s", msg)
            output["cad_error"] = str(exc)
            block("cad_reconstruction_failed", msg)
            if analytic_lattice_cad and output.get("recovered_shape") is not None:
                # The centrelines did not become a body, but the rasterized
                # lattice is still there and is still what was optimized.
                output["visualization_mode"] = "Manufactured Mesh"
    elif (
        output.get("recovered_shape") is not None
        and structure_options.mode != "solid"
    ):
        output["cad_reconstruction"] = {
            "method": "Lattice mesh (no exact B-rep)",
            "skipped": True,
            "editable": False,
            "reason": (
                "A minimal-surface or prismatic cell has no compact boundary "
                "representation, so this result is released as STL rather than "
                "as STEP. The strut families (BCC, Octet Truss) are "
                "reconstructed exactly from their centrelines instead."
            ),
        }
        output["visualization_mode"] = "Manufactured Mesh"
    elif (
        output.get("recovered_shape") is not None
        and structure_options.mode == "solid"
    ):
        output["cad_reconstruction"] = {
            "method": "Recovered surface (no exact B-rep)",
            "skipped": True,
            "editable": False,
            "reason": (
                "A general three-dimensional load path has no exact editable "
                "B-rep. Constrain the study to an extrusion axis for a solid "
                "CAD body, or take this surface as STL."
            ),
        }
        output["visualization_mode"] = "Surface"
    elif output.get("recovered_shape") is not None:
        output["cad_reconstruction"] = {
            "method": "Not requested",
            "skipped": True,
            "editable": None,
        }

    verification_passed = validation_completed and not failed_checks
    # Compatibility alias for existing graph consumers. This means the
    # checks configured in this study passed; it is not product certification
    # or an automatic release decision.
    output["release_ready"] = verification_passed
    output["verification_passed"] = verification_passed
    output["quality_gate"] = {
        "status": "verification passed" if verification_passed else "concept only",
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
