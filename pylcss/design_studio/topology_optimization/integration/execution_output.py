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

from ..geometry.surface_recovery import _recover_voxel_shape
from ..manufacturing.structures import (
    ManufacturingStructureOptions,
    build_manufacturing_field,
    passive_region_masks,
    structure_options_from_values,
)
from ..optimization.results import TopologyOptVoxelResult
from .boundary_mapping import _bounds_payload
from .execution_setup import PreparedTopologyStudy
from .voxelization import (
    _effective_density_cutoff,
    _fractional_cylinder_volume,
    _meaningful_material_components,
    _recovered_shape_volume,
    _source_material_fraction,
    _source_volume_fraction,
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
    logger.info("TopologyOptVoxelNode: %s", result.message)
    density = np.asarray(result.density, dtype=float)
    density_cutoff = _effective_density_cutoff(
        node.get_property("density_cutoff") or 0.45
    )
    print_ready = as_bool(node.get_property("print_ready_mesh"))
    decimate = float(node.get_property("mesh_decimate_ratio") or 1.0)
    try:
        structure_options = structure_options_from_values(
            node.get_property("structure_mode"),
            node.get_property("structure_cell_size_voxels"),
            node.get_property("structure_member_thickness_voxels"),
            node.get_property("structure_skin_thickness_voxels"),
            node.get_property("lattice_variable_density"),
            node.get_property("lattice_min_relative_density"),
            node.get_property("lattice_max_relative_density"),
            node.get_property("lattice_solid_transition_density"),
        )
    except (TypeError, ValueError) as exc:
        node.set_error(f"Invalid rib/lattice manufacturing settings: {exc}")
        return None
    if structure_options.mode != "solid":
        print_ready = True
    surface_backend = (
        "legacy"
        if str(node.get_property("surface_recovery_method") or "")
        .strip()
        .lower()
        .startswith("legacy")
        else "vtk_sdf"
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
    manufactured_density = None
    if structure_options.mode != "solid":
        manufactured_density = build_manufacturing_field(
            density,
            density_cutoff,
            structure_options,
            passive_solid_mask=passive_solid_mask,
            passive_void_mask=passive_void_mask,
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
    # Key on the actual density bytes — not id(result.density), which Python's
    # allocator can reuse across runs and silently return a stale recovered
    # shape for a different solve.
    density_view = np.ascontiguousarray(result.density)
    cache_key = (
        hash(density_view.tobytes()),
        density_view.shape,
        density_cutoff,
        print_ready,
        decimate,
        surface_backend,
        structure_options,
        str(mc.extrusion),
        str(bc.solid_boxes),
        str(bc.void_boxes),
        str(bc.solid_cylinders),
        str(bc.void_cylinders),
        str(bc.joint_pin_cylinders),
        hash(np.ascontiguousarray(passive_solid_mask).tobytes()),
        hash(np.ascontiguousarray(passive_void_mask).tobytes()),
    )
    if getattr(node, "_last_recovery_key", None) == cache_key:
        recovered = node._last_recovered_shape
    else:
        recovered = _recover_voxel_shape(
            density,
            bounds,
            density_cutoff,
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
            surface_backend=surface_backend,
        )
        node._last_recovery_key = cache_key
        node._last_recovered_shape = recovered
    if bounds is not None:
        mins, maxs = bounds
        total_volume = float(np.prod(np.maximum(maxs[:3] - mins[:3], 0.0)))
    else:
        total_volume = float(np.prod(density.shape))
    source_volume_fraction = _source_volume_fraction(density, design_domain)
    source_volume = float(total_volume * source_volume_fraction)
    source_material_fraction = _source_material_fraction(density, design_domain)
    final_volume = float(source_material_fraction * source_volume)
    material_density = float(material.get("rho", material.get("density", 0.0)))
    manufactured_material_fraction = (
        _source_material_fraction(manufactured_density, design_domain)
        if manufactured_density is not None
        else source_material_fraction
    )
    manufactured_volume = float(manufactured_material_fraction * source_volume)
    recovered_assembly_volume = _recovered_shape_volume(recovered)
    assembly_hardware_volume = float(
        sum(_fractional_cylinder_volume(pin, bounds) for pin in bc.joint_pin_cylinders)
    )
    recovered_design_volume = (
        max(0.0, recovered_assembly_volume - assembly_hardware_volume)
        if recovered_assembly_volume is not None
        else None
    )
    recovery_volume_delta_pct = (
        100.0 * (recovered_design_volume / final_volume - 1.0)
        if (recovered_design_volume is not None and final_volume > 1e-12)
        else None
    )
    output: dict[str, Any] = {
        "type": "topopt_voxel",
        "design_goal": design_goal,
        "objective_mode": problem.objective_mode,
        "physics_mode": problem.physics_mode,
        "optimizer_used": result.optimizer_used,
        "formulation_used": result.formulation_used,
        "load_aggregation": problem.load_aggregation,
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
        "density_cutoff": density_cutoff,
        "recovered_shape": recovered,
        "extrusion_axis": mc.extrusion,
        "visualization_mode": node.get_property("visualization") or "Density",
        "target_vol_frac": problem.volfrac,
        "final_vol_frac": source_material_fraction,
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
        "manufacturing": {
            "structure": structure_options.display_name,
            "cell_size_voxels": structure_options.cell_size_voxels,
            "member_thickness_voxels": (structure_options.member_thickness_voxels),
            "skin_thickness_voxels": structure_options.skin_thickness_voxels,
            "surface_backend": (
                recovered.get("surface_backend")
                if isinstance(recovered, dict)
                else None
            ),
            "material_fraction": manufactured_material_fraction,
            "volume": manufactured_volume,
            "mass": manufactured_volume * material_density,
            "requires_independent_reanalysis": (structure_options.mode != "solid"),
            "meaningful_component_count": component_count,
            "source_body_count": source_component_count,
            "component_voxels": component_voxels,
        },
        "lattice_optimization": (
            {
                "method": "density-guided explicit lattice interpretation",
                "cell_family": structure_options.display_name,
                "continuum_surrogate_penalty": float(problem.penal),
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
                "independent_validation_required": True,
            }
            if structure_options.mode
            in {"gyroid", "diamond", "honeycomb", "cubic", "octet"}
            and structure_options.variable_density
            else None
        ),
        "level_set_field": (
            np.asarray(result.level_set_field, dtype=float)
            if result.level_set_field is not None
            else None
        ),
        "compliance": (
            float(result.compliance_history[-1]) if result.compliance_history else None
        ),
        "stress_pnorm": (
            float(result.stress_history[-1]) if result.stress_history else None
        ),
        "iterations": result.n_iter,
        "max_iterations": problem.max_iter,
        "converged": result.converged,
        "message": result.message,
        "compliance_history": result.compliance_history,
        "objective_history": result.objective_history,
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
        },
    }

    return TopologyOutputContext(
        payload=output,
        manufactured_density=manufactured_density,
        structure_options=structure_options,
        result=result,
        study=study,
    )


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
    warnings_out: list[str] = []
    if not result.converged:
        warnings_out.append(
            f"The optimizer stopped at {result.n_iter} iterations before "
            "meeting its convergence criteria. Treat this result as a "
            "preview and increase iterations or revise the study before "
            "engineering release."
        )
    if (
        result.stress_history
        and float(result.stress_history[-1]) > float(problem.yield_stress) * 1.01
    ):
        warnings_out.append(
            f"The stress P-norm constraint is not satisfied "
            f"({float(result.stress_history[-1]):.3g} MPa > "
            f"{float(problem.yield_stress):.3g} MPa allowable). The "
            "minimum-mass result is not feasible."
        )
    validation_requested = as_bool(node.get_property("validate_after_optimize"))
    if structure_options.mode == "topology_ribs" and not validation_requested:
        warnings_out.append(
            f"{structure_options.display_name} is an explicit manufacturing "
            "interpretation of the optimized load-path envelope. Enable "
            "independent validation before making an engineering decision."
        )
    if float(result.min_source_volfrac or 0.0) > float(problem.volfrac) + 1e-6:
        warnings_out.append(
            "Target volume is below the fixed passive material; final "
            "volume cannot reach the requested fraction unless passive "
            "solid regions are reduced."
        )
    if validation_requested and problem.physics_mode == "thermal":
        warnings_out.append(
            "Structural CalculiX validation is not applicable to a "
            "thermal-only topology study."
        )
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
                    "compliance": validation.get("compliance"),
                    "converged": (
                        study.get("converged") if isinstance(study, dict) else None
                    ),
                }
                validated_stress = output["validation_summary"].get("max_stress")
                allowable = float(
                    material.get(
                        "yield_strength",
                        material.get("yield", 0.0),
                    )
                    or 0.0
                )
                if (
                    validated_stress is not None
                    and allowable > 0.0
                    and float(validated_stress) > allowable
                ):
                    warnings_out.append(
                        f"Independent CalculiX validation exceeds material "
                        f"yield ({float(validated_stress):.3g} MPa > "
                        f"{allowable:.3g} MPa). The manufactured result "
                        "fails the allowable-stress check."
                    )
            else:
                warnings_out.append(
                    "Validation was requested but did not produce a CalculiX result."
                )
        except Exception as exc:
            msg = f"Validation skipped/failed: {exc}"
            logger.warning("TopologyOptVoxelNode: %s", msg)
            output["validation_error"] = str(exc)
            warnings_out.append(msg)

    if as_bool(node.get_property("generate_cad_after_optimize")):
        try:
            cad_shape = node._run_embedded_cad_reconstruction(output)
            output["cad_shape"] = cad_shape
            output["shape"] = cad_shape
            try:
                solid = cad_shape.val() if hasattr(cad_shape, "val") else cad_shape
                output["cad_reconstruction"] = {
                    "method": "Recovered Shape",
                    "valid": bool(solid.isValid())
                    if hasattr(solid, "isValid")
                    else None,
                    "volume": float(solid.Volume())
                    if hasattr(solid, "Volume")
                    else None,
                }
            except Exception:
                output["cad_reconstruction"] = {
                    "method": "Recovered Shape",
                }
        except Exception as exc:
            msg = f"CAD reconstruction failed: {exc}"
            logger.warning("TopologyOptVoxelNode: %s", msg)
            output["cad_error"] = str(exc)
            warnings_out.append(msg)

    if warnings_out:
        output["warnings"] = warnings_out
    return output
