# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE.
"""Generate explicit graded cells from a completed solid TopOpt result."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter

from ..manufacturing import family_for
from ..manufacturing.structures import (
    ManufacturingStructureOptions,
    passive_region_masks,
)
from .lattice_fit import fit_lattice_dimensions, MAX_POST_LATTICE_VOXELS
from .voxelization import voxel_size_from_bounds

POST_LATTICE_DEFAULTS = {
    "post_lattice_type": "Gyroid Lattice",
    "post_lattice_cell_size_mm": 0.0,
    "post_lattice_min_thickness_mm": 0.0,
    "post_lattice_skin_thickness_mm": 0.0,
    "post_lattice_min_density": 0.15,
    "post_lattice_max_density": 0.60,
}


def topology_source(result: dict[str, Any]) -> dict[str, Any]:
    """Always regenerate from the original solve, including after reopening."""
    return result.get("topology_source") or result


def _copy_containers(value):
    """Give the archived solve independent containers for sidecar persistence."""
    if isinstance(value, dict):
        return {key: _copy_containers(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_copy_containers(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_copy_containers(item) for item in value)
    return value


def prepare_lattice_field(result: dict[str, Any], settings: dict[str, Any]):
    """Separate the optimized envelope from cell density and printable floors.

    A converged SIMP field is almost binary. Using its near-one values as cell
    density would make every cell solid. Remap the retained, smoothed design
    field into the chosen cell-density range instead; only passive pads stay
    solid. This is geometric density mapping, not a new mechanical optimum.
    """
    source = topology_source(result)
    values = {**POST_LATTICE_DEFAULTS, **settings}
    family = family_for(values["post_lattice_type"])
    if family is None:
        raise ValueError("Choose a supported lattice type.")
    raw = source.get("recovery_density")
    if raw is None:
        raw = source.get("density")
    field = np.asarray(raw, dtype=float)
    if field.ndim != 3 or not np.all(np.isfinite(field)):
        raise ValueError("Run TopOpt first to obtain a finite 3-D density field.")
    bounds_data = source.get("bounds") or {}
    lower = np.asarray(bounds_data.get("min"), dtype=float)
    upper = np.asarray(bounds_data.get("max"), dtype=float)
    if (
        lower.shape != (3,)
        or upper.shape != (3,)
        or not np.all(np.isfinite([lower, upper]))
        or np.any(upper <= lower)
    ):
        raise ValueError("The TopOpt result needs valid physical bounds.")
    bounds = (lower, upper)
    cutoff = float(source.get("recovery_cutoff") or source.get("density_cutoff") or 0.3)
    envelope = field >= cutoff
    domain = source.get("design_domain")
    if domain is not None:
        if np.shape(domain) != field.shape:
            raise ValueError("The design-domain mask does not match the solved grid.")
        envelope &= np.asarray(domain, dtype=bool)
    solid = np.zeros(field.shape, dtype=bool)
    void = np.zeros(field.shape, dtype=bool)
    for key, mask in (("passive_solid_mask", solid), ("passive_void_mask", void)):
        saved = source.get(key)
        if saved is not None:
            if np.shape(saved) != field.shape:
                raise ValueError("Preserved-region masks do not match the solved grid.")
            mask |= np.asarray(saved, dtype=bool)
    regions = source.get("passive_regions") or {}
    analytic_solid, analytic_void = passive_region_masks(
        field.shape,
        **{
            key: regions.get(key) or ()
            for key in (
                "solid_boxes",
                "void_boxes",
                "solid_cylinders",
                "void_cylinders",
            )
        },
    )
    solid |= analytic_solid
    void |= analytic_void
    solid &= ~void
    envelope = (envelope | solid) & ~void
    if not np.any(envelope & ~solid):
        raise ValueError("The result has no design material available for a lattice.")
    minimum = float(values["post_lattice_min_density"])
    maximum = float(values["post_lattice_max_density"])
    if (
        not np.isfinite([minimum, maximum]).all()
        or not 0.02 <= minimum <= maximum <= 0.80
    ):
        raise ValueError("Cell densities require 2% <= minimum <= maximum <= 80%.")
    lengths = [
        float(values[key])
        for key in (
            "post_lattice_cell_size_mm",
            "post_lattice_min_thickness_mm",
            "post_lattice_skin_thickness_mm",
        )
    ]
    if not np.isfinite(lengths).all() or min(lengths) < 0:
        raise ValueError("Lattice dimensions must be finite and non-negative.")
    design_region = envelope & ~solid
    cell, member, skin, solid_features, fit_report = fit_lattice_dimensions(
        design_region,
        bounds,
        family,
        requested_cell=lengths[0],
        requested_member=lengths[1],
        requested_skin=lengths[2],
    )
    fit_report["non_design_solid_voxels"] = int(np.count_nonzero(solid))
    values["_fit_report"] = fit_report
    solid |= solid_features
    options = ManufacturingStructureOptions(
        mode=family.key,
        cell_size_voxels=cell,
        member_thickness_voxels=member,
        skin_thickness_voxels=skin,
        variable_density=True,
        minimum_relative_density=minimum,
        maximum_relative_density=maximum,
        solid_transition_density=1.0,
    )
    design = source.get("design_density")
    design = field if design is None else np.asarray(design, dtype=float)
    if design.shape != field.shape or not np.isfinite(design).all():
        raise ValueError(
            "The saved design-density field does not match the solved grid."
        )
    weight = gaussian_filter(design_region.astype(float), 1.0)
    smooth = gaussian_filter(np.where(design_region, design, 0.0), 1.0) / np.maximum(
        weight, 1e-12
    )
    normalized = np.clip((smooth - cutoff) / max(1.0 - cutoff, 1e-6), 0.0, 1.0)
    graded = np.where(envelope, minimum + (maximum - minimum) * normalized, 0.0)
    graded[solid & ~void] = 1.0
    return source, values, graded, bounds, minimum * 0.5, envelope, solid, void, options


def generate_post_topopt_lattice(
    result: dict[str, Any],
    settings: dict[str, Any],
    *,
    cancel_callback=None,
    progress_callback=None,
) -> dict[str, Any]:
    from ..geometry.lattice_cad import topopt_export_formats
    from ..geometry.surface_recovery import _recover_voxel_shape
    from ..geometry.cad_features import source_bore_cylinders

    def stage(label):
        if callable(cancel_callback) and cancel_callback():
            raise InterruptedError("Lattice generation cancelled.")
        if callable(progress_callback):
            progress_callback({"status_only": True, "stage": label}, None, 0, 0)

    stage("Mapping optimized density to lattice thickness")
    source, values, field, bounds, cutoff, envelope, solid, void, options = (
        prepare_lattice_field(result, settings)
    )
    stage("Generating lattice cells and recovering the surface")
    regions = dict(source.get("passive_regions") or {})
    regions['void_cylinders'] = list(regions.get('void_cylinders') or ()) + source_bore_cylinders(source)
    recovered = _recover_voxel_shape(
        field,
        bounds,
        cutoff,
        source_mask=envelope,
        include_manufacturing_field=True,
        cubic_lattice_grid=True,
        max_recovery_voxels=MAX_POST_LATTICE_VOXELS,
        raise_errors=True,
        progress_stage_callback=stage,
        passive_solid_mask=solid,
        passive_void_mask=void,
        structure_options=options,
        explicit_solid_shapes=regions.get("explicit_solid_shapes") or (),
        explicit_void_shapes=regions.get("explicit_void_shapes") or (),
        source_cad_shape=source.get("source_cad_shape"),
        **{
            key: regions.get(key) or ()
            for key in (
                "solid_boxes",
                "void_boxes",
                "solid_cylinders",
                "void_cylinders",
                "joint_pin_cylinders",
            )
        },
    )
    stage("Checking generated lattice")
    if not recovered or not len(recovered.get("faces", [])):
        raise ValueError(
            "Lattice generation produced no surface. Increase cell size or minimum thickness."
        )
    if not recovered.get("is_watertight") or not recovered.get("volume", 0):
        raise ValueError(
            "The lattice surface is not closed. Increase cell size or minimum thickness."
        )
    manufactured = recovered.pop("manufactured_density", None)
    resolved = recovered.get("resolved_structure_options")
    if isinstance(resolved, dict):
        options = replace(options, **resolved)
    output = dict(source)
    # Invalidate all geometry-dependent analysis and CAD results.
    for key in (
        "cad_shape",
        "shape",
        "cad_preview_mesh",
        "cad_reconstruction",
        "cad_error",
        "validation",
        "validation_summary",
        "validation_error",
        "surface_layout",
        "member_plan",
        "lattice_optimization",
        "manufactured_density",
        "manufacturing",
        "_beam_lattice_surface",
        "_cad_preview",
    ):
        output.pop(key, None)
    output.update(
        topology_source=_copy_containers(source),
        study_kind="post_topopt_lattice",
        recovered_shape=recovered,
        recovery_density=field,
        recovery_cutoff=cutoff,
        density=field,
        density_cutoff=cutoff,
        design_domain=envelope,
        manufactured_density=manufactured,
        passive_solid_mask=solid,
        passive_void_mask=void,
        structure_options=options,
        requested_structure_options=options,
        visualization_mode="Lattice",
        export_formats=list(topopt_export_formats(options)),
        release_ready=False,
        verification_passed=False,
        quality_gate={
            "status": "independent lattice validation required",
            "release_ready": False,
            "verification_passed": False,
            "independent_validation_completed": False,
        },
        generated_lattice={
            "settings": {
                key: value
                for key, value in values.items()
                if key.startswith("post_lattice_")
            },
            "fit": values["_fit_report"],
            "cell_family": options.display_name,
            "cell_size_mm": options.cell_size_voxels
            * voxel_size_from_bounds(field.shape, bounds),
            "method": "Smoothed TopOpt density mapped to local cell thickness",
            "independent_validation_required": True,
            "lattice_sizing": recovered.get("lattice_sizing") or {},
        },
        warnings=[
            "Mechanical results refer to the original solid TopOpt solve. Validate the generated lattice separately."
        ],
    )
    volume = float(recovered["volume"])
    old_volume = float(source.get("volume") or 0.0)
    material_density = (
        float(source.get("mass") or 0.0) / old_volume if old_volume > 0 else 0.0
    )
    output.update(
        volume=volume,
        mass=volume * material_density,
        recovered_design_volume=volume,
        recovered_design_mass=volume * material_density,
        recovered_assembly_volume=volume,
        recovered_assembly_mass=volume * material_density,
    )
    source_volume = float(source.get("source_volume") or 0.0)
    if source_volume > 0:
        output["final_vol_frac"] = volume / source_volume
        output["source_volume_fraction"] = volume / source_volume
    # Never display original solid stresses/compliance as properties of the
    # generated cells. The archived solve retains them for restoring the solid.
    for key in (
        "compliance",
        "stress_pnorm",
        "max_stress",
        "peak_displacement",
        "density_equivalent_volume",
        "density_equivalent_mass",
        "recovery_volume_delta_pct",
        "structural_cases",
        "quality",
        "thermal_cases",
        "case_max_displacements",
        "case_compliances",
        "thermal_compliance",
        "fea_analysis_status",
        "fea_analysis_error",
    ):
        output.pop(key, None)
    for key in list(output):
        if key.endswith("_history"):
            output.pop(key)
    output["manufacturing"] = {
        "surface_quality": recovered.get("surface_quality") or {},
        "requires_independent_reanalysis": True,
    }
    output["export_formats"] = list(topopt_export_formats(output))
    return output
