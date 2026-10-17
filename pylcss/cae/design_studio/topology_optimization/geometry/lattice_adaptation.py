"""Continuous 2:1 cell refinement in a geometric boundary band.

Both family fields use the same physical phase origin and member floor. Their
overlap is a material union, not a coordinate warp or a conformal-cell claim.
Final source/void CSG and prescribed-region connectivity checks remain required.
"""
from __future__ import annotations

import numpy as np

from ..manufacturing import family_for
from ..manufacturing.lattice_families import tpms_specific_surface_area


class LatticeAdaptationError(ValueError):
    """A half-pitch cell cannot retain the requested physical member floor."""

    def __init__(self, message, report):
        super().__init__(message)
        self.report = report


def plan_lattice_adaptation(nominal_pitch, minimum_member, family, *, boundary_band=None, overlap=None):
    """Plan one aligned half-period boundary layer without changing dimensions.

    ``pitch`` denotes the cubic period for TPMS/strut cells and hexagon side
    length for Honeycomb. A TPMS floor must also fit below the evaluator's80%
    wall-derived occupancy cap; checking member<pitch alone is insufficient.
    """
    family = family_for(family) if isinstance(family, str) else family
    if family is None:
        raise ValueError("Choose a supported cell family for boundary refinement.")
    nominal_pitch, minimum_member = float(nominal_pitch), float(minimum_member)
    if not np.isfinite([nominal_pitch, minimum_member]).all() or min(nominal_pitch, minimum_member) <= 0:
        raise ValueError("Boundary refinement needs positive finite physical cell and member dimensions.")
    refined_pitch = nominal_pitch / 2.
    boundary_band = nominal_pitch / 2. if boundary_band is None else float(boundary_band)
    overlap = nominal_pitch / 4. if overlap is None else float(overlap)
    if not np.isfinite([boundary_band, overlap]).all() or not 0 < overlap < boundary_band:
        raise ValueError("Boundary refinement needs 0 < overlap < boundary-band depth.")
    if family.is_tpms:
        area_factor = tpms_specific_surface_area(family.key)
        if family.wall_mode == "skeletal":
            area_factor *= .5
        minimum_pitch = max(minimum_member, area_factor * minimum_member / .8)
        feasible = refined_pitch > minimum_member and refined_pitch >= minimum_pitch * (1. - 1e-12)
        constraint = "TPMS wall-derived occupancy must remain at or below80%"
    elif family.kind == "prismatic" and family.base == "honeycomb":
        # Opposite flats are sqrt(3)*side apart; inward wall halves must not
        # consume that opening. This is geometry, not a sampling allowance.
        minimum_pitch = minimum_member / np.sqrt(3.)
        feasible = refined_pitch > minimum_pitch
        constraint = "Honeycomb wall thickness must be less than its across-flats cell width"
    else:
        minimum_pitch = minimum_member
        feasible = refined_pitch > minimum_pitch
        constraint = "Strut diameter must remain below the refined cell pitch"
    if not feasible and minimum_pitch * 1.05 < nominal_pitch:
        refined_pitch = max(refined_pitch, minimum_pitch * 1.05)
        feasible = True
    report = {
        "method": "Boundary cell refinement with continuous overlapping material union (up to2:1)",
        "family": family.key,
        "nominal_pitch_mm": nominal_pitch,
        "refined_pitch_mm": refined_pitch,
        "sampling_pitch_mm": refined_pitch,
        "refinement_ratio": nominal_pitch / refined_pitch,
        "minimum_member_mm": minimum_member,
        "member_floor_preserved": True,
        "boundary_band_mm": boundary_band,
        "overlap_mm": overlap,
        "pure_refined_clearance_limit_mm": boundary_band - overlap,
        "pure_coarse_clearance_start_mm": boundary_band + overlap,
        "minimum_open_refined_pitch_mm": float(minimum_pitch),
        "refined_pitch_feasible": bool(feasible),
        "feasibility_constraint": constraint,
        "phase_origin_rule": "Both native family fields use the same physical origin",
        "physical_cell_refinement": True,
        "conformal_coordinate_warp": False,
        "connectivity_certified": False,
        "required_final_validation": "One final contour with closed/winding, source/void, bore and prescribed-material connectivity checks",
    }
    if not feasible:
        raise LatticeAdaptationError(
            f"Boundary refinement of{nominal_pitch:g} mm {family.display_name} cells needs "
            f"{refined_pitch:g} mm cells, which cannot retain the{minimum_member:g} mm minimum "
            f"wall/strut with an open cell ({constraint}). Increase nominal cell size or "
            "reduce the minimum thickness. Neither requested dimension was changed.", report)
    return report


def blend_boundary_lattice_fields(coarse_level, refined_level, clearance, *, nominal_pitch,
                                 minimum_member, family, boundary_band=None, overlap=None):
    """Union gated native fields using continuous signed CAD/STL clearance.

    For positive-inside level fields and clearance d:
        max(min(coarse,d-band+overlap),min(refined,band+overlap-d)).
    The positive material is purely half-period next to the boundary and purely
    nominal in the broad interior, with both present in a finite overlap band.
    All supplied fields remain unchanged. This helper does not trim the outer
    CAD boundary or turn a disconnected transition into a certified body.
    """
    report = plan_lattice_adaptation(nominal_pitch, minimum_member, family,
                                    boundary_band=boundary_band, overlap=overlap)
    coarse, refined, distance = map(np.asarray, (coarse_level, refined_level, clearance))
    if coarse.ndim != 3 or coarse.shape != refined.shape or coarse.shape != distance.shape:
        raise ValueError("Coarse cells, refined cells and geometric clearance need the same3-D grid.")
    if not all(np.issubdtype(a.dtype, np.number) and np.isfinite(a).all() for a in (coarse, refined, distance)):
        raise ValueError("Boundary refinement fields must contain finite numeric values.")
    dtype = np.result_type(coarse.dtype, refined.dtype, distance.dtype, np.float32)
    result = np.empty(coarse.shape, dtype=dtype)
    other = np.empty(coarse.shape, dtype=dtype)
    band, overlap = report["boundary_band_mm"], report["overlap_mm"]
    np.subtract(distance, band - overlap, out=result)
    np.minimum(coarse, result, out=result)
    np.subtract(band + overlap, distance, out=other)
    np.minimum(refined, other, out=other)
    np.maximum(result, other, out=result)
    # Counts refer to allowed geometric regions, not mesh triangle density.
    interior = distance > 0.
    near = interior & (distance < band - overlap)
    far = interior & (distance > band + overlap)
    transition = interior & ~near & ~far
    report.update(
        grid_shape=list(coarse.shape),
        refined_boundary_sample_count=int(np.count_nonzero(near)),
        transition_sample_count=int(np.count_nonzero(transition)),
        nominal_interior_sample_count=int(np.count_nonzero(far)),
        refined_material_sample_count=int(np.count_nonzero(near & (result > 0))),
        transition_material_sample_count=int(np.count_nonzero(transition & (result > 0))),
        nominal_material_sample_count=int(np.count_nonzero(far & (result > 0))),
        pure_refined_material_matches_native=bool(np.array_equal((result > 0)[near], (refined > 0)[near])),
        pure_nominal_material_matches_native=bool(np.array_equal((result > 0)[far], (coarse > 0)[far])),
        nominal_field_identical_sample_count=int(np.count_nonzero(far & (result == coarse))),
        continuity="Min/max of continuous native level sets and geometric clearance",
        transition="Both aligned native pitches are present in the overlap; final attachment gate is mandatory",
    )
    return result, report


__all__ = ["LatticeAdaptationError", "plan_lattice_adaptation", "blend_boundary_lattice_fields"]
