# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE.
"""Choose a printable periodic pitch from design-only local feature sizes."""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import skeletonize

from ..manufacturing.lattice_families import tpms_specific_surface_area

POST_LATTICE_QUALITY_BUDGETS = {"Standard": 8_000_000, "High": 64_000_000, "Professional": 128_000_000}
POST_LATTICE_QUALITY_SCALES = {"Standard": 1.0, "High": 1.5, "Professional": 2.0}
DEFAULT_POST_LATTICE_QUALITY = "High"
MAX_POST_LATTICE_VOXELS = POST_LATTICE_QUALITY_BUDGETS[DEFAULT_POST_LATTICE_QUALITY]
CELLS_ACROSS_FEATURE = 2.0


def resolve_lattice_quality(value=None):
    """Return one explicit sampling preset; physical dimensions are separate."""
    text = str(value or DEFAULT_POST_LATTICE_QUALITY).strip().lower()
    for name in POST_LATTICE_QUALITY_BUDGETS:
        if text == name.lower():
            return name
    raise ValueError("Lattice quality must be Standard, High, or Professional.")


def plan_lattice_sampling(span, pitch, member, family, surface_quality="High", max_samples=None,
                          *, refined_pitch=None):
    """Plan resolved continuous cells without changing pitch or member size.

    Each quality preset resolves at least its scaled family sampling guidance
    and three samples through the minimum wall/strut. Counts include the two
    exterior samples and interval endpoint used by geometry recovery.
    """
    span = np.asarray(span, dtype=float)
    if span.shape != (3,) or not np.isfinite(span).all() or np.any(span <= 0):
        raise ValueError("Lattice sampling needs finite positive 3-D bounds.")
    if not np.isfinite([pitch, member]).all() or min(pitch, member) <= 0:
        raise ValueError("Lattice sampling needs positive cell size and wall/strut thickness.")
    quality = resolve_lattice_quality(surface_quality)
    budget = POST_LATTICE_QUALITY_BUDGETS[quality] if max_samples is None else int(max_samples)
    if budget <= 0:
        raise ValueError("The lattice sampling budget must be positive.")
    samples_per_cell = POST_LATTICE_QUALITY_SCALES[quality] * float(family.minimum_cell_voxels)
    edge = min(float(pitch) / samples_per_cell, float(member) / 3.)
    if refined_pitch is not None:
        if not np.isfinite(refined_pitch) or not 0 < refined_pitch <= pitch:
            raise ValueError("Refined cell pitch must be positive and no larger than nominal pitch.")
        # Resolve both the nominal-cell quality guidance and every smaller
        # boundary cell's independent family minimum. No coarse grid is reused
        # for a finer physical pattern.
        edge = min(edge, float(refined_pitch) / float(family.minimum_cell_voxels))
    counts = np.maximum(2., np.ceil(span / edge)) + 3.
    required = float(np.prod(counts))
    if not np.isfinite(required) or required > budget:
        required_text = f"{required:,.0f}" if np.isfinite(required) else "an unbounded number of"
        raise ValueError(
            f"{quality} lattice quality needs {required_text} samples for the requested "
            f"{float(pitch):g} mm cell size and {float(member):g} mm minimum wall/strut; "
            f"its limit is {budget:,}. Choose a higher quality budget, a larger cell size, "
            "or a thicker minimum wall/strut. The requested dimensions were not changed."
        )
    from ..configuration.voxel_meshing import available_voxel_memory_bytes
    available = available_voxel_memory_bytes()
    estimated_memory = int(required*80.+512*1024**2)
    if available is not None and estimated_memory > .65*available:
        raise ValueError(
            f"The requested lattice needs approximately {estimated_memory/1024**3:.1f} GB of working memory; "
            f"only {available/1024**3:.1f} GB is currently available. Choose a lower sampling quality "
            "or a larger minimum wall/strut. Cell dimensions were not changed.")
    intervals = counts.astype(np.int64) - 3
    spacing = span / intervals
    report = {
        "quality": quality, "max_build_voxels": budget,
        "required_grid_samples": int(required), "grid_shape": counts.astype(int).tolist(),
        "requested_sampling_edge_mm": edge, "sampling_spacing_mm": spacing.tolist(),
        "minimum_cell_samples": samples_per_cell, "minimum_member_samples": 3,
        "refined_cell_pitch_mm": None if refined_pitch is None else float(refined_pitch),
        "minimum_refined_cell_samples": None if refined_pitch is None else float(family.minimum_cell_voxels),
        "physical_dimensions_preserved": True,
        "estimated_working_memory_bytes": estimated_memory,
    }
    return intervals, spacing, report


def design_feature_widths(design, spacing):
    """Approximate medial thickness in physical units, excluding passive pads.

    Sample distance on the medial skeleton, rather than over every boundary
    voxel. The latter mostly measures distance to the boundary and biases the
    size toward zero. Subtract one voxel edge for a conservative finite-grid
    estimate. A nearest-medial sample extends the widths to feature interiors.
    """
    design = np.asarray(design, dtype=bool)
    spacing = np.asarray(spacing, dtype=float)
    if not np.any(design):
        raise ValueError("No design material is available for lattice generation.")
    distance = ndi.distance_transform_edt(np.pad(design, 1), sampling=spacing)[
        1:-1, 1:-1, 1:-1
    ]
    medial = np.asarray(skeletonize(design), dtype=bool) & design
    # Lee thinning can remove even symmetric rods, including thin branches of
    # a larger body. Keep distance ridges as well so those features are sized.
    medial |= design & (distance == ndi.maximum_filter(distance, size=3))
    widths = np.maximum(2.0 * distance - np.min(spacing), np.min(spacing))
    samples = widths[medial]
    nearest = ndi.distance_transform_edt(
        ~medial, sampling=spacing, return_distances=False, return_indices=True
    )
    local = widths[tuple(nearest)]
    return np.where(design, local, 0.0), samples


def _budget_edge(span, budget=MAX_POST_LATTICE_VOXELS):
    """Cubic sampling edge within budget, including nodes and exterior border."""
    span = np.asarray(span, dtype=float)
    edge = float(np.cbrt(np.prod(span) / budget))
    for _ in range(20):
        # Continuous surface recovery samples the interval endpoints plus one
        # exterior node on each side, rather than a cell-centre voxel grid.
        counts = np.ceil(span / edge) + 3
        if np.prod(counts) <= budget:
            return edge
        edge *= max(1.002, float(np.cbrt(np.prod(counts) / budget)))
    return edge


def fit_lattice_dimensions(
    design,
    bounds,
    family,
    *,
    requested_cell=0.0,
    requested_member=0.0,
    requested_skin=0.0,
    surface_quality=DEFAULT_POST_LATTICE_QUALITY,
    max_recovery_voxels=None,
    boundary_adaptation=False,
):
    """Prefer two cells across 90% of the design-only medial samples.

    This optimizes a geometric fitting criterion. It does not optimize
    compliance and does not warp cells to conform to curved boundaries.
    Boundary cells and cells in thin features are trimmed to the envelope;
    design material is never promoted to non-design solid. A zero requested
    skin is literal, for both strut and TPMS families.
    """
    design = np.asarray(design, dtype=bool)
    if design.ndim != 3 or min(design.shape) < 1:
        raise ValueError("Lattice fitting needs a non-empty 3-D design mask.")
    span = np.asarray(bounds[1], dtype=float) - np.asarray(bounds[0], dtype=float)
    if span.shape != (3,) or not np.isfinite(span).all() or np.any(span <= 0.0):
        raise ValueError("Lattice fitting needs finite positive 3-D bounds.")
    lengths = np.asarray([requested_cell, requested_member, requested_skin], dtype=float)
    if not np.isfinite(lengths).all() or np.any(lengths < 0.0):
        raise ValueError("Lattice dimensions must be finite and non-negative.")
    spacing = span / np.asarray(design.shape, dtype=float)
    nominal_edge = float(np.mean(spacing))
    local, samples = design_feature_widths(design, spacing)
    feature = float(np.percentile(samples, 10))
    quality = resolve_lattice_quality(surface_quality)
    budget = POST_LATTICE_QUALITY_BUDGETS[quality] if max_recovery_voxels is None else int(max_recovery_voxels)
    if budget <= 0:
        raise ValueError("The lattice sampling budget must be positive.")
    budget_edge = _budget_edge(span, budget)
    automatic = requested_cell == 0.0
    # Automatic wall follows feature size but must remain resolvable. A stated
    # process minimum is authoritative and is never reduced to obtain a fit.
    if requested_member > 0.0:
        member = float(requested_member)
    elif automatic:
        member = max(feature / 16.0, 3.0 * budget_edge)
    else:
        area_factor = (tpms_specific_surface_area(family.key) / 0.80) if family.is_tpms else 1.0
        if family.is_tpms and family.wall_mode == "skeletal":
            area_factor *= 0.5
        max_member_for_cell = 0.70 * float(requested_cell) / max(area_factor, 1.0)
        member = min(0.5 * nominal_edge, max_member_for_cell)
        member = max(member, 3.0 * budget_edge)

    skin = float(requested_skin)
    resolution_floor = POST_LATTICE_QUALITY_SCALES[quality] * float(family.minimum_cell_voxels) * budget_edge

    # The TPMS renderer caps its wall-derived occupancy at 80%. Do not request
    # a thicker wall than that cap can represent and silently lose the floor.
    # Strut diameter must stay below the pitch to retain a cell opening.
    member_pitch = member
    area = 1.0
    if family.is_tpms:
        area = tpms_specific_surface_area(family.key)
        if family.wall_mode == "skeletal":
            area *= 0.5
        member_pitch = max(member_pitch, area * member / 0.80)
    floor = max(resolution_floor, member_pitch)
    if boundary_adaptation and automatic:
        # Automatic sizing must reserve room for the *smaller* boundary
        # cells. Previously a nominal TPMS cell could sit exactly at the
        # wall/occupancy floor, making every attempted refinement impossible.
        # Enlarge the automatic interior pitch; never thin a stated member.
        # Both scales must also be resolved within the chosen sampling budget.
        if family.kind == "prismatic" and family.base == "honeycomb":
            open_pitch = member / np.sqrt(3.)
        else:
            open_pitch = member_pitch
        boundary_resolution_floor = float(family.minimum_cell_voxels) * budget_edge
        floor = max(floor, 2. * open_pitch * 1.05, 2. * boundary_resolution_floor)
    target = max(0.0, (feature - 2.0 * skin) / CELLS_ACROSS_FEATURE)
    if automatic:
        cell = max(target, floor)
    else:
        cell = float(requested_cell)
    if cell <= member or cell < member_pitch * (1.0 - 1e-6):
        if automatic:
            cell = max(cell, member_pitch * 1.05, member * 1.1)
        else:
            raise ValueError(
                "The requested cell size cannot contain the minimum wall/strut with an open cell. "
                "Increase cell size or reduce the minimum wall/strut thickness; dimensions were not changed."
            )
    adaptation = None
    if boundary_adaptation:
        from ..geometry.lattice_adaptation import plan_lattice_adaptation
        adaptation = plan_lattice_adaptation(cell, member, family)
    _, _, sampling_report = plan_lattice_sampling(span, cell, member, family, quality, budget,
        refined_pitch=None if adaptation is None else adaptation["refined_pitch_mm"])

    # A full cell need only fit somewhere; thinner features may contain valid
    # partial cells. They receive the same lattice field, never solid fallback.
    too_thin = design & (local < cell + 2.0 * skin)
    fallback = np.zeros_like(design, dtype=bool)
    coverage = float(np.mean(samples >= CELLS_ACROSS_FEATURE * cell + 2.0 * skin))
    report = {
        "mode": "Automatic fit" if automatic else "Manual",
        "method": "Design-only medial thickness with printable and resolution bounds",
        "sizing_basis": "Geometry and sampling limits; no lattice strength optimization",
        "load_based_strength_sizing": False,
        "target_cells_across_feature": CELLS_ACROSS_FEATURE,
        "design_feature_thickness_mm": feature,
        "minimum_feasible_cell_mm": floor,
        "target_cell_size_mm": target,
        "selected_cell_size_mm": cell,
        "minimum_member_mm": member,
        "skin_thickness_mm": skin,
        "medial_fit_fraction": coverage,
        "constraint_limited": bool(automatic and cell > target * 1.001),
        "solid_feature_voxels": int(np.count_nonzero(fallback)),
        "solid_feature_fraction": float(
            np.count_nonzero(fallback) / np.count_nonzero(design)
        ),
        "partial_cell_fraction": float(
            np.count_nonzero(too_thin) / np.count_nonzero(design)
        ),
        "manual_unfit_fraction": float(
            np.count_nonzero(too_thin) / np.count_nonzero(design)
        )
        if not automatic
        else 0.0,
        "boundary_behavior": (
            "Cells trimmed to the source envelope, including thin design features"
            if skin == 0
            else "Trimmed cells joined to a boundary skin; no conformal warping"
        ),
        **sampling_report,
        "boundary_adaptation": adaptation,
    }
    return (
        cell / nominal_edge,
        member / nominal_edge,
        skin / nominal_edge,
        fallback,
        report,
    )
