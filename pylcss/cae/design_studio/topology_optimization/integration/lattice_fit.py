# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE.
"""Choose a printable periodic pitch from design-only local feature sizes."""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import skeletonize

from ..manufacturing.lattice_families import tpms_specific_surface_area

MAX_POST_LATTICE_VOXELS = 8_000_000
CELLS_ACROSS_FEATURE = 2.0


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
    budget_edge = _budget_edge(span)
    automatic = requested_cell == 0.0
    # Automatic wall follows feature size but must remain resolvable. A stated
    # process minimum is authoritative and is never reduced to obtain a fit.
    member = (
        float(requested_member)
        if requested_member > 0.0
        else (
            max(feature / 16.0, 3.0 * budget_edge) if automatic else 0.5 * nominal_edge
        )
    )
    skin = float(requested_skin)
    resolution_floor = 2.0 * float(family.minimum_cell_voxels) * budget_edge
    # The TPMS renderer caps its wall-derived occupancy at 80%. Do not request
    # a thicker wall than that cap can represent and silently lose the floor.
    # Strut diameter must stay below the pitch to retain a cell opening.
    member_pitch = member
    if family.is_tpms:
        area = tpms_specific_surface_area(family.key)
        if family.wall_mode == "skeletal":
            area *= 0.5
        member_pitch = max(member_pitch, area * member / 0.80)
    floor = max(resolution_floor, member_pitch)
    target = max(0.0, (feature - 2.0 * skin) / CELLS_ACROSS_FEATURE)
    cell = max(target, floor) if automatic else float(requested_cell)
    if (
        member < 3.0 * budget_edge * (1.0 - 1e-6)
        or cell < resolution_floor * (1.0 - 1e-6)
    ):
        raise ValueError(
            "The requested lattice cannot resolve its cell and minimum wall/strut "
            "within the build-grid budget. Increase cell size or minimum "
            "thickness. Design regions will not be replaced with solid material."
        )
    if member >= cell or cell < member_pitch * (1.0 - 1e-6):
        raise ValueError(
            "The minimum wall/strut thickness exceeds the cell's supported "
            "thickness or leaves no lattice opening. "
            "Increase cell size or reduce the minimum thickness."
        )
    # A full cell need only fit somewhere; thinner features may contain valid
    # partial cells. They receive the same lattice field, never solid fallback.
    too_thin = design & (local < cell + 2.0 * skin)
    fallback = np.zeros_like(design, dtype=bool)
    if not np.any(design & ~too_thin):
        raise ValueError(
            "No printable lattice cell fits the design material at the available "
            "build resolution. Reduce the minimum wall/strut or skin thickness, "
            "or choose a family requiring fewer cell samples. Design regions "
            "will not be replaced with solid material."
        )
    coverage = float(np.mean(samples >= CELLS_ACROSS_FEATURE * cell + 2.0 * skin))
    report = {
        "mode": "Automatic fit" if automatic else "Manual",
        "method": "Design-only medial thickness with printable and resolution bounds",
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
        "max_build_voxels": MAX_POST_LATTICE_VOXELS,
    }
    return (
        cell / nominal_edge,
        member / nominal_edge,
        skin / nominal_edge,
        fallback,
        report,
    )
