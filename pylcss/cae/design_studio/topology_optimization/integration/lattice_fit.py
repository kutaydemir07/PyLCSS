# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE.
"""Choose a printable periodic pitch from design-only local feature sizes."""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import skeletonize

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
    """Finest cubic grid within the voxel budget, including integer rounding."""
    span = np.asarray(span, dtype=float)
    edge = float(np.cbrt(np.prod(span) / budget))
    for _ in range(20):
        counts = np.maximum(1, np.ceil(span / edge))
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
    """Largest feasible uniform pitch for two cells across 90% of medial samples.

    This optimizes a geometric fitting criterion. It does not optimize
    compliance and does not warp cells to conform to curved boundaries.
    Small features below the printable pitch are kept solid in automatic mode.
    """
    span = np.asarray(bounds[1], dtype=float) - np.asarray(bounds[0], dtype=float)
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
    minimum_pitch = 2.0 * float(family.minimum_cell_voxels) * budget_edge
    skin_floor = float(requested_skin) if requested_skin > 0 else member
    target = max(0.0, (feature - 2.0 * skin_floor) / CELLS_ACROSS_FEATURE)
    if requested_skin <= 0:
        target = min(target, feature / (CELLS_ACROSS_FEATURE + 2.0 / 40.0))
    cell = max(target, minimum_pitch) if automatic else float(requested_cell)
    skin = (
        float(requested_skin)
        if requested_skin > 0
        else (0.0 if family.is_strut else max(member, cell / 40.0))
    )
    floor = minimum_pitch
    if automatic:
        cell = max(cell, floor)
    # A complete cell plus its surrounding skin needs this much local width.
    too_thin = design & (local < cell + 2.0 * skin)
    fallback = too_thin if automatic else np.zeros_like(design, dtype=bool)
    if automatic and not np.any(design & ~fallback):
        raise ValueError(
            "No printable lattice cell fits the design material. Reduce the "
            "minimum wall/strut or skin thickness, or keep this feature solid."
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
        "manual_unfit_fraction": float(
            np.count_nonzero(too_thin) / np.count_nonzero(design)
        )
        if not automatic
        else 0.0,
        "boundary_behavior": (
            "Trimmed beam members"
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
