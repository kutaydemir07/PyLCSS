# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Physical length-scale policy for density-based topology optimization."""

from __future__ import annotations

from dataclasses import dataclass

from typing import Sequence

import numpy as np


MIN_FEATURE_ELEMENTS = 3.0
RECOMMENDED_FEATURE_ELEMENTS = 4.0
MAX_FEATURE_ELEMENTS = 12.0
PROGRAM_CONTROLLED_FEATURE_ELEMENTS = 3.0

#: Largest threshold offset from 0.5 used by the robust formulation. The
#: closed-form length-scale relation below is derived assuming the filter
#: kernel fully spans the critical member, which stops holding as the eroded
#: threshold approaches 1. Checked numerically against a 1-D cone filter at
#: radii of 40, 200 and 800 cells: at eta_e = 0.75 the predicted member width
#: is within 0.12% of the measured one and still converging, at 0.60 within
#: 0.20%, while at 0.90 the error stalls at 8.2% and does not improve with
#: refinement -- the signature of the assumption failing rather than of
#: discretization. 0.25 is also the standard value in the literature, so the
#: validated regime and the conventional one coincide.
MAX_THRESHOLD_OFFSET = 0.25


@dataclass(frozen=True)
class LengthScale:
    """Cone-filter radius, requested feature sizes, and robust thresholds.

    ``eta_eroded`` and ``eta_dilated`` are the projection thresholds that make
    the requested sizes an actual constraint on the design rather than a label
    on it -- see :func:`resolve_physical_length_scale`.
    """

    filter_radius: float
    minimum_solid_size: float
    minimum_void_size: float
    eta_eroded: float = 0.5 + MAX_THRESHOLD_OFFSET
    eta_dilated: float = 0.5 - MAX_THRESHOLD_OFFSET


def extrusion_inactive_axes(extrusion: object) -> tuple[int, ...]:
    """Axes the density filter is not required to regularize.

    An extruded study pins the density to a constant along the extrusion axis,
    so the filter does no work in that direction and the number of layers the
    grid happens to use there says nothing about whether the design is
    regularized. Every check on "the coarsest voxel axis" has to agree on this
    or a study is accepted by one and rejected by the other: the suspension
    rocker resolved its 8 mm member against a 2.63 mm in-plane cell (3.04
    cells, accepted) and was then rejected by the solver for spanning 0.64 of
    the 6.25 mm extrusion-layer cell, which is not a direction its filter was
    ever going to act in.
    """
    return {
        "x": (0,),
        "y": (1,),
        "z": (2,),
    }.get(str(extrusion or "none").strip().lower(), ())


def coarsest_active_edge(
    voxel_edges: Sequence[float],
    inactive_axes: Sequence[int] = (),
) -> float:
    """Largest voxel edge among the axes the filter has to regularize."""
    edges = np.asarray(tuple(voxel_edges)[:3], dtype=float)
    inactive = {
        int(axis) for axis in inactive_axes if 0 <= int(axis) < edges.size
    }
    active = np.asarray(
        [edge for axis, edge in enumerate(edges) if axis not in inactive],
        dtype=float,
    )
    if active.size == 0:
        active = edges
    return float(np.max(active))


#: A design is reported as under its requested member size only when more than
#: this share of its material fails the probe. Opening rounds every free edge,
#: so even a plate comfortably thicker than the probe reads a few percent:
#: measured on slabs of 6 to 12 voxels against a 5-voxel probe, the residue ran
#: 2.9% to 5.5%. Real results separate far more cleanly than that band -- a
#: design holding its length scale measures 0.0%, and one that is not measures
#: 26% and up -- so 10% sits in the empty middle rather than on either
#: population.
THIN_MATERIAL_REPORTING_FRACTION = 0.10


def _probe_ball(voxel_counts: np.ndarray) -> np.ndarray:
    """Discrete ellipsoid spanning exactly ``voxel_counts`` voxels per axis.

    Sized in odd voxel counts rather than in a continuous radius so the element
    spans what it claims to. Sizing it by radius and letting ``ceil`` pick the
    array bounds produces an array one voxel wider than the ellipsoid inside
    it, and the size actually probed is then not the size that was asked for.
    """
    radii = (np.asarray(voxel_counts, dtype=float) - 1.0) / 2.0
    grids = np.meshgrid(
        *[np.arange(-int(radius), int(radius) + 1) for radius in radii],
        indexing="ij",
    )
    normalized = sum(
        (grid / max(float(radius), 1e-9)) ** 2
        for grid, radius in zip(grids, radii)
    )
    return normalized <= 1.0 + 1e-9


def thin_material_fraction(
    material_mask: np.ndarray,
    member_size: float,
    voxel_edges: Sequence[float],
) -> dict[str, float] | None:
    """Share of the material a ball of ``member_size`` diameter cannot reach.

    The morphological definition of minimum feature size: open the design with
    a ball of the requested diameter and see what does not survive. Material
    the ball cannot roll into is, by definition, in a member thinner than the
    ball -- which is the same statement the erosion in a robust formulation
    makes, measured after the fact instead of imposed during the solve.

    This replaced a ray-cast wall-thickness percentile for the comparison
    against the requested size, because that statistic does not respond to the
    quantity in question. Measured across filter radii from half the requested
    member to the full member, its 5th percentile stayed between 0.25 and 0.55
    of the request and its minimum between 0.01 and 0.32 -- no threshold on it
    separates a design that holds its length scale from one that does not,
    because a ray cast from a face centre near a member end or a fillet reads
    the local surface, not the member. The ray sample remains in the surface
    report, where a surface statistic is what is wanted.

    Returns ``fraction`` and the ``probe_size`` it was actually measured with,
    or ``None`` when the probe is smaller than the grid can represent.

    ``probe_size`` is reported because it is generally not ``member_size``. A
    symmetric discrete ball only has odd diameters in voxels, so a 6 mm request
    on a 1 mm grid can be probed at 5 mm or at 7 mm and not at 6. The smaller
    is chosen: probing above the request would report every design that exactly
    meets its size as failing, which is the useless direction to be wrong in.
    The cost is a blind spot of up to one voxel, and naming the size actually
    probed is what lets the reader see it -- a probe well below the request is
    itself the signal that the grid is too coarse to check the requirement.

    The reading also always carries a small residue from the rounding of free
    edges, so compare ``fraction`` against
    :data:`THIN_MATERIAL_REPORTING_FRACTION` rather than against zero.
    """
    mask = np.asarray(material_mask, dtype=bool)
    if mask.ndim != 3 or not np.any(mask):
        return None
    edges = np.asarray(tuple(voxel_edges)[:3], dtype=float)
    if edges.shape != (3,) or not np.all(np.isfinite(edges)) or np.any(edges <= 0.0):
        return None
    size = float(member_size)
    if not np.isfinite(size) or size <= 0.0:
        return None

    # The largest odd voxel count that does not exceed the request, per axis.
    # A symmetric discrete ball only has odd diameters, so an even request is
    # not representable and one of the two neighbours has to be chosen; the
    # smaller one is, because probing above the request reports a design that
    # exactly meets its size as failing. `probe_size` carries the consequence
    # back to the caller.
    counts = np.floor(size / edges).astype(int)
    counts = counts - (counts + 1) % 2
    if int(np.min(counts)) < 3:
        return None

    from scipy import ndimage as ndi

    element = _probe_ball(counts)
    probe_size = float(np.min(counts.astype(float) * edges))
    opened = ndi.binary_dilation(ndi.binary_erosion(mask, element), element)
    total = int(np.count_nonzero(mask))
    if total <= 0:
        return None
    fraction = float(total - int(np.count_nonzero(opened & mask))) / float(total)
    return {"fraction": fraction, "probe_size": probe_size}


def voxel_edge_lengths(
    bounds: tuple[np.ndarray, np.ndarray],
    shape: Sequence[int],
) -> np.ndarray:
    """Return the three physical voxel edge lengths."""
    mins = np.asarray(bounds[0], dtype=float)[:3]
    maxs = np.asarray(bounds[1], dtype=float)[:3]
    dims = np.maximum(np.asarray(tuple(shape)[:3], dtype=float), 1.0)
    return np.maximum(maxs - mins, 1e-12) / dims


def resolve_physical_length_scale(
    bounds: tuple[np.ndarray, np.ndarray],
    shape: Sequence[int],
    minimum_solid_size: float | None,
    minimum_void_size: float | None = None,
    *,
    inactive_axes: Sequence[int] = (),
) -> LengthScale:
    """Resolve requested or program-controlled sizes without changing units.

    Explicit sizes remain fixed in model units across mesh refinements.  A
    request below three cells is rejected because silently enlarging it changes
    the engineering requirement. Program-controlled studies use three cells on
    the coarsest voxel direction and report that resolved physical value.

    The returned radius and thresholds are what make the requested sizes bind.
    A density filter with a single projection at eta = 0.5 imposes no minimum
    length scale at all -- it blurs the field, and the projection then sharpens
    whatever survived, including members far thinner than the filter radius.
    The robust formulation (Wang, Lazarov & Sigmund 2011) is what turns the
    request into a constraint: the same filtered field is projected at three
    thresholds, and optimizing the *eroded* one makes a too-thin member vanish
    from the design being analysed, which the optimizer sees as lost stiffness.

    Sizing the thresholds. For a cone filter of radius ``R``, a solid bar of
    half-width ``b`` filters to ``rho = 2t - t^2`` at its centre and
    ``2t - t^2 - s^2`` at offset ``s*R``, with ``t = b/R`` (both in units of
    ``R``). The thinnest bar that survives erosion at ``eta_e`` is the one with
    ``2t - t^2 = eta_e``; its width in the blueprint is where that profile
    crosses 0.5, giving a minimum solid member size of::

        d_solid = 2 * R * sqrt(eta_e - 0.5)

    and, by the symmetric argument on a void gap closing in the dilated design,
    ``d_void = 2 * R * sqrt(0.5 - eta_d)``. Verified numerically against a 1-D
    cone filter to within 0.12% at the thresholds used here.

    Inverting those at the largest validated offset puts the radius at the
    larger of the two requested sizes and keeps both thresholds inside
    ``MAX_THRESHOLD_OFFSET``. Note this is twice the ``0.5 * solid`` radius
    this function used to return: under the relation above that radius delivers
    half the requested member, which is the arithmetic behind a minimum member
    size that measurably did not hold.

    The relation is one-dimensional, so it is exact for a wall or plate and
    conservative for a strut, whose two-sided curvature makes the filtered peak
    fall off faster -- a round strut lands slightly above the requested size
    rather than below it. Erring high is the correct direction for a
    manufacturing minimum.
    """
    edges = voxel_edge_lengths(bounds, shape)
    limiting_edge = coarsest_active_edge(edges, inactive_axes)
    solid = float(minimum_solid_size or 0.0)
    if solid <= 0.0:
        solid = PROGRAM_CONTROLLED_FEATURE_ELEMENTS * limiting_edge
    void = float(minimum_void_size or 0.0)
    if void <= 0.0:
        void = solid

    for label, value in (("solid", solid), ("void", void)):
        resolved_elements = value / limiting_edge
        if resolved_elements + 1e-9 < MIN_FEATURE_ELEMENTS:
            raise ValueError(
                f"Minimum {label} size {value:.6g} spans only "
                f"{resolved_elements:.2f} cells on the coarsest voxel axis; "
                f"at least {MIN_FEATURE_ELEMENTS:.0f} are required. Increase "
                "the quality/resolution or request a larger physical feature."
            )

    # Radius from the binding request at the largest validated offset, then
    # each threshold from its own size. Equal solid and void requests give the
    # textbook (0.75, 0.25) pair; an asymmetric request moves only the
    # threshold belonging to the smaller size, toward 0.5, which is the
    # direction that relaxes it.
    radius = max(solid, void) / (2.0 * np.sqrt(MAX_THRESHOLD_OFFSET))
    eta_eroded = 0.5 + (solid / (2.0 * radius)) ** 2
    eta_dilated = 0.5 - (void / (2.0 * radius)) ** 2
    return LengthScale(
        filter_radius=float(radius),
        minimum_solid_size=solid,
        minimum_void_size=void,
        eta_eroded=float(np.clip(eta_eroded, 0.5 + 1e-6, 0.5 + MAX_THRESHOLD_OFFSET)),
        eta_dilated=float(np.clip(eta_dilated, 0.5 - MAX_THRESHOLD_OFFSET, 0.5 - 1e-6)),
    )
