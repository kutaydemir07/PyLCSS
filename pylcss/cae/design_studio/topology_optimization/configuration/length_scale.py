# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Physical length-scale policy for density-based topology optimization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


MIN_FEATURE_ELEMENTS = 3.0
PROGRAM_CONTROLLED_FEATURE_ELEMENTS = 3.0

# Maximum offset for the cone-filter size relation used below.
MAX_THRESHOLD_OFFSET = 0.25


@dataclass(frozen=True)
class LengthScale:
    """Physical cone-filter radius, feature sizes, and robust projection thresholds."""

    filter_radius: float
    minimum_solid_size: float
    minimum_void_size: float
    eta_eroded: float = 0.5 + MAX_THRESHOLD_OFFSET
    eta_dilated: float = 0.5 - MAX_THRESHOLD_OFFSET


def extrusion_inactive_axes(extrusion: object) -> tuple[int, ...]:
    """Exclude the axis along which an extruded density field is constant."""
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
    """Resolve physical sizes and robust thresholds for a cone density filter.

    Explicit sizes stay fixed across mesh refinements and must span at least
    three cells on the coarsest active axis. An unspecified solid size uses
    three cells; an unspecified void size uses the resolved solid size.

    The nominal-feature planning relation used here is
    d = 2 * R * sqrt(threshold_offset).
    Choose R from the larger request at MAX_THRESHOLD_OFFSET, then derive
    each threshold from its own request. Equal requests give (0.75, 0.25).
    These settings drive the solver's robust projection. They state a target
    length scale; this helper does not measure the recovered 3-D feature sizes.
    """
    edges = voxel_edge_lengths(bounds, shape)
    limiting_edge = coarsest_active_edge(edges, inactive_axes)
    solid = float(minimum_solid_size or 0.0)
    void = float(minimum_void_size or 0.0)
    if not np.isfinite([solid, void]).all() or solid < 0 or void < 0:
        raise ValueError("Minimum solid and void sizes must be finite and non-negative.")
    if solid <= 0.0:
        solid = PROGRAM_CONTROLLED_FEATURE_ELEMENTS * limiting_edge
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
