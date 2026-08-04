# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Structured-grid coordinates and recovery region types."""

from __future__ import annotations

import logging
from typing import Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

CylinderRegion = tuple[str, float, float, float, float, float]
BoxRegion = tuple[float, float, float, float, float, float]


def _split_cylinder_region(
    cylinder: Sequence[object] | None,
) -> Optional[tuple[str, float, float, float, float, float, float]]:
    if cylinder is None or len(cylinder) < 6:
        return None
    axis, c0, c1, lo, hi, radius_a = cylinder[:6]
    radius_b = cylinder[6] if len(cylinder) > 6 else radius_a
    radius_a = float(radius_a)
    radius_b = float(radius_b)
    if radius_a <= 0.0 or radius_b <= 0.0:
        return None
    return (
        str(axis or "z").lower(),
        float(c0),
        float(c1),
        float(lo),
        float(hi),
        radius_a,
        radius_b,
    )


def _voxel_origin_cell(
    shape: tuple[int, int, int],
    bounds: Optional[tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    """Return physical origin and cell size for a structured voxel grid."""
    shape_arr = np.maximum(np.asarray(shape, dtype=float), 1.0)
    if bounds is not None:
        mins, maxs = bounds
        mins = np.asarray(mins, dtype=float)
        maxs = np.asarray(maxs, dtype=float)
        if mins.size >= 3 and maxs.size >= 3 and np.all(maxs[:3] > mins[:3]):
            return mins[:3], (maxs[:3] - mins[:3]) / shape_arr
    return -0.5 * shape_arr, np.ones(3, dtype=float)


def _regularize_extruded_density(grid: np.ndarray, axis: str) -> np.ndarray:
    """Force a density field to be constant along an extrusion axis.

    The solver already applies the extrusion map during optimization (see
    ``pymoto_modules._apply_extrusion``), so on the intended path the field is
    constant along ``axis`` and this average is a no-op to floating-point.

    It is not a no-op when the caller asks for an extrusion the field does not
    have -- the viewer's rebuild-from-current-density path can do this, and so
    can a source body thinner than its own design domain. Averaging then does
    not enforce a constraint, it invents a different part: a plate filling 4 of
    6 domain units averages to a uniform 0.667 over its footprint, the caller's
    count-matched level recalibration lands exactly *on* that plateau, and the
    result is a full-thickness slab. Measured on a two-hole plate that is
    +45% volume, and because the iso-field is identically zero through the whole
    slab the extraction degenerates: 3 components, not watertight, 965k
    triangles against 48k, 53 s against 4 s.

    Averaging is nevertheless kept unconditional, because skipping it is the
    more dangerous failure. The extrusion map runs on the design variable,
    *before* the passive clamp and the density filter, so a passive pad or bore
    that does not span the full thickness leaves a small genuine variation along
    the axis in an otherwise correctly prismatic result. Any threshold tight
    enough to catch the slab case above also trips on those, and silently
    dropping the constraint turns a prismatic part into a non-prismatic one --
    a much worse and much more common outcome than the edge case it guards.

    So: always average, and report the variation that was averaged away so a
    genuinely non-extruded input is visible in the log rather than silent.
    """
    axis_map = {"x": 0, "y": 1, "z": 2}
    ax = axis_map.get(str(axis or "").strip().lower())
    if ax is None or grid.ndim != 3:
        return grid
    profile = np.mean(grid, axis=ax, keepdims=True)
    # A constant cross-section deviates from its own mean only by round-off.
    # Large values mean the field was never extruded and the averaged result
    # will not describe the same part.
    deviation = float(np.max(np.abs(grid - profile))) if grid.size else 0.0
    if deviation > 0.25:
        logger.warning(
            "Extrusion axis %r was requested, but the density varies by %.3f "
            "along it (a constant cross-section varies by ~0), so this field "
            "was not produced by an extrusion-constrained solve. Enforcing the "
            "constant cross-section will change the part: material that fills "
            "only part of the domain thickness becomes a full-thickness slab.",
            str(axis).lower(),
            deviation,
        )
    return np.broadcast_to(profile, grid.shape).copy()


def _project_extruded_planes(
    vertices: np.ndarray,
    bounds: Optional[tuple[np.ndarray, np.ndarray]],
    axis: str,
    tolerance: float,
) -> np.ndarray:
    """Snap top/bottom vertices of an extruded result back onto exact planes."""
    axis_map = {"x": 0, "y": 1, "z": 2}
    ax = axis_map.get(str(axis or "").strip().lower())
    if ax is None or bounds is None or len(vertices) == 0:
        return vertices

    mins, maxs = bounds
    lo = float(np.asarray(mins, dtype=float)[ax])
    hi = float(np.asarray(maxs, dtype=float)[ax])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return vertices

    out = np.asarray(vertices, dtype=float).copy()
    tol = max(float(tolerance), 1e-9)
    out[out[:, ax] <= lo + tol, ax] = lo
    out[out[:, ax] >= hi - tol, ax] = hi
    return out
