# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Structured-grid coordinates and recovery region types."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

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
    """Force a density field to be constant along an extrusion axis."""
    axis_map = {"x": 0, "y": 1, "z": 2}
    ax = axis_map.get(str(axis or "").strip().lower())
    if ax is None or grid.ndim != 3:
        return grid
    profile = np.mean(grid, axis=ax, keepdims=True)
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
