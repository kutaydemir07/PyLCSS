# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Passive-region density and signed-distance operations."""

from __future__ import annotations

import logging
from typing import Optional, Sequence

import numpy as np

from .analytic_shapes import _convert_legacy_to_physical_shapes, smax, smin
from .recovery_grid import BoxRegion, CylinderRegion

logger = logging.getLogger(__name__)


def _apply_passive_density_regions(
    field: np.ndarray,
    *,
    solid_boxes: Sequence[BoxRegion] = (),
    void_boxes: Sequence[BoxRegion] = (),
    solid_cylinders: Sequence[CylinderRegion] = (),
    void_cylinders: Sequence[CylinderRegion] = (),
    bounds: Optional[tuple[np.ndarray, np.ndarray]] = None,
    cell: Optional[np.ndarray] = None,
    spacing: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Clamp passive regions on the high-resolution scalar field.

    The optimiser stores circular non-design regions analytically, but the
    density array itself is voxelised. Re-applying the analytic regions here
    prevents circular holes from becoming low-sided polygons during export.
    """
    if field.ndim != 3:
        return field

    if bounds is None:
        shape_arr = np.maximum(np.asarray(field.shape, dtype=float), 1.0)
        mins = -0.5 * shape_arr
        maxs = 0.5 * shape_arr
        bounds = (mins, maxs)

    shapes = _convert_legacy_to_physical_shapes(
        bounds,
        solid_boxes=solid_boxes,
        void_boxes=void_boxes,
        solid_cylinders=solid_cylinders,
        void_cylinders=void_cylinders,
    )
    if not shapes:
        return field

    mins, maxs = bounds
    mins = np.asarray(mins, dtype=float)[:3]
    maxs = np.asarray(maxs, dtype=float)[:3]
    span = np.maximum(maxs - mins, 1e-12)

    nx, ny, nz = field.shape

    if cell is None or spacing is None:
        cell = span / np.maximum(np.asarray(field.shape, dtype=float), 1.0)
        spacing = cell

    x_phys = mins[0] + 0.5 * spacing[0] + np.arange(nx, dtype=float) * spacing[0]
    y_phys = mins[1] + 0.5 * spacing[1] + np.arange(ny, dtype=float) * spacing[1]
    z_phys = mins[2] + 0.5 * spacing[2] + np.arange(nz, dtype=float) * spacing[2]

    # Create coordinate grid in physical space
    X, Y, Z = np.meshgrid(x_phys, y_phys, z_phys, indexing="ij")
    pts_grid = np.stack([X, Y, Z], axis=-1)

    solid_shapes = [shape for shape in shapes if shape.is_solid]
    void_shapes = [shape for shape in shapes if not shape.is_solid]

    for shape in solid_shapes:
        shape_sdf = shape.sdf(pts_grid)
        field[shape_sdf <= 0.0] = 1.0
    for shape in void_shapes:
        shape_sdf = shape.sdf(pts_grid)
        field[shape_sdf <= 0.0] = 0.0

    return field


def _apply_passive_cylinder_sdf(
    signed_distance: np.ndarray,
    *,
    pad: int,
    solid_boxes: Sequence[BoxRegion] = (),
    void_boxes: Sequence[BoxRegion] = (),
    solid_cylinders: Sequence[CylinderRegion] = (),
    void_cylinders: Sequence[CylinderRegion] = (),
    bounds: Optional[tuple[np.ndarray, np.ndarray]] = None,
    cell: Optional[np.ndarray] = None,
    spacing: Optional[np.ndarray] = None,
    blend_radius: float = 0.0,
) -> np.ndarray:
    """Re-impose analytic shape boundaries on the SDF.

    `signed_distance` uses negative values for material and positive values for
    void. A solid cylinder/shape is therefore a min operation; a through-hole/void
    is a max operation. Applying this after SDF smoothing keeps holes round instead of
    preserving the voxel-grid polygon.
    """
    if signed_distance.ndim != 3:
        return signed_distance

    if bounds is None:
        shape_arr = np.maximum(
            np.asarray(signed_distance.shape, dtype=float) - 2.0 * float(pad), 1.0
        )
        mins = -0.5 * shape_arr
        maxs = 0.5 * shape_arr
        bounds = (mins, maxs)

    shapes = _convert_legacy_to_physical_shapes(
        bounds,
        solid_boxes=solid_boxes,
        void_boxes=void_boxes,
        solid_cylinders=solid_cylinders,
        void_cylinders=void_cylinders,
    )
    if not shapes:
        return signed_distance

    mins, maxs = bounds
    mins = np.asarray(mins, dtype=float)[:3]
    maxs = np.asarray(maxs, dtype=float)[:3]
    span = np.maximum(maxs - mins, 1e-12)

    nx, ny, nz = signed_distance.shape

    if cell is None or spacing is None:
        shape_unpadded = np.maximum(
            np.asarray(signed_distance.shape, dtype=float) - 2.0 * float(pad), 1.0
        )
        cell = span / shape_unpadded
        spacing = cell

    x_phys = (
        mins[0]
        + 0.5 * spacing[0]
        + (np.arange(nx, dtype=float) - float(pad)) * spacing[0]
    )
    y_phys = (
        mins[1]
        + 0.5 * spacing[1]
        + (np.arange(ny, dtype=float) - float(pad)) * spacing[1]
    )
    z_phys = (
        mins[2]
        + 0.5 * spacing[2]
        + (np.arange(nz, dtype=float) - float(pad)) * spacing[2]
    )

    # Create coordinate grid in physical space
    X, Y, Z = np.meshgrid(x_phys, y_phys, z_phys, indexing="ij")
    pts_grid = np.stack([X, Y, Z], axis=-1)

    # Determine if signed_distance is in physical units or density units
    is_physical = (np.max(signed_distance) > 2.0) or (np.min(signed_distance) < -2.0)
    grid_spacing = 1.0
    if not is_physical:
        # Scale physical SDF to match unitless density field
        # A grid voxel has spacing. We can use the mean spacing as the scale factor.
        grid_spacing = np.mean(spacing)

    k = blend_radius if is_physical else (blend_radius / grid_spacing)

    solid_shapes = [shape for shape in shapes if shape.is_solid]
    void_shapes = [shape for shape in shapes if not shape.is_solid]

    for shape in solid_shapes:
        shape_sdf = shape.sdf(pts_grid)
        if not is_physical:
            shape_sdf = shape_sdf / grid_spacing

        if k > 0.0:
            signed_distance = smin(signed_distance, shape_sdf, k)
        else:
            np.minimum(signed_distance, shape_sdf, out=signed_distance)

    for shape in void_shapes:
        shape_sdf = shape.sdf(pts_grid)
        if not is_physical:
            shape_sdf = shape_sdf / grid_spacing

        if k > 0.0:
            signed_distance = smax(signed_distance, -shape_sdf, k)
        else:
            np.maximum(signed_distance, -shape_sdf, out=signed_distance)

    return signed_distance


def _resample_source_mask(
    source_mask: Optional[np.ndarray],
    target_shape: tuple[int, int, int],
) -> Optional[np.ndarray]:
    if source_mask is None:
        return None
    source = np.asarray(source_mask, dtype=bool)
    if source.ndim != 3 or min(source.shape) < 1:
        return None
    if tuple(source.shape) == tuple(target_shape):
        return source
    try:
        import scipy.ndimage as ndi

        zoom = tuple(
            float(target) / float(source_size)
            for target, source_size in zip(
                target_shape,
                source.shape,
                strict=True,
            )
        )
        resized = ndi.zoom(
            source.astype(float), zoom=zoom, order=0, mode="nearest", grid_mode=True
        )
        out = resized >= 0.5
        if out.shape != tuple(target_shape):
            cropped = np.zeros(target_shape, dtype=bool)
            common = tuple(
                slice(0, min(target, actual))
                for target, actual in zip(
                    cropped.shape,
                    out.shape,
                    strict=True,
                )
            )
            cropped[common] = out[common]
            return cropped
        return out
    except Exception:
        logger.debug("Failed to resample topology source mask", exc_info=True)
        return None
