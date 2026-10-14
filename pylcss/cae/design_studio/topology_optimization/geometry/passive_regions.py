# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Passive-region density and mask operations."""

from __future__ import annotations

import logging
from typing import Optional, Sequence

import numpy as np

from .analytic_shapes import (
    BoxRegion,
    CylinderRegion,
    _convert_legacy_to_physical_shapes,
)

logger = logging.getLogger(__name__)


def _apply_passive_density_regions(
    field: np.ndarray,
    *,
    solid_boxes: Sequence[BoxRegion] = (),
    void_boxes: Sequence[BoxRegion] = (),
    solid_cylinders: Sequence[CylinderRegion] = (),
    void_cylinders: Sequence[CylinderRegion] = (),
    joint_pin_cylinders: Sequence[CylinderRegion] = (),
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
        joint_pin_cylinders=joint_pin_cylinders,
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
