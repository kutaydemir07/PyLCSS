# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Boundary-condition and joint hardware mapping for topology nodes."""

from __future__ import annotations

import math
from typing import Any, Optional

import numpy as np

from ..models.study import VoxelBC
from .geometry_mapping import _entry_bboxes, _fraction_box


def _force_components_nonzero(values: Any) -> bool:
    """Recognize finite loads independently of the chosen force units."""
    try:
        vals = [float(v) for v in values]
    except (TypeError, ValueError, OverflowError):
        return False
    return all(math.isfinite(v) for v in vals) and any(v != 0.0 for v in vals)


def _bc_has_nonzero_load(bc: VoxelBC) -> bool:
    for case in (bc, *bc.load_cases):
        if any(
            boundary.cell_forces is not None
            and _force_components_nonzero(np.asarray(boundary.cell_forces).ravel())
            for boundary in case.boundary_forces
        ):
            return True
    for point in bc.point_forces:
        if _force_components_nonzero(point[3:6]):
            return True
    for box in bc.box_forces:
        if _force_components_nonzero(box[6:9]):
            return True
    for distributed in bc.distributed_forces:
        if _force_components_nonzero(distributed[1:4]):
            return True
    for load_case in bc.load_cases:
        for point in load_case.point_forces:
            if _force_components_nonzero(point[3:6]):
                return True
        for box in load_case.box_forces:
            if _force_components_nonzero(box[6:9]):
                return True
        for distributed in load_case.distributed_forces:
            if _force_components_nonzero(distributed[1:4]):
                return True
    return False


def _bc_has_support(bc: VoxelBC) -> bool:
    if any(
        boundary.fixed_dofs
        for case in (bc, *bc.load_cases)
        for boundary in case.boundary_supports
    ):
        return True
    if any(
        (
            bc.fixed_left_face_dofs,
            bc.fixed_right_face_dofs,
            bc.fixed_top_face_dofs,
            bc.fixed_bottom_face_dofs,
            bc.fixed_front_face_dofs,
            bc.fixed_back_face_dofs,
        )
    ):
        return True
    if any(bool(box[-1]) for box in bc.fixed_boxes):
        return True
    return any(
        any(bool(box[-1]) for box in load_case.fixed_boxes)
        for load_case in bc.load_cases
    )


def _bbox_pressure_resultant(
    bbox: tuple[float, float, float, float, float, float],
    bounds: tuple[np.ndarray, np.ndarray],
    pressure: float,
) -> tuple[float, float, float]:
    """Approximate a planar pressure face as one resultant force."""
    mins, maxs = bounds
    extents = np.asarray(
        [
            max(0.0, bbox[1] - bbox[0]),
            max(0.0, bbox[3] - bbox[2]),
            max(0.0, bbox[5] - bbox[4]),
        ],
        dtype=float,
    )
    axis = int(np.argmin(extents))
    area_axes = [idx for idx in range(3) if idx != axis]
    area = float(np.prod(extents[area_axes]))
    center = np.asarray(
        [
            0.5 * (bbox[0] + bbox[1]),
            0.5 * (bbox[2] + bbox[3]),
            0.5 * (bbox[4] + bbox[5]),
        ],
        dtype=float,
    )
    midpoint = 0.5 * (mins[:3] + maxs[:3])
    sign = 1.0 if center[axis] >= midpoint[axis] else -1.0
    force = np.zeros(3, dtype=float)
    force[axis] = float(pressure) * area * sign
    return float(force[0]), float(force[1]), float(force[2])


def _pressure_load_patches(
    load: dict[str, Any],
    bounds: tuple[np.ndarray, np.ndarray],
) -> list[tuple[tuple[float, float, float, float, float, float], float, float, float]]:
    try:
        pressure = float(load.get("pressure", load.get("magnitude", 0.0)))
    except Exception:
        return []
    patches: list[
        tuple[tuple[float, float, float, float, float, float], float, float, float]
    ] = []
    if load.get("geometries") or load.get("geometry") is not None:
        raise ValueError(
            "CAD pressure selections must use their surface geometry, not bounding-box patches."
        )
    if not np.isfinite(pressure):
        raise ValueError("Pressure must be finite.")
    for bbox in dict.fromkeys(_entry_bboxes(load)):
        extents = [bbox[i + 1] - bbox[i] for i in (0, 2, 4)]
        if sum(extent > 1e-9 for extent in extents) != 2:
            raise ValueError(
                "Pressure needs a planar face; a bounding volume does not identify its surface."
            )
        frac_box = _fraction_box(bbox, bounds)
        fx, fy, fz = _bbox_pressure_resultant(bbox, bounds, pressure)
        if _force_components_nonzero((fx, fy, fz)):
            patches.append((frac_box, fx, fy, fz))
    return patches


def _bounds_payload(
    bounds: Optional[tuple[np.ndarray, np.ndarray]],
) -> Optional[dict[str, list[float]]]:
    if bounds is None:
        return None
    mins, maxs = bounds
    return {
        "min": [float(v) for v in mins[:3]],
        "max": [float(v) for v in maxs[:3]],
    }
