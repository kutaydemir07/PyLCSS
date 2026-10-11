# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Boundary-condition and joint hardware mapping for topology nodes."""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np

from ..models.study import VoxelBC
from .geometry_mapping import _bbox_tuple, _entry_bboxes, _fraction_box

logger = logging.getLogger(__name__)


def _force_components_nonzero(values: Any) -> bool:
    try:
        vals = [float(v) for v in values]
    except Exception:
        return False
    return sum(v * v for v in vals) > 1e-24


def _bc_has_nonzero_load(bc: VoxelBC) -> bool:
    for case in (bc, *bc.load_cases):
        if any(np.any(np.asarray(boundary.cell_forces) != 0) for boundary in case.boundary_forces):
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
    if any(boundary.fixed_dofs for case in (bc, *bc.load_cases) for boundary in case.boundary_supports):
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


def _point_xyz(point: Any) -> Optional[np.ndarray]:
    try:
        return np.asarray([float(point.x), float(point.y), float(point.z)])
    except Exception:
        pass
    try:
        values = list(point)
        if len(values) >= 3:
            return np.asarray([float(values[0]), float(values[1]), float(values[2])])
    except Exception:
        pass
    return None


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
        raise ValueError("CAD pressure selections must use their surface geometry, not bounding-box patches.")
    if not np.isfinite(pressure):
        raise ValueError("Pressure must be finite.")
    for bbox in dict.fromkeys(_entry_bboxes(load)):
        extents = [bbox[i + 1] - bbox[i] for i in (0, 2, 4)]
        if sum(extent > 1e-9 for extent in extents) != 2:
            raise ValueError("Pressure needs a planar face; a bounding volume does not identify its surface.")
        frac_box = _fraction_box(bbox, bounds)
        fx, fy, fz = _bbox_pressure_resultant(bbox, bounds, pressure)
        if (fx * fx + fy * fy + fz * fz) > 1e-24:
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


def _bc_feature_bboxes(
    constraints: list[Any],
    loads: list[Any],
) -> list[tuple[float, float, float, float, float, float]]:
    bboxes: list[tuple[float, float, float, float, float, float]] = []
    for entry in list(constraints or []) + list(loads or []):
        if isinstance(entry, dict):
            bboxes.extend(_entry_bboxes(entry))
    return bboxes


def _cylindrical_face_is_bore(face: Any, surface: Any) -> Optional[bool]:
    """Is the material outside this cylindrical face (a bore) or inside it?

    A cylinder alone does not say. The surface normal does: it points away from
    the material, so on an internal bore it points inward toward the axis and on
    an external boss or shaft it points outward. Measured on a plain disc and a
    plain through-hole the two cases separate at -1.0 and +1.0, so a sign test
    is enough.

    Returns ``None`` when the face will not answer, in which case the caller
    keeps the historical assumption.
    """
    try:
        axis = surface.Axis()
        location = axis.Location()
        heading = axis.Direction()
        origin = np.asarray(
            [location.X(), location.Y(), location.Z()], dtype=float
        )
        direction = np.asarray(
            [heading.X(), heading.Y(), heading.Z()], dtype=float
        )
        vertices = face.Vertices()
        if not vertices:
            return None
        import cadquery as cq

        point = np.asarray(vertices[0].toTuple(), dtype=float)
        normal = face.normalAt(cq.Vector(*point.tolist()))
        normal_vector = np.asarray([normal.x, normal.y, normal.z], dtype=float)
    except Exception:
        return None

    norm = float(np.linalg.norm(direction))
    if norm <= 1e-12:
        return None
    direction = direction / norm
    relative = point - origin
    radial = relative - float(np.dot(relative, direction)) * direction
    radial_norm = float(np.linalg.norm(radial))
    if radial_norm <= 1e-9:
        return None
    alignment = float(np.dot(normal_vector, radial / radial_norm))
    if abs(alignment) < 0.25:
        # Nearly tangential: not a clean answer, so do not act on it.
        return None
    return alignment < 0.0


def _cylinder_void_region_from_face(
    face: Any,
    bounds: Optional[tuple[np.ndarray, np.ndarray]],
) -> Optional[tuple[Any, ...]]:
    if face is None or bounds is None:
        return None
    try:
        if str(face.geomType()).upper() != "CYLINDER":
            return None
    except Exception:
        return None
    try:
        surface = face._geomAdaptor()
        radius = float(surface.Radius())
    except Exception:
        return None
    if radius <= 0.0:
        return None
    # Only an internal bore becomes a void region. This used to treat every
    # cylindrical selection as a hole, so a load or restraint named on the
    # outside of a boss or a shaft hollowed the part out from the inside: on
    # the rocker benchmark, selecting the 60 mm boss walls voided both bosses
    # and left the arms hanging, which the solver then reported as a
    # disconnected load path and a compliance six orders of magnitude too high.
    if _cylindrical_face_is_bore(face, surface) is False:
        return None

    try:
        axis_dir = surface.Axis().Direction()
        direction = np.asarray(
            [float(axis_dir.X()), float(axis_dir.Y()), float(axis_dir.Z())],
            dtype=float,
        )
    except Exception:
        direction = np.zeros(3, dtype=float)
    try:
        bbox = _bbox_tuple(face.BoundingBox())
    except Exception:
        bbox = None
    if bbox is None:
        return None

    mins, maxs = bounds
    mins = np.asarray(mins[:3], dtype=float)
    maxs = np.asarray(maxs[:3], dtype=float)
    span = np.maximum(maxs - mins, 1e-12)
    ext = np.asarray(
        [bbox[1] - bbox[0], bbox[3] - bbox[2], bbox[5] - bbox[4]], dtype=float
    )
    axis_idx = (
        int(np.argmax(np.abs(direction))) if np.any(direction) else int(np.argmax(ext))
    )
    radial_axes = [idx for idx in range(3) if idx != axis_idx]
    axis_names = ["x", "y", "z"]

    try:
        center = _point_xyz(face.Center())
    except Exception:
        center = None
    if center is None:
        center = np.asarray(
            [
                0.5 * (bbox[0] + bbox[1]),
                0.5 * (bbox[2] + bbox[3]),
                0.5 * (bbox[4] + bbox[5]),
            ],
            dtype=float,
        )

    lo = (float(bbox[axis_idx * 2]) - mins[axis_idx]) / span[axis_idx]
    hi = (float(bbox[axis_idx * 2 + 1]) - mins[axis_idx]) / span[axis_idx]
    c0 = (float(center[radial_axes[0]]) - mins[radial_axes[0]]) / span[radial_axes[0]]
    c1 = (float(center[radial_axes[1]]) - mins[radial_axes[1]]) / span[radial_axes[1]]
    r0 = radius / span[radial_axes[0]]
    r1 = radius / span[radial_axes[1]]
    return (
        axis_names[axis_idx],
        float(np.clip(c0, 0.0, 1.0)),
        float(np.clip(c1, 0.0, 1.0)),
        float(np.clip(min(lo, hi), 0.0, 1.0)),
        float(np.clip(max(lo, hi), 0.0, 1.0)),
        float(r0),
        float(r1),
    )


def _append_region_once(
    regions: list[tuple[Any, ...]], region: tuple[Any, ...]
) -> None:
    key = (str(region[0]).lower(), np.asarray(region[1:], dtype=float))
    for existing in regions:
        if len(existing) != len(region):
            continue
        try:
            if str(existing[0]).lower() == key[0] and np.allclose(
                np.asarray(existing[1:], dtype=float),
                key[1],
                rtol=1e-6,
                atol=1e-8,
            ):
                return
        except Exception:
            continue
    regions.append(region)


def _joint_pin_from_anchor_cylinders(
    anchor_a: list[tuple[Any, ...]],
    anchor_b: list[tuple[Any, ...]],
) -> Optional[tuple[Any, ...]]:
    """Infer separate pin hardware from a pair of coaxial bore surfaces.

    The returned fractional cylinder uses 90% of the smaller bore radius,
    leaving a visible annular clearance in the recovered assembly. The pin
    spans both anchor bodies but remains outside the topology material budget.
    """
    candidates: list[tuple[float, tuple[Any, ...], tuple[Any, ...]]] = []
    for cyl_a in anchor_a:
        for cyl_b in anchor_b:
            if len(cyl_a) < 6 or len(cyl_b) < 6:
                continue
            axis_a = str(cyl_a[0] or "z").strip().lower()
            axis_b = str(cyl_b[0] or "z").strip().lower()
            if axis_a != axis_b:
                continue
            try:
                center_delta = float(
                    np.linalg.norm(
                        np.asarray(cyl_a[1:3], dtype=float)
                        - np.asarray(cyl_b[1:3], dtype=float)
                    )
                )
            except Exception:
                continue
            candidates.append((center_delta, cyl_a, cyl_b))
    if not candidates:
        return None

    center_delta, cyl_a, cyl_b = min(candidates, key=lambda item: item[0])
    # A physical pin is meaningful only for genuinely coaxial selections.
    if center_delta > 0.03:
        return None
    axis = str(cyl_a[0] or "z").strip().lower()
    try:
        c0 = 0.5 * (float(cyl_a[1]) + float(cyl_b[1]))
        c1 = 0.5 * (float(cyl_a[2]) + float(cyl_b[2]))
        lo = min(float(cyl_a[3]), float(cyl_a[4]), float(cyl_b[3]), float(cyl_b[4]))
        hi = max(float(cyl_a[3]), float(cyl_a[4]), float(cyl_b[3]), float(cyl_b[4]))
        a_r0 = float(cyl_a[5])
        b_r0 = float(cyl_b[5])
        a_r1 = float(cyl_a[6]) if len(cyl_a) > 6 else a_r0
        b_r1 = float(cyl_b[6]) if len(cyl_b) > 6 else b_r0
    except Exception:
        return None
    # Keep enough visual and geometric clearance to survive the structured
    # recovery grid. Contact is represented by the joint element, not by
    # accidentally merging the pin into either optimized body.
    r0 = 0.70 * min(a_r0, b_r0)
    r1 = 0.70 * min(a_r1, b_r1)
    if hi <= lo or r0 <= 0.0 or r1 <= 0.0:
        return None
    return (
        axis,
        float(np.clip(c0, 0.0, 1.0)),
        float(np.clip(c1, 0.0, 1.0)),
        float(np.clip(lo, 0.0, 1.0)),
        float(np.clip(hi, 0.0, 1.0)),
        r0,
        r1,
    )
