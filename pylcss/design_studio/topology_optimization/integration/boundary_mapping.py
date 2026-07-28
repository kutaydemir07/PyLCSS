# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Boundary-condition, contact, and joint mapping for topology nodes."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from ..models.study import JointDefinition, LoadCase, VoxelBC
from .geometry_mapping import _bbox_tuple, _entry_bboxes, _fraction_box


def _is_load_payload(entry: Any) -> bool:
    return isinstance(entry, dict) and str(entry.get("type") or "").lower() in {
        "force",
        "pressure",
        "gravity",
    }


def _force_components_nonzero(values: Any) -> bool:
    try:
        vals = [float(v) for v in values]
    except Exception:
        return False
    return sum(v * v for v in vals) > 1e-24


def _bc_has_nonzero_load(bc: VoxelBC) -> bool:
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


def _iter_load_faces(load: dict[str, Any]) -> list[Any]:
    faces = list(load.get("geometries") or [])
    if not faces and load.get("geometry") is not None:
        faces = [load.get("geometry")]
    return [face for face in faces if face is not None]


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


def _pressure_face_patches(
    face: Any,
    bounds: tuple[np.ndarray, np.ndarray],
    pressure: float,
    *,
    max_bins: tuple[int, int, int] = (12, 12, 4),
) -> list[tuple[tuple[float, float, float, float, float, float], float, float, float]]:
    """Approximate pressure on flat or curved CAD faces as voxel patch loads.

    The topology solver accepts nodal box forces, not native pressure BCs.
    Tessellating the face lets curved pressure, such as pressure on a hole
    wall, remain self-equilibrated instead of collapsing into one misleading
    resultant vector.
    """
    if abs(float(pressure)) <= 1e-30:
        return []

    try:
        vertices, triangles = face.tessellate(0.75)
    except TypeError:
        try:
            vertices, triangles = face.tessellate(1.0)
        except Exception:
            return []
    except Exception:
        return []

    points = [_point_xyz(v) for v in vertices]
    if not points or any(p is None for p in points):
        return []
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] < 3:
        return []

    mins, maxs = bounds
    span = np.maximum(maxs[:3] - mins[:3], 1e-12)
    bin_counts = np.maximum(np.asarray(max_bins, dtype=int), 1)
    bins: dict[tuple[int, int, int], list[Any]] = {}

    for tri in triangles:
        try:
            ids = [int(i) for i in list(tri)[:3]]
        except Exception:
            continue
        if len(ids) != 3 or any(i < 0 or i >= len(pts) for i in ids):
            continue

        tri_pts = pts[ids, :3]
        area_vec = 0.5 * np.cross(tri_pts[1] - tri_pts[0], tri_pts[2] - tri_pts[0])
        if float(np.linalg.norm(area_vec)) <= 1e-12:
            continue

        force = float(pressure) * area_vec
        center = np.mean(tri_pts, axis=0)
        frac = np.clip((center - mins[:3]) / span, 0.0, 1.0)
        key_arr = np.floor(frac * bin_counts).astype(int)
        key_arr = np.clip(key_arr, 0, bin_counts - 1)
        key = (int(key_arr[0]), int(key_arr[1]), int(key_arr[2]))

        lo = np.min(tri_pts, axis=0)
        hi = np.max(tri_pts, axis=0)
        bbox = [lo[0], hi[0], lo[1], hi[1], lo[2], hi[2]]
        if key not in bins:
            bins[key] = [bbox, force.astype(float)]
        else:
            acc_bbox, acc_force = bins[key]
            acc_bbox[0] = min(acc_bbox[0], bbox[0])
            acc_bbox[1] = max(acc_bbox[1], bbox[1])
            acc_bbox[2] = min(acc_bbox[2], bbox[2])
            acc_bbox[3] = max(acc_bbox[3], bbox[3])
            acc_bbox[4] = min(acc_bbox[4], bbox[4])
            acc_bbox[5] = max(acc_bbox[5], bbox[5])
            acc_force += force

    patches: list[
        tuple[tuple[float, float, float, float, float, float], float, float, float]
    ] = []
    for _, (bbox, force) in sorted(bins.items(), key=lambda item: item[0]):
        if float(np.linalg.norm(force)) <= 1e-12:
            continue
        frac_box = _fraction_box(tuple(float(v) for v in bbox), bounds, pad=0.012)
        patches.append((frac_box, float(force[0]), float(force[1]), float(force[2])))
    return patches


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
    for face in _iter_load_faces(load):
        patches.extend(_pressure_face_patches(face, bounds, pressure))
    if patches:
        return patches

    for bbox in _entry_bboxes(load):
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


def _cylinder_contact_solid_region(
    cylinder: tuple[Any, ...],
) -> Optional[tuple[Any, ...]]:
    """Create a passive solid sleeve around a graph-selected cylindrical face."""
    if cylinder is None or len(cylinder) < 6:
        return None
    try:
        axis, c0, c1, lo, hi, r0 = cylinder[:6]
        r1 = cylinder[6] if len(cylinder) > 6 else r0
        r0 = float(r0)
        r1 = float(r1)
    except Exception:
        return None
    if r0 <= 0.0 or r1 <= 0.0:
        return None

    wall0 = max(0.35 * r0, 0.012)
    wall1 = max(0.35 * r1, 0.012)
    return (
        str(axis or "z").strip().lower(),
        float(c0),
        float(c1),
        float(lo),
        float(hi),
        float(r0 + wall0),
        float(r1 + wall1),
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


def _add_cylindrical_contact_region(bc: VoxelBC, cylinder: tuple[Any, ...]) -> None:
    """Keep a selected cylindrical BC usable: solid sleeve, void bore."""
    solid = _cylinder_contact_solid_region(cylinder)
    if solid is not None:
        _append_region_once(bc.solid_cylinders, solid)
    _append_region_once(bc.void_cylinders, cylinder)


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


def _contact_solid_box_region(
    box: tuple[float, float, float, float, float, float],
    *,
    min_thickness: float = 0.012,
) -> tuple[float, float, float, float, float, float]:
    """Turn a load/support patch into a positive-thickness passive box."""
    values = [float(v) for v in box]
    out: list[float] = []
    thickness = max(float(min_thickness), 1e-5)
    for lo, hi in (
        (values[0], values[1]),
        (values[2], values[3]),
        (values[4], values[5]),
    ):
        lo, hi = sorted((max(0.0, min(1.0, lo)), max(0.0, min(1.0, hi))))
        if (hi - lo) < thickness:
            center = 0.5 * (lo + hi)
            if center <= 0.5 * thickness:
                lo, hi = 0.0, min(1.0, thickness)
            elif center >= 1.0 - 0.5 * thickness:
                lo, hi = max(0.0, 1.0 - thickness), 1.0
            else:
                lo = max(0.0, center - 0.5 * thickness)
                hi = min(1.0, center + 0.5 * thickness)
        out.extend((float(lo), float(hi)))
    return tuple(out)  # type: ignore[return-value]


def _append_box_once(
    regions: list[tuple[float, float, float, float, float, float]],
    region: tuple[float, float, float, float, float, float],
) -> None:
    key = np.asarray(region, dtype=float)
    for existing in regions:
        try:
            if np.allclose(
                np.asarray(existing, dtype=float), key, rtol=1e-6, atol=1e-8
            ):
                return
        except Exception:
            continue
    regions.append(region)


def _add_box_contact_region(
    bc: VoxelBC,
    box: tuple[float, float, float, float, float, float],
) -> None:
    _append_box_once(bc.solid_boxes, _contact_solid_box_region(box))


def _face_contact_box(
    face: str,
) -> Optional[tuple[float, float, float, float, float, float]]:
    face_name = str(face or "").strip().lower()
    boxes = {
        "left": (0.0, 0.0, 0.0, 1.0, 0.0, 1.0),
        "right": (1.0, 1.0, 0.0, 1.0, 0.0, 1.0),
        "bottom": (0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
        "top": (0.0, 1.0, 1.0, 1.0, 0.0, 1.0),
        "front": (0.0, 1.0, 0.0, 1.0, 0.0, 0.0),
        "back": (0.0, 1.0, 0.0, 1.0, 1.0, 1.0),
    }
    return boxes.get(face_name)


def _point_contact_box(
    x: float,
    y: float,
    z: float,
    *,
    half_width: float = 0.012,
) -> tuple[float, float, float, float, float, float]:
    w = max(float(half_width), 1e-5)

    def _interval(value: float) -> tuple[float, float]:
        center = max(0.0, min(1.0, float(value)))
        if center <= w:
            return 0.0, min(1.0, 2.0 * w)
        if center >= 1.0 - w:
            return max(0.0, 1.0 - 2.0 * w), 1.0
        return max(0.0, center - w), min(1.0, center + w)

    x0, x1 = _interval(x)
    y0, y1 = _interval(y)
    z0, z1 = _interval(z)
    return x0, x1, y0, y1, z0, z1


def _add_bc_contact_regions(bc: VoxelBC) -> None:
    """Freeze every active load/support interface as non-design material."""

    def _add_joint_contacts(joints: list[JointDefinition]) -> None:
        for joint in joints:
            _add_box_contact_region(
                bc,
                _point_contact_box(*joint.anchor_a, half_width=0.035),
            )
            _add_box_contact_region(
                bc,
                _point_contact_box(*joint.anchor_b, half_width=0.035),
            )

    _add_joint_contacts(list(bc.joints))
    face_supports = (
        ("left", bc.fixed_left_face_dofs),
        ("right", bc.fixed_right_face_dofs),
        ("top", bc.fixed_top_face_dofs),
        ("bottom", bc.fixed_bottom_face_dofs),
        ("front", bc.fixed_front_face_dofs),
        ("back", bc.fixed_back_face_dofs),
    )
    for face, dofs in face_supports:
        if not dofs:
            continue
        box = _face_contact_box(face)
        if box is not None:
            _add_box_contact_region(bc, box)

    for x0, x1, y0, y1, z0, z1, dofs in list(bc.fixed_boxes):
        if dofs:
            _add_box_contact_region(bc, (x0, x1, y0, y1, z0, z1))

    load_cases = list(bc.load_cases)
    if not load_cases and (bc.point_forces or bc.box_forces or bc.distributed_forces):
        load_cases = [
            LoadCase(
                name="LC1",
                point_forces=list(bc.point_forces),
                box_forces=list(bc.box_forces),
                distributed_forces=list(bc.distributed_forces),
            )
        ]
    for load_case in load_cases:
        _add_joint_contacts(list(load_case.joints))
        for x0, x1, y0, y1, z0, z1, dofs in load_case.fixed_boxes:
            if dofs:
                _add_box_contact_region(
                    bc,
                    (x0, x1, y0, y1, z0, z1),
                )
        for x, y, z, fx, fy, fz in load_case.point_forces:
            if (
                float(fx) * float(fx) + float(fy) * float(fy) + float(fz) * float(fz)
            ) <= 1e-24:
                continue
            _add_box_contact_region(bc, _point_contact_box(x, y, z))
        for x0, x1, y0, y1, z0, z1, fx, fy, fz in load_case.box_forces:
            if (
                float(fx) * float(fx) + float(fy) * float(fy) + float(fz) * float(fz)
            ) <= 1e-24:
                continue
            _add_box_contact_region(bc, (x0, x1, y0, y1, z0, z1))
        for face, fx, fy, fz in load_case.distributed_forces:
            if (
                float(fx) * float(fx) + float(fy) * float(fy) + float(fz) * float(fz)
            ) <= 1e-24:
                continue
            box = _face_contact_box(face)
            if box is not None:
                _add_box_contact_region(bc, box)
