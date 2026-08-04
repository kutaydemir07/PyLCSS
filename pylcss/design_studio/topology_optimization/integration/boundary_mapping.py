# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Boundary-condition, contact, and joint mapping for topology nodes."""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np

from ..models.study import JointDefinition, LoadCase, VoxelBC
from .geometry_mapping import _bbox_tuple, _entry_bboxes, _fraction_box

logger = logging.getLogger(__name__)


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


# Frozen bearing-collar wall, as a fraction of the bore radius.
COLLAR_WALL_RADIUS_FRACTION = 0.5


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

    # Bearing-collar wall. The factor is applied to the bore radius, which is
    # already a physical proportion, so the collar stays isotropic.
    #
    # 0.35*r gave a 2.1 mm wall on a 12 mm bore, and the guided voxel grid for
    # that part is ~1.95 mm — barely one voxel. A one-voxel ring cannot be
    # resolved by the optimizer or survive marching-cubes recovery, which is
    # why the bore looked like it had been optimized away. A half-radius wall
    # is roughly two voxels at the same resolution and is still a thin
    # bearing boss, so it does not give away much design freedom.
    wall0 = max(COLLAR_WALL_RADIUS_FRACTION * r0, 0.012)
    wall1 = max(COLLAR_WALL_RADIUS_FRACTION * r1, 0.012)
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


def _cylinder_fractional_bbox(
    cylinder: tuple[Any, ...],
) -> Optional[tuple[float, float, float, float, float, float]]:
    """Axis-aligned fractional bounds of a cylindrical passive region."""
    if cylinder is None or len(cylinder) < 6:
        return None
    try:
        axis = str(cylinder[0] or "z").strip().lower()
        c0, c1, lo, hi, r0 = (float(v) for v in cylinder[1:6])
        r1 = float(cylinder[6]) if len(cylinder) > 6 else r0
    except (TypeError, ValueError):
        return None
    radial = {"x": (1, 2), "y": (0, 2), "z": (0, 1)}.get(axis)
    if radial is None:
        return None
    axial = {"x": 0, "y": 1, "z": 2}[axis]
    box = [0.0] * 6
    box[2 * axial], box[2 * axial + 1] = min(lo, hi), max(lo, hi)
    box[2 * radial[0]], box[2 * radial[0] + 1] = c0 - r0, c0 + r0
    box[2 * radial[1]], box[2 * radial[1] + 1] = c1 - r1, c1 + r1
    return tuple(box)  # type: ignore[return-value]


def _box_matches_any(
    box: tuple[float, float, float, float, float, float],
    candidates: list[tuple[float, float, float, float, float, float]],
    *,
    tolerance: float = 0.02,
) -> bool:
    """True when ``box`` is essentially the bounding box of a known region."""
    try:
        values = [float(v) for v in box[:6]]
    except (TypeError, ValueError):
        return False
    for candidate in candidates:
        if all(
            abs(values[index] - float(candidate[index])) <= tolerance
            for index in range(6)
        ):
            return True
    return False


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
    min_thickness: float | tuple[float, float, float] = 0.012,
) -> tuple[float, float, float, float, float, float]:
    """Turn a load/support patch into a positive-thickness passive box.

    ``min_thickness`` may be one fraction for every axis, or a per-axis tuple
    describing a physically isotropic thickness (see
    :func:`_isotropic_thickness_fractions`).
    """
    values = [float(v) for v in box]
    out: list[float] = []
    if isinstance(min_thickness, (int, float)):
        thicknesses = (max(float(min_thickness), 1e-5),) * 3
    else:
        thicknesses = tuple(max(float(v), 1e-5) for v in min_thickness)
    for axis, (lo, hi) in enumerate(
        (
            (values[0], values[1]),
            (values[2], values[3]),
            (values[4], values[5]),
        )
    ):
        thickness = thicknesses[axis]
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


def _trim_box_to_design_domain(
    box: tuple[float, float, float, float, float, float],
    design_domain: Any,
) -> Optional[tuple[float, float, float, float, float, float]]:
    """Shrink a normalised box to the part it is supposed to sit on.

    Contact pads are centred on the selected entity, so a BC that lies on a
    domain boundary puts half the pad in empty space. Downstream consumers —
    the optimizer, surface recovery, the manufacturing masks — all read these
    boxes as must-keep material, so an untrimmed pad grows solid outside the
    design domain.

    The result is the input box clipped to the bounding extent of the domain
    voxels it actually covers, which keeps it a box. It never grows the input.
    Returns None when the box misses the domain entirely.
    """
    if design_domain is None:
        return box
    try:
        grid = np.asarray(design_domain, dtype=bool)
    except Exception:
        return box
    if grid.ndim != 3 or not grid.any():
        return box
    spans = [
        sorted((float(box[0]), float(box[1]))),
        sorted((float(box[2]), float(box[3]))),
        sorted((float(box[4]), float(box[5]))),
    ]
    windows: list[slice] = []
    for (lo, hi), n in zip(spans, grid.shape, strict=True):
        first = max(0, min(n - 1, int(np.floor(lo * n))))
        last = max(first + 1, min(n, int(np.ceil(hi * n))))
        windows.append(slice(first, last))
    block = grid[tuple(windows)]
    if not block.any():
        return None
    trimmed: list[float] = []
    for axis, ((lo, hi), n) in enumerate(zip(spans, grid.shape, strict=True)):
        occupied = np.flatnonzero(
            np.any(block, axis=tuple(a for a in range(3) if a != axis))
        )
        start = windows[axis].start + int(occupied[0])
        stop = windows[axis].start + int(occupied[-1]) + 1
        lo_out = max(lo, start / float(n))
        hi_out = min(hi, stop / float(n))
        if hi_out <= lo_out:
            return None
        trimmed.extend((lo_out, hi_out))
    return (
        trimmed[0],
        trimmed[1],
        trimmed[2],
        trimmed[3],
        trimmed[4],
        trimmed[5],
    )


def _add_box_contact_region(
    bc: VoxelBC,
    box: tuple[float, float, float, float, float, float],
    *,
    min_thickness: float | tuple[float, float, float] = 0.012,
    design_domain: Any = None,
) -> None:
    # Trimmed after the minimum-thickness pass, which would otherwise re-grow
    # the pad back across the domain boundary.
    region = _trim_box_to_design_domain(
        _contact_solid_box_region(box, min_thickness=min_thickness),
        design_domain,
    )
    if region is None:
        logger.warning(
            "TopologyOptVoxelNode: a load/support exclusion pad lies outside "
            "the design domain and was dropped. Check that the selected "
            "entity belongs to the optimized body."
        )
        return
    _append_box_once(bc.solid_boxes, region)


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


# Exclusion regions are sized from a physical length, expressed as a fraction
# of the domain's characteristic size. Applying one fraction to each axis
# separately — as this code used to — makes the frozen region physically
# anisotropic: on a 128 x 75 x 20 mm domain, "3.5%" is 9.0 mm across X but
# only 1.4 mm across Z, a 6.4x difference on the same part for the same
# setting. Commercial tools expose this as an isotropic exclusion thickness.
PROGRAM_CONTROLLED_EXCLUSION_FRACTION = 0.035
PROGRAM_CONTROLLED_EXCLUSION_LAYERS = 2.0


def _characteristic_length(span: Any) -> Optional[float]:
    """Return the geometric mean of the domain extents, or None."""
    if span is None:
        return None
    try:
        extents = np.abs(np.asarray(span, dtype=float).reshape(-1)[:3])
    except Exception:
        return None
    if extents.size < 3 or not np.all(np.isfinite(extents)) or np.any(extents <= 0.0):
        return None
    return float(np.cbrt(float(np.prod(extents))))


def _isotropic_thickness_fractions(
    span: Any,
    fraction: float = PROGRAM_CONTROLLED_EXCLUSION_FRACTION,
    *,
    physical: float | None = None,
) -> tuple[float, float, float]:
    """Per-axis fractional *full* thickness of one physically isotropic box.

    ``span`` is the domain extent in model units. Returning per-axis fractions
    that all correspond to the *same* physical length is what makes the frozen
    region a cube in millimetres instead of a slab that follows the aspect
    ratio of the bounding box.
    """
    characteristic = _characteristic_length(span)
    if characteristic is None:
        return (float(fraction),) * 3
    physical_width = (
        float(physical)
        if physical is not None
        else float(fraction) * characteristic
    )
    if not np.isfinite(physical_width) or physical_width <= 0.0:
        raise ValueError("Exclusion thickness must be a positive finite length.")
    extents = np.abs(np.asarray(span, dtype=float).reshape(-1)[:3])
    return tuple(
        float(np.clip(physical_width / max(float(extent), 1e-9), 1e-5, 0.5))
        for extent in extents
    )  # type: ignore[return-value]


def _characteristic_span(span: Any) -> float:
    """Representative size of the design domain, in model units.

    The geometric mean of the three extents. Used to bound sizes that would
    otherwise be stated against the domain's smallest dimension, which on a
    plate-like part is its thickness and makes any fraction of it meaningless.
    """
    try:
        extents = np.abs(np.asarray(span, dtype=float).reshape(-1)[:3])
    except Exception:
        return 0.0
    if extents.size < 3 or not np.all(np.isfinite(extents)) or np.any(extents <= 0.0):
        return 0.0
    return float(np.cbrt(np.prod(extents)))


def _program_controlled_exclusion_thickness(
    span: Any,
    grid_shape: Any,
) -> float | None:
    """Return two average element lengths in the model's length unit."""
    try:
        extents = np.abs(np.asarray(span, dtype=float).reshape(-1)[:3])
        cells = np.asarray(grid_shape, dtype=float).reshape(-1)[:3]
    except Exception:
        return None
    if (
        extents.size < 3
        or cells.size < 3
        or not np.all(np.isfinite(extents))
        or not np.all(np.isfinite(cells))
        or np.any(extents <= 0.0)
        or np.any(cells <= 0.0)
    ):
        return None
    average_element_length = float(np.cbrt(np.prod(extents / cells)))
    return PROGRAM_CONTROLLED_EXCLUSION_LAYERS * average_element_length


def _point_contact_box(
    x: float,
    y: float,
    z: float,
    *,
    half_width: float | tuple[float, float, float] = 0.012,
) -> tuple[float, float, float, float, float, float]:
    if isinstance(half_width, (int, float)):
        widths = (float(half_width),) * 3
    else:
        widths = tuple(float(v) for v in half_width)  # type: ignore[assignment]

    def _interval(value: float, w: float) -> tuple[float, float]:
        w = max(float(w), 1e-5)
        center = max(0.0, min(1.0, float(value)))
        if center <= w:
            return 0.0, min(1.0, 2.0 * w)
        if center >= 1.0 - w:
            return max(0.0, 1.0 - 2.0 * w), 1.0
        return max(0.0, center - w), min(1.0, center + w)

    x0, x1 = _interval(x, widths[0])
    y0, y1 = _interval(y, widths[1])
    z0, z1 = _interval(z, widths[2])
    return x0, x1, y0, y1, z0, z1


def _add_bc_contact_regions(
    bc: VoxelBC,
    span: Any = None,
    grid_shape: Any = None,
    *,
    scope: str = "All Loads and Supports",
    manual_thickness: float | None = None,
    design_domain: Any = None,
    lattice_cell_size: float | None = None,
) -> float | None:
    """Freeze every active load/support interface as non-design material.

    Program-controlled thickness is two average voxel element lengths.
    ``manual_thickness`` overrides that value in the model's length unit.
    The return value is the effective physical thickness, when available.

    ``lattice_cell_size`` is the manufactured cell pitch, in the same length
    unit, when the study builds a lattice. Two element lengths is the right
    frozen band for a solid study, where the surrounding material is
    continuous, and it is far too thin for a lattice one: the pad has to be at
    least a full cell deep or the load is introduced into open cells with no
    solid to spread through. Measured on the bundled crush block, the
    program-controlled pad came to 10.2 mm against a 40.9 mm cell — a quarter
    of one cell — so the Density view showed a fully covered load face while
    the manufactured mesh showed a paper-thin plate backed immediately by
    voids. That mismatch is not a conversion error in the lattice builder; it
    is the interface pad being sized for the wrong kind of part.

    ``design_domain`` is the voxelised source geometry. Pads are trimmed to it
    so a BC selected on a domain boundary cannot seed material off the part.
    """
    scope_key = str(scope or "All Loads and Supports").strip().lower()
    valid_scopes = {
        "all loads and supports",
        "loads only",
        "supports only",
        "none",
    }
    if scope_key not in valid_scopes:
        raise ValueError(
            "Automatic exclusion scope must be All Loads and Supports, "
            "Loads Only, Supports Only, or None."
        )
    if scope_key == "none":
        return None

    preserve_loads = scope_key in {"all loads and supports", "loads only"}
    preserve_supports = scope_key in {
        "all loads and supports",
        "supports only",
    }
    effective_thickness = (
        float(manual_thickness)
        if manual_thickness is not None
        else _program_controlled_exclusion_thickness(span, grid_shape)
    )
    if effective_thickness is not None and (
        not np.isfinite(effective_thickness) or effective_thickness <= 0.0
    ):
        raise ValueError("Manual exclusion thickness must be positive.")
    if (
        manual_thickness is None
        and lattice_cell_size is not None
        and np.isfinite(lattice_cell_size)
        and lattice_cell_size > 0.0
    ):
        # Aim for one full cell of solid before the lattice starts, but never
        # let the frozen band run away with the design space. An interface pad
        # is a full-cross-section slab at each end, so its cost is twice its
        # depth as a fraction of the part: asking for a whole cell without a
        # cap froze 17.8% of the bundled payload fitting, against 6.4% before,
        # and the volume budget that buys the actual structure went with it.
        #
        # When the cap binds, the cell is more than a sixteenth of the part and
        # the load has no room for a proper introduction layer whatever this
        # value is. That is a statement about the cell being too coarse for the
        # part, not about the pad, and `cell_resolution_warning` is where it
        # gets said. A manual thickness is the user's number and is left alone.
        characteristic = _characteristic_span(span)
        ceiling = (
            characteristic / 16.0
            if characteristic > 0.0
            else float(lattice_cell_size)
        )
        effective_thickness = max(
            float(effective_thickness or 0.0),
            min(float(lattice_cell_size), ceiling),
        )
    fractional_thickness = (
        _isotropic_thickness_fractions(span, physical=effective_thickness)
        if span is not None and effective_thickness is not None
        else _isotropic_thickness_fractions(span)
    )
    # `_point_contact_box` grows a box symmetrically about the point, so it
    # takes the half. Handing it the full thickness gave a point force a pad
    # twice as thick as every other selection: on a 200x60x15 mm domain at the
    # program-controlled 5 mm, a vertex support froze a 5 mm cube and a point
    # force at the same setting froze a 10 mm one.
    fractional_half_thickness = tuple(
        0.5 * float(value) for value in fractional_thickness
    )

    # A cylindrical selection has already registered its own analytic pad: a
    # sleeve around the bore, with the bore itself left open (see
    # `_add_cylindrical_contact_region`). Emitting a box for the same region on
    # top of that replaces a round collar with a square block the width of the
    # bore's bounding box -- a 76 mm boss came out as a 76 x 76 solid pad -- and
    # it fills the bore back in. Commercial practice scopes the automatic
    # exclusion to the elements the boundary condition is actually on, so the
    # box is skipped wherever the accurate region already exists.
    analytic_pads = [
        _cylinder_fractional_bbox(cylinder)
        for cylinder in list(bc.solid_cylinders) + list(bc.void_cylinders)
    ]
    analytic_pads = [pad for pad in analytic_pads if pad is not None]

    def _box(box: tuple[float, float, float, float, float, float]) -> None:
        if _box_matches_any(box, analytic_pads):
            return
        _add_box_contact_region(
            bc,
            box,
            min_thickness=fractional_thickness,
            design_domain=design_domain,
        )

    def _add_joint_contacts(joints: list[JointDefinition]) -> None:
        for joint in joints:
            _box(
                _point_contact_box(
                    *joint.anchor_a,
                    half_width=fractional_half_thickness,
                )
            )
            _box(
                _point_contact_box(
                    *joint.anchor_b,
                    half_width=fractional_half_thickness,
                )
            )

    if preserve_supports:
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
                _box(box)

        for x0, x1, y0, y1, z0, z1, dofs in list(bc.fixed_boxes):
            if dofs:
                _box((x0, x1, y0, y1, z0, z1))

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
        if preserve_supports:
            _add_joint_contacts(list(load_case.joints))
            for x0, x1, y0, y1, z0, z1, dofs in load_case.fixed_boxes:
                if dofs:
                    _box((x0, x1, y0, y1, z0, z1))
        if not preserve_loads:
            continue
        for x, y, z, fx, fy, fz in load_case.point_forces:
            if (
                float(fx) * float(fx) + float(fy) * float(fy) + float(fz) * float(fz)
            ) <= 1e-24:
                continue
            _box(
                _point_contact_box(
                    x,
                    y,
                    z,
                    half_width=fractional_half_thickness,
                )
            )
        for x0, x1, y0, y1, z0, z1, fx, fy, fz in load_case.box_forces:
            if (
                float(fx) * float(fx) + float(fy) * float(fy) + float(fz) * float(fz)
            ) <= 1e-24:
                continue
            _box((x0, x1, y0, y1, z0, z1))
        for face, fx, fy, fz in load_case.distributed_forces:
            if (
                float(fx) * float(fx) + float(fy) * float(fy) + float(fz) * float(fz)
            ) <= 1e-24:
                continue
            box = _face_contact_box(face)
            if box is not None:
                _box(box)
    return effective_thickness
