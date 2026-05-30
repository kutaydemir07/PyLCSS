# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.
"""The Topology Opt (Voxel) graph node — the Qt-facing wrapper around the solver."""
from __future__ import annotations

import logging
import json
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import numba

from pylcss.design_studio.core.base_node import CadQueryNode
from .boundary_conditions import (
    VoxelBC, LoadCase, ManufacturingConstraints,
    _parse_support, _parse_support_region_dofs, _parse_region_boxes,
    _parse_region_cylinders, _parse_load_cases,
)
from .presets import (
    industrial_topopt_defaults,
    INDUSTRIAL_WORKFLOW_MODES, INDUSTRIAL_DESIGN_GOALS,
    INDUSTRIAL_MANUFACTURING_PROCESSES,
)
from .solver import TopologyOptVoxelSolver, TopologyOptVoxelProblem
from .recovery import _recover_voxel_shape

logger = logging.getLogger(__name__)

def _flatten(values: Any) -> List[Any]:
    if values is None:
        return []
    if isinstance(values, (list, tuple)):
        out: List[Any] = []
        for item in values:
            out.extend(_flatten(item))
        return out
    return [values]


def _mesh_bounds(mesh: Any) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    if mesh is None or not hasattr(mesh, 'p'):
        return None
    pts = np.asarray(mesh.p, dtype=float)
    if pts.ndim != 2 or pts.shape[0] < 3 or pts.shape[1] == 0:
        return None
    return pts[:3].min(axis=1), pts[:3].max(axis=1)


def _bbox_tuple(bb: Any) -> Optional[Tuple[float, float, float, float, float, float]]:
    if bb is None:
        return None
    if isinstance(bb, dict):
        try:
            return (
                float(bb['xmin']), float(bb['xmax']),
                float(bb['ymin']), float(bb['ymax']),
                float(bb['zmin']), float(bb['zmax']),
            )
        except Exception:
            return None
    try:
        return (
            float(bb.xmin), float(bb.xmax),
            float(bb.ymin), float(bb.ymax),
            float(bb.zmin), float(bb.zmax),
        )
    except Exception:
        return None


def _entry_bboxes(entry: Dict[str, Any]) -> List[Tuple[float, float, float, float, float, float]]:
    bboxes: List[Tuple[float, float, float, float, float, float]] = []
    for face in entry.get('geometries') or []:
        try:
            bb = _bbox_tuple(face.BoundingBox())
        except Exception:
            bb = None
        if bb is not None:
            bboxes.append(bb)

    viz = entry.get('viz') if isinstance(entry.get('viz'), dict) else {}
    for face in viz.get('faces') or []:
        bb = _bbox_tuple(face.get('bbox') if isinstance(face, dict) else None)
        if bb is not None:
            bboxes.append(bb)
    bb = _bbox_tuple(viz.get('bbox'))
    if bb is not None:
        bboxes.append(bb)
    return bboxes


def _fraction(value: float, lo: float, hi: float, *, invert: bool = False) -> float:
    span = max(float(hi) - float(lo), 1e-12)
    frac = (float(value) - float(lo)) / span
    if invert:
        frac = 1.0 - frac
    return float(np.clip(frac, 0.0, 1.0))


def _fraction_box(
    bbox: Tuple[float, float, float, float, float, float],
    bounds: Tuple[np.ndarray, np.ndarray],
    pad: float = 0.02,
) -> Tuple[float, float, float, float, float, float]:
    mins, maxs = bounds
    vals = [
        _fraction(bbox[0], mins[0], maxs[0]),
        _fraction(bbox[1], mins[0], maxs[0]),
        _fraction(bbox[2], mins[1], maxs[1]),
        _fraction(bbox[3], mins[1], maxs[1]),
        _fraction(bbox[4], mins[2], maxs[2]),
        _fraction(bbox[5], mins[2], maxs[2]),
    ]
    for i in (0, 2, 4):
        if abs(vals[i + 1] - vals[i]) < pad:
            center = 0.5 * (vals[i] + vals[i + 1])
            vals[i] = max(0.0, center - pad)
            vals[i + 1] = min(1.0, center + pad)
    return tuple(vals)  # type: ignore[return-value]


def _fraction_center(
    bbox: Tuple[float, float, float, float, float, float],
    bounds: Tuple[np.ndarray, np.ndarray],
) -> Tuple[float, float, float]:
    mins, maxs = bounds
    return (
        _fraction(0.5 * (bbox[0] + bbox[1]), mins[0], maxs[0]),
        _fraction(0.5 * (bbox[2] + bbox[3]), mins[1], maxs[1]),
        _fraction(0.5 * (bbox[4] + bbox[5]), mins[2], maxs[2]),
    )


def _is_load_payload(entry: Any) -> bool:
    return (
        isinstance(entry, dict)
        and str(entry.get('type') or '').lower() in {'force', 'pressure', 'gravity'}
    )


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
    if any((
        bc.fixed_left_face_dofs,
        bc.fixed_right_face_dofs,
        bc.fixed_top_face_dofs,
        bc.fixed_bottom_face_dofs,
        bc.fixed_front_face_dofs,
        bc.fixed_back_face_dofs,
    )):
        return True
    return any(bool(box[-1]) for box in bc.fixed_boxes)


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


def _iter_load_faces(load: Dict[str, Any]) -> List[Any]:
    faces = list(load.get('geometries') or [])
    if not faces and load.get('geometry') is not None:
        faces = [load.get('geometry')]
    return [face for face in faces if face is not None]


def _bbox_pressure_resultant(
    bbox: Tuple[float, float, float, float, float, float],
    bounds: Tuple[np.ndarray, np.ndarray],
    pressure: float,
) -> Tuple[float, float, float]:
    """Approximate a planar pressure face as one resultant force."""
    mins, maxs = bounds
    extents = np.asarray([
        max(0.0, bbox[1] - bbox[0]),
        max(0.0, bbox[3] - bbox[2]),
        max(0.0, bbox[5] - bbox[4]),
    ], dtype=float)
    axis = int(np.argmin(extents))
    area_axes = [idx for idx in range(3) if idx != axis]
    area = float(np.prod(extents[area_axes]))
    center = np.asarray([
        0.5 * (bbox[0] + bbox[1]),
        0.5 * (bbox[2] + bbox[3]),
        0.5 * (bbox[4] + bbox[5]),
    ], dtype=float)
    midpoint = 0.5 * (mins[:3] + maxs[:3])
    sign = 1.0 if center[axis] >= midpoint[axis] else -1.0
    force = np.zeros(3, dtype=float)
    force[axis] = float(pressure) * area * sign
    return float(force[0]), float(force[1]), float(force[2])


def _pressure_face_patches(
    face: Any,
    bounds: Tuple[np.ndarray, np.ndarray],
    pressure: float,
    *,
    max_bins: Tuple[int, int, int] = (12, 12, 4),
) -> List[Tuple[Tuple[float, float, float, float, float, float], float, float, float]]:
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
    bins: Dict[Tuple[int, int, int], List[Any]] = {}

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

    patches: List[Tuple[Tuple[float, float, float, float, float, float], float, float, float]] = []
    for _, (bbox, force) in sorted(bins.items(), key=lambda item: item[0]):
        if float(np.linalg.norm(force)) <= 1e-12:
            continue
        frac_box = _fraction_box(tuple(float(v) for v in bbox), bounds, pad=0.012)
        patches.append((frac_box, float(force[0]), float(force[1]), float(force[2])))
    return patches


def _pressure_load_patches(
    load: Dict[str, Any],
    bounds: Tuple[np.ndarray, np.ndarray],
) -> List[Tuple[Tuple[float, float, float, float, float, float], float, float, float]]:
    try:
        pressure = float(load.get('pressure', load.get('magnitude', 0.0)))
    except Exception:
        return []
    patches: List[Tuple[Tuple[float, float, float, float, float, float], float, float, float]] = []
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


def _bounds_payload(bounds: Optional[Tuple[np.ndarray, np.ndarray]]) -> Optional[Dict[str, List[float]]]:
    if bounds is None:
        return None
    mins, maxs = bounds
    return {
        'min': [float(v) for v in mins[:3]],
        'max': [float(v) for v in maxs[:3]],
    }


def _bc_feature_bboxes(
    constraints: List[Any],
    loads: List[Any],
) -> List[Tuple[float, float, float, float, float, float]]:
    bboxes: List[Tuple[float, float, float, float, float, float]] = []
    for entry in list(constraints or []) + list(loads or []):
        if isinstance(entry, dict):
            bboxes.extend(_entry_bboxes(entry))
    return bboxes


def _cylinder_void_region_from_face(
    face: Any,
    bounds: Optional[Tuple[np.ndarray, np.ndarray]],
) -> Optional[Tuple[Any, ...]]:
    if face is None or bounds is None:
        return None
    try:
        if str(face.geomType()).upper() != 'CYLINDER':
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
    ext = np.asarray([bbox[1] - bbox[0], bbox[3] - bbox[2], bbox[5] - bbox[4]], dtype=float)
    axis_idx = int(np.argmax(np.abs(direction))) if np.any(direction) else int(np.argmax(ext))
    radial_axes = [idx for idx in range(3) if idx != axis_idx]
    axis_names = ['x', 'y', 'z']

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


def _cylinder_contact_solid_region(cylinder: Tuple[Any, ...]) -> Optional[Tuple[Any, ...]]:
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
        str(axis or 'z').strip().lower(),
        float(c0),
        float(c1),
        float(lo),
        float(hi),
        float(r0 + wall0),
        float(r1 + wall1),
    )


def _append_region_once(regions: List[Tuple[Any, ...]], region: Tuple[Any, ...]) -> None:
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


def _add_cylindrical_contact_region(bc: VoxelBC, cylinder: Tuple[Any, ...]) -> None:
    """Keep a selected cylindrical BC usable: solid sleeve, void bore."""
    solid = _cylinder_contact_solid_region(cylinder)
    if solid is not None:
        _append_region_once(bc.solid_cylinders, solid)
    _append_region_once(bc.void_cylinders, cylinder)


def _contact_solid_box_region(
    box: Tuple[float, float, float, float, float, float],
    *,
    min_thickness: float = 0.012,
) -> Tuple[float, float, float, float, float, float]:
    """Turn a load/support patch into a positive-thickness passive box."""
    values = [float(v) for v in box]
    out: List[float] = []
    thickness = max(float(min_thickness), 1e-5)
    for lo, hi in ((values[0], values[1]), (values[2], values[3]), (values[4], values[5])):
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
    regions: List[Tuple[float, float, float, float, float, float]],
    region: Tuple[float, float, float, float, float, float],
) -> None:
    key = np.asarray(region, dtype=float)
    for existing in regions:
        try:
            if np.allclose(np.asarray(existing, dtype=float), key, rtol=1e-6, atol=1e-8):
                return
        except Exception:
            continue
    regions.append(region)


def _add_box_contact_region(
    bc: VoxelBC,
    box: Tuple[float, float, float, float, float, float],
) -> None:
    _append_box_once(bc.solid_boxes, _contact_solid_box_region(box))


def _face_contact_box(face: str) -> Optional[Tuple[float, float, float, float, float, float]]:
    face_name = str(face or '').strip().lower()
    boxes = {
        'left':   (0.0, 0.0, 0.0, 1.0, 0.0, 1.0),
        'right':  (1.0, 1.0, 0.0, 1.0, 0.0, 1.0),
        'bottom': (0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
        'top':    (0.0, 1.0, 1.0, 1.0, 0.0, 1.0),
        'front':  (0.0, 1.0, 0.0, 1.0, 0.0, 0.0),
        'back':   (0.0, 1.0, 0.0, 1.0, 1.0, 1.0),
    }
    return boxes.get(face_name)


def _point_contact_box(
    x: float,
    y: float,
    z: float,
    *,
    half_width: float = 0.012,
) -> Tuple[float, float, float, float, float, float]:
    w = max(float(half_width), 1e-5)

    def _interval(value: float) -> Tuple[float, float]:
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
    face_supports = (
        ('left', bc.fixed_left_face_dofs),
        ('right', bc.fixed_right_face_dofs),
        ('top', bc.fixed_top_face_dofs),
        ('bottom', bc.fixed_bottom_face_dofs),
        ('front', bc.fixed_front_face_dofs),
        ('back', bc.fixed_back_face_dofs),
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
    if not load_cases:
        load_cases = [
            LoadCase(
                name='LC1',
                point_forces=list(bc.point_forces),
                box_forces=list(bc.box_forces),
                distributed_forces=list(bc.distributed_forces),
            )
        ]
    for load_case in load_cases:
        for x, y, z, fx, fy, fz in load_case.point_forces:
            if (float(fx) * float(fx) + float(fy) * float(fy) + float(fz) * float(fz)) <= 1e-24:
                continue
            _add_box_contact_region(bc, _point_contact_box(x, y, z))
        for x0, x1, y0, y1, z0, z1, fx, fy, fz in load_case.box_forces:
            if (float(fx) * float(fx) + float(fy) * float(fy) + float(fz) * float(fz)) <= 1e-24:
                continue
            _add_box_contact_region(bc, (x0, x1, y0, y1, z0, z1))
        for face, fx, fy, fz in load_case.distributed_forces:
            if (float(fx) * float(fx) + float(fy) * float(fy) + float(fz) * float(fz)) <= 1e-24:
                continue
            box = _face_contact_box(face)
            if box is not None:
                _add_box_contact_region(bc, box)


def _guided_voxel_grid(
    bounds: Optional[Tuple[np.ndarray, np.ndarray]],
    quality_preset: Any,
    feature_bboxes: Optional[List[Tuple[float, float, float, float, float, float]]] = None,
    feature_lengths: Optional[List[float]] = None,
) -> Optional[Tuple[int, int, int]]:
    """Choose the automatic aspect-correct grid for guided mode.

    Guided mode hides raw voxel counts, so it must derive them from the actual
    CAD extents and the smallest selected support/load/CAD features. The
    `quality_preset` argument is ignored and remains only for saved-graph
    compatibility.
    """
    if bounds is None:
        return None
    mins, maxs = bounds
    span = np.maximum(np.asarray(maxs[:3], dtype=float) - np.asarray(mins[:3], dtype=float), 1e-9)
    volume = float(np.prod(span))
    if volume <= 0.0:
        return None

    _ = quality_preset
    target_cells, min_axis, max_axis = (18_000, 5, 100)
    max_total_cells = 75_000
    voxel_size = (volume / float(target_cells)) ** (1.0 / 3.0)
    dims = np.ceil(span / max(voxel_size, 1e-12)).astype(int)
    dims = np.maximum(dims, int(min_axis))

    resolved_feature_lengths: List[float] = []
    for bbox in feature_bboxes or []:
        try:
            ext = np.asarray(
                [
                    float(bbox[1]) - float(bbox[0]),
                    float(bbox[3]) - float(bbox[2]),
                    float(bbox[5]) - float(bbox[4]),
                ],
                dtype=float,
            )
        except Exception:
            continue
        ext = np.abs(ext)
        positive = ext[ext > max(1e-6, float(np.max(span)) * 1e-6)]
        if positive.size:
            resolved_feature_lengths.append(float(np.min(positive)))
    for length in feature_lengths or []:
        try:
            value = float(length)
        except Exception:
            continue
        if value > max(1e-6, float(np.max(span)) * 1e-6):
            resolved_feature_lengths.append(value)
    if resolved_feature_lengths:
        feature_size = min(resolved_feature_lengths)
        target_across = 8.0
        feature_voxel_size = feature_size / max(float(target_across), 1.0)
        feature_dims = np.ceil(span / max(feature_voxel_size, 1e-12)).astype(int)
        dims = np.maximum(dims, feature_dims)

    longest = int(np.max(dims))
    guided_max_axis = max(int(max_axis), 120 if resolved_feature_lengths else int(max_axis))
    if longest > guided_max_axis:
        scale = float(guided_max_axis) / float(longest)
        dims = np.maximum(np.ceil(dims * scale).astype(int), int(min_axis))

    total_cells = int(np.prod(dims))
    if total_cells > int(max_total_cells):
        scale = (float(max_total_cells) / float(total_cells)) ** (1.0 / 3.0)
        dims = np.maximum(np.floor(dims * scale).astype(int), int(min_axis))
        while int(np.prod(dims)) > int(max_total_cells) and int(np.max(dims)) > int(min_axis):
            dims[int(np.argmax(dims))] -= 1

    return int(dims[0]), int(dims[1]), int(dims[2])


def _use_automatic_guided_grid(workflow_mode: Any, quality_preset: Any) -> bool:
    """Return True when guided mode should choose the grid from geometry."""
    if str(workflow_mode or 'Guided').strip().lower() == 'expert':
        return False
    _ = quality_preset
    return True


def _guided_rmin(nelx: int, nely: int, nelz: int) -> float:
    """Filter radius derived from the resolved voxel grid for guided mode.

    Guided mode auto-sizes the voxel grid from CAD geometry, so the SIMP
    filter radius must follow the same source-of-truth. Without this, a stale
    `rmin` stored in the .cad file silently overrides geometry intent and two
    studies on identical geometry can converge to different topologies.
    """
    max_dim = max(int(nelx), int(nely), int(nelz), 1)
    return round(max(1.2, min(5.0, max_dim * 0.030)), 2)


def _axis_radial_indices(axis: Any) -> Tuple[int, int]:
    axis_name = str(axis or 'z').strip().lower()
    if axis_name == 'x':
        return 1, 2
    if axis_name == 'y':
        return 0, 2
    return 0, 1


def _cylinder_actual_radii(
    cylinder: Tuple[Any, ...],
    span: np.ndarray,
) -> Optional[Tuple[str, float, float, float, float, float, float]]:
    if len(cylinder) < 6:
        return None
    axis = str(cylinder[0] or 'z').strip().lower()
    c0 = float(cylinder[1])
    c1 = float(cylinder[2])
    lo = float(cylinder[3])
    hi = float(cylinder[4])
    r0 = float(cylinder[5])
    r1 = float(cylinder[6]) if len(cylinder) > 6 else r0
    if r0 <= 0.0 or r1 <= 0.0:
        return None
    a0, a1 = _axis_radial_indices(axis)
    return axis, c0, c1, min(lo, hi), max(lo, hi), r0 * float(span[a0]), r1 * float(span[a1])


def _cylinder_feature_lengths(
    bounds: Optional[Tuple[np.ndarray, np.ndarray]],
    solid_cylinders: Optional[List[Tuple[Any, ...]]] = None,
    void_cylinders: Optional[List[Tuple[Any, ...]]] = None,
) -> List[float]:
    """Return physical feature lengths that should be resolved by the grid."""
    if bounds is None:
        return []
    mins, maxs = bounds
    span = np.maximum(np.asarray(maxs[:3], dtype=float) - np.asarray(mins[:3], dtype=float), 1e-12)
    solids = [
        c for c in (
            _cylinder_actual_radii(cylinder, span)
            for cylinder in (solid_cylinders or [])
        )
        if c is not None
    ]
    voids = [
        c for c in (
            _cylinder_actual_radii(cylinder, span)
            for cylinder in (void_cylinders or [])
        )
        if c is not None
    ]

    lengths: List[float] = []
    for cylinder in solids + voids:
        _, _, _, _, _, r0, r1 = cylinder
        lengths.extend([2.0 * r0, 2.0 * r1])

    center_tol = 1e-4
    for solid in solids:
        s_axis, s_c0, s_c1, s_lo, s_hi, s_r0, s_r1 = solid
        for void in voids:
            v_axis, v_c0, v_c1, v_lo, v_hi, v_r0, v_r1 = void
            if s_axis != v_axis:
                continue
            if abs(s_c0 - v_c0) > center_tol or abs(s_c1 - v_c1) > center_tol:
                continue
            if min(s_hi, v_hi) <= max(s_lo, v_lo):
                continue
            gap = min(abs(s_r0 - v_r0), abs(s_r1 - v_r1))
            if gap > 0.0:
                lengths.append(gap)
    return lengths


def _initial_design_density(
    nelx: int,
    nely: int,
    nelz: int,
    volfrac: float,
    design_domain: Optional[np.ndarray],
) -> np.ndarray:
    density = np.full(
        (max(1, int(nelx)), max(1, int(nely)), max(1, int(nelz))),
        float(volfrac),
        dtype=float,
    )
    if design_domain is not None:
        mask = np.asarray(design_domain, dtype=bool)
        if mask.shape == density.shape:
            density[~mask] = 1e-3
    return density


def _source_material_fraction(
    density: np.ndarray,
    design_domain: Optional[np.ndarray],
) -> float:
    rho = np.asarray(density, dtype=float)
    if design_domain is not None:
        mask = np.asarray(design_domain, dtype=bool)
        if mask.shape == rho.shape and np.any(mask):
            return float(np.mean(rho[mask]))
    return float(np.mean(rho)) if rho.size else 0.0


def _source_volume_fraction(
    density: np.ndarray,
    design_domain: Optional[np.ndarray],
) -> float:
    rho = np.asarray(density, dtype=float)
    if design_domain is not None:
        mask = np.asarray(design_domain, dtype=bool)
        if mask.shape == rho.shape and mask.size:
            return float(np.mean(mask))
    return 1.0 if rho.size else 0.0


def _effective_density_cutoff(cutoff: Any) -> float:
    """Use the saved threshold consistently for preview, recovery, and export."""
    try:
        cutoff_value = float(cutoff)
    except Exception:
        cutoff_value = 0.45
    return float(np.clip(cutoff_value, 0.01, 0.95))


@numba.njit(cache=True)
def _numba_voxelize_tets(
    pts: np.ndarray, tets: np.ndarray, mins: np.ndarray, sub_step: np.ndarray,
    nelx: int, nely: int, nelz: int, samples: int,
    xs: np.ndarray, ys: np.ndarray, zs: np.ndarray, active_samples: np.ndarray
):
    tol = 1e-9
    nx = nelx * samples
    ny = nely * samples
    nz = nelz * samples

    for i in range(len(tets)):
        tet = tets[i]
        if tet[0] < 0 or tet[1] < 0 or tet[2] < 0 or tet[3] < 0: continue
        if tet[0] >= len(pts) or tet[1] >= len(pts) or tet[2] >= len(pts) or tet[3] >= len(pts): continue

        v0 = pts[tet[0]]
        v1 = pts[tet[1]]
        v2 = pts[tet[2]]
        v3 = pts[tet[3]]

        lo_x = min(v0[0], v1[0], v2[0], v3[0]) - tol
        lo_y = min(v0[1], v1[1], v2[1], v3[1]) - tol
        lo_z = min(v0[2], v1[2], v2[2], v3[2]) - tol

        hi_x = max(v0[0], v1[0], v2[0], v3[0]) + tol
        hi_y = max(v0[1], v1[1], v2[1], v3[1]) + tol
        hi_z = max(v0[2], v1[2], v2[2], v3[2]) + tol

        sx_start = int(np.ceil((lo_x - mins[0]) / sub_step[0] - 0.5))
        sx_stop  = int(np.floor((hi_x - mins[0]) / sub_step[0] - 0.5)) + 1
        sy_start = int(np.ceil((lo_y - mins[1]) / sub_step[1] - 0.5))
        sy_stop  = int(np.floor((hi_y - mins[1]) / sub_step[1] - 0.5)) + 1
        sz_start = int(np.ceil((lo_z - mins[2]) / sub_step[2] - 0.5))
        sz_stop  = int(np.floor((hi_z - mins[2]) / sub_step[2] - 0.5)) + 1

        sx_start = max(sx_start, 0)
        sy_start = max(sy_start, 0)
        sz_start = max(sz_start, 0)
        sx_stop = min(sx_stop, nx)
        sy_stop = min(sy_stop, ny)
        sz_stop = min(sz_stop, nz)

        if sx_stop <= sx_start or sy_stop <= sy_start or sz_stop <= sz_start:
            continue

        mat = np.empty((3, 3), dtype=np.float64)
        mat[0, 0] = v1[0] - v0[0]; mat[0, 1] = v2[0] - v0[0]; mat[0, 2] = v3[0] - v0[0]
        mat[1, 0] = v1[1] - v0[1]; mat[1, 1] = v2[1] - v0[1]; mat[1, 2] = v3[1] - v0[1]
        mat[2, 0] = v1[2] - v0[2]; mat[2, 1] = v2[2] - v0[2]; mat[2, 2] = v3[2] - v0[2]

        # Skip degenerate tets explicitly. Numba nopython try/except only
        # catches user-raised exceptions; np.linalg.inv on a singular matrix
        # can return inf/nan rather than raising, so the determinant check is
        # the only reliable guard.
        det = (mat[0, 0] * (mat[1, 1] * mat[2, 2] - mat[1, 2] * mat[2, 1])
               - mat[0, 1] * (mat[1, 0] * mat[2, 2] - mat[1, 2] * mat[2, 0])
               + mat[0, 2] * (mat[1, 0] * mat[2, 1] - mat[1, 1] * mat[2, 0]))
        if abs(det) < 1e-18:
            continue

        try:
            inv = np.linalg.inv(mat)
        except Exception:
            continue

        for ix in range(sx_start, sx_stop):
            qx = xs[ix] - v0[0]
            vx = ix // samples
            lx = ix % samples
            for iy in range(sy_start, sy_stop):
                qy = ys[iy] - v0[1]
                vy = iy // samples
                ly = iy % samples
                for iz in range(sz_start, sz_stop):
                    qz = zs[iz] - v0[2]
                    vz = iz // samples
                    lz = iz % samples

                    b1 = inv[0, 0]*qx + inv[0, 1]*qy + inv[0, 2]*qz
                    b2 = inv[1, 0]*qx + inv[1, 1]*qy + inv[1, 2]*qz
                    b3 = inv[2, 0]*qx + inv[2, 1]*qy + inv[2, 2]*qz
                    b0 = 1.0 - b1 - b2 - b3

                    if (b0 >= -1e-8 and b1 >= -1e-8 and b2 >= -1e-8 and b3 >= -1e-8 and
                        b0 <= 1.0+1e-8 and b1 <= 1.0+1e-8 and b2 <= 1.0+1e-8 and b3 <= 1.0+1e-8):
                        sub = (lx * samples + ly) * samples + lz
                        active_samples[vx, vy, vz, sub] = True

def _mesh_design_domain_grid(
    mesh: Any,
    bounds: Optional[Tuple[np.ndarray, np.ndarray]],
    nelx: int,
    nely: int,
    nelz: int,
) -> Optional[np.ndarray]:
    """Voxelize the actual tetra mesh volume into the optimizer grid.

    This keeps FreeCAD cutouts and holes as voids automatically instead of
    treating the whole bounding box as designable material.
    """
    if mesh is None or bounds is None or not hasattr(mesh, 'p') or not hasattr(mesh, 't'):
        return None
    try:
        points = np.asarray(mesh.p, dtype=float)
        cells = np.asarray(mesh.t, dtype=int)
    except Exception:
        return None
    if points.ndim != 2 or points.shape[0] < 3 or points.shape[1] == 0:
        return None
    if cells.ndim != 2 or cells.shape[0] < 4 or cells.shape[1] == 0:
        return None

    nelx, nely, nelz = max(1, int(nelx)), max(1, int(nely)), max(1, int(nelz))
    mins, maxs = bounds
    mins = np.asarray(mins[:3], dtype=float)
    maxs = np.asarray(maxs[:3], dtype=float)
    span = np.maximum(maxs - mins, 1e-12)
    step = span / np.asarray([nelx, nely, nelz], dtype=float)

    n_cells = nelx * nely * nelz
    if n_cells <= 75_000:
        samples = 5
    elif n_cells <= 150_000:
        samples = 4
    else:
        samples = 3
    core_threshold = 0.45
    sub_step = step / float(samples)
    xs = mins[0] + (np.arange(nelx * samples, dtype=float) + 0.5) * sub_step[0]
    ys = mins[1] + (np.arange(nely * samples, dtype=float) + 0.5) * sub_step[1]
    zs = mins[2] + (np.arange(nelz * samples, dtype=float) + 0.5) * sub_step[2]
    active_samples = np.zeros((nelx, nely, nelz, samples ** 3), dtype=bool)

    pts = points[:3].T
    tets = cells[:4].T

    _numba_voxelize_tets(
        pts, tets, mins, sub_step, nelx, nely, nelz, samples,
        xs, ys, zs, active_samples
    )

    occupancy = np.mean(active_samples, axis=3)
    core = occupancy >= core_threshold
    touched = occupancy > 0.0

    # Preserve boundary cells that genuinely intersect the source body while
    # rejecting isolated one-sample slivers away from the coherent volume.
    if np.any(core):
        try:
            import scipy.ndimage as ndi
            near_core = ndi.binary_dilation(
                core,
                structure=np.ones((3, 3, 3), dtype=bool),
                iterations=1,
            )
            active = core | (touched & near_core)
        except Exception:
            min_sample_fraction = max(1.0 / float(samples ** 3), 0.08)
            active = occupancy >= min_sample_fraction
    else:
        active = touched

    if not np.any(active):
        logger.warning(
            "TopologyOptVoxelNode: source mesh produced an empty voxel design domain; "
            "continuing with passive contact regions only."
        )
    return active


# ---------------------------------------------------------------------------
# CadQueryNode
# ---------------------------------------------------------------------------

class TopologyOptVoxelNode(CadQueryNode):
    """3-D SIMP topology optimisation using pyMOTO on a structured voxel grid.

    The graph-facing inputs match the normal FEM solver flow: mesh, material,
    constraints, and loads. Solver resolution and optimiser controls live as
    node properties instead of cluttering the graph with hyperparameter ports.
    """
    __identifier__ = 'com.cad.sim.topopt_voxel'
    NODE_NAME = 'Topology Opt (3D Voxel)'

    def __init__(self):
        super().__init__()
        self.add_output('result', color=(180, 255, 180))
        self.add_output('recovered_shape', color=(100, 255, 100))
        self.add_input('mesh',        color=(200, 100, 200))
        self.add_input('material',    color=(200, 200, 200))
        self.add_input('constraints', color=(255, 100, 100), multi_input=True)
        self.add_input('loads',       color=(255, 255,   0), multi_input=True)
        # Guided workflow
        self.create_property('workflow_mode',
                             'Guided',
                             widget_type='combo',
                             items=list(INDUSTRIAL_WORKFLOW_MODES))
        self.create_property('design_goal',
                             'Lightweight Stiffness',
                             widget_type='combo',
                             items=list(INDUSTRIAL_DESIGN_GOALS))
        # Legacy saved studies may still carry a quality_preset. Guided mode
        # now chooses a single automatic feature-aware grid instead.
        self.create_property('quality_preset', 'Automatic', widget_type='string')
        self.create_property('manufacturing_process',
                             'None',
                             widget_type='combo',
                             items=list(INDUSTRIAL_MANUFACTURING_PROCESSES))
        self.create_property('advanced_settings_visible', False, widget_type='bool')
        self.create_property('validate_after_optimize', False, widget_type='bool')
        self.create_property('validation_quality',
                             'Standard',
                             widget_type='combo',
                             items=['Standard', 'Mesh Convergence'])
        self.create_property('generate_cad_after_optimize', False, widget_type='bool')
        self.create_property('cad_reconstruction_method',
                             'Recovered Shape',
                             widget_type='combo',
                             items=['Recovered Shape'])
        self.create_property('cad_export_filename',
                             'topology_optimized.step',
                             widget_type='string')
        # ── Domain ────────────────────────────────────────────────────────
        self.create_property('nelx',    30,    widget_type='int')
        self.create_property('nely',    20,    widget_type='int')
        self.create_property('nelz',    10,    widget_type='int')
        self.create_property('volfrac', 0.5,   widget_type='float')
        self.create_property('rmin',    1.5,   widget_type='float')
        self.create_property('penal',   3.0,   widget_type='float')
        self.create_property('density_cutoff', 0.45, widget_type='float')
        self.create_property('visualization', 'Density', widget_type='combo',
                             items=['Density', 'Recovered Shape'])
        # ── STL post-processing ───────────────────────────────────────────
        # When True, applies the trimesh print-ready pipeline (hole fill +
        # light Humphrey smoothing + optional decimation) on top of marching cubes.
        self.create_property('print_ready_mesh', False, widget_type='bool')
        # 1.0 = no decimation; lower values reduce face count (requires
        # `fast_simplification` to be installed — silently skipped otherwise).
        self.create_property('mesh_decimate_ratio', 1.0, widget_type='float')
        self.create_property('E0',      1.0,   widget_type='float')
        self.create_property('Emin',    1e-9,  widget_type='float')
        self.create_property('nu',      0.3,   widget_type='float')

        # ── Solver ────────────────────────────────────────────────────────
        self.create_property('optimizer', 'OC', widget_type='combo',
                             items=['OC', 'MMA'])
        self.create_property('max_iter', 80,   widget_type='int')
        # `tol` is the relative-compliance-change convergence threshold; the run
        # stops once it holds for `convergence_patience` consecutive iterations.
        self.create_property('tol',      0.01, widget_type='float')
        self.create_property('convergence_patience', 5, widget_type='int')

        # ── Stress constraint (engineering surface only) ──────────────────
        # User sees: Enable + Yield Stress.  The qp-approach exponent q,
        # PNorm exponent, and multi-LC aggregation are silent industrial
        # defaults inside the solver (q=0.5 Bruggi 2008, p=8.0 PNorm).
        self.create_property('stress_constraint', False, widget_type='bool')
        self.create_property('yield_stress',      1.0,   widget_type='float')

        # Legacy saved graphs may still carry this property. It is ignored:
        # graph-connected Constraint/Load nodes and explicit fields are the
        # only sources of topology-optimization boundary conditions.
        self.create_property('bc_preset', 'Custom', widget_type='string')

        # ── Manufacturing constraints (Phase 2) ───────────────────────────
        # `symmetry`: subset of {X,Y,Z} meaning "axes to mirror across the
        # domain centre".  E.g. 'Y' = mirror plane XZ (left-right symmetric);
        # 'XY' = mirror both X and Y.
        self.create_property('symmetry',
                             'None',
                             widget_type='combo',
                             items=['None', 'X', 'Y', 'Z', 'XY', 'XZ', 'YZ', 'XYZ'])
        self.create_property('extrusion',
                             'None',
                             widget_type='combo',
                             items=['None', 'X', 'Y', 'Z'])
        # AM build direction.  '+Y' means parts grow upward in +Y; voxels are
        # only allowed if supported by a 3×3 neighbourhood of denser voxels
        # in the layer below (45° self-supporting rule).
        self.create_property('overhang_build_axis',
                             'None',
                             widget_type='combo',
                             items=['None', '+X', '-X', '+Y', '-Y', '+Z', '-Z'])
        # Max member size — radius in voxels (0 = disabled); local-mean
        # threshold defaults to 0.6 and isn't user-exposed (industry default).
        self.create_property('max_member_size_voxels', 0.0, widget_type='float')
        # N-fold rotational pattern. 1 = disabled.
        self.create_property('pattern_repeat', 1, widget_type='int')
        self.create_property('pattern_axis',
                             'Y',
                             widget_type='combo',
                             items=['X', 'Y', 'Z'])

        # ── Face supports ─────────────────────────────────────────────────
        _support_items = ['None', 'Fix X', 'Fix Y', 'Fix Z',
                          'Fix XY', 'Fix YZ', 'Fix XZ', 'Fix XYZ']
        self.create_property('left_support',   'Fix XYZ', widget_type='combo', items=_support_items)
        self.create_property('right_support',  'None',    widget_type='combo', items=_support_items)
        self.create_property('top_support',    'None',    widget_type='combo', items=_support_items)
        self.create_property('bottom_support', 'None',    widget_type='combo', items=_support_items)
        self.create_property('front_support',  'None',    widget_type='combo', items=_support_items)
        self.create_property('back_support',   'None',    widget_type='combo', items=_support_items)
        self.create_property('support_regions', '[]', widget_type='text')

        # ── Non-design (passive) regions ──────────────────────────────────
        # JSON list of {"x":[a,b],"y":[a,b],"z":[a,b]} in fractional coords.
        # Voxels inside solid_regions are clamped to ρ=1, void_regions to ρ≈0.
        self.create_property('solid_regions', '[]', widget_type='text')
        self.create_property('void_regions',  '[]', widget_type='text')

        # ── Multi-load case JSON ──────────────────────────────────────────
        # When non-empty, overrides the single-force properties below.
        # Format:
        #   [{"name":"LC1","weight":1.0,
        #     "point_forces":[{"x":1.0,"y":0.5,"z":0.5,"fx":0,"fy":-1,"fz":0}],
        #     "distributed_forces":[{"face":"right","fx":0,"fy":-1,"fz":0}]}]
        self.create_property('load_cases', '[]', widget_type='text')

        # ── Force ─────────────────────────────────────────────────────────
        self.create_property('force_type', 'Point', widget_type='combo',
                             items=['Point', 'Distributed Face'])
        self.create_property('force_face', 'Right', widget_type='combo',
                             items=['Left', 'Right', 'Top', 'Bottom', 'Front', 'Back'])
        # Point force location as fractions of domain size [0, 1]
        self.create_property('force_ix_frac',   1.0,  widget_type='float')
        self.create_property('force_iy_frac',   0.5,  widget_type='float')
        self.create_property('force_iz_frac',   0.5,  widget_type='float')
        self.create_property('force_dir_x',     0.0,  widget_type='float')
        self.create_property('force_dir_y',    -1.0,  widget_type='float')
        self.create_property('force_dir_z',     0.0,  widget_type='float')
        self.create_property('force_magnitude', 1.0,  widget_type='float')

    # ── Property helpers ───────────────────────────────────────────────────

    def apply_guided_defaults(self) -> Dict[str, Any]:
        """Apply guided workflow defaults to this node and return the changes."""
        settings = industrial_topopt_defaults(
            self.get_property('design_goal'),
            'Automatic',
            self.get_property('manufacturing_process'),
            nelx=self.get_property('nelx') or 30,
            nely=self.get_property('nely') or 20,
            nelz=self.get_property('nelz') or 10,
        )
        for key, value in settings.items():
            self.set_property(key, value)
        return settings

    def _run_embedded_validation(
        self,
        topo_output: Dict[str, Any],
        material: Dict[str, Any],
        constraint_list: List[Any],
        load_list: List[Any],
    ) -> Optional[Dict[str, Any]]:
        """Run the validation stage as an internal TopOpt study action."""
        from .validation import run_topopt_validation

        quality = str(self.get_property('validation_quality') or 'Standard')
        return run_topopt_validation(
            topo_output,
            material,
            constraint_list,
            load_list,
            convergence_levels=2 if quality == 'Mesh Convergence' else 1,
            max_validation_elements=500000,
            run_external_solver=True,
            deck_only=False,
            analysis_type='Linear',
            visualization='Von Mises Stress',
            deformation_scale='Auto',
        )

    def _run_embedded_cad_reconstruction(
        self,
        topo_output: Dict[str, Any],
    ) -> Any:
        """Build a CAD body as an internal TopOpt study action."""
        from .cad_reconstruction import reconstruct_topopt_cad

        self.set_property('cad_reconstruction_method', 'Recovered Shape')
        topo_payload = dict(topo_output or {})
        return reconstruct_topopt_cad(
            topo_payload,
            source_geometry='Recovered Shape',
            sew_tolerance=1e-4,
            merge_angle_deg=0.0,
        )

    def _build_bc(self) -> VoxelBC:
        """Construct a VoxelBC from the current node properties."""
        bc = VoxelBC(
            fixed_left_face_dofs   = _parse_support(self.get_property('left_support')),
            fixed_right_face_dofs  = _parse_support(self.get_property('right_support')),
            fixed_top_face_dofs    = _parse_support(self.get_property('top_support')),
            fixed_bottom_face_dofs = _parse_support(self.get_property('bottom_support')),
            fixed_front_face_dofs  = _parse_support(self.get_property('front_support')),
            fixed_back_face_dofs   = _parse_support(self.get_property('back_support')),
        )

        regions_text = str(self.get_property('support_regions') or '').strip()
        if regions_text:
            try:
                regions = json.loads(regions_text)
                if not isinstance(regions, list):
                    raise ValueError("support_regions must be a JSON list")
                for region in regions:
                    if not isinstance(region, dict):
                        continue
                    x0, x1 = region.get('x', [0.0, 0.0])
                    y0, y1 = region.get('y', [0.0, 0.0])
                    z0, z1 = region.get('z', [0.0, 1.0])
                    dofs = _parse_support_region_dofs(region.get('dofs', ''))
                    bc.fixed_boxes.append((
                        float(x0), float(x1),
                        float(y0), float(y1),
                        float(z0), float(z1),
                        dofs,
                    ))
            except Exception as exc:
                raise ValueError(f"Invalid support_regions JSON: {exc}") from exc

        # ── non-design regions ────────────────────────────────────────────
        bc.solid_boxes = _parse_region_boxes(
            self.get_property('solid_regions'), 'solid_regions'
        )
        bc.solid_cylinders = _parse_region_cylinders(
            self.get_property('solid_regions'), 'solid_regions'
        )
        bc.void_boxes = _parse_region_boxes(
            self.get_property('void_regions'),  'void_regions'
        )
        bc.void_cylinders = _parse_region_cylinders(
            self.get_property('void_regions'),  'void_regions'
        )

        # ── multi-load case (overrides legacy single-force if present) ────
        bc.load_cases = _parse_load_cases(self.get_property('load_cases'))
        if bc.load_cases:
            return bc

        # ── legacy single-force mode ──────────────────────────────────────
        mag  = float(self.get_property('force_magnitude') or 1.0)
        fdx  = float(self.get_property('force_dir_x') or 0.0)
        fdy  = float(self.get_property('force_dir_y') or -1.0)
        fdz  = float(self.get_property('force_dir_z') or 0.0)

        norm = (fdx ** 2 + fdy ** 2 + fdz ** 2) ** 0.5 or 1.0
        fdx, fdy, fdz = (fdx / norm) * mag, (fdy / norm) * mag, (fdz / norm) * mag

        ftype = self.get_property('force_type')
        if ftype == 'Point':
            ix_f = float(self.get_property('force_ix_frac') or 1.0)
            iy_f = float(self.get_property('force_iy_frac') or 0.5)
            iz_f = float(self.get_property('force_iz_frac') or 0.5)
            bc.point_forces.append((ix_f, iy_f, iz_f, fdx, fdy, fdz))
        else:  # Distributed Face
            face = (self.get_property('force_face') or 'Right').lower()
            bc.distributed_forces.append((face, fdx, fdy, fdz))

        return bc

    def _merge_graph_bcs(
        self,
        bc: VoxelBC,
        mesh: Any,
        constraints: List[Any],
        loads: List[Any],
    ) -> None:
        """Add ordinary PyLCSS face constraints and loads to the voxel BCs."""
        bounds = _mesh_bounds(mesh)
        if bounds is None:
            if constraints or loads:
                logger.warning(
                    "TopologyOptVoxelNode: cannot map graph BCs without mesh bounds."
                )
            return

        for constraint in constraints:
            if not isinstance(constraint, dict):
                continue
            dofs = constraint.get('fixed_dofs') or []
            try:
                dofs = [int(d) for d in dofs if int(d) in (0, 1, 2)]
            except Exception:
                dofs = []
            if not dofs:
                continue
            for face in constraint.get('geometries') or []:
                cylinder = _cylinder_void_region_from_face(face, bounds)
                if cylinder is not None:
                    _add_cylindrical_contact_region(bc, cylinder)
            bboxes = _entry_bboxes(constraint)
            if not bboxes:
                logger.warning(
                    "TopologyOptVoxelNode: connected constraint did not "
                    "include face geometry or bbox metadata; it cannot be "
                    "mapped to voxel supports."
                )
                continue
            for bbox in bboxes:
                frac_box = _fraction_box(bbox, bounds)
                bc.fixed_boxes.append((*frac_box, dofs))

        for load in loads:
            if not isinstance(load, dict):
                continue
            load_type = str(load.get('type') or '').lower()
            if load_type == 'force':
                vector = load.get('vector')
                try:
                    fx, fy, fz = (float(vector[0]), float(vector[1]), float(vector[2]))
                except Exception:
                    continue
                for face in _iter_load_faces(load):
                    cylinder = _cylinder_void_region_from_face(face, bounds)
                    if cylinder is not None:
                        _add_cylindrical_contact_region(bc, cylinder)
                bboxes = _entry_bboxes(load)
                if not bboxes:
                    logger.warning(
                        "TopologyOptVoxelNode: connected force load did not "
                        "include face geometry or bbox metadata; it cannot be "
                        "mapped to voxel forces."
                    )
                    continue
                scale = 1.0 / max(1, len(bboxes))
                for bbox in bboxes:
                    frac_box = _fraction_box(bbox, bounds)
                    # Graph-selected load faces become distributed patch loads,
                    # not mathematical point loads.  The same patch is also kept
                    # local so CAD face loads stay tied to their source face.
                    bc.box_forces.append((*frac_box, fx * scale, fy * scale, fz * scale))
            elif load_type == 'pressure':
                for face in _iter_load_faces(load):
                    cylinder = _cylinder_void_region_from_face(face, bounds)
                    if cylinder is not None:
                        _add_cylindrical_contact_region(bc, cylinder)
                pressure_patches = _pressure_load_patches(load, bounds)
                if not pressure_patches:
                    logger.warning(
                        "TopologyOptVoxelNode: pressure load could not be "
                        "mapped to voxel patch forces."
                    )
                    continue
                for frac_box, fx, fy, fz in pressure_patches:
                    bc.box_forces.append((*frac_box, fx, fy, fz))
            else:
                logger.debug(
                    "TopologyOptVoxelNode: load type %r is not supported by "
                    "the voxel optimiser.",
                    load_type,
                )

    # ── Node run ───────────────────────────────────────────────────────────

    def run(self, progress_callback=None) -> Optional[Dict[str, Any]]:
        mesh = self.get_input_value('mesh', None)
        bounds = _mesh_bounds(mesh)
        material = self.get_input_value('material', None)
        material = material if isinstance(material, dict) else {}
        raw_constraints = _flatten(self.get_input_list('constraints'))
        constraint_list: List[Any] = []
        load_like_constraints: List[Any] = []
        for item in raw_constraints:
            if _is_load_payload(item):
                load_like_constraints.append(item)
            else:
                constraint_list.append(item)
        if load_like_constraints:
            logger.warning(
                "TopologyOptVoxelNode: %d load payload(s) were connected to "
                "the constraints input; treating them as loads.",
                len(load_like_constraints),
            )
        load_list = _flatten(self.get_input_list('loads')) + load_like_constraints
        bc = self._build_bc()
        if constraint_list:
            bc.fixed_left_face_dofs = []
            bc.fixed_right_face_dofs = []
            bc.fixed_top_face_dofs = []
            bc.fixed_bottom_face_dofs = []
            bc.fixed_front_face_dofs = []
            bc.fixed_back_face_dofs = []
            bc.fixed_boxes = []
        if load_list:
            bc.point_forces = []
            bc.box_forces = []
            bc.distributed_forces = []
            bc.load_cases = []  # graph-supplied loads override node-property load cases
        self._merge_graph_bcs(bc, mesh, constraint_list, load_list)
        _add_bc_contact_regions(bc)
        if constraint_list and not _bc_has_support(bc):
            self.set_error(
                "Connected constraints could not be mapped to voxel supports. "
                "Re-run the face selection on the current FreeCAD shape, then "
                "run TopOpt again."
            )
            return None
        if load_list and not _bc_has_nonzero_load(bc):
            self.set_error(
                "Connected loads could not be mapped to a non-zero voxel force. "
                "Check the selected load face and force components, then run "
                "TopOpt again."
            )
            return None

        nelx = int(self.get_property('nelx') or 30)
        nely = int(self.get_property('nely') or 20)
        nelz = int(self.get_property('nelz') or 10)
        guided_active = _use_automatic_guided_grid(
            self.get_property('workflow_mode'),
            self.get_property('quality_preset'),
        )
        if guided_active:
            guided_grid = _guided_voxel_grid(
                bounds,
                'Automatic',
                feature_bboxes=_bc_feature_bboxes(constraint_list, load_list),
                feature_lengths=_cylinder_feature_lengths(
                    bounds,
                    solid_cylinders=bc.solid_cylinders,
                    void_cylinders=bc.void_cylinders,
                ),
            )
            if guided_grid is not None:
                nelx, nely, nelz = guided_grid

        rmin_effective = float(self.get_property('rmin') or 1.5)
        if guided_active:
            rmin_effective = _guided_rmin(nelx, nely, nelz)
        design_domain = _mesh_design_domain_grid(mesh, bounds, nelx, nely, nelz)
        unitx = unity = unitz = 1.0
        if bounds is not None:
            mins, maxs = bounds
            span = np.maximum(maxs[:3] - mins[:3], 1e-12)
            unitx = float(span[0] / max(nelx, 1))
            unity = float(span[1] / max(nely, 1))
            unitz = float(span[2] / max(nelz, 1))

        mc = ManufacturingConstraints(
            symmetry            = (str(self.get_property('symmetry') or 'None')).lower(),
            extrusion           = (str(self.get_property('extrusion') or 'None')).lower(),
            overhang_build_axis = (str(self.get_property('overhang_build_axis') or 'None')).lower(),
            max_member_size_voxels = float(self.get_property('max_member_size_voxels') or 0.0),
            pattern_repeat = int(self.get_property('pattern_repeat') or 1),
            pattern_axis   = (str(self.get_property('pattern_axis') or 'Y')).lower(),
        )

        problem = TopologyOptVoxelProblem(
            nelx     = nelx,
            nely     = nely,
            nelz     = nelz,
            E0       = float(material.get('E', self.get_property('E0') or 1.0)),
            Emin     = float(self.get_property('Emin')  or 1e-9),
            nu       = float(material.get('nu', self.get_property('nu') or 0.3)),
            penal    = float(self.get_property('penal') or 3.0),
            volfrac  = float(self.get_property('volfrac') or 0.5),
            rmin     = rmin_effective,
            unitx    = unitx,
            unity    = unity,
            unitz    = unitz,
            optimizer = self.get_property('optimizer') or 'OC',
            max_iter = int(self.get_property('max_iter') or 80),
            tol      = float(self.get_property('tol')   or 0.01),
            patience = int(self.get_property('convergence_patience') or 5),
            bc       = bc,
            mc       = mc,
            design_domain = design_domain,
            stress_constraint_enabled = bool(self.get_property('stress_constraint')),
            yield_stress              = float(self.get_property('yield_stress') or 1.0),
            # Numerical hyperparameters use the dataclass defaults (industrial
            # values: q=0.5 Bruggi qp-approach, p=8.0 PNorm aggregation,
            # Heaviside three-field SIMP on with β: 1 → 16 stepping every 30
            # iters, η=0.5).  These are NOT user knobs in industrial codes.
        )

        def _preview_payload(density: np.ndarray, stage: Optional[str] = None) -> Dict[str, Any]:
            density_cutoff = _effective_density_cutoff(
                self.get_property('density_cutoff') or 0.45
            )
            payload = {
                'type': 'topopt_voxel',
                'density': density,
                'design_domain': design_domain,
                'grid_shape': density.shape,
                'bounds': _bounds_payload(bounds),
                'density_cutoff': density_cutoff,
                'target_vol_frac': problem.volfrac,
                'final_vol_frac': _source_material_fraction(density, design_domain),
                'bounding_vol_frac': float(np.mean(density)) if density.size else 0.0,
                '_preview': True,
            }
            if stage:
                payload['stage'] = stage
            return payload

        def _emit_preview(
            density: np.ndarray,
            step: int,
            total: int,
            stage: Optional[str] = None,
        ) -> None:
            if progress_callback is not None:
                try:
                    progress_callback(
                        _preview_payload(density, stage=stage),
                        density,
                        max(0, int(step)),
                        max(1, int(total)),
                    )
                except Exception:
                    pass

        _emit_preview(
            _initial_design_density(nelx, nely, nelz, problem.volfrac, design_domain),
            0,
            problem.max_iter,
            stage='Design domain preview',
        )

        solver = TopologyOptVoxelSolver(problem)

        def _cb(it: int, comp: float, change: float, density: np.ndarray) -> None:
            _emit_preview(density, max(0, it - 1), problem.max_iter)

        try:
            result = solver.run(callback=_cb)
        except Exception as exc:
            logger.exception("TopologyOptVoxelNode: solver error")
            self.set_error(str(exc))
            return None

        logger.info("TopologyOptVoxelNode: %s", result.message)
        density = np.asarray(result.density, dtype=float)
        density_cutoff = _effective_density_cutoff(
            self.get_property('density_cutoff') or 0.45
        )
        print_ready = bool(self.get_property('print_ready_mesh'))
        decimate    = float(self.get_property('mesh_decimate_ratio') or 1.0)
        # Key on the actual density bytes — not id(result.density), which Python's
        # allocator can reuse across runs and silently return a stale recovered
        # shape for a different solve.
        density_view = np.ascontiguousarray(result.density)
        cache_key = (
            hash(density_view.tobytes()),
            density_view.shape,
            density_cutoff,
            print_ready,
            decimate,
            str(mc.extrusion),
            str(bc.solid_boxes),
            str(bc.void_boxes),
            str(bc.solid_cylinders),
            str(bc.void_cylinders),
        )
        if getattr(self, '_last_recovery_key', None) == cache_key:
            recovered = self._last_recovered_shape
        else:
            recovered = _recover_voxel_shape(
                density, bounds, density_cutoff,
                print_ready=print_ready,
                decimate_ratio=decimate,
                solid_boxes=bc.solid_boxes,
                void_boxes=bc.void_boxes,
                solid_cylinders=bc.solid_cylinders,
                void_cylinders=bc.void_cylinders,
                extrusion_axis=mc.extrusion,
                source_mask=design_domain,
            )
            self._last_recovery_key = cache_key
            self._last_recovered_shape = recovered
        if bounds is not None:
            mins, maxs = bounds
            total_volume = float(np.prod(np.maximum(maxs[:3] - mins[:3], 0.0)))
        else:
            total_volume = float(np.prod(density.shape))
        source_volume_fraction = _source_volume_fraction(density, design_domain)
        source_volume = float(total_volume * source_volume_fraction)
        source_material_fraction = _source_material_fraction(density, design_domain)
        final_volume = float(source_material_fraction * source_volume)
        material_density = float(material.get('rho', material.get('density', 0.0)))
        output: Dict[str, Any] = {
            'type': 'topopt_voxel',
            'density': density,
            'design_domain': design_domain,
            'design_density': (
                np.asarray(result.design_density, dtype=float)
                if result.design_density is not None else None
            ),
            'grid_shape': density.shape,
            'bounds': _bounds_payload(bounds),
            'density_cutoff': density_cutoff,
            'recovered_shape': recovered,
            'extrusion_axis': mc.extrusion,
            'visualization_mode': self.get_property('visualization') or 'Density',
            'target_vol_frac': problem.volfrac,
            'final_vol_frac': source_material_fraction,
            'active_target_vol_frac': float(result.active_target_volfrac or 0.0),
            'passive_source_vol_frac': float(result.passive_source_volfrac or 0.0),
            'minimum_source_vol_frac': float(result.min_source_volfrac or 0.0),
            'bounding_vol_frac': float(np.mean(density)) if density.size else 0.0,
            'source_volume_fraction': source_volume_fraction,
            'volume': final_volume,
            'total_volume': total_volume,
            'source_volume': source_volume,
            'mass': final_volume * material_density,
            'compliance': (
                float(result.compliance_history[-1])
                if result.compliance_history else None
            ),
            'stress_pnorm': (
                float(result.stress_history[-1])
                if result.stress_history else None
            ),
            'iterations': result.n_iter,
            'converged': result.converged,
            'message': result.message,
            'compliance_history': result.compliance_history,
            'change_history': result.change_history,
            'stress_history': result.stress_history,
            'passive_regions': {
                'solid_boxes': list(bc.solid_boxes),
                'void_boxes': list(bc.void_boxes),
                'solid_cylinders': list(bc.solid_cylinders),
                'void_cylinders': list(bc.void_cylinders),
            },
        }

        warnings_out: List[str] = []
        if (
            float(result.min_source_volfrac or 0.0)
            > float(problem.volfrac) + 1e-6
        ):
            warnings_out.append(
                "Target volume is below the fixed passive material; final "
                "volume cannot reach the requested fraction unless passive "
                "solid regions are reduced."
            )
        if bool(self.get_property('validate_after_optimize')):
            try:
                validation = self._run_embedded_validation(
                    output, material, constraint_list, load_list,
                )
                if validation is not None:
                    output['validation'] = validation
                    study = validation.get('convergence_study') if isinstance(validation, dict) else None
                    output['validation_summary'] = {
                        'max_stress': validation.get('max_stress_gauss'),
                        'compliance': validation.get('compliance'),
                        'converged': (
                            study.get('converged')
                            if isinstance(study, dict) else None
                        ),
                    }
            except Exception as exc:
                msg = f"Validation skipped/failed: {exc}"
                logger.warning("TopologyOptVoxelNode: %s", msg)
                output['validation_error'] = str(exc)
                warnings_out.append(msg)

        if bool(self.get_property('generate_cad_after_optimize')):
            try:
                cad_shape = self._run_embedded_cad_reconstruction(output)
                output['cad_shape'] = cad_shape
                output['shape'] = cad_shape
                try:
                    solid = cad_shape.val() if hasattr(cad_shape, 'val') else cad_shape
                    output['cad_reconstruction'] = {
                        'method': 'Recovered Shape',
                        'valid': bool(solid.isValid()) if hasattr(solid, 'isValid') else None,
                        'volume': float(solid.Volume()) if hasattr(solid, 'Volume') else None,
                    }
                except Exception:
                    output['cad_reconstruction'] = {
                        'method': 'Recovered Shape',
                    }
            except Exception as exc:
                msg = f"CAD reconstruction failed: {exc}"
                logger.warning("TopologyOptVoxelNode: %s", msg)
                output['cad_error'] = str(exc)
                warnings_out.append(msg)

        if warnings_out:
            output['warnings'] = warnings_out
        return output
