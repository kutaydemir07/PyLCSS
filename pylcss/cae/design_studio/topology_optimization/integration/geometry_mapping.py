# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Geometry, bounds, and thermal payload mapping for topology nodes."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from ..models.study import JointDefinition, ThermalBC, ThermalLoadCase


def _flatten(values: Any) -> list[Any]:
    if values is None:
        return []
    if isinstance(values, (list, tuple)):
        out: list[Any] = []
        for item in values:
            out.extend(_flatten(item))
        return out
    return [values]


def _surface_mesh_arrays(mesh: Any) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """Return row-major vertices/triangles from imported or CAD geometry."""
    if mesh is None:
        return None
    if isinstance(mesh, dict):
        for key in ("mesh", "recovered_shape", "shape", "cad"):
            nested = mesh.get(key)
            if nested is not None and nested is not mesh:
                arrays = _surface_mesh_arrays(nested)
                if arrays is not None:
                    return arrays
        vertices = mesh.get("vertices")
        faces = mesh.get("faces")
        if vertices is not None and faces is not None:
            verts = np.asarray(vertices, dtype=float)
            tris = np.asarray(faces, dtype=int)
            if (
                verts.ndim == 2
                and verts.shape[1] >= 3
                and tris.ndim == 2
                and tris.shape[1] >= 3
            ):
                return verts[:, :3], tris[:, :3]

    if hasattr(mesh, "vertices") and hasattr(mesh, "faces"):
        raw_vertices = mesh.vertices
        raw_faces = mesh.faces
        if not callable(raw_vertices) and not callable(raw_faces):
            verts = np.asarray(raw_vertices, dtype=float)
            tris = np.asarray(raw_faces, dtype=int)
            if (
                verts.ndim == 2
                and verts.shape[1] >= 3
                and tris.ndim == 2
                and tris.shape[1] >= 3
            ):
                return verts[:, :3], tris[:, :3]

    shape = mesh
    if hasattr(shape, "toCompound"):
        try:
            shape = shape.toCompound()
        except Exception:
            pass
    elif hasattr(shape, "val"):
        try:
            shape = shape.val()
        except Exception:
            pass
    if hasattr(shape, "tessellate"):
        try:
            vertices, faces = shape.tessellate(0.5)
            verts = np.asarray(
                [
                    (float(v.x), float(v.y), float(v.z))
                    if hasattr(v, "x")
                    else tuple(v)[:3]
                    for v in vertices
                ],
                dtype=float,
            )
            tris = np.asarray(faces, dtype=int)
            if (
                verts.ndim == 2
                and verts.shape[1] >= 3
                and tris.ndim == 2
                and tris.shape[1] >= 3
            ):
                return verts[:, :3], tris[:, :3]
        except Exception:
            return None
    return None


def _mesh_bounds(mesh: Any) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """Bounds for tetra meshes, imported triangle meshes, or CAD solids."""
    if mesh is None:
        return None
    if isinstance(mesh, dict):
        for key in ("mesh", "recovered_shape", "shape", "cad"):
            nested = mesh.get(key)
            if nested is not None and nested is not mesh:
                bounds = _mesh_bounds(nested)
                if bounds is not None:
                    return bounds
    if hasattr(mesh, "p"):
        pts = np.asarray(mesh.p, dtype=float)
        if pts.ndim == 2 and pts.shape[0] >= 3 and pts.shape[1] > 0:
            return pts[:3].min(axis=1), pts[:3].max(axis=1)

    arrays = _surface_mesh_arrays(mesh)
    if arrays is not None:
        points = arrays[0]
        if len(points):
            return points.min(axis=0), points.max(axis=0)

    shape = mesh
    if hasattr(shape, "toCompound"):
        try:
            shape = shape.toCompound()
        except Exception:
            pass
    elif hasattr(shape, "val"):
        try:
            shape = shape.val()
        except Exception:
            pass
    try:
        bb = shape.BoundingBox()
        return (
            np.asarray([bb.xmin, bb.ymin, bb.zmin], dtype=float),
            np.asarray([bb.xmax, bb.ymax, bb.zmax], dtype=float),
        )
    except Exception:
        return None


def _bbox_tuple(bb: Any) -> Optional[tuple[float, float, float, float, float, float]]:
    if bb is None:
        return None
    if isinstance(bb, dict):
        try:
            return (
                float(bb["xmin"]),
                float(bb["xmax"]),
                float(bb["ymin"]),
                float(bb["ymax"]),
                float(bb["zmin"]),
                float(bb["zmax"]),
            )
        except Exception:
            return None
    try:
        return (
            float(bb.xmin),
            float(bb.xmax),
            float(bb.ymin),
            float(bb.ymax),
            float(bb.zmin),
            float(bb.zmax),
        )
    except Exception:
        return None


def _entry_bboxes(
    entry: dict[str, Any],
) -> list[tuple[float, float, float, float, float, float]]:
    bboxes: list[tuple[float, float, float, float, float, float]] = []
    for face in entry.get("geometries") or []:
        try:
            bb = _bbox_tuple(face.BoundingBox())
        except Exception:
            bb = None
        if bb is not None:
            bboxes.append(bb)

    viz = entry.get("viz") if isinstance(entry.get("viz"), dict) else {}
    for face in viz.get("faces") or []:
        bb = _bbox_tuple(face.get("bbox") if isinstance(face, dict) else None)
        if bb is not None:
            bboxes.append(bb)
    bb = _bbox_tuple(viz.get("bbox"))
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
    bbox: tuple[float, float, float, float, float, float],
    bounds: tuple[np.ndarray, np.ndarray],
    pad: float = 0.02,
) -> tuple[float, float, float, float, float, float]:
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


def _geometry_bbox(
    geometry: Any,
) -> Optional[tuple[float, float, float, float, float, float]]:
    try:
        return _bbox_tuple(geometry.BoundingBox())
    except Exception:
        if isinstance(geometry, dict):
            return _bbox_tuple(geometry.get("bbox"))
        return None


def _anchor_fraction(
    geometries: Any,
    bounds: tuple[np.ndarray, np.ndarray],
) -> tuple[float, float, float]:
    candidates = _flatten(geometries)
    bboxes = [
        bbox
        for bbox in (_geometry_bbox(geometry) for geometry in candidates)
        if bbox is not None
    ]
    if not bboxes:
        raise ValueError(
            "Joint anchors must come from selected CAD faces, edges, or vertices."
        )
    centers = np.asarray(
        [
            [
                0.5 * (bbox[0] + bbox[1]),
                0.5 * (bbox[2] + bbox[3]),
                0.5 * (bbox[4] + bbox[5]),
            ]
            for bbox in bboxes
        ],
        dtype=float,
    )
    center = np.mean(centers, axis=0)
    mins, maxs = bounds
    return tuple(
        _fraction(center[index], mins[index], maxs[index]) for index in range(3)
    )  # type: ignore[return-value]


def _joint_from_graph_payload(
    payload: Any,
    bounds: tuple[np.ndarray, np.ndarray],
) -> JointDefinition:
    if not isinstance(payload, dict) or str(payload.get("type")) != "topology_joint":
        raise ValueError("The joints input accepts TopOpt Joint nodes only.")
    return JointDefinition(
        name=str(payload.get("name") or "Joint"),
        kind=str(payload.get("joint_type") or "spherical"),
        anchor_a=_anchor_fraction(payload.get("anchor_a_geometries"), bounds),
        anchor_b=_anchor_fraction(payload.get("anchor_b_geometries"), bounds),
        axis=str(payload.get("axis") or "x"),
        relative_stiffness=float(payload.get("relative_stiffness") or 100.0),
    )


def _thermal_bc_from_graph_payloads(
    sink_payloads: list[Any],
    heat_payloads: list[Any],
    bounds: tuple[np.ndarray, np.ndarray],
) -> ThermalBC:
    thermal_bc = ThermalBC()
    for payload in sink_payloads:
        if not isinstance(payload, dict):
            continue
        bboxes = _entry_bboxes(payload)
        thermal_bc.fixed_boxes.extend(_fraction_box(bbox, bounds) for bbox in bboxes)

    grouped: dict[str, dict[str, Any]] = {}
    for payload in heat_payloads:
        if not isinstance(payload, dict):
            continue
        name = str(payload.get("name") or "Thermal Case")
        weight = float(payload.get("weight") or 1.0)
        total_heat = float(payload.get("total_heat") or 0.0)
        bboxes = _entry_bboxes(payload)
        if not bboxes:
            continue
        group = grouped.setdefault(
            name,
            {"weight": weight, "box_sources": []},
        )
        share = total_heat / float(len(bboxes))
        group["box_sources"].extend(
            (*_fraction_box(bbox, bounds), share) for bbox in bboxes
        )

    thermal_bc.load_cases = [
        ThermalLoadCase(
            name=name,
            weight=float(values["weight"]),
            box_sources=list(values["box_sources"]),
        )
        for name, values in grouped.items()
    ]
    return thermal_bc
