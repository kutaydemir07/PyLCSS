# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.
"""Topology optimisation export: recovered surface mesh -> stronger STEP B-rep.

This is a drop-in successor to ``cad_reconstruction.py``.  It keeps the same
public reconstruction entry point and node identifier, but improves STEP output
quality by adding:

* tolerance-aware vertex welding before sewing;
* connected-component filtering;
* boundary/non-manifold edge validation;
* consistent triangle orientation and outward winding;
* model-size-relative sewing tolerance;
* final OCC B-rep validation;
* more robust analytic passive feature re-application;
* optional physical-unit CAD feature metadata support.

The result is still intentionally a hybrid model: the topology-optimised organic
mass is represented as a sewed faceted B-rep, while known engineering features
such as sleeves, pads, bores and keep-out cuts are reconstructed analytically so
STEP consumers see real CAD cylinders/boxes where they matter.
"""
from __future__ import annotations

import logging
from collections import defaultdict, deque
from typing import Any, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_RECOVERED_MODE = "Recovered Shape"

CylinderRegion = Tuple[str, float, float, float, float, float, float]
BoxRegion = Tuple[float, float, float, float, float, float]


# ---------------------------------------------------------------------------
# Recovered mesh payload helpers
# ---------------------------------------------------------------------------

def _extract_recovered_mesh(payload: Any) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Return ``(vertices, triangle_faces)`` from a topology result payload."""
    if not isinstance(payload, dict):
        return None

    mesh = payload.get("recovered_shape")
    if mesh is None and "vertices" in payload and "faces" in payload:
        mesh = payload
    if not isinstance(mesh, dict):
        return None

    vertices = mesh.get("vertices")
    faces = mesh.get("faces")
    if vertices is None or faces is None:
        return None

    vertices = np.asarray(vertices, dtype=float)
    faces = np.asarray(faces, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] < 3:
        return None
    if faces.ndim != 2 or faces.shape[1] < 3:
        return None
    if len(vertices) < 4 or len(faces) < 4:
        return None
    return vertices[:, :3], faces[:, :3]


def _drop_degenerate_faces(
    vertices: np.ndarray,
    faces: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Remove invalid, zero-area, repeated-index, and duplicate triangles."""
    vertices = np.asarray(vertices, dtype=float)
    if vertices.ndim != 2 or vertices.shape[1] < 3:
        return np.empty((0, 3), dtype=float), np.empty((0, 3), dtype=np.int64)
    vertices = vertices[:, :3]

    faces = np.asarray(faces, dtype=np.int64)
    if faces.size == 0:
        return vertices, faces.reshape((0, 3))
    if faces.ndim != 2 or faces.shape[1] < 3:
        return vertices, np.empty((0, 3), dtype=np.int64)
    faces = faces[:, :3]

    in_range = np.all((faces >= 0) & (faces < len(vertices)), axis=1)
    faces = faces[in_range]
    if len(faces) == 0:
        return vertices, faces.reshape((0, 3))

    keep = (
        (faces[:, 0] != faces[:, 1])
        & (faces[:, 1] != faces[:, 2])
        & (faces[:, 0] != faces[:, 2])
    )
    faces = faces[keep]
    if len(faces) == 0:
        return vertices, faces.reshape((0, 3))

    tri = vertices[faces]
    finite = np.all(np.isfinite(tri), axis=(1, 2))
    tri = tri[finite]
    faces = faces[finite]
    if len(faces) == 0:
        return vertices, faces.reshape((0, 3))

    areas = 0.5 * np.linalg.norm(
        np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]),
        axis=1,
    )
    faces = faces[areas > 1e-12]
    if len(faces) == 0:
        return vertices, faces.reshape((0, 3))

    keys = np.sort(faces, axis=1)
    _, unique_idx = np.unique(keys, axis=0, return_index=True)
    unique_idx.sort()
    return vertices, faces[unique_idx]


def _bbox_diag(vertices: np.ndarray) -> float:
    """Return the model bounding-box diagonal, clamped to a non-zero value."""
    vertices = np.asarray(vertices, dtype=float)
    if vertices.size == 0:
        return 1.0
    lo = np.nanmin(vertices[:, :3], axis=0)
    hi = np.nanmax(vertices[:, :3], axis=0)
    diag = float(np.linalg.norm(hi - lo))
    return diag if np.isfinite(diag) and diag > 1e-12 else 1.0


def _effective_tolerance(
    vertices: np.ndarray,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> float:
    """Combine user absolute tolerance with model-size-relative tolerance."""
    abs_tol = float(absolute_tolerance or 0.0)
    rel_tol = float(relative_tolerance or 0.0) * _bbox_diag(vertices)
    return max(abs_tol, rel_tol, 1e-9)


def _merge_nearby_vertices(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    tolerance: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Weld vertices that land in the same tolerance grid cell.

    This is deliberately dependency-free.  It is not a full geometric remesher,
    but it fixes the common recovered-mesh problem where adjacent triangles use
    numerically distinct vertices along an otherwise shared edge.
    """
    vertices = np.asarray(vertices, dtype=float)[:, :3]
    faces = np.asarray(faces, dtype=np.int64)[:, :3]
    tol = float(tolerance or 0.0)
    if tol <= 0.0 or len(vertices) == 0 or len(faces) == 0:
        return vertices, faces

    keys = np.round(vertices / tol).astype(np.int64)
    key_to_new: dict[tuple[int, int, int], int] = {}
    inverse = np.empty(len(vertices), dtype=np.int64)
    sums: list[np.ndarray] = []
    counts: list[int] = []

    for old_idx, key_arr in enumerate(keys):
        key = (int(key_arr[0]), int(key_arr[1]), int(key_arr[2]))
        new_idx = key_to_new.get(key)
        if new_idx is None:
            new_idx = len(sums)
            key_to_new[key] = new_idx
            sums.append(vertices[old_idx].copy())
            counts.append(1)
        else:
            sums[new_idx] += vertices[old_idx]
            counts[new_idx] += 1
        inverse[old_idx] = new_idx

    welded = np.vstack([s / max(c, 1) for s, c in zip(sums, counts)]).astype(float)
    remapped_faces = inverse[faces]
    welded, remapped_faces = _drop_degenerate_faces(welded, remapped_faces)
    return welded, remapped_faces


def _edge_occurrences(faces: np.ndarray) -> dict[tuple[int, int], list[tuple[int, int, int]]]:
    """Map undirected edges to ``(face_index, oriented_start, oriented_end)``."""
    occ: dict[tuple[int, int], list[tuple[int, int, int]]] = defaultdict(list)
    for fi, tri in enumerate(np.asarray(faces, dtype=np.int64)[:, :3]):
        a, b, c = [int(v) for v in tri]
        for u, v in ((a, b), (b, c), (c, a)):
            occ[(min(u, v), max(u, v))].append((fi, u, v))
    return occ


def _find_boundary_edges(faces: np.ndarray) -> list[tuple[int, int]]:
    """Return mesh edges used by exactly one triangle."""
    return [edge for edge, items in _edge_occurrences(faces).items() if len(items) == 1]


def _find_nonmanifold_edges(faces: np.ndarray) -> list[tuple[int, int]]:
    """Return mesh edges used by more than two triangles."""
    return [edge for edge, items in _edge_occurrences(faces).items() if len(items) > 2]


def _connected_face_components(faces: np.ndarray) -> list[np.ndarray]:
    """Return connected components of faces using shared triangle edges."""
    faces = np.asarray(faces, dtype=np.int64)[:, :3]
    if len(faces) == 0:
        return []

    adjacency: list[set[int]] = [set() for _ in range(len(faces))]
    for items in _edge_occurrences(faces).values():
        if len(items) < 2:
            continue
        face_ids = [fi for fi, _, _ in items]
        for i, fi in enumerate(face_ids):
            adjacency[fi].update(face_ids[:i])
            adjacency[fi].update(face_ids[i + 1 :])

    seen = np.zeros(len(faces), dtype=bool)
    components: list[np.ndarray] = []
    for start in range(len(faces)):
        if seen[start]:
            continue
        queue: deque[int] = deque([start])
        seen[start] = True
        comp: list[int] = []
        while queue:
            fi = queue.popleft()
            comp.append(fi)
            for nxt in adjacency[fi]:
                if not seen[nxt]:
                    seen[nxt] = True
                    queue.append(nxt)
        components.append(np.asarray(comp, dtype=np.int64))
    return components


def _remove_tiny_components(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    min_faces: int = 8,
) -> Tuple[np.ndarray, np.ndarray]:
    """Drop tiny floating triangle islands while preserving all meaningful bodies."""
    faces = np.asarray(faces, dtype=np.int64)[:, :3]
    components = _connected_face_components(faces)
    if len(components) <= 1:
        return vertices, faces

    min_faces = int(max(min_faces, 1))
    keep_components = [comp for comp in components if len(comp) >= min_faces]
    if not keep_components:
        keep_components = [max(components, key=len)]

    keep_face_ids = np.concatenate(keep_components)
    keep_face_ids.sort()
    logger.info(
        "Recovered Shape mesh repair: kept %d/%d connected component(s).",
        len(keep_components),
        len(components),
    )
    return vertices, faces[keep_face_ids]


def _orient_faces_consistently(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Orient adjacent triangles consistently and make closed volumes outward."""
    vertices = np.asarray(vertices, dtype=float)[:, :3]
    faces = np.asarray(faces, dtype=np.int64)[:, :3].copy()
    if len(faces) == 0:
        return faces

    adjacency: list[list[tuple[int, bool]]] = [[] for _ in range(len(faces))]
    for items in _edge_occurrences(faces).values():
        if len(items) != 2:
            continue
        (f0, a0, b0), (f1, a1, b1) = items
        # For a consistently oriented manifold, neighbouring faces traverse the
        # shared edge in opposite directions.  Same direction means one side has
        # to be flipped relative to the other.
        same_direction = (a0 == a1 and b0 == b1)
        adjacency[f0].append((f1, same_direction))
        adjacency[f1].append((f0, same_direction))

    visited = np.zeros(len(faces), dtype=bool)
    flip = np.zeros(len(faces), dtype=bool)
    for start in range(len(faces)):
        if visited[start]:
            continue
        visited[start] = True
        queue: deque[int] = deque([start])
        while queue:
            fi = queue.popleft()
            for nb, same_direction in adjacency[fi]:
                desired_flip = flip[fi] ^ bool(same_direction)
                if not visited[nb]:
                    flip[nb] = desired_flip
                    visited[nb] = True
                    queue.append(nb)

    faces[flip, 1:] = faces[flip, 1:][:, ::-1]

    # If closed, use signed volume to make winding outward.  Positive signed
    # volume corresponds to outward orientation for right-handed coordinates.
    tri = vertices[faces]
    signed_volume = float(
        np.sum(np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2]))) / 6.0
    )
    if np.isfinite(signed_volume) and signed_volume < 0.0:
        faces[:, [1, 2]] = faces[:, [2, 1]]
    return faces


def _mesh_repair_and_validate(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    weld_tolerance: float,
    min_component_faces: int = 8,
    validate_watertight: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Repair common recovered mesh issues before OCC sewing."""
    vertices, faces = _drop_degenerate_faces(vertices, faces)
    if len(vertices) < 4 or len(faces) < 4:
        raise RuntimeError("Recovered Shape has too few valid triangles for STEP sewing.")

    vertices, faces = _merge_nearby_vertices(vertices, faces, tolerance=float(weld_tolerance) * 0.25)
    vertices, faces = _remove_tiny_components(vertices, faces, min_faces=min_component_faces)
    faces = _orient_faces_consistently(vertices, faces)
    vertices, faces = _drop_degenerate_faces(vertices, faces)

    boundary_edges = _find_boundary_edges(faces)
    nonmanifold_edges = _find_nonmanifold_edges(faces)
    if validate_watertight and (boundary_edges or nonmanifold_edges):
        raise RuntimeError(
            "Recovered Shape mesh is not a watertight 2-manifold "
            f"({len(boundary_edges)} boundary edge(s), "
            f"{len(nonmanifold_edges)} non-manifold edge(s))."
        )

    if boundary_edges or nonmanifold_edges:
        logger.warning(
            "Recovered Shape mesh has %d boundary edge(s) and %d non-manifold edge(s); "
            "continuing because validate_watertight=False.",
            len(boundary_edges),
            len(nonmanifold_edges),
        )
    return vertices, faces


# ---------------------------------------------------------------------------
# OCC sewing helpers used by the recovered-mesh STEP path.
# ---------------------------------------------------------------------------

def _shape_to_shells(shape: Any) -> list[Any]:
    """Extract TopoDS shells from an OCC shape or compound."""
    from OCP.TopAbs import TopAbs_SHELL
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS

    shells: list[Any] = []
    exp = TopExp_Explorer(shape, TopAbs_SHELL)
    while exp.More():
        try:
            shells.append(TopoDS.Shell_s(exp.Current()))
        except Exception:
            pass
        exp.Next()
    return shells


def _shell_to_solid(shell: Any) -> Any:
    """Fix a sewn shell and close it into an OCC solid."""
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeSolid
    from OCP.ShapeFix import ShapeFix_Shell, ShapeFix_Solid

    try:
        shell_fixer = ShapeFix_Shell(shell)
        shell_fixer.Perform()
        fixed_shell = shell_fixer.Shell()
    except Exception:
        fixed_shell = shell

    builder = BRepBuilderAPI_MakeSolid()
    builder.Add(fixed_shell)
    if not builder.IsDone():
        raise RuntimeError("Sewed recovered mesh did not close into a solid.")

    solid = builder.Solid()
    try:
        solid_fixer = ShapeFix_Solid(solid)
        solid_fixer.Perform()
        fixed_solid = solid_fixer.Solid()
        if fixed_solid is not None:
            solid = fixed_solid
    except Exception:
        logger.debug("ShapeFix_Solid failed for recovered mesh; using raw solid")
    return solid


def _assert_valid_occ_shape(shape: Any, *, label: str = "STEP body") -> None:
    """Raise a useful error if OpenCASCADE says the B-rep is invalid."""
    try:
        from OCP.BRepCheck import BRepCheck_Analyzer

        occ_shape = shape.wrapped if hasattr(shape, "wrapped") else shape
        analyzer = BRepCheck_Analyzer(occ_shape)
        if not analyzer.IsValid():
            raise RuntimeError(f"{label} is not a valid OpenCASCADE B-rep.")
    except RuntimeError:
        raise
    except Exception:
        logger.debug("BRepCheck_Analyzer unavailable or failed; skipped validation")


def _unify_same_domain_shape(shape: Any, merge_angle_deg: float = 0.0) -> Any:
    """Merge same-domain faces after sewing and analytic feature booleans."""
    if not merge_angle_deg or float(merge_angle_deg) <= 0.0:
        return shape
    try:
        import cadquery as cq
        from OCP.ShapeUpgrade import ShapeUpgrade_UnifySameDomain

        occ_shape = shape.wrapped if hasattr(shape, "wrapped") else shape
        up = ShapeUpgrade_UnifySameDomain(occ_shape, True, True, True)
        try:
            up.SetLinearTolerance(1e-4)
            up.SetAngularTolerance(float(np.radians(merge_angle_deg)))
        except Exception:
            pass
        up.Build()
        merged = cq.Shape.cast(up.Shape())
        if merged is not None and merged.isValid():
            return merged
    except Exception:
        logger.debug("ShapeUpgrade_UnifySameDomain failed; keeping recovered B-rep")
    return shape


def _solid_volume(shape: Any) -> Optional[float]:
    try:
        return float(shape.Volume())
    except Exception:
        return None


def _single_solid_if_possible(shape: Any) -> Any:
    """Return the only child solid from a compound when OCC keeps a wrapper."""
    try:
        solids = shape.Solids()
    except Exception:
        return shape
    if solids and len(solids) == 1:
        return solids[0]
    return shape


def _shells_to_cq_shape(shells: list[Any]) -> Any:
    """Convert one or more sewn shells into a CadQuery Solid or Compound."""
    import cadquery as cq

    solids = []
    errors = []
    for shell in shells:
        try:
            solid = cq.Solid(_shell_to_solid(shell))
            volume = _solid_volume(solid)
            if solid.isValid() and (volume is None or abs(volume) > 1e-12):
                solids.append(solid)
        except Exception as exc:
            errors.append(str(exc))

    if not solids:
        detail = f" ({'; '.join(errors[:3])})" if errors else ""
        raise RuntimeError(f"Recovered mesh sewing produced no valid solid{detail}.")
    if len(solids) == 1:
        return solids[0]
    return cq.Compound.makeCompound(solids)


# ---------------------------------------------------------------------------
# Passive feature preservation for STEP
# ---------------------------------------------------------------------------

def _compound_or_single(tools: list[Any]) -> Any:
    """Combine multiple OCC tools into a single compound for batch boolean."""
    import cadquery as cq

    tools = [t for t in tools if t is not None]
    if not tools:
        return None
    if len(tools) == 1:
        return tools[0]
    return cq.Compound.makeCompound(tools)



def _payload_bounds(payload: Any) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Return physical bounds stored by TopologyOptVoxelNode, if present."""
    if not isinstance(payload, dict):
        return None
    bounds = payload.get("bounds")
    mins = maxs = None
    if isinstance(bounds, dict):
        mins = bounds.get("min")
        if mins is None:
            mins = bounds.get("mins")
        if mins is None:
            mins = bounds.get("minimum")
        maxs = bounds.get("max")
        if maxs is None:
            maxs = bounds.get("maxs")
        if maxs is None:
            maxs = bounds.get("maximum")
    elif isinstance(bounds, (list, tuple)) and len(bounds) >= 2:
        mins, maxs = bounds[0], bounds[1]
    if mins is None or maxs is None:
        return None
    try:
        mins_arr = np.asarray(mins, dtype=float)[:3]
        maxs_arr = np.asarray(maxs, dtype=float)[:3]
    except Exception:
        return None
    if mins_arr.size < 3 or maxs_arr.size < 3:
        return None
    if not np.all(np.isfinite(mins_arr)) or not np.all(np.isfinite(maxs_arr)):
        return None
    if not np.all(maxs_arr > mins_arr):
        return None
    return mins_arr, maxs_arr


def _payload_region_list(payload: Any, key: str) -> list[Any]:
    if not isinstance(payload, dict):
        return []
    regions = payload.get("passive_regions")
    if not isinstance(regions, dict):
        return []
    value = regions.get(key) or []
    return list(value) if isinstance(value, (list, tuple)) else []


def _payload_region_boxes(payload: Any, key: str) -> list[BoxRegion]:
    out: list[BoxRegion] = []
    for item in _payload_region_list(payload, key):
        if isinstance(item, dict):
            vals = item.get("bounds") or item.get("box") or item.get("values")
            item = vals if vals is not None else item
        if not isinstance(item, (list, tuple)) or len(item) < 6:
            continue
        try:
            x0, x1, y0, y1, z0, z1 = [float(v) for v in item[:6]]
            out.append((x0, x1, y0, y1, z0, z1))
        except Exception:
            continue
    return out


def _payload_region_cylinders(payload: Any, key: str) -> list[CylinderRegion]:
    """Return passive cylinders, accepting both 6- and 7-value region tuples."""
    out: list[CylinderRegion] = []
    for item in _payload_region_list(payload, key):
        if isinstance(item, dict):
            vals = item.get("fractional") or item.get("region") or item.get("values")
            item = vals if vals is not None else item
        if not isinstance(item, (list, tuple)) or len(item) < 6:
            continue
        try:
            axis, c0, c1, lo, hi, r0 = item[:6]
            r1 = item[6] if len(item) > 6 else r0
            r0 = float(r0)
            r1 = float(r1)
            if r0 <= 0.0 or r1 <= 0.0:
                continue
            out.append((
                str(axis or "z").strip().lower(),
                float(c0),
                float(c1),
                float(lo),
                float(hi),
                r0,
                r1,
            ))
        except Exception:
            continue
    return out


def _region_box_to_solid(
    box: BoxRegion,
    bounds: Tuple[np.ndarray, np.ndarray],
) -> Any:
    import cadquery as cq

    mins, maxs = bounds
    span = np.maximum(maxs - mins, 1e-12)
    x0, x1, y0, y1, z0, z1 = [float(v) for v in box]
    lo = mins + np.asarray([min(x0, x1), min(y0, y1), min(z0, z1)], dtype=float) * span
    hi = mins + np.asarray([max(x0, x1), max(y0, y1), max(z0, z1)], dtype=float) * span
    size = hi - lo
    if not np.all(size > 1e-9):
        return None
    return cq.Solid.makeBox(
        float(size[0]),
        float(size[1]),
        float(size[2]),
        cq.Vector(float(lo[0]), float(lo[1]), float(lo[2])),
    )


def _region_cylinder_to_solid(
    cylinder: CylinderRegion,
    bounds: Tuple[np.ndarray, np.ndarray],
    *,
    axial_margin: float = 0.0,
    radial_margin: float = 0.0,
) -> Any:
    """Convert a fractional passive cylinder to an analytic CadQuery cylinder."""
    import cadquery as cq

    mins, maxs = bounds
    span = np.maximum(maxs - mins, 1e-12)
    axis, c0, c1, lo, hi, r0, r1 = cylinder
    axis = str(axis or "z").strip().lower()
    lo, hi = sorted((float(lo), float(hi)))
    lo -= float(axial_margin)
    hi += float(axial_margin)

    if axis == "x":
        radius = 0.5 * (float(r0) * span[1] + float(r1) * span[2])
        height = (hi - lo) * span[0]
        base = (
            mins[0] + lo * span[0],
            mins[1] + float(c0) * span[1],
            mins[2] + float(c1) * span[2],
        )
        direction = (1.0, 0.0, 0.0)
    elif axis == "y":
        radius = 0.5 * (float(r0) * span[0] + float(r1) * span[2])
        height = (hi - lo) * span[1]
        base = (
            mins[0] + float(c0) * span[0],
            mins[1] + lo * span[1],
            mins[2] + float(c1) * span[2],
        )
        direction = (0.0, 1.0, 0.0)
    else:
        radius = 0.5 * (float(r0) * span[0] + float(r1) * span[1])
        height = (hi - lo) * span[2]
        base = (
            mins[0] + float(c0) * span[0],
            mins[1] + float(c1) * span[1],
            mins[2] + lo * span[2],
        )
        direction = (0.0, 0.0, 1.0)

    radius += float(radial_margin or 0.0)
    if radius <= 1e-9 or height <= 1e-9:
        return None
    return cq.Solid.makeCylinder(
        float(radius),
        float(height),
        cq.Vector(float(base[0]), float(base[1]), float(base[2])),
        cq.Vector(float(direction[0]), float(direction[1]), float(direction[2])),
    )


def _as_float_vec(value: Any, *, length: int = 3) -> Optional[np.ndarray]:
    try:
        arr = np.asarray(value, dtype=float)[:length]
    except Exception:
        return None
    if arr.size < length or not np.all(np.isfinite(arr)):
        return None
    return arr.astype(float)


def _payload_physical_cad_features(payload: Any, role: str, feature_type: str) -> list[dict[str, Any]]:
    """Return optional exact CAD feature metadata in physical model units.

    Supported payload locations:
    ``payload['cad_features']`` or ``payload['passive_cad_features']``.

    Example cylinder entry::

        {
            "type": "cylinder",
            "role": "void",  # or "solid"
            "center": [x, y, z],
            "axis": [0, 0, 1],
            "radius": r,
            "height": h,
            "name": "front_left_bearing_bore"
        }

    Start/end form is also accepted using ``start`` and ``end`` instead of
    ``center``/``height``.
    """
    if not isinstance(payload, dict):
        return []
    raw = []
    for key in ("cad_features", "passive_cad_features"):
        value = payload.get(key)
        if isinstance(value, (list, tuple)):
            raw.extend(value)
    out: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        if str(item.get("type", "")).strip().lower() != feature_type:
            continue
        if str(item.get("role", "")).strip().lower() != role:
            continue
        out.append(item)
    return out


def _physical_box_to_solid(feature: dict[str, Any]) -> Any:
    import cadquery as cq

    lo = _as_float_vec(feature.get("min") or feature.get("mins") or feature.get("lo"))
    hi = _as_float_vec(feature.get("max") or feature.get("maxs") or feature.get("hi"))
    if lo is None or hi is None:
        center = _as_float_vec(feature.get("center"))
        size = _as_float_vec(feature.get("size") or feature.get("dimensions"))
        if center is None or size is None:
            return None
        lo = center - 0.5 * size
        hi = center + 0.5 * size
    size = hi - lo
    if not np.all(size > 1e-9):
        return None
    return cq.Solid.makeBox(
        float(size[0]),
        float(size[1]),
        float(size[2]),
        cq.Vector(float(lo[0]), float(lo[1]), float(lo[2])),
    )


def _physical_cylinder_to_solid(
    feature: dict[str, Any],
    *,
    axial_margin: float = 0.0,
    radial_margin: float = 0.0,
) -> Any:
    import cadquery as cq

    radius = feature.get("radius")
    try:
        radius = float(radius) + float(radial_margin or 0.0)
    except Exception:
        return None
    if radius <= 1e-9:
        return None

    start = _as_float_vec(feature.get("start"))
    end = _as_float_vec(feature.get("end"))
    if start is not None and end is not None:
        vec = end - start
        height = float(np.linalg.norm(vec))
        if height <= 1e-9:
            return None
        axis = vec / height
        base = start - axis * float(axial_margin or 0.0)
        height += 2.0 * float(axial_margin or 0.0)
    else:
        center = _as_float_vec(feature.get("center"))
        axis = _as_float_vec(feature.get("axis"))
        if center is None or axis is None:
            return None
        axis_norm = float(np.linalg.norm(axis))
        if axis_norm <= 1e-12:
            return None
        axis = axis / axis_norm
        try:
            height = float(feature.get("height") or feature.get("length"))
        except Exception:
            return None
        if height <= 1e-9:
            return None
        base = center - axis * (0.5 * height + float(axial_margin or 0.0))
        height += 2.0 * float(axial_margin or 0.0)

    return cq.Solid.makeCylinder(
        float(radius),
        float(height),
        cq.Vector(float(base[0]), float(base[1]), float(base[2])),
        cq.Vector(float(axis[0]), float(axis[1]), float(axis[2])),
    )


def _boolean_or_raise(result: Any, tool: Any, *, operation: str) -> Any:
    """Run a CadQuery boolean and give a targeted failure message."""
    try:
        candidate = result.fuse(tool) if operation == "fuse" else result.cut(tool)
    except Exception as exc:
        raise RuntimeError(f"Recovered Shape STEP failed during analytic {operation}.") from exc
    if candidate is None or (hasattr(candidate, "isValid") and not candidate.isValid()):
        raise RuntimeError(f"Recovered Shape STEP analytic {operation} produced an invalid body.")
    return _single_solid_if_possible(candidate)


def _apply_passive_regions_to_step(
    shape: Any,
    payload: Any,
    *,
    void_axial_margin: float = 0.02,
    void_radial_margin: float = 0.0,
    validate_after_boolean: bool = True,
) -> Any:
    """Re-apply passive boxes/cylinders as analytic STEP features."""
    import cadquery as cq

    result = shape
    solid_tools: list[Any] = []
    cut_tools: list[Any] = []

    bounds = _payload_bounds(payload)
    if bounds is not None:
        for box in _payload_region_boxes(payload, "solid_boxes"):
            tool = _region_box_to_solid(box, bounds)
            if tool is not None:
                solid_tools.append(tool)
        for cylinder in _payload_region_cylinders(payload, "solid_cylinders"):
            tool = _region_cylinder_to_solid(cylinder, bounds)
            if tool is not None:
                solid_tools.append(tool)

        for box in _payload_region_boxes(payload, "void_boxes"):
            tool = _region_box_to_solid(box, bounds)
            if tool is not None:
                cut_tools.append(tool)
        for cylinder in _payload_region_cylinders(payload, "void_cylinders"):
            tool = _region_cylinder_to_solid(
                cylinder,
                bounds,
                axial_margin=void_axial_margin,
                radial_margin=void_radial_margin,
            )
            if tool is not None:
                cut_tools.append(tool)

    # Optional newer exact-geometry metadata in physical CAD units.
    for feature in _payload_physical_cad_features(payload, "solid", "box"):
        tool = _physical_box_to_solid(feature)
        if tool is not None:
            solid_tools.append(tool)
    for feature in _payload_physical_cad_features(payload, "solid", "cylinder"):
        tool = _physical_cylinder_to_solid(feature)
        if tool is not None:
            solid_tools.append(tool)
    for feature in _payload_physical_cad_features(payload, "void", "box"):
        tool = _physical_box_to_solid(feature)
        if tool is not None:
            cut_tools.append(tool)
    for feature in _payload_physical_cad_features(payload, "void", "cylinder"):
        tool = _physical_cylinder_to_solid(
            feature,
            axial_margin=void_axial_margin,
            radial_margin=void_radial_margin,
        )
        if tool is not None:
            cut_tools.append(tool)

    solid_compound = _compound_or_single(solid_tools)
    if solid_compound is not None:
        result = _boolean_or_raise(result, solid_compound, operation="fuse")

    cut_compound = _compound_or_single(cut_tools)
    if cut_compound is not None:
        result = _boolean_or_raise(result, cut_compound, operation="cut")

    n_fused = len(solid_tools)
    n_cut = len(cut_tools)
    if n_fused or n_cut:
        if validate_after_boolean:
            _assert_valid_occ_shape(result, label="STEP body after passive booleans")
        logger.info(
            "Recovered Shape STEP: re-applied %d analytic passive solid(s), "
            "cut %d analytic passive void(s).",
            n_fused,
            n_cut,
        )
    result = _single_solid_if_possible(result)
    if not isinstance(result, cq.Shape):
        result = cq.Shape.cast(result)
    return result


# ---------------------------------------------------------------------------
# Single STEP path: recovered mesh -> faceted B-rep + analytic passive features
# ---------------------------------------------------------------------------

def _recovered_mesh_to_faceted_brep_solid(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    sew_tolerance: float = 1e-4,
    relative_sew_tolerance: float = 1e-6,
    merge_angle_deg: float = 0.0,
    max_faces: Optional[int] = None,
    repair_mesh: bool = False,
    validate_watertight: bool = False,
    validate_brep: bool = False,
    min_component_faces: int = 0,
) -> Any:
    """Sew the recovered triangle mesh into an OCC B-rep shape.

    ``max_faces`` is accepted only for compatibility with saved graphs.  It is
    not used to switch geometry style or trigger a fallback.
    """
    _ = max_faces

    import cadquery as cq
    from OCP.BRepBuilderAPI import (
        BRepBuilderAPI_MakeFace,
        BRepBuilderAPI_MakePolygon,
        BRepBuilderAPI_Sewing,
    )
    from OCP.gp import gp_Pnt

    effective_tol = _effective_tolerance(vertices, sew_tolerance, relative_sew_tolerance)
    if repair_mesh:
        vertices, faces = _mesh_repair_and_validate(
            vertices,
            faces,
            weld_tolerance=effective_tol,
            min_component_faces=min_component_faces,
            validate_watertight=validate_watertight,
        )
    else:
        vertices, faces = _drop_degenerate_faces(vertices, faces)

    if len(vertices) < 4 or len(faces) < 4:
        raise RuntimeError("Recovered Shape has too few valid triangles for STEP sewing.")

    sew = BRepBuilderAPI_Sewing(float(effective_tol))
    n_added = 0
    n_skipped = 0
    for tri in faces:
        poly = BRepBuilderAPI_MakePolygon()
        for idx in tri[:3]:
            p = vertices[int(idx)]
            poly.Add(gp_Pnt(float(p[0]), float(p[1]), float(p[2])))
        poly.Close()
        try:
            face_builder = BRepBuilderAPI_MakeFace(poly.Wire(), True)
            if hasattr(face_builder, "IsDone") and not face_builder.IsDone():
                n_skipped += 1
                continue
            sew.Add(face_builder.Face())
            n_added += 1
        except Exception:
            n_skipped += 1

    if n_added < 4:
        raise RuntimeError("Recovered Shape produced too few sewable triangles for STEP.")

    sew.Perform()
    shells = _shape_to_shells(sew.SewedShape())
    if not shells:
        raise RuntimeError("Recovered Shape sewing produced no shell.")

    shape = _shells_to_cq_shape(shells)
    if merge_angle_deg and float(merge_angle_deg) > 0.0:
        shape = _unify_same_domain_shape(shape, merge_angle_deg=merge_angle_deg)
    if not isinstance(shape, cq.Shape):
        shape = cq.Shape.cast(shape)
    if shape is None or not shape.isValid():
        raise RuntimeError("Recovered Shape STEP body is invalid after sewing.")
    if validate_brep:
        _assert_valid_occ_shape(shape, label="STEP body after mesh sewing")

    volume = _solid_volume(shape)
    logger.info(
        "Recovered Shape CAD reconstruction: %d triangles sewn%s, sew_tol=%g%s.",
        n_added,
        f", {n_skipped} skipped" if n_skipped else "",
        effective_tol,
        f", volume={volume:.3f}" if volume is not None else "",
    )
    return shape


def reconstruct_topopt_cad(
    payload: Any,
    *,
    source_geometry: str = _RECOVERED_MODE,
    sew_tolerance: float = 1e-4,
    relative_sew_tolerance: float = 1e-6,
    merge_angle_deg: float = 0.0,
    repair_mesh: bool = False,
    validate_watertight: bool = False,
    validate_brep: bool = False,
    min_component_faces: int = 0,
    void_axial_margin: float = 0.02,
    void_radial_margin: float = 0.0,
    **_ignored_legacy_kwargs: Any,
) -> Any:
    """Build a CadQuery workplane from the topology result's Recovered Shape."""
    mode = str(source_geometry or _RECOVERED_MODE).strip()
    if mode and mode.lower() != _RECOVERED_MODE.lower():
        logger.info(
            "CAD reconstruction source_geometry=%r ignored; using %s.",
            mode,
            _RECOVERED_MODE,
        )

    recovered = _extract_recovered_mesh(payload)
    if recovered is None:
        raise RuntimeError(
            "Recovered Shape STEP export needs topology_result['recovered_shape'] "
            "with vertices and faces."
        )

    vertices, faces = recovered
    solid = _recovered_mesh_to_faceted_brep_solid(
        vertices,
        faces,
        sew_tolerance=float(sew_tolerance),
        relative_sew_tolerance=float(relative_sew_tolerance or 0.0),
        merge_angle_deg=float(merge_angle_deg or 0.0),
        repair_mesh=bool(repair_mesh),
        validate_watertight=bool(validate_watertight),
        validate_brep=bool(validate_brep),
        min_component_faces=int(min_component_faces or 0),
    )
    solid = _apply_passive_regions_to_step(
        solid,
        payload,
        void_axial_margin=float(void_axial_margin or 0.0),
        void_radial_margin=float(void_radial_margin or 0.0),
        validate_after_boolean=bool(validate_brep),
    )
    if merge_angle_deg and float(merge_angle_deg) > 0.0:
        solid = _unify_same_domain_shape(solid, merge_angle_deg=float(merge_angle_deg))
    if validate_brep:
        _assert_valid_occ_shape(solid, label="final Recovered Shape STEP body")

    import cadquery as cq
    return cq.Workplane(obj=solid)


__all__ = ["reconstruct_topopt_cad"]
