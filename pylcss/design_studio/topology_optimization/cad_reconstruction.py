# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.
"""Phase 5 — CAD reconstruction: topology-opt result → parametric B-rep solid.

Turns a voxel topology-optimisation result into an OpenCASCADE B-rep solid
that exports to STEP and flows into the rest of the CadQuery node graph
(boolean ops, fillets, drawings, assembly fit checks).

Default reconstruction now uses the TopOpt node's smoothed recovered_shape:
marching cubes + smoothing/post-processing, sewn into a faceted B-rep STEP.
The voxel-boundary route below remains as a fast fallback/option.

## Voxel fallback approach

A topology optimiser on a structured voxel grid has a natural, exact boundary:
the exposed faces of the solid voxels.  We build the B-rep from **those
axis-aligned quad faces**, then run OpenCASCADE's `UnifySameDomain` to merge
coplanar quads.  Flat walls collapse from hundreds of little quads into a
handful of large planar faces — a clean, editable, CAD-native solid.

When smooth reconstruction fails or is disabled, the voxel-boundary route
still produces a valid STEP quickly by following the exact solid voxel faces.

## Honest scope

The smooth STEP is a faceted B-rep: many triangular planar faces following
the recovered mesh.  It is not compact NURBS/T-spline surface fitting.  The
separate "Fitted NURBS Surface" mode uses automatic patch fitting where the
mesh is clean enough, with triangle fallback inside the patch fitter.  The
voxel-boundary fallback is blocky, but often more robust and easier to edit.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np

from pylcss.design_studio.core.base_node import CadQueryNode, resolve_any_input

logger = logging.getLogger(__name__)

_SMOOTH_MODE = "Smooth Recovered Shape"
_VOXEL_MODE = "Voxel Boundary"
_AUTO_MODE = "Smooth, fallback to Voxel"
_SWEEP_MODE = "Skeleton NURBS Sweep"
_NURBS_MODE = "Fitted NURBS Surface"
_DEFAULT_MAX_SMOOTH_FACES = 3000


# Six faces of a unit voxel: (neighbour offset, 4 corner (dx,dy,dz) in winding
# order).  A face is on the boundary when the neighbour voxel is not solid.
_VOXEL_FACES = (
    ((-1, 0, 0), ((0, 0, 0), (0, 1, 0), (0, 1, 1), (0, 0, 1))),
    (( 1, 0, 0), ((1, 0, 0), (1, 0, 1), (1, 1, 1), (1, 1, 0))),
    (( 0, -1, 0), ((0, 0, 0), (0, 0, 1), (1, 0, 1), (1, 0, 0))),
    (( 0, 1, 0), ((0, 1, 0), (1, 1, 0), (1, 1, 1), (0, 1, 1))),
    (( 0, 0, -1), ((0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0))),
    (( 0, 0, 1), ((0, 0, 1), (0, 1, 1), (1, 1, 1), (1, 0, 1))),
)


# ---------------------------------------------------------------------------
# Voxel boundary → OCC B-rep solid
# ---------------------------------------------------------------------------

def _voxel_boundary_to_brep_solid(
    density: np.ndarray,
    bounds: Tuple[np.ndarray, np.ndarray],
    cutoff: float,
    merge_angle_deg: float = 1.0,
) -> Any:
    """Build a coplanar-merged B-rep solid from a thresholded voxel field.

    Returns a `cadquery.Solid`.  Raises RuntimeError on unrecoverable failure.
    """
    import cadquery as cq
    from OCP.gp import gp_Pnt
    from OCP.BRepBuilderAPI import (
        BRepBuilderAPI_MakePolygon,
        BRepBuilderAPI_MakeFace,
        BRepBuilderAPI_Sewing,
        BRepBuilderAPI_MakeSolid,
    )
    from OCP.TopoDS import TopoDS
    from OCP.TopAbs import TopAbs_SHELL
    from OCP.TopExp import TopExp_Explorer
    from OCP.ShapeFix import ShapeFix_Solid
    from OCP.ShapeUpgrade import ShapeUpgrade_UnifySameDomain

    density = np.asarray(density, dtype=float)
    if density.ndim != 3:
        raise RuntimeError("CAD reconstruction expects a 3-D voxel density field.")
    nelx, nely, nelz = density.shape

    solid = density >= float(cutoff)
    if not np.any(solid):
        nz = density[density > 0.0]
        if nz.size == 0:
            raise RuntimeError("Density field is empty — nothing to reconstruct.")
        solid = density >= float(np.percentile(nz, 50.0))

    mins = np.asarray(bounds[0], dtype=float)
    maxs = np.asarray(bounds[1], dtype=float)
    cell = (maxs - mins) / np.array([nelx, nely, nelz], dtype=float)

    # Pad with a False border so boundary detection needs no edge special-casing.
    padded = np.pad(solid, 1, constant_values=False)

    sew = BRepBuilderAPI_Sewing(1e-6)
    n_quads = 0
    for (i, j, k) in np.argwhere(solid):
        i, j, k = int(i), int(j), int(k)
        for (off, corners) in _VOXEL_FACES:
            if padded[i + 1 + off[0], j + 1 + off[1], k + 1 + off[2]]:
                continue  # neighbour solid → internal face, skip
            poly = BRepBuilderAPI_MakePolygon()
            for (dx, dy, dz) in corners:
                p = mins + np.array([i + dx, j + dy, k + dz], dtype=float) * cell
                poly.Add(gp_Pnt(float(p[0]), float(p[1]), float(p[2])))
            poly.Close()
            try:
                sew.Add(BRepBuilderAPI_MakeFace(poly.Wire(), True).Face())
                n_quads += 1
            except Exception:
                pass

    if n_quads == 0:
        raise RuntimeError("No boundary faces — density cutoff may be too high.")

    sew.Perform()
    sewed = sew.SewedShape()
    exp = TopExp_Explorer(sewed, TopAbs_SHELL)
    if not exp.More():
        raise RuntimeError("Sewing produced no closed shell from the voxel boundary.")
    shell = TopoDS.Shell_s(exp.Current())
    occ_solid = BRepBuilderAPI_MakeSolid(shell).Solid()

    try:
        fixer = ShapeFix_Solid(occ_solid)
        fixer.Perform()
        fixed = fixer.Solid()
        if fixed is not None:
            occ_solid = fixed
    except Exception:
        logger.debug("ShapeFix_Solid failed; using unrepaired solid")

    faceted = occ_solid
    if merge_angle_deg and merge_angle_deg > 0.0:
        try:
            up = ShapeUpgrade_UnifySameDomain(occ_solid, True, True, True)
            try:
                up.SetLinearTolerance(1e-4)
                up.SetAngularTolerance(float(np.radians(merge_angle_deg)))
            except Exception:
                pass
            up.Build()
            merged = up.Shape()
            cq_merged = cq.Solid(merged)
            if cq_merged.isValid() and cq_merged.Volume() > 0:
                logger.info(
                    "CAD reconstruction: %d boundary quads → coplanar-merged "
                    "B-rep at %.1f°.", n_quads, merge_angle_deg,
                )
                return cq_merged
            logger.warning("Coplanar merge produced an invalid solid; "
                           "falling back to faceted B-rep.")
        except Exception:
            logger.warning("Coplanar merge failed; falling back to faceted B-rep.")

    return cq.Solid(faceted)


# ---------------------------------------------------------------------------
# Smoothed recovered mesh -> faceted OCC B-rep solid
# ---------------------------------------------------------------------------

def _extract_recovered_mesh(
    payload: Any,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Return (vertices, triangle_faces) from a TopOpt recovered_shape payload."""
    if not isinstance(payload, dict):
        return None

    mesh = payload.get('recovered_shape')
    if mesh is None and 'vertices' in payload and 'faces' in payload:
        mesh = payload
    if not isinstance(mesh, dict):
        return None

    vertices = mesh.get('vertices')
    faces = mesh.get('faces')
    if vertices is None or faces is None:
        return None

    vertices = np.asarray(vertices, dtype=float)
    faces = np.asarray(faces, dtype=int)
    if vertices.ndim != 2 or vertices.shape[1] < 3:
        return None
    if faces.ndim != 2 or faces.shape[1] < 3:
        return None
    if len(vertices) < 4 or len(faces) < 4:
        return None
    return vertices[:, :3], faces[:, :3]


def _shape_to_shells(shape: Any) -> list[Any]:
    """Extract TopoDS shells from an OCC shape or compound."""
    from OCP.TopAbs import TopAbs_SHELL
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS

    shells = []
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


def _unify_same_domain_shape(shape: Any, merge_angle_deg: float = 1.0) -> Any:
    """Merge coplanar/same-domain faces after mesh sewing or analytic cuts."""
    if not merge_angle_deg or float(merge_angle_deg) <= 0.0:
        return shape
    try:
        import cadquery as cq
        from OCP.ShapeUpgrade import ShapeUpgrade_UnifySameDomain

        occ_shape = shape.wrapped if hasattr(shape, 'wrapped') else shape
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
        logger.debug("ShapeUpgrade_UnifySameDomain failed; keeping original shape")
    return shape


def _payload_region_cylinders(
    payload: Any,
    key: str,
) -> list[Tuple[str, float, float, float, float, float]]:
    if not isinstance(payload, dict):
        return []
    regions = payload.get('passive_regions')
    if not isinstance(regions, dict):
        return []
    cylinders = regions.get(key) or []
    out: list[Tuple[str, float, float, float, float, float]] = []
    for item in cylinders:
        if not isinstance(item, (list, tuple)) or len(item) < 6:
            continue
        axis, c0, c1, lo, hi, radius = item[:6]
        try:
            out.append((
                str(axis or 'z').lower(),
                float(c0), float(c1),
                float(lo), float(hi),
                float(radius),
            ))
        except Exception:
            continue
    return out


def _payload_void_cylinders(payload: Any) -> list[Tuple[str, float, float, float, float, float]]:
    return _payload_region_cylinders(payload, 'void_cylinders')


def _payload_extrusion_axis(payload: Any) -> str:
    if not isinstance(payload, dict):
        return 'none'
    axis = payload.get('extrusion_axis')
    if axis is None:
        manufacturing = payload.get('manufacturing')
        if isinstance(manufacturing, dict):
            axis = manufacturing.get('extrusion')
    axis = str(axis or 'none').strip().lower()
    return axis if axis in {'x', 'y', 'z'} else 'none'


def _cut_payload_void_cylinders(
    shape: Any,
    payload: Any,
    bounds: Optional[Tuple[np.ndarray, np.ndarray]],
) -> Any:
    """Cut analytic through-cylinders back into the reconstructed CAD body."""
    cylinders = _payload_void_cylinders(payload)
    if not cylinders or bounds is None:
        return shape

    import cadquery as cq

    mins, maxs = bounds
    mins = np.asarray(mins, dtype=float)[:3]
    maxs = np.asarray(maxs, dtype=float)[:3]
    span = np.maximum(maxs - mins, 1e-12)
    result = shape

    for axis, c0, c1, lo, hi, radius in cylinders:
        if radius <= 0.0:
            continue
        lo, hi = sorted((float(lo), float(hi)))
        margin = 0.05
        try:
            if axis == 'x':
                rad = float(radius) * float(min(span[1], span[2]))
                height = max((hi - lo) * span[0], span[0] * 1e-3) + 2.0 * margin * span[0]
                start = cq.Vector(
                    mins[0] + lo * span[0] - margin * span[0],
                    mins[1] + c0 * span[1],
                    mins[2] + c1 * span[2],
                )
                direction = cq.Vector(1, 0, 0)
            elif axis == 'y':
                rad = float(radius) * float(min(span[0], span[2]))
                height = max((hi - lo) * span[1], span[1] * 1e-3) + 2.0 * margin * span[1]
                start = cq.Vector(
                    mins[0] + c0 * span[0],
                    mins[1] + lo * span[1] - margin * span[1],
                    mins[2] + c1 * span[2],
                )
                direction = cq.Vector(0, 1, 0)
            else:
                rad = float(radius) * float(min(span[0], span[1]))
                height = max((hi - lo) * span[2], span[2] * 1e-3) + 2.0 * margin * span[2]
                start = cq.Vector(
                    mins[0] + c0 * span[0],
                    mins[1] + c1 * span[1],
                    mins[2] + lo * span[2] - margin * span[2],
                )
                direction = cq.Vector(0, 0, 1)

            cutter = cq.Solid.makeCylinder(rad, height, start, direction)
            cut = result.cut(cutter)
            if cut is not None and cut.isValid():
                result = cut
        except Exception:
            logger.debug("Analytic cylinder cut failed for passive void %s", (axis, c0, c1, lo, hi, radius))
    return result


def _resample_closed_polyline(points: np.ndarray, max_points: int = 180) -> np.ndarray:
    """Uniformly resample a closed 2-D polyline to a CAD-friendly point count."""
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 4:
        return pts
    if np.linalg.norm(pts[0] - pts[-1]) > 1e-9:
        pts = np.vstack([pts, pts[0]])
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    length = float(np.sum(seg))
    if length <= 1e-12:
        return pts[:-1]
    count = min(max(12, int(max_points)), max(12, len(pts) - 1))
    if len(pts) - 1 <= count:
        return pts[:-1]

    cumulative = np.concatenate([[0.0], np.cumsum(seg)])
    samples = np.linspace(0.0, length, count, endpoint=False)
    out = np.empty((count, 2), dtype=float)
    for i, s in enumerate(samples):
        idx = int(np.searchsorted(cumulative, s, side='right') - 1)
        idx = min(max(idx, 0), len(seg) - 1)
        t = (s - cumulative[idx]) / max(seg[idx], 1e-12)
        out[i] = (1.0 - t) * pts[idx] + t * pts[idx + 1]
    return out


def _polygon_area_2d(points: np.ndarray) -> float:
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 3:
        return 0.0
    x = pts[:, 0]
    y = pts[:, 1]
    return float(0.5 * np.sum(x * np.roll(y, -1) - y * np.roll(x, -1)))


_PROFILE_AXES = {
    # local profile X/Y axes for a prism extruded along the named global axis.
    # The order is chosen so CadQuery's plane yDir points in the positive global
    # direction for both profile axes.
    'x': (1, 2),  # profile coordinates are global Y/Z
    'y': (2, 0),  # profile coordinates are global Z/X
    'z': (0, 1),  # profile coordinates are global X/Y
}


def _axis_vector(axis: int) -> Tuple[float, float, float]:
    vec = [0.0, 0.0, 0.0]
    vec[int(axis)] = 1.0
    return float(vec[0]), float(vec[1]), float(vec[2])


def _extrusion_plane(axis: str, offset: float) -> Any:
    import cadquery as cq

    axis_idx = {'x': 0, 'y': 1, 'z': 2}[str(axis).lower()]
    plane_axes = _PROFILE_AXES[str(axis).lower()]
    origin = [0.0, 0.0, 0.0]
    origin[axis_idx] = float(offset)
    return cq.Plane(
        origin=tuple(origin),
        xDir=_axis_vector(plane_axes[0]),
        normal=_axis_vector(axis_idx),
    )


def _cylinder_center_on_profile(
    cylinder: Tuple[str, float, float, float, float, float],
    plane_axes: Tuple[int, int],
) -> Optional[Tuple[float, float]]:
    axis, c0, c1, _lo, _hi, _radius = cylinder
    center = [None, None, None]
    if str(axis).lower() == 'x':
        center[1], center[2] = float(c0), float(c1)
    elif str(axis).lower() == 'y':
        center[0], center[2] = float(c0), float(c1)
    elif str(axis).lower() == 'z':
        center[0], center[1] = float(c0), float(c1)
    else:
        return None
    if center[plane_axes[0]] is None or center[plane_axes[1]] is None:
        return None
    return float(center[plane_axes[0]]), float(center[plane_axes[1]])


def _wire_extrusion_from_points(
    points: np.ndarray,
    height: float,
    plane: Any,
    *,
    smooth: bool = True,
) -> Any:
    import cadquery as cq

    pts = [(float(x), float(y)) for x, y in np.asarray(points, dtype=float)]
    if len(pts) < 3:
        raise RuntimeError("Profile contour has too few points.")
    try:
        if smooth and len(pts) >= 5:
            solid = cq.Workplane(plane).spline(pts).close().extrude(float(height)).val()
        else:
            solid = cq.Workplane(plane).polyline(pts).close().extrude(float(height)).val()
    except Exception:
        solid = cq.Workplane(plane).polyline(pts).close().extrude(float(height)).val()
    return solid


def _extruded_profile_to_brep_solid(
    density: np.ndarray,
    bounds: Tuple[np.ndarray, np.ndarray],
    cutoff: float,
    payload: Any,
    merge_angle_deg: float = 1.0,
) -> Any:
    """Build a fast smooth CAD prism for any axis-extruded TopOpt result."""
    try:
        import scipy.ndimage as ndi
        from skimage import measure
    except ImportError as exc:
        raise RuntimeError("Extruded profile CAD needs scipy and scikit-image.") from exc

    grid = np.asarray(density, dtype=float)
    if grid.ndim != 3:
        raise RuntimeError("Extruded profile CAD needs a 3-D density field.")

    extrusion_axis = _payload_extrusion_axis(payload)
    if extrusion_axis not in _PROFILE_AXES:
        raise RuntimeError("Extruded profile CAD needs an extrusion axis.")
    axis_idx = {'x': 0, 'y': 1, 'z': 2}[extrusion_axis]
    plane_axes = _PROFILE_AXES[extrusion_axis]
    if min(grid.shape[plane_axes[0]], grid.shape[plane_axes[1]]) < 2:
        raise RuntimeError("Extruded profile CAD needs at least a 2-D profile.")

    mins, maxs = bounds
    mins = np.asarray(mins, dtype=float)[:3]
    maxs = np.asarray(maxs, dtype=float)[:3]
    span = np.maximum(maxs - mins, 1e-12)
    height = float(span[axis_idx])
    if height <= 1e-12:
        raise RuntimeError("Extruded profile CAD needs non-zero plate thickness.")

    clean_grid = np.nan_to_num(grid, nan=0.0, posinf=1.0, neginf=0.0)
    profile = np.mean(clean_grid, axis=axis_idx)
    remaining_axes = [i for i in range(3) if i != axis_idx]
    order = [remaining_axes.index(a) for a in plane_axes]
    if order != [0, 1]:
        profile = np.transpose(profile, axes=order)

    min_plane_dim = max(1, min(profile.shape))
    upsample = int(np.clip(np.ceil(240.0 / min_plane_dim), 1, 8))
    while upsample > 1 and np.prod(np.asarray(profile.shape) * upsample) > 900_000:
        upsample -= 1

    if upsample > 1:
        field = ndi.zoom(profile, zoom=float(upsample), order=3, mode='nearest')
    else:
        field = profile.copy()
    field = np.clip(ndi.gaussian_filter(field, sigma=0.35 if upsample > 1 else 0.20), 0.0, 1.0)

    nx, ny = field.shape
    xf = (np.arange(nx, dtype=float) + 0.5) / max(nx, 1)
    yf = (np.arange(ny, dtype=float) + 0.5) / max(ny, 1)
    xx_f, yy_f = np.meshgrid(xf, yf, indexing='ij')

    for axis, c0, c1, _lo, _hi, radius in _payload_region_cylinders(payload, 'solid_cylinders'):
        if axis == extrusion_axis and radius > 0.0:
            center = _cylinder_center_on_profile((axis, c0, c1, _lo, _hi, radius), plane_axes)
            if center is None:
                continue
            mask = (xx_f - center[0]) ** 2 + (yy_f - center[1]) ** 2 <= radius ** 2
            field[mask] = 1.0
    for axis, c0, c1, _lo, _hi, radius in _payload_region_cylinders(payload, 'void_cylinders'):
        if axis == extrusion_axis and radius > 0.0:
            center = _cylinder_center_on_profile((axis, c0, c1, _lo, _hi, radius), plane_axes)
            if center is None:
                continue
            mask = (xx_f - center[0]) ** 2 + (yy_f - center[1]) ** 2 <= radius ** 2
            field[mask] = 0.0

    pad = max(3, min(10, upsample))
    field = np.pad(field, pad_width=pad, mode='constant', constant_values=0.0)
    level = float(np.clip(cutoff, 1e-6, 0.999999))
    mask = field >= level
    if not np.any(mask) or np.all(mask):
        raise RuntimeError("Extruded profile CAD found no usable iso-contour.")

    plane_span = span[list(plane_axes)]
    plane_mins = mins[list(plane_axes)]
    spacing = plane_span / (np.asarray(profile.shape, dtype=float) * float(upsample))
    outside = ndi.distance_transform_edt(~mask, sampling=tuple(float(v) for v in spacing))
    inside = ndi.distance_transform_edt(mask, sampling=tuple(float(v) for v in spacing))
    sdf = ndi.gaussian_filter(outside - inside, sigma=0.65)

    nxp, nyp = sdf.shape
    xfp = (np.arange(nxp, dtype=float) - float(pad) + 0.5) / max(nx, 1)
    yfp = (np.arange(nyp, dtype=float) - float(pad) + 0.5) / max(ny, 1)
    xxp = plane_mins[0] + xfp[:, None] * plane_span[0]
    yyp = plane_mins[1] + yfp[None, :] * plane_span[1]

    for axis, c0, c1, _lo, _hi, radius in _payload_region_cylinders(payload, 'solid_cylinders'):
        if axis == extrusion_axis and radius > 0.0:
            center = _cylinder_center_on_profile((axis, c0, c1, _lo, _hi, radius), plane_axes)
            if center is None:
                continue
            cx = plane_mins[0] + center[0] * plane_span[0]
            cy = plane_mins[1] + center[1] * plane_span[1]
            rad = radius * min(plane_span[0], plane_span[1])
            cyl = np.sqrt((xxp - cx) ** 2 + (yyp - cy) ** 2) - rad
            np.minimum(sdf, cyl, out=sdf)
    for axis, c0, c1, _lo, _hi, radius in _payload_region_cylinders(payload, 'void_cylinders'):
        if axis == extrusion_axis and radius > 0.0:
            center = _cylinder_center_on_profile((axis, c0, c1, _lo, _hi, radius), plane_axes)
            if center is None:
                continue
            cx = plane_mins[0] + center[0] * plane_span[0]
            cy = plane_mins[1] + center[1] * plane_span[1]
            rad = radius * min(plane_span[0], plane_span[1])
            cyl = rad - np.sqrt((xxp - cx) ** 2 + (yyp - cy) ** 2)
            np.maximum(sdf, cyl, out=sdf)

    contours = measure.find_contours(sdf, level=0.0)
    if not contours:
        raise RuntimeError("Extruded profile CAD found no profile contours.")

    origin_xy = plane_mins + 0.5 * (plane_span / np.asarray(profile.shape, dtype=float)) - float(pad) * spacing
    physical_contours: list[np.ndarray] = []
    for contour in contours:
        if len(contour) < 8:
            continue
        pts = np.column_stack((
            origin_xy[0] + contour[:, 0] * spacing[0],
            origin_xy[1] + contour[:, 1] * spacing[1],
        ))
        area = abs(_polygon_area_2d(pts))
        if area > 1e-8 * float(plane_span[0] * plane_span[1]):
            physical_contours.append(pts)
    if not physical_contours:
        raise RuntimeError("Extruded profile CAD contours were too small.")

    physical_contours.sort(key=lambda p: abs(_polygon_area_2d(p)), reverse=True)
    outer = _resample_closed_polyline(physical_contours[0], max_points=220)
    solid = _wire_extrusion_from_points(
        outer,
        height,
        _extrusion_plane(extrusion_axis, float(mins[axis_idx])),
        smooth=True,
    )

    # Preserve design-created internal voids as extruded cuts, then re-cut
    # passive circular holes analytically so bolt/load holes select as cylinders.
    outer_area = abs(_polygon_area_2d(physical_contours[0]))
    for hole in physical_contours[1:]:
        if abs(_polygon_area_2d(hole)) < 0.0005 * outer_area:
            continue
        try:
            cut_profile = _resample_closed_polyline(hole, max_points=120)
            cutter = _wire_extrusion_from_points(
                cut_profile,
                height * 1.1,
                _extrusion_plane(extrusion_axis, float(mins[axis_idx] - 0.05 * height)),
                smooth=True,
            )
            cut = solid.cut(cutter)
            if cut is not None and cut.isValid():
                solid = cut
        except Exception:
            logger.debug("Extruded profile internal cut failed; continuing")

    solid = _cut_payload_void_cylinders(solid, payload, bounds)
    solid = _unify_same_domain_shape(solid, merge_angle_deg=merge_angle_deg)
    if not solid.isValid():
        raise RuntimeError("Extruded profile CAD produced an invalid solid.")
    try:
        if abs(float(solid.Volume())) <= 1e-12:
            raise RuntimeError("Extruded profile CAD produced zero volume.")
    except AttributeError:
        pass
    logger.info("Extruded profile CAD reconstruction produced a smooth prism solid.")
    return solid


def _drop_degenerate_faces(
    vertices: np.ndarray,
    faces: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    faces = np.asarray(faces, dtype=int)
    if faces.size == 0:
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
    areas = 0.5 * np.linalg.norm(
        np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]),
        axis=1,
    )
    faces = faces[areas > 1e-12]
    if len(faces) == 0:
        return vertices, faces.reshape((0, 3))

    # Remove duplicate triangles while preserving the original winding of the
    # first occurrence.
    keys = np.sort(faces, axis=1)
    _, unique_idx = np.unique(keys, axis=0, return_index=True)
    unique_idx.sort()
    faces = faces[unique_idx]
    return vertices, faces


def _cluster_simplify_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    target_faces: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Simple vertex-clustering simplifier used when quadric decimation is absent."""
    vertices = np.asarray(vertices, dtype=float)
    faces = np.asarray(faces, dtype=int)
    if len(faces) <= target_faces:
        return vertices, faces

    mins = np.min(vertices, axis=0)
    span = np.maximum(np.max(vertices, axis=0) - mins, 1e-9)
    diag = float(np.linalg.norm(span))
    if diag <= 0.0:
        return vertices, faces

    best_vertices, best_faces = vertices, faces
    cell = diag / 240.0
    for _ in range(18):
        keys = np.floor((vertices - mins) / max(cell, 1e-12)).astype(np.int64)
        _, inverse = np.unique(keys, axis=0, return_inverse=True)
        n_new = int(np.max(inverse)) + 1
        clustered = np.zeros((n_new, 3), dtype=float)
        counts = np.zeros(n_new, dtype=float)
        np.add.at(clustered, inverse, vertices)
        np.add.at(counts, inverse, 1.0)
        clustered /= np.maximum(counts[:, None], 1.0)

        remapped = inverse[faces]
        clustered, remapped = _drop_degenerate_faces(clustered, remapped)
        if len(remapped) > 0:
            best_vertices, best_faces = clustered, remapped
        if 0 < len(remapped) <= target_faces:
            return clustered, remapped
        cell *= 1.35

    return best_vertices, best_faces


def _simplify_recovered_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    target_faces: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Reduce recovered mesh size before OCC sewing when safe.

    We only accept a simplified mesh when it remains watertight and winding-
    consistent.  A rough vertex-clustering simplifier can damage topology and
    make OCC produce invalid solids, so dense but clean meshes are preferable
    to fast broken simplifications.
    """
    target_faces = max(64, int(target_faces))
    vertices, faces = _drop_degenerate_faces(
        np.asarray(vertices, dtype=float),
        np.asarray(faces, dtype=int),
    )
    if len(faces) <= target_faces:
        return vertices, faces

    try:
        import trimesh
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        simplified = mesh.simplify_quadric_decimation(face_count=target_faces)
        if simplified is not None and len(simplified.faces) > 0:
            if not simplified.is_watertight or not simplified.is_winding_consistent:
                raise RuntimeError("simplified mesh is not watertight")
            vertices = np.asarray(simplified.vertices, dtype=float)
            faces = np.asarray(simplified.faces, dtype=int)
            vertices, faces = _drop_degenerate_faces(vertices, faces)
            if len(faces) > 0:
                return vertices, faces
    except Exception:
        logger.info(
            "Watertight quadric decimation unavailable; sewing recovered mesh "
            "at original resolution."
        )

    return vertices, faces


def _limit_recovered_mesh_for_cad(
    vertices: np.ndarray,
    faces: np.ndarray,
    max_faces: int,
    *,
    label: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return a bounded-size recovered mesh or raise before OCC work explodes."""
    max_faces = max(64, int(max_faces or _DEFAULT_MAX_SMOOTH_FACES))
    vertices = np.asarray(vertices, dtype=float)
    faces = np.asarray(faces, dtype=int)
    if len(faces) <= max_faces:
        return vertices, faces

    original_faces = len(faces)
    vertices, faces = _simplify_recovered_mesh(vertices, faces, max_faces)
    if len(faces) > max_faces:
        raise RuntimeError(
            f"{label} needs at most {max_faces} recovered triangles, got "
            f"{original_faces}. Use Voxel Boundary, lower mesh_decimate_ratio, "
            "or export the recovered surface as STL."
        )

    logger.info(
        "%s simplified recovered mesh from %d to %d faces.",
        label, original_faces, len(faces),
    )
    return vertices, faces


def _recovered_mesh_to_faceted_brep_solid(
    vertices: np.ndarray,
    faces: np.ndarray,
    sew_tolerance: float = 1e-4,
    max_faces: int = _DEFAULT_MAX_SMOOTH_FACES,
    merge_angle_deg: float = 1.0,
) -> Any:
    """Sew the smoothed recovered triangle mesh into a faceted B-rep solid.

    The resulting STEP is smooth-looking because it follows the recovered
    marching-cubes/postprocessed surface, but it is still a faceted B-rep
    composed of planar triangular faces rather than fitted NURBS patches.
    """
    import cadquery as cq
    from OCP.gp import gp_Pnt
    from OCP.BRepBuilderAPI import (
        BRepBuilderAPI_MakePolygon,
        BRepBuilderAPI_MakeFace,
        BRepBuilderAPI_Sewing,
    )

    vertices = np.asarray(vertices, dtype=float)
    faces = np.asarray(faces, dtype=int)
    vertices, faces = _limit_recovered_mesh_for_cad(
        vertices,
        faces,
        int(max_faces),
        label="Smooth CAD reconstruction",
    )

    sew = BRepBuilderAPI_Sewing(float(sew_tolerance))
    n_faces = 0
    skipped = 0
    for tri in faces:
        try:
            pts = vertices[np.asarray(tri[:3], dtype=int), :3]
        except Exception:
            skipped += 1
            continue
        if not np.all(np.isfinite(pts)):
            skipped += 1
            continue
        area = 0.5 * np.linalg.norm(np.cross(pts[1] - pts[0], pts[2] - pts[0]))
        if area <= 1e-12:
            skipped += 1
            continue

        poly = BRepBuilderAPI_MakePolygon(
            gp_Pnt(float(pts[0, 0]), float(pts[0, 1]), float(pts[0, 2])),
            gp_Pnt(float(pts[1, 0]), float(pts[1, 1]), float(pts[1, 2])),
            gp_Pnt(float(pts[2, 0]), float(pts[2, 1]), float(pts[2, 2])),
            True,
        )
        if not poly.IsDone():
            skipped += 1
            continue
        face_builder = BRepBuilderAPI_MakeFace(poly.Wire(), True)
        if not face_builder.IsDone():
            skipped += 1
            continue
        sew.Add(face_builder.Face())
        n_faces += 1

    if n_faces < 4:
        raise RuntimeError("Recovered mesh did not contain enough valid faces.")

    sew.Perform()
    sewed_shape = sew.SewedShape()
    shells = _shape_to_shells(sewed_shape)
    if not shells:
        raise RuntimeError(
            "Sewing recovered mesh produced no shell; the recovered mesh may be open."
        )
    if len(shells) > 1:
        logger.warning(
            "Smooth CAD reconstruction found %d sewn shells; using the first.",
            len(shells),
        )

    occ_solid = _shell_to_solid(shells[0])
    solid = cq.Solid(occ_solid)
    if not solid.isValid():
        raise RuntimeError("Smooth recovered mesh produced an invalid B-rep solid.")
    try:
        if abs(float(solid.Volume())) <= 1e-12:
            raise RuntimeError("Smooth recovered mesh produced a zero-volume solid.")
    except AttributeError:
        pass

    solid = _unify_same_domain_shape(solid, merge_angle_deg=merge_angle_deg)

    logger.info(
        "Smooth CAD reconstruction: %d recovered triangles sewn into faceted "
        "B-rep solid (%d skipped).",
        n_faces, skipped,
    )
    return solid


# ---------------------------------------------------------------------------
# Input extraction
# ---------------------------------------------------------------------------

def _extract_density_field(
    payload: Any,
) -> Optional[Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray], float]]:
    """Pull (density_3d, (mins, maxs), cutoff) from a topology-opt result dict."""
    if not isinstance(payload, dict):
        return None
    density = payload.get('density')
    if density is None:
        return None
    density = np.asarray(density, dtype=float)
    if density.ndim != 3 or min(density.shape) < 1:
        return None

    bounds_payload = payload.get('bounds')
    if (isinstance(bounds_payload, dict)
            and 'min' in bounds_payload and 'max' in bounds_payload):
        mins = np.asarray(bounds_payload['min'], dtype=float)
        maxs = np.asarray(bounds_payload['max'], dtype=float)
        if mins.size < 3 or maxs.size < 3 or not np.all(maxs[:3] > mins[:3]):
            mins, maxs = np.zeros(3), np.asarray(density.shape, dtype=float)
    else:
        mins, maxs = np.zeros(3), np.asarray(density.shape, dtype=float)

    cutoff = float(payload.get('density_cutoff', 0.4) or 0.4)
    return density, (mins, maxs), cutoff


def reconstruct_topopt_cad(
    payload: Any,
    *,
    source_geometry: str = _AUTO_MODE,
    max_smooth_faces: int = _DEFAULT_MAX_SMOOTH_FACES,
    sew_tolerance: float = 1e-4,
    merge_angle_deg: float = 1.0,
    density_cutoff: float = 0.0,
    sweep_radius_scale: float = 1.0,
    sweep_spur_factor: float = 2.5,
    sweep_teasar_scale: float = 1.5,
    sweep_teasar_const: float = 4.0,
    sweep_max_sections: int = 40,
) -> Any:
    """Reconstruct a CAD B-rep from a topology result without a graph block."""
    mode = str(source_geometry or _AUTO_MODE).strip()
    mode_norm = mode.lower()

    if mode_norm in {
        _NURBS_MODE.lower(),
        "nurbs",
        "fitted nurbs",
        "fitted nurbs surface",
        "nurbs surface",
    }:
        recovered = _extract_recovered_mesh(payload)
        if recovered is None:
            raise RuntimeError(
                "Fitted NURBS Surface needs a topology result with recovered_shape "
                "vertices/faces."
            )

        vertices, faces = recovered
        vertices, faces = _limit_recovered_mesh_for_cad(
            vertices,
            faces,
            int(max_smooth_faces),
            label="Fitted NURBS Surface",
        )
        from pylcss.design_studio.topology_optimization.nurbs_fitting import (
            recovered_mesh_to_nurbs_brep_solid,
        )

        solid = recovered_mesh_to_nurbs_brep_solid(
            vertices,
            faces,
            sew_tolerance=float(sew_tolerance),
            fit_tol=0.0,
            region_angle_deg=30.0,
            max_patches=600,
            merge_angle_deg=float(merge_angle_deg or 0.0),
        )
        import cadquery as cq
        return cq.Workplane(obj=solid)

    if mode_norm == _SWEEP_MODE.lower():
        field = _extract_density_field(payload)
        if field is None:
            raise RuntimeError(
                "Skeleton NURBS Sweep needs a topology result with a voxel density field."
            )
        density, bounds, reported_cutoff = field
        cutoff = float(density_cutoff or 0.0) or reported_cutoff
        try:
            import cadquery as cq
            solid = _skeleton_sweep_to_brep_solid(
                density, bounds, cutoff,
                radius_scale=float(sweep_radius_scale),
                spur_factor=float(sweep_spur_factor),
                teasar_scale=float(sweep_teasar_scale),
                teasar_const=float(sweep_teasar_const),
                max_sections=int(sweep_max_sections),
                merge_angle_deg=float(merge_angle_deg),
            )
            return cq.Workplane(obj=solid)
        except Exception as exc:
            logger.warning(
                "Skeleton NURBS Sweep failed (%s); falling back to voxel-boundary B-rep.",
                exc,
            )
        use_smooth, allow_voxel_fallback = False, True
    else:
        use_smooth = mode_norm != _VOXEL_MODE.lower()
        allow_voxel_fallback = mode_norm in {
            _AUTO_MODE.lower(),
            "automatic",
            "auto",
            "smooth fallback",
        }

    if use_smooth:
        recovered = _extract_recovered_mesh(payload)
        if recovered is not None:
            vertices, faces = recovered
            if allow_voxel_fallback and len(faces) > int(max_smooth_faces):
                logger.warning(
                    "Automatic CAD skipped faceted smooth STEP: recovered mesh "
                    "has %d triangles, limit is %d. Falling back to voxel-boundary "
                    "B-rep.",
                    len(faces), int(max_smooth_faces),
                )
                use_smooth = False
            else:
                try:
                    solid = _recovered_mesh_to_faceted_brep_solid(
                        vertices,
                        faces,
                        sew_tolerance=float(sew_tolerance),
                        max_faces=int(max_smooth_faces),
                        merge_angle_deg=float(merge_angle_deg or 0.0),
                    )
                    solid = _unify_same_domain_shape(solid, merge_angle_deg=float(merge_angle_deg or 0.0))
                    import cadquery as cq
                    return cq.Workplane(obj=solid)
                except Exception as exc:
                    if not allow_voxel_fallback:
                        raise RuntimeError(f"Smooth CAD reconstruction failed: {exc}") from exc
                    logger.warning(
                        "Smooth CAD reconstruction failed (%s); falling back to "
                        "voxel-boundary B-rep.",
                        exc,
                    )
        elif not allow_voxel_fallback:
            raise RuntimeError(
                "Smooth CAD reconstruction needs a topology result with "
                "recovered_shape vertices/faces."
            )

    field = _extract_density_field(payload)
    if field is None:
        raise RuntimeError(
            "CAD reconstruction needs a topology result with recovered_shape "
            "or a voxel density field."
        )

    density, bounds, reported_cutoff = field
    cutoff = float(density_cutoff or 0.0)
    if cutoff <= 0.0:
        cutoff = reported_cutoff
    solid = _voxel_boundary_to_brep_solid(
        density, bounds, cutoff, merge_angle_deg=float(merge_angle_deg or 0.0),
    )
    solid = _cut_payload_void_cylinders(solid, payload, bounds)
    solid = _unify_same_domain_shape(solid, merge_angle_deg=float(merge_angle_deg or 0.0))
    try:
        logger.info(
            "Topology CAD reconstruction: solid valid=%s, volume=%.3f",
            bool(solid.isValid()), float(solid.Volume()),
        )
    except Exception:
        pass

    import cadquery as cq
    return cq.Workplane(obj=solid)


# ---------------------------------------------------------------------------
# Skeleton → variable-radius NURBS sweep (hybrid with voxel B-rep for plates)
# ---------------------------------------------------------------------------
#
# For truss / lattice-like results the compact, smooth, *editable* CAD form is
# a set of swept pipes following the members' centrelines, not a faceted skin.
# We extract a 3-D centreline graph + per-point radius with kimimaro (TEASAR),
# split it into branches, fit each into a lofted variable-radius B-spline solid,
# and boolean-union the members.  Regions the rod model does not cover (plates,
# bulk blocks — whose medial axis is a *surface*, not a curve) are reconstructed
# with the existing voxel-boundary B-rep and unioned in.  This is a hybrid by
# necessity: a swept circular section is the wrong primitive for a plate.
#
# Honest scope: members are approximated with (near-)circular cross-sections, so
# this reproduces the *topology and load paths* faithfully but not the exact
# optimised cross-section.  Best for rod-dominated results; plate-heavy parts
# fall back largely to the faceted residual.


def _solid_from_density(density: np.ndarray, cutoff: float) -> np.ndarray:
    """Threshold a density field to a boolean solid mask (shared fallback rule)."""
    density = np.asarray(density, dtype=float)
    if density.ndim != 3:
        raise RuntimeError("Skeleton sweep expects a 3-D voxel density field.")
    solid = density >= float(cutoff)
    if not np.any(solid):
        nz = density[density > 0.0]
        if nz.size == 0:
            raise RuntimeError("Density field is empty — nothing to reconstruct.")
        solid = density >= float(np.percentile(nz, 50.0))
    return solid


def _skeletonize_solid(
    solid: np.ndarray,
    dust_voxels: int,
    teasar_scale: float,
    teasar_const: float,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """kimimaro TEASAR skeleton → (vertices_vox, edges, radii_vox) merged over labels.

    Vertices are returned in *voxel index* coordinates (anisotropy=1) and radii
    in voxel units; the caller scales both to physical space.
    """
    import kimimaro

    labels = np.ascontiguousarray(solid.astype(np.uint8))
    if int(labels.max()) == 0:
        return None
    skels = kimimaro.skeletonize(
        labels,
        anisotropy=(1.0, 1.0, 1.0),
        dust_threshold=int(max(0, dust_voxels)),
        teasar_params={"scale": float(teasar_scale), "const": float(teasar_const)},
        fix_branching=True,
        fix_borders=True,
        progress=False,
        parallel=1,
    )
    if not skels:
        return None

    all_v, all_e, all_r, offset = [], [], [], 0
    for s in skels.values():
        v = np.asarray(s.vertices, dtype=float)
        e = np.asarray(s.edges, dtype=int)
        r = np.asarray(getattr(s, "radius", getattr(s, "radii", None)), dtype=float)
        if v.shape[0] == 0 or r.shape[0] != v.shape[0]:
            continue
        all_v.append(v)
        all_e.append(e + offset if e.size else e.reshape((0, 2)))
        all_r.append(r)
        offset += v.shape[0]
    if not all_v:
        return None
    edges = np.vstack([e for e in all_e if e.size]) if any(e.size for e in all_e) \
        else np.zeros((0, 2), dtype=int)
    return np.vstack(all_v), edges, np.concatenate(all_r)


def _trace_skeleton_branches(
    vertices: np.ndarray,
    edges: np.ndarray,
) -> list:
    """Split a skeleton graph into branches: ordered vertex-index polylines
    between endpoints (degree 1) and junctions (degree ≥ 3)."""
    from collections import defaultdict

    adj: Dict[int, set] = defaultdict(set)
    for a, b in edges:
        a, b = int(a), int(b)
        if a != b:
            adj[a].add(b)
            adj[b].add(a)
    if not adj:
        return []

    deg = {v: len(ns) for v, ns in adj.items()}
    special = {v for v, d in deg.items() if d != 2}
    visited = set()

    def ekey(a: int, b: int) -> Tuple[int, int]:
        return (a, b) if a < b else (b, a)

    branches: list = []
    starts = list(special) if special else [next(iter(adj))]
    for sv in starts:
        for nb in list(adj[sv]):
            if ekey(sv, nb) in visited:
                continue
            path = [sv, nb]
            visited.add(ekey(sv, nb))
            prev, cur = sv, nb
            while deg.get(cur, 0) == 2 and cur not in special:
                nxts = [x for x in adj[cur] if x != prev]
                if not nxts or ekey(cur, nxts[0]) in visited:
                    break
                nxt = nxts[0]
                visited.add(ekey(cur, nxt))
                path.append(nxt)
                prev, cur = cur, nxt
            branches.append(np.asarray(path, dtype=int))

    # Pure cycles (no special vertex): walk until we return to the start.
    for a in list(adj):
        for b in list(adj[a]):
            if ekey(a, b) in visited:
                continue
            path = [a, b]
            visited.add(ekey(a, b))
            prev, cur = a, b
            while cur != a:
                nxts = [x for x in adj[cur] if x != prev and ekey(cur, x) not in visited]
                if not nxts:
                    break
                nxt = nxts[0]
                visited.add(ekey(cur, nxt))
                path.append(nxt)
                prev, cur = cur, nxt
            branches.append(np.asarray(path, dtype=int))
    return branches


def _smooth_polyline(coords: np.ndarray, iterations: int = 2) -> np.ndarray:
    """Light moving-average smoothing of an open polyline; endpoints fixed."""
    c = np.asarray(coords, dtype=float).copy()
    if len(c) < 3:
        return c
    for _ in range(max(0, iterations)):
        nc = c.copy()
        nc[1:-1] = (c[:-2] + c[1:-1] + c[2:]) / 3.0
        c = nc
    return c


def _rotation_minimizing_frames(
    points: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Tangents + a non-twisting reference normal per point (double-reflection RMF).

    Used to orient each sweep circle so the lofted seams stay aligned and the
    pipe surface does not twist between sections.
    """
    pts = np.asarray(points, dtype=float)
    n = len(pts)
    t = np.zeros_like(pts)
    if n >= 3:
        t[1:-1] = pts[2:] - pts[:-2]
    t[0] = pts[1] - pts[0]
    t[-1] = pts[-1] - pts[-2]
    norm = np.linalg.norm(t, axis=1, keepdims=True)
    norm[norm < 1e-12] = 1.0
    t /= norm

    seed = np.array([1.0, 0.0, 0.0]) if abs(t[0, 0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    r0 = seed - np.dot(seed, t[0]) * t[0]
    n0 = np.linalg.norm(r0)
    r0 = r0 / n0 if n0 > 1e-12 else np.array([0.0, 1.0, 0.0])

    refs = np.zeros_like(pts)
    refs[0] = r0
    for i in range(n - 1):
        v1 = pts[i + 1] - pts[i]
        c1 = float(np.dot(v1, v1))
        if c1 < 1e-18:
            refs[i + 1] = refs[i]
            continue
        r_l = refs[i] - (2.0 / c1) * np.dot(v1, refs[i]) * v1
        t_l = t[i] - (2.0 / c1) * np.dot(v1, t[i]) * v1
        v2 = t[i + 1] - t_l
        c2 = float(np.dot(v2, v2))
        r = r_l if c2 < 1e-18 else r_l - (2.0 / c2) * np.dot(v2, r_l) * v2
        nr = np.linalg.norm(r)
        refs[i + 1] = r / nr if nr > 1e-12 else refs[i]
    return t, refs


def _branch_to_member_solid(
    coords: np.ndarray,
    radii: np.ndarray,
    *,
    max_sections: int = 40,
    min_radius: float = 1e-3,
) -> Any:
    """Loft variable-radius circles along a centreline into one smooth pipe solid."""
    import cadquery as cq
    from OCP.gp import gp_Pnt, gp_Dir, gp_Ax2, gp_Circ
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeWire

    coords = np.asarray(coords, dtype=float)
    radii = np.asarray(radii, dtype=float)
    if len(coords) < 2:
        return None

    # Drop coincident stations (would make zero-length tangents / loft failures).
    keep = [0]
    for i in range(1, len(coords)):
        if np.linalg.norm(coords[i] - coords[keep[-1]]) > 1e-6:
            keep.append(i)
    coords, radii = coords[keep], radii[keep]
    if len(coords) < 2:
        return None

    coords = _smooth_polyline(coords)
    radii = np.maximum(radii, float(min_radius))

    if len(coords) > int(max_sections):
        sel = np.unique(np.linspace(0, len(coords) - 1, int(max_sections)).round().astype(int))
        coords, radii = coords[sel], radii[sel]

    tangents, refs = _rotation_minimizing_frames(coords)
    wires = []
    for c, tg, rx, r in zip(coords, tangents, refs, radii):
        try:
            ax = gp_Ax2(
                gp_Pnt(float(c[0]), float(c[1]), float(c[2])),
                gp_Dir(float(tg[0]), float(tg[1]), float(tg[2])),
                gp_Dir(float(rx[0]), float(rx[1]), float(rx[2])),
            )
            edge = BRepBuilderAPI_MakeEdge(gp_Circ(ax, float(r))).Edge()
            wire = BRepBuilderAPI_MakeWire(edge).Wire()
            wires.append(cq.Wire(wire))
        except Exception:
            continue
    if len(wires) < 2:
        return None

    solid = None
    for ruled in (False, True):
        try:
            cand = cq.Solid.makeLoft(wires, ruled=ruled)
            if cand is not None and cand.isValid() and cand.Volume() > 1e-12:
                solid = cand
                break
        except Exception:
            continue
    return solid


def _skeleton_coverage_mask(
    shape: Tuple[int, int, int],
    vertices_vox: np.ndarray,
    radii_vox: np.ndarray,
    scale: float = 1.15,
) -> np.ndarray:
    """Voxels within (scale × nearest-skeleton-radius) of the centreline graph.

    This is what the swept rods are expected to cover; the complement inside the
    solid is treated as plate/bulk and handed to the voxel B-rep instead.
    """
    from scipy.ndimage import distance_transform_edt

    shape = tuple(int(s) for s in shape)
    skel = np.zeros(shape, dtype=bool)
    rad_grid = np.zeros(shape, dtype=float)

    idx = np.round(vertices_vox).astype(int)
    inb = np.all((idx >= 0) & (idx < np.array(shape)), axis=1)
    idx, rv = idx[inb], radii_vox[inb]
    if idx.shape[0] == 0:
        return np.zeros(shape, dtype=bool)
    ii, jj, kk = idx[:, 0], idx[:, 1], idx[:, 2]
    skel[ii, jj, kk] = True
    np.maximum.at(rad_grid, (ii, jj, kk), rv)

    edt, ind = distance_transform_edt(~skel, return_indices=True)
    nearest_r = rad_grid[ind[0], ind[1], ind[2]]
    return edt <= nearest_r * float(scale)


def _union_solids(solids: list) -> Any:
    """Boolean-union solids into one clean solid; on failure, a flat compound.

    Flattens any input compounds to their leaf solids first (never nests), does
    a single n-ary boolean (much more robust than incremental pairwise fusing on
    many overlapping members), then merges redundant faces with ``clean()``.
    """
    import cadquery as cq

    flat: list = []
    for s in solids:
        if s is None:
            continue
        try:
            leaves = s.Solids()
        except Exception:
            leaves = None
        if leaves:
            flat.extend(leaves)
        else:
            flat.append(s)
    if not flat:
        return None
    if len(flat) == 1:
        return flat[0]

    try:
        fused = flat[0].fuse(*flat[1:])
        if fused is not None and fused.isValid() and fused.Volume() > 1e-9:
            try:
                fused = fused.clean()  # UnifySameDomain — merges redundant faces
            except Exception:
                pass
            return fused
    except Exception:
        pass

    # Lossless fallback: a flat compound (a STEP can hold multiple solids).
    try:
        return cq.Compound.makeCompound(flat)
    except Exception:
        return flat[0]


def _prune_skeleton(
    vertices: np.ndarray,
    edges: np.ndarray,
    radii: np.ndarray,
    mean_cell: float,
    spur_factor: float = 2.5,
    max_passes: int = 25,
) -> np.ndarray:
    """Iteratively delete short leaf chains (spurs) from a skeleton edge graph.

    TEASAR over-segments bulk/plate regions into many tiny branches and splits
    straight members at spurious junctions.  Removing leaf chains shorter than
    ``spur_factor × local radius`` collapses that fuzz, leaving the genuine
    members; degree-2 continuations are re-merged downstream by the tracer.
    Returns a reduced (M, 2) edge array.
    """
    from collections import defaultdict

    adj: Dict[int, set] = defaultdict(set)
    for a, b in edges:
        a, b = int(a), int(b)
        if a != b:
            adj[a].add(b)
            adj[b].add(a)

    for _ in range(max_passes):
        leaves = [v for v, ns in adj.items() if len(ns) == 1]
        removed = False
        for lf in leaves:
            if lf not in adj or len(adj[lf]) != 1:
                continue
            chain = [lf]
            prev, cur = None, lf
            while True:
                nxts = [x for x in adj[cur] if x != prev]
                if len(nxts) != 1:
                    break
                nxt = nxts[0]
                chain.append(nxt)
                if len(adj[nxt]) != 2:  # reached a junction or another endpoint
                    break
                prev, cur = cur, nxt
            if len(chain) < 2:
                continue
            pts = vertices[chain]
            length = float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1))) * mean_cell
            rad = float(np.mean(radii[chain])) * mean_cell
            if length < spur_factor * max(rad, 1e-9):
                # Delete the spur but keep its terminal (junction) vertex.
                for i in range(len(chain) - 1):
                    u, w = chain[i], chain[i + 1]
                    adj[u].discard(w)
                    adj[w].discard(u)
                    if not adj[u]:
                        adj.pop(u, None)
                removed = True
        if not removed:
            break

    out = set()
    for v, ns in adj.items():
        for w in ns:
            out.add((v, w) if v < w else (w, v))
    return np.asarray(sorted(out), dtype=int) if out else np.zeros((0, 2), dtype=int)


def _skeleton_sweep_to_brep_solid(
    density: np.ndarray,
    bounds: Tuple[np.ndarray, np.ndarray],
    cutoff: float,
    *,
    radius_scale: float = 1.0,
    max_sections: int = 40,
    dust_voxels: int = 30,
    teasar_scale: float = 1.5,
    teasar_const: float = 4.0,
    spur_factor: float = 2.5,
    coverage_scale: float = 1.15,
    residual_min_voxels: int = 8,
    merge_angle_deg: float = 1.0,
) -> Any:
    """Reconstruct a topology result as swept NURBS members + a voxel-B-rep residual."""
    solid = _solid_from_density(density, cutoff)
    mins = np.asarray(bounds[0], dtype=float)
    maxs = np.asarray(bounds[1], dtype=float)
    cell = (maxs - mins) / np.asarray(solid.shape, dtype=float)
    mean_cell = float(np.mean(cell))

    sk = _skeletonize_solid(solid, dust_voxels, teasar_scale, teasar_const)
    if sk is None:
        raise RuntimeError("Skeletonisation produced no centreline (try a lower cutoff).")
    verts, edges, radii = sk
    edges = _prune_skeleton(verts, edges, radii, mean_cell, spur_factor=float(spur_factor))

    members, skipped = [], 0
    for br in _trace_skeleton_branches(verts, edges):
        if len(br) < 2:
            continue
        coords = mins + verts[br] * cell
        rad = np.maximum(radii[br] * mean_cell * float(radius_scale), mean_cell * 0.25)
        member = _branch_to_member_solid(coords, rad, max_sections=int(max_sections))
        if member is None:
            skipped += 1
        else:
            members.append(member)

    rods = _union_solids(members)

    # Coverage from the surviving (pruned) skeleton only, so spurs deleted from
    # plate/bulk regions don't mask those voxels — they fall to the residual.
    used = np.unique(edges) if edges.size else np.array([], dtype=int)
    cov_verts = verts[used] if used.size else verts[:0]
    cov_radii = radii[used] if used.size else radii[:0]
    covered = _skeleton_coverage_mask(solid.shape, cov_verts, cov_radii, scale=float(coverage_scale))
    residual = solid & (~covered)
    n_residual = int(np.count_nonzero(residual))
    plate = None
    if n_residual >= int(residual_min_voxels):
        try:
            plate = _voxel_boundary_to_brep_solid(
                residual.astype(float), bounds, 0.5, merge_angle_deg=float(merge_angle_deg),
            )
        except Exception:
            logger.warning("Skeleton sweep: residual (plate) B-rep failed; rods only.")
            plate = None

    pieces = [p for p in (rods, plate) if p is not None]
    result = _union_solids(pieces)
    if result is None:
        raise RuntimeError("Skeleton sweep produced no geometry.")

    logger.info(
        "Skeleton sweep: %d members (%d skipped), %d residual voxels -> %s",
        len(members), skipped, n_residual,
        "rods+plate" if plate is not None else "rods only",
    )
    return result


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

class _TopologyCadImplementation(CadQueryNode):
    """Reconstruct a CAD B-rep solid from a topology-optimisation result.

    Input:
        topology_result — connect the Topology Opt (Voxel) node output (its
                          full result dict with recovered_shape and density).

    Output:
        shape — a CadQuery solid; connect to Export STEP, or any boolean /
                fillet / detailing node in the graph.
    """
    __identifier__ = 'internal.topopt_cad'
    NODE_NAME = 'Internal Topology CAD'

    def __init__(self):
        super().__init__()
        self.add_input('topology_result', color=(180, 255, 180))
        self.add_output('shape', color=(100, 255, 100))

        self.create_property('source_geometry', _AUTO_MODE, widget_type='combo',
                             items=[
                                 _AUTO_MODE,
                                 "NURBS Surface",
                                 _SMOOTH_MODE,
                                 _NURBS_MODE,
                                 _VOXEL_MODE,
                                 _SWEEP_MODE,
                             ])
        # Guard for sewing dense marching-cubes meshes into STEP. Lower the
        # TopOpt mesh_decimate_ratio if this trips.
        self.create_property('max_smooth_faces', _DEFAULT_MAX_SMOOTH_FACES, widget_type='int')
        self.create_property('sew_tolerance', 1e-4, widget_type='float')
        # Coplanar faces within this angle are merged.  Voxel walls are exactly
        # axis-aligned so a small tolerance suffices; 0 disables merging.
        self.create_property('merge_angle_deg', 1.0, widget_type='float')
        # Threshold for "solid"; 0 = use the cutoff the optimiser reported.
        self.create_property('density_cutoff', 0.0, widget_type='float')

        # ── Skeleton NURBS Sweep controls (used by the _SWEEP_MODE) ──────────
        # Multiply every member's recovered radius (1.0 = as-measured).
        self.create_property('sweep_radius_scale', 1.0, widget_type='float')
        # Prune leaf branches shorter than this × local radius (kills skeleton
        # fuzz / spurs; raise to merge more, lower to keep fine members).
        self.create_property('sweep_spur_factor', 2.5, widget_type='float')
        # kimimaro TEASAR pruning: larger = fewer/longer members.
        self.create_property('sweep_teasar_scale', 1.5, widget_type='float')
        self.create_property('sweep_teasar_const', 4.0, widget_type='float')
        # Max circle sections lofted per member (caps cost on long branches).
        self.create_property('sweep_max_sections', 40, widget_type='int')

    def run(self):
        port = self.get_input('topology_result')
        payload = resolve_any_input(port) if port and port.connected_ports() else None
        if payload is None:
            payload = self.get_input_value('topology_result', None)

        mode = str(self.get_property('source_geometry') or _AUTO_MODE).strip()
        mode_norm = mode.lower()

        if mode_norm in {
            _NURBS_MODE.lower(),
            "nurbs",
            "fitted nurbs",
            "fitted nurbs surface",
            "nurbs surface",
        }:
            recovered = _extract_recovered_mesh(payload)
            if recovered is None:
                self.set_error(
                    "Fitted NURBS Surface needs a topology result with "
                    "recovered_shape vertices/faces."
                )
                return None
            try:
                vertices, faces = recovered
                vertices, faces = _limit_recovered_mesh_for_cad(
                    vertices,
                    faces,
                    int(self.get_property('max_smooth_faces') or _DEFAULT_MAX_SMOOTH_FACES),
                    label="Fitted NURBS Surface",
                )
                from pylcss.design_studio.topology_optimization.nurbs_fitting import (
                    recovered_mesh_to_nurbs_brep_solid,
                )

                solid = recovered_mesh_to_nurbs_brep_solid(
                    vertices,
                    faces,
                    sew_tolerance=float(self.get_property('sew_tolerance') or 1e-4),
                    fit_tol=0.0,
                    region_angle_deg=30.0,
                    max_patches=600,
                    merge_angle_deg=float(self.get_property('merge_angle_deg') or 0.0),
                )
                import cadquery as cq
                return cq.Workplane(obj=solid)
            except Exception as exc:
                logger.exception("Fitted NURBS Surface reconstruction failed")
                self.set_error(f"Fitted NURBS Surface reconstruction failed: {exc}")
                return None

        if mode_norm == _SWEEP_MODE.lower():
            field = _extract_density_field(payload)
            if field is None:
                self.set_error(
                    "Skeleton NURBS Sweep needs a topology result with a voxel "
                    "density field. Connect the Topology Opt (Voxel) node output."
                )
                return None
            density, bounds, reported_cutoff = field
            cutoff = float(self.get_property('density_cutoff') or 0.0) or reported_cutoff
            try:
                import cadquery as cq
                solid = _skeleton_sweep_to_brep_solid(
                    density, bounds, cutoff,
                    radius_scale=float(self.get_property('sweep_radius_scale') or 1.0),
                    spur_factor=float(self.get_property('sweep_spur_factor') or 2.5),
                    teasar_scale=float(self.get_property('sweep_teasar_scale') or 1.5),
                    teasar_const=float(self.get_property('sweep_teasar_const') or 4.0),
                    max_sections=int(self.get_property('sweep_max_sections') or 40),
                    merge_angle_deg=float(self.get_property('merge_angle_deg') or 0.0),
                )
                return cq.Workplane(obj=solid)
            except Exception as exc:
                logger.warning(
                    "Skeleton NURBS Sweep failed (%s); falling back to "
                    "voxel-boundary B-rep.", exc,
                )
            # Fall through to the robust voxel-boundary path on sweep failure.
            use_smooth, allow_voxel_fallback = False, True
        else:
            use_smooth = mode_norm != _VOXEL_MODE.lower()
            allow_voxel_fallback = mode_norm in {
                _AUTO_MODE.lower(),
                "auto",
                "smooth fallback",
            }

        if use_smooth:
            recovered = _extract_recovered_mesh(payload)
            if recovered is not None:
                vertices, faces = recovered
                max_faces = int(
                    self.get_property('max_smooth_faces') or _DEFAULT_MAX_SMOOTH_FACES
                )
                if allow_voxel_fallback and len(faces) > max_faces:
                    logger.warning(
                        "Automatic CAD skipped faceted smooth STEP: recovered mesh "
                        "has %d triangles, limit is %d. Falling back to "
                        "voxel-boundary B-rep.",
                        len(faces), max_faces,
                    )
                else:
                    try:
                        solid = _recovered_mesh_to_faceted_brep_solid(
                            vertices,
                            faces,
                            sew_tolerance=float(self.get_property('sew_tolerance') or 1e-4),
                            max_faces=max_faces,
                            merge_angle_deg=float(self.get_property('merge_angle_deg') or 0.0),
                        )
                        solid = _unify_same_domain_shape(
                            solid,
                            merge_angle_deg=float(self.get_property('merge_angle_deg') or 0.0),
                        )
                        import cadquery as cq
                        return cq.Workplane(obj=solid)
                    except Exception as exc:
                        if not allow_voxel_fallback:
                            logger.exception("Smooth CAD reconstruction failed")
                            self.set_error(f"Smooth CAD reconstruction failed: {exc}")
                            return None
                        logger.warning(
                            "Smooth CAD reconstruction failed (%s); falling back to "
                            "voxel-boundary B-rep.",
                            exc,
                        )
            elif not allow_voxel_fallback:
                self.set_error(
                    "Smooth CAD reconstruction needs a topology result with "
                    "recovered_shape vertices/faces."
                )
                return None

        field = _extract_density_field(payload)
        if field is None:
            self.set_error(
                "Topology Opt → CAD needs a topology-opt result with a density "
                "field. Connect the Topology Opt (Voxel) node output."
            )
            return None

        density, bounds, reported_cutoff = field
        cutoff = float(self.get_property('density_cutoff') or 0.0)
        if cutoff <= 0.0:
            cutoff = reported_cutoff
        merge_angle = float(self.get_property('merge_angle_deg') or 0.0)

        try:
            solid = _voxel_boundary_to_brep_solid(
                density, bounds, cutoff, merge_angle_deg=merge_angle,
            )
            solid = _cut_payload_void_cylinders(solid, payload, bounds)
            solid = _unify_same_domain_shape(solid, merge_angle_deg=merge_angle)
        except RuntimeError as exc:
            self.set_error(str(exc))
            return None
        except Exception as exc:  # OCC can raise low-level errors
            logger.exception("CAD reconstruction failed")
            self.set_error(f"CAD reconstruction failed: {exc}")
            return None

        try:
            logger.info(
                "Topology Opt → CAD: solid valid=%s, volume=%.3f",
                bool(solid.isValid()), float(solid.Volume()),
            )
        except Exception:
            pass

        import cadquery as cq
        return cq.Workplane(obj=solid)
