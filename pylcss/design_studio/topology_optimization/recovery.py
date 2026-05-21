# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.
"""Marching-cubes surface recovery and smoothing for the voxel-optimised field."""
from __future__ import annotations

import logging
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

CylinderRegion = Tuple[str, float, float, float, float, float]
BoxRegion = Tuple[float, float, float, float, float, float]


def _split_cylinder_region(cylinder) -> Optional[Tuple[str, float, float, float, float, float, float]]:
    if cylinder is None or len(cylinder) < 6:
        return None
    axis, c0, c1, lo, hi, radius_a = cylinder[:6]
    radius_b = cylinder[6] if len(cylinder) > 6 else radius_a
    radius_a = float(radius_a)
    radius_b = float(radius_b)
    if radius_a <= 0.0 or radius_b <= 0.0:
        return None
    return (
        str(axis or 'z').lower(),
        float(c0),
        float(c1),
        float(lo),
        float(hi),
        radius_a,
        radius_b,
    )


def _voxel_origin_cell(
    shape: Tuple[int, int, int],
    bounds: Optional[Tuple[np.ndarray, np.ndarray]],
) -> Tuple[np.ndarray, np.ndarray]:
    """Return physical origin and cell size for a structured voxel grid."""
    shape_arr = np.maximum(np.asarray(shape, dtype=float), 1.0)
    if bounds is not None:
        mins, maxs = bounds
        mins = np.asarray(mins, dtype=float)
        maxs = np.asarray(maxs, dtype=float)
        if mins.size >= 3 and maxs.size >= 3 and np.all(maxs[:3] > mins[:3]):
            return mins[:3], (maxs[:3] - mins[:3]) / shape_arr
    return -0.5 * shape_arr, np.ones(3, dtype=float)


def _regularize_extruded_density(grid: np.ndarray, axis: str) -> np.ndarray:
    """Force a density field to be constant along an extrusion axis."""
    axis_map = {'x': 0, 'y': 1, 'z': 2}
    ax = axis_map.get(str(axis or '').strip().lower())
    if ax is None or grid.ndim != 3:
        return grid
    profile = np.mean(grid, axis=ax, keepdims=True)
    return np.broadcast_to(profile, grid.shape).copy()


def _project_extruded_planes(
    vertices: np.ndarray,
    bounds: Optional[Tuple[np.ndarray, np.ndarray]],
    axis: str,
    tolerance: float,
) -> np.ndarray:
    """Snap top/bottom vertices of an extruded result back onto exact planes."""
    axis_map = {'x': 0, 'y': 1, 'z': 2}
    ax = axis_map.get(str(axis or '').strip().lower())
    if ax is None or bounds is None or len(vertices) == 0:
        return vertices

    mins, maxs = bounds
    lo = float(np.asarray(mins, dtype=float)[ax])
    hi = float(np.asarray(maxs, dtype=float)[ax])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return vertices

    out = np.asarray(vertices, dtype=float).copy()
    tol = max(float(tolerance), 1e-9)
    out[out[:, ax] <= lo + tol, ax] = lo
    out[out[:, ax] >= hi - tol, ax] = hi
    return out


def _project_passive_cylinder_surfaces(
    vertices: np.ndarray,
    bounds: Optional[Tuple[np.ndarray, np.ndarray]],
    cylinders: Sequence[CylinderRegion],
    tolerance: float,
) -> np.ndarray:
    """Snap mesh vertices near passive cylinder walls back to analytic radii."""
    if bounds is None or len(vertices) == 0 or not cylinders:
        return vertices

    mins, maxs = bounds
    mins = np.asarray(mins, dtype=float)[:3]
    maxs = np.asarray(maxs, dtype=float)[:3]
    span = np.maximum(maxs - mins, 1e-12)
    out = np.asarray(vertices, dtype=float).copy()
    frac = (out[:, :3] - mins[None, :]) / span[None, :]

    axis_map = {'x': (0, 1, 2), 'y': (1, 0, 2), 'z': (2, 0, 1)}
    for cylinder in cylinders or ():
        parsed = _split_cylinder_region(cylinder)
        if parsed is None:
            continue
        axis, c0, c1, lo, hi, radius_a, radius_b = parsed
        if axis not in axis_map:
            continue
        axial_idx, a_idx, b_idx = axis_map[axis]
        lo, hi = sorted((float(lo), float(hi)))

        # Convert the physical smoothing tolerance to the normalized fractional
        # space used by passive-region definitions.
        tol_frac = max(
            1e-4,
            float(tolerance) / max(min(float(span[a_idx]), float(span[b_idx])), 1e-12),
        )

        da = frac[:, a_idx] - float(c0)
        db = frac[:, b_idx] - float(c1)
        dist = np.sqrt((da / radius_a) ** 2 + (db / radius_b) ** 2)
        mask = (
            (frac[:, axial_idx] >= lo - tol_frac)
            & (frac[:, axial_idx] <= hi + tol_frac)
            & (dist > 1e-9)
            & (np.abs(dist - 1.0) <= tol_frac / max(min(radius_a, radius_b), 1e-12))
        )
        if not np.any(mask):
            continue
        scale = 1.0 / np.maximum(dist[mask], 1e-12)
        frac[mask, a_idx] = float(c0) + da[mask] * scale
        frac[mask, b_idx] = float(c1) + db[mask] * scale

    out[:, :3] = mins[None, :] + frac * span[None, :]
    return out


def _enhanced_mesh_postprocess(
    verts: np.ndarray,
    faces: np.ndarray,
    decimate_ratio: float = 1.0,
    smoothing_iterations: int = 2,
) -> Optional[Dict[str, np.ndarray]]:
    """Run a print-ready post-processing pipeline using trimesh.

    Pipeline:
        1. Split into connected components, keep ones above 1% of max volume.
        2. Fill small holes.
        3. Light Humphrey smoothing (volume-preserving alternative to Laplacian).
        4. Optional quadric decimation (requires `fast_simplification`).

    Returns None if trimesh is unavailable so the caller can fall back to the
    legacy Taubin path.
    """
    try:
        import trimesh
        import trimesh.smoothing
    except ImportError:
        return None
    if len(verts) == 0 or len(faces) == 0:
        return None

    mesh = trimesh.Trimesh(
        vertices=np.asarray(verts, dtype=float),
        faces=np.asarray(faces, dtype=np.int64),
        process=False,
    )

    # 1. Keep only meaningful connected components.
    try:
        components = mesh.split(only_watertight=False)
    except Exception:
        components = [mesh]
    if components:
        volumes = [abs(float(c.volume)) for c in components]
        if volumes:
            max_vol = max(volumes)
            kept = [c for c, v in zip(components, volumes)
                    if v >= 0.01 * max_vol]
            if kept:
                mesh = trimesh.util.concatenate(kept)

    # 2. Close small holes.
    try:
        mesh.fill_holes()
    except Exception:
        logger.debug("trimesh.fill_holes failed; continuing")

    # 3. Volume-preserving Humphrey smoothing.
    try:
        trimesh.smoothing.filter_humphrey(
            mesh,
            alpha=0.1,
            beta=0.5,
            iterations=max(1, int(smoothing_iterations)),
        )
    except Exception:
        logger.debug("Humphrey smoothing failed; falling back to Taubin")
        try:
            trimesh.smoothing.filter_taubin(
                mesh, iterations=max(1, int(smoothing_iterations))
            )
        except Exception:
            pass

    # 4. Optional decimation (requires `fast_simplification`).
    if 0.0 < float(decimate_ratio) < 1.0 and len(mesh.faces) > 0:
        target = max(64, int(len(mesh.faces) * float(decimate_ratio)))
        try:
            decimated = mesh.simplify_quadric_decimation(face_count=target)
            if decimated is not None and len(decimated.faces) > 0:
                mesh = decimated
        except ImportError:
            logger.info(
                "STL decimation skipped — install `fast_simplification` for "
                "quadric decimation support."
            )
        except Exception:
            logger.debug("Decimation failed; keeping un-decimated mesh")

    return {
        'vertices': np.asarray(mesh.vertices, dtype=float),
        'faces':    np.asarray(mesh.faces,    dtype=int),
    }


def _taubin_smooth_surface(
    verts: np.ndarray,
    faces: np.ndarray,
    iterations: int = 6,
) -> np.ndarray:
    """Light volume-preserving smoothing for marching-cubes output."""
    if len(verts) == 0 or len(faces) == 0 or iterations <= 0:
        return verts

    verts = np.asarray(verts, dtype=float).copy()
    faces = np.asarray(faces, dtype=int)
    edges = np.vstack([
        faces[:, [0, 1]],
        faces[:, [1, 2]],
        faces[:, [2, 0]],
    ])

    for _ in range(max(0, int(iterations))):
        for factor in (0.5, -0.53):
            neighbor_sum = np.zeros_like(verts)
            neighbor_count = np.zeros(len(verts), dtype=float)
            np.add.at(neighbor_sum, edges[:, 0], verts[edges[:, 1]])
            np.add.at(neighbor_count, edges[:, 0], 1.0)
            np.add.at(neighbor_sum, edges[:, 1], verts[edges[:, 0]])
            np.add.at(neighbor_count, edges[:, 1], 1.0)
            mask = neighbor_count > 0
            avg = neighbor_sum[mask] / neighbor_count[mask, None]
            verts[mask] += factor * (avg - verts[mask])
    return verts


def _axis_slice_from_fraction(
    coords: np.ndarray,
    lo: float,
    hi: float,
) -> slice:
    mask = (coords >= float(lo)) & (coords <= float(hi))
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return slice(0, 0)
    return slice(int(idx[0]), int(idx[-1]) + 1)


def _fractional_axes(shape: Tuple[int, int, int], pad: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return fractional sample coordinates for a padded or unpadded field."""
    nx, ny, nz = [max(1, int(v)) for v in shape]
    return (
        (np.arange(nx, dtype=float) - float(pad) + 0.5) / max(nx - 2 * int(pad), 1),
        (np.arange(ny, dtype=float) - float(pad) + 0.5) / max(ny - 2 * int(pad), 1),
        (np.arange(nz, dtype=float) - float(pad) + 0.5) / max(nz - 2 * int(pad), 1),
    )


def _apply_passive_density_regions(
    field: np.ndarray,
    *,
    solid_boxes: Sequence[BoxRegion] = (),
    void_boxes: Sequence[BoxRegion] = (),
    solid_cylinders: Sequence[CylinderRegion] = (),
    void_cylinders: Sequence[CylinderRegion] = (),
) -> np.ndarray:
    """Clamp passive regions on the high-resolution scalar field.

    The optimiser stores circular non-design regions analytically, but the
    density array itself is voxelised. Re-applying the analytic regions here
    prevents circular holes from becoming low-sided polygons during export.
    """
    if field.ndim != 3:
        return field

    x, y, z = _fractional_axes(tuple(field.shape), pad=0)

    def apply_box(box: BoxRegion, value: float) -> None:
        x0, x1, y0, y1, z0, z1 = [float(v) for v in box]
        xs = _axis_slice_from_fraction(x, min(x0, x1), max(x0, x1))
        ys = _axis_slice_from_fraction(y, min(y0, y1), max(y0, y1))
        zs = _axis_slice_from_fraction(z, min(z0, z1), max(z0, z1))
        field[xs, ys, zs] = float(value)

    def apply_cylinder(cylinder: CylinderRegion, value: float) -> None:
        parsed = _split_cylinder_region(cylinder)
        if parsed is None:
            return
        axis, c0, c1, lo, hi, radius_a, radius_b = parsed

        if axis == 'x':
            axial = x
            sl = [_axis_slice_from_fraction(axial, lo, hi), slice(None), slice(None)]
            dist2 = ((y[:, None] - c0) / radius_a) ** 2 + ((z[None, :] - c1) / radius_b) ** 2
            mask = dist2 <= 1.0
            block = field[tuple(sl)]
            if block.size:
                np.copyto(block, float(value), where=np.broadcast_to(mask[None, :, :], block.shape))
        elif axis == 'y':
            axial = y
            sl = [slice(None), _axis_slice_from_fraction(axial, lo, hi), slice(None)]
            dist2 = ((x[:, None] - c0) / radius_a) ** 2 + ((z[None, :] - c1) / radius_b) ** 2
            mask = dist2 <= 1.0
            block = field[tuple(sl)]
            if block.size:
                np.copyto(block, float(value), where=np.broadcast_to(mask[:, None, :], block.shape))
        else:
            axial = z
            sl = [slice(None), slice(None), _axis_slice_from_fraction(axial, lo, hi)]
            dist2 = ((x[:, None] - c0) / radius_a) ** 2 + ((y[None, :] - c1) / radius_b) ** 2
            mask = dist2 <= 1.0
            block = field[tuple(sl)]
            if block.size:
                np.copyto(block, float(value), where=np.broadcast_to(mask[:, :, None], block.shape))

    for box in solid_boxes or ():
        apply_box(box, 1.0)
    for cylinder in solid_cylinders or ():
        apply_cylinder(cylinder, 1.0)
    # Void wins where regions overlap, e.g. an annular solid around a hole.
    for box in void_boxes or ():
        apply_box(box, 0.0)
    for cylinder in void_cylinders or ():
        apply_cylinder(cylinder, 0.0)
    return field


def _apply_passive_cylinder_sdf(
    signed_distance: np.ndarray,
    *,
    pad: int,
    solid_cylinders: Sequence[CylinderRegion] = (),
    void_cylinders: Sequence[CylinderRegion] = (),
) -> np.ndarray:
    """Re-impose analytic circular cylinder boundaries on the SDF.

    `signed_distance` uses negative values for material and positive values for
    void. A solid cylinder is therefore a min operation; a through-hole is a max
    operation. Applying this after SDF smoothing keeps holes round instead of
    preserving the voxel-grid polygon.
    """
    if signed_distance.ndim != 3:
        return signed_distance

    x, y, z = _fractional_axes(tuple(signed_distance.shape), pad=int(pad))

    def apply(cylinder: CylinderRegion, solid: bool) -> None:
        parsed = _split_cylinder_region(cylinder)
        if parsed is None:
            return
        axis, c0, c1, lo, hi, radius_a, radius_b = parsed

        if axis == 'x':
            sl = [_axis_slice_from_fraction(x, lo, hi), slice(None), slice(None)]
            dist = np.sqrt(((y[:, None] - c0) / radius_a) ** 2 + ((z[None, :] - c1) / radius_b) ** 2)
            cyl_sdf = dist - 1.0 if solid else 1.0 - dist
            block = signed_distance[tuple(sl)]
            if block.size:
                op = np.minimum if solid else np.maximum
                op(block, cyl_sdf[None, :, :], out=block)
        elif axis == 'y':
            sl = [slice(None), _axis_slice_from_fraction(y, lo, hi), slice(None)]
            dist = np.sqrt(((x[:, None] - c0) / radius_a) ** 2 + ((z[None, :] - c1) / radius_b) ** 2)
            cyl_sdf = dist - 1.0 if solid else 1.0 - dist
            block = signed_distance[tuple(sl)]
            if block.size:
                op = np.minimum if solid else np.maximum
                op(block, cyl_sdf[:, None, :], out=block)
        else:
            sl = [slice(None), slice(None), _axis_slice_from_fraction(z, lo, hi)]
            dist = np.sqrt(((x[:, None] - c0) / radius_a) ** 2 + ((y[None, :] - c1) / radius_b) ** 2)
            cyl_sdf = dist - 1.0 if solid else 1.0 - dist
            block = signed_distance[tuple(sl)]
            if block.size:
                op = np.minimum if solid else np.maximum
                op(block, cyl_sdf[:, :, None], out=block)

    for cylinder in solid_cylinders or ():
        apply(cylinder, solid=True)
    for cylinder in void_cylinders or ():
        apply(cylinder, solid=False)
    return signed_distance


def _resample_source_mask(
    source_mask: Optional[np.ndarray],
    target_shape: Tuple[int, int, int],
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

        zoom = tuple(float(t) / float(s) for t, s in zip(target_shape, source.shape))
        resized = ndi.zoom(source.astype(float), zoom=zoom, order=0, mode='nearest')
        out = resized >= 0.5
        if out.shape != tuple(target_shape):
            cropped = np.zeros(target_shape, dtype=bool)
            common = tuple(slice(0, min(a, b)) for a, b in zip(cropped.shape, out.shape))
            cropped[common] = out[common]
            return cropped
        return out
    except Exception:
        logger.debug("Failed to resample topology source mask", exc_info=True)
        return None


def _recover_voxel_shape(
    density: np.ndarray,
    bounds: Optional[Tuple[np.ndarray, np.ndarray]],
    cutoff: float,
    print_ready: bool = False,
    decimate_ratio: float = 1.0,
    solid_boxes: Sequence[BoxRegion] = (),
    void_boxes: Sequence[BoxRegion] = (),
    solid_cylinders: Sequence[CylinderRegion] = (),
    void_cylinders: Sequence[CylinderRegion] = (),
    extrusion_axis: str = 'none',
    source_mask: Optional[np.ndarray] = None,
) -> Optional[Dict[str, np.ndarray]]:
    """Extract a recovered surface from a structured voxel density field.

    When `print_ready=True`, runs an additional trimesh pipeline after the
    marching-cubes + Taubin pass: connected-component filtering, hole-filling,
    light Humphrey smoothing, and (if `fast_simplification` is
    available) optional quadric decimation when `decimate_ratio < 1.0`.
    """
    try:
        from skimage import measure
        import scipy.ndimage as ndi
    except ImportError:
        return None

    try:
        grid = np.asarray(density, dtype=float)
        if grid.ndim != 3 or min(grid.shape) < 1:
            return None

        grid = np.nan_to_num(grid, nan=0.0, posinf=1.0, neginf=0.0)
        if float(np.max(grid)) <= 0.0:
            return None
        grid = _regularize_extruded_density(grid, extrusion_axis)

        axis_map = {'x': 0, 'y': 1, 'z': 2}
        extrusion_ax = axis_map.get(str(extrusion_axis or '').strip().lower())
        if extrusion_ax is not None:
            in_plane = [i for i in range(3) if i != extrusion_ax]
            min_plane_dim = max(1, min(int(grid.shape[i]) for i in in_plane))
            upsample = int(np.clip(np.ceil(240.0 / min_plane_dim), 1, 8))
            zoom_factors = np.ones(3, dtype=float)
            zoom_factors[in_plane] = float(upsample)
        else:
            min_dim = max(1, min(grid.shape))
            upsample = int(np.clip(np.ceil(48.0 / min_dim), 1, 12))
            zoom_factors = np.full(3, float(upsample), dtype=float)

        while np.any(zoom_factors > 1.0) and np.prod(np.asarray(grid.shape) * zoom_factors) > 2_500_000:
            largest = int(np.argmax(zoom_factors))
            zoom_factors[largest] = max(1.0, zoom_factors[largest] - 1.0)

        if np.any(zoom_factors > 1.0):
            field = ndi.zoom(grid, zoom=tuple(float(v) for v in zoom_factors), order=3, mode='nearest')
        else:
            field = grid.copy()

        field = np.clip(field, 0.0, 1.0)
        source_field = _resample_source_mask(source_mask, tuple(field.shape))
        sigma = 0.20 if float(np.max(zoom_factors)) <= 1.0 else 0.35
        field = ndi.gaussian_filter(field, sigma=sigma)
        field = _apply_passive_density_regions(
            field,
            solid_boxes=solid_boxes,
            void_boxes=void_boxes,
            solid_cylinders=solid_cylinders,
            void_cylinders=void_cylinders,
        )
        if source_field is not None:
            field[~source_field] = 0.0
        pad = max(3, min(10, int(np.ceil(float(np.max(zoom_factors))))))
        field = np.pad(field, pad_width=pad, mode='constant', constant_values=0.0)

        origin, cell = _voxel_origin_cell(tuple(grid.shape), bounds)
        spacing = cell / zoom_factors
        level = float(np.clip(cutoff, 1e-6, 0.999999))
        mask = field >= level
        if not np.any(mask):
            nonzero = field[field > 0.0]
            if nonzero.size == 0:
                return None
            mask = field >= float(np.percentile(nonzero, 75.0))
        if np.all(mask):
            return None

        # Build a signed iso-field directly from the filtered physical density:
        # negative is material, positive is void.  Passive cylinders are applied
        # before marching cubes so circular holes do not need post-hoc vertex
        # snapping, which can fold triangles near tight bolt holes.
        iso_field = level - field
        mc_level = 0.0
        passive_cylinders = tuple(solid_cylinders or ()) + tuple(void_cylinders or ())
        analytic_cylinder_iso = False
        if passive_cylinders:
            iso_field = _apply_passive_cylinder_sdf(
                iso_field,
                pad=pad,
                solid_cylinders=solid_cylinders,
                void_cylinders=void_cylinders,
            )
            analytic_cylinder_iso = True

        # Prefer the filtered physical density field itself for the visible
        # boundary.  The previous binary-mask -> distance-transform SDF route
        # made thin topology webs look like terraced contour plots because the
        # signed-distance bands were smoothed after thresholding.
        if not (float(np.min(iso_field)) < mc_level < float(np.max(iso_field))):
            outside = ndi.distance_transform_edt(~mask, sampling=tuple(float(v) for v in spacing))
            inside = ndi.distance_transform_edt(mask, sampling=tuple(float(v) for v in spacing))
            iso_field = outside - inside
            iso_field = ndi.gaussian_filter(iso_field, sigma=0.35)
            if passive_cylinders:
                iso_field = _apply_passive_cylinder_sdf(
                    iso_field,
                    pad=pad,
                    solid_cylinders=solid_cylinders,
                    void_cylinders=void_cylinders,
                )
                analytic_cylinder_iso = True
            mc_level = 0.0
            if not (float(np.min(iso_field)) < mc_level < float(np.max(iso_field))):
                return None

        verts, faces, _, _ = measure.marching_cubes(
            iso_field,
            level=float(mc_level),
            spacing=tuple(float(v) for v in spacing),
            gradient_direction='ascent',
        )
        if len(verts) == 0 or len(faces) == 0:
            return None

        surface_origin = origin + 0.5 * cell - float(pad) * spacing
        verts = verts + surface_origin
        verts = _taubin_smooth_surface(verts, faces, iterations=2)
        verts = _project_extruded_planes(
            verts, bounds, extrusion_axis,
            tolerance=float(np.max(spacing)) * 2.5,
        )
        if passive_cylinders and not analytic_cylinder_iso:
            verts = _project_passive_cylinder_surfaces(
                verts,
                bounds,
                passive_cylinders,
                tolerance=float(np.max(spacing)) * 3.0,
            )

        # Print-ready: trimesh pipeline (hole-fill, Humphrey, optional decimate).
        if print_ready:
            improved = _enhanced_mesh_postprocess(
                verts, faces,
                decimate_ratio=float(decimate_ratio),
                smoothing_iterations=2,
            )
            if improved is not None and len(improved.get('faces', [])) > 0:
                improved['vertices'] = _project_extruded_planes(
                    improved['vertices'], bounds, extrusion_axis,
                    tolerance=float(np.max(spacing)) * 2.5,
                )
                if passive_cylinders and not analytic_cylinder_iso:
                    improved['vertices'] = _project_passive_cylinder_surfaces(
                        improved['vertices'],
                        bounds,
                        passive_cylinders,
                        tolerance=float(np.max(spacing)) * 3.0,
                    )
                return improved

        return {
            'vertices': np.asarray(verts, dtype=float),
            'faces': np.asarray(faces, dtype=int),
        }
    except Exception:
        logger.exception("Voxel shape recovery failed")
        return None


