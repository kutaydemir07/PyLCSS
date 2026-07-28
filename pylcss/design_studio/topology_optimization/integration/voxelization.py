# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Grid sizing, voxelization, and recovered-volume helpers."""

from __future__ import annotations

import logging
from typing import Any, Optional

import numba
import numpy as np

from .geometry_mapping import _surface_mesh_arrays

logger = logging.getLogger(__name__)


def _guided_voxel_grid(
    bounds: Optional[tuple[np.ndarray, np.ndarray]],
    quality_preset: Any,
    feature_bboxes: Optional[
        list[tuple[float, float, float, float, float, float]]
    ] = None,
    feature_lengths: Optional[list[float]] = None,
    max_total_cells: int = 12_000,
) -> Optional[tuple[int, int, int]]:
    """Choose the automatic aspect-correct grid for guided mode.

    Guided mode hides raw voxel counts, so it must derive them from the actual
    CAD extents and the smallest selected support/load/CAD features. The
    `quality_preset` remains only for saved-graph compatibility. The automatic
    policy deliberately targets an interactive study size; release decisions
    belong to the independent mesh-convergence validation, not an opaque
    increase in optimizer voxels.
    """
    if bounds is None:
        return None
    mins, maxs = bounds
    span = np.maximum(
        np.asarray(maxs[:3], dtype=float) - np.asarray(mins[:3], dtype=float), 1e-9
    )
    volume = float(np.prod(span))
    if volume <= 0.0:
        return None

    _ = quality_preset
    # A globally structured voxel grid cannot refine only around a small bore.
    # Keep guided studies interactive by balancing the whole-domain target with
    # four cells across the smallest selected engineering feature. More local
    # resolution belongs in the independent recovered-shape validation mesh.
    # Roughly 12k active cells is still interactive on a workstation, while
    # avoiding the visibly terraced branches produced by the former 6k-cell
    # default on ordinary brackets.
    max_total_cells = max(2_000, int(max_total_cells))
    target_cells, min_axis, max_axis = (
        min(12_000, max_total_cells),
        7,
        100,
    )
    voxel_size = (volume / float(target_cells)) ** (1.0 / 3.0)
    dims = np.ceil(span / max(voxel_size, 1e-12)).astype(int)
    dims = np.maximum(dims, int(min_axis))

    resolved_feature_lengths: list[float] = []
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
        target_across = 4.0
        feature_voxel_size = feature_size / max(float(target_across), 1.0)
        feature_dims = np.ceil(span / max(feature_voxel_size, 1e-12)).astype(int)
        dims = np.maximum(dims, feature_dims)

    longest = int(np.max(dims))
    guided_max_axis = max(
        int(max_axis), 120 if resolved_feature_lengths else int(max_axis)
    )
    if longest > guided_max_axis:
        scale = float(guided_max_axis) / float(longest)
        dims = np.maximum(np.ceil(dims * scale).astype(int), int(min_axis))

    total_cells = int(np.prod(dims))
    if total_cells > int(max_total_cells):
        scale = (float(max_total_cells) / float(total_cells)) ** (1.0 / 3.0)
        dims = np.maximum(np.floor(dims * scale).astype(int), int(min_axis))
        while int(np.prod(dims)) > int(max_total_cells) and int(np.max(dims)) > int(
            min_axis
        ):
            dims[int(np.argmax(dims))] -= 1

    return int(dims[0]), int(dims[1]), int(dims[2])


def _use_automatic_guided_grid(workflow_mode: Any, quality_preset: Any) -> bool:
    """Return True when guided mode should choose the grid from geometry."""
    if str(workflow_mode or "Guided").strip().lower() == "expert":
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


def _axis_radial_indices(axis: Any) -> tuple[int, int]:
    axis_name = str(axis or "z").strip().lower()
    if axis_name == "x":
        return 1, 2
    if axis_name == "y":
        return 0, 2
    return 0, 1


def _cylinder_actual_radii(
    cylinder: tuple[Any, ...],
    span: np.ndarray,
) -> Optional[tuple[str, float, float, float, float, float, float]]:
    if len(cylinder) < 6:
        return None
    axis = str(cylinder[0] or "z").strip().lower()
    c0 = float(cylinder[1])
    c1 = float(cylinder[2])
    lo = float(cylinder[3])
    hi = float(cylinder[4])
    r0 = float(cylinder[5])
    r1 = float(cylinder[6]) if len(cylinder) > 6 else r0
    if r0 <= 0.0 or r1 <= 0.0:
        return None
    a0, a1 = _axis_radial_indices(axis)
    return (
        axis,
        c0,
        c1,
        min(lo, hi),
        max(lo, hi),
        r0 * float(span[a0]),
        r1 * float(span[a1]),
    )


def _cylinder_feature_lengths(
    bounds: Optional[tuple[np.ndarray, np.ndarray]],
    solid_cylinders: Optional[list[tuple[Any, ...]]] = None,
    void_cylinders: Optional[list[tuple[Any, ...]]] = None,
) -> list[float]:
    """Return physical feature lengths that should be resolved by the grid."""
    if bounds is None:
        return []
    mins, maxs = bounds
    span = np.maximum(
        np.asarray(maxs[:3], dtype=float) - np.asarray(mins[:3], dtype=float), 1e-12
    )
    solids = [
        c
        for c in (
            _cylinder_actual_radii(cylinder, span)
            for cylinder in (solid_cylinders or [])
        )
        if c is not None
    ]
    voids = [
        c
        for c in (
            _cylinder_actual_radii(cylinder, span)
            for cylinder in (void_cylinders or [])
        )
        if c is not None
    ]

    lengths: list[float] = []
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


def _recovered_shape_volume(recovered: Any) -> Optional[float]:
    """Return the actual watertight recovered-mesh volume in model units."""
    if not isinstance(recovered, dict):
        return None
    try:
        vertices = np.asarray(recovered.get("vertices"), dtype=float)
        faces = np.asarray(recovered.get("faces"), dtype=np.int64)
        if (
            vertices.ndim != 2
            or vertices.shape[1] < 3
            or faces.ndim != 2
            or faces.shape[1] < 3
        ):
            return None
        import trimesh

        mesh = trimesh.Trimesh(
            vertices=vertices[:, :3],
            faces=faces[:, :3],
            process=False,
        )
        parts = mesh.split(only_watertight=False)
        volume = float(sum(abs(float(part.volume)) for part in parts))
        return volume if np.isfinite(volume) and volume > 0.0 else None
    except Exception:
        logger.debug("Could not measure recovered topology volume", exc_info=True)
        return None


def _fractional_cylinder_volume(
    cylinder: tuple[Any, ...],
    bounds: Optional[tuple[np.ndarray, np.ndarray]],
) -> float:
    """Return one fractional elliptic-cylinder volume in physical units."""
    if bounds is None or len(cylinder) < 6:
        return 0.0
    try:
        axis = str(cylinder[0] or "z").strip().lower()
        lo, hi = sorted((float(cylinder[3]), float(cylinder[4])))
        r0 = float(cylinder[5])
        r1 = float(cylinder[6]) if len(cylinder) > 6 else r0
        span = np.maximum(
            np.asarray(bounds[1], dtype=float)[:3]
            - np.asarray(bounds[0], dtype=float)[:3],
            0.0,
        )
        axis_idx = {"x": 0, "y": 1, "z": 2}.get(axis, 2)
        radial = [idx for idx in range(3) if idx != axis_idx]
        length = (hi - lo) * float(span[axis_idx])
        radius_a = r0 * float(span[radial[0]])
        radius_b = r1 * float(span[radial[1]])
        volume = float(np.pi * radius_a * radius_b * length)
        return volume if np.isfinite(volume) and volume > 0.0 else 0.0
    except Exception:
        return 0.0


def _meaningful_material_components(
    density: np.ndarray,
    cutoff: float,
    design_domain: Optional[np.ndarray],
) -> tuple[int, list[int]]:
    """Count materially significant 3-D components in a recovered voxel field."""
    from scipy import ndimage

    rho = np.asarray(density, dtype=float)
    if rho.ndim != 3 or rho.size == 0:
        return 0, []
    solid = rho >= float(cutoff)
    if design_domain is not None:
        source = np.asarray(design_domain, dtype=bool)
        if source.shape == solid.shape:
            solid &= source
    solid_count = int(np.count_nonzero(solid))
    if solid_count == 0:
        return 0, []
    labels, count = ndimage.label(
        solid,
        # The validation tet mesh joins face-adjacent voxels only. Corner- or
        # edge-touching cells are not a load path and must be reported as
        # disconnected before the CalculiX handoff.
        structure=ndimage.generate_binary_structure(3, 1),
    )
    sizes = np.bincount(labels.ravel())[1:]
    minimum = max(3, int(np.ceil(0.002 * solid_count)))
    meaningful = sorted(
        (int(value) for value in sizes if int(value) >= minimum),
        reverse=True,
    )
    return len(meaningful), meaningful


def _effective_density_cutoff(cutoff: Any) -> float:
    """Use the saved threshold consistently for preview, recovery, and export."""
    try:
        cutoff_value = float(cutoff)
    except Exception:
        cutoff_value = 0.45
    return float(np.clip(cutoff_value, 0.01, 0.95))


@numba.njit(cache=True)
def _numba_voxelize_tets(
    pts: np.ndarray,
    tets: np.ndarray,
    mins: np.ndarray,
    sub_step: np.ndarray,
    nelx: int,
    nely: int,
    nelz: int,
    samples: int,
    xs: np.ndarray,
    ys: np.ndarray,
    zs: np.ndarray,
    active_samples: np.ndarray,
) -> None:
    tol = 1e-9
    nx = nelx * samples
    ny = nely * samples
    nz = nelz * samples

    for i in range(len(tets)):
        tet = tets[i]
        if tet[0] < 0 or tet[1] < 0 or tet[2] < 0 or tet[3] < 0:
            continue
        if (
            tet[0] >= len(pts)
            or tet[1] >= len(pts)
            or tet[2] >= len(pts)
            or tet[3] >= len(pts)
        ):
            continue
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
        sx_stop = int(np.floor((hi_x - mins[0]) / sub_step[0] - 0.5)) + 1
        sy_start = int(np.ceil((lo_y - mins[1]) / sub_step[1] - 0.5))
        sy_stop = int(np.floor((hi_y - mins[1]) / sub_step[1] - 0.5)) + 1
        sz_start = int(np.ceil((lo_z - mins[2]) / sub_step[2] - 0.5))
        sz_stop = int(np.floor((hi_z - mins[2]) / sub_step[2] - 0.5)) + 1

        sx_start = max(sx_start, 0)
        sy_start = max(sy_start, 0)
        sz_start = max(sz_start, 0)
        sx_stop = min(sx_stop, nx)
        sy_stop = min(sy_stop, ny)
        sz_stop = min(sz_stop, nz)

        if sx_stop <= sx_start or sy_stop <= sy_start or sz_stop <= sz_start:
            continue

        mat = np.empty((3, 3), dtype=np.float64)
        mat[0, 0] = v1[0] - v0[0]
        mat[0, 1] = v2[0] - v0[0]
        mat[0, 2] = v3[0] - v0[0]
        mat[1, 0] = v1[1] - v0[1]
        mat[1, 1] = v2[1] - v0[1]
        mat[1, 2] = v3[1] - v0[1]
        mat[2, 0] = v1[2] - v0[2]
        mat[2, 1] = v2[2] - v0[2]
        mat[2, 2] = v3[2] - v0[2]

        # Skip degenerate tets explicitly. Numba nopython try/except only
        # catches user-raised exceptions; np.linalg.inv on a singular matrix
        # can return inf/nan rather than raising, so the determinant check is
        # the only reliable guard.
        det = (
            mat[0, 0] * (mat[1, 1] * mat[2, 2] - mat[1, 2] * mat[2, 1])
            - mat[0, 1] * (mat[1, 0] * mat[2, 2] - mat[1, 2] * mat[2, 0])
            + mat[0, 2] * (mat[1, 0] * mat[2, 1] - mat[1, 1] * mat[2, 0])
        )
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

                    b1 = inv[0, 0] * qx + inv[0, 1] * qy + inv[0, 2] * qz
                    b2 = inv[1, 0] * qx + inv[1, 1] * qy + inv[1, 2] * qz
                    b3 = inv[2, 0] * qx + inv[2, 1] * qy + inv[2, 2] * qz
                    b0 = 1.0 - b1 - b2 - b3

                    if (
                        b0 >= -1e-8
                        and b1 >= -1e-8
                        and b2 >= -1e-8
                        and b3 >= -1e-8
                        and b0 <= 1.0 + 1e-8
                        and b1 <= 1.0 + 1e-8
                        and b2 <= 1.0 + 1e-8
                        and b3 <= 1.0 + 1e-8
                    ):
                        sub = (lx * samples + ly) * samples + lz
                        active_samples[vx, vy, vz, sub] = True


def _surface_design_domain_grid(
    mesh: Any,
    bounds: tuple[np.ndarray, np.ndarray],
    nelx: int,
    nely: int,
    nelz: int,
) -> Optional[np.ndarray]:
    """Voxelize a watertight imported/CAD triangle surface at cell centres."""
    arrays = _surface_mesh_arrays(mesh)
    if arrays is None:
        return None
    try:
        import trimesh

        vertices, faces = arrays
        surface = trimesh.Trimesh(
            vertices=np.asarray(vertices, dtype=float),
            faces=np.asarray(faces, dtype=int),
            process=True,
            validate=True,
        )
        if surface.is_empty or len(surface.faces) < 4:
            return None
        if not surface.is_watertight:
            try:
                trimesh.repair.fill_holes(surface)
                trimesh.repair.fix_normals(surface)
            except Exception:
                pass
        if not surface.is_watertight:
            logger.warning(
                "TopologyOptVoxelNode: imported/CAD surface is not watertight; "
                "it cannot define a reliable design volume."
            )
            return None

        mins, maxs = bounds
        mins = np.asarray(mins[:3], dtype=float)
        maxs = np.asarray(maxs[:3], dtype=float)
        dims = np.asarray([nelx, nely, nelz], dtype=int)
        step = np.maximum((maxs - mins) / np.maximum(dims, 1), 1e-12)
        axes = [mins[i] + (np.arange(int(dims[i])) + 0.5) * step[i] for i in range(3)]
        xx, yy, zz = np.meshgrid(*axes, indexing="ij")
        points = np.column_stack((xx.ravel(), yy.ravel(), zz.ravel()))
        inside = np.zeros(len(points), dtype=bool)
        for start in range(0, len(points), 20_000):
            stop = min(start + 20_000, len(points))
            inside[start:stop] = surface.contains(points[start:stop])
        return inside.reshape((nelx, nely, nelz))
    except Exception:
        logger.exception(
            "TopologyOptVoxelNode: failed to voxelize surface design domain"
        )
        return None


def _mesh_design_domain_grid(
    mesh: Any,
    bounds: Optional[tuple[np.ndarray, np.ndarray]],
    nelx: int,
    nely: int,
    nelz: int,
) -> Optional[np.ndarray]:
    """Voxelize the actual tetra mesh volume into the optimizer grid.

    This keeps FreeCAD cutouts and holes as voids automatically instead of
    treating the whole bounding box as designable material.
    """
    if mesh is None or bounds is None:
        return None
    if not hasattr(mesh, "p") or not hasattr(mesh, "t"):
        return _surface_design_domain_grid(mesh, bounds, nelx, nely, nelz)
    try:
        points = np.asarray(mesh.p, dtype=float)
        cells = np.asarray(mesh.t, dtype=int)
    except Exception:
        return _surface_design_domain_grid(mesh, bounds, nelx, nely, nelz)
    if points.ndim != 2 or points.shape[0] < 3 or points.shape[1] == 0:
        return None
    if cells.ndim != 2 or cells.shape[0] < 4 or cells.shape[1] == 0:
        return _surface_design_domain_grid(mesh, bounds, nelx, nely, nelz)

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
    active_samples = np.zeros((nelx, nely, nelz, samples**3), dtype=bool)

    pts = points[:3].T
    tets = cells[:4].T

    _numba_voxelize_tets(
        pts, tets, mins, sub_step, nelx, nely, nelz, samples, xs, ys, zs, active_samples
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
            min_sample_fraction = max(1.0 / float(samples**3), 0.08)
            active = occupancy >= min_sample_fraction
    else:
        active = touched

    if not np.any(active):
        logger.warning(
            "TopologyOptVoxelNode: source mesh produced an empty voxel design domain; "
            "continuing with passive contact regions only."
        )
    return active


def _non_design_region_masks(
    payloads: list[Any],
    bounds: tuple[np.ndarray, np.ndarray],
    nelx: int,
    nely: int,
    nelz: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Voxelize closed CAD solids supplied as explicit non-design regions."""
    shape = (int(nelx), int(nely), int(nelz))
    solid = np.zeros(shape, dtype=bool)
    void = np.zeros(shape, dtype=bool)
    for payload in payloads:
        if (
            not isinstance(payload, dict)
            or str(payload.get("type") or "").lower() != "topology_non_design_region"
        ):
            raise ValueError(
                "The non-design input accepts Non-Design Region nodes only."
            )
        geometry = payload.get("geometry")
        mask = _mesh_design_domain_grid(geometry, bounds, nelx, nely, nelz)
        if mask is None or not np.any(mask):
            raise ValueError(
                "A Non-Design Region could not be voxelized. Connect a closed "
                "CAD solid, not an open face or surface."
            )
        if str(payload.get("region_type") or "solid").lower() == "void":
            void |= np.asarray(mask, dtype=bool)
        else:
            solid |= np.asarray(mask, dtype=bool)
    solid &= ~void
    return solid, void
