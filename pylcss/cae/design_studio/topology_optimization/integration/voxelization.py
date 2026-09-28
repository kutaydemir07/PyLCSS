# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Grid sizing, voxelization, and recovered-volume helpers."""

from __future__ import annotations

import logging
from typing import Any, Optional, Sequence

import numba
import numpy as np

from ..configuration.length_scale import (
    MAX_FEATURE_ELEMENTS,
    MIN_FEATURE_ELEMENTS,
    PROGRAM_CONTROLLED_FEATURE_ELEMENTS,
    resolve_physical_length_scale,
)
from .geometry_mapping import _surface_mesh_arrays

logger = logging.getLogger(__name__)


def _multigrid_aligned_grid(
    dims: np.ndarray,
    span: np.ndarray,
    max_total_cells: int,
    minimum_dims: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Align a guided grid for two geometric-multigrid levels.

    ``minimum_dims`` is a floor the cell ceiling may not cut through. The
    ceiling is a cost limit; a requested physical feature size is an
    engineering requirement, and a grid too coarse to represent it is rejected
    downstream rather than solved, so the requirement wins.
    """
    aligned = np.asarray(dims, dtype=int).copy()
    for axis, value in enumerate(aligned):
        if value >= 4:
            aligned[axis] = max(4, int(round(float(value) / 4.0)) * 4)
        elif value > 1:
            aligned[axis] = value + value % 2

    minima = np.where(aligned >= 4, 4, np.where(aligned > 1, 2, 1))
    if minimum_dims is not None:
        required = np.maximum(np.asarray(minimum_dims, dtype=int), 1)
        # Round the floor up onto the same alignment, so holding the length
        # scale does not cost a multigrid level as well.
        required = np.where(
            required >= 4,
            np.ceil(required / 4.0).astype(int) * 4,
            np.where(required > 1, required + required % 2, required),
        )
        minima = np.maximum(minima, required)
        aligned = np.maximum(aligned, minima)
    while int(np.prod(aligned)) > int(max_total_cells):
        candidates = [
            axis for axis in range(3) if aligned[axis] > minima[axis]
        ]
        if not candidates:
            break
        # Reduce the most over-resolved physical axis first. This retains
        # near-isotropic voxels instead of merely shortening the longest axis.
        axis = max(
            candidates,
            key=lambda index: float(aligned[index]) / max(float(span[index]), 1e-12),
        )
        aligned[axis] -= 4 if aligned[axis] >= 8 else 2
    return aligned


def _guided_voxel_grid(
    bounds: Optional[tuple[np.ndarray, np.ndarray]],
    element_size: Optional[float] = None,
    feature_bboxes: Optional[
        list[tuple[float, float, float, float, float, float]]
    ] = None,
    feature_lengths: Optional[list[float]] = None,
    max_total_cells: int = 160_000,
    inactive_axes: Sequence[int] = (),
    required_feature_size: Optional[float] = None,
) -> Optional[tuple[int, int, int]]:
    """Choose the automatic aspect-correct grid for guided mode.

    Guided mode hides raw voxel counts, so it must derive them from the actual
    CAD extents and the smallest selected support/load/CAD features. Pass an
    explicit ``element_size`` to bypass that and state the discretization
    directly; ``max_total_cells`` is the ceiling the caller lowers for multiple
    load cases or coupled physics. Release decisions still belong to the
    independent recovered-solid mesh-convergence validation.

    ``required_feature_size`` is a physical length the study has *stated* — a
    minimum member or void size — as opposed to a feature merely worth
    resolving. It must span ``MIN_FEATURE_ELEMENTS`` cells or the study is
    rejected when its length scale is resolved, so it is treated as a floor the
    cell ceiling cannot cut through. Without it the sizer could return a grid
    that the very next step refuses to solve.
    """
    if bounds is None:
        return None
    mins, maxs = bounds
    span = np.maximum(
        np.asarray(maxs[:3], dtype=float) - np.asarray(mins[:3], dtype=float), 1e-9
    )
    inactive = {
        int(axis)
        for axis in inactive_axes
        if isinstance(axis, (int, np.integer)) and 0 <= int(axis) < 3
    }
    # At least four layers across an extruded thickness, so a planar-extrusion
    # study still gets a real through-thickness discretization check rather
    # than the bare minimum the multigrid path needs.
    #
    # It is a floor, never a ceiling. The extrusion axis carries no design
    # freedom, but it still carries the physical cone filter, and coarsening it
    # past the in-plane pitch squashes the filtered field: on the 120x60x10
    # cantilever a fixed four layers put a 2.5 mm pitch against 0.83 mm in
    # plane, the filter then spanned a single layer between two void-padded
    # faces, and the field it produced ran 0.087..0.888 instead of 0..1. The
    # recovered profile picked up slivers the spline B-rep could not hold and
    # CAD reconstruction failed outright. Resolve the thickness at the in-plane
    # pitch and the same study reconstructs.
    inactive_layer_count = 4

    def _inactive_dims(dims: np.ndarray, active_pitch: float) -> np.ndarray:
        """Resolve inactive axes at the active pitch, floored at four layers."""
        for axis in inactive:
            dims[axis] = max(
                inactive_layer_count,
                int(np.ceil(float(span[axis]) / max(active_pitch, 1e-9))),
            )
        return dims
    volume = float(np.prod(span))
    if volume <= 0.0:
        return None

    try:
        elem_size = float(element_size)
        if elem_size > 0.0:
            dims = np.maximum(np.ceil(span / max(elem_size, 1e-6)).astype(int), 2)
            dims = _inactive_dims(dims, max(elem_size, 1e-6))
            dims = np.maximum(dims - (dims % 2), 2)
            return int(dims[0]), int(dims[1]), int(dims[2])
    except (TypeError, ValueError):
        pass

    # Pure physical feature-resolution calculation:
    min_feature_dim = float("inf")
    for bbox in feature_bboxes or []:
        try:
            exts = [
                abs(float(bbox[1]) - float(bbox[0])),
                abs(float(bbox[3]) - float(bbox[2])),
                abs(float(bbox[5]) - float(bbox[4])),
            ]
            for axis, e in enumerate(exts):
                if axis in inactive:
                    continue
                if e > 1e-6:
                    min_feature_dim = min(min_feature_dim, e)
        except Exception:
            pass

    for fl in feature_lengths or []:
        try:
            val = float(fl)
            if val > 1e-6:
                min_feature_dim = min(min_feature_dim, val)
        except Exception:
            pass

    target_cells = int(max_total_cells)
    # Take the minimum of four cells across the smallest selected feature and
    # the whole-domain cell ceiling. The hard cell cap is applied below.
    global_voxel_size = (volume / max(float(target_cells), 1.0)) ** (1.0 / 3.0)
    if np.isfinite(min_feature_dim) and min_feature_dim > 0.0:
        feature_voxel_size = max(min_feature_dim / 4.0, 1e-6)
        target_voxel_size = min(feature_voxel_size, global_voxel_size)
    else:
        target_voxel_size = global_voxel_size

    # The coarsest edge a stated feature size tolerates. Only the active axes
    # are constrained, matching how the length scale measures its limiting
    # edge: an extrusion axis carries no design freedom.
    required_dims = np.ones(3, dtype=int)
    try:
        required_feature = float(required_feature_size or 0.0)
    except (TypeError, ValueError):
        required_feature = 0.0
    if required_feature > 0.0:
        required_edge = required_feature / MIN_FEATURE_ELEMENTS
        for axis in range(3):
            if axis in inactive:
                continue
            required_dims[axis] = int(
                np.ceil(float(span[axis]) / max(required_edge, 1e-12))
            )

    dims = np.ceil(span / target_voxel_size).astype(int)
    dims = _inactive_dims(dims, target_voxel_size)
    total_cells = int(np.prod(dims))

    max_total_cells = max(2_000, int(max_total_cells))
    if total_cells > max_total_cells:
        scale = (float(max_total_cells) / float(total_cells)) ** (1.0 / 3.0)
        dims = np.maximum(np.floor(dims * scale).astype(int), 4)
    dims = np.maximum(dims, required_dims)

    dims = _multigrid_aligned_grid(
        dims,
        span,
        max_total_cells,
        minimum_dims=required_dims,
    )
    return int(dims[0]), int(dims[1]), int(dims[2])


def _use_automatic_guided_grid(workflow_mode: Any) -> bool:
    """Return True when guided mode should choose the grid from geometry."""
    return str(workflow_mode or "Guided").strip().lower() != "expert"


# Compatibility aliases for older callers that still inspect the historical
# element-count policy. New guided studies resolve and store physical lengths.
MIN_MEMBER_ELEMENTS_FLOOR = MIN_FEATURE_ELEMENTS
MIN_MEMBER_ELEMENTS_CEILING = MAX_FEATURE_ELEMENTS
RMIN_FLOOR = MIN_MEMBER_ELEMENTS_FLOOR / 2.0
RMIN_CEILING = MIN_MEMBER_ELEMENTS_CEILING / 2.0

# When no size is requested, use three cells on the coarsest voxel direction.
PROGRAM_CONTROLLED_MEMBER_ELEMENTS = PROGRAM_CONTROLLED_FEATURE_ELEMENTS
PROGRAM_CONTROLLED_RMIN = PROGRAM_CONTROLLED_MEMBER_ELEMENTS / 2.0


def _guided_rmin(
    nelx: int,
    nely: int,
    nelz: int,
    bounds: Optional[tuple[np.ndarray, np.ndarray]] = None,
    minimum_member_size: Optional[float] = None,
) -> float:
    """Backward-compatible nominal filter radius, in physical model units.

    Explicit sizes stay in model units. Under-resolved requests raise instead
    of being silently changed. The helper retains its historical name because
    saved studies and extension code import it directly.
    """
    if bounds is None:
        return PROGRAM_CONTROLLED_RMIN
    resolved = resolve_physical_length_scale(
        bounds,
        (nelx, nely, nelz),
        minimum_member_size,
        robust=False,
    )
    return float(resolved.filter_radius)


def voxel_size_from_bounds(
    shape: tuple[int, int, int],
    bounds: Optional[tuple[np.ndarray, np.ndarray]],
) -> Optional[float]:
    """Average edge length of one analysis voxel, in model units."""
    if bounds is None:
        return None
    try:
        mins = np.asarray(bounds[0], dtype=float)[:3]
        maxs = np.asarray(bounds[1], dtype=float)[:3]
        dims = np.maximum(np.asarray(shape, dtype=float)[:3], 1.0)
    except (TypeError, ValueError, IndexError):
        return None
    size = float(np.mean(np.maximum(maxs - mins, 1e-12) / dims))
    return size if np.isfinite(size) and size > 0.0 else None


def lattice_voxels_from_length(
    length: Any,
    voxel_size: Optional[float],
    fallback_voxels: float,
    allow_zero: bool = False,
) -> float:
    """Convert a physical lattice length to analysis voxels.

    Lattice cell size, member thickness and skin thickness were specified in
    voxels. The guided grid is sized from the CAD bounding box against a cell
    budget, so a voxel is a different physical length on every part and on
    every quality preset — "8 voxels" silently meant a different lattice on
    each study, and there was no way to ask for the 5 mm cell a printer's
    powder-removal or minimum-feature capability actually dictates.

    A non-positive length keeps the existing voxel-denominated value unless
    allow_zero is True (e.g. for skin thickness where 0 mm disables skin).
    """
    try:
        if length is None or str(length).strip() == "":
            physical = -1.0
        else:
            physical = float(length)
    except (TypeError, ValueError):
        physical = -1.0

    if physical < 0.0 or (physical == 0.0 and not allow_zero) or voxel_size is None or voxel_size <= 0.0:
        return float(fallback_voxels)
    return float(physical / voxel_size)


def guided_minimum_member_size(
    nelx: int,
    nely: int,
    nelz: int,
    rmin: float,
    bounds: Optional[tuple[np.ndarray, np.ndarray]] = None,
) -> Optional[float]:
    """Physical minimum member size implied by a physical filter radius.

    The inverse of `_guided_rmin`. Professional tools always echo the length
    scale they resolved, in model units, because a radius in elements is not
    something a user can check against a drawing or a process capability.
    """
    _ = (nelx, nely, nelz, bounds)
    value = 2.0 * float(rmin)
    return value if np.isfinite(value) and value > 0.0 else None


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


def _effective_density_cutoff(cutoff: Any, density: Optional[np.ndarray] = None, target_volfrac: Optional[float] = None) -> float:
    """Use adaptive volume-preserving bisection or the saved threshold consistently."""
    if density is not None and target_volfrac is not None and (cutoff is None or str(cutoff).strip().lower() in ("auto", "automatic", "")):
        try:
            flat = np.ascontiguousarray(density.ravel())
            target_count = int(round(float(target_volfrac) * flat.size))
            target_count = min(max(target_count, 1), flat.size - 1)
            kth = flat.size - target_count
            cutoff_val = float(np.partition(flat, kth)[kth])
            return float(np.clip(cutoff_val, 0.05, 0.95))
        except Exception:
            pass
    try:
        cutoff_value = float(cutoff)
    except Exception:
        cutoff_value = 0.30
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


def _stencil_voxel_grid(
    vertices: np.ndarray,
    faces: np.ndarray,
    mins: np.ndarray,
    maxs: np.ndarray,
    dims: np.ndarray,
) -> Optional[np.ndarray]:
    """Rasterize a closed triangle surface onto the cell-centre grid with VTK.

    The same question as :meth:`trimesh.Trimesh.contains` — is this cell centre
    inside the body — answered by ``vtkPolyDataToImageStencil`` in one C++ pass
    over the whole image instead of one Python ray cast per point.

    Not an approximation of the ray test: both classify a point by the parity of
    a ray crossing the same closed surface. Verified voxel-identical on a box, a
    sphere, a torus, an annulus and a two-body assembly (0 of 115,200 voxels
    disagreeing on each), which covers the genus, through-hole and multibody
    cases a design domain is allowed to have.

    What changes is the cost, and it is the difference between a usable node and
    an unusable one: 20.3 s to 0.01 s on the sphere above, and on the grid a
    lattice infill actually builds (1.96M voxels) 93 s of ray casting became
    0.01 s. Returns ``None`` if VTK is unavailable so the caller keeps the ray
    path.
    """
    try:
        import vtk
        from vtkmodules.util import numpy_support
    except ImportError:
        return None
    try:
        points = vtk.vtkPoints()
        points.SetData(
            numpy_support.numpy_to_vtk(
                np.ascontiguousarray(vertices, dtype=np.float64), deep=1
            )
        )
        triangles = np.ascontiguousarray(faces, dtype=np.int64)
        # VTK cell arrays are (count, i, j, k) per polygon.
        flattened = np.hstack(
            (np.full((len(triangles), 1), 3, dtype=np.int64), triangles)
        ).ravel()
        cells = vtk.vtkCellArray()
        cells.SetCells(
            len(triangles),
            numpy_support.numpy_to_vtkIdTypeArray(flattened, deep=1),
        )
        polydata = vtk.vtkPolyData()
        polydata.SetPoints(points)
        polydata.SetPolys(cells)

        step = np.maximum((maxs - mins) / np.maximum(dims, 1), 1e-12)
        origin = mins + 0.5 * step

        image = vtk.vtkImageData()
        image.SetOrigin(*(float(value) for value in origin))
        image.SetSpacing(*(float(value) for value in step))
        image.SetDimensions(*(int(value) for value in dims))
        image.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)
        numpy_support.vtk_to_numpy(image.GetPointData().GetScalars())[:] = 1

        stencil = vtk.vtkPolyDataToImageStencil()
        stencil.SetInputData(polydata)
        stencil.SetOutputOrigin(*(float(value) for value in origin))
        stencil.SetOutputSpacing(*(float(value) for value in step))
        stencil.SetOutputWholeExtent(image.GetExtent())
        stencil.SetTolerance(0.0)
        stencil.Update()

        painter = vtk.vtkImageStencil()
        painter.SetInputData(image)
        painter.SetStencilConnection(stencil.GetOutputPort())
        painter.ReverseStencilOff()
        painter.SetBackgroundValue(0)
        painter.Update()

        scalars = numpy_support.vtk_to_numpy(
            painter.GetOutput().GetPointData().GetScalars()
        )
        # VTK images run X fastest; the design domain is indexed (x, y, z).
        return (
            scalars.reshape(int(dims[2]), int(dims[1]), int(dims[0]))
            .transpose(2, 1, 0)
            > 0
        )
    except Exception:
        logger.debug(
            "VTK stencil voxelization unavailable; using the ray test.",
            exc_info=True,
        )
        return None


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
        stencilled = _stencil_voxel_grid(
            np.asarray(surface.vertices, dtype=float),
            np.asarray(surface.faces, dtype=np.int64),
            mins,
            maxs,
            dims,
        )
        if stencilled is not None:
            return stencilled
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
                "The non-design input accepts Topology Preserved Region nodes only."
            )
        geometry = payload.get("geometry")
        mask = _mesh_design_domain_grid(geometry, bounds, nelx, nely, nelz)
        if mask is None or not np.any(mask):
            raise ValueError(
                "A Topology Preserved Region could not be voxelized. Connect a closed "
                "CAD solid, not an open face or surface."
            )
        if str(payload.get("region_type") or "solid").lower() == "void":
            void |= np.asarray(mask, dtype=bool)
        else:
            solid |= np.asarray(mask, dtype=bool)
    solid &= ~void
    return solid, void
