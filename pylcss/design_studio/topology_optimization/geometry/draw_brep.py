# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Exact B-rep reconstruction for draw-direction (cast/moulded) topology.

A cast or moulded result is not a general 3-D body.  The pull-out constraint in
:func:`..manufacturing.constraints._apply_pull_out` takes a running minimum of
the density along the withdrawal direction, so once a column turns to void it
can never become solid again.  The retained material in every column is
therefore a single interval that starts at the closed face of the tool, which
means the optimized body *is* a height field over the parting plane.

That is an exact structural fact about the result, not an approximation of it,
and it converts to a small editable B-rep the same way an extrusion does:

* the silhouette on the parting plane becomes one faired periodic B-spline
  outline per loop, extruded into a prism;
* the level crossing along each column becomes a single approximated B-spline
  surface, fitted to the demoulding tolerance;
* the prism is split by that surface and the material side is kept.

The result is a solid whose faces are the mould's own features -- one parting
face, one draw surface, one side wall per outline -- rather than thousands of
freeform patches with no manufacturing meaning.  Nothing here is accepted on
faith: demouldability, topology, volume, sampled surface deviation and
above-cutoff point coverage are all measured before the body is returned.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np

from .brep_validation import protected_point_coverage
from .recovery_grid import _voxel_origin_cell
from .spline_brep import (
    _loop_nesting_depths,
    _mesh_volume,
    _point_in_polygon,
    _profile_wire,
    _sampled_surface_deviation,
)

logger = logging.getLogger(__name__)

_AXIS_INDEX = {"x": 0, "y": 1, "z": 2}
_MAX_SURFACE_SAMPLES = 80
_MIN_SURFACE_SAMPLES = 8
_EXTENSION_RELAXATIONS = 64
_FAIRING_SIGMA_CELLS = 0.75


def _draw_axis(value: Any) -> Optional[tuple[int, int]]:
    """Return ``(axis, step)`` for a signed pull-out direction such as ``+Z``."""
    key = str(value or "").strip().lower()
    if not key or key == "none":
        return None
    step = 1
    if key[0] in "+-":
        step = -1 if key[0] == "-" else 1
        key = key[1:]
    axis = _AXIS_INDEX.get(key.strip())
    return None if axis is None else (axis, step)


def _draw_height_field(
    field: np.ndarray,
    level: float,
    *,
    axis: int,
    step: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return per-column height, silhouette, material mask and retained depth.

    Heights are measured in cells from the closed face of the tool.  A column
    that is solid all the way through returns the full depth.
    """
    ordered = field if step > 0 else np.flip(field, axis=axis)
    ordered = np.moveaxis(ordered, axis, -1)
    depth = int(ordered.shape[-1])
    above = ordered >= float(level)
    counts = np.count_nonzero(above, axis=-1)

    # Demouldability is a statement about the retained material, so test it on
    # the thresholded field rather than on the raw densities: every above-cutoff
    # run must be a prefix that starts at the closed face. A trailing island is
    # exactly the undercut the tool would be trapped by.
    prefix = np.arange(depth)[None, None, :] < counts[..., None]
    if not np.array_equal(above, prefix):
        trapped = int(np.count_nonzero(np.any(above != prefix, axis=-1)))
        raise RuntimeError(
            f"{trapped} column(s) of the recovered field contain material the "
            "tool cannot clear, so the result is not a draw-direction body. "
            "Check that the pull-out constraint was active for this study."
        )

    last = np.clip(counts - 1, 0, max(depth - 2, 0))
    lower = np.take_along_axis(ordered, last[..., None], axis=-1)[..., 0]
    upper = np.take_along_axis(
        ordered,
        np.minimum(last + 1, depth - 1)[..., None],
        axis=-1,
    )[..., 0]
    span = lower - upper
    fraction = np.where(
        span > 1.0e-12,
        (lower - float(level)) / np.where(span > 1.0e-12, span, 1.0),
        0.0,
    )
    heights = last.astype(float) + 0.5 + np.clip(fraction, 0.0, 1.0)
    heights = np.where(counts >= depth, float(depth), heights)
    material = counts > 0
    heights = np.where(material, heights, 0.0)
    return heights, ordered[..., 0], material, counts


def _column_protected_points(
    counts: np.ndarray,
    material: np.ndarray,
    *,
    origin: np.ndarray,
    cell: np.ndarray,
    plane_axes: tuple[int, int],
    base_axis: int,
    base_coordinate: float,
    step: int,
) -> np.ndarray:
    """Return the topmost retained cell centre of every column.

    A reconstructed draw body is the silhouette prism intersected with the
    region under the draw surface, so a column whose highest retained centre is
    inside the body contains every centre below it as well.  Checking the top
    cell of each column is therefore the same statement about deleted material
    as checking all of them, at a fraction of the cost -- on a ribbed housing
    it is 3.7k points instead of 31k.
    """
    rows, columns = np.nonzero(material)
    if not len(rows):
        return np.empty((0, 3), dtype=float)
    first, second = plane_axes
    top = counts[rows, columns] - 1
    points = np.empty((len(rows), 3), dtype=float)
    points[:, first] = origin[first] + (rows + 0.5) * cell[first]
    points[:, second] = origin[second] + (columns + 0.5) * cell[second]
    points[:, base_axis] = float(base_coordinate) + float(step) * (
        top + 0.5
    ) * float(cell[base_axis])
    return points


def _footprint_loops(
    silhouette: np.ndarray,
    level: float,
    *,
    origin: np.ndarray,
    cell: np.ndarray,
    plane_axes: tuple[int, int],
    base_axis: int,
    base_coordinate: float,
) -> list[np.ndarray]:
    """Return closed silhouette outlines as 3-D loops on the parting plane."""
    from skimage import measure

    # Padding with void closes every outline and lets a footprint that reaches
    # the domain wall cross between the last cell centre and the wall, which is
    # the same convention the recovered isosurface is extracted with.
    padded = np.pad(np.asarray(silhouette, dtype=float), 1, constant_values=0.0)
    contours = measure.find_contours(padded, float(level))
    first, second = plane_axes
    loops: list[np.ndarray] = []
    for contour in contours:
        samples = np.asarray(contour, dtype=float) - 1.0
        if len(samples) < 4:
            continue
        if np.linalg.norm(samples[0] - samples[-1]) <= 1.0e-9:
            samples = samples[:-1]
        if len(samples) < 3:
            continue
        loop = np.empty((len(samples), 3), dtype=float)
        loop[:, first] = origin[first] + (samples[:, 0] + 0.5) * cell[first]
        loop[:, second] = origin[second] + (samples[:, 1] + 0.5) * cell[second]
        loop[:, base_axis] = float(base_coordinate)
        loops.append(loop)
    return loops


def _resampled_height_grid(
    heights: np.ndarray,
    material: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fill the void columns, then bring the grid into the fitting size band.

    Outside the silhouette the height is meaningless -- the surface there is
    trimmed away by the prism -- but the approximation still needs a complete
    array.  The extension is harmonic rather than nearest-neighbour: a nearest
    fill is only piecewise constant, and its ridges are high-frequency content
    the C2 approximation has to chase, which multiplied the control net of a
    ribbed housing several times over for detail that is then trimmed away.
    """
    from scipy import ndimage

    filled = np.asarray(heights, dtype=float).copy()
    solid = np.asarray(material, dtype=bool)
    if not solid.all() and solid.any():
        _distance, indices = ndimage.distance_transform_edt(
            ~solid,
            return_indices=True,
            return_distances=True,
        )
        filled = filled[tuple(indices)]
        # Jacobi sweeps with the silhouette values held fixed relax the fill
        # towards the harmonic extension of the draw surface.
        outside = ~solid
        for _iteration in range(_EXTENSION_RELAXATIONS):
            smoothed = ndimage.uniform_filter(filled, size=3, mode="nearest")
            filled = np.where(outside, smoothed, filled)

    # A projected density field crosses the cutoff inside one cell, so the raw
    # column heights are quantized to the cell pitch and the draw surface
    # inherits that staircase. Fairing it away is the same correction the
    # extruded profile applies to its section outline, and it is done under the
    # same constraint: the swept volume the optimizer sized is preserved
    # exactly, so smoothing cannot quietly add or remove material.
    volume_before = float(np.sum(filled[solid])) if solid.any() else 0.0
    faired = ndimage.gaussian_filter(filled, sigma=_FAIRING_SIGMA_CELLS, mode="nearest")
    volume_after = float(np.sum(faired[solid])) if solid.any() else 0.0
    if volume_before > 0.0 and volume_after > 0.0:
        faired *= volume_before / volume_after
    filled = faired

    rows = np.arange(filled.shape[0], dtype=float)
    columns = np.arange(filled.shape[1], dtype=float)
    target = [
        int(np.clip(size, _MIN_SURFACE_SAMPLES, _MAX_SURFACE_SAMPLES))
        for size in filled.shape
    ]
    if target != list(filled.shape):
        sample_rows = np.linspace(0.0, filled.shape[0] - 1.0, target[0])
        sample_columns = np.linspace(0.0, filled.shape[1] - 1.0, target[1])
        mesh = np.meshgrid(sample_rows, sample_columns, indexing="ij")
        filled = ndimage.map_coordinates(
            filled,
            np.asarray(mesh, dtype=float),
            order=1,
            mode="nearest",
        )
        rows, columns = sample_rows, sample_columns
    return filled, rows, columns


def _draw_surface_face(
    heights: np.ndarray,
    rows: np.ndarray,
    columns: np.ndarray,
    *,
    origin: np.ndarray,
    cell: np.ndarray,
    plane_axes: tuple[int, int],
    base_axis: int,
    base_coordinate: float,
    step: int,
    fit_tolerance: float,
) -> tuple[Any, int]:
    """Approximate the draw surface as one C2 B-spline face in model space."""
    import cadquery as cq
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace
    from OCP.GeomAbs import GeomAbs_C2
    from OCP.GeomAPI import GeomAPI_PointsToBSplineSurface
    from OCP.gp import gp_Pnt
    from OCP.TColgp import TColgp_Array2OfPnt

    first, second = plane_axes
    grid = TColgp_Array2OfPnt(1, len(rows), 1, len(columns))
    for i, row in enumerate(rows):
        first_coordinate = origin[first] + (float(row) + 0.5) * cell[first]
        for j, column in enumerate(columns):
            point = np.empty(3, dtype=float)
            point[first] = first_coordinate
            point[second] = origin[second] + (float(column) + 0.5) * cell[second]
            point[base_axis] = float(base_coordinate) + float(step) * float(
                heights[i, j]
            ) * float(cell[base_axis])
            grid.SetValue(i + 1, j + 1, gp_Pnt(*point))

    approximation = GeomAPI_PointsToBSplineSurface(
        grid,
        3,
        8,
        GeomAbs_C2,
        max(float(fit_tolerance), 1.0e-7),
    )
    if not approximation.IsDone():
        raise RuntimeError("The draw surface could not be approximated.")
    surface = approximation.Surface()
    face = BRepBuilderAPI_MakeFace(surface, 1.0e-6).Face()
    control_points = int(surface.NbUPoles() * surface.NbVPoles())
    return cq.Face(face), control_points


def _prism_from_footprint(
    loops: list[np.ndarray],
    *,
    plane_axes: tuple[int, int],
    base_axis: int,
    sweep: np.ndarray,
    fit_tolerance: float,
) -> tuple[list[Any], int, int]:
    """Extrude the faired silhouette outlines into a prism with its holes cut."""
    import cadquery as cq

    loops_2d = [loop[:, list(plane_axes)] for loop in loops]
    depths = _loop_nesting_depths(loops_2d)
    wires: list[Any] = []
    control_points = 0
    edge_count = 0
    for loop in loops:
        wire, count, _fairness = _profile_wire(loop, fit_tolerance=fit_tolerance)
        wires.append(wire)
        control_points += int(count)
        edge_count += len(wire.Edges())

    solids: list[Any] = []
    for index, depth in enumerate(depths):
        if depth % 2:
            continue
        hole_indices = [
            hole_index
            for hole_index, hole_depth in enumerate(depths)
            if hole_depth == depth + 1
            and _point_in_polygon(loops_2d[hole_index][0], loops_2d[index])
        ]
        material = cq.Solid.extrudeLinear(
            wires[index],
            [],
            cq.Vector(*(float(value) for value in sweep)),
        )
        if hole_indices:
            material = material.cut(
                cq.Compound.makeCompound(
                    [
                        cq.Solid.extrudeLinear(
                            wires[hole_index],
                            [],
                            cq.Vector(*(float(value) for value in sweep)),
                        )
                        for hole_index in hole_indices
                    ]
                )
            )
        solids.append(material)
    if not solids:
        raise RuntimeError("The silhouette produced no extrudable material region.")
    return solids, control_points, edge_count


def _draw_direction_brep(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    density: np.ndarray,
    level: float,
    bounds: Optional[tuple[np.ndarray, np.ndarray]],
    pull_direction: Any,
    absolute_fit_tolerance: float,
    relative_fit_tolerance: float,
    maximum_volume_delta: float,
    maximum_relative_deviation: float,
    protected_points: Optional[np.ndarray] = None,
) -> tuple[Any, dict[str, Any]]:
    """Build and validate the exact height-field B-rep of a demouldable result."""
    import cadquery as cq

    draw = _draw_axis(pull_direction)
    if draw is None:
        raise RuntimeError(
            "Draw-direction reconstruction requires an explicit pull-out axis."
        )
    axis, step = draw

    grid = np.asarray(density, dtype=float)
    if grid.ndim != 3 or min(grid.shape) < 2:
        raise RuntimeError(
            "Draw-direction reconstruction needs a three-dimensional density field."
        )
    origin, cell = _voxel_origin_cell(tuple(grid.shape), bounds)

    vertices = np.asarray(vertices, dtype=float)[:, :3]
    faces = np.asarray(faces, dtype=np.int64)[:, :3]
    lower = np.min(vertices, axis=0)
    upper = np.max(vertices, axis=0)
    diagonal = float(np.linalg.norm(upper - lower))
    fit_tolerance = max(
        float(absolute_fit_tolerance or 0.0),
        float(relative_fit_tolerance or 0.0) * diagonal,
        1.0e-7,
    )

    heights, silhouette, material, counts = _draw_height_field(
        grid,
        float(level),
        axis=axis,
        step=step,
    )
    if not material.any():
        raise RuntimeError("The recovered field retains no material to reconstruct.")

    plane_axes = tuple(index for index in range(3) if index != axis)
    depth_cells = int(grid.shape[axis])
    base_coordinate = (
        float(origin[axis])
        if step > 0
        else float(origin[axis] + depth_cells * cell[axis])
    )
    maximum_height = float(np.max(heights[material])) * float(cell[axis])
    if not np.isfinite(maximum_height) or maximum_height <= fit_tolerance:
        raise RuntimeError("The draw-direction result has no usable depth.")

    loops = _footprint_loops(
        silhouette,
        float(level),
        origin=origin,
        cell=cell,
        plane_axes=plane_axes,
        base_axis=axis,
        base_coordinate=base_coordinate,
    )
    if not loops:
        raise RuntimeError("No closed silhouette was found on the parting plane.")

    # The prism has to stand clear of the draw surface everywhere so the split
    # is a clean two-piece cut rather than a tangency.
    sweep = np.zeros(3, dtype=float)
    sweep[axis] = float(step) * (maximum_height + max(2.0 * float(cell[axis]), fit_tolerance))
    prism_solids, outline_control_points, outline_edges = _prism_from_footprint(
        loops,
        plane_axes=plane_axes,
        base_axis=axis,
        sweep=sweep,
        fit_tolerance=fit_tolerance,
    )

    filled, rows, columns = _resampled_height_grid(heights, material)
    # The column heights are only resolved to the cell the level crossing falls
    # in, so approximating them an order of magnitude tighter than that buys no
    # accuracy and pays for it in control points: on a ribbed housing the same
    # surface took 3690 poles at the curve-fitting tolerance and 1188 at a
    # quarter of a cell, for deviation that the gate below cannot tell apart.
    surface_tolerance = max(fit_tolerance, 0.25 * float(cell[axis]))
    surface_face, surface_control_points = _draw_surface_face(
        filled,
        rows,
        columns,
        origin=origin,
        cell=cell,
        plane_axes=plane_axes,
        base_axis=axis,
        base_coordinate=base_coordinate,
        step=step,
        fit_tolerance=surface_tolerance,
    )

    from scipy.interpolate import RegularGridInterpolator

    height_lookup = RegularGridInterpolator(
        (rows, columns),
        filled,
        bounds_error=False,
        fill_value=None,
    )

    def _keeps_material(solid: Any) -> bool:
        centre = solid.Center()
        model = np.asarray([centre.x, centre.y, centre.z], dtype=float)
        sample = np.asarray(
            [
                (model[plane_axes[0]] - origin[plane_axes[0]]) / cell[plane_axes[0]]
                - 0.5,
                (model[plane_axes[1]] - origin[plane_axes[1]]) / cell[plane_axes[1]]
                - 0.5,
            ],
            dtype=float,
        )
        expected = float(height_lookup(sample[None, :])[0]) * float(cell[axis])
        actual = float(step) * (model[axis] - base_coordinate)
        return actual < expected

    kept: list[Any] = []
    for prism in prism_solids:
        pieces = prism.split(surface_face)
        candidates = list(pieces.Solids())
        if not candidates:
            raise RuntimeError("The draw surface did not split the silhouette prism.")
        below = [solid for solid in candidates if _keeps_material(solid)]
        if not below:
            raise RuntimeError(
                "Splitting the silhouette prism left no material below the draw "
                "surface."
            )
        kept.extend(below)

    candidate: Any = kept[0] if len(kept) == 1 else cq.Compound.makeCompound(kept)
    if not candidate.isValid():
        raise RuntimeError("The draw-direction reconstruction produced an invalid body.")

    source_volume = _mesh_volume(vertices, faces)
    candidate_volume = float(candidate.Volume())
    volume_delta = None
    if source_volume is not None:
        volume_delta = (candidate_volume - source_volume) / source_volume
        if not np.isfinite(volume_delta) or abs(float(volume_delta)) > float(
            maximum_volume_delta
        ):
            raise RuntimeError(
                "Draw-direction reconstruction changed enclosed volume by "
                f"{float(volume_delta):+.2%}; the limit is "
                f"{float(maximum_volume_delta):.2%}."
            )

    sampled_deviation = _sampled_surface_deviation(
        vertices,
        faces,
        candidate,
        tessellation_tolerance=max(fit_tolerance * 0.5, diagonal * 1.0e-5),
    )
    deviation_limit = max(
        fit_tolerance * 4.0,
        float(maximum_relative_deviation) * diagonal,
    )
    if not np.isfinite(sampled_deviation) or sampled_deviation > deviation_limit:
        raise RuntimeError(
            "Draw-direction reconstruction sampled surface deviation is "
            f"{sampled_deviation:g}, above the {deviation_limit:g} limit."
        )

    # The recovery stage only projects protected centres for an extrusion, so a
    # cast study arrives with none. Derive them from the same field the height
    # was read out of: without this the coverage gate would silently pass on
    # every draw-direction body.
    column_points = _column_protected_points(
        counts,
        material,
        origin=origin,
        cell=cell,
        plane_axes=plane_axes,
        base_axis=axis,
        base_coordinate=base_coordinate,
        step=step,
    )
    supplied = (
        np.asarray(protected_points, dtype=float)
        if protected_points is not None
        else np.empty((0, 3), dtype=float)
    )
    if supplied.ndim != 2 or supplied.shape[1] != 3:
        supplied = np.empty((0, 3), dtype=float)
    coverage = protected_point_coverage(
        candidate,
        np.vstack((column_points, supplied)),
        fit_tolerance=fit_tolerance,
        description="Draw-direction reconstruction",
    )

    report = {
        "method": "Smooth",
        "representation": "draw-direction height-field B-rep",
        "editable": True,
        "smooth": True,
        "fallback_used": False,
        "draw_direction": f"{'+' if step > 0 else '-'}{'XYZ'[axis]}",
        "parting_plane_coordinate": float(base_coordinate),
        "draw_depth": float(maximum_height),
        "silhouette_loops": len(loops),
        "silhouette_spline_edges": int(outline_edges),
        "silhouette_control_points": int(outline_control_points),
        "draw_surface_control_points": int(surface_control_points),
        "draw_surface_samples": [int(len(rows)), int(len(columns))],
        "draw_surface_fit_tolerance": float(surface_tolerance),
        "source_triangle_count": int(len(faces)),
        "cad_face_count": int(len(candidate.Faces())),
        "fit_tolerance": float(fit_tolerance),
        "max_sampled_surface_deviation": float(sampled_deviation),
        "maximum_allowed_surface_deviation": float(deviation_limit),
        **coverage,
        "source_mesh_volume": source_volume,
        "cad_volume_before_feature_healing": candidate_volume,
        "volume_delta_pct": (
            float(volume_delta * 100.0) if volume_delta is not None else None
        ),
    }
    logger.info(
        "Topology CAD: draw-direction %s body from %d silhouette loop(s) into "
        "%d B-rep face(s), max sampled deviation=%g, volume delta=%s.",
        report["draw_direction"],
        len(loops),
        report["cad_face_count"],
        sampled_deviation,
        f"{volume_delta:+.2%}" if volume_delta is not None else "unknown",
    )
    return candidate, report


__all__ = ["_draw_axis", "_draw_direction_brep"]
