# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Density-field to manufacturing-surface recovery orchestration."""

from __future__ import annotations

import logging
from typing import Optional, Sequence

import numpy as np

from ..manufacturing.structures import (
    LAST_STRUCTURE_DIAGNOSTICS,
    ManufacturingStructureOptions,
    build_manufacturing_field,
    resolve_target_relative_density,
)
from ..manufacturing.member_sizing import OptimizedMemberPlan
from .analytic_shapes import (
    _convert_legacy_to_physical_shapes,
    _project_passive_shapes_surfaces,
)
from .mesh_postprocess import _enhanced_mesh_postprocess, _taubin_smooth_surface
from .passive_regions import (
    _apply_passive_cylinder_sdf,
    _apply_passive_density_regions,
    _resample_source_mask,
)
from .print_surface import (
    extract_flying_edges_surface,
    extract_surface_nets_surface,
    extract_volume_calibrated_surface,
    volume_preserving_level_field,
)
from .recovery_grid import (
    BoxRegion,
    CylinderRegion,
    _project_extruded_planes,
    _regularize_extruded_density,
    _voxel_origin_cell,
)
from . import analytic_shapes as _analytic_shapes
from . import mesh_postprocess as _mesh_postprocess_module
from . import passive_regions as _passive_regions
from . import recovery_grid as _recovery_grid

logger = logging.getLogger(__name__)

# Diagnostics from the most recent lattice sizing, for the caller to report.
# The recovery entry point returns a mesh, so there is no result dict to thread
# this through, and a user given a relative-density target has to be told what
# was actually achieved -- silently missing a mass budget is the failure mode
# this control exists to remove.
LAST_LATTICE_SIZING: dict[str, float] = {}


def _recovery_smoothing_sigmas(
    zoom_factors: np.ndarray,
    *,
    smooth_input: bool,
) -> tuple[float, float, float]:
    """Return axis-aware Gaussian sigmas for the recovery grid.

    Extruded studies refine only their in-plane axes. Scaling a scalar sigma by
    the largest zoom factor also scales the unrefined thickness axis and can
    erase a valid grey load path through the part. Each axis therefore carries
    its own source-voxel-to-recovery-voxel conversion.
    """
    zoom = np.asarray(zoom_factors, dtype=float).reshape(3)
    if float(np.max(zoom)) <= 1.0:
        value = 0.08 if smooth_input else 0.20
        return (value, value, value)
    source_voxel_sigma = 0.12 if smooth_input else 0.32
    values = np.maximum(0.05, source_voxel_sigma * zoom)
    return tuple(float(value) for value in values)


def _protected_voxel_core_sdf(
    material_mask: np.ndarray,
    target_shape: tuple[int, int, int],
    *,
    pad: int,
    radius_voxels: float = 0.56,
    extrusion_axis: str = "none",
) -> np.ndarray | None:
    """Return a smooth keep-material core through every retained source cell.

    Surface fairing is allowed to move the boundary between voxel centres, but
    it must not move it *past* a cell that the analyzed projected density kept.
    A smooth core through every retained cell centre provides that local
    invariant without forcing recovery to keep the full blocky voxel cube.

    The distance metric is expressed in source-voxel coordinates.  A radius a
    prismatic study connects neighbouring in-plane centres with line segments,
    including diagonal neighbours, then rounds that graph with a sub-voxel
    radius. This is deliberately a core, not a replacement shape: the filtered
    density still determines the visible free boundary.
    """
    from scipy import ndimage as ndi

    source = np.asarray(material_mask, dtype=bool)
    if source.ndim != 3 or not np.any(source):
        return None
    target = np.asarray(target_shape, dtype=int)
    if target.shape != (3,) or np.any(target < 1):
        return None

    source_shape = np.asarray(source.shape, dtype=float)
    axis = {"x": 0, "y": 1, "z": 2}.get(
        str(extrusion_axis or "").strip().lower()
    )
    if axis is not None:
        # A prismatic protection field must itself be prismatic. Spherical
        # three-dimensional cores leave a bead at every unrefined thickness
        # sample and turn a clean extrusion into hundreds of thousands of side
        # triangles. Protect the in-plane centres and broadcast their smooth
        # distance field through the requested sweep instead.
        in_plane = [index for index in range(3) if index != axis]
        source_2d = np.any(source, axis=axis)
        target_2d = target[in_plane]
        source_shape_2d = source_shape[in_plane]
        scale_2d = target_2d.astype(float) / source_shape_2d
        centres_2d = np.argwhere(source_2d)
        mapped_2d = np.rint(
            (centres_2d.astype(float) + 0.5) * scale_2d - 0.5
        ).astype(int)
        mapped_2d = np.clip(mapped_2d, 0, target_2d - 1)
        seeds_2d = np.zeros(tuple(int(value) for value in target_2d), dtype=bool)
        seeds_2d[tuple(mapped_2d.T)] = True
        try:
            from skimage.draw import line

            # Preserve the connectivity visible in the density view. A chain
            # of diagonally adjacent retained cells is a deliberate sloping
            # member, not a collection of isolated islands. Rasterize its
            # centre graph before taking a distance field so recovery produces
            # one smooth diagonal capsule instead of disconnected round bumps.
            neighbour_offsets = ((0, 1), (1, -1), (1, 0), (1, 1))
            source_rows, source_columns = source_2d.shape
            for centre in np.argwhere(source_2d):
                for offset in neighbour_offsets:
                    neighbour = centre + np.asarray(offset, dtype=int)
                    if (
                        neighbour[0] < 0
                        or neighbour[0] >= source_rows
                        or neighbour[1] < 0
                        or neighbour[1] >= source_columns
                        or not source_2d[tuple(neighbour)]
                    ):
                        continue
                    endpoints = np.vstack((centre, neighbour)).astype(float)
                    endpoints = np.rint(
                        (endpoints + 0.5) * scale_2d - 0.5
                    ).astype(int)
                    endpoints = np.clip(endpoints, 0, target_2d - 1)
                    rows, columns = line(
                        int(endpoints[0, 0]),
                        int(endpoints[0, 1]),
                        int(endpoints[1, 0]),
                        int(endpoints[1, 1]),
                    )
                    seeds_2d[rows, columns] = True
        except Exception:
            logger.debug(
                "Diagonal protected-path rasterization failed; keeping centre cores.",
                exc_info=True,
            )
        distance_2d = ndi.distance_transform_edt(
            ~seeds_2d,
            sampling=tuple(
                float(value)
                for value in source_shape_2d / target_2d.astype(float)
            ),
        )
        # The connecting graph, not overlapping circles, supplies continuity.
        # A smaller round radius avoids the bead/scallop outline that centre
        # spheres produced at every diagonal voxel.
        core_2d = distance_2d - min(
            0.40, float(np.clip(radius_voxels, 0.20, 0.62))
        )
        expanded = np.expand_dims(core_2d, axis=axis)
        core = np.broadcast_to(
            expanded,
            tuple(int(value) for value in target),
        ).copy()
    else:
        scale = target.astype(float) / source_shape
        centres = np.argwhere(source)
        mapped = np.rint(
            (centres.astype(float) + 0.5) * scale - 0.5
        ).astype(int)
        mapped = np.clip(mapped, 0, target - 1)
        seeds = np.zeros(tuple(int(value) for value in target), dtype=bool)
        seeds[tuple(mapped.T)] = True

        # `sampling` converts recovery-grid index distances back to source
        # voxels, yielding ellipsoids when the source grid is anisotropic.
        source_voxel_sampling = source_shape / target.astype(float)
        distance = ndi.distance_transform_edt(
            ~seeds,
            sampling=tuple(float(value) for value in source_voxel_sampling),
        )
        core = distance - float(np.clip(radius_voxels, 0.501, 0.62))
    if int(pad) > 0:
        outside = max(float(np.max(core)) + 1.0, 1.0)
        core = np.pad(
            core,
            pad_width=int(pad),
            mode="constant",
            constant_values=outside,
        )
    return np.asarray(core, dtype=float)


def _surface_point_coverage(
    vertices: np.ndarray,
    faces: np.ndarray,
    points: np.ndarray,
) -> tuple[int, int] | None:
    """Return how many protected points lie inside a closed recovered mesh."""
    query_points = np.asarray(points, dtype=float)
    if query_points.ndim != 2 or query_points.shape[1] != 3:
        return None
    if len(query_points) == 0:
        return (0, 0)
    try:
        import vtk
        from vtk.util.numpy_support import vtk_to_numpy

        surface = vtk.vtkPolyData()
        surface_points = vtk.vtkPoints()
        surface_points.SetNumberOfPoints(len(vertices))
        for index, point in enumerate(np.asarray(vertices, dtype=float)):
            surface_points.SetPoint(index, *(float(value) for value in point))
        surface.SetPoints(surface_points)
        polygons = vtk.vtkCellArray()
        for face in np.asarray(faces, dtype=np.int64):
            triangle = vtk.vtkTriangle()
            for corner, vertex_id in enumerate(face[:3]):
                triangle.GetPointIds().SetId(corner, int(vertex_id))
            polygons.InsertNextCell(triangle)
        surface.SetPolys(polygons)

        query = vtk.vtkPolyData()
        vtk_query_points = vtk.vtkPoints()
        vtk_query_points.SetNumberOfPoints(len(query_points))
        for index, point in enumerate(query_points):
            vtk_query_points.SetPoint(index, *(float(value) for value in point))
        query.SetPoints(vtk_query_points)

        enclosed = vtk.vtkSelectEnclosedPoints()
        enclosed.SetInputData(query)
        enclosed.SetSurfaceData(surface)
        enclosed.SetTolerance(1.0e-7)
        enclosed.Update()
        selected = enclosed.GetOutput().GetPointData().GetArray("SelectedPoints")
        if selected is None:
            return None
        inside = np.asarray(vtk_to_numpy(selected), dtype=bool)
        return int(np.count_nonzero(inside)), int(len(inside))
    except Exception:
        logger.debug("Protected-voxel coverage check failed.", exc_info=True)
        return None


def _count_matched_level(field: np.ndarray, target_count: int) -> float:
    """Choose an iso-level enclosing approximately ``target_count`` samples."""
    values = np.asarray(field, dtype=float).reshape(-1)
    if values.size == 0 or target_count <= 0:
        return 0.5
    count = int(np.clip(target_count, 1, values.size - 1))
    ordered = np.sort(values)
    split = values.size - count
    lower = float(ordered[max(0, split - 1)])
    upper = float(ordered[min(values.size - 1, split)])
    return float(np.clip(0.5 * (lower + upper), 0.02, 0.98))


def _record_lattice_density(
    target: float,
    achieved: float,
    member_thickness_voxels: float,
    spacing: np.ndarray,
) -> None:
    """Log and record what the lattice sizing actually reached."""
    try:
        member_mm = float(member_thickness_voxels) * float(np.mean(spacing))
    except Exception:
        member_mm = float("nan")
    LAST_LATTICE_SIZING.clear()
    LAST_LATTICE_SIZING.update(
        target_relative_density=float(target),
        achieved_relative_density=float(achieved),
        member_thickness_voxels=float(member_thickness_voxels),
        member_thickness_model_units=member_mm,
    )
    if not np.isfinite(achieved):
        return
    if abs(achieved - target) <= 0.015:
        logger.info(
            "Lattice sized to %.1f%% relative density (target %.1f%%) with a "
            "%.3g unit member.",
            achieved * 100.0,
            target * 100.0,
            member_mm,
        )
        return
    logger.warning(
        "Lattice reached %.1f%% relative density against a %.1f%% target; the "
        "%.3g unit member is at the limit this cell pitch and analysis grid can "
        "resolve. Change the cell pitch or the quality preset to move it.",
        achieved * 100.0,
        target * 100.0,
        member_mm,
    )

_HELPER_MODULES = (
    _recovery_grid,
    _analytic_shapes,
    _mesh_postprocess_module,
    _passive_regions,
)


def __getattr__(name: str) -> object:
    """Resolve helpers moved out of this formerly monolithic module."""
    for module in _HELPER_MODULES:
        if hasattr(module, name):
            return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    helper_names = {
        name
        for module in _HELPER_MODULES
        for name in dir(module)
        if name.startswith("_") or not name.startswith("__")
    }
    return sorted(set(globals()) | helper_names)


def _recover_voxel_shape(
    density: np.ndarray,
    bounds: Optional[tuple[np.ndarray, np.ndarray]],
    cutoff: float,
    print_ready: bool = False,
    decimate_ratio: float = 1.0,
    solid_boxes: Sequence[BoxRegion] = (),
    void_boxes: Sequence[BoxRegion] = (),
    solid_cylinders: Sequence[CylinderRegion] = (),
    void_cylinders: Sequence[CylinderRegion] = (),
    joint_pin_cylinders: Sequence[CylinderRegion] = (),
    extrusion_axis: str = "none",
    source_mask: Optional[np.ndarray] = None,
    passive_solid_mask: Optional[np.ndarray] = None,
    passive_void_mask: Optional[np.ndarray] = None,
    blend_radius: float = 0.0,
    structure_options: Optional[ManufacturingStructureOptions] = None,
    member_plan: Optional[OptimizedMemberPlan] = None,
    surface_backend: str = "vtk_sdf",
    surface_quality: str = "Professional",
) -> Optional[dict[str, np.ndarray]]:
    """Extract a recovered surface from a structured voxel density field.

    With ``print_ready=True``, the preferred path uses a volume-preserving
    signed-distance field and VTK Flying Edges, followed by mesh repair and
    optional decimation. ``surface_backend="legacy"`` retains marching cubes.

    ``structure_options`` can replace the solid envelope with a TPMS, strut or
    honeycomb lattice. This is manufacturing geometry and must be verified
    independently from the continuous SIMP result.
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
        # ``cutoff`` has already been matched to the physical density used by
        # the FE solve.  Preserve this pre-recovery material measure.  The
        # support gate, interpolation and analytic-feature reapplication below
        # are geometric operations; none of them is allowed to redefine the
        # optimizer's material budget.
        source_material_mask = grid >= cutoff
        if source_mask is not None and np.shape(source_mask) == grid.shape:
            source_material_mask &= np.asarray(source_mask, dtype=bool)
        if (
            passive_void_mask is not None
            and np.shape(passive_void_mask) == grid.shape
        ):
            source_material_mask &= ~np.asarray(passive_void_mask, dtype=bool)
        if (
            passive_solid_mask is not None
            and np.shape(passive_solid_mask) == grid.shape
        ):
            source_material_mask |= np.asarray(passive_solid_mask, dtype=bool)
        source_target_count = int(np.count_nonzero(source_material_mask))
        # Preserve the analyzed thresholded cells as full cells. Re-interpreting
        # their boundary with a second interpolation scheme changes the
        # optimizer's material budget, especially when valid material lies only
        # slightly above the selected cutoff.
        origin_for_protection, analysis_cell = _voxel_origin_cell(
            tuple(grid.shape), bounds
        )
        analyzed_reference_volume = (
            float(source_target_count) * float(np.prod(analysis_cell))
        )
        source_material_centres = (
            origin_for_protection
            + (np.argwhere(source_material_mask).astype(float) + 0.5)
            * analysis_cell
            if source_target_count
            else np.empty((0, 3), dtype=float)
        )
        protection_axis = {"x": 0, "y": 1, "z": 2}.get(
            str(extrusion_axis or "").strip().lower()
        )
        protected_profile_points = np.empty((0, 3), dtype=float)
        if protection_axis is not None and len(source_material_centres):
            protected_profile_points = source_material_centres.copy()
            if bounds is not None:
                protected_profile_points[:, protection_axis] = 0.5 * (
                    float(bounds[0][protection_axis])
                    + float(bounds[1][protection_axis])
                )
            protected_profile_points = np.unique(
                np.round(protected_profile_points, decimals=12),
                axis=0,
            )
        grid = _regularize_extruded_density(grid, extrusion_axis)
        if str(extrusion_axis or "").strip().lower() in {"x", "y", "z"}:
            # Averaging enforces the requested constant cross-section but also
            # changes the scalar distribution. Reusing the pre-average cutoff
            # can delete every grey load path and leave only passive bosses.
            # Recalibrate the level so extrusion preserves the analyzed volume.
            cutoff = _count_matched_level(grid, source_target_count)

        quality_key = str(surface_quality or "Professional").strip().lower()
        (
            min_dimension_target,
            recovery_point_cap,
            vtk_iterations,
            vtk_pass_band,
        ) = {
            "standard": (24.0, 1_000_000, 16, 0.09),
            "high": (36.0, 2_000_000, 26, 0.055),
            "professional": (48.0, 3_500_000, 36, 0.035),
        }.get(quality_key, (48.0, 3_500_000, 36, 0.035))

        axis_map = {"x": 0, "y": 1, "z": 2}
        extrusion_ax = axis_map.get(str(extrusion_axis or "").strip().lower())
        if extrusion_ax is not None:
            in_plane = [i for i in range(3) if i != extrusion_ax]
            min_plane_dim = max(1, min(int(grid.shape[i]) for i in in_plane))
            extrusion_target = max(240.0, 4.0 * min_dimension_target)
            upsample = int(
                np.clip(np.ceil(extrusion_target / min_plane_dim), 1, 10)
            )
            zoom_factors = np.ones(3, dtype=float)
            zoom_factors[in_plane] = float(upsample)
            # The fast planar solver may retain only two 3-D element layers.
            # Flying Edges needs at least four scalar samples through a closed
            # slab to form a stable two-manifold. Refine only the recovery field
            # to that minimum; this does not add design freedom or FE accuracy,
            # and the physical thickness/volume remain unchanged through the
            # correspondingly reduced sample spacing.
            zoom_factors[extrusion_ax] = max(
                1.0,
                4.0 / max(float(grid.shape[extrusion_ax]), 1.0),
            )
        else:
            min_dim = max(1, min(grid.shape))
            # Cubic interpolation beyond roughly 36 samples on the thinnest
            # axis added tens of thousands of nearly coplanar triangles but no
            # recoverable design information. Print-ready smoothing and
            # quadric decimation provide the useful finish at much lower cost.
            upsample = int(
                np.clip(np.ceil(min_dimension_target / min_dim), 1, 12)
            )
            # A lattice study needs the grid to resolve its *cell*, not just the
            # part's thinnest dimension. `min_dimension_target` was written for
            # solid recovery, where the only question is how finely the free
            # boundary is sampled, and on a lattice study it is the wrong
            # question: it capped the bundled crush block at 2x while the point
            # budget allowed 3.85x, so 86% of the allowance went unspent and the
            # cell had to be made coarse enough to survive the grid it was
            # given. That is backwards -- the grid should be refined to fit the
            # cell the user asked for, and only then should the cell be limited.
            # Asking for the family's own resolved floor here is what lets the
            # pitch itself come down.
            if structure_options is not None and structure_options.mode != "solid":
                family = structure_options.family
                required = float(
                    family.minimum_cell_voxels if family is not None else 8.0
                )
                requested_cell = max(
                    float(structure_options.cell_size_voxels), 1e-6
                )
                upsample = int(
                    np.clip(
                        max(upsample, np.ceil(required / requested_cell)), 1, 12
                    )
                )
            zoom_factors = np.full(3, float(upsample), dtype=float)

        # Fit the point budget with a single fractional factor shared by every
        # refined axis. Two things were wrong with stepping the largest axis down
        # by one: it left the grid anisotropic -- a 122x62x20 study resolved to
        # zoom (2, 3, 3), stretching each recovery voxel 1.5x along X, which
        # terraces the recovered surface along the stretched axis and lands a
        # requested strut 1.19x heavy because the lattice sizes below are
        # denominated in voxels -- and stepping in whole integers instead
        # squanders the budget, dropping that study from 3 to 2 and using 1.2M of
        # the 3.5M points it was allowed. The coarser grid inflates a thin strut
        # more than the anisotropy did. A fractional factor keeps the voxels
        # cubic and spends the whole allowance. An extruded study deliberately
        # leaves its extrusion axis at 1.0, so only refined axes are scaled.
        refined = zoom_factors > 1.0
        n_refined = int(np.count_nonzero(refined))
        if n_refined:
            base_points = float(np.prod(np.asarray(grid.shape, dtype=float)))
            allowed = (
                float(recovery_point_cap) / max(base_points, 1.0)
            ) ** (1.0 / float(n_refined))
            zoom_factors[refined] = max(
                1.0, min(float(np.max(zoom_factors)), allowed)
            )

        # Sizes stated in source voxels convert by the factor applied to the
        # axes that were refined, which are now all equal. Averaging in an
        # unrefined extrusion axis would under-size every in-plane member.
        structure_resolution_scale = float(
            np.max(zoom_factors) if np.any(zoom_factors > 1.0) else 1.0
        )

        if np.any(zoom_factors > 1.0):
            field = ndi.zoom(
                grid,
                zoom=tuple(float(v) for v in zoom_factors),
                # The optimizer has already supplied a physically cone-filtered
                # physical density. Cubic resampling can overshoot between
                # voxels and create the repeating humps/ruts that users read
                # as fake load paths. Linear interpolation is monotone; the
                # signed-distance and windowed-sinc stages below provide the
                # geometric fairing without inventing extrema.
                order=1,
                mode="nearest",
                grid_mode=True,
            )
        else:
            field = grid.copy()

        origin, cell = _voxel_origin_cell(tuple(grid.shape), bounds)
        spacing = cell / zoom_factors

        field = np.clip(field, 0.0, 1.0)
        source_field = _resample_source_mask(source_mask, tuple(field.shape))
        explicit_solid_field = _resample_source_mask(
            passive_solid_mask, tuple(field.shape)
        )
        explicit_void_field = _resample_source_mask(
            passive_void_mask, tuple(field.shape)
        )
        # Express fairing in *source voxel* units. A fixed sigma of 0.35 on a
        # 10x recovery grid is only 0.035 of a source voxel and therefore
        # leaves every voxel terrace visible in the manufactured surface.
        #
        # How much is needed depends on what the caller supplied. A filtered
        # (pre-projection) density already varies smoothly over ~2*rmin voxels,
        # so it needs only enough blur to remove interpolation creases; blurring
        # it as hard as a binary field thins genuine load paths, because the
        # two smoothing passes compound. A near-binary field — the legacy
        # projected input, or an explicit lattice field built by
        # `build_manufacturing_field` — still needs the full third of a voxel to
        # bury its terraces. Measure the input instead of assuming.
        interior = field[(field > 0.02) & (field < 0.98)]
        grey_fraction = float(interior.size) / float(max(field.size, 1))
        smooth_input = grey_fraction >= 0.25
        sigma = _recovery_smoothing_sigmas(
            zoom_factors,
            smooth_input=smooth_input,
        )
        field = ndi.gaussian_filter(field, sigma=sigma, mode="nearest")
        if source_field is not None:
            # The source CAD is voxelized before optimization. Hard clipping
            # to a nearest-neighbour Boolean mask reintroduces stair steps even
            # after cubic density interpolation. Use a sub-voxel, monotone
            # support gate for recovery only; the later level calibration
            # preserves material and the analytic interface regions are
            # re-imposed below.
            source_gate = ndi.gaussian_filter(
                source_field.astype(float),
                sigma=tuple(
                    float(value)
                    for value in np.maximum(0.5, 0.40 * zoom_factors)
                ),
                mode="nearest",
            )
            field *= np.clip(source_gate, 0.0, 1.0)
            field[source_gate < 1e-5] = 0.0
        if explicit_solid_field is not None:
            field[explicit_solid_field] = 1.0
        # Source masks are voxel approximations of the input body; passive keep
        # and cut regions are analytic constraints and must be re-imposed after
        # that coarse clipping so round holes/bosses stay round in recovery.
        field = _apply_passive_density_regions(
            field,
            solid_boxes=solid_boxes,
            void_boxes=void_boxes,
            solid_cylinders=solid_cylinders,
            void_cylinders=void_cylinders,
            bounds=bounds,
            cell=cell,
            spacing=spacing,
        )
        # A requested keep-void is the last word, and it has to be, because
        # `solid_boxes` is not only the user's keep-material: it also carries
        # the pads this study generated automatically around every load and
        # support. Written before the analytic pass, an explicit bore was
        # simply overwritten by the pad sitting on top of it -- the solve had
        # the bore open (`_assemble_passive_masks` applies the void mask last)
        # and the recovered surface came back with it filled, so the delivered
        # geometry was not the geometry that was analyzed. Commercial practice
        # is the same ordering: an automatic boundary-condition exclusion is a
        # convenience underneath the user's non-design statement, never over it.
        if explicit_void_field is not None:
            field[explicit_void_field] = 0.0
        design_count_before_hardware = int(
            np.count_nonzero(
                field >= float(np.clip(cutoff, 1e-6, 0.999999))
            )
        )
        # Joint pins are separate assembly hardware, not passive optimizer
        # material. Add them only to the manufactured/recovered field and only
        # after through-bores have been re-imposed.
        field = _apply_passive_density_regions(
            field,
            solid_cylinders=joint_pin_cylinders,
            bounds=bounds,
            cell=cell,
            spacing=spacing,
        )
        assembly_hardware_sample_count = max(
            0,
            int(
                np.count_nonzero(
                    field >= float(np.clip(cutoff, 1e-6, 0.999999))
                )
            )
            - design_count_before_hardware,
        )
        lattice_sizing: dict[str, float] = {}
        lattice_connectivity: dict[str, object] = {}
        if structure_options is not None and structure_options.mode != "solid":
            build_options = structure_options
            # A stated relative density is a mass budget, so it has to be met on
            # the grid that actually gets manufactured. Solve for the member
            # thickness here rather than at analysis resolution: the two
            # disagree by roughly 20% because this grid is supersampled.
            if structure_options.target_relative_density > 0.0:
                build_options, reached = resolve_target_relative_density(
                    field,
                    cutoff,
                    structure_options,
                    resolution_scale=structure_resolution_scale,
                    passive_solid_mask=explicit_solid_field,
                    passive_void_mask=explicit_void_field,
                    member_plan=member_plan,
                )
                _record_lattice_density(
                    structure_options.target_relative_density,
                    reached,
                    build_options.member_thickness_voxels,
                    spacing,
                )
                lattice_sizing = dict(LAST_LATTICE_SIZING)
            field = build_manufacturing_field(
                field,
                cutoff,
                build_options,
                resolution_scale=structure_resolution_scale,
                passive_solid_mask=explicit_solid_field,
                passive_void_mask=explicit_void_field,
                member_plan=member_plan,
            )
            # How much of the optimized envelope the built lattice actually
            # reached, and what it cost to keep it in one piece. Captured from
            # the build that is delivered, after any relative-density search.
            lattice_connectivity = dict(LAST_STRUCTURE_DIAGNOSTICS)
            # Contact pads, bores, and other passive engineering features have
            # priority over the generated infill representation.
            field = _apply_passive_density_regions(
                field,
                solid_boxes=solid_boxes,
                void_boxes=void_boxes,
                solid_cylinders=solid_cylinders,
                void_cylinders=void_cylinders,
                bounds=bounds,
                cell=cell,
                spacing=spacing,
            )
            # And a requested keep-void outranks all of them, for the reason
            # given above: the automatic load/support pads travel in
            # `solid_boxes` and must not close a bore the study asked for.
            if explicit_void_field is not None:
                field[explicit_void_field] = 0.0
            field = _apply_passive_density_regions(
                field,
                solid_cylinders=joint_pin_cylinders,
                bounds=bounds,
                cell=cell,
                spacing=spacing,
            )
        # Volume target for the level recalibration below.
        #
        # A manufactured lattice is intentionally created on this supersampled
        # grid, so its thresholded volume is the correct reference.  A solid
        # envelope is different: its analyzed material volume was fixed before
        # recovery.  Recounting after interpolation, support gating and
        # smoothing silently makes those operations a second optimizer.  A
        # coarse grey boundary can then lose an entire voxel layer (16% on the
        # regression block), even though the final level calibration perfectly
        # matches the *already shrunken* count.  Carry the analyzed physical
        # volume through instead, plus only the separate joint-pin hardware.
        recovery_voxel_volume = float(np.prod(spacing))
        postprocessed_material_count = float(
            np.count_nonzero(field >= float(np.clip(cutoff, 1e-6, 0.999999)))
        )
        if structure_options is not None and structure_options.mode != "solid":
            target_material_count = postprocessed_material_count
            recovery_target_volume = (
                target_material_count * recovery_voxel_volume
            )
        else:
            recovery_target_volume = analyzed_reference_volume + (
                float(assembly_hardware_sample_count) * recovery_voxel_volume
            )
            target_material_count = (
                recovery_target_volume / recovery_voxel_volume
                if recovery_voxel_volume > 0.0
                else float(source_target_count)
            )
        # This is the actual manufactured-volume reference used by recovery,
        # measured on the supersampled grid on which members were sized.  The
        # coarser analysis-grid field can differ substantially for one-voxel
        # lattice members and must not be used to judge recovered CAD volume.
        manufacturing_reference_volume = (
            recovery_target_volume
            if structure_options is not None
            and structure_options.mode != "solid"
            else None
        )
        pad = max(3, min(12, int(np.ceil(float(np.max(zoom_factors))))))
        solid_envelope = (
            structure_options is None or structure_options.mode == "solid"
        )
        protected_core_sdf = (
            _protected_voxel_core_sdf(
                source_material_mask,
                tuple(int(value) for value in field.shape),
                pad=pad,
                extrusion_axis=extrusion_axis,
            )
            if solid_envelope and source_target_count > 0
            else None
        )
        field = np.pad(field, pad_width=pad, mode="constant", constant_values=0.0)

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
        # negative is material, positive is void.  Passive shapes are applied
        # before marching cubes so circular holes do not need post-hoc vertex
        # snapping, which can fold triangles near tight bolt holes.
        iso_field = level - field
        if protected_core_sdf is not None:
            # Fairing may move the free boundary between retained cell centres,
            # never across one.  The passive analytic regions below still have
            # final authority, so a prescribed bore is not filled by this core.
            iso_field = np.minimum(iso_field, protected_core_sdf)
        mc_level = 0.0
        passive_shapes_present = bool(
            solid_cylinders
            or void_cylinders
            or solid_boxes
            or void_boxes
            or joint_pin_cylinders
        )
        analytic_cylinder_iso = False
        if passive_shapes_present:
            iso_field = _apply_passive_cylinder_sdf(
                iso_field,
                pad=pad,
                solid_boxes=solid_boxes,
                void_boxes=void_boxes,
                solid_cylinders=solid_cylinders,
                void_cylinders=void_cylinders,
                bounds=bounds,
                cell=cell,
                spacing=spacing,
                blend_radius=blend_radius,
            )
            iso_field = _apply_passive_cylinder_sdf(
                iso_field,
                pad=pad,
                solid_cylinders=joint_pin_cylinders,
                bounds=bounds,
                cell=cell,
                spacing=spacing,
                blend_radius=blend_radius,
            )
            analytic_cylinder_iso = True

        # Prefer the filtered physical density field itself for the visible
        # boundary.  The previous binary-mask -> distance-transform SDF route
        # made thin topology webs look like terraced contour plots because the
        # signed-distance bands were smoothed after thresholding.
        if not (float(np.min(iso_field)) < mc_level < float(np.max(iso_field))):
            outside = ndi.distance_transform_edt(
                ~mask, sampling=tuple(float(v) for v in spacing)
            )
            inside = ndi.distance_transform_edt(
                mask, sampling=tuple(float(v) for v in spacing)
            )
            iso_field = outside - inside
            iso_field = ndi.gaussian_filter(iso_field, sigma=0.35)
            if protected_core_sdf is not None:
                iso_field = np.minimum(iso_field, protected_core_sdf)
            if passive_shapes_present:
                iso_field = _apply_passive_cylinder_sdf(
                    iso_field,
                    pad=pad,
                    solid_boxes=solid_boxes,
                    void_boxes=void_boxes,
                    solid_cylinders=solid_cylinders,
                    void_cylinders=void_cylinders,
                    bounds=bounds,
                    cell=cell,
                    spacing=spacing,
                    blend_radius=blend_radius,
                )
                iso_field = _apply_passive_cylinder_sdf(
                    iso_field,
                    pad=pad,
                    solid_cylinders=joint_pin_cylinders,
                    bounds=bounds,
                    cell=cell,
                    spacing=spacing,
                    blend_radius=blend_radius,
                )
                analytic_cylinder_iso = True
            mc_level = 0.0
            if not (float(np.min(iso_field)) < mc_level < float(np.max(iso_field))):
                return None

        surface_origin = origin + 0.5 * spacing - float(pad) * spacing

        active_shapes = ()
        if passive_shapes_present:
            active_shapes = _convert_legacy_to_physical_shapes(
                bounds,
                solid_boxes=solid_boxes,
                void_boxes=void_boxes,
                solid_cylinders=solid_cylinders,
                void_cylinders=void_cylinders,
                # The pin is a separate solid inside the bore. Including it in
                # the projection set keeps its analytic circular surface
                # protected if a non-SDF fallback is used.
                # It is deliberately appended after the bore definition.
                # `_convert_legacy_to_physical_shapes` is order independent for
                # projection, while scalar composition above enforces order.
            )
            active_shapes = tuple(active_shapes) + tuple(
                _convert_legacy_to_physical_shapes(
                    bounds,
                    solid_cylinders=joint_pin_cylinders,
                )
            )

        def _finish_surface(
            raw_vertices: np.ndarray,
            raw_faces: np.ndarray,
            *,
            finish_iterations: int,
        ) -> np.ndarray:
            """Apply every geometric step that moves vertices, in pipeline order.

            Extracted as a closure so volume calibration can measure the surface
            the user actually receives. Calibrating the bare isosurface instead
            was measurably wrong: on the bundled cantilever the extraction hit
            its target to 0.0007% and the delivered solid was still 1.5% heavy,
            because snapping the extruded end caps out onto the exact design
            planes afterwards adds 1.35% -- correctly, since the part does fill
            its full thickness, but after the level had already been fixed.

            Only volume-neutral cleanup stays outside this function.
            """
            finished = _taubin_smooth_surface(
                raw_vertices,
                raw_faces,
                iterations=int(finish_iterations),
                shapes=active_shapes,
                tolerance=float(np.max(spacing)) * 3.0,
                lock_open_boundaries=True,
            )
            finished = _project_extruded_planes(
                finished,
                bounds,
                extrusion_axis,
                tolerance=float(np.max(spacing)) * 2.5,
            )
            if passive_shapes_present and not analytic_cylinder_iso:
                finished = _project_passive_shapes_surfaces(
                    finished,
                    active_shapes,
                    tolerance=float(np.max(spacing)) * 3.0,
                )
            if bounds is not None:
                finished = np.clip(
                    finished,
                    np.asarray(bounds[0], dtype=float)[:3],
                    np.asarray(bounds[1], dtype=float)[:3],
                )
            return finished

        backend_name = "scikit-image marching cubes"
        vtk_surface = None
        backend_key = str(surface_backend or "").strip().lower()
        use_surface_nets = backend_key in {"surface_nets", "surfacenets", "nets"}
        use_vtk_sdf = use_surface_nets or backend_key not in {
            "legacy",
            "marching_cubes",
            "skimage",
        }
        if use_vtk_sdf:
            material_mask = iso_field <= float(mc_level)
            try:
                base_sdf, _ = volume_preserving_level_field(
                    iso_field,
                    material_mask,
                    target_material_count=target_material_count,
                )

                def _field_at_level(level_shift: float) -> np.ndarray:
                    """Move the optimizer's free boundary, then restore features.

                    Volume calibration is allowed to move the boundary the
                    optimizer chose. It is not allowed to resize a bore or an
                    interface pad, so the analytic shapes are re-composited after
                    the shift rather than being carried along by it.
                    """
                    shifted = base_sdf - float(level_shift)
                    if protected_core_sdf is not None:
                        shifted = np.minimum(shifted, protected_core_sdf)
                    if passive_shapes_present:
                        shifted = _apply_passive_cylinder_sdf(
                            shifted,
                            pad=pad,
                            solid_boxes=solid_boxes,
                            void_boxes=void_boxes,
                            solid_cylinders=solid_cylinders,
                            void_cylinders=void_cylinders,
                            bounds=bounds,
                            cell=cell,
                            spacing=spacing,
                            blend_radius=blend_radius,
                        )
                        shifted = _apply_passive_cylinder_sdf(
                            shifted,
                            pad=pad,
                            solid_cylinders=joint_pin_cylinders,
                            bounds=bounds,
                            cell=cell,
                            spacing=spacing,
                            blend_radius=blend_radius,
                        )
                    return shifted

                def _finished_at_level(level_shift: float):
                    field_now = _field_at_level(level_shift)
                    extracted = (
                        extract_surface_nets_surface(
                            field_now,
                            spacing,
                            surface_origin,
                            smoothing_iterations=vtk_iterations,
                        )
                        if use_surface_nets
                        else extract_flying_edges_surface(
                            field_now,
                            spacing,
                            surface_origin,
                            smoothing_iterations=vtk_iterations,
                            pass_band=vtk_pass_band,
                        )
                    )
                    if extracted is None:
                        return None
                    raw_faces = np.asarray(extracted["faces"], dtype=int)
                    return {
                        "vertices": _finish_surface(
                            np.asarray(extracted["vertices"], dtype=float),
                            raw_faces,
                            finish_iterations=4 if print_ready else 2,
                        ),
                        "faces": raw_faces,
                        "surface_backend": extracted["surface_backend"],
                    }

                vtk_surface = extract_volume_calibrated_surface(
                    _finished_at_level,
                    target_volume=recovery_target_volume,
                )
            except Exception:
                logger.exception(
                    "Signed-distance print recovery failed; using marching cubes"
                )

        if vtk_surface is not None:
            verts = np.asarray(vtk_surface["vertices"], dtype=float)
            faces = np.asarray(vtk_surface["faces"], dtype=int)
            backend_name = str(vtk_surface["surface_backend"])
        else:
            verts, faces, _, _ = measure.marching_cubes(
                iso_field,
                level=float(mc_level),
                spacing=tuple(float(v) for v in spacing),
                gradient_direction="ascent",
            )
            verts = verts + surface_origin
        if len(verts) == 0 or len(faces) == 0:
            return None

        # The VTK path already ran `_finish_surface` inside the volume
        # calibration, on the level it settled on. Only the marching-cubes
        # fallback still needs finishing here, and it needs a heavier Taubin
        # pass because it has had no windowed-sinc fairing.
        #
        # `maximum_step` is deliberately left unset. It bounds how far a vertex
        # can travel per pass, which protects a thin member on a coarse grid, but
        # the vectors it clamps are exactly the high-curvature terrace corners
        # this pass exists to remove, and on a well-resolved grid it perturbs
        # vertices by a third of a voxel for no measurable volume benefit.
        if vtk_surface is None:
            verts = _finish_surface(
                verts,
                faces,
                finish_iterations=12 if print_ready else 4,
            )
        if bounds is not None:
            bound_min = np.asarray(bounds[0], dtype=float)[:3]
            bound_max = np.asarray(bounds[1], dtype=float)[:3]

        # Always run topology-preserving cleanup so every result reports
        # watertightness, open edges, and component count. Decimation remains
        # opt-in because it can trade away small geometric detail.
        def _attach_local_density_report(payload: dict[str, object]) -> None:
            if not solid_envelope or source_target_count <= 0:
                return
            coverage = _surface_point_coverage(
                np.asarray(payload["vertices"], dtype=float),
                np.asarray(payload["faces"], dtype=np.int64),
                source_material_centres,
            )
            if coverage is None:
                payload["protected_voxel_center_coverage"] = None
                return
            inside_count, total_count = coverage
            fraction = (
                float(inside_count) / float(total_count)
                if total_count > 0
                else 1.0
            )
            payload["protected_voxel_centers_inside"] = int(inside_count)
            payload["protected_voxel_centers_total"] = int(total_count)
            payload["protected_voxel_center_coverage"] = float(fraction)
            payload["local_density_preserved"] = bool(inside_count == total_count)
            if inside_count != total_count:
                logger.warning(
                    "Recovered surface excluded %d of %d above-cutoff voxel "
                    "centres (%.3f%% coverage).",
                    total_count - inside_count,
                    total_count,
                    100.0 * fraction,
                )

        improved = _enhanced_mesh_postprocess(
            verts,
            faces,
            decimate_ratio=float(decimate_ratio) if print_ready else 1.0,
            smoothing_iterations=0,
            preserve_components=solid_envelope,
        )
        if improved is not None and len(improved.get("faces", [])) > 0:
            improved["vertices"] = _project_extruded_planes(
                improved["vertices"],
                bounds,
                extrusion_axis,
                tolerance=float(np.max(spacing)) * 2.5,
            )
            if passive_shapes_present and not analytic_cylinder_iso:
                shapes = _convert_legacy_to_physical_shapes(
                    bounds,
                    solid_boxes=solid_boxes,
                    void_boxes=void_boxes,
                    solid_cylinders=solid_cylinders,
                    void_cylinders=void_cylinders,
                )
                shapes = tuple(shapes) + tuple(
                    _convert_legacy_to_physical_shapes(
                        bounds,
                        solid_cylinders=joint_pin_cylinders,
                    )
                )
                improved["vertices"] = _project_passive_shapes_surfaces(
                    improved["vertices"],
                    shapes,
                    tolerance=float(np.max(spacing)) * 3.0,
                )
            if bounds is not None:
                improved["vertices"] = np.clip(
                    improved["vertices"], bound_min, bound_max
                )
            improved["surface_backend"] = backend_name
            improved["surface_quality_preset"] = quality_key.title()
            improved["manufacturing_structure"] = (
                structure_options.display_name
                if structure_options is not None
                else "Solid Envelope"
            )
            improved["manufacturing_reference_volume"] = (
                manufacturing_reference_volume
            )
            improved["manufacturing_reference_sample_count"] = (
                target_material_count
            )
            improved["analyzed_reference_volume"] = analyzed_reference_volume
            improved["recovery_target_volume"] = recovery_target_volume
            improved["lattice_sizing"] = lattice_sizing
            improved["lattice_connectivity"] = lattice_connectivity
            improved["effective_recovery_cutoff"] = float(cutoff)
            improved["protected_profile_points"] = protected_profile_points
            # Carried on this branch too, not only the fallback below. A cell
            # pitch is requested in analysis voxels but resolved on the
            # supersampled build grid, and this is the factor between them; the
            # study gate judges lattice connectivity with it, so omitting it
            # here left that check reading the analysis grid on every result
            # that reached the normal postprocess path — which is all of them.
            improved["structure_resolution_scale"] = float(
                structure_resolution_scale
            )
            _attach_local_density_report(improved)
            return improved

        recovered_shape = {
            "vertices": np.asarray(verts, dtype=float),
            "faces": np.asarray(faces, dtype=int),
            "surface_backend": backend_name,
            "surface_quality_preset": quality_key.title(),
            "manufacturing_structure": (
                structure_options.display_name
                if structure_options is not None
                else "Solid Envelope"
            ),
            "manufacturing_reference_volume": manufacturing_reference_volume,
            "manufacturing_reference_sample_count": target_material_count,
            "analyzed_reference_volume": analyzed_reference_volume,
            "recovery_target_volume": recovery_target_volume,
            "lattice_sizing": lattice_sizing,
            "lattice_connectivity": lattice_connectivity,
            "effective_recovery_cutoff": float(cutoff),
            "protected_profile_points": protected_profile_points,
            # How far the build grid was supersampled past the analysis grid.
            # A cell pitch is requested in analysis voxels but is *resolved* on
            # this grid, and that is the number that decides whether a periodic
            # surface stays connected.
            "structure_resolution_scale": float(structure_resolution_scale),
        }
        _attach_local_density_report(recovered_shape)
        return recovered_shape
    except Exception:
        logger.exception("Voxel shape recovery failed")
        return None
