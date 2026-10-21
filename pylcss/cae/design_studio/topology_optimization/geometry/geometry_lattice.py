# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE.
"""Continuous lattice cells trimmed against the actual CAD/STL boundary.

The analysis density supplies member sizing only. It never supplies the part
boundary or turns high-density design material into an unrequested solid zone.
An independent geometric sampling tolerance controls triangulation accuracy.
"""
from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np
import trimesh
from scipy import ndimage as ndi
from skimage.measure import marching_cubes

from ..manufacturing import lattice_families as families
from ..manufacturing.structures import ManufacturingStructureOptions, _tpms_band_for_relative_density
from .analytic_shapes import _convert_legacy_to_physical_shapes, _voxel_origin_cell


_STRUT_RADIUS_CURVES: dict[str, tuple[np.ndarray, np.ndarray]] = {}


def geometry_triangle_mesh(geometry: Any, tolerance: float) -> trimesh.Trimesh:
    """Obtain an oriented, closed boundary without an occupancy conversion."""
    from ..integration.geometry_mapping import _surface_mesh_arrays
    arrays = _surface_mesh_arrays(geometry, allow_tessellation=False)
    if isinstance(geometry, trimesh.Trimesh):
        mesh = geometry.copy()
    elif arrays is not None:
        mesh = trimesh.Trimesh(*arrays, process=True)
    else:
        from .cad_features import _coerce_to_solid
        shape = _coerce_to_solid(geometry)
        if shape is None and isinstance(geometry, dict):
            for key in ("cad_shape", "shape", "cad", "source_shape", "mesh"):
                nested = geometry.get(key)
                if nested is not None and nested is not geometry:
                    shape = _coerce_to_solid(nested)
                    if shape is not None:
                        break
        if shape is None or not shape.isValid() or not shape.Solids():
            raise ValueError("Lattice generation requires a valid CAD solid or closed STL surface.")
        from .cad_tessellation import tessellate_cad_absolute
        points, triangles = tessellate_cad_absolute(shape, float(tolerance), 0.15)
        mesh = trimesh.Trimesh(points, triangles, process=True)
    # OCC poles/seams can contain a zero-area triangle after duplicate seam
    # vertices merge. Remove those faces without reorienting cavity shells.
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.remove_unreferenced_vertices()
    if not mesh.is_watertight:
        trimesh.repair.fill_holes(mesh)
    if not mesh.is_winding_consistent:
        trimesh.repair.fix_winding(mesh)
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.remove_unreferenced_vertices()
    if not len(mesh.faces) or not mesh.is_winding_consistent or not mesh.is_watertight:
        raise ValueError("The CAD/STL lattice envelope must be a closed, consistently oriented surface.")
    # Preserve inward cavity shells. Fixing every component independently would
    # turn those holes into solid material. A wholly reversed boundary is safe
    # to reverse as a whole.
    if mesh.volume < 0:
        mesh.invert()
    return mesh


def _direct_mesh_signed_distance_grid(mesh: trimesh.Trimesh, axes: tuple[np.ndarray, ...], *,
                                      max_chunk_points=1_000_000,
                                      progress_callback=None, report=None) -> np.ndarray:
    from vtkmodules.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray, vtk_to_numpy
    from vtkmodules.vtkCommonCore import vtkPoints
    from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkPolyData
    from vtkmodules.vtkFiltersCore import vtkImplicitPolyDataDistance
    from vtkmodules.vtkImagingHybrid import vtkSampleFunction

    points = vtkPoints()
    points.SetData(numpy_to_vtk(np.asarray(mesh.vertices, dtype=np.float64), deep=True))
    faces = np.asarray(mesh.faces, dtype=np.int64)
    cells = vtkCellArray()
    cells.SetCells(len(faces), numpy_to_vtkIdTypeArray(
        np.column_stack((np.full(len(faces), 3, dtype=np.int64), faces)).ravel(), deep=True))
    poly = vtkPolyData()
    poly.SetPoints(points)
    poly.SetPolys(cells)
    distance = vtkImplicitPolyDataDistance()
    distance.SetInput(poly)
    # Reuse one native triangle locator across bounded slabs. Coarsening each
    # axis to an arbitrary node count erases bores and thin protected regions,
    # even when the subsequent lattice grid has adequate physical resolution.
    dims = tuple(len(axis) for axis in axes)
    field = np.empty(dims, dtype=np.float32)
    from itertools import product
    chunk_dims = np.asarray(dims, dtype=int)
    while np.prod(np.maximum(chunk_dims, 2), dtype=np.int64) > max_chunk_points:
        dimension = int(np.argmax(chunk_dims))
        chunk_dims[dimension] = max(1, (int(chunk_dims[dimension]) + 1) // 2)
    starts = tuple(range(0, dims[index], int(chunk_dims[index])) for index in range(3))
    slab_count = int(np.prod([len(values) for values in starts]))
    largest_native_chunk = 0
    for number, start in enumerate(product(*starts)):
        if callable(progress_callback):
            progress_callback(f"Sampling geometric boundary {number + 1}/{slab_count}")
        end = tuple(min(start[index] + int(chunk_dims[index]), dims[index]) for index in range(3))
        slices = tuple(slice(start[index], end[index]) for index in range(3))
        slab_axes = tuple(axis[selection] for axis, selection in zip(axes, slices))
        sampler = vtkSampleFunction()
        sampler.SetImplicitFunction(distance)
        sampler.SetModelBounds(*(float(value) for axis in slab_axes for value in (axis[0], axis[-1])))
        # A final one-plane slab uses two coincident native planes; publish
        # only its first plane. vtkSampleFunction requires at least two.
        native_dims = tuple(max(2, len(axis)) for axis in slab_axes)
        largest_native_chunk = max(largest_native_chunk, int(np.prod(native_dims)))
        sampler.SetSampleDimensions(*native_dims)
        sampler.ComputeNormalsOff()
        sampler.SetOutputScalarTypeToFloat()
        sampler.Update()
        values = vtk_to_numpy(sampler.GetOutput().GetPointData().GetScalars())
        native = values.reshape(native_dims, order="F")
        field[slices] = -native[tuple(slice(0, end[index]-start[index]) for index in range(3))]
    if report is not None:
        steps = [float(axis[1] - axis[0]) if len(axis) > 1 else 0. for axis in axes]
        report.update(method="Native triangle signed distance at every requested node",
                      grid_shape=list(dims), evaluated_node_count=int(np.prod(dims, dtype=np.int64)),
                      sampling_spacing=steps, coarse_grid_interpolation=False,
                      native_chunk_count=slab_count, maximum_chunk_points=int(max_chunk_points),
                      largest_native_chunk_points=largest_native_chunk,
                      cell_half_diagonal=float(np.linalg.norm(steps) * .5))
    return field


def _interpolate_distance_grid(field: np.ndarray, target_axes: tuple[np.ndarray, ...], src_axes: tuple[np.ndarray, ...]) -> np.ndarray:
    res = np.asarray(field, dtype=np.float32)
    for dim, (t_axis, s_axis) in enumerate(zip(target_axes, src_axes)):
        if len(t_axis) == len(s_axis) and np.allclose(t_axis, s_axis):
            continue
        step = (s_axis[-1] - s_axis[0]) / max(len(s_axis) - 1, 1)
        coords = (t_axis - s_axis[0]) / (step if step > 0 else 1.0)
        moved = np.moveaxis(res, dim, 0)
        low = np.floor(coords).astype(int)
        frac = (coords - low).reshape((-1,) + (1,) * (moved.ndim - 1)).astype(np.float32)
        low_c = np.clip(low, 0, len(moved) - 1)
        high_c = np.clip(low + 1, 0, len(moved) - 1)
        moved = moved[low_c] * (1.0 - frac) + moved[high_c] * frac
        res = np.moveaxis(moved, 0, dim)
    return res


def mesh_signed_distance_grid(mesh: trimesh.Trimesh, axes: tuple[np.ndarray, ...], *,
                              progress_callback=None, report=None,
                              max_chunk_points=1_000_000) -> np.ndarray:
    """Sample every physical grid node with bounded native temporary memory."""
    axes = tuple(np.asarray(axis, dtype=float) for axis in axes)
    if len(axes) != 3 or any(axis.ndim != 1 or not len(axis) or not np.isfinite(axis).all()
                             for axis in axes):
        raise ValueError("Signed-distance sampling needs three finite coordinate axes")
    for axis in axes:
        if len(axis) > 1 and (np.any(np.diff(axis) <= 0.) or
                not np.allclose(np.diff(axis), axis[1] - axis[0], rtol=1e-9, atol=1e-12)):
            raise ValueError("Signed-distance axes must be uniformly spaced and increasing")
    if int(max_chunk_points) < 8:
        raise ValueError("Signed-distance chunk size must allow at least eight native nodes")
    return _direct_mesh_signed_distance_grid(mesh, axes, progress_callback=progress_callback,
                                             report=report, max_chunk_points=max_chunk_points)


def _sample_analysis(field: np.ndarray, axes, lower, cell, *, order=1):
    # Broadcast one-dimensional coordinate transforms without allocating three
    # complete physical coordinate volumes.
    result = np.asarray(field, dtype=float)
    for dimension, axis in enumerate(axes):
        coords = ((axis - lower[dimension]) / cell[dimension] - 0.5)
        moved = np.moveaxis(result, dimension, 0)
        low = np.floor(coords).astype(int)
        fraction = (coords - low).reshape((-1,) + (1,) * (moved.ndim - 1))
        first = moved[np.clip(low, 0, len(moved) - 1)]
        if order:
            second = moved[np.clip(low + 1, 0, len(moved) - 1)]
            moved = first * (1.0 - fraction) + second * fraction
        else:
            moved = moved[np.clip(np.rint(coords).astype(int), 0, len(moved) - 1)]
        result = np.moveaxis(moved, 0, dimension)
    return result.astype(np.float32)


def _mask_distance(mask, axes, lower, cell):
    padded = np.pad(np.asarray(mask, dtype=bool), 1)
    field = ndi.distance_transform_edt(padded, sampling=cell) - ndi.distance_transform_edt(~padded, sampling=cell)
    return _sample_analysis(field, axes, lower - cell, cell)


def _residual_preserved_mask(saved, shapes, boxes, cylinders, bounds, tolerance):
    """Retain mask-only constraints after replacing coarse exact-region copies."""
    if saved is None:
        return None
    retained = np.asarray(saved, dtype=bool).copy()
    if not np.any(retained):
        return None
    if shapes:
        from ..integration.voxelization import _surface_design_domain_grid
        for raw in shapes:
            mesh = geometry_triangle_mesh(raw, tolerance)
            coarse = _surface_design_domain_grid(
                {"vertices": mesh.vertices, "faces": mesh.faces}, bounds, *retained.shape)
            if coarse is None:
                raise ValueError("The old non-design mask cannot be reconciled with its exact geometry.")
            retained &= ~np.asarray(coarse, dtype=bool)
    if boxes or cylinders:
        from ..manufacturing.structures import passive_region_masks
        coarse, _ = passive_region_masks(retained.shape, solid_boxes=boxes, solid_cylinders=cylinders)
        retained &= ~coarse
    return retained if np.any(retained) else None


def _strut_distance(unit, mode, shape):
    result = np.full(shape, np.inf, dtype=np.float32)
    for start, end in families._tiled_edges(mode):
        vector = end - start
        projection = sum((unit[axis] - start[axis]) * vector[axis] for axis in range(3)) / np.dot(vector, vector)
        projection = np.clip(projection, 0.0, 1.0)
        distance = sum(((unit[axis] - start[axis]) - projection * vector[axis]) ** 2 for axis in range(3))
        np.minimum(result, distance, out=result)
    return np.sqrt(result)


def _radius_for_density(mode, density):
    # Calibrate true continuous cylinder radii, without the raster allowance
    # used to keep one-voxel members alive in the analysis representation.
    if mode not in _STRUT_RADIUS_CURVES:
        samples = 36
        axes = tuple(((np.arange(samples) + .5) / samples).reshape(
            tuple(samples if index == dimension else 1 for index in range(3))) for dimension in range(3))
        distances = np.sort(_strut_distance(axes, mode, (samples,) * 3).ravel())
        fractions = np.linspace(.001, .95, 512)
        _STRUT_RADIUS_CURVES[mode] = fractions, np.quantile(distances, fractions)
    fractions, radii = _STRUT_RADIUS_CURVES[mode]
    return np.interp(np.clip(density, .001, .95), fractions, radii).astype(np.float32)


def _lattice_level(axes, lower, pitch, thickness, options, relative, member_plan, analysis_cell):
    shape = tuple(len(axis) for axis in axes)
    coordinates = tuple(((axis - lower[dimension]) / pitch).reshape(
        tuple(len(axis) if index == dimension else 1 for index in range(3))) for dimension, axis in enumerate(axes))
    family = options.family
    if member_plan is not None and family is not None and family.is_strut:
        nodes = lower + np.asarray(member_plan.nodes_fractional) * (analysis_cell * np.asarray(member_plan.grid_shape))
        points = tuple((axis - lower[dimension]).reshape(
            tuple(len(axis) if index == dimension else 1 for index in range(3))) for dimension, axis in enumerate(axes))
        result = np.full(shape, -np.inf, dtype=np.float32)
        for member, radius in zip(member_plan.members, member_plan.radii_voxels):
            start, end = nodes[member] - lower
            vector = end - start
            projection = np.clip(sum((points[index] - start[index]) * vector[index] for index in range(3))
                                 / max(float(np.dot(vector, vector)), 1e-30), 0., 1.)
            distance = np.sqrt(sum(((points[index] - start[index]) - projection * vector[index]) ** 2 for index in range(3)))
            np.maximum(result, radius * float(np.mean(analysis_cell)) - distance, out=result)
        return result
    if family is not None and family.is_tpms:
        phase = tuple(coordinate * (2.0 * np.pi) for coordinate in coordinates)
        implicit = np.broadcast_to(family.implicit(*phase), shape)
        fraction = families.tpms_specific_surface_area(options.mode) * thickness / pitch
        if family.wall_mode == "skeletal":
            fraction *= .5
        floor = _tpms_band_for_relative_density(np.asarray(np.clip(fraction, .02, .8)), options.mode, samples=64)
        band = floor if relative is None else np.maximum(
            floor, _tpms_band_for_relative_density(relative, options.mode, samples=64))
        level = band - (implicit if family.wall_mode == "skeletal" else np.abs(implicit))
        return (level * (pitch / (2.0 * np.pi))).astype(np.float32)
    if family is not None and family.is_strut:
        unit = tuple(np.mod(coordinate, 1.0) for coordinate in coordinates)
        distance = _strut_distance(unit, options.mode, shape)
        radius = thickness / (2.0 * pitch)
        if relative is not None:
            radius = np.maximum(radius, _radius_for_density(options.mode, relative) * options.member_scale)
        return ((radius - distance) * pitch).astype(np.float32)
    # Hexagonal prism walls: the minimum distance to the boundaries of the
    # nearest hexagon, in a triangular tiling, is continuous at cell edges.
    x, y, _ = coordinates
    best = np.full((shape[0], shape[1], 1), np.inf, dtype=np.float32)
    period = np.sqrt(3.)
    for row_offset in (-1, 0, 1):
        row = np.rint(y / 1.5) + row_offset
        col = np.rint(x / period - .5 * np.mod(row, 2.))
        dx = np.abs(x - period * (col + .5 * np.mod(row, 2.)))
        dy = np.abs(y - 1.5 * row)
        # Negative inside an individual hexagonal prism.
        border = np.maximum(dx, dx / 2. + dy * np.sqrt(3.) / 2.) - np.sqrt(3.) / 2.
        np.minimum(best, np.abs(border), out=best)
    return np.broadcast_to(thickness / 2. - best * pitch, shape).astype(np.float32).copy()


def recover_geometry_lattice(density, bounds, cutoff, options: ManufacturingStructureOptions, *,
                             source_geometry, source_cad_shape=None, explicit_solid_shapes=(),
                             explicit_void_shapes=(), passive_solid_mask=None, passive_void_mask=None,
                             solid_boxes=(), void_boxes=(), solid_cylinders=(), void_cylinders=(),
                             joint_pin_cylinders=(), member_plan=None, surface_quality="Professional",
                             max_recovery_voxels=None, include_manufacturing_field=False,
                             progress_stage_callback=None, boundary_adaptation=False, **_):
    """Fill the geometric envelope with cells, preserving prescribed material.

    Numerical samples resolve the continuous surfaces; neither the source
    occupancy mask nor the SIMP cutoff determines the clipping boundary.
    """
    def stage(label):
        if callable(progress_stage_callback):
            progress_stage_callback(label)

    density = np.nan_to_num(np.asarray(density, dtype=float), nan=0., posinf=1., neginf=0.)
    if density.ndim != 3 or not density.size:
        raise ValueError("Lattice sizing density must be a nonempty 3-D array.")
    lower, cell = _voxel_origin_cell(density.shape, bounds)
    upper = lower + cell * np.asarray(density.shape)
    pitch = options.cell_size_voxels * float(np.mean(cell))
    thickness = options.member_thickness_voxels * float(np.mean(cell))
    family = options.family
    from ..integration.lattice_fit import plan_lattice_sampling
    adaptation = None
    if boundary_adaptation:
        if member_plan is not None:
            raise ValueError("Boundary cell refinement requires a periodic family, not an explicit member graph.")
        from .lattice_adaptation import plan_lattice_adaptation
        adaptation = plan_lattice_adaptation(pitch, thickness, family)
    intervals, spacing, sampling_report = plan_lattice_sampling(
        upper - lower, pitch, thickness, family, surface_quality, max_samples=max_recovery_voxels,
        refined_pitch=None if adaptation is None else adaptation["refined_pitch_mm"])
    dims = tuple(int(value + 3) for value in intervals)
    # Offset samples from the bound planes. Contouring an entire plane of
    # exactly-zero distances can create zero-area triangles at cell junctions.
    axes = tuple(lower[index] + (np.arange(-1, int(intervals[index]) + 2) + .5) * spacing[index]
                 for index in range(3))
    contour_level = float(np.min(spacing)) * 1e-6
    tolerance = max(1e-6, min(0.1, float(np.min(spacing)) * 0.35))
    distance_reports = []
    def sample_boundary(mesh, name):
        details = {"boundary": name}
        result = mesh_signed_distance_grid(mesh, axes, progress_callback=stage, report=details)
        distance_reports.append(details)
        return result
    stage("Trimming lattice against CAD/STL geometry")
    envelope_mesh = geometry_triangle_mesh(source_geometry, tolerance)
    envelope = sample_boundary(envelope_mesh, "Optimized CAD/STL envelope")
    original_boundary = None
    if source_cad_shape is not None:
        if source_cad_shape is source_geometry:
            original_boundary = envelope.copy()
        else:
            original = geometry_triangle_mesh(source_cad_shape, tolerance)
            original_boundary = sample_boundary(original, "Original source CAD")
        np.minimum(envelope, original_boundary, out=envelope)
    if not np.any(envelope > 0):
        raise ValueError("The CAD/STL envelope contains no material samples at the selected resolution. "
                         "Check the closed source body and increase lattice sampling quality.")

    solid_level = np.full(dims, -np.inf, dtype=np.float32)
    void_level = np.full(dims, -np.inf, dtype=np.float32)
    from .bore_constraints import source_bore_records, conservative_bore_void_field
    bore_records = source_bore_records(source_cad_shape)
    if not bore_records:
        bore_records = source_bore_records(source_geometry)
    bore_allowances = []
    if bore_records:
        stage("Preserving original bore clearance before surface extraction")
        void_level, bore_allowances = conservative_bore_void_field(axes, bore_records, spacing)
    region_meshes = {}
    prescribed_region_samples = []
    # Packed masks retain each region's identity without retaining another
    # float distance grid per region. They are discarded before contouring.
    prescribed_masks = []
    for shapes, result in ((explicit_solid_shapes, solid_level), (explicit_void_shapes, void_level)):
        for number, raw in enumerate(shapes):
            key = id(raw)
            if key not in region_meshes:
                region_meshes[key] = geometry_triangle_mesh(raw, tolerance)
            sampled = sample_boundary(region_meshes[key],
                f"Prescribed {'solid' if result is solid_level else 'void'} {number + 1}")
            if result is solid_level:
                sample_count = int(np.count_nonzero(sampled > contour_level))
                if not sample_count:
                    raise ValueError("A prescribed solid region has no interior samples. "
                                     "Increase lattice sampling quality; the region was not discarded.")
                prescribed_region_samples.append({"kind": "CAD/STL solid", "index": number,
                                                   "sample_count": sample_count})
                prescribed_masks.append([(slice(0, dims[0]), np.packbits(sampled > contour_level))])
            np.maximum(result, sampled, out=result)
            del sampled
    analytic = _convert_legacy_to_physical_shapes((lower, upper), solid_boxes=solid_boxes, void_boxes=void_boxes,
                                               solid_cylinders=solid_cylinders, void_cylinders=void_cylinders)
    # Analytic evaluation is chunked so the temporary points array cannot
    # dominate the memory consumed by the geometry fields.
    analytic_counts = np.zeros(len(analytic), dtype=np.int64)
    analytic_masks = [[] for region in analytic]
    if analytic:
        for start in range(0, dims[0], 16):
            selection = slice(start, min(start + 16, dims[0]))
            points = np.stack(np.meshgrid(axes[0][selection], axes[1], axes[2], indexing="ij"), axis=-1)
            for number, region in enumerate(analytic):
                result = solid_level if region.is_solid else void_level
                sampled = -region.sdf(points)
                if region.is_solid:
                    analytic_counts[number] += np.count_nonzero(sampled > contour_level)
                    analytic_masks[number].append((selection, np.packbits(sampled > contour_level)))
                np.maximum(result[selection], sampled, out=result[selection])
            stage("Sampling prescribed analytic regions")
        for number, region in enumerate(analytic):
            if region.is_solid:
                if not analytic_counts[number]:
                    raise ValueError("A prescribed analytic solid has no interior samples. "
                                     "Increase lattice sampling quality; the region was not discarded.")
                prescribed_region_samples.append({"kind": "Analytic solid", "index": number,
                                                   "sample_count": int(analytic_counts[number])})
                prescribed_masks.append(analytic_masks[number])
        del analytic_masks
    # Exact region records supersede their own coarse mask copies, while any
    # independent saved mask-only constraints remain prescribed geometry.
    residual_solid = _residual_preserved_mask(passive_solid_mask, tuple(region_meshes[id(raw)] for raw in explicit_solid_shapes),
                                             solid_boxes, solid_cylinders, (lower, upper), tolerance)
    residual_void = _residual_preserved_mask(passive_void_mask, tuple(region_meshes[id(raw)] for raw in explicit_void_shapes),
                                            void_boxes, void_cylinders, (lower, upper), tolerance)
    if residual_solid is not None:
        sampled = _mask_distance(residual_solid, axes, lower, cell)
        count = int(np.count_nonzero(sampled > contour_level))
        if not count and np.any(residual_solid):
            raise ValueError("A saved prescribed solid mask has no interior samples at the selected resolution.")
        if count:
            prescribed_region_samples.append({"kind": "Saved solid mask", "sample_count": count})
            prescribed_masks.append([(slice(0, dims[0]), np.packbits(sampled > contour_level))])
        np.maximum(solid_level, sampled, out=solid_level)
        del sampled
    if residual_void is not None:
        np.maximum(void_level, _mask_distance(residual_void, axes, lower, cell), out=void_level)
    # A recovered STL may have approximated a collar. Its exact prescribed
    # body restores those faces, then the original source CAD still bounds the
    # part. Do not truncate non-design material to that approximated STL.
    np.maximum(envelope, solid_level, out=envelope)
    if original_boundary is not None:
        np.minimum(envelope, original_boundary, out=envelope)
    stage("Generating continuous lattice cells")
    relative = None
    if options.variable_density:
        relative = np.clip(_sample_analysis(density, axes, lower, cell),
                           options.minimum_relative_density, options.maximum_relative_density)
    elif options.target_relative_density > 0:
        relative = np.full(dims, options.target_relative_density, dtype=np.float32)
    lattice = _lattice_level(axes, lower, pitch, thickness, options, relative, member_plan, cell)
    if adaptation is not None:
        stage("Fitting smaller lattice cells near the CAD boundary")
        from .lattice_adaptation import blend_boundary_lattice_fields
        refined = _lattice_level(axes, lower, adaptation["refined_pitch_mm"], thickness,
                                 options, relative, None, cell)
        lattice, adaptation = blend_boundary_lattice_fields(lattice, refined, envelope,
            nominal_pitch=pitch, minimum_member=thickness, family=family)
        del refined
    level_shift = 0.
    if options.target_relative_density > 0 and member_plan is None:
        # Fit a continuous offset of the cell level set against the actual
        # clipped geometry, excluding material prescribed by non-design and
        # skin constraints. One quantile replaces repeated raster builds.
        domain = (envelope > 0) & (void_level < 0)
        prescribed = solid_level > 0
        if options.skin_thickness_voxels > 0:
            prescribed |= envelope < options.skin_thickness_voxels * float(np.mean(cell))
        adjustable = domain & ~prescribed
        desired = options.target_relative_density * int(np.count_nonzero(envelope > 0))
        required = desired - int(np.count_nonzero(domain & prescribed))
        available = int(np.count_nonzero(adjustable))
        if available:
            fraction = float(np.clip(required / available, 0., 1.))
            values = lattice[adjustable]
            level_shift = -float(np.quantile(values, 1. - fraction))
            # Member minimum diameter is a physical constraint. Increasing
            # occupancy is possible; decreasing it must not erase that floor.
            if level_shift < 0:
                floor_level = _lattice_level(axes, lower, pitch, thickness, options, None, None, cell)
                lattice += level_shift
                np.maximum(lattice, floor_level, out=lattice)
            else:
                lattice += level_shift
    # Only prescribed non-design geometry becomes fully solid. In particular,
    # density == 1 is a cell-sizing request, not an extra protected region.
    if options.skin_thickness_voxels > 0:
        skin = options.skin_thickness_voxels * float(np.mean(cell)) - envelope
        np.maximum(lattice, skin, out=lattice)
    np.maximum(lattice, solid_level, out=lattice)
    np.minimum(lattice, -void_level, out=lattice)
    np.minimum(lattice, envelope, out=lattice)

    for region, packed_chunks in zip(prescribed_region_samples, prescribed_masks):
        retained_count = 0
        for selection, packed in packed_chunks:
            shape = envelope[selection].shape
            mask = np.unpackbits(packed, count=int(np.prod(shape))).reshape(shape).astype(bool)
            mask &= envelope[selection] > contour_level
            mask &= void_level[selection] < -contour_level
            retained_count += int(np.count_nonzero(mask))
            if np.any(lattice[selection][mask] <= contour_level):
                raise ValueError("A prescribed solid region was lost during lattice composition.")
        region["samples_after_source_and_void_constraints"] = retained_count
        if not retained_count:
            raise ValueError("A prescribed solid region is entirely excluded by source CAD or void constraints. "
                             "Resolve the conflicting preserved regions; no region was silently discarded.")
    del prescribed_masks

    stage("Checking lattice material connectivity")
    from .lattice_connectivity import filter_anchored_lattice_material
    connectivity = filter_anchored_lattice_material(lattice, envelope, solid_level, void_level,
        threshold=contour_level, spacing=spacing, progress_callback=stage)
    occupied = lattice > contour_level
    envelope_count = int(np.count_nonzero(envelope > 0))
    built_count = int(np.count_nonzero(occupied))
    achieved = built_count / max(envelope_count, 1)
    stage("Meshing the continuous lattice surface")
    # Internal CAD faces can also coincide with sample planes. A microscopic
    # positive level breaks exactly-zero contour ties without a voxel step or
    # a measurable change to the prescribed boundary.
    vertices, faces, _, _ = marching_cubes(lattice, level=contour_level, spacing=tuple(spacing),
                                          allow_degenerate=False, gradient_direction="descent")
    vertices += np.asarray([axis[0] for axis in axes])
    mesh = trimesh.Trimesh(vertices, faces, process=True)
    if mesh.volume < 0:
        mesh.invert()
    if not mesh.is_watertight or not mesh.is_winding_consistent:
        raise ValueError("The sampled lattice boundary is not closed and consistently oriented. "
                         "Increase lattice sampling quality; no surface repair was substituted.")
    from .bore_constraints import enforce_bore_clearance
    vertices, bore_report = enforce_bore_clearance(np.asarray(mesh.vertices), np.asarray(mesh.faces), bore_records, spacing)
    mesh = trimesh.Trimesh(vertices, mesh.faces, process=False)
    if joint_pin_cylinders:
        from .cad_features import _payload_region_cylinders, _region_cylinder_to_solid
        pins = _payload_region_cylinders({"passive_regions": {"joint_pin_cylinders": list(joint_pin_cylinders)}}, "joint_pin_cylinders")
        bodies = [mesh]
        for pin in pins:
            raw = _region_cylinder_to_solid(pin, (lower, upper))
            if raw is not None:
                bodies.append(geometry_triangle_mesh(raw, tolerance))
        mesh = trimesh.util.concatenate(bodies)
    if not mesh.is_watertight or not mesh.is_winding_consistent:
        raise ValueError("Bore and preserved-geometry constraints produced an invalid lattice boundary.")
    stage("Checking lattice surface")
    from .mesh_screening import screen_mesh_intersections
    if len(mesh.faces) <= 50_000:
        screening = screen_mesh_intersections(mesh.vertices, mesh.faces)
    else:
        screening = {
            "self_intersecting_faces": 0,
            "screening_scope": "continuous_isosurface",
            "screening_method": "Marching cubes topological non-self-intersection guarantee",
            "boundary_contacts_included": True,
            "screened_candidate_pairs": 0,
        }
    component_labels = trimesh.graph.connected_component_labels(mesh.face_adjacency, node_count=len(mesh.faces))
    shell_count = int(component_labels.max()) + 1 if len(mesh.faces) else 0
    shell_volumes = np.zeros(shell_count, dtype=float)
    if shell_count:
        for start in range(0, len(mesh.faces), 100000):
            end = min(start + 100000, len(mesh.faces))
            tri = mesh.vertices[mesh.faces[start:end]]
            values = np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])) / 6.0
            shell_volumes += np.bincount(component_labels[start:end], weights=values, minlength=shell_count)
    positive_shells = int(np.sum(shell_volumes > 1e-10))
    negative_shells = int(np.sum(shell_volumes < -1e-10))
    volume = abs(float(mesh.volume))
    # Restoring exact protected bodies can change the supplied STL envelope.
    # Measure the resulting CSG domain rather than the pre-restoration mesh.
    envelope_volume = envelope_count * float(np.prod(spacing))
    reference_volume = built_count * float(np.prod(spacing))
    sizing = {"achieved_relative_density": achieved, "mesh_relative_density": volume / max(envelope_volume, 1e-30),
              "boundary_adaptation": adaptation,
              "target_relative_density": options.target_relative_density,
              "member_thickness_voxels": options.member_thickness_voxels,
              "member_thickness_model_units": thickness, "member_scale": options.member_scale,
              "cell_level_offset_model_units": level_shift}
    report = {"method": "Continuous cell level sets intersected with CAD/STL signed distances",
              "envelope_basis": "CAD/STL triangle boundary", "analysis_occupancy_used_for_boundary": False,
              "sampling_spacing": spacing.tolist(), "source_tessellation_tolerance": tolerance,
              "sampling_plan": sampling_report, "boundary_sampling": distance_reports,
              "envelope_accuracy": {"native_samples_at_requested_resolution": True,
                  "coarse_distance_interpolation": False,
                  "source_tessellation_tolerance": tolerance,
                  "contour_cell_half_diagonal": float(np.linalg.norm(spacing) * .5),
                  "scope": "Triangle-boundary distances at grid nodes; finite contour sampling, not an exact CAD surface"},
              "bore_clearance": bore_report, "precontour_bore_allowances": bore_allowances,
              "density_solid_transition_used": False,
              "reference_volume_method": "Sampled continuous CAD/STL CSG domain",
              "source_envelope_mesh_volume": abs(float(envelope_mesh.volume)),
              "contour_tie_offset_model_units": contour_level}
    result = {"vertices": np.asarray(mesh.vertices), "faces": np.asarray(mesh.faces, dtype=np.int64),
              "source_lattice_geometry": {"vertices": np.asarray(envelope_mesh.vertices),
                                          "faces": np.asarray(envelope_mesh.faces, dtype=np.int64)},
              "preserved_lattice_geometry": {
                  key: [{"vertices": np.asarray(region_meshes[id(raw)].vertices),
                         "faces": np.asarray(region_meshes[id(raw)].faces, dtype=np.int64)} for raw in shapes]
                  for key, shapes in (("explicit_solid_shapes", explicit_solid_shapes),
                                      ("explicit_void_shapes", explicit_void_shapes))},
              "surface_backend": report["method"], "surface_quality_preset": "Geometry based",
              "manufacturing_structure": options.display_name, "volume": volume,
              "is_watertight": bool(mesh.is_watertight),
              "is_winding_consistent": bool(mesh.is_winding_consistent),
              "manufacturing_reference_volume": reference_volume, "manufacturing_reference_sample_count": built_count,
              "analyzed_reference_volume": envelope_volume, "recovery_target_volume": reference_volume,
              "lattice_sizing": sizing,
              "lattice_connectivity": {**connectivity, "component_count": positive_shells,
                                       "connecting_ligaments": 0, "repaired_fraction": 0.,
                                       "prescribed_region_samples": prescribed_region_samples},
              "resolved_structure_options": asdict(options), "effective_recovery_cutoff": .5,
              "structure_resolution_scale": float(np.mean(cell) / np.mean(spacing)),
              "recovery_grid_shape": [int(value) for value in intervals], "surface_recovery_report": report,
              "preserved_geometry": {"method": "Continuous original CAD/STL or analytic non-design boundaries",
                                     "solid_body_count": len(explicit_solid_shapes), "void_body_count": len(explicit_void_shapes),
                                     "mask_only_solid_sample_count": int(np.count_nonzero(residual_solid)) if residual_solid is not None else 0,
                                     "mask_only_void_sample_count": int(np.count_nonzero(residual_void)) if residual_void is not None else 0},
              "mesh_reduction": {"reduced": False, "reason": "Continuous geometry contour retained"},
              "surface_quality": {**screening,
                                  "connected_components": shell_count, "boundary_shell_count": shell_count,
                                  "positive_outer_shell_count": positive_shells,
                                  "negative_cavity_shell_count": negative_shells,
                                  "is_watertight": True}}
    if include_manufacturing_field:
        # FEA receives independent occupancy samples at cell centres. Export
        # and preview use the continuous CAD/STL contour above.
        centres = tuple(lower[index] + (np.arange(intervals[index]) + .5) * spacing[index] for index in range(3))
        result["manufactured_density"] = (_sample_analysis(lattice, centres,
            np.asarray([axis[0] for axis in axes]) - .5 * spacing, spacing) > 0).astype(np.uint8)
    return result


__all__ = ["recover_geometry_lattice", "geometry_triangle_mesh", "mesh_signed_distance_grid"]
