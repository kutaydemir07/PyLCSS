# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE.
"""SimulationRenderingMixin implementation for the CAD viewer."""

from __future__ import annotations

import logging

import numpy as np
import vtk

from .boundary_visuals import (
    BC_PALETTE,
    compact_constraint_label,
    constraint_color,
    overlay_scale,
)

__all__ = ["SimulationRenderingMixin"]

logger = logging.getLogger(__name__)

_NODAL_STRESS_DISPLAY_PERCENTILE = 99.5


def _stress_display_range(values, stress_location=""):
    """Return a useful contour range without hiding the reported raw peak.

    CalculiX solid stresses are extrapolated from integration points to nodes.
    At idealized fixed or bonded free edges, one or two extrapolated nodal
    values can be orders of magnitude above the surrounding structural field.
    Using that mathematical singularity as the mapper maximum collapses the
    useful contour into one blue band.  Clip only the *display* range for this
    explicitly identified result type; result tables and extrema still retain
    the unmodified peak.
    """
    finite = np.asarray(values, dtype=float).reshape(-1)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 0.0, 1.0, False

    minimum = float(np.min(finite))
    maximum = float(np.max(finite))
    if (
        str(stress_location or "").strip().lower() == "nodal_extrapolated"
        and finite.size >= 200
    ):
        robust_maximum = float(
            np.percentile(finite, _NODAL_STRESS_DISPLAY_PERCENTILE)
        )
        if (
            robust_maximum > minimum
            and maximum > 3.0 * robust_maximum
        ):
            return minimum, robust_maximum, True
    return minimum, maximum, False


def _stress_scalar_bar_title(display_clipped):
    if display_clipped:
        return (
            "Von Mises Stress\n"
            f"(MPa; display P{_NODAL_STRESS_DISPLAY_PERCENTILE:g})"
        )
    return "Von Mises Stress (MPa)"


def build_surface_polydata(vertices, faces):
    """Build a vtkPolyData from vertex/face arrays, or None if unusable.

    Two reasons this exists rather than the obvious per-element loop.

    Speed: recovered topology surfaces are large — a Professional-quality
    recovery on a full-size grid returns 600k-750k triangles, and inserting
    those one vtkTriangle at a time took ~9 s on the main thread, which reads
    as the application hanging right when a solve finishes.

    Safety: VTK indexes the point array through these ids without bounds
    checking, so a single out-of-range or negative index is an out-of-bounds
    read at render time — a hard crash with no Python traceback, which is
    exactly the failure mode where the log simply stops. Non-finite
    coordinates are equally unsafe: they propagate into bounds computation and
    leave the camera and locators in an undefined state. Both are validated
    here instead of being trusted.
    """
    from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray

    try:
        verts = np.asarray(vertices, dtype=np.float64)
        cells = np.asarray(faces)
    except (TypeError, ValueError):
        return None
    if verts.ndim != 2 or verts.shape[1] < 3 or len(verts) == 0:
        return None
    verts = np.ascontiguousarray(verts[:, :3])
    if not np.all(np.isfinite(verts)):
        logger.warning(
            "Surface has %d non-finite vertex coordinates; not rendering.",
            int(np.count_nonzero(~np.isfinite(verts))),
        )
        return None
    if cells.ndim != 2 or cells.shape[1] < 3 or len(cells) == 0:
        return None
    try:
        cells = np.ascontiguousarray(cells.astype(np.int64, casting="unsafe"))
    except (TypeError, ValueError):
        return None

    valid = np.all((cells >= 0) & (cells < len(verts)), axis=1)
    if not np.all(valid):
        dropped = int(np.count_nonzero(~valid))
        logger.warning(
            "Dropping %d of %d faces with out-of-range vertex indices.",
            dropped,
            len(cells),
        )
        cells = cells[valid]
        if len(cells) == 0:
            return None

    points = vtk.vtkPoints()
    points.SetData(numpy_to_vtk(verts, deep=True))

    width = int(cells.shape[1])
    packed = np.empty((len(cells), width + 1), dtype=np.int64)
    packed[:, 0] = width
    packed[:, 1:] = cells
    array = vtk.vtkCellArray()
    array.SetCells(len(cells), numpy_to_vtkIdTypeArray(packed.ravel(), deep=True))

    poly_data = vtk.vtkPolyData()
    poly_data.SetPoints(points)
    poly_data.SetPolys(array)
    return poly_data


def vtk_points_from_array(coordinates):
    """vtkPoints from a (3, n) coordinate array, without a Python loop."""
    from vtk.util.numpy_support import numpy_to_vtk

    array = np.ascontiguousarray(
        np.asarray(coordinates, dtype=np.float64).T[:, :3]
    )
    points = vtk.vtkPoints()
    points.SetData(numpy_to_vtk(array, deep=True))
    return points


def vtk_scalar_array(values, name):
    """Named vtkFloatArray from a 1-D sequence, without a Python loop."""
    from vtk.util.numpy_support import numpy_to_vtk

    array = np.ascontiguousarray(
        np.asarray(values, dtype=np.float32).reshape(-1)
    )
    scalars = numpy_to_vtk(array, deep=True, array_type=vtk.VTK_FLOAT)
    scalars.SetName(str(name))
    return scalars


def _packed_cell_array(connectivity):
    """vtkCellArray from an (n_cells, nodes_per_cell) index array."""
    from vtk.util.numpy_support import numpy_to_vtkIdTypeArray

    cells = np.ascontiguousarray(np.asarray(connectivity, dtype=np.int64))
    width = int(cells.shape[1])
    packed = np.empty((len(cells), width + 1), dtype=np.int64)
    packed[:, 0] = width
    packed[:, 1:] = cells
    array = vtk.vtkCellArray()
    array.SetCells(len(cells), numpy_to_vtkIdTypeArray(packed.ravel(), deep=True))
    return array


def set_uniform_cells(grid, connectivity, cell_type, n_points, cache_owner=None):
    """Populate `grid` with one cell type, vectorized and bounds-checked.

    The per-element alternative — allocating a vtkCell and calling SetId once
    per node — cost 8.0 s for a 150k-element tet10 mesh, on the main thread,
    and `render_simulation` rebuilds the grid on every field switch. That is
    the result viewer freezing when a field is selected.

    ``cache_owner`` may be an analysis mesh whose connectivity is fixed for the
    life of the object. Crash playback re-renders the same mesh once per frame
    with only coordinates and the result field changing, and packing the cell
    array is 88% of that work (34 ms of 39 ms at 150k tet10) — pure waste,
    since connectivity is identical every frame. Caching it on the mesh takes
    the ceiling from roughly 25 fps to over 200. The cached array is never
    mutated, so sharing it between grids is safe under VTK's reference
    counting.

    Indices are range-checked for the same reason as in
    `build_surface_polydata`: VTK dereferences them without bounds checking, so
    a stale or malformed connectivity array is an out-of-bounds read at render
    time rather than an exception.
    """
    cells = np.asarray(connectivity, dtype=np.int64)
    if cells.ndim != 2 or len(cells) == 0:
        return False

    key = (int(cell_type), tuple(cells.shape), int(n_points))
    if cache_owner is not None:
        cached = getattr(cache_owner, "_pylcss_vtk_cell_cache", None)
        if cached is not None and cached[0] == key:
            grid.SetCells(int(cell_type), cached[1])
            return True

    valid = np.all((cells >= 0) & (cells < int(n_points)), axis=1)
    if not np.all(valid):
        dropped = int(np.count_nonzero(~valid))
        logger.warning(
            "Dropping %d of %d cells with out-of-range node indices.",
            dropped,
            len(cells),
        )
        cells = cells[valid]
        if len(cells) == 0:
            return False
    array = _packed_cell_array(cells)
    grid.SetCells(int(cell_type), array)
    if cache_owner is not None and len(cells) == key[1][0]:
        # Only cache when nothing was dropped, so a mesh that failed validation
        # never has its rejected cells silently reinstated on a later frame.
        try:
            cache_owner._pylcss_vtk_cell_cache = (key, array)
        except AttributeError:
            pass
    return True


def set_mixed_cells(grid, blocks, n_points):
    """Populate `grid` from [(vtk_cell_type, connectivity), ...] in one pass."""
    from vtk.util.numpy_support import numpy_to_vtk

    kept: list[tuple[int, np.ndarray]] = []
    for cell_type, connectivity in blocks:
        cells = np.asarray(connectivity, dtype=np.int64)
        if cells.ndim != 2 or len(cells) == 0:
            continue
        valid = np.all((cells >= 0) & (cells < int(n_points)), axis=1)
        if not np.all(valid):
            logger.warning(
                "Dropping %d of %d cells with out-of-range node indices.",
                int(np.count_nonzero(~valid)),
                len(cells),
            )
            cells = cells[valid]
        if len(cells):
            kept.append((int(cell_type), cells))
    if not kept:
        return False

    total = sum(len(cells) for _, cells in kept)
    offsets = np.empty(total + 1, dtype=np.int64)
    connectivity = np.empty(
        sum(cells.size for _, cells in kept), dtype=np.int64
    )
    types = np.empty(total, dtype=np.uint8)
    cell_index = 0
    value_index = 0
    offsets[0] = 0
    for cell_type, cells in kept:
        width = int(cells.shape[1])
        count = len(cells)
        connectivity[value_index : value_index + cells.size] = cells.ravel()
        offsets[cell_index + 1 : cell_index + count + 1] = (
            value_index + width * np.arange(1, count + 1, dtype=np.int64)
        )
        types[cell_index : cell_index + count] = cell_type
        cell_index += count
        value_index += cells.size

    from vtk.util.numpy_support import numpy_to_vtkIdTypeArray

    array = vtk.vtkCellArray()
    array.SetData(
        numpy_to_vtkIdTypeArray(offsets, deep=True),
        numpy_to_vtkIdTypeArray(connectivity, deep=True),
    )
    type_array = numpy_to_vtk(types, deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
    grid.SetCells(type_array, array)
    return True


class SimulationRenderingMixin:
    def update_simulation_field(self, mesh, values, field_name="Density"):
        """Update scalar field on existing mesh for real-time visualization."""
        data = {"mesh": mesh, "visualization_mode": field_name}
        if field_name == "Density":
            data["density"] = values
        elif field_name == "Von Mises Stress":
            data["stress"] = values

        self.render_simulation(data)

    def _topopt_grid_geometry(self, data, density):
        """Return physical origin and spacing for a structured TopOpt field."""
        nelx, nely, nelz = density.shape
        bounds = data.get("bounds") if isinstance(data.get("bounds"), dict) else None
        if bounds and "min" in bounds and "max" in bounds:
            mins = np.asarray(bounds["min"], dtype=float)
            maxs = np.asarray(bounds["max"], dtype=float)
            if mins.size >= 3 and maxs.size >= 3 and np.all(maxs[:3] > mins[:3]):
                spacing = (maxs[:3] - mins[:3]) / np.array(
                    [nelx, nely, nelz], dtype=float
                )
                return mins[:3], spacing
        spacing = np.ones(3, dtype=float)
        origin = np.array([-0.5 * nelx, -0.5 * nely, -0.5 * nelz], dtype=float)
        return origin, spacing

    def _render_smooth_topopt_density(self, data, density):
        """Render the continuous density boundary used by the optimizer.

        The former default used a cube glyph for every retained voxel. That is
        useful for diagnosing the structured analysis grid, but it necessarily
        makes a valid result look terraced and disconnected. Flying Edges
        interpolates the physical density field, and windowed-sinc fairing
        removes display-scale grid noise without changing mesh connectivity.
        """
        from vtk.util.numpy_support import numpy_to_vtk

        origin, spacing = self._topopt_grid_geometry(data, density)
        cutoff = float(np.clip(data.get("density_cutoff", 0.30), 0.01, 0.95))
        values = np.asarray(density, dtype=np.float32)
        # Prefer the filtered field the recovered shape was built from. The
        # projected density is nearly binary once beta continuation finishes,
        # so contouring it here drew voxel terraces and disagreed with the
        # recovered/exported geometry from the same solve.
        recovery_values = data.get("recovery_density")
        if recovery_values is not None:
            candidate = np.asarray(recovery_values, dtype=np.float32)
            if candidate.shape == values.shape:
                values = candidate
                cutoff = float(
                    np.clip(data.get("recovery_cutoff", cutoff), 0.01, 0.99)
                )
        # A diverged solve can leave NaN/inf in the field. Flying Edges does not
        # reject them: it emitted a surface with more points than polygons for a
        # partly-NaN field, i.e. loose fragments the user reads as a broken
        # result. Clamp to the physical 0-1 range so the contour is meaningful.
        if not np.all(np.isfinite(values)):
            logger.warning(
                "Density field has %d non-finite values; clamping for display.",
                int(np.count_nonzero(~np.isfinite(values))),
            )
            values = np.nan_to_num(values, nan=0.0, posinf=1.0, neginf=0.0)
        values = np.clip(values, 0.0, 1.0)
        if not np.isfinite(cutoff):
            cutoff = 0.45
        # Voxel values live at cell centres. Add an explicit void halo so a
        # design domain touching the analysis-grid boundary produces a closed
        # exterior surface. Without it, Flying Edges only drew the internal
        # transitions and the pre-solve domain looked like a thin border or an
        # open cover rather than a solid design space.
        values = np.pad(
            values,
            pad_width=1,
            mode="constant",
            constant_values=0.0,
        )
        if not (float(np.min(values)) < cutoff < float(np.max(values))):
            positive = values[values > 1e-3]
            if positive.size == 0:
                return False
            cutoff = float(np.clip(np.percentile(positive, 35.0), 0.01, 0.95))

        # Densities are defined at voxel centres, hence the half-cell origin.
        image = vtk.vtkImageData()
        image.SetDimensions(*(int(value) for value in values.shape))
        image.SetSpacing(*(float(value) for value in spacing))
        image.SetOrigin(
            *(float(value) for value in (origin - 0.5 * np.asarray(spacing)))
        )
        scalars = numpy_to_vtk(
            np.ascontiguousarray(values.ravel(order="F")),
            deep=True,
            array_type=vtk.VTK_FLOAT,
        )
        scalars.SetName("Physical density")
        image.GetPointData().SetScalars(scalars)

        contour = vtk.vtkFlyingEdges3D()
        contour.SetInputData(image)
        contour.SetValue(0, cutoff)
        contour.ComputeNormalsOff()
        contour.ComputeGradientsOff()

        triangles = vtk.vtkTriangleFilter()
        triangles.SetInputConnection(contour.GetOutputPort())
        triangles.PassLinesOff()
        triangles.PassVertsOff()

        clean = vtk.vtkCleanPolyData()
        clean.SetInputConnection(triangles.GetOutputPort())
        clean.ToleranceIsAbsoluteOff()
        clean.SetTolerance(1e-8)

        fair = vtk.vtkWindowedSincPolyDataFilter()
        fair.SetInputConnection(clean.GetOutputPort())
        fair.SetNumberOfIterations(12 if data.get("_preview") else 24)
        fair.SetPassBand(0.08 if data.get("_preview") else 0.055)
        fair.NormalizeCoordinatesOn()
        fair.BoundarySmoothingOff()
        fair.FeatureEdgeSmoothingOff()
        fair.NonManifoldSmoothingOff()

        normals = vtk.vtkPolyDataNormals()
        normals.SetInputConnection(fair.GetOutputPort())
        normals.ConsistencyOn()
        normals.AutoOrientNormalsOn()
        normals.SplittingOff()
        normals.Update()
        poly_data = normals.GetOutput()
        if (
            poly_data is None
            or poly_data.GetNumberOfPoints() == 0
            or poly_data.GetNumberOfPolys() == 0
        ):
            return False

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(poly_data)
        mapper.ScalarVisibilityOff()

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(0.70, 0.76, 0.84)
        actor.GetProperty().SetOpacity(1.0)
        actor.GetProperty().SetInterpolationToPhong()
        actor.GetProperty().SetEdgeVisibility(1 if self._show_edges else 0)
        actor.GetProperty().SetEdgeColor(0.10, 0.13, 0.18)
        try:
            actor.GetProperty().SetMetallic(0.18)
            actor.GetProperty().SetRoughness(0.36)
        except AttributeError:
            pass

        self.renderer.AddActor(actor)
        self.current_actor = actor
        self.actors.append(actor)
        if self._section_enabled:
            self._update_section_cut()
        self._pickable_surface_dataset = vtk.vtkPolyData()
        self._pickable_surface_dataset.DeepCopy(poly_data)
        self._render_topopt_joint_overlays(data)
        self._update_scalar_bar("", 0.0, 1.0, None)
        self.renderer.ResetCamera()
        self.vtkWidget.GetRenderWindow().Render()
        return True

    def _render_density_field(self, data, density):
        """Colour the analysis grid by density, with a legend.

        "Analysis Density" previously drew the same solid-grey isosurface as
        the recovered shape, with scalar visibility off and the scalar bar
        hidden, so it showed a *shape* and never the field it is named after.
        Intermediate density is the main thing to inspect during a study — it
        is how you see whether the projection has converged or the result is
        still grey — so it needs the element colouring and legend that every
        commercial density plot has.

        The projected field is deliberately used here rather than the filtered
        one: this view answers "what did the optimizer actually decide", which
        is a question about the physical densities that drove the FE solve.
        """
        from vtk.util.numpy_support import numpy_to_vtk

        origin, spacing = self._topopt_grid_geometry(data, density)
        values = np.asarray(density, dtype=np.float32)
        if not np.all(np.isfinite(values)):
            values = np.nan_to_num(values, nan=0.0, posinf=1.0, neginf=0.0)
        values = np.clip(values, 0.0, 1.0)
        if values.size == 0:
            return False

        image = vtk.vtkImageData()
        # Cell-centred data: the point dimensions are one larger per axis.
        image.SetDimensions(*(int(n) + 1 for n in values.shape))
        image.SetSpacing(*(float(v) for v in spacing))
        image.SetOrigin(*(float(v) for v in origin))
        scalars = numpy_to_vtk(
            np.ascontiguousarray(values.ravel(order="F")),
            deep=True,
            array_type=vtk.VTK_FLOAT,
        )
        scalars.SetName("Density")
        image.GetCellData().SetScalars(scalars)

        cutoff = float(np.clip(data.get("density_cutoff", 0.30), 0.01, 0.95))
        # Hide voxels below cutoff threshold so void space is empty and carved out
        threshold = vtk.vtkThreshold()
        threshold.SetInputData(image)
        threshold.SetInputArrayToProcess(
            0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_CELLS, "Density"
        )
        threshold.SetLowerThreshold(cutoff)
        threshold.SetUpperThreshold(1.1)
        threshold.SetThresholdFunction(
            getattr(vtk.vtkThreshold, "THRESHOLD_BETWEEN", 0)
        )
        threshold.Update()
        if threshold.GetOutput().GetNumberOfCells() == 0:
            return False

        lut = vtk.vtkLookupTable()
        # Blue (void-ish) through to red (solid), the convention every
        # commercial density plot uses.
        lut.SetHueRange(0.667, 0.0)
        lut.SetSaturationRange(0.85, 0.95)
        lut.SetValueRange(0.80, 0.95)
        lut.SetTableRange(0.0, 1.0)
        lut.Build()

        mapper = vtk.vtkDataSetMapper()
        mapper.SetInputConnection(threshold.GetOutputPort())
        mapper.SetScalarModeToUseCellData()
        mapper.SelectColorArray("Density")
        mapper.SetScalarRange(0.0, 1.0)
        mapper.SetLookupTable(lut)
        mapper.ScalarVisibilityOn()

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetEdgeVisibility(1 if self._show_edges else 0)
        actor.GetProperty().SetEdgeColor(0.10, 0.13, 0.18)

        self.renderer.AddActor(actor)
        self.current_actor = actor
        self.actors.append(actor)
        if self._section_enabled:
            self._update_section_cut()
        self._render_topopt_joint_overlays(data)
        self._update_scalar_bar("Density", 0.0, 1.0, lut)
        self.renderer.ResetCamera()
        self.vtkWidget.GetRenderWindow().Render()
        return True

    def _render_voxel_topopt(self, data):
        """Render a TopOpt result as a density field, boundary, or debug voxels."""
        density = np.asarray(data.get("density"), dtype=float)
        grid_shape = data.get("grid_shape")
        if density.ndim == 1 and grid_shape:
            density = density.reshape(tuple(int(v) for v in grid_shape))
        if density.ndim != 3:
            return

        mode = str(data.get("visualization_mode") or "Density").lower()
        if "density" in mode and self._render_density_field(data, density):
            return
        if "voxel" not in mode and "density" not in mode:
            if self._render_smooth_topopt_density(data, density):
                return

        nelx, nely, nelz = density.shape
        origin, cell = self._topopt_grid_geometry(data, density)

        cutoff = float(np.clip(data.get("density_cutoff", 0.30), 0.01, 0.95))
        mask = density >= cutoff
        if not np.any(mask) and density.size:
            mask = density >= float(np.percentile(density, 90))

        points = vtk.vtkPoints()
        scalars = vtk.vtkFloatArray()
        scalars.SetName("Density")

        for ix, iy, iz in np.argwhere(mask):
            points.InsertNextPoint(
                origin[0] + (ix + 0.5) * cell[0],
                origin[1] + (iy + 0.5) * cell[1],
                origin[2] + (iz + 0.5) * cell[2],
            )
            scalars.InsertNextValue(float(density[ix, iy, iz]))

        poly_data = vtk.vtkPolyData()
        poly_data.SetPoints(points)
        poly_data.GetPointData().SetScalars(scalars)

        cube = vtk.vtkCubeSource()
        cube.SetXLength(float(cell[0] * 0.92))
        cube.SetYLength(float(cell[1] * 0.92))
        cube.SetZLength(float(cell[2] * 0.92))

        glyph = vtk.vtkGlyph3D()
        glyph.SetInputData(poly_data)
        glyph.SetSourceConnection(cube.GetOutputPort())
        glyph.SetScaleFactor(1.0)
        glyph.SetColorModeToColorByScalar()
        try:
            glyph.ScalingOff()
        except AttributeError:
            pass

        lut = vtk.vtkLookupTable()
        lut.SetHueRange(0.67, 0.16)
        lut.SetSaturationRange(0.75, 0.95)
        lut.SetValueRange(0.65, 1.0)
        lut.Build()

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(glyph.GetOutputPort())
        mapper.SetScalarModeToUsePointData()
        mapper.SetScalarRange(0.0, 1.0)
        mapper.SetLookupTable(lut)

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetRepresentationToSurface()
        actor.GetProperty().SetEdgeVisibility(1 if self._show_edges else 0)
        actor.GetProperty().SetEdgeColor(0.08, 0.09, 0.10)

        self.renderer.AddActor(actor)
        self.current_actor = actor
        self.actors.append(actor)
        if self._section_enabled:
            self._update_section_cut()
        self._render_topopt_joint_overlays(data)
        self._update_scalar_bar("Density", 0.0, 1.0, lut)
        self.renderer.ResetCamera()
        self.vtkWidget.GetRenderWindow().Render()

    def _clear_topopt_joint_overlays(self):
        """Remove multibody joint reference geometry from the previous result."""
        for actor in list(getattr(self, "_topopt_joint_actors", [])):
            self.renderer.RemoveActor(actor)
            if actor in self.actors:
                self.actors.remove(actor)
        self._topopt_joint_actors = []

    def _topopt_beam_lattice_surface(self, data):
        """Tessellate a strut-lattice result, cached on the result dict.

        Returns ``(vertices, faces)`` or ``None`` when this result is not a
        strut lattice. Cached because building the lattice from the density
        field costs a few tenths of a second and the viewer re-renders on every
        camera-independent state change — a section plane, an edge toggle.
        """
        if not isinstance(data, dict):
            return None
        cached = data.get("_beam_lattice_surface")
        if cached is not None:
            return cached
        try:
            from pylcss.design_studio.topology_optimization.geometry.cad_reconstruction import (
                build_topopt_beam_lattice,
            )
            from pylcss.design_studio.topology_optimization.geometry.lattice_cad import (
                beam_lattice_triangles,
                lattice_cad_strategy,
            )

            options = data.get("structure_options")
            if lattice_cad_strategy(options) != "beam":
                return None
            lattice = build_topopt_beam_lattice(
                data, options, data.get("member_plan")
            )
            surface = beam_lattice_triangles(lattice)
        except Exception:
            logger.debug("Analytic lattice display unavailable.", exc_info=True)
            return None
        data["_beam_lattice_surface"] = surface
        return surface

    def _render_topopt_beam_lattice(self, data):
        """Render a strut lattice from its members. True if it was handled."""
        surface = self._topopt_beam_lattice_surface(data)
        if surface is None:
            return False
        poly_data = build_surface_polydata(surface[0], surface[1])
        if poly_data is None:
            return False

        self.clear()
        self.scalar_bar.VisibilityOff()
        normals = vtk.vtkPolyDataNormals()
        normals.SetInputData(poly_data)
        normals.ConsistencyOn()
        normals.AutoOrientNormalsOn()
        # Each member is a closed prism of its own, so the shells meet at hard
        # creases at every joint. Splitting the normals there keeps those edges
        # crisp instead of smearing a fake shading gradient across them.
        normals.SplittingOn()
        normals.SetFeatureAngle(45.0)

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(normals.GetOutputPort())
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        self.apply_cad_material(actor)
        # Every member contributes its own facet edges, so wireframe on a
        # lattice is an unreadable thicket. Leave it off regardless of the
        # global toggle.
        actor.GetProperty().SetEdgeVisibility(0)

        self.renderer.AddActor(actor)
        self.current_actor = actor
        self.actors.append(actor)
        self._pickable_surface_dataset = vtk.vtkPolyData()
        self._pickable_surface_dataset.DeepCopy(poly_data)
        self._update_scalar_bar("", 0, 1, None)
        if self._section_enabled:
            self._update_section_cut()
        self._render_topopt_joint_overlays(data)
        self.renderer.ResetCamera()
        self.vtkWidget.GetRenderWindow().Render()
        return True

    def _render_topopt_joint_overlays(self, data):
        """Show kinematic joint pins without treating them as optimized material."""
        self._clear_topopt_joint_overlays()
        multibody = data.get("multibody") if isinstance(data, dict) else None
        joints = multibody.get("global_joints") if isinstance(multibody, dict) else None
        bounds = data.get("bounds") if isinstance(data, dict) else None
        if not joints or not isinstance(bounds, dict):
            return
        try:
            mins = np.asarray(bounds["min"], dtype=float)[:3]
            maxs = np.asarray(bounds["max"], dtype=float)[:3]
        except Exception:
            return
        span = maxs - mins
        if not np.all(np.isfinite(span)) or np.any(span <= 0.0):
            return

        radius = max(0.45, 0.016 * float(np.max(span)))
        axis_index = {"x": 0, "y": 1, "z": 2}
        for joint in joints:
            try:
                start = mins + np.asarray(joint["anchor_a"], dtype=float)[:3] * span
                end = mins + np.asarray(joint["anchor_b"], dtype=float)[:3] * span
            except Exception:
                continue
            if float(np.linalg.norm(end - start)) < 1e-9:
                index = axis_index.get(str(joint.get("axis") or "z").lower(), 2)
                direction = np.zeros(3, dtype=float)
                direction[index] = max(4.0 * radius, 0.12 * float(span[index]))
                start = start - 0.5 * direction
                end = end + 0.5 * direction

            line = vtk.vtkLineSource()
            line.SetPoint1(*(float(value) for value in start))
            line.SetPoint2(*(float(value) for value in end))
            tube = vtk.vtkTubeFilter()
            tube.SetInputConnection(line.GetOutputPort())
            tube.SetRadius(radius)
            tube.SetNumberOfSides(24)
            tube.CappingOn()
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(tube.GetOutputPort())
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            # A pin is reference hardware, not optimized density.  Use a
            # deliberately distinct cyan so it cannot be mistaken for the
            # amber force/support glyphs or for result material.
            actor.GetProperty().SetColor(0.12, 0.82, 1.0)
            try:
                actor.GetProperty().SetMetallic(0.35)
                actor.GetProperty().SetRoughness(0.32)
            except AttributeError:
                pass
            self.renderer.AddActor(actor)
            self.actors.append(actor)
            self._topopt_joint_actors.append(actor)
            label = vtk.vtkBillboardTextActor3D()
            label.SetInput(
                f"JOINT: {str(joint.get('name') or joint.get('type') or 'connector')}"
            )
            midpoint = 0.5 * (start + end)
            label.SetPosition(*(float(value) for value in midpoint))
            text = label.GetTextProperty()
            text.SetColor(0.12, 0.82, 1.0)
            text.SetFontSize(15)
            text.BoldOn()
            self.renderer.AddActor(label)
            self.actors.append(label)
            self._topopt_joint_actors.append(label)

    def _reset_scene_for_simulation(self):
        """Drop the previous render, keeping only the exact BC overlays.

        ``render_shape`` starts from a full ``clear()``, but this path used to
        remove ``current_actor`` alone.  Everything else a previous render had
        put in the renderer therefore survived and was drawn underneath the new
        result: the source body's wireframe/vertex actors, selection
        highlights, and the fallback load/support markers appended to
        ``self.actors``.  After an FEA run that reads as the original solid
        sitting on top of the result — the duplicated shape users see.

        BC overlay props are deliberately kept: ``render_bc_overlays`` owns
        their lifetime and re-renders them itself, and field/scale switches
        must not silently discard them.
        """
        self._clear_topopt_joint_overlays()
        self._clear_undeformed_overlay()
        self._clear_result_extrema()

        # Selection highlights belong to the body being replaced.  An active
        # picking session rebuilds its own highlights on every click, so leave
        # those alone rather than fighting the session.
        if not (
            self._picking_mode
            or self._edge_picking_mode
            or self._vertex_picking_mode
        ):
            self._clear_highlight_actors()
            self._clear_edge_highlight_actors()
            self._clear_vertex_highlight_actors()

        if self._edge_actor is not None:
            self.renderer.RemoveActor(self._edge_actor)
            self._edge_actor = None
        self._all_occ_edges = []
        self._edge_pd_list = []
        self._edge_cell_map = {}

        if self._vertex_actor is not None:
            self.renderer.RemoveActor(self._vertex_actor)
            self._vertex_actor = None
        self._all_occ_vertices = []

        self.forget_section_state()
        if self.current_actor:
            self.renderer.RemoveActor(self.current_actor)
            self.current_actor = None

        kept = []
        for actor in list(self.actors):
            if getattr(actor, "_bc_overlay", False):
                kept.append(actor)
                continue
            self.renderer.RemoveActor(actor)
            try:
                self.renderer.RemoveActor2D(actor)
            except Exception:
                logging.getLogger(__name__).debug(
                    "Optional UI operation failed.", exc_info=True
                )
        self.actors = kept

        self._last_simulation_mesh = None
        self._face_map = {}
        self._all_occ_faces = []
        self._face_polydata_list = []
        self._pickable_surface_dataset = None

    def render_simulation(self, data, is_sub_render: bool = False):
        """
        Render simulation results (Mesh, FEA Result, or list of FEA components/meshes).
        """
        if data is None:
            return
        if isinstance(data, (list, tuple)) and data:
            if not is_sub_render:
                self._reset_scene_for_simulation()
            for item in data:
                if item is not None:
                    self.render_simulation(item, is_sub_render=True)
            self.renderer.ResetCamera()
            self.vtkWidget.GetRenderWindow().Render()
            return

        preserve_field_selector = bool(
            isinstance(data, dict)
            and data.get("_preserve_field_selector")
        )
        if not preserve_field_selector and not is_sub_render:
            self._configure_result_fields(data)

        # Auto-start crash animation playback when solver result with frames arrives
        if (
            isinstance(data, dict)
            and data.get("type") == "crash"
            and data.get("frames")
        ):
            self.start_crash_playback(data)
            return

        # Leaving crash playback: drop the rigid-wall overlay on any render that
        # is not a crash animation frame (e.g. an FEA result or plain geometry).
        if getattr(self, "_crash_wall_actor", None) is not None and not (
            isinstance(data, dict) and data.get("type") == "crash_frame"
        ):
            self._clear_crash_wall_visuals()
            self._crash_wall_info = None

        # 1. Clean up previous render
        if not is_sub_render:
            self._reset_scene_for_simulation()

        # A TopOpt result owns both the design field and, when requested, a
        # full independent CalculiX result. Keep one field selector for the
        # parent study and pass the cached validation mesh through the normal
        # FEA renderer so stress, displacement, probing, extrema, deformation,
        # and sectioning behave exactly like a standalone verification run.
        if isinstance(data, dict) and data.get("type") == "topopt_voxel":
            validation_modes = {
                "Validated Von Mises Stress": "Von Mises Stress",
                "Validated Displacement": "Displacement",
            }
            requested_mode = str(data.get("visualization_mode") or "")
            if requested_mode == "Manufactured Mesh":
                if self._render_topopt_beam_lattice(data):
                    return
                data = dict(data)
                data["visualization_mode"] = "Surface"
            if requested_mode in {"CAD", "Reconstructed CAD (B-rep)"}:
                # A strut lattice is displayed from its centrelines, not from
                # its B-rep. The bodies are exact either way; the difference is
                # that OpenCASCADE tessellates a compound one face at a time,
                # which measured 15 s for a 668-member octet -- and that is a
                # small lattice. Tessellating the members directly is the same
                # picture in 3 ms and gets rounder struts, because the segment
                # count is chosen rather than inherited from a chord tolerance.
                if self._render_topopt_beam_lattice(data):
                    return
                cad_shape = data.get("cad_shape")
                if cad_shape is None:
                    cad_shape = data.get("shape")
                if cad_shape is not None:
                    reconstruction = data.get("cad_reconstruction")
                    smooth_freeform = bool(
                        isinstance(reconstruction, dict)
                        and reconstruction.get("method") == "Smooth Freeform"
                    )
                    cad_preview = data.get("cad_preview_mesh")
                    if (
                        smooth_freeform
                        and isinstance(cad_preview, dict)
                        and cad_preview.get("vertices") is not None
                        and cad_preview.get("faces") is not None
                    ):
                        # OCC face-by-face tessellation and edge sampling of a
                        # 3,000-patch freeform B-rep can block the Qt thread for
                        # tens of seconds. CAD systems display a cached
                        # tessellation too; keep the exact live B-rep for
                        # picking/export and use its saved display mesh here.
                        data = dict(data)
                        data["recovered_shape"] = cad_preview
                        data["visualization_mode"] = "Surface"
                        data["_cad_preview"] = True
                    else:
                        self.render_shape(
                            cad_shape,
                            initial_edge_visibility=(
                                False if smooth_freeform else None
                            ),
                            smooth_across_faces=smooth_freeform,
                        )
                        self._render_topopt_joint_overlays(data)
                        self.vtkWidget.GetRenderWindow().Render()
                        return
                # Keep the last recovered surface visible while an on-demand
                # reconstruction is running in the background.
                data = dict(data)
                data["visualization_mode"] = "Surface"
            validation = data.get("validation")
            if requested_mode in validation_modes and isinstance(validation, dict):
                validation_payload = dict(validation)
                validation_payload["visualization_mode"] = validation_modes[
                    requested_mode
                ]
                validation_payload["_preserve_field_selector"] = True
                self.render_simulation(validation_payload)
                return

        # --- RECOVERED SURFACE VIZ (Triangulated Surface) ---
        # (see _render_topopt_beam_lattice for the strut-lattice CAD path)
        if (
            isinstance(data, dict)
            and data.get("visualization_mode")
            in {
                "Surface",
                "Manufactured Mesh",
                "Recovered Shape",
                "Recovered Surface (Mesh)",
            }
        ):
            rec = data.get("recovered_shape")
            if rec and "vertices" in rec and "faces" in rec:
                poly_data = build_surface_polydata(rec["vertices"], rec["faces"])
            else:
                poly_data = None
            if poly_data is not None:
                normals = vtk.vtkPolyDataNormals()
                normals.SetInputData(poly_data)
                normals.ConsistencyOn()
                normals.AutoOrientNormalsOn()
                normals.SplittingOff()
                normals.SetFeatureAngle(80.0)

                mapper = vtk.vtkPolyDataMapper()
                mapper.SetInputConnection(normals.GetOutputPort())

                actor = vtk.vtkActor()
                actor.SetMapper(mapper)
                self.apply_cad_material(actor)
                actor.GetProperty().SetOpacity(1.0)
                actor.GetProperty().SetEdgeVisibility(
                    1
                    if self._show_edges and not data.get("_cad_preview")
                    else 0
                )
                actor.GetProperty().SetEdgeColor(0.06, 0.12, 0.16)

                self.renderer.AddActor(actor)
                self.current_actor = actor
                self.actors.append(actor)
                if self._section_enabled:
                    self._update_section_cut()
                self._render_topopt_joint_overlays(data)

                self._pickable_surface_dataset = vtk.vtkPolyData()
                self._pickable_surface_dataset.DeepCopy(poly_data)

                # Update scalar bar to be empty for geometry view
                self._update_scalar_bar("", 0, 1, None)

                self.renderer.ResetCamera()
                self.vtkWidget.GetRenderWindow().Render()
                return

        # Imported STL/OBJ meshes arrive as a plain surface dict.  Render that
        # surface directly so users can inspect it before a volume remesh runs.
        if (
            isinstance(data, dict)
            and data.get("type") != "topopt_voxel"
            and "vertices" in data
            and "faces" in data
        ):
            poly_data = build_surface_polydata(data["vertices"], data["faces"])
            if poly_data is not None:
                normals = vtk.vtkPolyDataNormals()
                normals.SetInputData(poly_data)
                normals.ConsistencyOn()
                normals.AutoOrientNormalsOn()
                normals.SplittingOn()
                normals.SetFeatureAngle(45.0)

                mapper = vtk.vtkPolyDataMapper()
                mapper.SetInputConnection(normals.GetOutputPort())

                actor = vtk.vtkActor()
                actor.SetMapper(mapper)
                actor.GetProperty().SetColor(0.72, 0.76, 0.78)
                actor.GetProperty().SetOpacity(1.0)
                try:
                    actor.GetProperty().SetInterpolationToPhong()
                except AttributeError:
                    pass
                actor.GetProperty().SetEdgeVisibility(
                    1 if self._show_edges else 0
                )

                self.renderer.AddActor(actor)
                self.current_actor = actor
                self.actors.append(actor)
                if self._section_enabled:
                    self._update_section_cut()

                self._pickable_surface_dataset = vtk.vtkPolyData()
                self._pickable_surface_dataset.DeepCopy(poly_data)

                self._update_scalar_bar("", 0, 1, None)
                self.renderer.ResetCamera()
                self.vtkWidget.GetRenderWindow().Render()
                return

        if isinstance(data, dict) and data.get("type") == "topopt_voxel":
            self._render_voxel_topopt(data)
            return

        # Check if it's a Mesh object or Result dict
        mesh = None
        displacement = None
        density = None
        stress = None
        scalar_name = "VonMises"
        scalar_title = "Von Mises Stress (MPa)"
        visualization_mode = "Von Mises Stress"
        density_cutoff = 0.30
        stress_location = ""
        locked_scalar_range = (
            None  # (lo, hi) supplied by crash playback for stable colormap
        )
        _def_scale = 1.0  # visualisation deformation scale factor

        # Detect skfem Mesh
        if hasattr(data, "p") and hasattr(data, "t"):
            mesh = data
        # Detect Result Dict
        elif isinstance(data, dict) and "mesh" in data:
            mesh = data["mesh"]
            if "displacement" in data:
                displacement = data["displacement"]
            if "density" in data:
                density = data["density"]
            if "stress" in data:
                stress = data["stress"]
            stress_location = str(data.get("stress_location") or "")
            if "visualization_mode" in data:
                visualization_mode = data["visualization_mode"]
            if "density_cutoff" in data:
                density_cutoff = float(data["density_cutoff"])
            if "_scalar_range" in data:
                locked_scalar_range = data["_scalar_range"]
            if "deformation_scale" in data:
                raw_scale = data["deformation_scale"]
                try:
                    if isinstance(raw_scale, str):
                        text_scale = raw_scale.strip().lower()
                        _def_scale = (
                            1.0
                            if text_scale == "auto"
                            else float(text_scale.rstrip("x"))
                        )
                    else:
                        _def_scale = float(raw_scale)
                except Exception:
                    _def_scale = 1.0
            tensor = np.asarray(data.get("stress_tensor", []), dtype=float)
            component_fields = {
                "Stress XX": (0, "StressXX", "Normal Stress XX (MPa)"),
                "Stress YY": (1, "StressYY", "Normal Stress YY (MPa)"),
                "Stress ZZ": (2, "StressZZ", "Normal Stress ZZ (MPa)"),
                "Stress XY": (3, "StressXY", "Shear Stress XY (MPa)"),
                "Stress YZ": (4, "StressYZ", "Shear Stress YZ (MPa)"),
                "Stress ZX": (5, "StressZX", "Shear Stress ZX (MPa)"),
            }
            if (
                tensor.ndim == 2
                and tensor.shape[1] >= 6
                and tensor.shape[0] == np.asarray(mesh.p).shape[1]
            ):
                if visualization_mode in component_fields:
                    column, scalar_name, scalar_title = component_fields[
                        visualization_mode
                    ]
                    stress = tensor[:, column]
                elif visualization_mode in {
                    "Maximum Principal Stress",
                    "Minimum Principal Stress",
                }:
                    matrices = np.zeros((tensor.shape[0], 3, 3), dtype=float)
                    matrices[:, 0, 0] = tensor[:, 0]
                    matrices[:, 1, 1] = tensor[:, 1]
                    matrices[:, 2, 2] = tensor[:, 2]
                    matrices[:, 0, 1] = matrices[:, 1, 0] = tensor[:, 3]
                    matrices[:, 1, 2] = matrices[:, 2, 1] = tensor[:, 4]
                    matrices[:, 2, 0] = matrices[:, 0, 2] = tensor[:, 5]
                    principal = np.linalg.eigvalsh(matrices)
                    if visualization_mode == "Maximum Principal Stress":
                        stress = principal[:, 2]
                        scalar_name = "MaxPrincipal"
                        scalar_title = "Maximum Principal Stress (MPa)"
                    else:
                        stress = principal[:, 0]
                        scalar_name = "MinPrincipal"
                        scalar_title = "Minimum Principal Stress (MPa)"

        if mesh is None:
            return
        self._active_result_mode = visualization_mode

        # 2. Create VTK Unstructured Grid
        points = vtk.vtkPoints()
        grid = vtk.vtkUnstructuredGrid()

        pts = mesh.p
        n_points = pts.shape[1]

        # Apply displacement if available (scaled for visualisation)
        if displacement is not None:
            if len(displacement) == 3 * n_points:
                disp_3n = displacement.reshape((3, n_points), order="F")
                pts = pts + disp_3n * _def_scale

        points = vtk_points_from_array(pts)
        grid.SetPoints(points)

        base_tets = np.asarray(mesh.t, dtype=int)
        is_shell_mesh = base_tets.shape[0] == 3
        quadratic_tets = None
        if not is_shell_mesh:
            from pylcss.solver_backends.mesh import tet10_connectivity

            quadratic_tets = tet10_connectivity(mesh)
        tets = quadratic_tets if quadratic_tets is not None else base_tets
        is_quadratic_tet = quadratic_tets is not None
        # Detect shell (triangle) meshes — first dimension is 3 instead of 4/10.
        # The viewer otherwise emits vtkTetra cells, which would silently render
        # nothing for a surface mesh.
        cell_blocks = getattr(mesh, "cell_blocks", None)

        # Add Density Data if available
        if density is not None:
            grid.GetCellData().SetScalars(vtk_scalar_array(density, "Density"))

        if stress is not None:
            grid.GetPointData().AddArray(vtk_scalar_array(stress, scalar_name))
            if visualization_mode != "Displacement":
                grid.GetPointData().SetActiveScalars(scalar_name)

        if displacement is not None:
            if len(displacement) == 3 * n_points:
                disp_3n = displacement.reshape((3, n_points), order="F")
                mag = np.linalg.norm(disp_3n, axis=0)
                grid.GetPointData().AddArray(vtk_scalar_array(mag, "Displacement"))
                if visualization_mode == "Displacement":
                    grid.GetPointData().SetActiveScalars("Displacement")

        if cell_blocks:
            vtk_cell_types = {
                "vertex": (vtk.VTK_VERTEX, 1),
                "line": (vtk.VTK_LINE, 2),
                "triangle": (vtk.VTK_TRIANGLE, 3),
                "quad": (vtk.VTK_QUAD, 4),
                "tetra": (vtk.VTK_TETRA, 4),
                "tetra10": (vtk.VTK_QUADRATIC_TETRA, 10),
                "hexahedron": (vtk.VTK_HEXAHEDRON, 8),
                "wedge": (vtk.VTK_WEDGE, 6),
                "pyramid": (vtk.VTK_PYRAMID, 5),
            }
            blocks = []
            for cell_type, connectivity in cell_blocks:
                cell_spec = vtk_cell_types.get(str(cell_type))
                if cell_spec is None:
                    continue
                vtk_type, node_count = cell_spec
                connectivity = np.asarray(connectivity, dtype=np.int64)
                if connectivity.ndim != 2 or connectivity.shape[1] < node_count:
                    continue
                blocks.append((vtk_type, connectivity[:, :node_count]))
            set_mixed_cells(grid, blocks, n_points)
        elif is_shell_mesh:
            set_uniform_cells(
                grid, tets[:3].T, vtk.VTK_TRIANGLE, n_points, cache_owner=mesh
            )
        elif is_quadratic_tet:
            set_uniform_cells(
                grid,
                tets[:10].T,
                vtk.VTK_QUADRATIC_TETRA,
                n_points,
                cache_owner=mesh,
            )
        else:
            set_uniform_cells(
                grid, tets[:4].T, vtk.VTK_TETRA, n_points, cache_owner=mesh
            )

        # 3. Mapper and Actor
        mapper = vtk.vtkDataSetMapper()

        if density is not None:
            cutoff = float(np.clip(density_cutoff, 0.05, 0.95))
            lower = max(0.01, cutoff)
            upper = 1.1

            threshold = vtk.vtkThreshold()
            threshold.SetInputData(grid)
            threshold.SetInputArrayToProcess(
                0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_CELLS, "Density"
            )
            threshold.SetLowerThreshold(lower)
            threshold.SetUpperThreshold(upper)
            threshold.SetThresholdFunction(
                getattr(vtk.vtkThreshold, "THRESHOLD_BETWEEN", 0)
            )

            try:
                threshold.SetPassPointArrays(True)
            except AttributeError:
                pass

            threshold.Update()

            threshold_output = threshold.GetOutput()

            if threshold_output.GetNumberOfCells() == 0 and len(density) > 0:
                relaxed = float(max(0.01, np.percentile(density, 10)))
                threshold.SetLowerThreshold(relaxed)
                threshold.SetUpperThreshold(upper)
                threshold.SetThresholdFunction(vtk.vtkThreshold.THRESHOLD_BETWEEN)
                threshold.Update()
                threshold_output = threshold.GetOutput()

            mapper.SetInputData(threshold_output)
            mapper.SetScalarRange(0, 1)
            section_volume = threshold_output
        else:
            surface = vtk.vtkDataSetSurfaceFilter()
            surface.SetInputData(grid)
            surface.Update()
            mapper.SetInputData(surface.GetOutput())
            # Drawing the extracted skin keeps rendering cheap, but the volume
            # has to stay reachable: a section clip of the skin alone is a
            # hollow shell with no field on the cut face.
            section_volume = grid

            dataset = surface.GetOutput()
            self._pickable_surface_dataset = vtk.vtkPolyData()
            self._pickable_surface_dataset.DeepCopy(dataset)

            if dataset.GetPointData().GetScalars() is not None:
                scalars = dataset.GetPointData().GetScalars()
                stress_display_clipped = False
                if locked_scalar_range is not None:
                    min_val, max_val = locked_scalar_range
                elif (
                    scalars.GetName() == "VonMises"
                    and visualization_mode == "Von Mises Stress"
                    and stress is not None
                ):
                    (
                        min_val,
                        max_val,
                        stress_display_clipped,
                    ) = _stress_display_range(stress, stress_location)
                else:
                    min_val, max_val = scalars.GetRange()

                if max_val - min_val < 1e-10:
                    max_val = min_val + 1.0

                mapper.SetScalarModeToUsePointData()
                mapper.SelectColorArray(scalars.GetName())
                mapper.SetScalarRange(min_val, max_val)

                lut = self._result_lut()  # Blue to Red, banded if requested
                mapper.SetLookupTable(lut)

                scalar_name = scalars.GetName()
                if scalar_name == "VonMises":
                    if visualization_mode == "Plastic Strain":
                        self._update_scalar_bar(
                            "Equivalent Plastic Strain", min_val, max_val, lut
                        )
                    elif visualization_mode == "Failed Elements":
                        self._update_scalar_bar(
                            "Element Failure (0=intact, 1=failed)",
                            min_val,
                            max_val,
                            lut,
                        )
                    else:
                        self._update_scalar_bar(
                            _stress_scalar_bar_title(stress_display_clipped),
                            min_val,
                            max_val,
                            lut,
                        )
                elif scalar_name == "Displacement":
                    _disp_title = (
                        f"Displacement (mm)  [scale: {_def_scale:.0f}\u00d7]"
                        if _def_scale != 1.0
                        else "Displacement (mm)"
                    )
                    self._update_scalar_bar(_disp_title, min_val, max_val, lut)
                else:
                    self._update_scalar_bar(
                        scalar_title,
                        min_val,
                        max_val,
                        lut,
                    )

            elif stress is not None and len(stress) > 0:
                stress_display_clipped = False
                if locked_scalar_range is not None:
                    min_s, max_s = locked_scalar_range
                else:
                    (
                        min_s,
                        max_s,
                        stress_display_clipped,
                    ) = _stress_display_range(stress, stress_location)

                if max_s - min_s < 1e-10:
                    max_s = min_s + 1.0

                mapper.SetScalarRange(min_s, max_s)
                mapper.SetScalarModeToUsePointData()

                lut = self._result_lut()
                mapper.SetLookupTable(lut)

                self._update_scalar_bar(
                    _stress_scalar_bar_title(stress_display_clipped),
                    min_s,
                    max_s,
                    lut,
                )

        if (
            density is not None
            and stress is not None
            and visualization_mode == "Von Mises Stress"
        ):
            mapper_input = mapper.GetInput()
            if mapper_input is not None and mapper_input.GetPointData() is not None:
                mapper_input.GetPointData().SetActiveScalars("VonMises")
            mapper.SetScalarModeToUsePointData()
            mapper.SelectColorArray("VonMises")

            # FIX: Skip void elements when calculating stress range for colorbar
            if (
                hasattr(mesh, "t")
                and mesh.t.shape[1] == len(density)
                and len(stress) == mesh.p.shape[1]
            ):
                # Nodes connected to solid elements
                solid_elems = np.where(
                    density >= float(np.clip(density_cutoff, 0.05, 0.95))
                )[0]
                if len(solid_elems) > 0:
                    solid_nodes = np.unique(mesh.t[:, solid_elems])
                    valid_stress = stress[solid_nodes]
                    min_s, max_s = (
                        float(np.min(valid_stress)),
                        float(np.max(valid_stress)),
                    )
                else:
                    min_s, max_s = float(np.min(stress)), float(np.max(stress))
            else:
                min_s, max_s = float(np.min(stress)), float(np.max(stress))

            if max_s - min_s < 1e-10:
                max_s = min_s + 1.0

            mapper.SetScalarRange(min_s, max_s)

            lut = self._result_lut()
            mapper.SetLookupTable(lut)
            self._update_scalar_bar("Von Mises Stress (MPa)", min_s, max_s, lut)

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)

        if density is not None and visualization_mode == "Density":
            actor.GetProperty().SetColor(0.9, 0.7, 0.1)
            actor.GetProperty().SetRepresentationToSurface()
            actor.GetProperty().SetEdgeVisibility(1 if self._show_edges else 0)
        elif stress is not None or displacement is not None:
            # FEA / crash result contour — overlay the element edges so the
            # mesh stays visible on top of the stress / displacement field
            # instead of a smooth, mesh-less colored surface.
            actor.GetProperty().SetRepresentationToSurface()
            actor.GetProperty().SetEdgeVisibility(1 if self._show_edges else 0)
            actor.GetProperty().SetEdgeColor(0.15, 0.15, 0.17)
            try:
                actor.GetProperty().SetLineWidth(1.0)
            except Exception:
                logging.getLogger(__name__).debug(
                    "Optional UI operation failed.", exc_info=True
                )
        else:
            # Bare mesh (no field data) — the "Generate Mesh" / "Remesh"
            # preview.  Show element edges so it actually reads as a mesh
            # instead of a smooth solid that looks identical to the CAD body.
            actor.GetProperty().SetColor(0.72, 0.76, 0.78)
            actor.GetProperty().SetRepresentationToSurface()
            actor.GetProperty().SetEdgeVisibility(1 if self._show_edges else 0)
            actor.GetProperty().SetEdgeColor(0.16, 0.18, 0.21)
            try:
                actor.GetProperty().SetLineWidth(1.0)
            except Exception:
                logging.getLogger(__name__).debug(
                    "Optional UI operation failed.", exc_info=True
                )

        self.renderer.AddActor(actor)
        self.current_actor = actor
        self.actors.append(actor)
        self.register_section_volume(actor, section_volume)
        if self._section_enabled:
            self._update_section_cut()
        self._capture_result_extrema(mapper, visualization_mode)
        self._last_simulation_mesh = mesh if displacement is not None else None
        if displacement is not None and self._show_undeformed:
            self._render_undeformed_overlay(mesh)

        # 4. Debug Visualization (Loads/Constraints after solve)
        # If exact BC overlays are cached from the graph, prefer those over the
        # approximate centroid/bbox debug markers to avoid duplicate clutter.
        _has_exact_bc_overlay = self._cached_bc_data is not None
        if isinstance(data, dict) and not _has_exact_bc_overlay:
            try:
                scale = overlay_scale(self.current_actor.GetBounds())
                base_len = 2.4 * scale
                glyph_size = 0.82 * scale
            except Exception:
                base_len, glyph_size = 0.08, 0.03

            if "debug_loads" in data and data["debug_loads"]:
                for load in data["debug_loads"]:
                    if "vector" not in load:
                        continue
                    raw_vec = np.array(load["vector"], dtype=float)
                    raw_mag = float(np.linalg.norm(raw_vec))
                    if raw_mag < 1e-9:
                        continue
                    norm_dir = raw_vec / raw_mag

                    center = load.get("start") or load.get("pos")
                    if not center and "bbox" in load:
                        bb = load["bbox"]
                        center = [
                            (bb.xmin + bb.xmax) / 2,
                            (bb.ymin + bb.ymax) / 2,
                            (bb.zmin + bb.zmax) / 2,
                        ]
                    if not center:
                        continue

                    rel_mag = load.get("relative_mag", 1.0)
                    arrow_len = base_len * max(0.2, min(1.0, rel_mag)) * 1.5
                    c_rgb = BC_PALETTE["force"]
                    before_count = len(self.actors)
                    self._add_arrow(
                        center, norm_dir, color=c_rgb, scale=arrow_len / 5.0
                    )

                    # Magnitude label at the arrow tip (only for loads that
                    # actually carry a magnitude; relative_mag * raw_mag
                    # recovers the absolute Newton value used by the solver).
                    label_pos = [center[i] + norm_dir[i] * arrow_len for i in range(3)]
                    label = f"F {self._format_force_magnitude(raw_mag)}"
                    if label:
                        self._add_force_label(label_pos, label, color=c_rgb)
                    for fallback_actor in self.actors[before_count:]:
                        fallback_actor._bc_result_fallback = True

            if "debug_constraints" in data and data["debug_constraints"]:
                for const in data["debug_constraints"]:
                    center = const.get("pos")
                    if not center:
                        continue
                    metadata = dict(const)
                    metadata.setdefault("constraint_type", "Fixed")
                    c_rgb = constraint_color(metadata)
                    fixed_dofs = const.get("fixed_dofs")
                    before_count = len(self.actors)
                    self._add_constraint_glyph(
                        center, fixed_dofs=fixed_dofs, color=c_rgb, size=glyph_size
                    )
                    self._add_force_label(
                        center,
                        compact_constraint_label(metadata),
                        color=c_rgb,
                    )
                    for fallback_actor in self.actors[before_count:]:
                        fallback_actor._bc_result_fallback = True

            self._apply_bc_visibility(render=False)

        # 5. Re-apply cached BC face overlays on top of the simulation result
        #    so they remain visible after FEA/TopOpt solve.
        # Crash frames replace only the deformed mesh; the reference-condition
        # actors remain valid. Re-tessellating every selected face for every
        # animation frame caused avoidable flicker and playback stalls.
        skip_bc_replay = bool(
            isinstance(data, dict)
            and data.get("type") == "crash_frame"
            and self._bc_overlay_actors
        )
        if not skip_bc_replay and self._cached_bc_data is not None:
            c_faces, l_faces, l_vecs = self._cached_bc_data
            self.render_bc_overlays(
                constraint_faces=c_faces or None,
                load_faces=l_faces or None,
                load_vectors=l_vecs or None,
            )

        # Reset camera only when explicitly requested (first crash frame) or for
        # non-animation renders.  Skipping ResetCamera on subsequent animation
        # frames keeps the viewpoint fixed so mesh deformation is clearly visible
        # rather than being hidden by continuous re-centring.
        should_reset = True
        if isinstance(data, dict) and data.get("type") == "crash_frame":
            should_reset = bool(data.get("_reset_camera", False))
        if should_reset:
            self.renderer.ResetCamera()
        self.vtkWidget.GetRenderWindow().Render()

    def _clear_result_extrema(self):
        """Remove result-extrema markers and their cached locations."""
        for actor in list(getattr(self, "_result_extrema_actors", [])):
            self.renderer.RemoveActor(actor)
            self.bc_renderer.RemoveActor(actor)
        self._result_extrema_actors = []
        self._result_extrema_cache = []

    def _capture_result_extrema(self, mapper, visualization_mode):
        """Cache finite extrema from the exact dataset displayed by the mapper."""
        try:
            mapper.Update()
            dataset = mapper.GetInput()
            if dataset is None:
                return
            point_data = dataset.GetPointData()
            cell_data = dataset.GetCellData()
            scalars = point_data.GetScalars() if point_data is not None else None
            association = "point"
            if scalars is None and cell_data is not None:
                scalars = cell_data.GetScalars()
                association = "cell"
            if scalars is None or scalars.GetNumberOfTuples() == 0:
                return

            values = np.asarray(
                [
                    float(scalars.GetComponent(index, 0))
                    for index in range(scalars.GetNumberOfTuples())
                ],
                dtype=float,
            )
            finite = np.flatnonzero(np.isfinite(values))
            if finite.size == 0:
                return
            minimum_id = int(finite[np.argmin(values[finite])])
            maximum_id = int(finite[np.argmax(values[finite])])
            indices = [("MIN", minimum_id), ("MAX", maximum_id)]
            if minimum_id == maximum_id:
                indices = [("VALUE", minimum_id)]

            scalar_name = str(scalars.GetName() or visualization_mode or "Result")
            units = {
                "Displacement": "mm",
                "Density": "",
                "MaxPrincipal": "MPa",
                "MinPrincipal": "MPa",
                "StressXX": "MPa",
                "StressYY": "MPa",
                "StressZZ": "MPa",
                "StressXY": "MPa",
                "StressYZ": "MPa",
                "StressZX": "MPa",
            }.get(scalar_name, "")
            if scalar_name == "VonMises" and visualization_mode == "Von Mises Stress":
                units = "MPa"
            cached = []
            for label, index in indices:
                if association == "point":
                    position = dataset.GetPoint(index)
                else:
                    bounds = dataset.GetCell(index).GetBounds()
                    position = (
                        0.5 * (bounds[0] + bounds[1]),
                        0.5 * (bounds[2] + bounds[3]),
                        0.5 * (bounds[4] + bounds[5]),
                    )
                value = float(values[index])
                suffix = f" {units}" if units else ""
                cached.append(
                    {
                        "label": f"{label}  {value:.5g}{suffix}",
                        "position": tuple(float(v) for v in position),
                        "kind": label,
                    }
                )
            self._result_extrema_cache = cached
            if self._show_extrema:
                self._render_result_extrema()
        except Exception:
            logging.getLogger(__name__).debug(
                "Could not calculate result extrema.", exc_info=True
            )

    def _render_result_extrema(self):
        """Draw compact, camera-facing markers for cached extrema."""
        if not self._result_extrema_cache or self._result_extrema_actors:
            return
        try:
            bounds = self.current_actor.GetBounds() if self.current_actor else None
            if bounds and all(np.isfinite(bounds)):
                diagonal = float(
                    np.linalg.norm(
                        [
                            bounds[1] - bounds[0],
                            bounds[3] - bounds[2],
                            bounds[5] - bounds[4],
                        ]
                    )
                )
                model_center = np.asarray(
                    [
                        0.5 * (bounds[0] + bounds[1]),
                        0.5 * (bounds[2] + bounds[3]),
                        0.5 * (bounds[4] + bounds[5]),
                    ],
                    dtype=float,
                )
            else:
                diagonal = 1.0
                model_center = np.zeros(3, dtype=float)
            radius = max(diagonal * 0.008, 1.0e-6)
            for item in self._result_extrema_cache:
                color = (
                    (1.0, 0.33, 0.16)
                    if item["kind"] in {"MAX", "VALUE"}
                    else (0.10, 0.78, 1.0)
                )
                source = vtk.vtkSphereSource()
                source.SetCenter(*item["position"])
                source.SetRadius(radius)
                source.SetThetaResolution(16)
                source.SetPhiResolution(12)
                marker_mapper = vtk.vtkPolyDataMapper()
                marker_mapper.SetInputConnection(source.GetOutputPort())
                marker = vtk.vtkActor()
                marker.SetMapper(marker_mapper)
                marker.GetProperty().SetColor(*color)
                self.bc_renderer.AddActor(marker)
                self._result_extrema_actors.append(marker)

                text = vtk.vtkBillboardTextActor3D()
                offset_position = np.asarray(item["position"], dtype=float)
                outward = offset_position - model_center
                outward_norm = float(np.linalg.norm(outward))
                if outward_norm > 1.0e-12:
                    offset_position += outward * (5.0 * radius / outward_norm)
                else:
                    offset_position[2] += radius * 2.2
                offset_position[2] += (
                    2.5 * radius
                    if item["kind"] in {"MAX", "VALUE"}
                    else -2.5 * radius
                )
                text.SetPosition(*offset_position)
                text.SetInput(item["label"])
                # Keep the two captions readable even when the extrema occur
                # on adjacent nodes.  This screen-space stagger is stable
                # while orbiting, unlike a large model-space translation.
                try:
                    if item["kind"] == "MIN":
                        text.SetDisplayOffset(-10, -24)
                    elif item["kind"] in {"MAX", "VALUE"}:
                        text.SetDisplayOffset(10, 24)
                except AttributeError:
                    pass
                text_property = text.GetTextProperty()
                text_property.SetColor(*color)
                text_property.SetFontFamilyToArial()
                text_property.SetFontSize(14)
                text_property.BoldOn()
                text_property.ItalicOff()
                text_property.ShadowOff()
                # Anchor captions away from one another when extrema occur on
                # neighbouring nodes.  Display offsets alone move both labels
                # but still let their text extend in the same direction.
                if item["kind"] == "MIN":
                    text_property.SetJustificationToRight()
                    text_property.SetVerticalJustificationToTop()
                else:
                    text_property.SetJustificationToLeft()
                    text_property.SetVerticalJustificationToBottom()
                self.bc_renderer.AddActor(text)
                self._result_extrema_actors.append(text)
        except Exception:
            logging.getLogger(__name__).debug(
                "Could not draw result extrema.", exc_info=True
            )

    def _clear_undeformed_overlay(self):
        """Remove the previous reference-configuration wireframe."""
        actor = getattr(self, "_undeformed_actor", None)
        if actor is None:
            return
        self.renderer.RemoveActor(actor)
        if actor in self.actors:
            self.actors.remove(actor)
        self._undeformed_actor = None

    def _render_undeformed_overlay(self, mesh):
        """Build a neutral wireframe of the original analysis coordinates."""
        if mesh is None or not hasattr(mesh, "p") or not hasattr(mesh, "t"):
            return
        points_array = np.asarray(mesh.p, dtype=float)
        connectivity = np.asarray(mesh.t, dtype=int)
        if points_array.ndim != 2 or connectivity.ndim != 2:
            return

        # This overlay is rebuilt alongside the deformed result, so it carries
        # the same per-element cost and must use the vectorized path too.
        grid = vtk.vtkUnstructuredGrid()
        grid.SetPoints(vtk_points_from_array(points_array))
        n_points = points_array.shape[1]
        if connectivity.shape[0] == 3:
            cell_type, width = vtk.VTK_TRIANGLE, 3
        elif connectivity.shape[0] >= 10:
            cell_type, width = vtk.VTK_QUADRATIC_TETRA, 10
        elif connectivity.shape[0] >= 4:
            cell_type, width = vtk.VTK_TETRA, 4
        else:
            return
        if not set_uniform_cells(
            grid, connectivity[:width].T, cell_type, n_points
        ):
            return

        surface = vtk.vtkDataSetSurfaceFilter()
        surface.SetInputData(grid)
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(surface.GetOutputPort())
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetRepresentationToWireframe()
        actor.GetProperty().SetColor(0.65, 0.68, 0.72)
        actor.GetProperty().SetOpacity(0.65)
        actor.GetProperty().SetLineWidth(1.0)
        actor.SetVisibility(1 if self._show_undeformed else 0)
        self.renderer.AddActor(actor)
        self.actors.append(actor)
        self._undeformed_actor = actor
