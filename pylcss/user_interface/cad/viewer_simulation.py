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


class SimulationRenderingMixin:
    def update_simulation_field(self, mesh, values, field_name="Density"):
        """Update scalar field on existing mesh for real-time visualization."""
        data = {"mesh": mesh, "visualization_mode": field_name}
        if field_name == "Density":
            data["density"] = values
        elif field_name == "Von Mises Stress":
            data["stress"] = values

        self.render_simulation(data)

    def _render_voxel_topopt(self, data):
        """Render pyMOTO structured-density results as actual voxel cubes."""
        density = np.asarray(data.get("density"), dtype=float)
        grid_shape = data.get("grid_shape")
        if density.ndim == 1 and grid_shape:
            density = density.reshape(tuple(int(v) for v in grid_shape))
        if density.ndim != 3:
            return

        nelx, nely, nelz = density.shape
        bounds = data.get("bounds") if isinstance(data.get("bounds"), dict) else None
        if bounds and "min" in bounds and "max" in bounds:
            mins = np.asarray(bounds["min"], dtype=float)
            maxs = np.asarray(bounds["max"], dtype=float)
            if mins.size >= 3 and maxs.size >= 3 and np.all(maxs[:3] > mins[:3]):
                cell = (maxs[:3] - mins[:3]) / np.array([nelx, nely, nelz], dtype=float)
                origin = mins[:3]
            else:
                cell = np.ones(3, dtype=float)
                origin = np.array([-0.5 * nelx, -0.5 * nely, -0.5 * nelz], dtype=float)
        else:
            cell = np.ones(3, dtype=float)
            origin = np.array([-0.5 * nelx, -0.5 * nely, -0.5 * nelz], dtype=float)

        cutoff = float(np.clip(data.get("density_cutoff", 0.35), 0.01, 0.95))
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
        actor.GetProperty().EdgeVisibilityOn()
        actor.GetProperty().SetEdgeColor(0.08, 0.09, 0.10)

        self.renderer.AddActor(actor)
        self.current_actor = actor
        self.actors.append(actor)
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

    def render_simulation(self, data):
        """
        Render simulation results (Mesh or FEA Result).
        """
        if data is None:
            return

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
        self._clear_topopt_joint_overlays()
        if self.current_actor:
            self.renderer.RemoveActor(self.current_actor)
            self.current_actor = None
        self._face_map = {}
        self._all_occ_faces = []
        self._face_polydata_list = []
        self._pickable_surface_dataset = None

        # --- RECOVERED SHAPE VIZ (Triangulated Surface) ---
        if (
            isinstance(data, dict)
            and data.get("visualization_mode") == "Recovered Shape"
        ):
            rec = data.get("recovered_shape")
            if rec and "vertices" in rec and "faces" in rec:
                verts = rec["vertices"]
                faces = rec["faces"]

                poly_data = vtk.vtkPolyData()
                points = vtk.vtkPoints()
                for v in verts:
                    points.InsertNextPoint(v[0], v[1], v[2])
                poly_data.SetPoints(points)

                cells = vtk.vtkCellArray()
                for f in faces:
                    triangle = vtk.vtkTriangle()
                    triangle.GetPointIds().SetId(0, f[0])
                    triangle.GetPointIds().SetId(1, f[1])
                    triangle.GetPointIds().SetId(2, f[2])
                    cells.InsertNextCell(triangle)
                poly_data.SetPolys(cells)

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
                # Soft premium gray
                actor.GetProperty().SetColor(0.8, 0.8, 0.8)
                actor.GetProperty().SetOpacity(1.0)
                try:
                    actor.GetProperty().SetInterpolationToPhong()
                except AttributeError:
                    pass
                actor.GetProperty().EdgeVisibilityOff()

                self.renderer.AddActor(actor)
                self.current_actor = actor
                self.actors.append(actor)
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
            try:
                verts = np.asarray(data["vertices"], dtype=float)
                faces = np.asarray(data["faces"], dtype=int)
            except Exception:
                verts = faces = None
            if (
                verts is not None
                and faces is not None
                and verts.ndim == 2
                and verts.shape[1] >= 3
                and faces.ndim == 2
                and faces.shape[1] >= 3
                and len(verts) > 0
                and len(faces) > 0
            ):
                poly_data = vtk.vtkPolyData()
                points = vtk.vtkPoints()
                for v in verts:
                    points.InsertNextPoint(v[0], v[1], v[2])
                poly_data.SetPoints(points)

                cells = vtk.vtkCellArray()
                for f in faces:
                    if len(f) == 3:
                        cell = vtk.vtkTriangle()
                        for j in range(3):
                            cell.GetPointIds().SetId(j, int(f[j]))
                    else:
                        cell = vtk.vtkPolygon()
                        cell.GetPointIds().SetNumberOfIds(len(f))
                        for j, idx in enumerate(f):
                            cell.GetPointIds().SetId(j, int(idx))
                    cells.InsertNextCell(cell)
                poly_data.SetPolys(cells)

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
                actor.GetProperty().EdgeVisibilityOff()

                self.renderer.AddActor(actor)
                self.current_actor = actor
                self.actors.append(actor)

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
        visualization_mode = "Von Mises Stress"
        density_cutoff = 0.5
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

        if mesh is None:
            return

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

        for i in range(n_points):
            points.InsertNextPoint(pts[0, i], pts[1, i], pts[2, i])

        grid.SetPoints(points)

        base_tets = np.asarray(mesh.t, dtype=int)
        is_shell_mesh = base_tets.shape[0] == 3
        quadratic_tets = None
        if not is_shell_mesh:
            from pylcss.solver_backends.mesh import tet10_connectivity

            quadratic_tets = tet10_connectivity(mesh)
        tets = quadratic_tets if quadratic_tets is not None else base_tets
        is_quadratic_tet = quadratic_tets is not None
        n_tets = tets.shape[1]
        # Detect shell (triangle) meshes — first dimension is 3 instead of 4/10.
        # The viewer otherwise emits vtkTetra cells, which would silently render
        # nothing for a surface mesh.
        cell_blocks = getattr(mesh, "cell_blocks", None)

        # Add Density Data if available
        if density is not None:
            density_array = vtk.vtkFloatArray()
            density_array.SetName("Density")
            for d in density:
                density_array.InsertNextValue(float(d))
            grid.GetCellData().SetScalars(density_array)

        if stress is not None:
            s_array = vtk.vtkFloatArray()
            s_array.SetName("VonMises")
            for s in stress:
                s_array.InsertNextValue(float(s))
            grid.GetPointData().AddArray(s_array)
            if visualization_mode in (
                "Von Mises Stress",
                "Plastic Strain",
                "Failed Elements",
            ):
                grid.GetPointData().SetActiveScalars("VonMises")

        if displacement is not None:
            if len(displacement) == 3 * n_points:
                disp_3n = displacement.reshape((3, n_points), order="F")
                mag = np.linalg.norm(disp_3n, axis=0)
                mag_array = vtk.vtkFloatArray()
                mag_array.SetName("Displacement")
                for m in mag:
                    mag_array.InsertNextValue(m)
                grid.GetPointData().AddArray(mag_array)
                if visualization_mode == "Displacement":
                    grid.GetPointData().SetActiveScalars("Displacement")

        if cell_blocks:
            vtk_cell_types = {
                "vertex": (vtk.vtkVertex, 1),
                "line": (vtk.vtkLine, 2),
                "triangle": (vtk.vtkTriangle, 3),
                "quad": (vtk.vtkQuad, 4),
                "tetra": (vtk.vtkTetra, 4),
                "tetra10": (vtk.vtkQuadraticTetra, 10),
                "hexahedron": (vtk.vtkHexahedron, 8),
                "wedge": (vtk.vtkWedge, 6),
                "pyramid": (vtk.vtkPyramid, 5),
            }
            for cell_type, connectivity in cell_blocks:
                cell_spec = vtk_cell_types.get(str(cell_type))
                if cell_spec is None:
                    continue
                cell_class, node_count = cell_spec
                connectivity = np.asarray(connectivity, dtype=int)
                if connectivity.ndim != 2 or connectivity.shape[1] < node_count:
                    continue
                for conn in connectivity:
                    cell = cell_class()
                    for j in range(node_count):
                        cell.GetPointIds().SetId(j, int(conn[j]))
                    grid.InsertNextCell(cell.GetCellType(), cell.GetPointIds())
        elif is_shell_mesh:
            for i in range(n_tets):
                tri = vtk.vtkTriangle()
                for j in range(3):
                    tri.GetPointIds().SetId(j, int(tets[j, i]))
                grid.InsertNextCell(tri.GetCellType(), tri.GetPointIds())
        elif is_quadratic_tet:
            for i in range(n_tets):
                tet = vtk.vtkQuadraticTetra()
                for j in range(10):
                    tet.GetPointIds().SetId(j, int(tets[j, i]))
                grid.InsertNextCell(tet.GetCellType(), tet.GetPointIds())
        else:
            for i in range(n_tets):
                tet = vtk.vtkTetra()
                for j in range(4):
                    tet.GetPointIds().SetId(j, int(tets[j, i]))
                grid.InsertNextCell(tet.GetCellType(), tet.GetPointIds())

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
        else:
            surface = vtk.vtkDataSetSurfaceFilter()
            surface.SetInputData(grid)
            surface.Update()
            mapper.SetInputData(surface.GetOutput())

            dataset = surface.GetOutput()
            self._pickable_surface_dataset = vtk.vtkPolyData()
            self._pickable_surface_dataset.DeepCopy(dataset)

            if dataset.GetPointData().GetScalars() is not None:
                scalars = dataset.GetPointData().GetScalars()
                if locked_scalar_range is not None:
                    min_val, max_val = locked_scalar_range
                else:
                    min_val, max_val = scalars.GetRange()

                if max_val - min_val < 1e-10:
                    max_val = min_val + 1.0

                mapper.SetScalarModeToUsePointData()
                mapper.SelectColorArray(scalars.GetName())
                mapper.SetScalarRange(min_val, max_val)

                lut = vtk.vtkLookupTable()
                lut.SetHueRange(0.667, 0.0)  # Blue to Red
                lut.Build()
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
                            "Von Mises Stress (MPa)", min_val, max_val, lut
                        )
                elif scalar_name == "Displacement":
                    _disp_title = (
                        f"Displacement (mm)  [scale: {_def_scale:.0f}\u00d7]"
                        if _def_scale != 1.0
                        else "Displacement (mm)"
                    )
                    self._update_scalar_bar(_disp_title, min_val, max_val, lut)

            elif stress is not None and len(stress) > 0:
                if locked_scalar_range is not None:
                    min_s, max_s = locked_scalar_range
                else:
                    min_s, max_s = float(np.min(stress)), float(np.max(stress))

                if max_s - min_s < 1e-10:
                    max_s = min_s + 1.0

                mapper.SetScalarRange(min_s, max_s)
                mapper.SetScalarModeToUsePointData()

                lut = vtk.vtkLookupTable()
                lut.SetHueRange(0.667, 0.0)
                lut.Build()
                mapper.SetLookupTable(lut)

                self._update_scalar_bar("Von Mises Stress (MPa)", min_s, max_s, lut)

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

            lut = vtk.vtkLookupTable()
            lut.SetHueRange(0.667, 0.0)
            lut.Build()
            mapper.SetLookupTable(lut)
            self._update_scalar_bar("Von Mises Stress (MPa)", min_s, max_s, lut)

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)

        if density is not None and visualization_mode == "Density":
            actor.GetProperty().SetColor(0.9, 0.7, 0.1)
            actor.GetProperty().SetRepresentationToSurface()
            actor.GetProperty().EdgeVisibilityOn()
        elif stress is not None or displacement is not None:
            # FEA / crash result contour — overlay the element edges so the
            # mesh stays visible on top of the stress / displacement field
            # instead of a smooth, mesh-less colored surface.
            actor.GetProperty().SetRepresentationToSurface()
            actor.GetProperty().EdgeVisibilityOn()
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
            actor.GetProperty().EdgeVisibilityOn()
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
