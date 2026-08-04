# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE.
"""SceneRenderingMixin implementation for the CAD viewer."""

from __future__ import annotations

import logging

import numpy as np
import vtk
from PySide6 import QtCore

from .navigation_cube import NavCubeWidget
from .viewer_constants import (
    MESH_COMPONENT_INDEX_BASE as _MESH_COMPONENT_INDEX_BASE,
    MESH_COMPONENT_INDEX_STRIDE as _MESH_COMPONENT_INDEX_STRIDE,
    MESH_FEATURE_INDEX_BASE as _MESH_FEATURE_INDEX_BASE,
    MESH_PICKING_MAX_SURFACE_CELLS as _MESH_PICKING_MAX_SURFACE_CELLS,
)

__all__ = ["SceneRenderingMixin"]


class SceneRenderingMixin:
    def showEvent(self, event):
        """Restore the Qt navigation overlay after a parent tab is shown."""
        super().showEvent(event)
        if hasattr(self, "_nav_cube"):
            QtCore.QTimer.singleShot(0, self.ensure_navigation_cube_visible)

    def clear(self):
        """Clear the viewer and release memory."""
        self._clear_highlight_actors()
        self._clear_edge_highlight_actors()
        self._clear_vertex_highlight_actors()
        self._clear_result_extrema()
        self.clear_silhouettes()
        self._clear_hover_actor()

        # Remove wireframe edge actor
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
        self._picked_occ_vertices = []
        self._picked_vertex_indices = []

        # Clear BC overlay actors
        for actor in list(self._bc_overlay_actors):
            self.renderer.RemoveActor(actor)
            self.bc_renderer.RemoveActor(actor)
            try:
                self.renderer.RemoveActor2D(actor)
                self.bc_renderer.RemoveActor2D(actor)
            except Exception:
                logging.getLogger(__name__).debug(
                    "Optional UI operation failed.", exc_info=True
                )
        self._bc_overlay_actors = []
        # Reset cached BC data so overlays don't auto-replay on wrong shapes
        self._cached_bc_data = None

        # Remove the crash rigid-wall overlay
        self._clear_crash_wall_visuals()
        self._crash_wall_info = None

        # Clear legacy single actor if present
        self.forget_section_state()
        if self.current_actor:
            self.renderer.RemoveActor(self.current_actor)
            if self.current_actor.GetMapper():
                self.current_actor.GetMapper().RemoveAllInputConnections(0)
            self.current_actor = None

        # Clear all actors in the list (excluding bc_overlay which were
        # already removed above to avoid double-removal).
        # NB: vtkBillboardTextActor3D and other text actors do not have a
        #     GetMapper() method, so guard the mapper teardown with hasattr.
        for actor in list(self.actors):
            if getattr(actor, "_bc_overlay", False):
                continue
            self.renderer.RemoveActor(actor)
            get_mapper = getattr(actor, "GetMapper", None)
            if callable(get_mapper):
                mapper = get_mapper()
                if mapper is not None:
                    try:
                        mapper.RemoveAllInputConnections(0)
                    except Exception:
                        logging.getLogger(__name__).debug(
                            "Optional UI operation failed.", exc_info=True
                        )
        self.actors = []
        self._topopt_joint_actors = []
        self._undeformed_actor = None
        self._last_simulation_mesh = None

        self.scalar_bar.VisibilityOff()
        self.vtkWidget.GetRenderWindow().Render()

    def clear_cached_results(self):
        """Drop result/playback data that could repopulate an empty scene."""
        self._last_result_data = None
        self._active_result_mode = None
        self._last_simulation_mesh = None
        self._contour_bands = None
        self._result_extrema_cache = []

        try:
            self._crash_timer.stop()
        except Exception:
            logging.getLogger(__name__).debug(
                "Could not stop crash playback while clearing the viewer.",
                exc_info=True,
            )
        self._crash_frames = []
        self._crash_base_data = None
        self._crash_frame_idx = 0
        self._crash_playing = False
        self._crash_scalar_range = (0.0, 1.0)
        self._crash_wall_info = None

        for name in ("_field_label", "_field_combo", "_scale_label", "_scale_combo"):
            widget = getattr(self, name, None)
            if widget is not None:
                self._set_toolbar_widget_visible(widget, False)

        field_combo = getattr(self, "_field_combo", None)
        if field_combo is not None:
            field_combo.blockSignals(True)
            try:
                field_combo.clear()
            finally:
                field_combo.blockSignals(False)

        for name in ("_btn_extrema", "_btn_undeformed"):
            button = getattr(self, name, None)
            if button is not None:
                button.blockSignals(True)
                try:
                    button.setChecked(False)
                finally:
                    button.blockSignals(False)
        self._show_extrema = False
        self._show_undeformed = False

        panel = getattr(self, "_crash_panel", None)
        if panel is not None:
            panel.hide()
        play_button = getattr(self, "_play_btn", None)
        if play_button is not None:
            play_button.setText("\u25b6")
        frame_label = getattr(self, "_crash_frame_lbl", None)
        if frame_label is not None:
            frame_label.setText("0 / 0")
        self._position_nav_cube()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, "_nav_cube"):
            self._position_nav_cube()

    def _position_nav_cube(self):
        """Keep the NavCube anchored to the bottom-left of the VTK area."""
        margin = 8
        playback_clearance = self._bottom_overlay_clearance(margin)
        y = self.height() - NavCubeWidget.SIZE - margin - playback_clearance
        y = max(margin, y)
        self._nav_cube.move(margin, y)
        self._nav_cube.raise_()
        self._nav_cube.show()

    def _bottom_overlay_clearance(self, margin):
        if not hasattr(self, "_crash_panel") or not self._crash_panel.isVisible():
            return 0
        panel_height = max(
            self._crash_panel.height(), self._crash_panel.sizeHint().height()
        )
        return panel_height + margin

    def ensure_navigation_cube_visible(self):
        """Synchronize and raise the cube after tab, layout, or native-window changes."""
        if not hasattr(self, "_nav_cube") or not hasattr(self, "vtkWidget"):
            return
        self._position_nav_cube()
        self._nav_cube.update_rotation(self.renderer.GetActiveCamera())
        self._nav_cube.setVisible(True)
        self._nav_cube.raise_()
        self._nav_cube.update()

    def _on_vtk_render(self):
        """Sync NavCube rotation to camera after every VTK render."""
        try:
            self._nav_cube.update_rotation(self.renderer.GetActiveCamera())
            self._nav_cube.raise_()
        except Exception:
            logging.getLogger(__name__).debug(
                "Optional UI operation failed.", exc_info=True
            )
        # First render is the earliest point the OpenGL context can be
        # inspected; shader-dependent features stay off until it passes.
        try:
            self._on_first_render()
        except Exception:
            logging.getLogger(__name__).debug(
                "Optional UI operation failed.", exc_info=True
            )

    def _toggle_grid(self, state):
        """Toggle grid visibility."""
        if state:
            if not self._grid_actor:
                self._grid_actor, self._axes_actor, self._axis_label_actors = (
                    self._build_grid_actors()
                )
            self.renderer.AddActor(self._grid_actor)
            self.renderer.AddActor(self._axes_actor)
            for actor in self._axis_label_actors:
                self.renderer.AddActor(actor)
        else:
            if self._grid_actor:
                self.renderer.RemoveActor(self._grid_actor)
                self.renderer.RemoveActor(self._axes_actor)
                for actor in self._axis_label_actors:
                    self.renderer.RemoveActor(actor)
        self.vtkWidget.GetRenderWindow().Render()

    def _build_edge_actor(self, topo_shape, *, initial_visibility=None):
        """
        Tessellate every OCC edge of *topo_shape* into a combined line actor.

        The actor is added to the renderer immediately:
         - invisible by default (shown only when _show_edges or edge picking is active)
         - not pickable by default (edge picking turns it on temporarily)
        Per-edge vtkPolyData objects are cached in self._edge_pd_list for
        highlight rendering; self._edge_cell_map maps VTK cell ids to edge indices.
        """
        occ_edges = []
        try:
            if hasattr(topo_shape, "Edges"):
                occ_edges = topo_shape.Edges()
        except Exception:
            logging.getLogger(__name__).debug(
                "Optional UI operation failed.", exc_info=True
            )

        self._all_occ_edges = occ_edges
        self._edge_pd_list = []
        self._edge_cell_map = {}

        if not occ_edges:
            return

        combined_pts = vtk.vtkPoints()
        combined_lines = vtk.vtkCellArray()
        cell_idx = 0

        for edge_idx, occ_edge in enumerate(occ_edges):
            pts_list = []
            try:
                if hasattr(occ_edge, "wrapped"):
                    from OCP.BRepAdaptor import BRepAdaptor_Curve
                    from OCP.GCPnts import GCPnts_UniformAbscissa

                    curve = BRepAdaptor_Curve(occ_edge.wrapped)
                    sampler = GCPnts_UniformAbscissa(curve, 24)
                    if sampler.NbPoints() > 0:
                        for i in range(1, sampler.NbPoints() + 1):
                            p = curve.Value(sampler.Parameter(i))
                            pts_list.append((p.X(), p.Y(), p.Z()))
            except Exception:
                logging.getLogger(__name__).debug(
                    "Optional UI operation failed.", exc_info=True
                )

            if not pts_list:
                try:
                    sp = occ_edge.startPoint()
                    ep = occ_edge.endPoint()
                    pts_list = [(sp.x, sp.y, sp.z), (ep.x, ep.y, ep.z)]
                except Exception:
                    logging.getLogger(__name__).debug(
                        "Optional UI operation failed.", exc_info=True
                    )

            if len(pts_list) < 2:
                self._edge_pd_list.append(None)
                continue

            # Per-edge polydata (for highlight rendering)
            edge_pts = vtk.vtkPoints()
            edge_lines = vtk.vtkCellArray()
            local_base = combined_pts.GetNumberOfPoints()
            for pt in pts_list:
                combined_pts.InsertNextPoint(pt[0], pt[1], pt[2])
                edge_pts.InsertNextPoint(pt[0], pt[1], pt[2])
            for i in range(len(pts_list) - 1):
                combined_lines.InsertNextCell(2)
                combined_lines.InsertCellPoint(local_base + i)
                combined_lines.InsertCellPoint(local_base + i + 1)
                self._edge_cell_map[cell_idx] = edge_idx
                cell_idx += 1

                edge_lines.InsertNextCell(2)
                edge_lines.InsertCellPoint(i)
                edge_lines.InsertCellPoint(i + 1)

            edge_pd = vtk.vtkPolyData()
            edge_pd.SetPoints(edge_pts)
            edge_pd.SetLines(edge_lines)
            self._edge_pd_list.append(edge_pd)

        if combined_pts.GetNumberOfPoints() == 0:
            return

        pd = vtk.vtkPolyData()
        pd.SetPoints(combined_pts)
        pd.SetLines(combined_lines)

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(pd)
        mapper.ScalarVisibilityOff()

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(0.12, 0.12, 0.12)
        actor.GetProperty().SetLineWidth(1.5)
        actor.GetProperty().LightingOff()
        actor.SetPickable(0)
        visible = (
            self._show_edges if initial_visibility is None else bool(initial_visibility)
        )
        actor.SetVisibility(1 if visible else 0)

        self.renderer.AddActor(actor)
        self._edge_actor = actor

    def _build_vertex_actor(self, topo_shape):
        """Build a point actor whose point ids match OCC vertex indices."""
        try:
            occ_vertices = (
                list(topo_shape.Vertices()) if hasattr(topo_shape, "Vertices") else []
            )
        except Exception:
            occ_vertices = []
        self._all_occ_vertices = occ_vertices
        if not occ_vertices:
            return

        points = vtk.vtkPoints()
        verts = vtk.vtkCellArray()
        kept_vertices = []
        for vertex in occ_vertices:
            try:
                center = vertex.Center()
                point = (float(center.x), float(center.y), float(center.z))
            except Exception:
                try:
                    point = (
                        float(vertex.X),
                        float(vertex.Y),
                        float(vertex.Z),
                    )
                except Exception:
                    continue
            point_id = points.InsertNextPoint(*point)
            verts.InsertNextCell(1)
            verts.InsertCellPoint(point_id)
            kept_vertices.append(vertex)

        self._all_occ_vertices = kept_vertices
        if points.GetNumberOfPoints() == 0:
            return
        poly_data = vtk.vtkPolyData()
        poly_data.SetPoints(points)
        poly_data.SetVerts(verts)
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(poly_data)
        mapper.ScalarVisibilityOff()
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(0.0, 0.9, 1.0)
        actor.GetProperty().SetPointSize(11.0)
        actor.GetProperty().LightingOff()
        try:
            actor.GetProperty().RenderPointsAsSpheresOn()
        except AttributeError:
            pass
        actor.SetPickable(0)
        actor.SetVisibility(0)
        self.renderer.AddActor(actor)
        self._vertex_actor = actor

    def _build_grid_actors(self):
        """Creates a 3-plane (XY/XZ/YZ) reference grid and thick main axes for spatial reference."""
        size = 500.0  # Big enough to feel infinite
        step = 10.0

        pts = vtk.vtkPoints()
        lines = vtk.vtkCellArray()
        colors = vtk.vtkUnsignedCharArray()
        colors.SetNumberOfComponents(3)
        colors.SetName("Colors")

        idx = 0

        def add_line(p0, p1, r, g, b):
            nonlocal idx
            pts.InsertNextPoint(*p0)
            pts.InsertNextPoint(*p1)
            lines.InsertNextCell(2)
            lines.InsertCellPoint(idx)
            lines.InsertCellPoint(idx + 1)
            colors.InsertNextTuple3(r, g, b)
            idx += 2

        ticks = np.arange(-size, size + step, step)

        # XY plane (z=0) — neutral gray
        for v in ticks:
            add_line((v, -size, 0), (v, size, 0), 58, 58, 58)
            add_line((-size, v, 0), (size, v, 0), 58, 58, 58)

        # XZ plane (y=0) — faint blue tint
        for v in ticks:
            add_line((v, 0, -size), (v, 0, size), 48, 48, 65)
            add_line((-size, 0, v), (size, 0, v), 48, 48, 65)

        # YZ plane (x=0) — faint red tint
        for v in ticks:
            add_line((0, v, -size), (0, v, size), 65, 48, 48)
            add_line((0, -size, v), (0, size, v), 65, 48, 48)

        grid_pd = vtk.vtkPolyData()
        grid_pd.SetPoints(pts)
        grid_pd.SetLines(lines)
        grid_pd.GetCellData().SetScalars(colors)

        grid_mapper = vtk.vtkPolyDataMapper()
        grid_mapper.SetInputData(grid_pd)
        grid_mapper.SetColorModeToDirectScalars()
        grid_actor = vtk.vtkActor()
        grid_actor.SetMapper(grid_mapper)
        grid_actor.GetProperty().SetLineWidth(1.0)
        grid_actor.GetProperty().LightingOff()
        grid_actor.SetPickable(0)
        grid_actor.UseBoundsOff()  # exclude from ResetCamera bounds

        # Thick Central Axes Crosshair
        axes_pd = vtk.vtkPolyData()
        axes_pts = vtk.vtkPoints()
        axes_lines = vtk.vtkCellArray()
        axes_colors = vtk.vtkUnsignedCharArray()
        axes_colors.SetNumberOfComponents(3)

        # X Red
        axes_pts.InsertNextPoint(-size, 0, 0)
        axes_pts.InsertNextPoint(size, 0, 0)
        axes_lines.InsertNextCell(2, [0, 1])
        axes_colors.InsertNextTuple3(255, 50, 50)

        # Y Green
        axes_pts.InsertNextPoint(0, -size, 0)
        axes_pts.InsertNextPoint(0, size, 0)
        axes_lines.InsertNextCell(2, [2, 3])
        axes_colors.InsertNextTuple3(50, 255, 50)

        # Z Blue
        axes_pts.InsertNextPoint(0, 0, -size)
        axes_pts.InsertNextPoint(0, 0, size)
        axes_lines.InsertNextCell(2, [4, 5])
        axes_colors.InsertNextTuple3(50, 50, 255)

        axes_pd.SetPoints(axes_pts)
        axes_pd.SetLines(axes_lines)
        axes_pd.GetCellData().SetScalars(axes_colors)

        axes_mapper = vtk.vtkPolyDataMapper()
        axes_mapper.SetInputData(axes_pd)
        axes_mapper.SetColorModeToDirectScalars()
        axes_actor = vtk.vtkActor()
        axes_actor.SetMapper(axes_mapper)
        axes_actor.GetProperty().SetLineWidth(2.5)  # Thicker than grid
        axes_actor.GetProperty().LightingOff()
        axes_actor.SetPickable(0)
        axes_actor.UseBoundsOff()  # exclude from ResetCamera bounds

        axis_label_actors = [
            self._build_axis_label_actor("X", (size * 1.04, 0, 0), (1.0, 0.24, 0.20)),
            self._build_axis_label_actor("Y", (0, size * 1.04, 0), (0.24, 0.95, 0.35)),
            self._build_axis_label_actor("Z", (0, 0, size * 1.04), (0.35, 0.55, 1.0)),
        ]

        return grid_actor, axes_actor, axis_label_actors

    def _build_axis_label_actor(self, label, position, color):
        actor = vtk.vtkBillboardTextActor3D()
        actor.SetInput(label)
        actor.SetPosition(*position)
        actor.SetPickable(0)
        prop = actor.GetTextProperty()
        prop.SetColor(*color)
        prop.SetFontSize(24)
        prop.BoldOn()
        prop.SetJustificationToCentered()
        prop.SetVerticalJustificationToCentered()
        try:
            actor.UseBoundsOff()
        except AttributeError:
            pass
        return actor

    def _set_camera_view(self, position, view_up):
        """Sets the camera to look at the focal point from the given relative direction."""
        camera = self.renderer.GetActiveCamera()

        # Normalize the position direction vector so distance is consistent
        pos = np.array(position, dtype=float)
        mag = np.linalg.norm(pos)
        if mag > 1e-10:
            pos = pos / mag

        # Compute focal point and distance from visible geometry bounds
        bounds = self.renderer.ComputeVisiblePropBounds()
        if bounds and len(bounds) == 6 and bounds[0] <= bounds[1]:
            dx = bounds[1] - bounds[0]
            dy = bounds[3] - bounds[2]
            dz = bounds[5] - bounds[4]
            max_dim = max(dx, dy, dz, 1e-3)
            distance = max_dim * 2.5
            focal_point = (
                (bounds[0] + bounds[1]) / 2.0,
                (bounds[2] + bounds[3]) / 2.0,
                (bounds[4] + bounds[5]) / 2.0,
            )
        else:
            distance = camera.GetDistance()
            focal_point = camera.GetFocalPoint()

        # Every standard view is framed to fit, which ResetCamera does properly
        # for the current aspect ratio and view angle — the bounds estimate
        # above does not.  Pose the camera, let ResetCamera solve the distance,
        # then rewind and animate to that solved target so the eased motion
        # lands exactly where the old instant jump did.
        start_pose = (
            camera.GetPosition(),
            camera.GetFocalPoint(),
            camera.GetViewUp(),
        )
        try:
            camera.SetFocalPoint(*focal_point)
            camera.SetPosition(*(np.asarray(focal_point) + pos * distance))
            camera.SetViewUp(*view_up)
            self.renderer.ResetCamera()
            focal_point = camera.GetFocalPoint()
            distance = camera.GetDistance()
        finally:
            camera.SetPosition(*start_pose[0])
            camera.SetFocalPoint(*start_pose[1])
            camera.SetViewUp(*start_pose[2])

        # Ease into the standard view rather than cutting to it — a hard snap
        # loses which way the part turned. _animate_camera_to keeps the
        # NavCube in sync on every frame and applies the pose directly when
        # animation is switched off.
        self._animate_camera_to(pos, focal_point, view_up, distance)

    def _roll_camera(self, angle):
        """Animate a VTK-consistent roll around the current viewing axis."""
        camera = self.renderer.GetActiveCamera()
        focal = np.asarray(camera.GetFocalPoint(), dtype=float)
        direction = np.asarray(camera.GetPosition(), dtype=float) - focal
        distance = float(np.linalg.norm(direction))
        if distance <= 1e-12:
            return
        direction /= distance

        start_up = tuple(camera.GetViewUp())
        camera.Roll(angle)
        camera.OrthogonalizeViewUp()
        target_up = tuple(camera.GetViewUp())
        camera.SetViewUp(*start_up)
        self._animate_camera_to(direction, focal, target_up, distance)

    # Control-point ramps in RGB.  The rainbow entry reproduces the
    # blue→cyan→green→yellow→red convention FE post-processors use, but as
    # explicit stops rather than a raw HSV hue sweep — sweeping hue at full
    # saturation spends most of its range in cyan/green and compresses the
    # high end, which is what makes the stock VTK ramp look harsh.
    _RESULT_COLOR_STOPS = (
        (0.00, (0.00, 0.00, 0.60)),
        (0.25, (0.00, 0.70, 0.95)),
        (0.50, (0.20, 0.85, 0.25)),
        (0.75, (0.98, 0.85, 0.10)),
        (1.00, (0.70, 0.02, 0.02)),
    )

    def _result_lut(self):
        """Build the fixed classic FEA spectrum, banded if requested.

        FE post-processors default to a *banded* legend rather than a smooth
        ramp: discrete levels let a value be read straight off the plot and
        turn colour boundaries into contour lines.  ``_contour_bands`` is set
        by the viewer's "Bands" combo; ``None`` restores a continuous
        256-entry ramp.
        """
        bands = getattr(self, "_contour_bands", None)
        count = int(bands) if bands else 256
        count = max(2, count)
        lut = vtk.vtkLookupTable()
        lut.SetNumberOfTableValues(count)
        positions = np.array([p for p, _ in self._RESULT_COLOR_STOPS], dtype=float)
        colors = np.array([c for _, c in self._RESULT_COLOR_STOPS], dtype=float)
        # Band i spans [i/count, (i+1)/count); sampling its centre keeps the
        # extreme bands from collapsing onto the ramp end points.
        samples = (np.arange(count) + 0.5) / count if bands else np.linspace(
            0.0, 1.0, count
        )
        for index, t in enumerate(samples):
            rgb = [np.interp(t, positions, colors[:, channel]) for channel in range(3)]
            lut.SetTableValue(index, rgb[0], rgb[1], rgb[2], 1.0)
        # Values outside the mapped range are drawn in grey rather than being
        # clamped into the extreme band, so a clipped legend is visible as
        # clipped instead of reading as a genuine maximum.
        try:
            lut.SetNanColor(0.55, 0.55, 0.58, 1.0)
            lut.SetBelowRangeColor(0.35, 0.35, 0.38, 1.0)
            lut.SetAboveRangeColor(0.85, 0.35, 0.85, 1.0)
        except Exception:
            logging.getLogger(__name__).debug(
                "Out-of-range legend colours unavailable.", exc_info=True
            )
        lut.Build()
        return lut

    def _style_scalar_bar(self):
        """Give the VTK colour legend a flat, modern look.

        VTK's default scalar bar uses a bold *italic* drop-shadowed font that
        looks dated (the serif-ish numbers).  This switches both the title and
        the tick labels to a clean, non-italic, shadow-free sans-serif, slims
        the colour strip, and tightens the number formatting.  Wrapped in
        try/except per call so it degrades gracefully on older VTK builds.
        """
        bar = self.scalar_bar
        # With banded contours the legend should tick every band boundary, so
        # a colour read off the model maps to an exact interval. Capped at 13
        # so a 20-band legend stays readable.
        bands = getattr(self, "_contour_bands", None)
        n_labels = min(int(bands) + 1, 13) if bands else 6
        for setter in (
            lambda: bar.SetNumberOfLabels(n_labels),
            lambda: bar.SetLabelFormat("%.3g"),
            lambda: bar.SetUnconstrainedFontSize(True),
            lambda: bar.SetBarRatio(0.22),  # slim colour strip, room for labels
            lambda: bar.SetTextPad(6),
            lambda: bar.DrawFrameOff(),
            lambda: bar.DrawBackgroundOff(),
        ):
            try:
                setter()
            except Exception:
                logging.getLogger(__name__).debug(
                    "Optional UI operation failed.", exc_info=True
                )

        title_tp = bar.GetTitleTextProperty()
        label_tp = bar.GetLabelTextProperty()
        for tp in (title_tp, label_tp):
            try:
                tp.SetFontFamilyToArial()
                tp.ItalicOff()
                tp.ShadowOff()
                tp.SetColor(*getattr(self, "_scalar_text_color", (0.90, 0.92, 0.96)))
            except Exception:
                logging.getLogger(__name__).debug(
                    "Optional UI operation failed.", exc_info=True
                )
        try:
            title_tp.BoldOn()
            title_tp.SetFontSize(12)
            label_tp.BoldOff()
            label_tp.SetFontSize(11)
            label_tp.SetColor(*getattr(self, "_scalar_text_color", (0.78, 0.82, 0.88)))
        except Exception:
            logging.getLogger(__name__).debug(
                "Optional UI operation failed.", exc_info=True
            )

    def _update_scalar_bar(self, title, min_val, max_val, lut=None):
        """Update and show the scalar bar."""
        if not title:
            self.scalar_bar.VisibilityOff()
            return

        compact_titles = {
            "Von Mises Stress (MPa)": "Von Mises Stress\n(MPa)",
            "Equivalent Plastic Strain": "Plastic Strain",
            "Element Failure (0=intact, 1=failed)": "Element Failure",
        }
        self.scalar_bar.SetTitle(compact_titles.get(str(title), str(title)))

        if lut:
            self.scalar_bar.SetLookupTable(lut)
        elif self.current_actor and self.current_actor.GetMapper():
            self.scalar_bar.SetLookupTable(
                self.current_actor.GetMapper().GetLookupTable()
            )

        # Re-assert the flat font/format styling — a freshly assigned lookup
        # table can otherwise leave the actor on VTK's italic defaults.
        self._style_scalar_bar()
        self.scalar_bar.VisibilityOn()

    def render_sketch(self, sketch):
        """
        Render a 2D sketch (CadQuery Workplane with 2D geometry) as polylines.
        Works with sketches that have wires but no 3D solid.
        """
        if sketch is None:
            return

        self.clear()

        # Try to extract edges from the sketch
        edges = []
        try:
            # Method 1: CadQuery Workplane with pending wires
            if hasattr(sketch, "ctx") and hasattr(sketch.ctx, "pendingWires"):
                try:
                    wires = sketch.ctx.pendingWires
                    if wires:
                        for wire in wires:
                            if hasattr(wire, "Edges"):
                                edges.extend(wire.Edges())
                except Exception:
                    logging.getLogger(__name__).debug(
                        "Optional UI operation failed.", exc_info=True
                    )

            # Method 2: Try to get edges directly from the unwrapped shape
            if not edges:
                shape = sketch
                if hasattr(sketch, "val"):
                    try:
                        shape = sketch.val()
                    except Exception:
                        logging.getLogger(__name__).debug(
                            "Optional UI operation failed.", exc_info=True
                        )

                if hasattr(shape, "Edges"):
                    try:
                        edge_list = shape.Edges()
                        if edge_list:
                            edges = edge_list
                    except Exception:
                        logging.getLogger(__name__).debug(
                            "Optional UI operation failed.", exc_info=True
                        )

            # Method 3: Try the CadQuery .edges() API
            if not edges and hasattr(sketch, "edges"):
                try:
                    edge_objects = sketch.edges().vals()
                    if edge_objects:
                        edges = edge_objects
                except Exception:
                    logging.getLogger(__name__).debug(
                        "Optional UI operation failed.", exc_info=True
                    )

            # Method 4: Check for wires on the shape
            if not edges:
                shape = sketch
                if hasattr(sketch, "val"):
                    try:
                        shape = sketch.val()
                    except Exception:
                        logging.getLogger(__name__).debug(
                            "Optional UI operation failed.", exc_info=True
                        )

                if hasattr(shape, "Wires"):
                    try:
                        wires = shape.Wires()
                        for wire in wires:
                            if hasattr(wire, "Edges"):
                                edges.extend(wire.Edges())
                    except Exception:
                        logging.getLogger(__name__).debug(
                            "Optional UI operation failed.", exc_info=True
                        )

            # Method 5: Try _edges attribute (fallback)
            if not edges and hasattr(sketch, "_edges"):
                edges = sketch._edges

        except Exception:
            return

        if not edges:
            return

        # Create VTK points and lines
        points = vtk.vtkPoints()
        lines = vtk.vtkCellArray()

        point_id = 0

        for edge in edges:
            try:
                pts_extracted = []

                # Method 1: Use OCCT curve sampling (best for circles, arcs, splines)
                if hasattr(edge, "wrapped"):
                    try:
                        from OCP.BRepAdaptor import BRepAdaptor_Curve
                        from OCP.GCPnts import GCPnts_UniformAbscissa

                        curve = BRepAdaptor_Curve(edge.wrapped)
                        sampler = GCPnts_UniformAbscissa(curve, 30)

                        if sampler.NbPoints() > 0:
                            for i in range(1, sampler.NbPoints() + 1):
                                p = curve.Value(sampler.Parameter(i))
                                pts_extracted.append((p.X(), p.Y(), p.Z()))
                    except Exception:
                        logging.getLogger(__name__).debug(
                            "Optional UI operation failed.", exc_info=True
                        )

                # Method 2: CadQuery edge discretization
                if not pts_extracted and hasattr(edge, "discretize"):
                    try:
                        pts_extracted = edge.discretize(30)
                    except Exception:
                        logging.getLogger(__name__).debug(
                            "Optional UI operation failed.", exc_info=True
                        )

                # Method 3: Simple start/end for straight lines
                if (
                    not pts_extracted
                    and hasattr(edge, "startPoint")
                    and hasattr(edge, "endPoint")
                ):
                    sp = edge.startPoint()
                    ep = edge.endPoint()
                    if (
                        abs(sp.x - ep.x) > 1e-6
                        or abs(sp.y - ep.y) > 1e-6
                        or abs(sp.z - ep.z) > 1e-6
                    ):
                        pts_extracted = [(sp.x, sp.y, sp.z), (ep.x, ep.y, ep.z)]

                if pts_extracted and len(pts_extracted) >= 2:
                    start_id = point_id
                    for pt in pts_extracted:
                        if hasattr(pt, "x"):
                            points.InsertNextPoint(
                                pt.x, pt.y, pt.z if hasattr(pt, "z") else 0
                            )
                        else:
                            points.InsertNextPoint(
                                pt[0], pt[1], pt[2] if len(pt) > 2 else 0
                            )
                        point_id += 1

                    for i in range(start_id, point_id - 1):
                        line = vtk.vtkLine()
                        line.GetPointIds().SetId(0, i)
                        line.GetPointIds().SetId(1, i + 1)
                        lines.InsertNextCell(line)

            except Exception:
                continue

        if point_id == 0:
            return

        poly_data = vtk.vtkPolyData()
        poly_data.SetPoints(points)
        poly_data.SetLines(lines)

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(poly_data)

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(0.0, 0.9, 0.9)  # Cyan
        actor.GetProperty().SetLineWidth(2.0)

        self.renderer.AddActor(actor)
        self.current_actor = actor
        self.actors.append(actor)

        camera = self.renderer.GetActiveCamera()
        camera.SetPosition(0, 0, 100)
        camera.SetFocalPoint(0, 0, 0)
        camera.SetViewUp(0, 1, 0)

        self.renderer.ResetCamera()
        self.vtkWidget.GetRenderWindow().Render()

    def render_shape(
        self,
        shape,
        *,
        initial_edge_visibility=None,
        smooth_across_faces=False,
        material_color=None,
    ):
        """
        Accepts a CadQuery Workplane or Shape, tessellates it, and renders it.
        Also builds the per-face mapping required for interactive picking.
        """
        self.scalar_bar.VisibilityOff()
        if shape is None:
            return

        self.clear()

        # Reset picking data from previous shape
        self._face_map = {}
        self._all_occ_faces = []
        self._face_polydata_list = []
        self._pickable_surface_dataset = None

        topo_shape = shape
        if isinstance(topo_shape, dict):
            if "shape" in topo_shape:
                topo_shape = topo_shape["shape"]
            elif "components" in topo_shape:
                topo_shape = topo_shape["components"]

        if hasattr(shape, "toCompound"):
            try:
                topo_shape = shape.toCompound()
            except Exception:
                return

        try:
            if hasattr(topo_shape, "val"):
                topo_shape = topo_shape.val()
        except Exception:
            logging.getLogger(__name__).debug(
                "Optional UI operation failed.", exc_info=True
            )

        if not hasattr(topo_shape, "tessellate"):
            if isinstance(topo_shape, (list, tuple)) and topo_shape:
                def _flatten(items):
                    res = []
                    for it in items:
                        if isinstance(it, (list, tuple)):
                            res.extend(_flatten(it))
                        elif it is not None:
                            res.append(it)
                    return res
                all_items = _flatten(topo_shape)
                valid_items = []
                for item in all_items:
                    v = item.val() if hasattr(item, "val") else item
                    if hasattr(v, "tessellate") or hasattr(v, "wrapped"):
                        valid_items.append(v)
                if valid_items:
                    if len(valid_items) == 1:
                        topo_shape = valid_items[0]
                    else:
                        try:
                            import cadquery as cq
                            cq_items = []
                            for s in valid_items:
                                if hasattr(s, "wrapped"):
                                    cq_items.append(s)
                                else:
                                    try:
                                        cq_items.append(cq.Shape.cast(s))
                                    except Exception:
                                        pass
                            if cq_items:
                                topo_shape = cq.Compound.makeCompound(cq_items)
                            else:
                                topo_shape = valid_items[0]
                        except Exception:
                            topo_shape = valid_items[0]

        if not hasattr(topo_shape, "tessellate") and hasattr(topo_shape, "objects"):
            try:
                for obj in getattr(topo_shape, "objects", []):
                    if hasattr(obj, "val"):
                        v = obj.val()
                        if hasattr(v, "tessellate"):
                            topo_shape = v
                            break
                    if hasattr(obj, "tessellate"):
                        topo_shape = obj
                        break
            except Exception:
                logging.getLogger(__name__).debug(
                    "Optional UI operation failed.", exc_info=True
                )

        if not hasattr(topo_shape, "tessellate"):
            self.render_sketch(shape)
            return

        # ── Build per-face tessellation for picking ──
        # Try to get OCC faces list for picking
        occ_faces = []
        try:
            if hasattr(topo_shape, "Faces"):
                occ_faces = topo_shape.Faces()
        except Exception:
            logging.getLogger(__name__).debug(
                "Optional UI operation failed.", exc_info=True
            )

        self._all_occ_faces = occ_faces

        # Build combined poly data for render + per-face polydata for picking
        combined_points = vtk.vtkPoints()
        combined_polys = vtk.vtkCellArray()
        global_cell_idx = 0
        self._face_polydata_list = []

        # We'll try per-face tessellation first (enables picking), fall back to
        # whole-shape tessellation if per-face isn't available.
        use_per_face = len(occ_faces) > 0

        if use_per_face:
            for face_idx, occ_face in enumerate(occ_faces):
                face_pd = vtk.vtkPolyData()
                face_points = vtk.vtkPoints()
                face_polys = vtk.vtkCellArray()

                try:
                    tri = occ_face.tessellate(tolerance=0.1, angularTolerance=0.2)
                    if isinstance(tri, dict):
                        verts = tri.get("vertices") or tri.get("verts") or []
                        triangles = tri.get("triangles") or tri.get("faces") or []
                    else:
                        verts, triangles = tri[0], tri[1]

                    if not verts:
                        self._face_polydata_list.append(None)
                        continue

                    pt_offset = combined_points.GetNumberOfPoints()
                    local_pt_offset = face_points.GetNumberOfPoints()

                    for v in verts:
                        combined_points.InsertNextPoint(v.x, v.y, v.z)
                        face_points.InsertNextPoint(v.x, v.y, v.z)

                    for t in triangles:
                        # Combined mesh
                        combined_polys.InsertNextCell(3)
                        combined_polys.InsertCellPoint(t[0] + pt_offset)
                        combined_polys.InsertCellPoint(t[1] + pt_offset)
                        combined_polys.InsertCellPoint(t[2] + pt_offset)
                        self._face_map[global_cell_idx] = face_idx
                        global_cell_idx += 1

                        # Per-face mesh
                        face_polys.InsertNextCell(3)
                        face_polys.InsertCellPoint(t[0] + local_pt_offset)
                        face_polys.InsertCellPoint(t[1] + local_pt_offset)
                        face_polys.InsertCellPoint(t[2] + local_pt_offset)

                    face_pd.SetPoints(face_points)
                    face_pd.SetPolys(face_polys)
                    self._face_polydata_list.append(face_pd)

                except Exception:
                    self._face_polydata_list.append(None)
                    continue

            poly_data = vtk.vtkPolyData()
            poly_data.SetPoints(combined_points)
            poly_data.SetPolys(combined_polys)

        else:
            # Fallback: whole-shape tessellation (no per-face picking)
            try:
                triangulation = topo_shape.tessellate(
                    tolerance=0.1, angularTolerance=0.2
                )
                if isinstance(triangulation, dict):
                    verts = triangulation.get("vertices") or triangulation.get("verts")
                    triangles = triangulation.get("triangles") or triangulation.get(
                        "faces"
                    )
                else:
                    verts, triangles = triangulation[0], triangulation[1]
            except Exception:
                self.render_sketch(shape)
                return

            if not verts or len(verts) == 0:
                self.render_sketch(shape)
                return

            points = vtk.vtkPoints()
            polys = vtk.vtkCellArray()

            for v in verts:
                points.InsertNextPoint(v.x, v.y, v.z)
            for t in triangles:
                polys.InsertNextCell(3)
                polys.InsertCellPoint(t[0])
                polys.InsertCellPoint(t[1])
                polys.InsertCellPoint(t[2])

            poly_data = vtk.vtkPolyData()
            poly_data.SetPoints(points)
            poly_data.SetPolys(polys)

        if combined_points.GetNumberOfPoints() == 0 and not use_per_face:
            self.render_sketch(shape)
            return

        # Calculate normals for smooth shading. OCC tessellates each B-rep face
        # independently, so coincident vertices on a subdivision-patch seam are
        # duplicated in ``poly_data``. Welding only for the render pipeline lets
        # VTK average those normals while the original cell/face mapping remains
        # intact for CAD picking. Splitting at a real 50-degree feature retains
        # the crease lines the subdivision reconstruction deliberately marked.
        normals = vtk.vtkPolyDataNormals()
        if smooth_across_faces:
            clean = vtk.vtkCleanPolyData()
            clean.SetInputData(poly_data)
            clean.ToleranceIsAbsoluteOff()
            clean.SetTolerance(1e-8)
            normals.SetInputConnection(clean.GetOutputPort())
            normals.ConsistencyOn()
            normals.AutoOrientNormalsOn()
            normals.SplittingOn()
            normals.SetFeatureAngle(50.0)
        else:
            normals.SetInputData(poly_data)
        normals.Update()

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(normals.GetOutputPort())

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        if material_color is None:
            self.apply_cad_material(actor)
        else:
            self.apply_cad_material(actor, color=material_color)

        self.renderer.AddActor(actor)
        self.current_actor = actor
        if self._section_enabled:
            self._update_section_cut()

        # Build wireframe edge actor from OCC edges (used by Edges toggle + edge picking)
        self._build_edge_actor(
            topo_shape,
            initial_visibility=initial_edge_visibility,
        )
        self._build_vertex_actor(topo_shape)

        self.renderer.ResetCamera()
        # Ambient occlusion is scaled in world units, so it has to be re-fitted
        # whenever the model that fills the view changes size.
        self._update_scene_scale()
        self._refresh_render_quality()
        self.vtkWidget.GetRenderWindow().Render()

    def _polydata_from_cells(self, dataset, cell_ids):
        """Build standalone polydata from selected cells in a VTK dataset."""
        if dataset is None or not cell_ids:
            return None
        try:
            pts = vtk.vtkPoints()
            polys = vtk.vtkCellArray()
            point_map = {}
            for cell_id in cell_ids:
                cell = dataset.GetCell(int(cell_id))
                if cell is None or cell.GetNumberOfPoints() < 3:
                    continue
                local_ids = []
                for i in range(cell.GetNumberOfPoints()):
                    pid = int(cell.GetPointId(i))
                    if pid not in point_map:
                        point_map[pid] = pts.InsertNextPoint(dataset.GetPoint(pid))
                    local_ids.append(point_map[pid])
                polys.InsertNextCell(len(local_ids))
                for pid in local_ids:
                    polys.InsertCellPoint(pid)
            if pts.GetNumberOfPoints() == 0 or polys.GetNumberOfCells() == 0:
                return None
            pd = vtk.vtkPolyData()
            pd.SetPoints(pts)
            pd.SetPolys(polys)
            return pd
        except Exception:
            return None

    def _configure_mesh_face_picking_from_surface(self, dataset):
        """Expose feature-bounded curved and planar regions for mesh picking."""
        self._face_map = {}
        self._all_occ_faces = []
        self._face_polydata_list = []
        self._mesh_picking_too_dense = False

        if dataset is None:
            return
        try:
            n_cells = int(dataset.GetNumberOfCells())
            n_points = int(dataset.GetNumberOfPoints())
            if n_cells == 0 or n_points == 0:
                return
            if n_cells > _MESH_PICKING_MAX_SURFACE_CELLS:
                # Splitting hundreds of thousands of triangles into connected
                # pick patches on the GUI thread makes the application appear
                # frozen.  The Properties panel will prepare/render the
                # remeshed upstream geometry instead; if none exists, picking is
                # deliberately unavailable for this dense raw surface.
                self._mesh_picking_too_dense = True
                return

            all_points = np.array(
                [dataset.GetPoint(i) for i in range(n_points)], dtype=float
            )
            mins = np.min(all_points, axis=0)
            maxs = np.max(all_points, axis=0)
            spans = maxs - mins
            body_center = 0.5 * (mins + maxs)

            valid_cell_ids = []
            valid_point_ids = []
            centers = []
            normals = []
            areas = []
            edge_lengths = []
            for cell_id in range(n_cells):
                cell = dataset.GetCell(cell_id)
                if cell is None or cell.GetNumberOfPoints() < 3:
                    continue
                point_ids = [
                    int(cell.GetPointId(i)) for i in range(cell.GetNumberOfPoints())
                ]
                coords = np.array(
                    [dataset.GetPoint(point_id) for point_id in point_ids],
                    dtype=float,
                )
                if coords.shape[0] < 3:
                    continue
                normal = np.cross(coords[1] - coords[0], coords[2] - coords[0])
                norm = float(np.linalg.norm(normal))
                if norm < 1e-12:
                    continue
                normal = normal / norm
                center = np.mean(coords, axis=0)
                if float(np.dot(normal, center - body_center)) < 0.0:
                    normal *= -1.0
                valid_cell_ids.append(cell_id)
                valid_point_ids.append(point_ids)
                centers.append(center)
                normals.append(normal)
                areas.append(0.5 * norm)
                for i in range(coords.shape[0]):
                    edge_lengths.append(
                        float(
                            np.linalg.norm(
                                coords[(i + 1) % coords.shape[0]] - coords[i]
                            )
                        )
                    )

            if not valid_cell_ids:
                return

            valid_cell_ids = np.asarray(valid_cell_ids, dtype=int)
            centers = np.asarray(centers, dtype=float)
            normals = np.asarray(normals, dtype=float)
            areas = np.asarray(areas, dtype=float)
            edge_lengths = np.asarray(
                [v for v in edge_lengths if v > 1e-9], dtype=float
            )
            edge_scale = float(np.median(edge_lengths)) if edge_lengths.size else 1.0

            def _components_for_mask(mask):
                selected = np.where(np.asarray(mask, dtype=bool))[0]
                if selected.size == 0:
                    return []

                edge_to_faces = {}
                for local_idx, valid_idx in enumerate(selected):
                    ids = valid_point_ids[int(valid_idx)]
                    if len(ids) < 3:
                        continue
                    for i in range(len(ids)):
                        edge = tuple(
                            sorted((int(ids[i]), int(ids[(i + 1) % len(ids)])))
                        )
                        edge_to_faces.setdefault(edge, []).append(local_idx)

                adjacency = [set() for _ in range(len(selected))]
                for owners in edge_to_faces.values():
                    if len(owners) < 2:
                        continue
                    for owner in owners:
                        adjacency[owner].update(v for v in owners if v != owner)

                components = []
                visited = np.zeros(len(selected), dtype=bool)
                for start in range(len(selected)):
                    if visited[start]:
                        continue
                    stack = [start]
                    visited[start] = True
                    current = []
                    while stack:
                        item = stack.pop()
                        current.append(item)
                        for nxt in adjacency[item]:
                            if not visited[nxt]:
                                visited[nxt] = True
                                stack.append(nxt)
                    components.append(selected[np.asarray(current, dtype=int)])

                def _component_sort_key(component):
                    comp_area = float(np.sum(areas[component]))
                    if comp_area > 1e-12:
                        comp_center = (
                            np.sum(
                                centers[component] * areas[component, None],
                                axis=0,
                            )
                            / comp_area
                        )
                    else:
                        comp_center = np.mean(centers[component], axis=0)
                    return (
                        -comp_area,
                        float(comp_center[0]),
                        float(comp_center[1]),
                        float(comp_center[2]),
                    )

                components.sort(key=_component_sort_key)
                return components

            directions = (
                ("<X", 0, -1.0),
                (">X", 0, 1.0),
                ("<Y", 1, -1.0),
                (">Y", 1, 1.0),
                ("<Z", 2, -1.0),
                (">Z", 2, 1.0),
            )
            face_cell_scores = {}
            for direction_index, (selector, axis, sign) in enumerate(directions):
                direction = np.zeros(3, dtype=float)
                direction[axis] = sign
                normal_mask = np.dot(normals, direction) >= 0.30
                if np.any(normal_mask):
                    candidate_centers = centers[normal_mask]
                    extreme = (
                        float(np.min(candidate_centers[:, axis]))
                        if sign < 0.0
                        else float(np.max(candidate_centers[:, axis]))
                    )
                    axis_span = float(spans[axis])
                    tol = max(
                        1e-6,
                        0.005 * axis_span,
                        min(3.0 * edge_scale, 0.08 * axis_span)
                        if axis_span > 1e-9
                        else 3.0 * edge_scale,
                    )
                    near_mask = (
                        centers[:, axis] <= extreme + tol
                        if sign < 0.0
                        else centers[:, axis] >= extreme - tol
                    )
                    mask = normal_mask & near_mask
                else:
                    extreme = (
                        float(np.min(centers[:, axis]))
                        if sign < 0.0
                        else float(np.max(centers[:, axis]))
                    )
                    axis_span = float(spans[axis])
                    tol = max(
                        1e-6,
                        0.005 * axis_span,
                        min(3.0 * edge_scale, 0.08 * axis_span)
                        if axis_span > 1e-9
                        else 3.0 * edge_scale,
                    )
                    mask = (
                        centers[:, axis] <= extreme + tol
                        if sign < 0.0
                        else centers[:, axis] >= extreme - tol
                    )

                for component_index, component in enumerate(_components_for_mask(mask)):
                    cell_ids = [int(v) for v in valid_cell_ids[component].tolist()]
                    if not cell_ids:
                        continue
                    face_idx = len(self._all_occ_faces)
                    stored_index = (
                        _MESH_COMPONENT_INDEX_BASE
                        + direction_index * _MESH_COMPONENT_INDEX_STRIDE
                        + component_index
                    )
                    comp_area = float(np.sum(areas[component]))
                    if comp_area > 1e-12:
                        comp_center = (
                            np.sum(
                                centers[component] * areas[component, None],
                                axis=0,
                            )
                            / comp_area
                        )
                    else:
                        comp_center = np.mean(centers[component], axis=0)
                    self._all_occ_faces.append(
                        {
                            "mesh_virtual_face": True,
                            "selector": selector,
                            "component_index": int(component_index),
                            "stored_index": int(stored_index),
                            "label": f"{selector} patch {component_index + 1}",
                            "face_index": face_idx,
                            "center": [float(v) for v in comp_center.tolist()],
                            "area": comp_area,
                        }
                    )
                    self._face_polydata_list.append(
                        self._polydata_from_cells(dataset, cell_ids)
                    )
                    scores = np.dot(normals[component], direction)
                    for local_cell_id, score in zip(cell_ids, scores):
                        previous = face_cell_scores.get(int(local_cell_id), -1e9)
                        if float(score) > previous:
                            face_cell_scores[int(local_cell_id)] = float(score)
                            self._face_map[int(local_cell_id)] = face_idx

            # Directional patches above preserve compatibility with saved
            # projects. New clicks resolve to geometric feature regions, so a
            # cylindrical side, cone, fillet, or freeform smooth face can be
            # selected as one surface instead of only ±X/±Y/±Z end patches.
            from pylcss.design_studio.nodes.selection import (
                _feature_surface_component_indices,
            )

            triangle_ids = np.asarray([ids[:3] for ids in valid_point_ids], dtype=int)
            feature_components = _feature_surface_component_indices(
                all_points,
                triangle_ids,
                normals,
                centers,
                areas,
            )
            for component_index, component in enumerate(feature_components):
                cell_ids = [int(v) for v in valid_cell_ids[component].tolist()]
                if not cell_ids:
                    continue
                face_idx = len(self._all_occ_faces)
                comp_area = float(np.sum(areas[component]))
                comp_center = np.sum(
                    centers[component] * areas[component, None], axis=0
                ) / max(comp_area, 1e-12)
                self._all_occ_faces.append(
                    {
                        "mesh_virtual_face": True,
                        "selector": "Feature",
                        "component_index": int(component_index),
                        "stored_index": int(_MESH_FEATURE_INDEX_BASE + component_index),
                        "label": f"Surface {component_index + 1}",
                        "face_index": face_idx,
                        "center": [float(v) for v in comp_center.tolist()],
                        "area": comp_area,
                    }
                )
                self._face_polydata_list.append(
                    self._polydata_from_cells(dataset, cell_ids)
                )
                # Feature regions intentionally win over the legacy
                # directional map for interactive clicks.
                for cell_id in cell_ids:
                    self._face_map[int(cell_id)] = face_idx
        except Exception:
            self._face_map = {}
            self._all_occ_faces = []
            self._face_polydata_list = []

    def ensure_mesh_face_picking(self):
        """Build mesh virtual-face picking data lazily for the current render."""
        try:
            self._mesh_picking_too_dense = False
            if self._all_occ_faces and any(
                pd is not None for pd in self._face_polydata_list
            ):
                return True
            dataset = getattr(self, "_pickable_surface_dataset", None)
            if dataset is None:
                return False
            self._configure_mesh_face_picking_from_surface(dataset)
            return bool(self._all_occ_faces) and any(
                pd is not None for pd in self._face_polydata_list
            )
        except Exception:
            return False

    def set_bc_overlay_data(
        self, constraint_faces=None, load_faces=None, load_vectors=None
    ):
        """
        Cache BC overlay data so it can be re-applied after render_simulation().
        Pass None to all arguments to clear the cached data.
        """
        has_data = bool(
            (constraint_faces and any(f is not None for f in constraint_faces))
            or (load_faces and any(f is not None for f in load_faces))
            or (load_vectors and len(load_vectors) > 0)
        )
        if has_data:
            self._cached_bc_data = (
                list(constraint_faces) if constraint_faces else [],
                list(load_faces) if load_faces else [],
                list(load_vectors) if load_vectors else [],
            )
        else:
            self._cached_bc_data = None
