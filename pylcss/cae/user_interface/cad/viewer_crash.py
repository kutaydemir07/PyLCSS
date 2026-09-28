# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE.
"""CrashPlaybackMixin implementation for the CAD viewer."""

from __future__ import annotations

import logging

import numpy as np
import vtk
from PySide6 import QtCore, QtWidgets

from .boundary_visuals import (
    BC_PALETTE,
)

__all__ = ["CrashPlaybackMixin"]


# The deck writers tag the wall they emitted as "stationary" (moving body into
# a fixed barrier), "moving" (rigid impactor driven into a fixed specimen) or
# "prescribed" (massless platen on a prescribed path). These have to match the
# wording the setup overlay uses before the run, so the label does not change
# meaning once results arrive.
_WALL_LABELS = {
    "stationary": "FIXED WALL",
    "moving": "IMPACTOR",
    "prescribed": "MOVING WALL",
}


def _wall_label(info: dict) -> str:
    """Return the viewer caption for the rigid wall described by *info*."""
    label = _WALL_LABELS.get(str(info.get("type") or "").strip().lower())
    if label is not None:
        return label
    # Older results and user-supplied decks may omit the tag; a wall with no
    # velocity of its own is a fixed barrier.
    try:
        wall_speed = float(info.get("v0_mm_per_ms", 0.0) or 0.0)
    except (TypeError, ValueError):
        wall_speed = 0.0
    return "FIXED WALL" if wall_speed <= 0.0 else "MOVING WALL"


class CrashPlaybackMixin:
    def _build_crash_panel(self):
        """Construct the crash animation playback toolbar (initially hidden)."""
        panel = QtWidgets.QWidget(self)
        panel.setObjectName("crash_panel")
        panel.setFixedHeight(52)
        panel.setStyleSheet(
            "#crash_panel {"
            "  background: rgba(15, 20, 35, 230);"
            "  border-top: 2px solid #e74c3c;"
            "}"
            "QLabel { color: #ecf0f1; font-size: 11px; }"
            "QPushButton {"
            "  background: #2c3e50; color: white; border: 1px solid #4a4a6a;"
            "  border-radius: 4px; padding: 3px 10px; font-size: 12px;"
            "}"
            "QPushButton:hover { background: #e74c3c; }"
            "QSlider::groove:horizontal { height: 4px; background: #4a4a6a; border-radius: 2px; }"
            "QSlider::handle:horizontal {"
            "  background: #e74c3c; width: 14px; height: 14px;"
            "  border-radius: 7px; margin: -5px 0;"
            "}"
            "QSlider::sub-page:horizontal { background: #e74c3c; border-radius: 2px; }"
            "QComboBox {"
            "  background: #2c3e50; color: white; border: 1px solid #4a4a6a;"
            "  border-radius: 3px; padding: 2px 6px; font-size: 11px;"
            "}"
        )

        lay = QtWidgets.QHBoxLayout(panel)
        lay.setContentsMargins(10, 4, 10, 4)
        lay.setSpacing(8)

        # Crash icon label
        icon_lbl = QtWidgets.QLabel("   IMPACT PLAYBACK")
        icon_lbl.setStyleSheet("color: #e74c3c; font-weight: bold; font-size: 12px;")
        lay.addWidget(icon_lbl)

        # Rewind button
        btn_rew = QtWidgets.QPushButton("\u23ee")  # ⏮
        btn_rew.setFixedWidth(32)
        btn_rew.setToolTip("Rewind to start")
        btn_rew.clicked.connect(self._crash_rewind)
        lay.addWidget(btn_rew)

        # Play/Pause button
        self._play_btn = QtWidgets.QPushButton("\u25b6")  # ▶
        self._play_btn.setFixedWidth(32)
        self._play_btn.setToolTip("Play / Pause (Space)")
        self._play_btn.clicked.connect(self._toggle_crash_play)
        lay.addWidget(self._play_btn)

        # Slider
        self._crash_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self._crash_slider.setMinimum(0)
        self._crash_slider.setMaximum(0)
        self._crash_slider.setValue(0)
        self._crash_slider.setToolTip("Drag to scrub through time")
        self._crash_slider.valueChanged.connect(self._on_crash_slider_changed)
        lay.addWidget(self._crash_slider, stretch=1)

        # Time label
        self._crash_time_lbl = QtWidgets.QLabel("t = 0.000 ms")
        self._crash_time_lbl.setFixedWidth(95)
        lay.addWidget(self._crash_time_lbl)

        # Speed combo
        lay.addWidget(QtWidgets.QLabel("Speed:"))
        self._speed_combo = QtWidgets.QComboBox()
        for label in [
            "0.25\u00d7",
            "0.5\u00d7",
            "1\u00d7",
            "2\u00d7",
            "4\u00d7",
            "8\u00d7",
        ]:
            self._speed_combo.addItem(label)
        self._speed_combo.setCurrentIndex(2)  # 1×
        self._speed_combo.setFixedWidth(62)
        self._speed_combo.currentIndexChanged.connect(self._on_speed_changed)
        lay.addWidget(self._speed_combo)

        # Frame counter
        self._crash_frame_lbl = QtWidgets.QLabel("0 / 0")
        self._crash_frame_lbl.setFixedWidth(55)
        lay.addWidget(self._crash_frame_lbl)

        # Close button
        btn_close = QtWidgets.QPushButton("\u2715")  # ✕
        btn_close.setFixedWidth(28)
        btn_close.setToolTip("Close playback panel")
        btn_close.clicked.connect(self.stop_crash_playback)
        lay.addWidget(btn_close)

        return panel

    def start_crash_playback(self, crash_data):
        """
        Load crash result data into the playback panel and begin animating.
        Call this instead of render_simulation() when crash results arrive.
        """
        frames = crash_data.get("frames", [])
        if not frames:
            # No frames recorded — fall back to static render
            self.render_simulation(crash_data)
            return

        self._crash_frames = frames
        self._crash_base_data = crash_data
        self._crash_frame_idx = 0
        self._crash_playing = False
        # Rigid impact wall geometry (None for user-supplied decks).
        self._clear_crash_wall_visuals()
        self._crash_wall_info = (
            crash_data.get("wall") if isinstance(crash_data, dict) else None
        )
        # Preserve cached setup overlays during animation. They are shown as
        # reference-configuration conditions in the foreground layer.

        # Compute global scalar range for stable colourmap across all frames
        viz_mode = crash_data.get("visualization_mode", "Von Mises Stress")
        all_vals = []
        for fr in frames:
            field = self._crash_field_for_frame(fr, viz_mode)
            if field is not None and len(field) > 0:
                # Visual deformation exaggeration must never alter the
                # physical values shown by the colour legend.
                all_vals.append(float(np.min(field)))
                all_vals.append(float(np.max(field)))
        if all_vals:
            lo, hi = min(all_vals), max(all_vals)
            if hi - lo < 1e-10:
                hi = lo + 1.0
            self._crash_scalar_range = (lo, hi)
        else:
            self._crash_scalar_range = (0.0, 1.0)

        # Configure slider
        self._crash_slider.blockSignals(True)
        self._crash_slider.setMaximum(len(frames) - 1)
        self._crash_slider.setValue(0)
        self._crash_slider.blockSignals(False)

        # ── Set camera to encompass the fully-deformed bounding box ──────────
        # Using the last frame (maximum deformation) ensures the whole range
        # of motion is visible from the very first frame rendered, without the
        # camera ever being reset to the undeformed (frame-0) state.
        mesh = crash_data.get("mesh")
        disp_scale = float(crash_data.get("disp_scale", 1.0))
        if mesh is not None and frames:
            last_disp = frames[-1].get("displacement")
            self._set_camera_for_crash(mesh, last_disp, disp_scale)

        # Show panel
        self._crash_panel.show()
        self._position_nav_cube()

        # Render first frame (camera is already positioned — never reset it again)
        self._render_crash_frame(0)
        self._toggle_crash_play()  # start playing

    def stop_crash_playback(self):
        """Stop playback and hide the crash panel."""
        self._crash_timer.stop()
        self._crash_playing = False
        self._play_btn.setText("\u25b6")
        self._crash_panel.hide()
        self._position_nav_cube()
        self._crash_frames = []
        self._crash_base_data = None
        self._clear_crash_wall_visuals()
        self._crash_wall_info = None

    def _set_camera_for_crash(self, mesh, last_displacement, disp_scale=1.0):
        """
        Position the camera so that both the original mesh AND the most-deformed
        state are fully visible.  Called once when crash playback starts; the
        camera is then kept fixed for the entire animation (including loops).

        Parameters
        ----------
        mesh            : skfem MeshTet – original (undeformed) mesh
        last_displacement: (3·N,) displacement array for the final frame
        disp_scale       : visualisation scale factor applied to displacement
        """
        p0 = mesh.p  # (3, N) original coords
        n = p0.shape[1]

        # Start from original bounding box
        x0_min, x0_max = float(p0[0].min()), float(p0[0].max())
        y0_min, y0_max = float(p0[1].min()), float(p0[1].max())
        z0_min, z0_max = float(p0[2].min()), float(p0[2].max())

        # Expand to include the final deformed state
        if last_displacement is not None and len(last_displacement) == 3 * n:
            d3 = last_displacement.reshape((3, n), order="F") * disp_scale
            pf = p0 + d3
            x0_min = min(x0_min, float(pf[0].min()))
            x0_max = max(x0_max, float(pf[0].max()))
            y0_min = min(y0_min, float(pf[1].min()))
            y0_max = max(y0_max, float(pf[1].max()))
            z0_min = min(z0_min, float(pf[2].min()))
            z0_max = max(z0_max, float(pf[2].max()))

        # Add 10 % margin on every side
        mx = (x0_max - x0_min) * 0.10
        my = (y0_max - y0_min) * 0.10
        mz = (z0_max - z0_min) * 0.10
        bounds = [
            x0_min - mx,
            x0_max + mx,
            y0_min - my,
            y0_max + my,
            z0_min - mz,
            z0_max + mz,
        ]
        self.renderer.ResetCamera(bounds)

    @staticmethod
    def _crash_field_for_frame(frame, viz_mode):
        """Return the correct scalar field from a frame dict for a given viz mode."""
        if viz_mode == "Von Mises Stress":
            return frame.get("stress_vm")
        elif viz_mode == "Displacement":
            u = frame.get("displacement")
            if u is not None:
                n = len(u) // 3
                return np.linalg.norm(u.reshape(n, 3), axis=1)
            return None
        elif viz_mode == "Plastic Strain":
            return frame.get("eps_p")
        elif viz_mode == "Failed Elements":
            return frame.get("failed")
        return frame.get("stress_vm")

    def _render_crash_frame(self, idx):
        """Render a single crash animation frame at the given index."""
        if not self._crash_frames or not self._crash_base_data:
            return
        idx = max(0, min(idx, len(self._crash_frames) - 1))
        frame = self._crash_frames[idx]

        # Build a view-data dict that re-uses the cached mesh
        viz_mode = self._crash_base_data.get("visualization_mode", "Von Mises Stress")
        disp_scale = float(self._crash_base_data.get("disp_scale", 1.0))
        field = self._crash_field_for_frame(frame, viz_mode)

        # Keep the physical displacement field unchanged.  The common
        # simulation renderer applies ``deformation_scale`` to coordinates
        # only, so the legend retains engineering units.
        raw_disp = frame["displacement"]
        vis_disp = raw_disp * disp_scale if disp_scale != 1.0 else raw_disp

        view_data = {
            "type": "crash_frame",  # avoids re-triggering start_crash_playback
            "mesh": self._crash_base_data["mesh"],
            "displacement": raw_disp,
            "deformation_scale": disp_scale,
            "stress": field,
            "visualization_mode": viz_mode,
            "_scalar_range": self._crash_scalar_range,  # pass locked range
            # Camera is set once in start_crash_playback based on the fully-
            # deformed bounding box — never reset it again during playback so
            # the viewer provides a stable, fixed-viewpoint reference throughout
            # the animation (including subsequent loops).
            "_reset_camera": False,
        }
        self.render_simulation(view_data)

        # Snap the rigid impact wall to this frame's deformed mesh (vis_disp is
        # the same view-scaled displacement the mesh is drawn with).
        self._update_crash_wall(vis_disp)

        # Update UI labels
        t_ms = float(frame.get("time", 0.0))
        if frame.get("time_is_normalized", False):
            self._crash_time_lbl.setText(f"t = {t_ms:.3f} norm")
        else:
            self._crash_time_lbl.setText(f"t = {t_ms:.3f} ms")
        self._crash_frame_lbl.setText(f"{idx + 1} / {len(self._crash_frames)}")

        # Keep slider in sync without triggering valueChanged recursion
        self._crash_slider.blockSignals(True)
        self._crash_slider.setValue(idx)
        self._crash_slider.blockSignals(False)

    def _clear_crash_wall_visuals(self) -> None:
        if self._crash_wall_actor is not None:
            self.renderer.RemoveActor(self._crash_wall_actor)
            self._crash_wall_actor = None
        if self._crash_wall_outline_actor is not None:
            self.bc_renderer.RemoveActor(self._crash_wall_outline_actor)
            self._crash_wall_outline_actor = None
        if self._crash_wall_label_actor is not None:
            self.bc_renderer.RemoveActor(self._crash_wall_label_actor)
            self._crash_wall_label_actor = None

    def _update_crash_wall(self, vis_disp) -> None:
        """Draw / refresh the rigid impact wall (barrier) for the current frame.

        OpenRadioss does not export the rigid wall's trajectory in the
        animation files, so rather than *guessing* its position kinematically
        (which drifts out of the imported mesh's coordinate frame and cuts
        through the part), we **snap the wall to the deformed mesh itself**:
        the wall plane is placed just outside the leading crush face — the
        extreme of the deformed nodes along the impact direction.  Because this
        is computed from the exact node positions the viewer draws (``mesh.p``
        + view-scaled displacement), the wall can never penetrate the part and
        automatically tracks the crush at any ``disp_scale``.

        For a fixed barrier (moving body + fixed wall, ``v0 == 0``) the wall is
        stationary, so we keep its original placement.
        """
        info = self._crash_wall_info
        if not isinstance(info, dict):
            self._clear_crash_wall_visuals()
            return
        try:
            pt = np.asarray(info.get("pt", [0.0, 0.0, 0.0]), dtype=float)[:3]
            normal = np.asarray(info.get("normal", [0.0, 0.0, 1.0]), dtype=float)[:3]
            half = float(info.get("half_extent", 0.0) or 0.0)
            v0 = float(info.get("v0_mm_per_ms", 0.0) or 0.0)
        except Exception:
            return
        if half <= 0.0:
            return

        n = np.asarray(normal, dtype=float)
        nn = float(np.linalg.norm(n))
        n = n / nn if nn > 1e-9 else np.array([0.0, 0.0, 1.0])

        center = None
        mesh = (
            self._crash_base_data.get("mesh")
            if isinstance(self._crash_base_data, dict)
            else None
        )
        wall_half_extents = (half, half)
        if mesh is not None and hasattr(mesh, "p"):
            wall_half_extents = self._wall_half_extents_for_mesh(
                mesh,
                n,
                fallback=half,
            )
        if v0 > 0.0 and mesh is not None and hasattr(mesh, "p"):
            # Moving wall: snap to the leading face of the deformed mesh.
            try:
                base = np.asarray(mesh.p, dtype=float)
                base = base.T if base.shape[0] in (2, 3) else base
                base = base[:, :3]
                d = np.asarray(vis_disp, dtype=float).reshape(-1, 3)
                pts = base + d if d.shape[0] == base.shape[0] else base
                proj = pts @ n
                span = float(proj.max() - proj.min())
                gap = max(span * 0.01, 1e-6)
                wall_proj = float(proj.min()) - gap  # just outside the lowest node
                centroid = pts.mean(axis=0)
                center = centroid + n * (wall_proj - float(centroid @ n))
            except Exception:
                center = None
        if center is None:
            center = pt  # fixed barrier (or fallback): keep original placement

        pd = self._make_wall_polydata(center, n, wall_half_extents)
        if pd is None:
            return
        if self._crash_wall_actor is None:
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputData(pd)
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            prop = actor.GetProperty()
            prop.SetColor(0.85, 0.55, 0.20)  # amber barrier
            prop.SetOpacity(0.14)
            prop.SetEdgeVisibility(1)
            prop.SetEdgeColor(0.95, 0.70, 0.30)
            prop.SetLineWidth(1.5)
            try:
                prop.LightingOff()
            except Exception:
                logging.getLogger(__name__).debug(
                    "Optional UI operation failed.", exc_info=True
                )
            actor.SetPickable(0)
            self.renderer.AddActor(actor)
            self._crash_wall_actor = actor

            edges = vtk.vtkFeatureEdges()
            edges.SetInputData(pd)
            edges.BoundaryEdgesOn()
            edges.FeatureEdgesOff()
            edges.ManifoldEdgesOff()
            edges.NonManifoldEdgesOff()
            outline_mapper = vtk.vtkPolyDataMapper()
            outline_mapper.SetInputConnection(edges.GetOutputPort())
            outline_actor = vtk.vtkActor()
            outline_actor.SetMapper(outline_mapper)
            outline_actor.GetProperty().SetColor(*BC_PALETTE["impact"])
            outline_actor.GetProperty().SetLineWidth(2.5)
            outline_actor.GetProperty().LightingOff()
            outline_actor.SetPickable(0)
            self.bc_renderer.AddActor(outline_actor)
            self._crash_wall_outline_actor = outline_actor

            label_actor = vtk.vtkBillboardTextActor3D()
            label_actor.SetInput(_wall_label(info))
            label_actor.SetPosition(
                *self._wall_annotation_position(center, n, wall_half_extents)
            )
            label_actor.SetPickable(0)
            label_actor._bc_text_label = True
            label_actor.SetVisibility(1 if self._bc_text_visible() else 0)
            label_property = label_actor.GetTextProperty()
            label_property.SetFontSize(14)
            label_property.BoldOn()
            wall_text_color = np.asarray(BC_PALETTE["impact"], dtype=float)
            if self._light_mode:
                wall_text_color *= 0.56
            label_property.SetColor(*wall_text_color)
            label_property.SetBackgroundColor(
                *(0.95, 0.97, 0.99) if self._light_mode else (0.04, 0.05, 0.07)
            )
            label_property.SetBackgroundOpacity(0.88)
            self.bc_renderer.AddActor(label_actor)
            self._crash_wall_label_actor = label_actor
            self._apply_bc_visibility(render=False)
        else:
            self._crash_wall_actor.GetMapper().SetInputData(pd)
            self._crash_wall_actor.GetMapper().Modified()
            if self._crash_wall_outline_actor is not None:
                edge_source = (
                    self._crash_wall_outline_actor.GetMapper().GetInputConnection(0, 0)
                )
                producer = (
                    edge_source.GetProducer() if edge_source is not None else None
                )
                if producer is not None and hasattr(producer, "SetInputData"):
                    producer.SetInputData(pd)
                    producer.Modified()
            if self._crash_wall_label_actor is not None:
                self._crash_wall_label_actor.SetPosition(
                    *self._wall_annotation_position(
                        center,
                        n,
                        wall_half_extents,
                    )
                )

    @staticmethod
    def _wall_axes(normal):
        """Return stable in-plane axes for a wall normal."""
        n = np.asarray(normal, dtype=float)
        norm = float(np.linalg.norm(n))
        n = n / norm if norm > 1.0e-9 else np.array([0.0, 0.0, 1.0])
        seed = (
            np.array([1.0, 0.0, 0.0])
            if abs(n[0]) < 0.9
            else np.array([0.0, 1.0, 0.0])
        )
        first = np.cross(n, seed)
        first_norm = float(np.linalg.norm(first))
        if first_norm <= 1.0e-9:
            return None
        first /= first_norm
        second = np.cross(n, first)
        return first, second

    @classmethod
    def _wall_half_extents_for_mesh(cls, mesh, normal, fallback):
        """Fit a padded rectangular wall to the model cross-section.

        Older results stored a square half-size based on the full 3-D model
        diagonal. For a long crash rail that makes the wall several times
        larger than the impacted section. Recompute the two transverse
        half-extents from the displayed mesh so cached results also render
        correctly without being rerun.
        """
        axes = cls._wall_axes(normal)
        if axes is None:
            return float(fallback), float(fallback)
        try:
            points = np.asarray(mesh.p, dtype=float)
            points = points.T if points.shape[0] in (2, 3) else points
            points = points[:, :3]
            first, second = axes
            first_span = float(np.ptp(points @ first))
            second_span = float(np.ptp(points @ second))
            model_diagonal = float(np.linalg.norm(np.ptp(points, axis=0)))
            minimum_half = max(0.035 * model_diagonal, 1.0e-6)
            return (
                max(minimum_half, 0.54 * first_span),
                max(minimum_half, 0.54 * second_span),
            )
        except Exception:
            return float(fallback), float(fallback)

    @classmethod
    def _wall_annotation_position(cls, center, normal, half_extents):
        """Place the wall label near a corner instead of over the result field."""
        axes = cls._wall_axes(normal)
        if axes is None:
            return np.asarray(center, dtype=float)
        first, second = axes
        half_values = np.asarray(half_extents, dtype=float).reshape(-1)
        if half_values.size == 1:
            half_first = half_second = float(half_values[0])
        else:
            half_first = float(half_values[0])
            half_second = float(half_values[1])
        return (
            np.asarray(center, dtype=float)
            + 0.78 * half_first * first
            + 0.90 * half_second * second
        )

    @classmethod
    def _make_wall_polydata(cls, center, normal, half_extents):
        """Rectangular wall fitted to the impacted model cross-section."""
        axes = cls._wall_axes(normal)
        if axes is None:
            return None
        u, v = axes
        half_values = np.asarray(half_extents, dtype=float).reshape(-1)
        if half_values.size == 1:
            half_u = half_v = float(half_values[0])
        else:
            half_u = float(half_values[0])
            half_v = float(half_values[1])
        if (
            not np.isfinite(half_u)
            or not np.isfinite(half_v)
            or half_u <= 0.0
            or half_v <= 0.0
        ):
            return None
        c = np.asarray(center, dtype=float)
        corners = [
            c - u * half_u - v * half_v,
            c + u * half_u - v * half_v,
            c + u * half_u + v * half_v,
            c - u * half_u + v * half_v,
        ]
        pts = vtk.vtkPoints()
        for corner in corners:
            pts.InsertNextPoint(float(corner[0]), float(corner[1]), float(corner[2]))
        quad = vtk.vtkQuad()
        for i in range(4):
            quad.GetPointIds().SetId(i, i)
        cells = vtk.vtkCellArray()
        cells.InsertNextCell(quad)
        pd = vtk.vtkPolyData()
        pd.SetPoints(pts)
        pd.SetPolys(cells)
        return pd

    def _on_crash_timer(self):
        """Timer callback: advance one frame."""
        if not self._crash_frames:
            self._crash_timer.stop()
            return
        next_idx = self._crash_frame_idx + 1
        if next_idx >= len(self._crash_frames):
            # Loop back to start
            next_idx = 0
        self._crash_frame_idx = next_idx
        self._render_crash_frame(next_idx)

    def _toggle_crash_play(self):
        """Toggle play / pause."""
        if self._crash_playing:
            self._crash_timer.stop()
            self._crash_playing = False
            self._play_btn.setText("\u25b6")
        else:
            self._crash_playing = True
            self._play_btn.setText("\u23f8")  # ⏸
            self._update_timer_interval()
            self._crash_timer.start()

    def _crash_rewind(self):
        """Jump to frame 0."""
        self._crash_timer.stop()
        self._crash_playing = False
        self._play_btn.setText("\u25b6")
        self._crash_frame_idx = 0
        self._render_crash_frame(0)

    def _on_crash_slider_changed(self, value):
        """User dragged the slider — pause and jump to that frame."""
        self._crash_timer.stop()
        self._crash_playing = False
        self._play_btn.setText("\u25b6")
        self._crash_frame_idx = value
        self._render_crash_frame(value)

    def _on_speed_changed(self, _):
        """Playback speed changed — update timer interval."""
        if self._crash_playing:
            self._update_timer_interval()
            self._crash_timer.start()

    def _update_timer_interval(self):
        """Set QTimer interval based on speed combo selection."""
        # Target ~30 fps at 1×; speed multiplier makes frames go faster/slower
        speed_map = {0: 0.25, 1: 0.5, 2: 1.0, 3: 2.0, 4: 4.0, 5: 8.0}
        speed = speed_map.get(self._speed_combo.currentIndex(), 1.0)
        interval_ms = max(8, int(33.0 / speed))
        self._crash_timer.setInterval(interval_ms)
