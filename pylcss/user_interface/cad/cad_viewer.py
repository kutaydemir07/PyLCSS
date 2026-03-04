# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

import vtk
import numpy as np
from PySide6 import QtWidgets, QtCore, QtGui
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor


class CQ3DViewer(QtWidgets.QWidget):
    """
    Professional 3D Viewer for CadQuery using VTK.
    Embeds directly into PySide6 layouts.
    Supports interactive face picking mode for FEA boundary condition assignment.
    """

    # Signals emitted during interactive picking
    face_picked = QtCore.Signal(list)      # list of OCC face objects
    picking_cancelled = QtCore.Signal()

    def __init__(self, parent=None):
        super(CQ3DViewer, self).__init__(parent)
        self.main_layout = QtWidgets.QVBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)

        # --- Picking toolbar (hidden by default) ---
        self._picking_toolbar = QtWidgets.QWidget(self)
        self._picking_toolbar.setObjectName("picking_toolbar")
        self._picking_toolbar.setStyleSheet(
            "#picking_toolbar {"
            "  background: rgba(30, 90, 180, 220);"
            "  border-bottom: 2px solid #4a9eff;"
            "}"
        )
        tb_layout = QtWidgets.QHBoxLayout(self._picking_toolbar)
        tb_layout.setContentsMargins(12, 6, 12, 6)

        self._pick_icon_lbl = QtWidgets.QLabel("Face Picking Mode  --  Click faces to select")
        self._pick_icon_lbl.setStyleSheet("color: white; font-weight: bold; font-size: 13px;")
        tb_layout.addWidget(self._pick_icon_lbl)

        tb_layout.addStretch()

        self._pick_count_lbl = QtWidgets.QLabel("0 selected")
        self._pick_count_lbl.setStyleSheet("color: #aad4ff; font-size: 12px; margin-right: 16px;")
        tb_layout.addWidget(self._pick_count_lbl)

        self._pick_hint_lbl = QtWidgets.QLabel("Ctrl+Click = multi-select")
        self._pick_hint_lbl.setStyleSheet("color: #aad4ff; font-size: 11px; margin-right: 16px;")
        tb_layout.addWidget(self._pick_hint_lbl)

        btn_done = QtWidgets.QPushButton("Done")
        btn_done.setStyleSheet(
            "QPushButton { background: #27ae60; color: white; border-radius: 4px;"
            "  padding: 4px 14px; font-weight: bold; font-size: 13px; }"
            "QPushButton:hover { background: #2ecc71; }"
        )
        btn_done.clicked.connect(self._on_pick_done)
        tb_layout.addWidget(btn_done)

        btn_cancel = QtWidgets.QPushButton("Cancel")
        btn_cancel.setStyleSheet(
            "QPushButton { background: #c0392b; color: white; border-radius: 4px;"
            "  padding: 4px 14px; font-weight: bold; font-size: 13px; }"
            "QPushButton:hover { background: #e74c3c; }"
        )
        btn_cancel.clicked.connect(self._on_pick_cancel)
        tb_layout.addWidget(btn_cancel)

        self._picking_toolbar.hide()
        self.main_layout.addWidget(self._picking_toolbar)

        # VTK Widget
        self.vtkWidget = QVTKRenderWindowInteractor(self)
        self.main_layout.addWidget(self.vtkWidget)

        # Renderer
        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(0.2, 0.2, 0.2)  # Dark Gray Background

        # Scalar Bar (Legend)
        self.scalar_bar = vtk.vtkScalarBarActor()
        self.scalar_bar.SetOrientationToVertical()
        self.scalar_bar.SetWidth(0.1)
        self.scalar_bar.SetHeight(0.8)
        self.scalar_bar.SetPosition(0.85, 0.1)
        self.scalar_bar.VisibilityOff()
        self.renderer.AddActor(self.scalar_bar)

        self.vtkWidget.GetRenderWindow().AddRenderer(self.renderer)
        self.interactor = self.vtkWidget.GetRenderWindow().GetInteractor()

        # Initialize axes (XYZ arrows)
        axes = vtk.vtkAxesActor()
        self.marker_widget = vtk.vtkOrientationMarkerWidget()
        self.marker_widget.SetOrientationMarker(axes)
        self.marker_widget.SetInteractor(self.interactor)
        self.marker_widget.SetViewport(0.0, 0.0, 0.2, 0.2)
        self.marker_widget.SetEnabled(1)

        # State
        self.current_actor = None
        self.actors = []  # List of all active actors

        # --- Picking State ---
        self._picking_mode = False
        self._multi_select = True
        self._picked_face_indices = []       # list of int (OCC face indices)
        self._picked_occ_faces = []          # list of OCC face objects
        self._highlight_actors = []          # VTK actors for selected face highlights
        self._face_map = {}                  # vtk_cell_id -> occ_face_index
        self._all_occ_faces = []             # list of OCC face objects (from last render_shape)
        self._face_polydata_list = []        # per-face vtkPolyData for highlighting
        self._pick_callback_id = None

        # --- BC Overlay State ---
        self._bc_overlay_actors = []     # dedicated list for load/support overlay actors
        self._cached_bc_data = None      # (constraint_faces, load_faces, load_vectors) – replayed after sim render

        # --- Crash Playback State ---
        self._crash_frames      = []         # list of frame dicts
        self._crash_base_data   = None       # original crash result (mesh, viz_mode, etc.)
        self._crash_frame_idx   = 0          # current frame index
        self._crash_playing     = False
        self._crash_scalar_range = (0.0, 1.0)  # global min/max for stable colourmap
        self._crash_timer       = QtCore.QTimer(self)
        self._crash_timer.timeout.connect(self._on_crash_timer)

        # Crash panel (hidden until crash results arrive)
        self._crash_panel = self._build_crash_panel()
        self._crash_panel.hide()
        self.main_layout.addWidget(self._crash_panel)

        self.interactor.Initialize()
        self.interactor.Start()

    # ──────────────────────────────────────────────────────────────────────────
    # CRASH ANIMATION PLAYBACK
    # ──────────────────────────────────────────────────────────────────────────

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
        icon_lbl = QtWidgets.QLabel("   CRASH PLAYBACK")
        icon_lbl.setStyleSheet("color: #e74c3c; font-weight: bold; font-size: 12px;")
        lay.addWidget(icon_lbl)

        # Rewind button
        btn_rew = QtWidgets.QPushButton("\u23EE")   # ⏮
        btn_rew.setFixedWidth(32)
        btn_rew.setToolTip("Rewind to start")
        btn_rew.clicked.connect(self._crash_rewind)
        lay.addWidget(btn_rew)

        # Play/Pause button
        self._play_btn = QtWidgets.QPushButton("\u25B6")   # ▶
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
        for label in ["0.25\u00d7", "0.5\u00d7", "1\u00d7", "2\u00d7", "4\u00d7", "8\u00d7"]:
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

    # ── Public entry point ────────────────────────────────────────────────────

    def start_crash_playback(self, crash_data):
        """
        Load crash result data into the playback panel and begin animating.
        Call this instead of render_simulation() when crash results arrive.
        """
        frames = crash_data.get('frames', [])
        if not frames:
            # No frames recorded — fall back to static render
            self.render_simulation(crash_data)
            return

        self._crash_frames    = frames
        self._crash_base_data = crash_data
        self._crash_frame_idx = 0
        self._crash_playing   = False

        # Compute global scalar range for stable colourmap across all frames
        viz_mode = crash_data.get('visualization_mode', 'Von Mises Stress')
        disp_scale_val = float(crash_data.get('disp_scale', 1.0))
        all_vals = []
        for fr in frames:
            field = self._crash_field_for_frame(fr, viz_mode)
            if field is not None and len(field) > 0:
                # For Displacement mode the global range must reflect the
                # scaled magnitudes that will actually be painted on the mesh.
                range_scale = disp_scale_val if viz_mode == 'Displacement' else 1.0
                all_vals.append(float(np.min(field)) * range_scale)
                all_vals.append(float(np.max(field)) * range_scale)
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
        mesh = crash_data.get('mesh')
        disp_scale = float(crash_data.get('disp_scale', 1.0))
        if mesh is not None and frames:
            last_disp = frames[-1].get('displacement')
            self._set_camera_for_crash(mesh, last_disp, disp_scale)

        # Show panel
        self._crash_panel.show()

        # Render first frame (camera is already positioned — never reset it again)
        self._render_crash_frame(0)
        self._toggle_crash_play()   # start playing

    def stop_crash_playback(self):
        """Stop playback and hide the crash panel."""
        self._crash_timer.stop()
        self._crash_playing = False
        self._play_btn.setText("\u25B6")
        self._crash_panel.hide()
        self._crash_frames      = []
        self._crash_base_data   = None

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
        p0 = mesh.p                                   # (3, N) original coords
        n  = p0.shape[1]

        # Start from original bounding box
        x0_min, x0_max = float(p0[0].min()), float(p0[0].max())
        y0_min, y0_max = float(p0[1].min()), float(p0[1].max())
        z0_min, z0_max = float(p0[2].min()), float(p0[2].max())

        # Expand to include the final deformed state
        if last_displacement is not None and len(last_displacement) == 3 * n:
            d3 = last_displacement.reshape((3, n), order='F') * disp_scale
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
            x0_min - mx, x0_max + mx,
            y0_min - my, y0_max + my,
            z0_min - mz, z0_max + mz,
        ]
        self.renderer.ResetCamera(bounds)

    # ── Internal helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _crash_field_for_frame(frame, viz_mode):
        """Return the correct scalar field from a frame dict for a given viz mode."""
        if viz_mode == 'Von Mises Stress':
            return frame.get('stress_vm')
        elif viz_mode == 'Displacement':
            u = frame.get('displacement')
            if u is not None:
                n = len(u) // 3
                return np.linalg.norm(u.reshape(n, 3), axis=1)
            return None
        elif viz_mode == 'Plastic Strain':
            return frame.get('eps_p')
        elif viz_mode == 'Failed Elements':
            return frame.get('failed')
        return frame.get('stress_vm')

    def _render_crash_frame(self, idx):
        """Render a single crash animation frame at the given index."""
        if not self._crash_frames or not self._crash_base_data:
            return
        idx = max(0, min(idx, len(self._crash_frames) - 1))
        frame = self._crash_frames[idx]

        # Build a view-data dict that re-uses the cached mesh
        viz_mode   = self._crash_base_data.get('visualization_mode', 'Von Mises Stress')
        disp_scale = float(self._crash_base_data.get('disp_scale', 1.0))
        field      = self._crash_field_for_frame(frame, viz_mode)

        # Scale displacement for visualisation (disp_scale > 1 magnifies the
        # deformation so small structural changes become clearly visible).
        raw_disp = frame['displacement']
        vis_disp = raw_disp * disp_scale if disp_scale != 1.0 else raw_disp

        view_data = {
            'type':               'crash_frame',   # avoids re-triggering start_crash_playback
            'mesh':               self._crash_base_data['mesh'],
            'displacement':       vis_disp,
            'stress':             field,
            'visualization_mode': viz_mode,
            '_scalar_range':      self._crash_scalar_range,  # pass locked range
            # Camera is set once in start_crash_playback based on the fully-
            # deformed bounding box — never reset it again during playback so
            # the viewer provides a stable, fixed-viewpoint reference throughout
            # the animation (including subsequent loops).
            '_reset_camera':      False,
        }
        self.render_simulation(view_data)

        # Update UI labels
        t_ms = float(frame.get('time', 0.0))
        self._crash_time_lbl.setText(f"t = {t_ms:.3f} ms")
        self._crash_frame_lbl.setText(f"{idx + 1} / {len(self._crash_frames)}")

        # Keep slider in sync without triggering valueChanged recursion
        self._crash_slider.blockSignals(True)
        self._crash_slider.setValue(idx)
        self._crash_slider.blockSignals(False)

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
            self._play_btn.setText("\u25B6")
        else:
            self._crash_playing = True
            self._play_btn.setText("\u23F8")   # ⏸
            self._update_timer_interval()
            self._crash_timer.start()

    def _crash_rewind(self):
        """Jump to frame 0."""
        self._crash_timer.stop()
        self._crash_playing = False
        self._play_btn.setText("\u25B6")
        self._crash_frame_idx = 0
        self._render_crash_frame(0)

    def _on_crash_slider_changed(self, value):
        """User dragged the slider — pause and jump to that frame."""
        self._crash_timer.stop()
        self._crash_playing = False
        self._play_btn.setText("\u25B6")
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

    # ──────────────────────────────────────────────────────────────────────────
    # PICKING MODE
    # ──────────────────────────────────────────────────────────────────────────

    def enable_picking_mode(self, multi_select=True):
        """Switch the viewer into face-selection picking mode."""
        self._picking_mode = True
        self._multi_select = multi_select
        self._picked_face_indices = []
        self._picked_occ_faces = []
        self._clear_highlight_actors()

        # Show toolbar
        self._picking_toolbar.show()
        self._pick_count_lbl.setText("0 selected")

        # Install VTK left-button press callback
        self._pick_callback_id = self.interactor.AddObserver(
            "LeftButtonPressEvent", self._on_vtk_pick
        )

        # Change cursor
        self.vtkWidget.setCursor(QtGui.QCursor(QtCore.Qt.CrossCursor))

    def disable_picking_mode(self):
        """Exit picking mode, restore normal orbit interaction."""
        self._picking_mode = False
        self._picking_toolbar.hide()
        self._clear_highlight_actors()

        if self._pick_callback_id is not None:
            try:
                self.interactor.RemoveObserver(self._pick_callback_id)
            except Exception:
                pass
            self._pick_callback_id = None

        self.vtkWidget.setCursor(QtGui.QCursor(QtCore.Qt.ArrowCursor))

    def _on_vtk_pick(self, obj, event):
        """VTK callback: called on left-button press in picking mode."""
        if not self._picking_mode:
            return

        x, y = self.interactor.GetEventPosition()
        modifiers = QtWidgets.QApplication.keyboardModifiers()
        ctrl_held = bool(modifiers & QtCore.Qt.ControlModifier)

        # Use cell picker for face selection
        picker = vtk.vtkCellPicker()
        picker.SetTolerance(0.001)
        picker.Pick(x, y, 0, self.renderer)

        cell_id = picker.GetCellId()
        if cell_id < 0:
            return  # Missed — no geometry at this pixel

        face_idx = self._face_map.get(cell_id, None)
        if face_idx is None:
            return

        if not ctrl_held:
            # Replace selection
            self._picked_face_indices = [face_idx]
            self._picked_occ_faces = [self._all_occ_faces[face_idx]] if face_idx < len(self._all_occ_faces) else []
        else:
            # Toggle face in/out of selection
            if face_idx in self._picked_face_indices:
                self._picked_face_indices.remove(face_idx)
                if face_idx < len(self._all_occ_faces):
                    face = self._all_occ_faces[face_idx]
                    if face in self._picked_occ_faces:
                        self._picked_occ_faces.remove(face)
            else:
                self._picked_face_indices.append(face_idx)
                if face_idx < len(self._all_occ_faces):
                    self._picked_occ_faces.append(self._all_occ_faces[face_idx])

        # Update highlights
        self._update_highlight_actors()
        n = len(self._picked_face_indices)
        self._pick_count_lbl.setText(f"{n} face{'s' if n != 1 else ''} selected")

    def _update_highlight_actors(self):
        """Re-render face highlights for currently selected faces."""
        self._clear_highlight_actors()

        for face_idx in self._picked_face_indices:
            if face_idx >= len(self._face_polydata_list):
                continue
            face_pd = self._face_polydata_list[face_idx]
            if face_pd is None:
                continue

            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputData(face_pd)
            mapper.ScalarVisibilityOff()

            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(1.0, 0.55, 0.0)   # Orange highlight
            actor.GetProperty().SetOpacity(0.8)
            actor.GetProperty().SetLineWidth(2.0)
            actor.GetProperty().EdgeVisibilityOn()
            actor.GetProperty().SetEdgeColor(1.0, 0.8, 0.0)

            self.renderer.AddActor(actor)
            self._highlight_actors.append(actor)

        self.vtkWidget.GetRenderWindow().Render()

    def _clear_highlight_actors(self):
        """Remove all highlight actors from the scene."""
        for actor in self._highlight_actors:
            self.renderer.RemoveActor(actor)
        self._highlight_actors = []

    def _on_pick_done(self):
        """User confirmed picking — emit signal and exit picking mode."""
        picked = list(self._picked_occ_faces)
        picked_indices = list(self._picked_face_indices)
        self.disable_picking_mode()
        # Re-add highlights in a passive/dimmed color to keep them visible
        for idx in picked_indices:
            if idx < len(self._face_polydata_list) and self._face_polydata_list[idx] is not None:
                mapper = vtk.vtkPolyDataMapper()
                mapper.SetInputData(self._face_polydata_list[idx])
                mapper.ScalarVisibilityOff()
                actor = vtk.vtkActor()
                actor.SetMapper(mapper)
                actor.GetProperty().SetColor(1.0, 0.6, 0.1)
                actor.GetProperty().SetOpacity(0.5)
                self.renderer.AddActor(actor)
                self.actors.append(actor)
        self.vtkWidget.GetRenderWindow().Render()
        self.face_picked.emit(picked)

    def _on_pick_cancel(self):
        """User cancelled picking."""
        self.disable_picking_mode()
        self.picking_cancelled.emit()

    def highlight_faces(self, face_indices):
        """Public method to highlight specific face indices matching current geometry."""
        self._clear_highlight_actors()
        for idx in face_indices:
            if idx < len(self._face_polydata_list) and self._face_polydata_list[idx] is not None:
                mapper = vtk.vtkPolyDataMapper()
                mapper.SetInputData(self._face_polydata_list[idx])
                mapper.ScalarVisibilityOff()
                actor = vtk.vtkActor()
                actor.SetMapper(mapper)
                actor.GetProperty().SetColor(1.0, 0.6, 0.1)  # Dim orange
                actor.GetProperty().SetOpacity(0.5)
                self.renderer.AddActor(actor)
                self.actors.append(actor)
        self.vtkWidget.GetRenderWindow().Render()

    # ──────────────────────────────────────────────────────────────────────────
    # CONFIRM / CANCEL from external code
    # ──────────────────────────────────────────────────────────────────────────

    def confirm_picking(self):
        self._on_pick_done()

    def cancel_picking(self):
        self._on_pick_cancel()

    # ──────────────────────────────────────────────────────────────────────────
    # CLEAR
    # ──────────────────────────────────────────────────────────────────────────

    def clear(self):
        """Clear the viewer and release memory."""
        self._clear_highlight_actors()

        # Clear BC overlay actors
        for actor in list(self._bc_overlay_actors):
            self.renderer.RemoveActor(actor)
        self._bc_overlay_actors = []
        # Reset cached BC data so overlays don't auto-replay on wrong shapes
        self._cached_bc_data = None

        # Clear legacy single actor if present
        if self.current_actor:
            self.renderer.RemoveActor(self.current_actor)
            if self.current_actor.GetMapper():
                self.current_actor.GetMapper().RemoveAllInputConnections(0)
            self.current_actor = None

        # Clear all actors in the list (excluding bc_overlay which were
        # already removed above to avoid double-removal)
        for actor in list(self.actors):
            if not getattr(actor, '_bc_overlay', False):
                self.renderer.RemoveActor(actor)
                if actor.GetMapper():
                    actor.GetMapper().RemoveAllInputConnections(0)
        self.actors = []

        self.scalar_bar.VisibilityOff()
        self.vtkWidget.GetRenderWindow().Render()

    def _update_scalar_bar(self, title, min_val, max_val, lut=None):
        """Update and show the scalar bar."""
        self.scalar_bar.SetTitle(title)
        self.scalar_bar.SetNumberOfLabels(5)

        if lut:
            self.scalar_bar.SetLookupTable(lut)
        elif self.current_actor and self.current_actor.GetMapper():
            self.scalar_bar.SetLookupTable(self.current_actor.GetMapper().GetLookupTable())

        self.scalar_bar.VisibilityOn()

    # ──────────────────────────────────────────────────────────────────────────
    # SKETCH RENDER
    # ──────────────────────────────────────────────────────────────────────────

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
            if hasattr(sketch, 'ctx') and hasattr(sketch.ctx, 'pendingWires'):
                try:
                    wires = sketch.ctx.pendingWires
                    if wires:
                        for wire in wires:
                            if hasattr(wire, 'Edges'):
                                edges.extend(wire.Edges())
                except Exception:
                    pass

            # Method 2: Try to get edges directly from the unwrapped shape
            if not edges:
                shape = sketch
                if hasattr(sketch, 'val'):
                    try:
                        shape = sketch.val()
                    except Exception:
                        pass

                if hasattr(shape, 'Edges'):
                    try:
                        edge_list = shape.Edges()
                        if edge_list:
                            edges = edge_list
                    except Exception:
                        pass

            # Method 3: Try the CadQuery .edges() API
            if not edges and hasattr(sketch, 'edges'):
                try:
                    edge_objects = sketch.edges().vals()
                    if edge_objects:
                        edges = edge_objects
                except Exception:
                    pass

            # Method 4: Check for wires on the shape
            if not edges:
                shape = sketch
                if hasattr(sketch, 'val'):
                    try:
                        shape = sketch.val()
                    except Exception:
                        pass

                if hasattr(shape, 'Wires'):
                    try:
                        wires = shape.Wires()
                        for wire in wires:
                            if hasattr(wire, 'Edges'):
                                edges.extend(wire.Edges())
                    except Exception:
                        pass

            # Method 5: Try _edges attribute (fallback)
            if not edges and hasattr(sketch, '_edges'):
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
                if hasattr(edge, 'wrapped'):
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
                        pass

                # Method 2: CadQuery edge discretization
                if not pts_extracted and hasattr(edge, 'discretize'):
                    try:
                        pts_extracted = edge.discretize(30)
                    except Exception:
                        pass

                # Method 3: Simple start/end for straight lines
                if not pts_extracted and hasattr(edge, 'startPoint') and hasattr(edge, 'endPoint'):
                    sp = edge.startPoint()
                    ep = edge.endPoint()
                    if abs(sp.x - ep.x) > 1e-6 or abs(sp.y - ep.y) > 1e-6 or abs(sp.z - ep.z) > 1e-6:
                        pts_extracted = [(sp.x, sp.y, sp.z), (ep.x, ep.y, ep.z)]

                if pts_extracted and len(pts_extracted) >= 2:
                    start_id = point_id
                    for pt in pts_extracted:
                        if hasattr(pt, 'x'):
                            points.InsertNextPoint(pt.x, pt.y, pt.z if hasattr(pt, 'z') else 0)
                        else:
                            points.InsertNextPoint(pt[0], pt[1], pt[2] if len(pt) > 2 else 0)
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

    # ──────────────────────────────────────────────────────────────────────────
    # SHAPE RENDER (with per-face picking support)
    # ──────────────────────────────────────────────────────────────────────────

    def render_shape(self, shape):
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

        topo_shape = shape

        if hasattr(shape, 'toCompound'):
            try:
                topo_shape = shape.toCompound()
            except Exception:
                return

        try:
            if hasattr(topo_shape, 'val'):
                topo_shape = topo_shape.val()
        except Exception:
            pass

        if not hasattr(topo_shape, 'tessellate'):
            if isinstance(topo_shape, (list, tuple)) and topo_shape:
                found = None
                for item in topo_shape:
                    if hasattr(item, 'tessellate'):
                        found = item
                        break
                    if hasattr(item, 'val'):
                        try:
                            v = item.val()
                            if hasattr(v, 'tessellate'):
                                found = v
                                break
                        except Exception:
                            continue
                if found is not None:
                    topo_shape = found

        if not hasattr(topo_shape, 'tessellate') and hasattr(topo_shape, 'objects'):
            try:
                for obj in getattr(topo_shape, 'objects', []):
                    if hasattr(obj, 'val'):
                        v = obj.val()
                        if hasattr(v, 'tessellate'):
                            topo_shape = v
                            break
                    if hasattr(obj, 'tessellate'):
                        topo_shape = obj
                        break
            except Exception:
                pass

        if not hasattr(topo_shape, 'tessellate'):
            self.render_sketch(shape)
            return

        # ── Build per-face tessellation for picking ──
        # Try to get OCC faces list for picking
        occ_faces = []
        try:
            if hasattr(topo_shape, 'Faces'):
                occ_faces = topo_shape.Faces()
        except Exception:
            pass

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
                        verts = tri.get('vertices') or tri.get('verts') or []
                        triangles = tri.get('triangles') or tri.get('faces') or []
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
                triangulation = topo_shape.tessellate(tolerance=0.1, angularTolerance=0.2)
                if isinstance(triangulation, dict):
                    verts = triangulation.get('vertices') or triangulation.get('verts')
                    triangles = triangulation.get('triangles') or triangulation.get('faces')
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

        # Calculate Normals for smooth shading
        normals = vtk.vtkPolyDataNormals()
        normals.SetInputData(poly_data)
        normals.Update()

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(normals.GetOutputPort())

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)

        # Styling (Metallic Blue/Grey)
        actor.GetProperty().SetColor(0.7, 0.75, 0.8)
        actor.GetProperty().SetSpecular(0.5)
        actor.GetProperty().SetSpecularPower(20)

        self.renderer.AddActor(actor)
        self.current_actor = actor

        self.renderer.ResetCamera()
        self.vtkWidget.GetRenderWindow().Render()

    # ──────────────────────────────────────────────────────────────────────────
    # BC OVERLAYS
    # ──────────────────────────────────────────────────────────────────────────

    def set_bc_overlay_data(self, constraint_faces=None, load_faces=None, load_vectors=None):
        """
        Cache BC overlay data so it can be re-applied after render_simulation().
        Pass None to all arguments to clear the cached data.
        """
        has_data = bool(
            (constraint_faces and any(f is not None for f in constraint_faces)) or
            (load_faces and any(f is not None for f in load_faces)) or
            (load_vectors and len(load_vectors) > 0)
        )
        if has_data:
            self._cached_bc_data = (
                list(constraint_faces) if constraint_faces else [],
                list(load_faces) if load_faces else [],
                list(load_vectors) if load_vectors else [],
            )
        else:
            self._cached_bc_data = None

    def render_bc_overlays(self, constraint_faces=None, load_faces=None, load_vectors=None):
        """
        Highlight boundary condition faces in the viewer:
          - constraint_faces: list of OCC face objects → red highlight
          - load_faces: list of OCC face objects → yellow highlight
          - load_vectors: list of (centroid [x,y,z], vector [fx,fy,fz]) → force arrows
        Always replaces previously drawn BC overlays (no stale arrows/highlights).
        """
        # ── Remove all previous BC overlay actors ──────────────────────────────
        for actor in list(self._bc_overlay_actors):
            self.renderer.RemoveActor(actor)
            # Also pull out of the generic actors list if it ended up there
            if actor in self.actors:
                self.actors.remove(actor)
        self._bc_overlay_actors = []

        def _face_to_polydata(occ_face):
            try:
                tri = occ_face.tessellate(tolerance=0.15, angularTolerance=0.3)
                if isinstance(tri, dict):
                    verts = tri.get('vertices') or tri.get('verts') or []
                    tris = tri.get('triangles') or tri.get('faces') or []
                else:
                    verts, tris = tri[0], tri[1]
                if not verts:
                    return None
                pts = vtk.vtkPoints()
                polys = vtk.vtkCellArray()
                for v in verts:
                    pts.InsertNextPoint(v.x, v.y, v.z)
                for t in tris:
                    polys.InsertNextCell(3)
                    polys.InsertCellPoint(t[0])
                    polys.InsertCellPoint(t[1])
                    polys.InsertCellPoint(t[2])
                pd = vtk.vtkPolyData()
                pd.SetPoints(pts)
                pd.SetPolys(polys)
                return pd
            except Exception:
                return None

        def _add_face_overlay(occ_face, color, opacity=0.55):
            pd = _face_to_polydata(occ_face)
            if pd is None:
                return
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputData(pd)
            mapper.ScalarVisibilityOff()
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(*color)
            actor.GetProperty().SetOpacity(opacity)
            actor.GetProperty().EdgeVisibilityOn()
            actor.GetProperty().SetEdgeColor(*color)
            actor._bc_overlay = True
            self.renderer.AddActor(actor)
            self._bc_overlay_actors.append(actor)

        def _add_bc_arrow(start, vector, color):
            """Arrow tagged as a BC overlay so it is cleaned up with the rest."""
            import numpy as _np
            length = _np.linalg.norm(vector)
            if length < 1e-9:
                return
            arrow_source = vtk.vtkArrowSource()
            visual_length = 5.0
            norm_vec = vector / length
            transform = vtk.vtkTransform()
            transform.Translate(start[0], start[1], start[2])
            default_dir = _np.array([1.0, 0.0, 0.0])
            axis = _np.cross(default_dir, norm_vec)
            angle_deg = _np.degrees(_np.arccos(_np.clip(_np.dot(default_dir, norm_vec), -1.0, 1.0)))
            if _np.linalg.norm(axis) > 1e-6:
                transform.RotateWXYZ(angle_deg, axis)
            elif _np.dot(default_dir, norm_vec) < 0:
                transform.RotateWXYZ(180, [0, 1, 0])
            transform.Scale(visual_length, visual_length, visual_length)
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(arrow_source.GetOutputPort())
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.SetUserTransform(transform)
            actor.GetProperty().SetColor(color)
            actor._bc_overlay = True
            self.renderer.AddActor(actor)
            self._bc_overlay_actors.append(actor)

        if constraint_faces:
            for face in constraint_faces:
                if face is not None:
                    _add_face_overlay(face, color=(0.9, 0.15, 0.15))  # Red

        if load_faces:
            for face in load_faces:
                if face is not None:
                    _add_face_overlay(face, color=(1.0, 0.85, 0.0))   # Yellow

        if load_vectors:
            for centroid, vec in load_vectors:
                _add_bc_arrow(centroid, vec, color=(1.0, 0.85, 0.0))

        self.vtkWidget.GetRenderWindow().Render()

    # ──────────────────────────────────────────────────────────────────────────
    # HELPERS
    # ──────────────────────────────────────────────────────────────────────────

    def _add_arrow(self, start, vector, color=(1, 1, 0), scale=1.0):
        """Add a 3D arrow to the scene representing a vector."""
        arrow_source = vtk.vtkArrowSource()

        length = np.linalg.norm(vector)
        if length < 1e-9:
            return

        visual_length = 5.0 * scale

        normalized_vector = vector / length

        transform = vtk.vtkTransform()
        transform.Translate(start[0], start[1], start[2])

        default_dir = np.array([1.0, 0.0, 0.0])
        axis = np.cross(default_dir, normalized_vector)
        angle_rad = np.arccos(np.clip(np.dot(default_dir, normalized_vector), -1.0, 1.0))
        angle_deg = np.degrees(angle_rad)

        if np.linalg.norm(axis) > 1e-6:
            transform.RotateWXYZ(angle_deg, axis)
        elif np.dot(default_dir, normalized_vector) < 0:
            transform.RotateWXYZ(180, [0, 1, 0])

        transform.Scale(visual_length, visual_length, visual_length)

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(arrow_source.GetOutputPort())

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.SetUserTransform(transform)
        actor.GetProperty().SetColor(color)

        self.renderer.AddActor(actor)
        self.actors.append(actor)

    def _add_cube_marker(self, pos, color=(1, 0, 0), size=1.0):
        """Add a cube marker (e.g. for constraints)."""
        source = vtk.vtkCubeSource()
        source.SetCenter(pos[0], pos[1], pos[2])
        source.SetXLength(size)
        source.SetYLength(size)
        source.SetZLength(size)

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(source.GetOutputPort())

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(color)

        self.renderer.AddActor(actor)
        self.actors.append(actor)

    def update_simulation_field(self, mesh, values, field_name="Density"):
        """Update scalar field on existing mesh for real-time visualization."""
        data = {
            'mesh': mesh,
            'visualization_mode': field_name
        }
        if field_name == 'Density':
            data['density'] = values
        elif field_name == 'Von Mises Stress':
            data['stress'] = values

        self.render_simulation(data)

    # ──────────────────────────────────────────────────────────────────────────
    # SIMULATION RENDER
    # ──────────────────────────────────────────────────────────────────────────

    def render_simulation(self, data):
        """
        Render simulation results (Mesh or FEA Result).
        """
        if data is None:
            return

        # Auto-start crash animation playback when solver result with frames arrives
        if (isinstance(data, dict) and
                data.get('type') == 'crash' and
                data.get('frames')):
            self.start_crash_playback(data)
            return

        # 1. Clean up previous render
        if self.current_actor:
            self.renderer.RemoveActor(self.current_actor)
            self.current_actor = None

        # Check if it's a Mesh object or Result dict
        mesh = None
        displacement = None
        density = None
        stress = None
        visualization_mode = 'Von Mises Stress'
        density_cutoff = 0.5
        locked_scalar_range = None   # (lo, hi) supplied by crash playback for stable colormap

        # Detect skfem Mesh
        if hasattr(data, 'p') and hasattr(data, 't'):
            mesh = data
        # Detect Result Dict
        elif isinstance(data, dict) and 'mesh' in data:
            mesh = data['mesh']
            if 'displacement' in data:
                displacement = data['displacement']
            if 'density' in data:
                density = data['density']
            if 'stress' in data:
                stress = data['stress']
            if 'visualization_mode' in data:
                visualization_mode = data['visualization_mode']
            if 'density_cutoff' in data:
                density_cutoff = float(data['density_cutoff'])
            if '_scalar_range' in data:
                locked_scalar_range = data['_scalar_range']

        if mesh is None:
            return

        # 2. Create VTK Unstructured Grid
        points = vtk.vtkPoints()
        grid = vtk.vtkUnstructuredGrid()

        pts = mesh.p
        n_points = pts.shape[1]

        # Apply displacement if available
        if displacement is not None:
            if len(displacement) == 3 * n_points:
                disp_3n = displacement.reshape((3, n_points), order='F')
                pts = pts + disp_3n * 1.0

        for i in range(n_points):
            points.InsertNextPoint(pts[0, i], pts[1, i], pts[2, i])

        grid.SetPoints(points)

        tets = mesh.t
        n_tets = tets.shape[1]

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
            if visualization_mode in ('Von Mises Stress', 'Plastic Strain', 'Failed Elements'):
                grid.GetPointData().SetActiveScalars("VonMises")

        if displacement is not None:
            if len(displacement) == 3 * n_points:
                disp_3n = displacement.reshape((3, n_points), order='F')
                mag = np.linalg.norm(disp_3n, axis=0)
                mag_array = vtk.vtkFloatArray()
                mag_array.SetName("Displacement")
                for m in mag:
                    mag_array.InsertNextValue(m)
                grid.GetPointData().AddArray(mag_array)
                if visualization_mode == 'Displacement':
                    grid.GetPointData().SetActiveScalars("Displacement")

        for i in range(n_tets):
            tet = vtk.vtkTetra()
            for j in range(4):
                tet.GetPointIds().SetId(j, tets[j, i])
            grid.InsertNextCell(tet.GetCellType(), tet.GetPointIds())

        # 3. Mapper and Actor
        mapper = vtk.vtkDataSetMapper()

        if density is not None:
            cutoff = float(np.clip(density_cutoff, 0.05, 0.95))
            lower = max(0.01, cutoff)
            upper = 1.1

            threshold = vtk.vtkThreshold()
            threshold.SetInputData(grid)
            threshold.SetInputArrayToProcess(0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_CELLS, "Density")
            threshold.SetLowerThreshold(lower)
            threshold.SetUpperThreshold(upper)
            threshold.SetThresholdFunction(getattr(vtk.vtkThreshold, 'THRESHOLD_BETWEEN', 0))

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
                    if visualization_mode == 'Plastic Strain':
                        self._update_scalar_bar("Equivalent Plastic Strain", min_val, max_val, lut)
                    elif visualization_mode == 'Failed Elements':
                        self._update_scalar_bar("Element Failure (0=intact, 1=failed)", min_val, max_val, lut)
                    else:
                        self._update_scalar_bar("Von Mises Stress (MPa)", min_val, max_val, lut)
                elif scalar_name == "Displacement":
                    self._update_scalar_bar("Displacement (mm)", min_val, max_val, lut)

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

        if density is not None and stress is not None and visualization_mode == 'Von Mises Stress':
            mapper_input = mapper.GetInput()
            if mapper_input is not None and mapper_input.GetPointData() is not None:
                mapper_input.GetPointData().SetActiveScalars("VonMises")

            mapper.SetScalarModeToUsePointData()
            mapper.SelectColorArray("VonMises")

            # FIX: Skip void elements when calculating stress range for colorbar
            if hasattr(mesh, 't') and mesh.t.shape[1] == len(density) and len(stress) == mesh.p.shape[1]:
                # Nodes connected to solid elements
                solid_elems = np.where(density >= float(np.clip(density_cutoff, 0.05, 0.95)))[0]
                if len(solid_elems) > 0:
                    solid_nodes = np.unique(mesh.t[:, solid_elems])
                    valid_stress = stress[solid_nodes]
                    min_s, max_s = float(np.min(valid_stress)), float(np.max(valid_stress))
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

        if density is not None and visualization_mode == 'Density':
            actor.GetProperty().SetColor(0.9, 0.7, 0.1)
            actor.GetProperty().SetRepresentationToSurface()
            actor.GetProperty().EdgeVisibilityOn()
        elif stress is not None or displacement is not None:
            actor.GetProperty().SetRepresentationToSurface()
            actor.GetProperty().EdgeVisibilityOn()
        else:
            actor.GetProperty().SetColor(0.8, 0.4, 0.4)
            actor.GetProperty().SetRepresentationToSurface()
            actor.GetProperty().EdgeVisibilityOn()

        self.renderer.AddActor(actor)
        self.current_actor = actor
        self.actors.append(actor)

        # 4. Debug Visualization (Loads/Constraints)
        if isinstance(data, dict):
            if 'debug_loads' in data and data['debug_loads']:
                for load in data['debug_loads']:
                    self._add_arrow(load['start'], load['vector'], color=(1, 1, 0), scale=1.0)

            if 'debug_constraints' in data and data['debug_constraints']:
                for const in data['debug_constraints']:
                    self._add_cube_marker(const['pos'], color=(1, 0, 0), size=2.0)

        # 5. Re-apply cached BC face overlays on top of the simulation result
        #    so they remain visible after FEA/TopOpt solve.
        #    (Skip for TopOpt iteration previews to keep frame rate high.)
        skip_bc_replay = isinstance(data, dict) and data.get('type') in ('topopt', 'crash_frame')
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
        if isinstance(data, dict) and data.get('type') == 'crash_frame':
            should_reset = bool(data.get('_reset_camera', False))
        if should_reset:
            self.renderer.ResetCamera()
        self.vtkWidget.GetRenderWindow().Render()