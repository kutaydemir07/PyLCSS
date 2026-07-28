# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE.
"""Embedded CAD/FEA viewer assembled from focused rendering mixins."""

from __future__ import annotations

import logging

import numpy as np
import vtk
from PySide6 import QtCore, QtWidgets
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

from .boundary_visuals import BC_PALETTE
from .navigation_cube import NavCubeWidget
from .viewer_boundaries import BoundaryOverlayMixin
from .viewer_crash import CrashPlaybackMixin
from .viewer_picking import PickingMixin
from .viewer_scene import SceneRenderingMixin
from .viewer_simulation import SimulationRenderingMixin

__all__ = ["CQ3DViewer", "NavCubeWidget"]


class CQ3DViewer(
    CrashPlaybackMixin,
    PickingMixin,
    SceneRenderingMixin,
    BoundaryOverlayMixin,
    SimulationRenderingMixin,
    QtWidgets.QWidget,
):
    """Professional embedded CadQuery/VTK viewer."""

    face_picked = QtCore.Signal(list)
    edge_picked = QtCore.Signal(list)
    vertex_picked = QtCore.Signal(list)
    picking_cancelled = QtCore.Signal()
    face_picking_requested = QtCore.Signal()

    SCALAR_BAR_WIDTH = 0.10
    SCALAR_BAR_RIGHT_MARGIN = 0.012
    SCALAR_BAR_X = 1.0 - SCALAR_BAR_WIDTH - SCALAR_BAR_RIGHT_MARGIN

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._show_bc_labels = True
        self.main_layout = QtWidgets.QVBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)

        # --- View Orientation Toolbar ---
        self._view_toolbar = QtWidgets.QWidget(self)
        self._view_toolbar.setObjectName("view_toolbar")
        self._view_toolbar.setStyleSheet(
            "#view_toolbar { background: rgba(30, 30, 30, 180); border-bottom: 1px solid #444; }"
            "QPushButton { background: transparent; color: #ccc; font-weight: bold; border-radius: 3px; padding: 4px 10px; }"
            "QPushButton:hover { background: rgba(80, 80, 80, 200); color: white; }"
            "QPushButton:checked { background: rgba(32, 126, 168, 210); color: white; }"
        )
        vtb_layout = QtWidgets.QHBoxLayout(self._view_toolbar)
        vtb_layout.setContentsMargins(5, 2, 5, 2)
        vtb_layout.addStretch()

        self._btn_bc_labels = QtWidgets.QPushButton("BC Labels")
        self._btn_bc_labels.setCheckable(True)
        self._btn_bc_labels.setChecked(True)
        self._btn_bc_labels.setToolTip(
            "Show or hide condition text only. Supports, arrows, highlighted "
            "regions, walls, and the condition legend remain visible."
        )
        self._btn_bc_labels.toggled.connect(self._toggle_bc_labels)
        vtb_layout.addWidget(self._btn_bc_labels)

        self._btn_pick_faces = None

        # NOTE: the "Grid" reference-grid toggle was removed from the toolbar —
        # it confused more than it helped.  _toggle_grid / _build_grid_actors
        # remain available for programmatic use but are no longer surfaced.

        self.main_layout.addWidget(self._view_toolbar)

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

        self._pick_icon_lbl = QtWidgets.QLabel(
            "Face Picking Mode  --  Click faces to select"
        )
        self._pick_icon_lbl.setStyleSheet(
            "color: white; font-weight: bold; font-size: 13px;"
        )
        tb_layout.addWidget(self._pick_icon_lbl)

        tb_layout.addStretch()

        self._pick_count_lbl = QtWidgets.QLabel("0 selected")
        self._pick_count_lbl.setStyleSheet(
            "color: #aad4ff; font-size: 12px; margin-right: 16px;"
        )
        tb_layout.addWidget(self._pick_count_lbl)

        self._pick_hint_lbl = QtWidgets.QLabel("Ctrl+Click = multi-select")
        self._pick_hint_lbl.setStyleSheet(
            "color: #aad4ff; font-size: 11px; margin-right: 16px;"
        )
        tb_layout.addWidget(self._pick_hint_lbl)

        self._btn_lock = QtWidgets.QPushButton("🔒 Lock")
        self._btn_lock.setCheckable(True)
        self._btn_lock.setChecked(True)
        self._btn_lock.setToolTip(
            "Locked: left-click picks geometry.\nRotate: left-drag orbits freely."
        )
        self._btn_lock.setStyleSheet(
            "QPushButton { background:#555; color:white; border-radius:4px;"
            "  padding:4px 12px; font-size:12px; }"
            "QPushButton:checked { background:#b8860b; color:#ffe; }"
            "QPushButton:hover { background:#777; }"
        )
        self._btn_lock.toggled.connect(self._on_lock_toggled)
        tb_layout.addWidget(self._btn_lock)

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
        self.scalar_bar.SetWidth(self.SCALAR_BAR_WIDTH)
        self.scalar_bar.SetHeight(0.6)
        # Anchor the entire legend box to the right viewport border. Labels are
        # laid out inside this box, so the model keeps almost all of the canvas.
        self.scalar_bar.SetPosition(self.SCALAR_BAR_X, 0.2)
        self.scalar_bar.VisibilityOff()
        self._style_scalar_bar()
        self.renderer.AddActor(self.scalar_bar)

        render_window = self.vtkWidget.GetRenderWindow()
        render_window.SetNumberOfLayers(2)
        self.renderer.SetLayer(0)
        render_window.AddRenderer(self.renderer)

        # A transparent foreground renderer shares the model camera. Surface
        # fills stay in the main renderer and remain depth-correct; semantic
        # glyphs, outlines and labels use this foreground layer so a support,
        # load arrow, or wall annotation cannot disappear behind the result.
        self.bc_renderer = vtk.vtkRenderer()
        self.bc_renderer.SetLayer(1)
        self.bc_renderer.SetActiveCamera(self.renderer.GetActiveCamera())
        self.bc_renderer.InteractiveOff()
        self.bc_renderer.PreserveColorBufferOn()
        self.bc_renderer.PreserveDepthBufferOff()
        render_window.AddRenderer(self.bc_renderer)
        self.interactor = render_window.GetInteractor()

        # Initialize sophisticated View Axes
        axes = vtk.vtkAxesActor()
        self.orientation_axes = axes
        axes.SetTotalLength(1.0, 1.0, 1.0)
        axes.SetShaftTypeToCylinder()
        axes.SetCylinderRadius(0.02)
        axes.SetConeRadius(0.08)

        # Style the labels
        for txt in [
            axes.GetXAxisCaptionActor2D(),
            axes.GetYAxisCaptionActor2D(),
            axes.GetZAxisCaptionActor2D(),
        ]:
            txt.GetCaptionTextProperty().SetColor(1, 1, 1)
            txt.GetCaptionTextProperty().SetFontFamilyToArial()
            txt.GetCaptionTextProperty().BoldOn()
            txt.GetCaptionTextProperty().ItalicOff()
            txt.GetCaptionTextProperty().ShadowOff()
            txt.GetCaptionTextProperty().SetFontSize(24)
            txt.SetWidth(0.1)

        self.marker_widget = vtk.vtkOrientationMarkerWidget()
        self.marker_widget.SetOrientationMarker(axes)
        self.marker_widget.SetInteractor(self.interactor)
        self.marker_widget.SetViewport(0.0, 0.0, 0.2, 0.2)
        self.marker_widget.SetEnabled(1)
        self.marker_widget.InteractiveOff()  # Prevent user from dragging it around

        # Initialize Grid state
        self._grid_actor = None
        self._axes_actor = None
        self._axis_label_actors = []

        # State
        self.current_actor = None
        self.actors = []  # List of all active actors

        # --- Picking State ---
        self._picking_mode = False
        self._multi_select = True
        self._picked_face_indices = []  # list of int (OCC face indices)
        self._picked_occ_faces = []  # list of OCC face objects
        self._highlight_actors = []  # VTK actors for selected face highlights
        self._face_map = {}  # vtk_cell_id -> occ_face_index
        self._all_occ_faces = []  # list of OCC face objects (from last render_shape)
        self._face_polydata_list = []  # per-face vtkPolyData for highlighting
        self._pickable_surface_dataset = None
        self._pick_callback_id = None
        self._rotation_locked = (
            False  # True → no-op interactor style installed; left-click picks
        )

        # --- Edge Display State ---
        self._show_edges = False
        self._edge_actor = None  # wireframe line actor (OCC edges)

        # --- Edge Picking State ---
        self._edge_picking_mode = False
        self._picked_edge_indices = []  # list of int (OCC edge indices)
        self._picked_occ_edges = []  # list of OCC edge objects
        self._edge_highlight_actors = []  # VTK actors for selected edge highlights
        self._all_occ_edges = []  # list of OCC edge objects (from last render_shape)
        self._edge_pd_list = []  # per-edge vtkPolyData for highlighting
        self._edge_cell_map = {}  # vtk_cell_id -> occ_edge_index (picking actor)
        self._edge_pick_callback_id = None

        # --- Vertex Picking State ---
        self._vertex_picking_mode = False
        self._picked_vertex_indices = []
        self._picked_occ_vertices = []
        self._vertex_highlight_actors = []
        self._all_occ_vertices = []
        self._vertex_actor = None
        self._vertex_pick_callback_id = None

        # --- BC Overlay State ---
        self._bc_overlay_actors = []  # dedicated list for load/support overlay actors
        self._light_mode = False
        self._cached_bc_data = None  # (constraint_faces, load_faces, load_vectors) – replayed after sim render

        # --- Crash Playback State ---
        self._crash_frames = []  # list of frame dicts
        self._crash_base_data = None  # original crash result (mesh, viz_mode, etc.)
        self._crash_frame_idx = 0  # current frame index
        self._crash_playing = False
        self._crash_scalar_range = (0.0, 1.0)  # global min/max for stable colourmap
        self._crash_timer = QtCore.QTimer(self)
        self._crash_timer.timeout.connect(self._on_crash_timer)
        # Rigid impact wall/barrier overlay (from the crash result's 'wall' meta).
        self._crash_wall_actor = None
        self._crash_wall_outline_actor = None
        self._crash_wall_label_actor = None
        self._crash_wall_info = None

        # Crash panel (hidden until crash results arrive)
        self._crash_panel = self._build_crash_panel()
        self._crash_panel.hide()
        self.main_layout.addWidget(self._crash_panel)

        self.interactor.Initialize()
        self.interactor.Start()

        # Cache the default trackball interactor style so we can restore it
        # after picking. A no-op style is installed during locked picking to
        # prevent the camera from rotating while the user clicks faces/edges.
        self._default_interactor_style = self.interactor.GetInteractorStyle()
        self._noop_interactor_style = vtk.vtkInteractorStyleUser()

        # NavCube overlay – replaces the VTK axes orientation marker widget
        self.marker_widget.SetEnabled(0)
        self._nav_cube = NavCubeWidget(self)
        self._nav_cube.view_requested.connect(self._set_camera_view)
        self._nav_cube.roll_requested.connect(self._roll_camera)
        self._nav_cube.raise_()
        self._nav_cube.show()

        self._position_nav_cube()
        self.vtkWidget.GetRenderWindow().AddObserver(
            "EndEvent", lambda o, e: self._on_vtk_render()
        )

    def _toggle_bc_labels(self, visible):
        """Toggle condition billboards without changing engineering geometry."""
        self._show_bc_labels = bool(visible)
        actors = list(getattr(self, "_bc_overlay_actors", []))
        actors.extend(getattr(self, "actors", []))
        crash_label = getattr(self, "_crash_wall_label_actor", None)
        if crash_label is not None:
            actors.append(crash_label)
        seen = set()
        for actor in actors:
            if actor is None or id(actor) in seen:
                continue
            seen.add(id(actor))
            if getattr(actor, "_bc_text_label", False):
                actor.SetVisibility(1 if self._show_bc_labels else 0)
        try:
            self.vtkWidget.GetRenderWindow().Render()
        except Exception:
            logging.getLogger(__name__).debug(
                "Optional UI operation failed.", exc_info=True
            )

    def apply_theme(self, theme):
        """Update the VTK canvas, legends, and orientation overlay."""
        light = str(theme).strip().lower() == "light"
        self._light_mode = light
        text_color = (0.12, 0.14, 0.17) if light else (0.90, 0.92, 0.96)
        self.renderer.SetBackground(
            *(0.91, 0.93, 0.95) if light else (0.20, 0.20, 0.20)
        )
        self._scalar_text_color = text_color
        try:
            self._nav_cube.set_light_mode(light)
        except Exception:
            logging.getLogger(__name__).debug(
                "Optional UI operation failed.", exc_info=True
            )
        try:
            for caption in (
                self.orientation_axes.GetXAxisCaptionActor2D(),
                self.orientation_axes.GetYAxisCaptionActor2D(),
                self.orientation_axes.GetZAxisCaptionActor2D(),
            ):
                caption.GetCaptionTextProperty().SetColor(*text_color)
        except Exception:
            logging.getLogger(__name__).debug(
                "Optional UI operation failed.", exc_info=True
            )
        try:
            if self._crash_wall_label_actor is not None:
                wall_prop = self._crash_wall_label_actor.GetTextProperty()
                wall_color = np.asarray(BC_PALETTE["impact"], dtype=float)
                if light:
                    wall_color *= 0.56
                wall_prop.SetColor(*wall_color)
                wall_prop.SetBackgroundColor(
                    *(0.95, 0.97, 0.99) if light else (0.04, 0.05, 0.07)
                )
        except Exception:
            logging.getLogger(__name__).debug(
                "Optional UI operation failed.", exc_info=True
            )
        self._style_scalar_bar()
        try:
            if self._cached_bc_data is not None:
                constraints, faces, vectors = self._cached_bc_data
                self.render_bc_overlays(
                    constraint_faces=constraints or None,
                    load_faces=faces or None,
                    load_vectors=vectors or None,
                )
            else:
                self.vtkWidget.GetRenderWindow().Render()
        except Exception:
            logging.getLogger(__name__).debug(
                "Optional UI operation failed.", exc_info=True
            )
