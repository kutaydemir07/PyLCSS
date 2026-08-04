# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE.
"""Embedded CAD/FEA viewer assembled from focused rendering mixins."""

from __future__ import annotations

import logging

import numpy as np
import vtk
from PySide6 import QtCore, QtGui, QtWidgets
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

from .boundary_visuals import BC_PALETTE
from .navigation_cube import NavCubeWidget
from .viewer_boundaries import BoundaryOverlayMixin
from .viewer_crash import CrashPlaybackMixin
from .viewer_interaction import ViewerInteractionMixin
from .viewer_picking import PickingMixin
from .viewer_quality import RenderQualityMixin
from .viewer_scene import SceneRenderingMixin
from .viewer_simulation import SimulationRenderingMixin

__all__ = ["CQ3DViewer", "NavCubeWidget"]


class CQ3DViewer(
    CrashPlaybackMixin,
    PickingMixin,
    RenderQualityMixin,
    ViewerInteractionMixin,
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
    result_field_changed = QtCore.Signal(str)

    SCALAR_BAR_WIDTH = 0.10
    SCALAR_BAR_RIGHT_MARGIN = 0.012
    SCALAR_BAR_X = 1.0 - SCALAR_BAR_WIDTH - SCALAR_BAR_RIGHT_MARGIN

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._show_bc_overlay = True
        self._show_bc_labels = True
        self.main_layout = QtWidgets.QVBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)

        # --- View Orientation Toolbar ---
        self._build_view_toolbar()

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
        camera = self.renderer.GetActiveCamera()
        camera.ParallelProjectionOff()
        camera.SetPosition(1.0, -1.0, 1.0)
        camera.SetFocalPoint(0.0, 0.0, 0.0)
        camera.SetViewUp(0.0, 0.0, 1.0)
        camera.OrthogonalizeViewUp()

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
        # Shaded-with-edges is the default display style in every commercial
        # CAD/CAE package, so the viewer opens that way too.
        self._show_edges = True
        self._edge_actor = None  # wireframe line actor (OCC edges)
        self._show_undeformed = False
        self._undeformed_actor = None
        self._section_enabled = False
        self._section_plane = vtk.vtkPlane()
        # Live section-clip pipeline. The unclipped dataset and its bounds are
        # cached so the actor can be restored, and so the slider keeps mapping
        # to the whole model rather than to whatever is left after the cut.
        self._section_actor = None
        self._section_mapper = None
        self._section_source = None
        self._section_clipper = None
        self._section_bounds = None
        # A surface actor extracted from a volume mesh registers that volume
        # here. Clipping the skin can only ever expose the far side of a shell,
        # so the section prefers the volume whenever one is available.
        self._section_volume_actor = None
        self._section_volume_source = None
        self._last_simulation_mesh = None
        self._last_result_data = None
        self._active_result_mode = None
        self._updating_field_selector = False
        # Results use a continuous colour ramp by default.  Discrete bands are
        # still supported internally for exported/specialist views, but they
        # are not exposed as a raw numeric control in the primary toolbar.
        self._contour_bands = None
        self._show_extrema = False
        self._result_extrema_cache = []
        self._result_extrema_actors = []

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

        # Camera/pointer behaviour, then presentation quality. The interactor
        # has to be initialised first: both attach observers to it, and
        # _install_cad_interactor_style replaces the style cached just above.
        self._init_interaction()
        self._install_cad_interactor_style()
        self._init_render_quality()

        # NavCube overlay – replaces the VTK axes orientation marker widget
        self.marker_widget.SetEnabled(0)
        self._nav_cube = NavCubeWidget(self)
        self._nav_cube.view_requested.connect(self._set_camera_view)
        self._nav_cube.roll_requested.connect(self._roll_camera)
        self._nav_cube.update_rotation(self.renderer.GetActiveCamera())
        self._nav_cube.raise_()
        self._nav_cube.show()

        self.ensure_navigation_cube_visible()
        self.vtkWidget.GetRenderWindow().AddObserver(
            "EndEvent", lambda o, e: self._on_vtk_render()
        )

    # ------------------------------------------------------------------
    # view toolbar
    # ------------------------------------------------------------------
    @staticmethod
    def _toolbar_icon(name, color="#d6dae1"):
        """qtawesome icon, or a blank one when the package is unavailable.

        qtawesome is an optional dependency here, exactly as in the Design
        Studio toolbar; the actions keep their text and tooltips either way.
        """
        try:
            import qtawesome as qta

            return qta.icon(name, color=color)
        except Exception:
            logging.getLogger(__name__).debug(
                "Icon theme unavailable.", exc_info=True
            )
            return QtGui.QIcon()

    @staticmethod
    def _view_toolbar_stylesheet(light):
        """Toolbar chrome for the active theme."""
        if light:
            return """
            #view_toolbar {
                background: rgba(244, 246, 249, 230);
                border: 0; border-bottom: 1px solid rgba(0,0,0,34);
                spacing: 2px; padding: 3px 6px;
            }
            #view_toolbar QToolButton {
                padding: 4px 6px; border-radius: 4px; color: #2b313a;
            }
            #view_toolbar QToolButton:hover { background: rgba(0,0,0,20); }
            #view_toolbar QToolButton:checked {
                background: rgba(32, 126, 168, 200); color: #ffffff;
            }
            #view_toolbar QToolBar::separator {
                background: rgba(0,0,0,38); width: 1px; margin: 5px 5px;
            }
            #view_toolbar QLabel { color: #4a5361; padding-left: 6px; }
            """
        return """
            #view_toolbar {
                background: rgba(30, 32, 36, 200);
                border: 0; border-bottom: 1px solid rgba(255,255,255,26);
                spacing: 2px; padding: 3px 6px;
            }
            #view_toolbar QToolButton {
                padding: 4px 6px; border-radius: 4px; color: #d6dae1;
            }
            #view_toolbar QToolButton:hover { background: rgba(255,255,255,22); }
            #view_toolbar QToolButton:checked {
                background: rgba(32, 126, 168, 210); color: #ffffff;
            }
            #view_toolbar QToolBar::separator {
                background: rgba(255,255,255,30); width: 1px; margin: 5px 5px;
            }
            #view_toolbar QLabel { color: #aeb7c2; padding-left: 6px; }
            """

    def _restyle_toolbar_icons(self, light):
        """Recolour the toolbar glyphs for the active theme."""
        color = "#2b313a" if light else "#d6dae1"
        for button, icon_name in getattr(self, "_toolbar_icon_names", ()):
            try:
                button.setIcon(self._toolbar_icon(icon_name, color))
            except Exception:
                logging.getLogger(__name__).debug(
                    "Optional UI operation failed.", exc_info=True
                )

    def _add_toolbar_widget(self, widget):
        """Add a widget to the view toolbar and remember its owning action.

        A widget inside a QToolBar is wrapped in a QAction; hiding the widget
        alone leaves the wrapper behind and the toolbar keeps its gap, so the
        action is what visibility has to be driven through.
        """
        action = self._view_toolbar.addWidget(widget)
        self._toolbar_actions[widget] = action
        return action

    def _set_toolbar_widget_visible(self, widget, visible):
        action = self._toolbar_actions.get(widget)
        if action is not None:
            action.setVisible(bool(visible))
        widget.setVisible(bool(visible))

    def _build_view_toolbar(self):
        """Icon toolbar grouped the way CAD/CAE packages group these commands.

        Replaces the previous row of text push-buttons: labelled buttons for
        display toggles are the clearest visual difference from commercial
        software, and the grouping (view / display / result) is what makes a
        dense toolbar readable.
        """
        self._toolbar_actions = {}

        self._view_toolbar = QtWidgets.QToolBar(self)
        self._view_toolbar.setObjectName("view_toolbar")
        self._view_toolbar.setMovable(False)
        self._view_toolbar.setIconSize(QtCore.QSize(16, 16))
        self._view_toolbar.setToolButtonStyle(QtCore.Qt.ToolButtonIconOnly)
        self._view_toolbar.setStyleSheet(
            self._view_toolbar_stylesheet(getattr(self, "_light_mode", False))
        )

        # ── View ────────────────────────────────────────────────────────
        self._btn_fit = QtWidgets.QToolButton()
        self._btn_fit.setIcon(self._toolbar_icon("fa5s.expand"))
        self._btn_fit.setToolTip("Fit all visible geometry in the 3-D view (F)")
        self._btn_fit.clicked.connect(self._fit_visible_geometry)
        self._add_toolbar_widget(self._btn_fit)

        self._view_toolbar.addSeparator()

        # ── Display ─────────────────────────────────────────────────────
        self._btn_edges = QtWidgets.QToolButton()
        self._btn_edges.setIcon(self._toolbar_icon("fa5s.border-all"))
        self._btn_edges.setCheckable(True)
        self._btn_edges.setChecked(True)
        self._btn_edges.setToolTip(
            "Show or hide CAD feature edges and analysis element edges. "
            "Shaded-with-edges is the default display style."
        )
        self._btn_edges.toggled.connect(self._toggle_edges)
        self._add_toolbar_widget(self._btn_edges)

        self._btn_bc_overlay = QtWidgets.QToolButton()
        self._btn_bc_overlay.setIcon(self._toolbar_icon("fa5s.anchor"))
        self._btn_bc_overlay.setCheckable(True)
        self._btn_bc_overlay.setChecked(True)
        self._btn_bc_overlay.setToolTip(
            "Show or hide the whole boundary-condition overlay: supports, "
            "loads, highlighted regions, rigid walls, text, and the legend. "
            "Display only — the conditions sent to the solver are unchanged."
        )
        self._btn_bc_overlay.toggled.connect(self._toggle_bc_overlay)
        self._add_toolbar_widget(self._btn_bc_overlay)

        self._btn_bc_labels = QtWidgets.QToolButton()
        self._btn_bc_labels.setIcon(self._toolbar_icon("fa5s.tags"))
        self._btn_bc_labels.setCheckable(True)
        self._btn_bc_labels.setChecked(True)
        self._btn_bc_labels.setToolTip(
            "Show or hide condition text only. Supports, arrows, highlighted "
            "regions, walls, and the condition legend remain visible."
        )
        self._btn_bc_labels.toggled.connect(self._toggle_bc_labels)
        self._add_toolbar_widget(self._btn_bc_labels)

        self._btn_section = QtWidgets.QToolButton()
        self._btn_section.setIcon(self._toolbar_icon("fa5s.cut"))
        self._btn_section.setCheckable(True)
        self._btn_section.setToolTip(
            "Clip the active CAD or result actor with an interactive X, Y, or "
            "Z section plane. This is display-only and does not change results."
        )
        self._btn_section.toggled.connect(self._toggle_section_cut)
        self._add_toolbar_widget(self._btn_section)

        self._section_axis = QtWidgets.QComboBox()
        self._section_axis.addItems(["X", "Y", "Z"])
        self._section_axis.setMaximumWidth(48)
        self._section_axis.setToolTip("Section-plane normal")
        self._section_axis.currentTextChanged.connect(self._update_section_cut)
        self._add_toolbar_widget(self._section_axis)
        self._set_toolbar_widget_visible(self._section_axis, False)

        self._section_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self._section_slider.setRange(0, 100)
        self._section_slider.setValue(50)
        self._section_slider.setMaximumWidth(110)
        self._section_slider.setToolTip(
            "Move the section plane through the active actor."
        )
        self._section_slider.valueChanged.connect(self._update_section_cut)
        self._add_toolbar_widget(self._section_slider)
        self._set_toolbar_widget_visible(self._section_slider, False)

        self._view_toolbar.addSeparator()

        # ── Result post-processing ──────────────────────────────────────
        self._btn_extrema = QtWidgets.QToolButton()
        self._btn_extrema.setIcon(self._toolbar_icon("fa5s.crosshairs"))
        self._btn_extrema.setCheckable(True)
        self._btn_extrema.setToolTip(
            "Mark the physical locations of the minimum and maximum active "
            "result value. Markers update with each impact-animation frame."
        )
        self._btn_extrema.toggled.connect(self._toggle_result_extrema)
        self._add_toolbar_widget(self._btn_extrema)

        self._btn_undeformed = QtWidgets.QToolButton()
        self._btn_undeformed.setIcon(self._toolbar_icon("fa5s.clone"))
        self._btn_undeformed.setCheckable(True)
        self._btn_undeformed.setToolTip(
            "Superimpose the undeformed analysis mesh as a neutral wireframe."
        )
        self._btn_undeformed.toggled.connect(self._toggle_undeformed_overlay)
        self._add_toolbar_widget(self._btn_undeformed)

        self._field_label = QtWidgets.QLabel("Field")
        self._add_toolbar_widget(self._field_label)
        self._set_toolbar_widget_visible(self._field_label, False)
        self._field_combo = QtWidgets.QComboBox()
        self._field_combo.setToolTip(
            "Switch among fields already present in the solver result. "
            "This does not re-run the analysis."
        )
        self._field_combo.setMinimumContentsLength(15)
        self._field_combo.setMaximumWidth(230)
        self._field_combo.currentTextChanged.connect(self._switch_result_field)
        self._add_toolbar_widget(self._field_combo)
        self._set_toolbar_widget_visible(self._field_combo, False)

        # Deformation scale is a post-processing setting, not an analysis
        # input.  It re-renders the cached result and never touches the graph.
        self._scale_label = QtWidgets.QLabel("Deform")
        self._add_toolbar_widget(self._scale_label)
        self._set_toolbar_widget_visible(self._scale_label, False)
        self._scale_combo = QtWidgets.QComboBox()
        self._scale_combo.setToolTip(
            "Displacement exaggeration used for the deformed shape. "
            "'True (1x)' shows real scale; 'Auto' scales the peak "
            "displacement to 5% of the model size. Display only — the "
            "solved displacements are unchanged."
        )
        self._scale_combo.addItems(
            ["Auto", "True (1x)", "2x", "5x", "10x", "25x", "50x", "100x", "200x"]
        )
        self._scale_combo.setMaximumWidth(110)
        self._scale_combo.currentTextChanged.connect(self._switch_deformation_scale)
        self._add_toolbar_widget(self._scale_combo)
        self._set_toolbar_widget_visible(self._scale_combo, False)

        spacer = QtWidgets.QWidget()
        spacer.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred
        )
        self._view_toolbar.addWidget(spacer)

        # Recoloured on theme change — qtawesome bakes the colour into the
        # pixmap, so the icons have to be rebuilt rather than restyled.
        self._toolbar_icon_names = (
            (self._btn_fit, "fa5s.expand"),
            (self._btn_edges, "fa5s.border-all"),
            (self._btn_bc_overlay, "fa5s.anchor"),
            (self._btn_bc_labels, "fa5s.tags"),
            (self._btn_section, "fa5s.cut"),
            (self._btn_extrema, "fa5s.crosshairs"),
            (self._btn_undeformed, "fa5s.clone"),
        )

        self._btn_pick_faces = None

        # NOTE: the "Grid" reference-grid toggle was removed from the toolbar —
        # it confused more than it helped.  _toggle_grid / _build_grid_actors
        # remain available for programmatic use but are no longer surfaced.

        self.main_layout.addWidget(self._view_toolbar)

    def _iter_bc_overlay_actors(self):
        """Yield every prop the condition overlay owns, each one once.

        Constraint glyphs and the rigid wall are added to the generic actor
        lists rather than to ``_bc_overlay_actors``, so tag-based collection is
        what keeps the display switches from missing part of the overlay.
        """
        seen = set()
        candidates = list(getattr(self, "_bc_overlay_actors", []))
        candidates.extend(
            actor
            for actor in getattr(self, "actors", [])
            if getattr(actor, "_bc_overlay", False)
            or getattr(actor, "_bc_kind", None)
            or getattr(actor, "_bc_result_fallback", False)
            or getattr(actor, "_bc_text_label", False)
        )
        candidates.extend(
            getattr(self, name, None)
            for name in (
                "_crash_wall_actor",
                "_crash_wall_outline_actor",
                "_crash_wall_label_actor",
            )
        )
        for actor in candidates:
            if actor is None or id(actor) in seen:
                continue
            seen.add(id(actor))
            yield actor

    def _bc_text_visible(self):
        """True when condition text passes both display switches."""
        return bool(getattr(self, "_show_bc_overlay", True)) and bool(
            getattr(self, "_show_bc_labels", True)
        )

    def _apply_bc_visibility(self, render=True):
        """Apply the overlay switch and its text sub-switch to every prop.

        Overlays are rebuilt on every graph selection and every result render,
        so the switches are re-applied there instead of being stored per actor.
        """
        show_overlay = bool(getattr(self, "_show_bc_overlay", True))
        show_text = self._bc_text_visible()
        for actor in self._iter_bc_overlay_actors():
            is_text = getattr(actor, "_bc_text_label", False)
            actor.SetVisibility(1 if (show_text if is_text else show_overlay) else 0)
        if not render:
            return
        try:
            self.vtkWidget.GetRenderWindow().Render()
        except Exception:
            logging.getLogger(__name__).debug(
                "Optional UI operation failed.", exc_info=True
            )

    def _toggle_bc_overlay(self, visible):
        """Toggle the whole condition overlay without changing the model."""
        self._show_bc_overlay = bool(visible)
        # The text switch only has meaning while the overlay is drawn.
        button = getattr(self, "_btn_bc_labels", None)
        if button is not None:
            button.setEnabled(self._show_bc_overlay)
        self._apply_bc_visibility()

    def _toggle_bc_labels(self, visible):
        """Toggle condition billboards without changing engineering geometry."""
        self._show_bc_labels = bool(visible)
        self._apply_bc_visibility()

    def _toggle_edges(self, visible):
        """Toggle CAD feature and result-mesh edges without re-solving.

        The two cases need different sources.  On an analysis mesh the cell
        edges *are* the element edges, so the property flag is right.  On a
        CAD solid the cells are tessellation triangles — drawing them covers
        every planar face in a fan of meaningless lines — so the B-rep edge
        actor built by ``_build_edge_actor`` is the only correct source.
        """
        self._show_edges = bool(visible)
        actor = self.current_actor
        if actor is not None:
            is_tessellated_cad = getattr(actor, "_pylcss_material", None) == "cad"
            try:
                actor.GetProperty().SetEdgeVisibility(
                    1 if (self._show_edges and not is_tessellated_cad) else 0
                )
            except Exception:
                logging.getLogger(__name__).debug(
                    "Could not toggle result edges.", exc_info=True
                )
        if self._edge_actor is not None:
            self._edge_actor.SetVisibility(1 if self._show_edges else 0)
        try:
            self.vtkWidget.GetRenderWindow().Render()
        except Exception:
            logging.getLogger(__name__).debug(
                "Optional UI operation failed.", exc_info=True
            )

    def _fit_visible_geometry(self):
        """Reset the active camera to all visible engineering actors."""
        try:
            self.renderer.ResetCamera()
            self._update_scene_scale()
            self.vtkWidget.GetRenderWindow().Render()
        except Exception:
            logging.getLogger(__name__).debug(
                "Optional UI operation failed.", exc_info=True
            )

    def _configure_result_fields(self, data):
        """Expose post-processing fields that are already cached in a result."""
        if not isinstance(data, dict) or data.get("type") in {
            "crash",
            "crash_frame",
        }:
            self._set_toolbar_widget_visible(self._field_label, False)
            self._set_toolbar_widget_visible(self._field_combo, False)
            if not isinstance(data, dict):
                self._last_result_data = None
            return

        fields = []
        if data.get("stress") is not None:
            fields.append("Von Mises Stress")
        tensor = np.asarray(data.get("stress_tensor", []))
        if tensor.ndim == 2 and tensor.shape[1] >= 6 and tensor.shape[0]:
            fields.extend(
                [
                    "Maximum Principal Stress",
                    "Minimum Principal Stress",
                    "Stress XX",
                    "Stress YY",
                    "Stress ZZ",
                    "Stress XY",
                    "Stress YZ",
                    "Stress ZX",
                ]
            )
        if data.get("displacement") is not None:
            fields.append("Displacement")
        # One shape view for a TopOpt result: the B-rep and the recovered
        # surface are the same geometry behind STEP and STL, so they share the
        # "CAD" entry and the renderer falls back to the surface until the
        # B-rep is available.
        structure_options = data.get("structure_options")
        is_lattice_structure = bool(
            structure_options is not None
            and getattr(structure_options, "mode", "solid") != "solid"
        )
        if data.get("type") == "topopt_voxel" and (
            data.get("cad_shape") is not None
            or data.get("shape") is not None
            or data.get("recovered_shape") is not None
        ):
            fields.append(
                "Manufactured Mesh" if is_lattice_structure else "CAD"
            )
        if data.get("density") is not None:
            fields.append("Density")
        validation = data.get("validation")
        if (
            data.get("type") == "topopt_voxel"
            and isinstance(validation, dict)
        ):
            if validation.get("stress") is not None:
                fields.append("Validated Von Mises Stress")
            if validation.get("displacement") is not None:
                fields.append("Validated Displacement")
        if not fields:
            self._set_toolbar_widget_visible(self._field_label, False)
            self._set_toolbar_widget_visible(self._field_combo, False)
            return

        self._last_result_data = dict(data)
        current = str(data.get("visualization_mode") or fields[0])
        current = {
            "Recovered Shape": (
                "Manufactured Mesh" if is_lattice_structure else "CAD"
            ),
            "Recovered Surface (Mesh)": (
                "Manufactured Mesh" if is_lattice_structure else "CAD"
            ),
            "Reconstructed CAD (B-rep)": "CAD",
            "Voxel Density": "Density",
        }.get(current, current)
        if data.get("type") == "topopt_voxel":
            if current == "Surface":
                current = (
                    "Manufactured Mesh" if is_lattice_structure else "CAD"
                )
            elif "density" in current.lower() or "voxel" in current.lower():
                current = "Density"
        if current not in fields:
            current = fields[0]
        self._updating_field_selector = True
        try:
            self._field_combo.clear()
            self._field_combo.addItems(fields)
            self._field_combo.setCurrentText(current)
        finally:
            self._updating_field_selector = False
        visible = len(fields) > 1
        self._set_toolbar_widget_visible(self._field_label, visible)
        self._set_toolbar_widget_visible(self._field_combo, visible)

        # Deformation scale is meaningful only when the result carries a
        # displacement field.
        has_displacement = data.get("displacement") is not None
        self._set_toolbar_widget_visible(self._scale_label, has_displacement)
        self._set_toolbar_widget_visible(self._scale_combo, has_displacement)

    def _switch_result_field(self, field_name):
        """Re-render one cached result field without solving again."""
        if self._updating_field_selector or not self._last_result_data:
            return
        # The workbench owns the live node result and any lazy CAD
        # reconstruction. Tell it about toolbar changes as well as rendering a
        # viewer-local copy; otherwise selecting CAD here can only show the
        # recovered-surface fallback and the result card remains on Density.
        self.result_field_changed.emit(str(field_name))
        payload = dict(self._last_result_data)
        payload["visualization_mode"] = str(field_name)
        self.render_simulation(payload)

    def _switch_deformation_scale(self, choice):
        """Re-render the cached result at a different display exaggeration."""
        if self._updating_field_selector or not self._last_result_data:
            return
        text = str(choice).strip()
        if text.startswith("True"):
            scale = 1.0
        elif text.lower() == "auto":
            scale = self._last_result_data.get("auto_deformation_scale") or 1.0
        else:
            try:
                scale = float(text.rstrip("xX"))
            except ValueError:
                return
        payload = dict(self._last_result_data)
        payload["deformation_scale"] = float(scale)
        payload["disp_scale"] = float(scale)
        # Keep whichever field the user is currently looking at.
        payload["visualization_mode"] = (
            self._active_result_mode
            or payload.get("visualization_mode")
        )
        self.render_simulation(payload)

    def _switch_contour_bands(self, choice):
        """Set discrete contour banding and re-render the cached result."""
        text = str(choice).strip()
        self._contour_bands = None if text == "Smooth" else int(text)
        if self._updating_field_selector or not self._last_result_data:
            return
        payload = dict(self._last_result_data)
        payload["visualization_mode"] = (
            self._active_result_mode
            or payload.get("visualization_mode")
        )
        self.render_simulation(payload)

    def _toggle_undeformed_overlay(self, visible):
        """Show or hide the cached undeformed analysis-mesh wireframe."""
        self._show_undeformed = bool(visible)
        if (
            self._show_undeformed
            and self._undeformed_actor is None
            and self._last_simulation_mesh is not None
        ):
            self._render_undeformed_overlay(self._last_simulation_mesh)
        if self._undeformed_actor is not None:
            self._undeformed_actor.SetVisibility(
                1 if self._show_undeformed else 0
            )
        try:
            self.vtkWidget.GetRenderWindow().Render()
        except Exception:
            logging.getLogger(__name__).debug(
                "Optional UI operation failed.", exc_info=True
            )

    def _toggle_section_cut(self, visible):
        """Enable a display-only clipping plane on the active engineering actor."""
        self._section_enabled = bool(visible)
        self._set_toolbar_widget_visible(self._section_axis, self._section_enabled)
        self._set_toolbar_widget_visible(self._section_slider, self._section_enabled)
        self._update_section_cut()

    def _clear_section_clip(self):
        """Put a previously sectioned actor back on its unclipped dataset."""
        mapper = self._section_mapper
        source = self._section_source
        self._section_actor = None
        self._section_mapper = None
        self._section_source = None
        self._section_clipper = None
        self._section_bounds = None
        if mapper is None or source is None:
            return
        try:
            mapper.RemoveAllInputConnections(0)
            mapper.SetInputData(source)
        except Exception:
            logging.getLogger(__name__).debug(
                "Could not restore the unsectioned dataset.", exc_info=True
            )

    def register_section_volume(self, actor, dataset):
        """Record the volume mesh a surface actor was extracted from.

        The result renderers hand the mapper an extracted skin because drawing
        the full volume every frame is wasteful.  The section, however, needs
        the interior back — see ``_arm_section_clip``.
        """
        self._section_volume_actor = actor
        self._section_volume_source = dataset

    def forget_section_state(self):
        """Drop cached section datasets when the actor they belong to goes.

        Teardown removes the actor and its mapper connections itself, so this
        only releases the references — unlike ``_clear_section_clip`` it does
        not try to restore a mapper that is on its way out.
        """
        self._section_actor = None
        self._section_mapper = None
        self._section_source = None
        self._section_clipper = None
        self._section_bounds = None
        self._section_volume_actor = None
        self._section_volume_source = None

    def _section_input_for(self, actor):
        """Return the volume registered for this actor, or None.

        Cutting a skin can only expose the far side of a shell, whereas
        cutting the volume regenerates the elements the plane passes through
        and carries the result field onto the cut face.
        """
        if (
            self._section_volume_actor is not actor
            or self._section_volume_source is None
        ):
            return None
        try:
            if self._section_volume_source.GetNumberOfCells() > 0:
                return self._section_volume_source
        except Exception:
            logging.getLogger(__name__).debug(
                "Ignoring an unusable section volume.", exc_info=True
            )
        return None

    def _build_section_pipeline(self, dataset, keep_scalars=False):
        """Return a clip filter for ``dataset``, or None if it cannot be cut.

        Every branch keeps the half-space the plane normal points into, so the
        slider direction is the same whichever one runs.
        """
        if not isinstance(dataset, vtk.vtkPolyData):
            # Volume mesh: the clip regenerates the cut cells and interpolates
            # point data onto them, so the cut face shows the result field.
            clipper = vtk.vtkTableBasedClipDataSet()
            clipper.SetInputData(dataset)
            clipper.SetClipFunction(self._section_plane)
            return clipper
        if dataset.GetNumberOfPolys() == 0:
            # Sketches and wireframes have no faces to cut.
            return None
        if keep_scalars:
            # vtkClipClosedSurface discards point data, which would drop the
            # colour map. Keeping the field beats capping the opening, so cut
            # without a cap here; a surface carrying a rendered field is a
            # fallback path anyway — results normally section as a volume.
            #
            # Decided from the mapper, not from the dataset: an isosurface
            # carries the contour scalars it was built from even when the
            # mapper ignores them, and testing the array alone sent every
            # topology result down this uncapped branch.
            clipper = vtk.vtkClipPolyData()
            clipper.SetInputData(dataset)
            clipper.SetClipFunction(self._section_plane)
            return clipper
        # A closed surface has no interior to reveal, so cap the opening.
        # vtkPolyDataPlaneClipper only caps convex single-loop cross-sections
        # and emits loose line segments otherwise, which is why a cut through
        # a bracket with a hole used to read as hollow; vtkClipClosedSurface
        # triangulates multi-loop cross-sections into a real cap.
        planes = vtk.vtkPlaneCollection()
        planes.AddItem(self._section_plane)
        clipper = vtk.vtkClipClosedSurface()
        clipper.SetInputData(dataset)
        clipper.SetClippingPlanes(planes)
        clipper.GenerateFacesOn()
        clipper.GenerateOutlineOff()
        clipper.SetScalarModeToNone()
        return clipper

    def _arm_section_clip(self, actor):
        """Point the actor's mapper at a real clip of its own dataset.

        A mapper clipping plane only discards fragments at render time, so a
        cut through a closed surface shows nothing but the far side of the
        shell.  Clipping the dataset instead regenerates the cells the plane
        passes through, which is what exposes the element faces inside a
        volume mesh and gives a solid cut face.
        """
        mapper = actor.GetMapper()
        if mapper is None:
            return False
        try:
            mapper.Update()
            # Captured before the swap: this is what the actor goes back to
            # when the section is switched off, and it is the displayed
            # surface even when the cut itself runs on the volume.
            displayed = mapper.GetInput()
        except Exception:
            logging.getLogger(__name__).debug(
                "Could not read the dataset to section.", exc_info=True
            )
            return False
        if displayed is None:
            return False
        dataset = self._section_input_for(actor) or displayed
        bounds = dataset.GetBounds()
        if bounds is None or len(bounds) != 6 or bounds[0] > bounds[1]:
            return False
        try:
            keep_scalars = bool(
                mapper.GetScalarVisibility()
                and dataset.GetPointData().GetScalars() is not None
            )
        except Exception:
            keep_scalars = False
        try:
            tail = self._build_section_pipeline(dataset, keep_scalars)
            if tail is None:
                return False
            mapper.SetInputConnection(tail.GetOutputPort())
        except Exception:
            logging.getLogger(__name__).debug(
                "Could not build the section clip pipeline.", exc_info=True
            )
            return False
        self._section_actor = actor
        self._section_mapper = mapper
        self._section_source = displayed
        self._section_clipper = tail
        self._section_bounds = tuple(bounds)
        return True

    def _update_section_cut(self, *_args):
        actor = self.current_actor
        if actor is None or actor.GetMapper() is None:
            return
        try:
            if not self._section_enabled:
                self._clear_section_clip()
                self.vtkWidget.GetRenderWindow().Render()
                return
            if (
                self._section_actor is not actor
                or self._section_mapper is not actor.GetMapper()
            ):
                self._clear_section_clip()
                if not self._arm_section_clip(actor):
                    return
            # Positioned against the unclipped bounds: the actor's own bounds
            # shrink as the cut advances, which would make the slider drift.
            bounds = self._section_bounds
            axis = {"X": 0, "Y": 1, "Z": 2}.get(
                self._section_axis.currentText(),
                0,
            )
            fraction = 0.01 * float(self._section_slider.value())
            origin = [
                0.5 * (bounds[0] + bounds[1]),
                0.5 * (bounds[2] + bounds[3]),
                0.5 * (bounds[4] + bounds[5]),
            ]
            origin[axis] = bounds[2 * axis] + fraction * (
                bounds[2 * axis + 1] - bounds[2 * axis]
            )
            normal = [0.0, 0.0, 0.0]
            normal[axis] = 1.0
            # The clip filters observe this plane, so moving it re-executes
            # them on the next render without rebuilding the pipeline.
            self._section_plane.SetOrigin(*origin)
            self._section_plane.SetNormal(*normal)
            self.vtkWidget.GetRenderWindow().Render()
        except Exception:
            logging.getLogger(__name__).debug(
                "Could not update section plane.", exc_info=True
            )

    def _toggle_result_extrema(self, visible):
        """Show or hide cached extrema markers without re-running the study."""
        self._show_extrema = bool(visible)
        if self._show_extrema and not self._result_extrema_actors:
            self._render_result_extrema()
        for actor in self._result_extrema_actors:
            actor.SetVisibility(1 if self._show_extrema else 0)
        try:
            self.vtkWidget.GetRenderWindow().Render()
        except Exception:
            logging.getLogger(__name__).debug(
                "Could not toggle result extrema.", exc_info=True
            )

    def apply_theme(self, theme):
        """Update the VTK canvas, legends, and orientation overlay."""
        light = str(theme).strip().lower() == "light"
        self._light_mode = light
        text_color = (0.12, 0.14, 0.17) if light else (0.90, 0.92, 0.96)
        self._scalar_text_color = text_color
        self._apply_viewport_background(light)
        try:
            self._view_toolbar.setStyleSheet(self._view_toolbar_stylesheet(light))
            self._restyle_toolbar_icons(light)
        except Exception:
            logging.getLogger(__name__).debug(
                "Optional UI operation failed.", exc_info=True
            )
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
