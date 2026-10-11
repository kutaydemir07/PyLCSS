# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE.
"""Top-level Design Studio workbench assembled from focused mixins."""

from __future__ import annotations

import logging
import os
import sys

from NodeGraphQt import NodeGraph
from PySide6 import QtCore, QtGui, QtWidgets

from pylcss.cae.design_studio.node_library import NODE_CLASS_MAPPING

from .cad_viewer import CQ3DViewer
from .inspector import PropertiesPanel
from .panels import EventLog, LibraryPanel, ResultsPanel
from .workbench_actions import WorkbenchActionMixin
from .workbench_boundaries import WorkbenchBoundaryMixin
from .workbench_export import WorkbenchExportMixin
from .workbench_graph import WorkbenchGraphMixin
from .workbench_project import WorkbenchProjectMixin
from .workbench_rendering import WorkbenchRenderingMixin
from .workbench_topology import WorkbenchTopologyMixin

logger = logging.getLogger(__name__)

__all__ = ["ProfessionalCadApp", "main"]


class ProfessionalCadApp(
    WorkbenchRenderingMixin,
    WorkbenchBoundaryMixin,
    WorkbenchGraphMixin,
    WorkbenchActionMixin,
    WorkbenchProjectMixin,
    WorkbenchExportMixin,
    WorkbenchTopologyMixin,
    QtWidgets.QMainWindow,
):
    """Professional node-based CAD and simulation workbench."""

    _SILENT_PROP_NAMES = frozenset(
        {
            "error_state",
            "error_message",
        }
    )

    def closeEvent(self, event):
        if not self._ensure_idle_for_io("closing PyLCSS CAE"):
            event.ignore()
            return
        super().closeEvent(event)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)

        self.setWindowTitle("Design Studio")
        self.resize(1280, 800)

        # Initialize data
        self.undo_stack = []
        self.redo_stack = []
        self.current_file = None

        # Initialize worker as None
        self.worker = None
        # Newest execution request that arrived while a run was in flight.
        # Without this, a refresh triggered mid-run was dropped and the viewer
        # kept showing the previous geometry.
        self._pending_execute = None

        # Loading state flag to suppress events during project load
        self._is_loading = False
        # Files can contain Python-bearing CadQuery Code nodes. New projects
        # and bundled examples are trusted; external files must be approved
        # before their graph can execute.
        self._project_code_trusted = True

        # Add mutex for thread safety
        self.result_mutex = QtCore.QMutex()

        # Create graph
        self.graph = NodeGraph()
        self._register_nodes()

        # Connect graph signals
        self.graph.property_changed.connect(self._on_graph_property_changed)
        self.graph.port_connected.connect(self._on_connection_changed)
        self.graph.port_disconnected.connect(self._on_connection_changed)

        # Prevent double-click popup
        self.graph.node_double_clicked.connect(self._on_node_double_clicked)
        # Tear down per-node FreeCAD launcher / watcher state when a node is
        # removed from the graph so the QProcess + QFileSystemWatcher don't
        # leak and so reopening the deleted node's file ID later starts
        # fresh state.
        try:
            self.graph.nodes_deleted.connect(self._on_nodes_deleted)
        except Exception:
            logger.debug("Optional UI operation failed.", exc_info=True)

        # Setup UI
        self._setup_toolbar()
        self._create_ui()
        self._setup_shortcuts()

        # Connect graph selection
        self.graph.node_selected.connect(self._on_node_selected)
        self.graph.node_selection_changed.connect(self._on_node_selection_changed)

        from pylcss.user_interface.common.theme_manager import current_theme

        self.apply_theme(current_theme())

    def _register_nodes(self):
        """Register all available nodes."""
        for node_class in dict.fromkeys(NODE_CLASS_MAPPING.values()):
            try:
                self.graph.register_node(node_class)
            except Exception as exc:
                logger.warning(
                    "Could not register CAD node %s: %s",
                    getattr(node_class, "__name__", node_class),
                    exc,
                )

    def _create_ui(self):
        """Create the main UI layout with 3D Viewer."""
        # Main content widget for QMainWindow
        content_widget = QtWidgets.QWidget()
        main_h_layout = QtWidgets.QHBoxLayout(content_widget)
        main_h_layout.setContentsMargins(0, 0, 0, 0)

        # Left panel: Component Library
        self.library = LibraryPanel(self._spawn_node)
        self.library.setMinimumWidth(220)

        left_splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        left_splitter.addWidget(self.library)
        left_splitter.setSizes([500])
        left_splitter.setMinimumWidth(220)

        # CENTER AREA: Flexible 3D Viewer + Node Graph (Top Viewer, Bottom Graph by default)
        self.center_splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        self.viewer = CQ3DViewer()
        self.viewer.face_picking_requested.connect(self._start_viewer_face_picking)
        self.viewer.result_field_changed.connect(
            self._on_viewer_result_field_changed
        )
        self.center_splitter.addWidget(self.viewer)

        graph_widget = self.graph.widget
        # Enable drops
        try:
            graph_widget.setAcceptDrops(True)
            graph_widget.installEventFilter(self)
        except Exception:
            logger.debug("Optional UI operation failed.", exc_info=True)
        self._graph_widget = graph_widget
        self.center_splitter.addWidget(graph_widget)
        self.center_splitter.setSizes([520, 380])
        self.center_splitter.setMinimumWidth(480)

        # Setup context menu for the graph
        try:
            menu = self.graph.get_context_menu("graph")
            menu.add_command("Fit to View", self._fit_all, "F")
        except Exception:
            logger.debug("Optional UI operation failed.", exc_info=True)

        # Per-node context menu commands
        try:
            node_menu = self.graph.get_context_menu("nodes")
            node_menu.add_command(
                "Open in FreeCAD",
                lambda g, n: self._open_freecad_for_node(n),
                node_type="com.cad.freecad_part.FreeCadPartNode",
            )
        except Exception:
            pass

        # Right panel: Inspector & Results Container
        self.right_container = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(self.right_container)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)

        self.right_header = QtWidgets.QWidget()
        self.right_header.setFixedHeight(28)
        hdr_layout = QtWidgets.QHBoxLayout(self.right_header)
        hdr_layout.setContentsMargins(8, 2, 6, 2)
        self._hdr_title = QtWidgets.QLabel("Inspector")
        self._hdr_title.setStyleSheet("font-weight: 600; font-size: 11px;")
        hdr_layout.addWidget(self._hdr_title, 1)

        self.btn_toggle_inspector = QtWidgets.QToolButton()
        self.btn_toggle_inspector.setText("›")
        self.btn_toggle_inspector.setToolTip("Collapse Inspector")
        self.btn_toggle_inspector.setFixedWidth(22)
        self.btn_toggle_inspector.clicked.connect(self._toggle_inspector_visibility)
        hdr_layout.addWidget(self.btn_toggle_inspector)

        right_layout.addWidget(self.right_header)

        self.right_body = QtWidgets.QWidget()
        self.right_body.setObjectName("InspectorBody")
        body_layout = QtWidgets.QVBoxLayout(self.right_body)
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.setSpacing(0)

        self.results = ResultsPanel()
        self.results.setVisible(False)
        try:
            self.results.view_mode_requested.connect(
                self._on_results_view_mode_requested
            )
            self.results.undeformed_requested.connect(
                self._on_results_undeformed_requested
            )
            self.results.extrema_requested.connect(
                self._on_results_extrema_requested
            )
            self.results.deform_scale_requested.connect(
                self._on_results_deform_scale_requested
            )
        except Exception:
            logger.debug("Could not connect ResultsPanel signals.", exc_info=True)

        self.properties = PropertiesPanel()
        try:
            self.properties.property_changed.connect(self._on_property_changed)
        except Exception:
            logger.debug("Optional UI operation failed.", exc_info=True)

        # Session events go to the log, not to a panel
        self.timeline = EventLog()

        # 3D view mode selector upside at top, properties filling below (single continuous panel)
        body_layout.addWidget(self.results)
        body_layout.addWidget(self.properties, 1)

        self.right_splitter = self.right_body
        self.right_body.sizes = lambda: [1, 1]
        self.right_body.setSizes = lambda sizes: None

        right_layout.addWidget(self.right_body, 1)
        self.right_container.setMinimumWidth(280)

        # Main horizontal splitter
        self.main_splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.main_splitter.addWidget(left_splitter)
        self.main_splitter.addWidget(self.center_splitter)
        self.main_splitter.addWidget(self.right_container)
        self.main_splitter.setCollapsible(0, False)
        self.main_splitter.setCollapsible(1, False)
        self.main_splitter.setCollapsible(2, False)
        self.main_splitter.setSizes([230, 950, 340])
        self.main_splitter.setStretchFactor(0, 0)
        self.main_splitter.setStretchFactor(1, 4)
        self.main_splitter.setStretchFactor(2, 1)

        main_h_layout.addWidget(self.main_splitter)

        # Set central widget for QMainWindow
        self.setCentralWidget(content_widget)
        self._create_progress_indicator()

    def _create_progress_indicator(self):
        """Add the permanent solve-progress readout to the status bar.

        Transient ``showMessage`` text is overwritten by anything else that
        reports status, so an iteration counter posted that way is easy to
        miss entirely. A permanent widget keeps the optimizer's iteration
        number in one fixed place for the whole run.
        """
        status = self.statusBar()
        self._progress_label = QtWidgets.QLabel("")
        self._progress_label.setVisible(False)
        self._progress_bar = QtWidgets.QProgressBar()
        self._progress_bar.setTextVisible(False)
        self._progress_bar.setFixedWidth(140)
        self._progress_bar.setVisible(False)
        status.addPermanentWidget(self._progress_label)
        status.addPermanentWidget(self._progress_bar)

    def _set_solve_progress(self, text, value=None, maximum=None):
        """Show *text* (and an optional determinate bar) in the status bar."""
        label = getattr(self, "_progress_label", None)
        bar = getattr(self, "_progress_bar", None)
        if label is None or bar is None:
            return
        label.setText(str(text))
        label.setVisible(True)
        if maximum is None or int(maximum) <= 0:
            # Indeterminate: work is running but has no countable steps yet.
            bar.setRange(0, 0)
        else:
            bar.setRange(0, int(maximum))
            bar.setValue(max(0, min(int(value or 0), int(maximum))))
        bar.setVisible(True)

    def _clear_solve_progress(self):
        """Hide the solve-progress readout once nothing is running."""
        for name in ("_progress_label", "_progress_bar"):
            widget = getattr(self, name, None)
            if widget is not None:
                widget.setVisible(False)

    def _active_interactive_face_node(self):
        """Return the interactive face-selection node implied by the current UI."""
        candidates = []
        try:
            candidates.extend(list(self.graph.selected_nodes()))
        except Exception:
            logger.debug("Optional UI operation failed.", exc_info=True)

        for attr in ("current_node",):
            node = getattr(self.properties, attr, None)
            if node is not None and node not in candidates:
                candidates.append(node)

        last = getattr(self, "_last_rendered_node", None)
        if last is not None and last not in candidates:
            candidates.append(last)

        for node in candidates:
            if (
                node is not None
                and node.__class__.__name__ == "InteractiveSelectFaceNode"
            ):
                return node

        try:
            all_interactive = [
                n
                for n in self.graph.all_nodes()
                if n.__class__.__name__ == "InteractiveSelectFaceNode"
            ]
        except Exception:
            all_interactive = []

        if len(all_interactive) == 1:
            return all_interactive[0]
        return None

    def _start_viewer_face_picking(self):
        """Start face picking from the VTK viewer toolbar."""
        if self._execution_is_active():
            message = "Wait for the preview to finish."
            self.statusBar().showMessage(message)
            QtWidgets.QMessageBox.information(self, "Preview Running", message)
            return

        node = self._active_interactive_face_node()
        if node is None:
            QtWidgets.QMessageBox.information(
                self,
                "Pick Geometry",
                "Add or select a Pick Geometry node.",
            )
            return

        self.properties.display_node(node)
        self._last_rendered_node = node

        _, geometry = self._get_render_context_for_node(node)
        if geometry is None:
            self._execute_graph(skip_simulation=True)
            self.statusBar().showMessage("Preparing geometry...")
            return

        self._render_result_in_viewer(geometry)
        self.properties._start_picking_session(node)

    def _setup_shortcuts(self):
        """Setup keyboard shortcuts."""
        from PySide6.QtGui import QShortcut, QKeySequence

        # File shortcuts
        QShortcut(QKeySequence.New, self, self._new_project)
        QShortcut(QKeySequence.Open, self, self._open_project)
        QShortcut(QKeySequence.Save, self, self._save_project)

        # Edit shortcuts.  NodeGraphQt's graph_widget already binds Ctrl+Z /
        # Ctrl+Y internally, so registering them as QShortcut on the same
        # parent caused Qt's "Ambiguous shortcut overload" warning at
        # runtime.  We bind through the toolbar actions instead (see
        # _create_toolbar's act_undo / act_redo) so there's exactly one
        # owner per key chord.  Delete is unique to us -- keep it.
        QShortcut(QKeySequence.Delete, self, self._delete_selected)

        # Custom shortcuts
        QShortcut(QKeySequence("F5"), self, self._execute_graph)  # Run
        QShortcut(QKeySequence("F"), self, self._fit_all)  # Fit all
        QShortcut(QKeySequence("R"), self, self._reset_view)  # Reset view

    def _setup_toolbar(self):
        """Create toolbar and add to layout."""
        try:
            import qtawesome as qta
        except Exception:
            qta = None

        def _icon(name, color="#E0E0E0"):
            if qta is None:
                return QtGui.QIcon()
            try:
                return qta.icon(name, color=color)
            except Exception:
                return QtGui.QIcon()

        self.toolbar = QtWidgets.QToolBar("Main Toolbar")
        self.toolbar.setMovable(False)
        self.toolbar.setIconSize(QtCore.QSize(18, 18))
        self.toolbar.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        self.toolbar.setStyleSheet(
            """
            QToolBar { spacing: 4px; padding: 4px; border: 0; }
            QToolButton { padding: 4px 8px; border-radius: 4px; }
            QToolButton:hover { background: rgba(255,255,255,18); }
            QToolBar::separator { background: rgba(255,255,255,28); width: 1px; margin: 4px 4px; }
            """
        )
        self.addToolBar(self.toolbar)

        # ── File / project ──────────────────────────────────────────────
        act_new = self.toolbar.addAction(_icon("fa5s.file"), "New", self._new_project)
        act_new.setShortcut("Ctrl+N")
        act_open = self.toolbar.addAction(
            _icon("fa5s.folder-open"), "Open", self._open_project
        )
        act_open.setShortcut("Ctrl+O")
        act_examples = self.toolbar.addAction(
            _icon("fa5s.book-open"), "Examples", self._open_example
        )
        act_examples.setToolTip("Browse examples")
        act_save = self.toolbar.addAction(
            _icon("fa5s.save"), "Save", self._save_project
        )
        act_save.setShortcut("Ctrl+S")
        act_import = self.toolbar.addAction(
            _icon("fa5s.file-import"), "Import", self._import_cad
        )
        act_import.setToolTip(
            "Import STEP, IGES, BREP, STL, OBJ, or 3MF geometry"
        )
        self.toolbar.addSeparator()

        # ── Edit ────────────────────────────────────────────────────────
        from PySide6.QtGui import QKeySequence

        act_undo = self.toolbar.addAction(_icon("fa5s.undo"), "", self._undo)
        act_undo.setShortcut(QKeySequence.Undo)
        act_undo.setShortcutContext(QtCore.Qt.WindowShortcut)
        act_undo.setToolTip("Undo (Ctrl+Z)")
        act_redo = self.toolbar.addAction(_icon("fa5s.redo"), "", self._redo)
        act_redo.setShortcut(QKeySequence.Redo)
        act_redo.setShortcutContext(QtCore.Qt.WindowShortcut)
        act_redo.setToolTip("Redo (Ctrl+Y)")
        self.toolbar.addSeparator()

        # ── Execution ───────────────────────────────────────────────────
        self.run_action = self.toolbar.addAction(
            _icon("fa5s.play", "#3fb950"), "Run", self._run_action
        )
        self.run_action.setShortcut("F5")
        self.run_action.setToolTip("Run (F5)")
        self.stop_action = self.toolbar.addAction(
            _icon("fa5s.stop", "#f85149"), "Stop", self._cancel_execution
        )
        self.stop_action.setShortcut("Shift+F5")
        self.stop_action.setToolTip("Stop (Shift+F5)")

        self.validate_action = self.toolbar.addAction(
            _icon("fa5s.check-circle", "#d29922"), "Validate", self._validate_model
        )
        self.validate_action.setToolTip("Check the graph")
        self.toolbar.addSeparator()

        # ── Analysis & Export ───────────────────────────────────────────
        self.save_results_action = self.toolbar.addAction(
            _icon("fa5s.file-export"), "Export Results", self._export_simulation_results
        )
        self.save_results_action.setToolTip("Export the current result")
        self.toolbar.addSeparator()

        # ── Layout Modes ────────────────────────────────────────────────
        self._layout_group = QtGui.QActionGroup(self)
        self._layout_group.setExclusive(True)
        self._layout_actions = {}

        self.act_layout_3d = self.toolbar.addAction(
            _icon("fa5s.cube"), "3D Focus", lambda: self.set_layout_mode("3d")
        )
        self.act_layout_3d.setCheckable(True)
        self.act_layout_3d.setToolTip("Focus on 3D Viewport")
        self._layout_group.addAction(self.act_layout_3d)
        self._layout_actions["3d"] = self.act_layout_3d

        self.act_layout_split = self.toolbar.addAction(
            _icon("fa5s.columns"), "Split View", lambda: self.set_layout_mode("split")
        )
        self.act_layout_split.setCheckable(True)
        self.act_layout_split.setChecked(True)
        self.act_layout_split.setToolTip("Split View (3D View + Node Graph)")
        self._layout_group.addAction(self.act_layout_split)
        self._layout_actions["split"] = self.act_layout_split

        self.act_layout_graph = self.toolbar.addAction(
            _icon("fa5s.project-diagram"), "Graph Focus", lambda: self.set_layout_mode("graph")
        )
        self.act_layout_graph.setCheckable(True)
        self.act_layout_graph.setToolTip("Focus on Node Graph Canvas")
        self._layout_group.addAction(self.act_layout_graph)
        self._layout_actions["graph"] = self.act_layout_graph

        self.act_layout_orient = self.toolbar.addAction(
            _icon("fa5s.exchange-alt"), "", self.toggle_split_orientation
        )
        self.act_layout_orient.setToolTip("Toggle Split Orientation: Stacked (Top/Bottom) / Side-by-Side")
        self.toolbar.addSeparator()

        # ── View ────────────────────────────────────────────────────────
        act_fit = self.toolbar.addAction(_icon("fa5s.expand"), "", self._fit_all)
        act_fit.setToolTip("Fit view to all (F)")
        act_reset = self.toolbar.addAction(_icon("fa5s.compress-arrows-alt"), "", self._reset_view)
        act_reset.setToolTip("Reset 3D View (R)")

        self._theme_toolbar_icons = (
            (act_new, "fa5s.file"),
            (act_open, "fa5s.folder-open"),
            (act_examples, "fa5s.book-open"),
            (act_save, "fa5s.save"),
            (act_import, "fa5s.file-import"),
            (act_undo, "fa5s.undo"),
            (act_redo, "fa5s.redo"),
            (self.save_results_action, "fa5s.file-export"),
            (self.act_layout_3d, "fa5s.cube"),
            (self.act_layout_split, "fa5s.columns"),
            (self.act_layout_graph, "fa5s.project-diagram"),
            (self.act_layout_orient, "fa5s.exchange-alt"),
            (act_fit, "fa5s.expand"),
            (act_reset, "fa5s.compress-arrows-alt"),
        )
        self._theme_semantic_toolbar_icons = (
            (self.run_action, "fa5s.play", "success", "#3fb950"),
            (self.stop_action, "fa5s.stop", "danger", "#f85149"),
            (self.validate_action, "fa5s.check-circle", "primary", "#d29922"),
        )

    def set_layout_mode(self, mode: str):
        """Switch center area between 3D focus, Split view, and Node Graph focus."""
        mode = mode.lower()
        self._current_layout_mode = mode

        if hasattr(self, "_layout_actions"):
            act = self._layout_actions.get(mode)
            if act and not act.isChecked():
                act.setChecked(True)

        if mode == "3d":
            self.viewer.setVisible(True)
            self._graph_widget.setVisible(False)
            self.center_splitter.setSizes([1000, 0])
        elif mode == "graph":
            self.viewer.setVisible(False)
            self._graph_widget.setVisible(True)
            self.center_splitter.setSizes([0, 1000])
        else:  # "split"
            self.viewer.setVisible(True)
            self._graph_widget.setVisible(True)
            sizes = self.center_splitter.sizes()
            total = sum(sizes) if sum(sizes) > 0 else 1000
            self.center_splitter.setSizes([int(total * 0.58), int(total * 0.42)])

    def toggle_split_orientation(self):
        """Toggle center splitter orientation between side-by-side and stacked."""
        current = self.center_splitter.orientation()
        new_orient = (
            QtCore.Qt.Vertical
            if current == QtCore.Qt.Horizontal
            else QtCore.Qt.Horizontal
        )
        self.center_splitter.setOrientation(new_orient)
        sizes = self.center_splitter.sizes()
        total = sum(sizes) if sum(sizes) > 0 else 1000
        self.center_splitter.setSizes([int(total * 0.55), int(total * 0.45)])

    def _toggle_inspector_visibility(self):
        """Toggle right inspector collapse / expand."""
        is_visible = not self.right_splitter.isHidden()
        self.right_splitter.setVisible(not is_visible)
        if is_visible:
            self.btn_toggle_inspector.setText("‹")
            self.btn_toggle_inspector.setToolTip("Expand Inspector")
        else:
            self.btn_toggle_inspector.setText("›")
            self.btn_toggle_inspector.setToolTip("Collapse Inspector")
        if is_visible:
            self.right_container.setMinimumWidth(32)
            sizes = self.main_splitter.sizes()
            if len(sizes) == 3:
                self.main_splitter.setSizes([sizes[0], sizes[1] + sizes[2] - 32, 32])
        else:
            self.right_container.setMinimumWidth(280)
            sizes = self.main_splitter.sizes()
            if len(sizes) == 3 and sizes[2] < 200:
                self.main_splitter.setSizes([sizes[0], max(400, sizes[1] - 300), 300])

    def set_stage(self, stage: str):
        """Adapt CAE workspace layout and focus for the active engineering stage."""
        stage = stage.lower()
        self._current_stage = stage
        if stage in ("geometry", "cad"):
            self.set_layout_mode("3d")
            self.statusBar().showMessage("Stage: Parametric CAD & Geometry")
        elif stage in ("setup", "physics"):
            self.set_layout_mode("split")
            self.statusBar().showMessage("Stage: Physics Pre-Processing & Boundary Conditions")
        elif stage in ("solvers", "simulation"):
            self.set_layout_mode("split")
            self.statusBar().showMessage("Stage: Multi-Physics Solvers & Optimization")
        elif stage in ("results", "insights"):
            self.set_layout_mode("3d")
            self._show_results_panel()
            self.statusBar().showMessage("Stage: Post-Processing & Insights")

    def apply_theme(self, theme_name):
        """Apply theme colours to Design Studio's non-palette surfaces."""
        from pylcss.user_interface.common.theme_manager import (
            THEMES,
            retheme_node_graph,
            retheme_widget_styles,
        )

        theme_name = str(theme_name).lower()
        colors = THEMES[theme_name]
        try:
            import qtawesome as qta

            for action, icon_name in self._theme_toolbar_icons:
                action.setIcon(qta.icon(icon_name, color=colors["text_main"]))
            for action, icon_name, role, dark_color in self._theme_semantic_toolbar_icons:
                color = colors[role] if theme_name == "light" else dark_color
                action.setIcon(qta.icon(icon_name, color=color))
        except Exception:
            logger.debug("Optional UI operation failed.", exc_info=True)
        try:
            retheme_node_graph(self.graph, theme_name)
        except Exception:
            logger.debug("Optional UI operation failed.", exc_info=True)
        try:
            self.viewer.apply_theme(theme_name)
        except Exception:
            logger.debug("Optional UI operation failed.", exc_info=True)
        try:
            retheme_widget_styles(self, theme_name)
        except Exception:
            logger.debug("Optional UI operation failed.", exc_info=True)
        self.results.apply_theme(theme_name)
        if hasattr(self, "right_header") and self.right_header is not None:
            self.right_header.setStyleSheet(
                f"""
                background-color: {colors["bg_dark"]};
                border-bottom: 1px solid {colors["border"]};
                color: {colors["text_main"]};
                """
            )

    def _spawn_node(self, node_id, label, x=None, y=None):
        """Spawn a new node in the graph."""
        # Choose a default position if none provided
        existing = [n.pos() for n in self.graph.all_nodes()]
        x0, y0 = 250, 50
        if existing:
            y0 = max(n[1] for n in existing) + 150
        if x is None:
            x = x0
        if y is None:
            y = y0

        node_classes = NODE_CLASS_MAPPING

        node_class = node_classes.get(node_id)
        if not node_class:
            self.statusBar().showMessage(f"Unknown node: {node_id}")
            return

        try:
            node = node_class()
            node.set_name(label)
            node.set_pos(x, y)
            if node_id == "com.cad.sim.gravity":
                node.set_property("load_type", "Gravity")
            from pylcss.cae.design_studio.core.port_schema import (
                apply_display_port_labels,
            )

            apply_display_port_labels(node)
            if node_id in (
                "com.cad.sim.topopt_voxel",
                "com.cad.sim.lattice_voxel",
            ):
                try:
                    self._suppress_graph_property_changed = True
                    self._apply_topopt_industrial_defaults(node)
                except Exception:
                    logger.debug("Optional UI operation failed.", exc_info=True)
                finally:
                    self._suppress_graph_property_changed = False
            # A newly created node is the active editing target. NodeGraphQt
            # otherwise keeps the old node selected as well, which turns a
            # scoped workflow run into a multi-selection/full-graph run.
            self.graph.clear_selection()
            self.graph.add_node(node, selected=True)
            # NodeGraphQt ignores the QApplication palette, so style a newly
            # inserted node immediately instead of waiting for a theme toggle.
            from pylcss.user_interface.common.theme_manager import (
                current_theme,
                retheme_node_graph,
            )

            retheme_node_graph(self.graph, current_theme())
            # record undo action
            try:
                self._push_undo({"type": "add_node", "node": node})
            except Exception:
                logger.debug("Optional UI operation failed.", exc_info=True)

            self.timeline.add_event(f"Added {label}")
            self.statusBar().showMessage(f"Added {label}")
            return node
        except Exception as e:
            self.statusBar().showMessage(f"Error: {e}")
            return None


def main() -> None:
    """Launch the CAD application."""
    os.environ["QT_API"] = "pyside6"

    app = QtWidgets.QApplication(sys.argv)

    # Apply dark theme
    app.setStyle("Fusion")
    palette = QtGui.QPalette()
    palette.setColor(QtGui.QPalette.Window, QtGui.QColor(45, 45, 45))
    palette.setColor(QtGui.QPalette.WindowText, QtCore.Qt.white)
    palette.setColor(QtGui.QPalette.Button, QtGui.QColor(53, 53, 53))
    palette.setColor(QtGui.QPalette.ButtonText, QtCore.Qt.white)
    palette.setColor(QtGui.QPalette.Base, QtGui.QColor(35, 35, 35))
    palette.setColor(QtGui.QPalette.Text, QtCore.Qt.white)
    palette.setColor(QtGui.QPalette.Highlight, QtGui.QColor(42, 130, 218))
    app.setPalette(palette)

    try:
        window = ProfessionalCadApp()
        window.show()
        sys.exit(app.exec())
    except RuntimeError as e:
        # Suppress NodeGraphQt internal errors
        if "layout direction not valid" in str(e):
            pass
            sys.exit(0)
        else:
            raise
