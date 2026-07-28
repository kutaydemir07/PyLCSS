# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE.
"""Top-level Design Studio workbench assembled from focused mixins."""

from __future__ import annotations

import logging
import os
import sys

from NodeGraphQt import NodeGraph
from PySide6 import QtCore, QtGui, QtWidgets

from pylcss.design_studio.node_library import NODE_CLASS_MAPPING

from .cad_viewer import CQ3DViewer
from .inspector import PropertiesPanel
from .panels import LibraryPanel, ResultsPanel, TimelinePanel
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

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)

        self.setWindowTitle("Engineering Design Studio")
        self.resize(1280, 800)

        # Initialize data
        self.undo_stack = []
        self.redo_stack = []
        self.current_file = None

        # Initialize worker as None
        self.worker = None
        self._last_preview_update_time = 0.0

        # Loading state flag to suppress events during project load
        self._is_loading = False

        # Add mutex for thread safety
        self.result_mutex = QtCore.QMutex()

        # Create graph
        self.graph = NodeGraph()
        self._register_nodes()

        # Initialize Command Dispatcher for LLM actions (lazy import to avoid circular)
        from pylcss.assistant_systems.api.dispatcher import CommandDispatcher

        self.command_dispatcher = CommandDispatcher(main_window=self)

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

    def _register_nodes(self):
        """Register all available nodes."""
        for node_class in NODE_CLASS_MAPPING.values():
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

        # Left panel: Library
        self.library = LibraryPanel(self._spawn_node)
        left_splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        left_splitter.addWidget(self.library)
        left_splitter.setSizes([500])
        left_splitter.setMinimumWidth(190)

        # CENTER AREA
        center_splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        self.viewer = CQ3DViewer()
        self.viewer.face_picking_requested.connect(self._start_viewer_face_picking)
        center_splitter.addWidget(self.viewer)

        graph_widget = self.graph.widget
        # Enable drops
        try:
            graph_widget.setAcceptDrops(True)
            graph_widget.installEventFilter(self)
        except Exception:
            logger.debug("Optional UI operation failed.", exc_info=True)
        self._graph_widget = graph_widget
        center_splitter.addWidget(graph_widget)
        center_splitter.setSizes([600, 400])
        center_splitter.setMinimumWidth(480)

        # Setup context menu for the graph
        try:
            menu = self.graph.get_context_menu("graph")
            menu.add_command("Fit to View", self._fit_all, "F")
        except Exception:
            logger.debug("Optional UI operation failed.", exc_info=True)

        # Per-node context menu commands. NodeGraphQt scopes node-context
        # menu items to a specific node type via its identifier, so this
        # only appears on FreeCadPartNode right-click. The handler receives
        # (graph, node).
        try:
            node_menu = self.graph.get_context_menu("nodes")
            node_menu.add_command(
                "Open in FreeCAD",
                lambda g, n: self._open_freecad_for_node(n),
                node_type="com.cad.freecad_part.FreeCadPartNode",
            )
        except Exception:
            # NodeGraphQt's node-context menu API changes between versions;
            # if this signature isn't supported, double-click still works.
            pass

        # Right panel
        right_splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        self.properties = PropertiesPanel()
        try:
            self.properties.property_changed.connect(self._on_property_changed)
        except Exception:
            logger.debug("Optional UI operation failed.", exc_info=True)
        self.results = ResultsPanel()
        self.timeline = TimelinePanel()

        lower_tabs = QtWidgets.QTabWidget()
        lower_tabs.setDocumentMode(True)
        lower_tabs.addTab(self.results, "Results")
        lower_tabs.addTab(self.timeline, "History")
        self._engineering_tabs = lower_tabs
        right_splitter.addWidget(self.properties)
        right_splitter.addWidget(lower_tabs)
        right_splitter.setSizes([500, 260])
        right_splitter.setMinimumWidth(300)

        # Main splitter
        main_splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        main_splitter.addWidget(left_splitter)
        main_splitter.addWidget(center_splitter)
        main_splitter.addWidget(right_splitter)
        main_splitter.setSizes([210, 850, 330])
        main_splitter.setStretchFactor(0, 0)
        main_splitter.setStretchFactor(1, 1)
        main_splitter.setStretchFactor(2, 0)

        main_h_layout.addWidget(main_splitter)

        # Set central widget for QMainWindow
        self.setCentralWidget(content_widget)

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
            message = (
                "Wait for the current CAD preview to finish before picking geometry."
            )
            self.statusBar().showMessage(message)
            QtWidgets.QMessageBox.information(self, "Preview In Progress", message)
            return

        node = self._active_interactive_face_node()
        if node is None:
            QtWidgets.QMessageBox.information(
                self,
                "Select Geometry Picking Node",
                "Add or select a Select Geometry (Interactive) node connected "
                "to the shape, then click Pick Geometry again.",
            )
            return

        self.properties.display_node(node)
        self._last_rendered_node = node

        _, geometry = self._get_render_context_for_node(node)
        if geometry is None:
            self._execute_graph(skip_simulation=True)
            self.statusBar().showMessage(
                "Updating the CAD preview for geometry picking; click Pick "
                "Geometry again when it finishes."
            )
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
        act_save = self.toolbar.addAction(
            _icon("fa5s.save"), "Save", self._save_project
        )
        act_save.setShortcut("Ctrl+S")
        act_import = self.toolbar.addAction(
            _icon("fa5s.file-import"), "Import Geometry", self._import_cad
        )
        act_import.setToolTip("Import a STEP / IGES / STL / OBJ file as a node")
        self.toolbar.addSeparator()

        # ── Edit (icon-only — short labels add noise) ───────────────────
        # setShortcut() on the action makes Qt route Ctrl+Z / Ctrl+Y to our
        # handler from a single owner; the previous QShortcut bindings were
        # removed in _setup_shortcuts to stop the "Ambiguous shortcut" warning.
        # QKeySequence is imported locally because _setup_shortcuts (the
        # other caller in this file) does the same -- there's no
        # module-level import to lean on.
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

        # ── Run is the most-used action — give it accent color via icon ─
        self.run_action = self.toolbar.addAction(
            _icon("fa5s.play", "#66BB6A"), "Run", self._run_action
        )
        self.run_action.setToolTip(
            "Run the selected node and its inputs only (siblings/downstream are "
            "skipped). With nothing selected, runs the whole graph."
        )
        self.run_action.setShortcut("F5")
        self.run_action.setToolTip("Execute the node graph (F5)")
        self.stop_action = self.toolbar.addAction(
            _icon("fa5s.stop", "#EF5350"), "Stop", self._cancel_execution
        )
        self.stop_action.setShortcut("Shift+F5")
        self.stop_action.setToolTip(
            "Stop TopOpt at the next iteration boundary (Shift+F5)"
        )

        self.toolbar.addAction(
            _icon("fa5s.check-circle", "#4FC3F7"), "Validate", self._validate_model
        ).setToolTip("Check the graph for disconnected nodes and obvious mistakes")
        act_report = self.toolbar.addAction(
            _icon("fa5s.clipboard-list"), "Report", self._generate_report
        )
        act_report.setToolTip("Generate a text summary of the current model")
        self.save_results_action = self.toolbar.addAction(
            _icon("fa5s.file-export"), "Save Results", self._export_simulation_results
        )
        self.save_results_action.setToolTip(
            "Save the already-computed FEA / TopOpt / Crash result as portable JSON or HDF5"
        )
        self.toolbar.addSeparator()

        # ── View ────────────────────────────────────────────────────────
        act_fit = self.toolbar.addAction(_icon("fa5s.expand"), "", self._fit_all)
        act_fit.setToolTip("Fit view to all (F)")
        self._theme_toolbar_icons = (
            (act_new, "fa5s.file"),
            (act_open, "fa5s.folder-open"),
            (act_save, "fa5s.save"),
            (act_import, "fa5s.file-import"),
            (act_undo, "fa5s.undo"),
            (act_redo, "fa5s.redo"),
            (act_report, "fa5s.clipboard-list"),
            (self.save_results_action, "fa5s.file-export"),
            (act_fit, "fa5s.expand"),
        )

    def apply_theme(self, theme_name):
        """Apply theme colours to Design Studio's non-palette surfaces."""
        from pylcss.user_interface.common.theme_manager import (
            THEMES,
            retheme_node_graph,
        )

        theme_name = str(theme_name).lower()
        colors = THEMES[theme_name]
        try:
            import qtawesome as qta

            for action, icon_name in self._theme_toolbar_icons:
                action.setIcon(qta.icon(icon_name, color=colors["text_main"]))
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
            if node_id == "com.cad.sim.topopt_voxel":
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

            self.timeline.add_event(f"Added {label} node")
            self.statusBar().showMessage(f"Created {label}")
            return node
        except Exception as e:
            self.statusBar().showMessage(f"Error: {e}")
            return None


def main() -> None:
    """Launch the professional CAD software."""
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
