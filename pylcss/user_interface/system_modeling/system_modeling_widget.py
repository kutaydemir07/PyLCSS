# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""
System modeling editor for PyLCSS.

This module provides the main visual modeling interface using NodeGraphQt,
allowing users to create and edit system models through a node-based
graphical interface.
"""

import logging
import re

import qtawesome as qta
from PySide6 import QtCore, QtGui, QtWidgets

from pylcss.system_modeling.compilation import (
    ModelCompilationError,
    compile_systems,
)
from pylcss.user_interface.common.theme_manager import (
    COLORS,
    THEMES,
    current_theme,
    retheme_node_graph,
)
from .actions import (
    load_graph,
    load_graph_from_file,
    save_graph,
    save_graph_to_file,
    validate_graph,
)
from .system_manager import SystemManager
from .system_node_types import (
    CodeEditorDialog,
    CustomBlockNode,
    InputNode,
    IntermediateNode,
    OutputNode,
    apply_system_node_style,
)
from .training import train_selected_node_surrogate

logger = logging.getLogger(__name__)

# Use the application's professional palette so this feature remains visually
# consistent with the rest of the desktop application.
_MDO_ACCENT = COLORS["primary"]  # amber #d29922
_MDO_TOOLBAR_QSS = f"""
    QToolBar {{
        background: {COLORS["bg_panel"]}; border: none;
        border-bottom: 1px solid {COLORS["bg_dark"]};
        spacing: 3px; padding: 5px 6px;
    }}
    QToolBar::separator {{ background: {COLORS["bg_input"]}; width: 1px; margin: 5px 7px; }}
    QToolButton {{
        background: transparent; color: {COLORS["text_dim"]};
        border: 1px solid transparent; border-radius: 6px;
        padding: 5px 10px; font-weight: 600;
    }}
    QToolButton:hover {{
        background: {COLORS["bg_input"]}; border: 1px solid {COLORS["primary"]};
        color: {COLORS["text_main"]};
    }}
    QToolButton:pressed {{ background: {COLORS["bg_dark"]}; }}
    QLabel {{ color: {COLORS["text_dim"]}; font-weight: 600; }}
    QLineEdit {{
        background: {COLORS["bg_input"]}; border: 1px solid {COLORS["bg_dark"]};
        border-radius: 6px; padding: 4px 8px; color: {COLORS["text_main"]};
        selection-background-color: {COLORS["primary"]};
    }}
    QLineEdit:focus {{ border: 1px solid {COLORS["primary"]}; }}
"""

__all__ = ["ModelingWidget"]


class ModelingWidget(QtWidgets.QWidget):
    """
    Main widget for the system modeling environment.

    Provides a complete node-based modeling interface with system management,
    graph editing, validation, and code generation capabilities.
    """

    build_requested = QtCore.Signal()

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        """
        Initialize the modeling widget.

        Sets up the UI components including system manager, graph area,
        toolbar with modeling tools, and search functionality.
        """
        super().__init__(parent)

        # --- Layout & Panels ---
        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.setSpacing(15)  # Add breathing room between elements
        self.layout.setContentsMargins(20, 20, 20, 20)  # Add padding around the edges

        self.current_graph = None
        self._updating_flag = False  # Prevent recursion in property changes

        self.system_manager = SystemManager(self, create_default=False)
        self.system_manager.system_selected.connect(self.on_system_selected)
        self.system_manager.system_added.connect(self.setup_new_graph)

        # Splitter
        self.main_splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.main_splitter.addWidget(self.system_manager.panel)
        self.main_splitter.addWidget(self.system_manager.graph_area)
        self.main_splitter.setSizes([200, 1200])  # Set initial proportions
        self.main_splitter.setStretchFactor(0, 0)  # Left panel doesn't stretch
        self.main_splitter.setStretchFactor(1, 1)  # Right panel stretches
        self.layout.addWidget(self.main_splitter)

        # --- Toolbar ---
        self.toolbar = QtWidgets.QToolBar()
        self.toolbar.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        self.toolbar.setIconSize(QtCore.QSize(16, 16))
        self.toolbar.setStyleSheet(_MDO_TOOLBAR_QSS)
        self.toolbar.setMovable(False)
        self.layout.insertWidget(0, self.toolbar)

        def _ico(name, color="#cdd2d9"):
            return qta.icon(name, color=color)

        self.action_save = QtGui.QAction(_ico("fa5s.save"), "Save", self)
        self.action_save.triggered.connect(self.save_graph)
        self.toolbar.addAction(self.action_save)

        self.action_load = QtGui.QAction(_ico("fa5s.folder-open"), "Load", self)
        self.action_load.triggered.connect(self.load_graph)
        self.toolbar.addAction(self.action_load)

        self.toolbar.addSeparator()

        # Primary action — theme success green.
        self.action_build = QtGui.QAction(
            _ico("fa5s.hammer", COLORS["success"]), "Build Model", self
        )
        self.action_build.triggered.connect(self.build_requested.emit)
        self.toolbar.addAction(self.action_build)

        self.action_validate = QtGui.QAction(
            _ico("fa5s.check-circle", _MDO_ACCENT), "Validate", self
        )
        self.action_validate.triggered.connect(self.validate_graph)
        self.toolbar.addAction(self.action_validate)

        self.toolbar.addSeparator()

        self.action_undo = QtGui.QAction(_ico("fa5s.undo"), "Undo", self)
        self.action_undo.setShortcut(QtGui.QKeySequence.Undo)
        self.action_undo.triggered.connect(self.undo)
        self.toolbar.addAction(self.action_undo)

        self.action_redo = QtGui.QAction(_ico("fa5s.redo"), "Redo", self)
        self.action_redo.setShortcut(QtGui.QKeySequence.Redo)
        self.action_redo.triggered.connect(self.redo)
        self.toolbar.addAction(self.action_redo)

        self.toolbar.addSeparator()

        self.action_delete = QtGui.QAction(
            _ico("fa5s.trash-alt", COLORS["danger"]), "Delete", self
        )
        self.action_delete.triggered.connect(self.delete_current)
        self.action_delete.setShortcut(QtGui.QKeySequence.Delete)
        self.action_delete.setShortcutContext(QtCore.Qt.WidgetWithChildrenShortcut)
        self.toolbar.addAction(self.action_delete)

        self.toolbar.addSeparator()

        # Node-creation buttons — icon colour matches each node's role colour
        # (blue = design variable, slate = intermediate, teal = QoI, violet =
        # function block) so the toolbar reads as a legend for the graph.
        self.action_add_input = QtGui.QAction(
            _ico("fa5s.sliders-h", COLORS["primary"]), "Design Var", self
        )
        self.action_add_input.setToolTip(
            "Design Variable — an optimizer-controlled input with units and bounds"
        )
        self.action_add_input.triggered.connect(self.add_input_node)
        self.toolbar.addAction(self.action_add_input)

        self.action_add_intermediate = QtGui.QAction(
            _ico("fa5s.dot-circle", "#9aa3b2"), "Intermediate", self
        )
        self.action_add_intermediate.setToolTip(
            "Intermediate Variable — route or rename a value between disciplines"
        )
        self.action_add_intermediate.triggered.connect(self.add_intermediate_node)
        self.toolbar.addAction(self.action_add_intermediate)

        self.action_add_output = QtGui.QAction(
            _ico("fa5s.bullseye", "#3fc093"), "QoI", self
        )
        self.action_add_output.setToolTip(
            "Quantity of Interest — a result, constraint, or optimization objective"
        )
        self.action_add_output.triggered.connect(self.add_output_node)
        self.toolbar.addAction(self.action_add_output)

        self.action_add_function = QtGui.QAction(
            _ico("fa5s.code", "#a98be0"), "Function Block", self
        )
        self.action_add_function.setToolTip(
            "Function / Discipline — Python, FEA, crash, or TopOpt calculation"
        )
        self.action_add_function.triggered.connect(self.add_function_node)
        self.toolbar.addAction(self.action_add_function)

        self._theme_action_icons = {
            self.action_save: ("fa5s.save", "neutral"),
            self.action_load: ("fa5s.folder-open", "neutral"),
            self.action_build: ("fa5s.hammer", "success"),
            self.action_validate: ("fa5s.check-circle", "primary"),
            self.action_undo: ("fa5s.undo", "neutral"),
            self.action_redo: ("fa5s.redo", "neutral"),
            self.action_delete: ("fa5s.trash-alt", "danger"),
            self.action_add_input: ("fa5s.sliders-h", "primary"),
            self.action_add_intermediate: ("fa5s.dot-circle", "muted"),
            self.action_add_output: ("fa5s.bullseye", "success"),
            self.action_add_function: ("fa5s.code", "function"),
        }

        # Search Bar
        self.toolbar.addSeparator()
        lbl_search = QtWidgets.QLabel(" Search ")
        self.toolbar.addWidget(lbl_search)
        self.search_bar = QtWidgets.QLineEdit()
        self.search_bar.setPlaceholderText("Find node...")
        self.search_bar.setMaximumWidth(170)
        self.search_bar.setClearButtonEnabled(True)
        self.search_bar.returnPressed.connect(self.find_node)
        self.toolbar.addWidget(self.search_bar)

        # Init
        self._current_undo_connection = None
        self._current_redo_connection = None
        self.update_undo_redo_actions()
        self.on_system_selected()

    def apply_theme(self, theme_name):
        """Refresh icons and every modeling graph, including hidden systems."""
        theme_name = str(theme_name).lower()
        self._active_theme = theme_name
        colors = THEMES[theme_name]
        role_colors = {
            "neutral": colors["text_main"],
            "muted": colors["text_dim"],
            "primary": colors["primary"],
            "success": colors["success"],
            "danger": colors["danger"],
            "function": "#8250df" if theme_name == "light" else "#a98be0",
        }
        for action, (icon_name, role) in self._theme_action_icons.items():
            action.setIcon(qta.icon(icon_name, color=role_colors[role]))
        for system in self.system_manager.systems:
            graph = system.get("graph") if isinstance(system, dict) else None
            if graph is not None:
                retheme_node_graph(graph, theme_name)

    @QtCore.Slot(bool)
    def _safe_set_undo_enabled(self, enabled):
        try:
            if hasattr(self, "action_undo"):
                self.action_undo.setEnabled(enabled)
        except RuntimeError:
            pass

    @QtCore.Slot(bool)
    def _safe_set_redo_enabled(self, enabled):
        try:
            if hasattr(self, "action_redo"):
                self.action_redo.setEnabled(enabled)
        except RuntimeError:
            pass

    def update_undo_redo_actions(self):
        """Update undo/redo action connections for the current graph."""
        # Disconnect previous connections
        if self._current_undo_connection:
            try:
                self._current_undo_connection.disconnect()
            except (RuntimeError, TypeError):
                pass
            self._current_undo_connection = None

        if self._current_redo_connection:
            try:
                self._current_redo_connection.disconnect()
            except (RuntimeError, TypeError):
                pass
            self._current_redo_connection = None

        # Connect to current graph's undo/redo if available
        if self.current_graph and hasattr(self.current_graph, "undo_stack"):
            undo_stack = self.current_graph.undo_stack()
            if hasattr(undo_stack, "canUndoChanged"):
                self._current_undo_connection = undo_stack.canUndoChanged.connect(
                    self._safe_set_undo_enabled
                )
            if hasattr(undo_stack, "canRedoChanged"):
                self._current_redo_connection = undo_stack.canRedoChanged.connect(
                    self._safe_set_redo_enabled
                )

            # Update button states immediately
            self._safe_set_undo_enabled(undo_stack.canUndo())
            self._safe_set_redo_enabled(undo_stack.canRedo())

    def find_node(self):
        """Search for nodes by name or variable name in the current graph."""
        text = self.search_bar.text().lower().strip()
        if not text or not self.current_graph:
            return

        found = False
        for node in self.current_graph.all_nodes():
            if (
                text in node.name().lower()
                or text in str(node.get_property("var_name")).lower()
            ):
                self.current_graph.clear_selection()
                node.set_selected(True)
                self.current_graph.center_on([node])
                found = True
                break

        if not found:
            self.search_bar.setStyleSheet("border: 1px solid red;")
            QtCore.QTimer.singleShot(1000, lambda: self.search_bar.setStyleSheet(""))

    def on_system_selected(self):
        """Handle system selection changes."""
        self.current_graph = self.system_manager.current_graph

        # Ensure the context menu is set up for the current graph
        if self.current_graph:
            self.setup_context_menu_for_graph(self.current_graph)
            for node in self.current_graph.all_nodes():
                apply_system_node_style(node)
            retheme_node_graph(
                self.current_graph,
                getattr(self, "_active_theme", current_theme()),
            )

        # Update undo/redo connections
        self.update_undo_redo_actions()

    def setup_new_graph(self, graph):
        """Set up event connections for a newly created graph."""
        graph.node_double_clicked.connect(self.on_node_double_clicked)
        graph.port_connected.connect(self.on_port_connected)
        graph.property_changed.connect(self.on_property_changed)
        self.setup_context_menu_for_graph(graph)
        retheme_node_graph(
            graph,
            getattr(self, "_active_theme", current_theme()),
        )

        # --- FIX: Use .scene() instead of ._viewer._scene ---
        # The internal attribute _scene is gone/private. Use the public API.
        if hasattr(graph, "scene"):
            scene = graph.scene()
            if scene:
                # Disable BSP indexing to prevent crashes during rapid remove/add operations
                scene.setItemIndexMethod(QtWidgets.QGraphicsScene.NoIndex)

        # --- FIX: Set Undo Limit Here (Only once per graph) ---
        if hasattr(graph, "undo_stack"):
            # Only set limit if the stack is new/empty to avoid Qt warnings
            if graph.undo_stack().count() == 0:
                graph.undo_stack().setUndoLimit(50)

    def undo(self):
        """Undo the last action in the current graph."""
        if self.current_graph:
            self.current_graph.undo_stack().undo()

    def redo(self):
        """Redo the last undone action in the current graph."""
        if self.current_graph:
            self.current_graph.undo_stack().redo()

    def delete_current(self):
        if self.current_graph:
            self.current_graph.delete_nodes(self.current_graph.selected_nodes())

    def add_input_node(self):
        if self.current_graph:
            self.create_node_for_graph(self.current_graph, InputNode)

    def add_intermediate_node(self):
        if self.current_graph:
            self.create_node_for_graph(self.current_graph, IntermediateNode)

    def add_output_node(self):
        if self.current_graph:
            self.create_node_for_graph(self.current_graph, OutputNode)

    def add_function_node(self):
        if self.current_graph:
            self.create_node_for_graph(self.current_graph, CustomBlockNode)

    def setup_context_menu_for_graph(self, graph):
        menu = graph.context_menu()

        # Track if commands have already been added to prevent duplication
        if hasattr(menu, "_commands_added") and menu._commands_added:
            return

        # Remove default undo/redo actions from the context menu
        qmenu = menu.qmenu
        actions_to_remove = []
        for action in qmenu.actions():
            if action.text() in ["&Undo", "&Redo"]:
                actions_to_remove.append(action)

        for action in actions_to_remove:
            qmenu.removeAction(action)

        menu.add_command(
            "Create Function / Discipline",
            lambda: self.create_node_for_graph(graph, CustomBlockNode),
            "Shift+F",
        )
        menu.add_command(
            "Create Design Variable",
            lambda: self.create_node_for_graph(graph, InputNode),
            "Shift+I",
        )
        menu.add_command(
            "Create Quantity of Interest",
            lambda: self.create_node_for_graph(graph, OutputNode),
            "Shift+O",
        )
        menu.add_command(
            "Create Intermediate Variable",
            lambda: self.create_node_for_graph(graph, IntermediateNode),
            "Shift+V",
        )
        menu.add_separator()
        menu.add_command(
            "Delete", lambda: graph.delete_nodes(graph.selected_nodes()), "Del"
        )
        menu.add_separator()
        menu.add_command("Fit to View", graph.fit_to_selection, "F")

        # Mark commands as added
        menu._commands_added = True

    def create_node_for_graph(self, graph, node_class):
        target_id = node_class.__identifier__
        registered = graph.registered_nodes()
        full_id = None
        if target_id in registered:
            full_id = target_id
        else:
            expected_id = f"{target_id}.{node_class.__name__}"
            if expected_id in registered:
                full_id = expected_id
            else:
                for nid in registered:
                    if nid.startswith(target_id):
                        full_id = nid
                        break
        if not full_id:
            QtWidgets.QMessageBox.warning(
                self,
                "Node Error",
                f"Could not find registered node for {target_id}\nAvailable: {registered}",
            )
            return None
        try:
            node = graph.create_node(full_id, pos=[0, 0])
            apply_system_node_style(node)
            return node
        except Exception as e:
            QtWidgets.QMessageBox.warning(
                self, "Node Error", f"Failed to create node {full_id}:\n{e}"
            )
            return None

    def save_graph(self):
        save_graph(self)

    def load_graph(self):
        load_graph(self)

    def on_node_double_clicked(self, node):
        if node.type_.startswith("com.pfd.custom_block"):
            code = node.get_property("code_content")
            dialog = CodeEditorDialog(code, node=node, parent=self)
            if dialog.exec_() == QtWidgets.QDialog.Accepted:
                new_code = dialog.get_code()
                node.set_property("code_content", new_code)

    def on_port_connected(self, port_in, port_out):
        # Wrap in try/except to prevent crashing the Viewer state if logic fails
        try:
            node_in = port_in.node()
            node_out = port_out.node()

            # 1. InputNode -> CustomBlockNode
            if node_out.type_.startswith("com.pfd.input") and node_in.type_.startswith(
                "com.pfd.custom_block"
            ):
                var_name = node_out.get_property("var_name")
                if var_name:
                    # Count how many input_nodes with same var_name are connected to this custom_block
                    count = 0
                    for inp_port in node_in.input_ports():
                        for connected_port in inp_port.connected_ports():
                            connected_node = connected_port.node()
                            if (
                                connected_node.type_.startswith("com.pfd.input")
                                and connected_node.get_property("var_name") == var_name
                            ):
                                count += 1
                    if count > 1:
                        # Disconnect and warn
                        port_in.disconnect_from(port_out)
                        QtWidgets.QMessageBox.warning(
                            self,
                            "Connection Error",
                            f"Cannot connect multiple inputs with the same variable name '{var_name}' to the same function block.",
                        )
                        return
                    if port_in.name() != var_name:
                        self.rename_port(
                            node_in,
                            port_in.name(),
                            var_name,
                            "input",
                            node_out,
                            preferred_target=var_name,
                        )
                    if port_out.name() != var_name:
                        self.rename_port(
                            node_out,
                            port_out.name(),
                            var_name,
                            "output",
                            node_in,
                            preferred_target=var_name,
                        )

            # 2. CustomBlockNode -> OutputNode
            if node_out.type_.startswith(
                "com.pfd.custom_block"
            ) and node_in.type_.startswith("com.pfd.output"):
                var_name = node_in.get_property("var_name")
                if var_name:
                    # Count how many output_nodes with same var_name are connected to this custom_block
                    count = 0
                    for out_port in node_out.output_ports():
                        for connected_port in out_port.connected_ports():
                            connected_node = connected_port.node()
                            if (
                                connected_node.type_.startswith("com.pfd.output")
                                and connected_node.get_property("var_name") == var_name
                            ):
                                count += 1
                    if count > 1:
                        # Disconnect and warn
                        port_in.disconnect_from(port_out)
                        QtWidgets.QMessageBox.warning(
                            self,
                            "Connection Error",
                            f"Cannot connect multiple quantities of interest with the same name '{var_name}' to the same function block.",
                        )
                        return
                    if port_out.name() != var_name:
                        self.rename_port(
                            node_out,
                            port_out.name(),
                            var_name,
                            "output",
                            node_in,
                            preferred_target=var_name,
                        )
                    if port_in.name() != var_name:
                        self.rename_port(
                            node_in,
                            port_in.name(),
                            var_name,
                            "input",
                            node_out,
                            preferred_target=var_name,
                        )

            # 3. CustomBlock -> IntermediateNode
            if node_out.type_.startswith(
                "com.pfd.custom_block"
            ) and node_in.type_.startswith("com.pfd.intermediate"):
                var_name = node_in.get_property("var_name")
                if var_name:
                    # Count how many intermediate_nodes with same var_name are connected to this custom_block
                    count = 0
                    for out_port in node_out.output_ports():
                        for connected_port in out_port.connected_ports():
                            connected_node = connected_port.node()
                            if (
                                connected_node.type_.startswith("com.pfd.intermediate")
                                and connected_node.get_property("var_name") == var_name
                            ):
                                count += 1
                    if count > 1:
                        # Disconnect and warn
                        port_in.disconnect_from(port_out)
                        QtWidgets.QMessageBox.warning(
                            self,
                            "Connection Error",
                            f"Cannot connect multiple intermediates with the same name '{var_name}' to the same function block.",
                        )
                        return
                    if port_out.name() != var_name:
                        self.rename_port(
                            node_out,
                            port_out.name(),
                            var_name,
                            "output",
                            node_in,
                            preferred_target=var_name,
                            fallback_target=port_in.name(),
                        )
                    if port_in.name() != var_name:
                        self.rename_port(
                            node_in,
                            port_in.name(),
                            var_name,
                            "input",
                            node_out,
                            preferred_target=port_out.name(),
                        )

            # 4. IntermediateNode -> CustomBlock
            if node_out.type_.startswith(
                "com.pfd.intermediate"
            ) and node_in.type_.startswith("com.pfd.custom_block"):
                var_name = node_out.get_property("var_name")
                if var_name:
                    # Count how many intermediate_nodes with same var_name are connected to this custom_block
                    count = 0
                    for inp_port in node_in.input_ports():
                        for connected_port in inp_port.connected_ports():
                            connected_node = connected_port.node()
                            if (
                                connected_node.type_.startswith("com.pfd.intermediate")
                                and connected_node.get_property("var_name") == var_name
                            ):
                                count += 1
                    if count > 1:
                        # Disconnect and warn
                        port_in.disconnect_from(port_out)
                        QtWidgets.QMessageBox.warning(
                            self,
                            "Connection Error",
                            f"Cannot connect multiple intermediates with the same variable name '{var_name}' to the same function block.",
                        )
                        return
                    if port_in.name() != var_name:
                        self.rename_port(
                            node_in,
                            port_in.name(),
                            var_name,
                            "input",
                            node_out,
                            preferred_target=var_name,
                            fallback_target=port_out.name(),
                        )
                    if port_out.name() != var_name:
                        self.rename_port(
                            node_out,
                            port_out.name(),
                            var_name,
                            "output",
                            node_in,
                            preferred_target=var_name,
                            fallback_target=port_in.name(),
                        )
        except Exception:
            logger.exception("Error in on_port_connected")

    def on_property_changed(self, node, prop_name, value):
        # Temporarily disconnect the signal to prevent recursion
        try:
            node.property_changed.disconnect(self.on_property_changed)
        except (RuntimeError, TypeError):
            pass  # Signal might not be connected

        try:
            # --- NEW: Surrogate Training Trigger ---
            # Handle the node's surrogate-training trigger.
            if prop_name == "surrogate_train_trigger":
                # Select the node first so the global function knows which one to train
                if self.current_graph:
                    self.current_graph.clear_selection()
                    node.set_selected(True)
                    # Call the training function (defined globally in this file)
                    train_selected_node_surrogate(self)
                return  # Early return, logic done, 'finally' block will reconnect signal

            # --- NEW: Update Surrogate Widget Status UI ---
            # Updates the embedded widget label if the status property changes (e.g. after training)
            if prop_name == "surrogate_status":
                if hasattr(node, "surrogate_widget"):
                    node.surrogate_widget.set_status(value)

            # --- EXISTING: Variable/Function Name Changes & Port Renaming ---
            if prop_name == "var_name" or prop_name == "func_name":
                node.set_name(value)

                # --- INPUT NODE RENAMING ---
                if node.type_.startswith("com.pfd.input"):
                    # 1. Identify the current port and its connections
                    outputs = node.output_ports()
                    if outputs:
                        old_port = outputs[0]
                        old_port_name = old_port.name()

                        # Only proceed if name actually changed
                        if old_port_name != value:
                            # Store connections to restore them later
                            # each entry is a port object on the OTHER node
                            connected_target_ports = old_port.connected_ports()

                            # Disconnect everything first
                            for cp in connected_target_ports:
                                old_port.disconnect_from(cp)

                            def rename_input_task():
                                # 2. Properly Delete and Add the port
                                node.delete_output(old_port_name)
                                node.add_output(value)

                                # 3. Reconnect to the same targets
                                new_port = node.get_output(value)
                                if new_port:
                                    for cp in connected_target_ports:
                                        # This 'connect_to' will trigger 'on_port_connected',
                                        # which detects the name mismatch on the OTHER end
                                        # and auto-updates the Function Block.
                                        new_port.connect_to(cp)

                            QtCore.QTimer.singleShot(0, rename_input_task)

                # --- OUTPUT NODE RENAMING ---
                elif node.type_.startswith("com.pfd.output"):
                    inputs = node.input_ports()
                    if inputs:
                        old_port = inputs[0]
                        old_port_name = old_port.name()

                        if old_port_name != value:
                            connected_source_ports = old_port.connected_ports()

                            for cp in connected_source_ports:
                                old_port.disconnect_from(cp)

                            def rename_output_task():
                                node.delete_input(old_port_name)
                                node.add_input(value)

                                new_port = node.get_input(value)
                                if new_port:
                                    for cp in connected_source_ports:
                                        # Reconnect (Triggers propagation)
                                        cp.connect_to(new_port)

                            QtCore.QTimer.singleShot(0, rename_output_task)

                # --- INTERMEDIATE NODE RENAMING ---
                elif node.type_.startswith("com.pfd.intermediate"):
                    # Must handle both input and output ports
                    in_ports = node.input_ports()
                    out_ports = node.output_ports()

                    old_in_name = in_ports[0].name() if in_ports else None
                    old_out_name = out_ports[0].name() if out_ports else None

                    # Store connections
                    in_connections = in_ports[0].connected_ports() if in_ports else []
                    out_connections = (
                        out_ports[0].connected_ports() if out_ports else []
                    )

                    # Disconnect
                    if in_ports:
                        in_ports[0].clear_connections()
                    if out_ports:
                        out_ports[0].clear_connections()

                    def rename_intermediate_task():
                        # Delete old
                        if old_in_name:
                            node.delete_input(old_in_name)
                        if old_out_name:
                            node.delete_output(old_out_name)

                        # Add new
                        node.add_input(value)
                        node.add_output(value)

                        # Reconnect
                        new_in = node.get_input(value)
                        new_out = node.get_output(value)

                        if new_in:
                            for cp in in_connections:
                                cp.connect_to(new_in)
                        if new_out:
                            for cp in out_connections:
                                new_out.connect_to(cp)

                    QtCore.QTimer.singleShot(0, rename_intermediate_task)

        except Exception:
            logger.exception("Error in on_property_changed")
        finally:
            # Reconnect the signal
            try:
                node.property_changed.connect(self.on_property_changed)
            except (RuntimeError, TypeError):
                logger.debug(
                    "Could not reconnect the node property signal.",
                    exc_info=True,
                )

    def rename_port(
        self,
        node,
        old_name,
        new_name,
        port_type,
        other_node,
        preferred_target=None,
        fallback_target=None,
    ):
        graph = self.current_graph

        def do_rename():
            # Safety check: ensure nodes still exist in graph
            if node not in graph.all_nodes() or other_node not in graph.all_nodes():
                return
            try:
                graph.port_connected.disconnect(self.on_port_connected)
            except TypeError:
                pass
            try:
                graph.property_changed.disconnect(self.on_property_changed)
            except TypeError:
                pass

            existing = [
                p.name()
                for p in (
                    node.input_ports() if port_type == "input" else node.output_ports()
                )
            ]
            if new_name in existing:
                return

            # START UNDO MACRO
            graph.undo_stack().beginMacro("Auto-Rename Port")

            try:
                node.set_port_deletion_allowed(True)

                if node.type_.startswith("com.pfd.custom_block"):
                    code = node.get_property("code_content")
                    if code:
                        pattern = r"\b" + re.escape(old_name) + r"\b"
                        new_code = re.sub(pattern, new_name, code)
                        if new_code != code:
                            node.set_property("code_content", new_code)

                if port_type == "input":
                    port_obj = node.get_input(old_name)
                    if port_obj:
                        for cp in port_obj.connected_ports():
                            if hasattr(graph, "disconnect_ports"):
                                graph.disconnect_ports(port_obj, cp)
                            else:
                                port_obj.disconnect_from(cp)
                    node.delete_input(old_name)
                    node.add_input(new_name)
                    new_port = node.get_input(new_name)
                    new_port.model.name = new_name
                    new_port.view.name = new_name
                    node.view.draw_node()
                    new_port.view.update()
                    node.view.update()

                    target_port = None
                    if preferred_target:
                        if other_node.type_.startswith(
                            "com.pfd.input"
                        ) or other_node.type_.startswith("com.pfd.intermediate"):
                            outputs = other_node.output_ports()
                            for p in outputs:
                                if p.name() == preferred_target:
                                    target_port = p
                                    break
                        elif other_node.type_.startswith("com.pfd.custom_block"):
                            outputs = other_node.output_ports()
                            for p in outputs:
                                if p.name() == preferred_target:
                                    target_port = p
                                    break
                    if not target_port and fallback_target:
                        if other_node.type_.startswith(
                            "com.pfd.input"
                        ) or other_node.type_.startswith("com.pfd.intermediate"):
                            outputs = other_node.output_ports()
                            for p in outputs:
                                if p.name() == fallback_target:
                                    target_port = p
                                    break
                        elif other_node.type_.startswith("com.pfd.custom_block"):
                            outputs = other_node.output_ports()
                            for p in outputs:
                                if p.name() == fallback_target:
                                    target_port = p
                                    break
                    if not target_port and (
                        other_node.type_.startswith("com.pfd.input")
                        or other_node.type_.startswith("com.pfd.intermediate")
                    ):
                        outputs = other_node.output_ports()
                        if outputs:
                            target_port = outputs[0]
                    if target_port:
                        if hasattr(graph, "connect_ports"):
                            graph.connect_ports(target_port, new_port)
                        else:
                            target_port.connect_to(new_port)

                else:  # port_type == 'output'
                    port_obj = node.get_output(old_name)
                    if port_obj:
                        for cp in port_obj.connected_ports():
                            if hasattr(graph, "disconnect_ports"):
                                graph.disconnect_ports(port_obj, cp)
                            else:
                                port_obj.disconnect_from(cp)
                    node.delete_output(old_name)
                    node.add_output(new_name)
                    new_port = node.get_output(new_name)
                    new_port.model.name = new_name
                    new_port.view.name = new_name
                    node.view.draw_node()
                    new_port.view.update()
                    node.view.update()

                    target_port = None
                    if preferred_target:
                        if other_node.type_.startswith(
                            "com.pfd.output"
                        ) or other_node.type_.startswith("com.pfd.intermediate"):
                            inputs = other_node.input_ports()
                            for p in inputs:
                                if p.name() == preferred_target:
                                    target_port = p
                                    break
                        elif other_node.type_.startswith("com.pfd.custom_block"):
                            inputs = other_node.input_ports()
                            for p in inputs:
                                if p.name() == preferred_target:
                                    target_port = p
                                    break
                    if not target_port and fallback_target:
                        if other_node.type_.startswith(
                            "com.pfd.output"
                        ) or other_node.type_.startswith("com.pfd.intermediate"):
                            inputs = other_node.input_ports()
                            for p in inputs:
                                if p.name() == fallback_target:
                                    target_port = p
                                    break
                        elif other_node.type_.startswith("com.pfd.custom_block"):
                            inputs = other_node.input_ports()
                            for p in inputs:
                                if p.name() == fallback_target:
                                    target_port = p
                                    break
                    if not target_port and (
                        other_node.type_.startswith("com.pfd.output")
                        or other_node.type_.startswith("com.pfd.intermediate")
                    ):
                        inputs = other_node.input_ports()
                        if inputs:
                            target_port = inputs[0]
                    if target_port:
                        if hasattr(graph, "connect_ports"):
                            graph.connect_ports(new_port, target_port)
                        else:
                            new_port.connect_to(target_port)
                graph._viewer.update()
            except Exception:
                logger.exception("Rename port error")
            finally:
                # END UNDO MACRO
                graph.undo_stack().endMacro()
                graph.port_connected.connect(self.on_port_connected)
                graph.property_changed.connect(self.on_property_changed)

        QtCore.QTimer.singleShot(0, do_rename)

    def validate_graph(self):
        return validate_graph(self)

    def get_compiled_code(self):
        try:
            return compile_systems(self.system_manager.systems)
        except ModelCompilationError as exc:
            QtWidgets.QMessageBox.critical(self, "Build Error", str(exc))
            return []

    def save_graph_to_file(self, folder_path):
        """Save the current graph to a file in the specified folder."""
        import os

        path = os.path.join(folder_path, "systems.json")
        save_graph_to_file(self, path)

    def load_graph_from_file(self, folder_path):
        """Load a graph from a file in the specified folder."""
        import os

        path = os.path.join(folder_path, "systems.json")
        if os.path.exists(path):
            load_graph_from_file(self, path)
