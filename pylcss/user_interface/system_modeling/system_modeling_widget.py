# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""
System modeling editor for PyLCSS.

This module provides the main visual modeling interface using NodeGraphQt,
allowing users to create and edit system models through a node-based
graphical interface.
"""

import re
from PySide6 import QtWidgets, QtCore, QtGui
import numpy as np
import logging

logger = logging.getLogger(__name__)

# Patch NodeGraphQt to use packaging.version instead of distutils.version
import sys
from packaging.version import Version

# Create a fake distutils.version module that redirects to packaging.version
if 'distutils.version' not in sys.modules:
    import types
    distutils_version = types.ModuleType('distutils.version')
    distutils_version.LooseVersion = Version
    sys.modules['distutils.version'] = distutils_version

from NodeGraphQt import NodeGraph
from NodeGraphQt.base.commands import NodesRemovedCmd

# Monkey patch NodesRemovedCmd to keep strong references to node views
# to prevent garbage collection during undo/redo operations
_original_init = NodesRemovedCmd.__init__

def _patched_init(self, graph, nodes, emit_signal=True):
    _original_init(self, graph, nodes, emit_signal)
    # Keep strong references to the views to prevent GC
    self.node_views = [node.view for node in nodes]

NodesRemovedCmd.__init__ = _patched_init
from .system_node_types import CustomBlockNode, InputNode, OutputNode, IntermediateNode, CodeEditorDialog
from pylcss.system_modeling.model_builder import GraphBuilder
from pylcss.system_modeling.system_list_manager import SystemManager
from pylcss.system_modeling.graph_validation import validate_graph
from pylcss.system_modeling.model_compilation import get_compiled_code
from pylcss.system_modeling.graph_input_output import save_graph, load_graph

class ModelingWidget(QtWidgets.QWidget):
    """
    Main widget for the system modeling environment.

    Provides a complete node-based modeling interface with system management,
    graph editing, validation, and code generation capabilities.
    """

    build_requested = QtCore.Signal()

    def __init__(self, parent=None):
        """
        Initialize the modeling widget.

        Sets up the UI components including system manager, graph area,
        toolbar with modeling tools, and search functionality.
        """
        super(ModelingWidget, self).__init__(parent)
        
        # --- Layout & Panels ---
        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.setSpacing(15)  # Add breathing room between elements
        self.layout.setContentsMargins(20, 20, 20, 20) # Add padding around the edges
        
        self.current_graph = None
        self._updating_flag = False  # Prevent recursion in property changes
        
        self.system_manager = SystemManager(self)
        self.system_manager.system_selected.connect(self.on_system_selected)
        self.system_manager.system_added.connect(self.setup_new_graph)
        
        # Clear default systems
        self.system_manager.system_selected.disconnect(self.on_system_selected)
        self.system_manager.systems_list.clear()
        while self.system_manager.graph_stack.count() > 0:
            w = self.system_manager.graph_stack.widget(0)
            self.system_manager.graph_stack.removeWidget(w)
        self.system_manager.systems = []
        self.system_manager.current_graph = None
        self.system_manager.system_selected.connect(self.on_system_selected)
        
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
        self.layout.insertWidget(0, self.toolbar)
        
        self.action_save = QtGui.QAction("Save", self)
        self.action_save.triggered.connect(self.save_graph)
        self.toolbar.addAction(self.action_save)
        
        self.action_load = QtGui.QAction("Load", self)
        self.action_load.triggered.connect(self.load_graph)
        self.toolbar.addAction(self.action_load)
        
        self.toolbar.addSeparator()
        
        self.action_build = QtGui.QAction("Build Model", self)
        self.action_build.triggered.connect(self.build_requested.emit)
        self.toolbar.addAction(self.action_build)
        
        self.action_validate = QtGui.QAction("Validate", self)
        self.action_validate.triggered.connect(self.validate_graph)
        self.toolbar.addAction(self.action_validate)
        
        self.toolbar.addSeparator()

        self.action_undo = QtGui.QAction("Undo", self)
        self.action_undo.setShortcut(QtGui.QKeySequence.Undo)
        self.action_undo.triggered.connect(self.undo)
        self.toolbar.addAction(self.action_undo)

        self.action_redo = QtGui.QAction("Redo", self)
        self.action_redo.setShortcut(QtGui.QKeySequence.Redo)
        self.action_redo.triggered.connect(self.redo)
        self.toolbar.addAction(self.action_redo)
        
        self.toolbar.addSeparator()
        
        self.action_delete = QtGui.QAction("Delete", self)
        self.action_delete.triggered.connect(self.delete_current)
        self.action_delete.setShortcut(QtGui.QKeySequence.Delete)
        self.action_delete.setShortcutContext(QtCore.Qt.WidgetWithChildrenShortcut)
        self.toolbar.addAction(self.action_delete)
        
        self.toolbar.addSeparator()
        
        # Node Buttons
        self.action_add_input = QtGui.QAction("Design Var", self)
        self.action_add_input.triggered.connect(self.add_input_node)
        self.toolbar.addAction(self.action_add_input)
        
        self.action_add_intermediate = QtGui.QAction("Intermediate", self)
        self.action_add_intermediate.triggered.connect(self.add_intermediate_node)
        self.toolbar.addAction(self.action_add_intermediate)
        
        self.action_add_output = QtGui.QAction("QoI", self)
        self.action_add_output.triggered.connect(self.add_output_node)
        self.toolbar.addAction(self.action_add_output)
        
        self.action_add_function = QtGui.QAction("Function Block", self)
        self.action_add_function.triggered.connect(self.add_function_node)
        self.toolbar.addAction(self.action_add_function)
        
        # Search Bar
        self.toolbar.addSeparator()
        lbl_search = QtWidgets.QLabel(" Search: ")
        self.toolbar.addWidget(lbl_search)
        self.search_bar = QtWidgets.QLineEdit()
        self.search_bar.setPlaceholderText("Find node...")
        self.search_bar.setMaximumWidth(150)
        self.search_bar.returnPressed.connect(self.find_node)
        self.toolbar.addWidget(self.search_bar)
        
        # Init
        self._current_undo_connection = None
        self._current_redo_connection = None
        self.update_undo_redo_actions()
        self.on_system_selected()

    @QtCore.Slot(bool)
    def _safe_set_undo_enabled(self, enabled):
        try:
            if hasattr(self, 'action_undo'):
                self.action_undo.setEnabled(enabled)
        except RuntimeError:
            pass

    @QtCore.Slot(bool)
    def _safe_set_redo_enabled(self, enabled):
        try:
            if hasattr(self, 'action_redo'):
                self.action_redo.setEnabled(enabled)
        except RuntimeError:
            pass

    def update_undo_redo_actions(self):
        """Update undo/redo action connections for the current graph."""
        # Disconnect previous connections
        if self._current_undo_connection:
            try:
                self._current_undo_connection.disconnect()
            except:
                pass
            self._current_undo_connection = None
            
        if self._current_redo_connection:
            try:
                self._current_redo_connection.disconnect()
            except:
                pass
            self._current_redo_connection = None
        
        # Connect to current graph's undo/redo if available
        if self.current_graph and hasattr(self.current_graph, 'undo_stack'):
            undo_stack = self.current_graph.undo_stack()
            if hasattr(undo_stack, 'canUndoChanged'):
                self._current_undo_connection = undo_stack.canUndoChanged.connect(self._safe_set_undo_enabled)
            if hasattr(undo_stack, 'canRedoChanged'):
                self._current_redo_connection = undo_stack.canRedoChanged.connect(self._safe_set_redo_enabled)
                
            # Update button states immediately
            self._safe_set_undo_enabled(undo_stack.canUndo())
            self._safe_set_redo_enabled(undo_stack.canRedo())

    def find_node(self):
        """Search for nodes by name or variable name in the current graph."""
        text = self.search_bar.text().lower().strip()
        if not text or not self.current_graph: return
        
        found = False
        for node in self.current_graph.all_nodes():
            if text in node.name().lower() or text in str(node.get_property('var_name')).lower():
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
        
        # Update undo/redo connections
        self.update_undo_redo_actions()

    def setup_new_graph(self, graph):
        """Set up event connections for a newly created graph."""
        graph.node_double_clicked.connect(self.on_node_double_clicked)
        graph.port_connected.connect(self.on_port_connected)
        graph.property_changed.connect(self.on_property_changed)
        self.setup_context_menu_for_graph(graph)
        
        # --- FIX: Use .scene() instead of ._viewer._scene ---
        # The internal attribute _scene is gone/private. Use the public API.
        if hasattr(graph, 'scene'):
            scene = graph.scene()
            if scene:
                # Disable BSP indexing to prevent crashes during rapid remove/add operations
                scene.setItemIndexMethod(QtWidgets.QGraphicsScene.NoIndex)

        # --- FIX: Set Undo Limit Here (Only once per graph) ---
        if hasattr(graph, 'undo_stack'):
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
        if self.current_graph: self.current_graph.delete_nodes(self.current_graph.selected_nodes())
    def add_input_node(self):
        if self.current_graph: self.create_node_for_graph(self.current_graph, InputNode)
    def add_intermediate_node(self):
        if self.current_graph: self.create_node_for_graph(self.current_graph, IntermediateNode)
    def add_output_node(self):
        if self.current_graph: self.create_node_for_graph(self.current_graph, OutputNode)
    def add_function_node(self):
        if self.current_graph: self.create_node_for_graph(self.current_graph, CustomBlockNode)

    def setup_context_menu_for_graph(self, graph):
        menu = graph.context_menu()

        # Track if commands have already been added to prevent duplication
        if hasattr(menu, '_commands_added') and menu._commands_added:
            return

        # Remove default undo/redo actions from the context menu
        qmenu = menu.qmenu
        actions_to_remove = []
        for action in qmenu.actions():
            if action.text() in ['&Undo', '&Redo']:
                actions_to_remove.append(action)
        
        for action in actions_to_remove:
            qmenu.removeAction(action)

        menu.add_command('Create Black Box Function', lambda: self.create_node_for_graph(graph, CustomBlockNode), 'Shift+F')
        menu.add_command('Create Design Variable', lambda: self.create_node_for_graph(graph, InputNode), 'Shift+I')
        menu.add_command('Create Quantity of Interest', lambda: self.create_node_for_graph(graph, OutputNode), 'Shift+O')
        menu.add_command('Create Intermediate Variable', lambda: self.create_node_for_graph(graph, IntermediateNode), 'Shift+V')
        menu.add_separator()
        menu.add_command('Delete', lambda: graph.delete_nodes(graph.selected_nodes()), 'Del')
        menu.add_separator()
        menu.add_command('Fit to View', graph.fit_to_selection, 'F')

        # Mark commands as added
        menu._commands_added = True

    def create_node_for_graph(self, graph, node_class):
        target_id = node_class.__identifier__
        registered = graph.registered_nodes()
        full_id = None
        if target_id in registered: full_id = target_id
        else:
            expected_id = f"{target_id}.{node_class.__name__}"
            if expected_id in registered: full_id = expected_id
            else:
                for nid in registered:
                    if nid.startswith(target_id):
                        full_id = nid
                        break
        if not full_id:
            QtWidgets.QMessageBox.warning(self, "Node Error", f"Could not find registered node for {target_id}\nAvailable: {registered}")
            return None
        try:
            node = graph.create_node(full_id, pos=[0, 0])
            return node
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Node Error", f"Failed to create node {full_id}:\n{e}")
            return None

    def save_graph(self):
        save_graph(self)

    def load_graph(self):
        load_graph(self)

    def on_node_double_clicked(self, node):
        if node.type_.startswith('com.pfd.custom_block'):
            code = node.get_property('code_content')
            dialog = CodeEditorDialog(code, node=node, parent=self)
            if dialog.exec_() == QtWidgets.QDialog.Accepted:
                new_code = dialog.get_code()
                node.set_property('code_content', new_code)

    def on_port_connected(self, port_in, port_out):
        # Wrap in try/except to prevent crashing the Viewer state if logic fails
        try:
            node_in = port_in.node()
            node_out = port_out.node()
            
            # 1. InputNode -> CustomBlockNode
            if node_out.type_.startswith('com.pfd.input') and node_in.type_.startswith('com.pfd.custom_block'):
                var_name = node_out.get_property('var_name')
                if var_name:
                    # Count how many input_nodes with same var_name are connected to this custom_block
                    count = 0
                    for inp_port in node_in.input_ports():
                        for connected_port in inp_port.connected_ports():
                            connected_node = connected_port.node()
                            if connected_node.type_.startswith('com.pfd.input') and connected_node.get_property('var_name') == var_name:
                                count += 1
                    if count > 1:
                        # Disconnect and warn
                        port_in.disconnect_from(port_out)
                        QtWidgets.QMessageBox.warning(self, "Connection Error", 
                            f"Cannot connect multiple inputs with the same variable name '{var_name}' to the same function block.")
                        return
                    if port_in.name() != var_name:
                        self.rename_port(node_in, port_in.name(), var_name, 'input', node_out, preferred_target=var_name)
                    if port_out.name() != var_name:
                        self.rename_port(node_out, port_out.name(), var_name, 'output', node_in, preferred_target=var_name)
                        
            # 2. CustomBlockNode -> OutputNode
            if node_out.type_.startswith('com.pfd.custom_block') and node_in.type_.startswith('com.pfd.output'):
                var_name = node_in.get_property('var_name')
                if var_name:
                    # Count how many output_nodes with same var_name are connected to this custom_block
                    count = 0
                    for out_port in node_out.output_ports():
                        for connected_port in out_port.connected_ports():
                            connected_node = connected_port.node()
                            if connected_node.type_.startswith('com.pfd.output') and connected_node.get_property('var_name') == var_name:
                                count += 1
                    if count > 1:
                        # Disconnect and warn
                        port_in.disconnect_from(port_out)
                        QtWidgets.QMessageBox.warning(self, "Connection Error", 
                            f"Cannot connect multiple quantities of interest with the same name '{var_name}' to the same function block.")
                        return
                    if port_out.name() != var_name:
                        self.rename_port(node_out, port_out.name(), var_name, 'output', node_in, preferred_target=var_name)
                    if port_in.name() != var_name:
                        self.rename_port(node_in, port_in.name(), var_name, 'input', node_out, preferred_target=var_name)

            # 3. CustomBlock -> IntermediateNode
            if node_out.type_.startswith('com.pfd.custom_block') and node_in.type_.startswith('com.pfd.intermediate'):
                var_name = node_in.get_property('var_name')
                if var_name:
                    # Count how many intermediate_nodes with same var_name are connected to this custom_block
                    count = 0
                    for out_port in node_out.output_ports():
                        for connected_port in out_port.connected_ports():
                            connected_node = connected_port.node()
                            if connected_node.type_.startswith('com.pfd.intermediate') and connected_node.get_property('var_name') == var_name:
                                count += 1
                    if count > 1:
                        # Disconnect and warn
                        port_in.disconnect_from(port_out)
                        QtWidgets.QMessageBox.warning(self, "Connection Error", 
                            f"Cannot connect multiple intermediates with the same name '{var_name}' to the same function block.")
                        return
                    if port_out.name() != var_name:
                        self.rename_port(node_out, port_out.name(), var_name, 'output', node_in, preferred_target=var_name, fallback_target=port_in.name())
                    if port_in.name() != var_name:
                        self.rename_port(node_in, port_in.name(), var_name, 'input', node_out, preferred_target=port_out.name())

            # 4. IntermediateNode -> CustomBlock
            if node_out.type_.startswith('com.pfd.intermediate') and node_in.type_.startswith('com.pfd.custom_block'):
                var_name = node_out.get_property('var_name')
                if var_name:
                    # Count how many intermediate_nodes with same var_name are connected to this custom_block
                    count = 0
                    for inp_port in node_in.input_ports():
                        for connected_port in inp_port.connected_ports():
                            connected_node = connected_port.node()
                            if connected_node.type_.startswith('com.pfd.intermediate') and connected_node.get_property('var_name') == var_name:
                                count += 1
                    if count > 1:
                        # Disconnect and warn
                        port_in.disconnect_from(port_out)
                        QtWidgets.QMessageBox.warning(self, "Connection Error", 
                            f"Cannot connect multiple intermediates with the same variable name '{var_name}' to the same function block.")
                        return
                    if port_in.name() != var_name:
                        self.rename_port(node_in, port_in.name(), var_name, 'input', node_out, preferred_target=var_name, fallback_target=port_out.name())
                    if port_out.name() != var_name:
                        self.rename_port(node_out, port_out.name(), var_name, 'output', node_in, preferred_target=var_name, fallback_target=port_in.name())
        except Exception as e:
            logger.exception("Error in on_port_connected")

    def on_property_changed(self, node, prop_name, value):
        # Temporarily disconnect the signal to prevent recursion
        try:
            node.property_changed.disconnect(self.on_property_changed)
        except:
            pass  # Signal might not be connected
        
        try:
            # --- NEW: Surrogate Training Trigger ---
            # Handles the signal from the new SurrogateControlWidget 'Train' button
            if prop_name == 'surrogate_train_trigger':
                # Select the node first so the global function knows which one to train
                if self.current_graph:
                    self.current_graph.clear_selection()
                    node.set_selected(True)
                    # Call the training function (defined globally in this file)
                    train_selected_node_surrogate(self)
                return  # Early return, logic done, 'finally' block will reconnect signal
                
            # --- NEW: Update Surrogate Widget Status UI ---
            # Updates the embedded widget label if the status property changes (e.g. after training)
            if prop_name == 'surrogate_status':
                if hasattr(node, 'surrogate_widget'):
                    node.surrogate_widget.set_status(value)

            # --- EXISTING: Variable/Function Name Changes & Port Renaming ---
            if prop_name == 'var_name' or prop_name == 'func_name':
                node.set_name(value)
                
                # --- INPUT NODE RENAMING ---
                if node.type_.startswith('com.pfd.input'):
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
                elif node.type_.startswith('com.pfd.output'):
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
                elif node.type_.startswith('com.pfd.intermediate'):
                    # Must handle both input and output ports
                    in_ports = node.input_ports()
                    out_ports = node.output_ports()
                    
                    old_in_name = in_ports[0].name() if in_ports else None
                    old_out_name = out_ports[0].name() if out_ports else None
                    
                    # Store connections
                    in_connections = in_ports[0].connected_ports() if in_ports else []
                    out_connections = out_ports[0].connected_ports() if out_ports else []
                    
                    # Disconnect
                    if in_ports: in_ports[0].clear_connections()
                    if out_ports: out_ports[0].clear_connections()
                    
                    def rename_intermediate_task():
                        # Delete old
                        if old_in_name: node.delete_input(old_in_name)
                        if old_out_name: node.delete_output(old_out_name)
                        
                        # Add new
                        node.add_input(value)
                        node.add_output(value)
                        
                        # Reconnect
                        new_in = node.get_input(value)
                        new_out = node.get_output(value)
                        
                        if new_in:
                            for cp in in_connections: cp.connect_to(new_in)
                        if new_out:
                            for cp in out_connections: new_out.connect_to(cp)

                    QtCore.QTimer.singleShot(0, rename_intermediate_task)

        except Exception as e:
            logger.exception("Error in on_property_changed")
        finally:
            # Reconnect the signal
            try:
                node.property_changed.connect(self.on_property_changed)
            except:
                pass

    def rename_port(self, node, old_name, new_name, port_type, other_node, preferred_target=None, fallback_target=None):
        graph = self.current_graph
        def do_rename():
            # Safety check: ensure nodes still exist in graph
            if node not in graph.all_nodes() or other_node not in graph.all_nodes():
                return
            try: graph.port_connected.disconnect(self.on_port_connected)
            except TypeError: pass
            try: graph.property_changed.disconnect(self.on_property_changed)
            except TypeError: pass
            
            existing = [p.name() for p in (node.input_ports() if port_type == 'input' else node.output_ports())]
            if new_name in existing: return

            # START UNDO MACRO
            graph.undo_stack().beginMacro("Auto-Rename Port")

            try:
                node.set_port_deletion_allowed(True)
                
                if node.type_.startswith('com.pfd.custom_block'):
                    code = node.get_property('code_content')
                    if code:
                        pattern = r'\b' + re.escape(old_name) + r'\b'
                        new_code = re.sub(pattern, new_name, code)
                        if new_code != code:
                            node.set_property('code_content', new_code)

                if port_type == 'input':
                    port_obj = node.get_input(old_name)
                    if port_obj:
                        for cp in port_obj.connected_ports():
                            if hasattr(graph, 'disconnect_ports'): graph.disconnect_ports(port_obj, cp)
                            else: port_obj.disconnect_from(cp)
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
                        if other_node.type_.startswith('com.pfd.input') or other_node.type_.startswith('com.pfd.intermediate'):
                            outputs = other_node.output_ports()
                            for p in outputs:
                                if p.name() == preferred_target: target_port = p; break
                        elif other_node.type_.startswith('com.pfd.custom_block'):
                             outputs = other_node.output_ports()
                             for p in outputs:
                                if p.name() == preferred_target: target_port = p; break
                    if not target_port and fallback_target:
                        if other_node.type_.startswith('com.pfd.input') or other_node.type_.startswith('com.pfd.intermediate'):
                            outputs = other_node.output_ports()
                            for p in outputs:
                                if p.name() == fallback_target: target_port = p; break
                        elif other_node.type_.startswith('com.pfd.custom_block'):
                             outputs = other_node.output_ports()
                             for p in outputs:
                                if p.name() == fallback_target: target_port = p; break
                    if not target_port and (other_node.type_.startswith('com.pfd.input') or other_node.type_.startswith('com.pfd.intermediate')):
                        outputs = other_node.output_ports()
                        if outputs: target_port = outputs[0]
                    if target_port:
                        if hasattr(graph, 'connect_ports'): graph.connect_ports(target_port, new_port)
                        else: target_port.connect_to(new_port)

                else: # port_type == 'output'
                    port_obj = node.get_output(old_name)
                    if port_obj:
                        for cp in port_obj.connected_ports():
                            if hasattr(graph, 'disconnect_ports'): graph.disconnect_ports(port_obj, cp)
                            else: port_obj.disconnect_from(cp)
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
                        if other_node.type_.startswith('com.pfd.output') or other_node.type_.startswith('com.pfd.intermediate'):
                            inputs = other_node.input_ports()
                            for p in inputs:
                                if p.name() == preferred_target: target_port = p; break
                        elif other_node.type_.startswith('com.pfd.custom_block'):
                             inputs = other_node.input_ports()
                             for p in inputs:
                                if p.name() == preferred_target: target_port = p; break
                    if not target_port and fallback_target:
                        if other_node.type_.startswith('com.pfd.output') or other_node.type_.startswith('com.pfd.intermediate'):
                            inputs = other_node.input_ports()
                            for p in inputs:
                                if p.name() == fallback_target: target_port = p; break
                        elif other_node.type_.startswith('com.pfd.custom_block'):
                             inputs = other_node.input_ports()
                             for p in inputs:
                                if p.name() == fallback_target: target_port = p; break
                    if not target_port and (other_node.type_.startswith('com.pfd.output') or other_node.type_.startswith('com.pfd.intermediate')):
                        inputs = other_node.input_ports()
                        if inputs: target_port = inputs[0]
                    if target_port:
                        if hasattr(graph, 'connect_ports'): graph.connect_ports(new_port, target_port)
                        else: new_port.connect_to(target_port)
                graph._viewer.update()
            except Exception as e:
                logger.exception("Rename port error")
            finally:
                # END UNDO MACRO
                graph.undo_stack().endMacro()
                graph.port_connected.connect(self.on_port_connected)
                graph.property_changed.connect(self.on_property_changed)
        QtCore.QTimer.singleShot(0, do_rename)
    
    def validate_graph(self):
        validate_graph(self)
    
    def get_compiled_code(self):
        return get_compiled_code(self)

    def save_graph_to_file(self, folder_path):
        """Save the current graph to a file in the specified folder."""
        import os
        from pylcss.system_modeling.graph_input_output import save_graph_to_file
        path = os.path.join(folder_path, 'systems.json')
        save_graph_to_file(self, path)

    def load_graph_from_file(self, folder_path):
        """Load a graph from a file in the specified folder."""
        import os
        from pylcss.system_modeling.graph_input_output import load_graph_from_file
        path = os.path.join(folder_path, 'systems.json')
        if os.path.exists(path):
            load_graph_from_file(self, path)


class NodeTrainingWorker(QtCore.QThread):
    """
    Worker thread for training surrogate models for selected nodes.
    Generates training data by sampling the system up to the target node,
    then trains a neural network surrogate model.
    """

    progress_updated = QtCore.Signal(int, str)  # progress_percent, status_message
    training_finished = QtCore.Signal(bool, str)  # success, message

    def __init__(self, graph_builder, nodes, input_nodes, output_nodes, target_node,
                 num_samples=1000, test_size=0.2, random_state=42):
        super().__init__()
        self.graph_builder = graph_builder
        self.nodes = nodes
        self.input_nodes = input_nodes
        self.output_nodes = output_nodes
        self.target_node = target_node
        self.num_samples = num_samples
        self.test_size = test_size
        self.random_state = random_state

    def run(self):
        try:
            # Import sklearn modules here to avoid numpy compatibility issues at module load
            import joblib
            import warnings
            from sklearn.neural_network import MLPRegressor
            from sklearn.preprocessing import StandardScaler
            from sklearn.pipeline import Pipeline
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import mean_squared_error
        
        except ImportError:
            self.training_finished.emit(False, "Scikit-learn not installed. Please run: pip install scikit-learn joblib")
            return

        try:
            self.progress_updated.emit(0, "Generating spy model code...")

            # Build spy model to capture training data
            spy_code, spy_inputs, spy_outputs = self.graph_builder.build_spy_model(
                self.nodes, self.input_nodes, self.output_nodes,
                self.target_node.id, "spy_model"
            )

            self.progress_updated.emit(10, "Compiling spy model...")

            # --- FIX START ---
            # Use a single dictionary for both globals and locals so functions can see each other
            exec_context = {"np": np}
            exec(spy_code, exec_context, exec_context)
            spy_func = exec_context["spy_model"]
            # --- FIX END ---

            self.progress_updated.emit(20, f"Generating {self.num_samples} training samples...")

            # Generate training data by sampling input space
            X_data = []
            y_data = []

            # Get input bounds
            input_bounds = []
            for inp_node in self.input_nodes:
                if inp_node.has_property('input_props'):
                    props = inp_node.get_property('input_props')
                    min_val = float(props.get('min', '0.0'))
                    max_val = float(props.get('max', '10.0'))
                else:
                    min_val = float(inp_node.get_property('min'))
                    max_val = float(inp_node.get_property('max'))
                input_bounds.append((min_val, max_val))

            # Sample input space
            np.random.seed(self.random_state)
            for i in range(self.num_samples):
                # Generate random input sample within bounds
                sample_inputs = []
                for min_val, max_val in input_bounds:
                    sample_inputs.append(np.random.uniform(min_val, max_val))

                # Execute spy model to get corresponding outputs
                inputs_dict, outputs_dict = spy_func(*sample_inputs)

                # Extract input and output values
                X_sample = [inputs_dict[f'input_{j}'] for j in range(len(spy_inputs))]
                y_sample = [outputs_dict[f'output_{j}'] for j in range(len(spy_outputs))]

                X_data.append(X_sample)
                y_data.append(y_sample)

                if (i + 1) % 100 == 0:
                    progress = 20 + int(50 * (i + 1) / self.num_samples)
                    self.progress_updated.emit(progress, f"Generated {i + 1}/{self.num_samples} samples...")

            X = np.array(X_data)
            y = np.array(y_data)

            self.progress_updated.emit(70, "Training neural network surrogate...")

            # Create and train surrogate model
            # Use pipeline with StandardScaler and MLPRegressor
            model = Pipeline([
                ('scaler', StandardScaler()),
                ('regressor', MLPRegressor(
                    hidden_layer_sizes=(100, 50),
                    activation='relu',
                    solver='adam',
                    alpha=0.001,
                    batch_size='auto',
                    learning_rate='adaptive',
                    learning_rate_init=0.01,
                    max_iter=1000,
                    random_state=self.random_state,
                    early_stopping=True,
                    validation_fraction=0.1,
                    n_iter_no_change=10
                ))
            ])

            # Split data for training and validation
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.test_size, random_state=self.random_state
            )

            # Train model
            model.fit(X_train, y_train)

            self.progress_updated.emit(90, "Evaluating model performance...")

            # Evaluate model performance
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)

            self.progress_updated.emit(95, "Saving surrogate model...")

            # Save model to file
            model_path = f"surrogate_{self.target_node.id.replace('-', '_')}.joblib"
            joblib.dump(model, model_path)

            # Update node properties
            self.target_node.set_property('surrogate_model_path', model_path)
            self.target_node.set_property('surrogate_status', f'Trained (RMSE: {rmse:.4f})')

            self.progress_updated.emit(100, "Training completed successfully!")
            self.training_finished.emit(True, f"Surrogate model trained successfully. RMSE: {rmse:.4f}")

        except Exception as e:
            self.training_finished.emit(False, f"Training failed: {str(e)}")


def train_selected_node_surrogate(modeling_widget, num_samples=1000):
    """
    Train a surrogate model for the currently selected node.

    Args:
        modeling_widget: The ModelingWidget instance
        num_samples: Number of training samples to generate
    """
    # Ensure a graph is loaded
    if not modeling_widget.current_graph:
        QtWidgets.QMessageBox.warning(
            modeling_widget, "No Graph",
            "No graph is currently loaded. Please load or create a graph."
        )
        return

    # Get selected nodes
    selected_nodes = modeling_widget.current_graph.selected_nodes()
    if not selected_nodes:
        QtWidgets.QMessageBox.warning(
            modeling_widget, "No Selection",
            "Please select a custom block node to train a surrogate model."
        )
        return

    if len(selected_nodes) > 1:
        QtWidgets.QMessageBox.warning(
            modeling_widget, "Multiple Selection",
            "Please select only one node for surrogate training."
        )
        return

    target_node = selected_nodes[0]
    if not target_node.type_.startswith('com.pfd.custom_block'):
        QtWidgets.QMessageBox.warning(
            modeling_widget, "Invalid Selection",
            "Please select a custom block node for surrogate training."
        )
        return

    # Ask user for sample count
    val, ok = QtWidgets.QInputDialog.getInt(
        modeling_widget, "Training Settings", 
        "Number of Samples:", 
        value=num_samples, 
        min=100, 
        max=1000000, 
        step=1000
    )
    if not ok:
        return
    num_samples = val

    # Confirm training
    reply = QtWidgets.QMessageBox.question(
        modeling_widget, "Train Surrogate Model",
        f"Train surrogate model for node '{target_node.name()}'?\n\n"
        f"This will generate {num_samples} training samples and may take some time.",
        QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No
    )

    if reply != QtWidgets.QMessageBox.Yes:
        return

    # Get graph data
    all_nodes = modeling_widget.current_graph.all_nodes()
    input_nodes = [n for n in all_nodes if n.type_.startswith('com.pfd.input')]
    output_nodes = [n for n in all_nodes if n.type_.startswith('com.pfd.output')]

    # Pass the current_graph to GraphBuilder
    graph_builder = GraphBuilder(modeling_widget.current_graph)

    worker = NodeTrainingWorker(
        graph_builder, all_nodes, input_nodes, output_nodes, target_node, num_samples
    )

    # Create progress dialog
    progress_dialog = QtWidgets.QProgressDialog(
        "Training surrogate model...", "Cancel", 0, 100, modeling_widget
    )
    progress_dialog.setWindowModality(QtCore.Qt.WindowModal)
    progress_dialog.setAutoClose(True)
    progress_dialog.setAutoReset(True)

    # Connect signals
    worker.progress_updated.connect(
        lambda progress, msg: (
            progress_dialog.setValue(progress),
            progress_dialog.setLabelText(msg)
        )
    )

    worker.training_finished.connect(
        lambda success, msg: (
            progress_dialog.close(),
            QtWidgets.QMessageBox.information(
                modeling_widget, "Training Complete" if success else "Training Failed", msg
            ) if success else QtWidgets.QMessageBox.critical(
                modeling_widget, "Training Failed", msg
            ),
            # Refresh node display if successful
            target_node.view.update() if success else None
        )
    )

    # Handle cancellation
    progress_dialog.canceled.connect(worker.terminate)

    # Start training
    worker.start()
    progress_dialog.exec_()







