# Copyright (c) 2025 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""
Graph serialization and deserialization utilities for PyLCSS.

This module provides functions to save and load system graphs to/from JSON files,
enabling persistence of modeling work and sharing of system designs.
"""

from PySide6 import QtWidgets
import logging

logger = logging.getLogger(__name__)

def save_graph(widget):
    """
    Save all system graphs to a JSON file with file dialog.
    """
    # Get product name, default to "Product" if empty
    product_name = widget.system_manager.product_name.text().strip() or "Product"

    # Show save file dialog
    path, _ = QtWidgets.QFileDialog.getSaveFileName(
        widget,
        "Save Systems",
        f"{product_name}.json",
        "JSON (*.json)"
    )

    if path:
        save_graph_to_file(widget, path)

def save_graph_to_file(widget, path):
    """
    Save all system graphs to a specific JSON file path.
    """
    try:
        import json

        # Get product name, default to "Product" if empty
        product_name = widget.system_manager.product_name.text().strip() or "Product"

        # Prepare data structure
        data = {"product_name": product_name, "systems": []}

        # Serialize each system's graph
        for sys in widget.system_manager.systems:
            session_data = sys['graph'].serialize_session()
            data["systems"].append({
                "name": sys['name'],
                "graph": session_data
            })

        # Write to file
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

        # Only show message if widget is provided and has parent (to avoid popup during automated tests/saves)
        if widget and isinstance(widget, QtWidgets.QWidget) and widget.isVisible():
             QtWidgets.QMessageBox.information(widget, "Saved", "Systems saved successfully.")

    except Exception as e:
        if widget and isinstance(widget, QtWidgets.QWidget) and widget.isVisible():
            QtWidgets.QMessageBox.critical(widget, "Error", f"Failed to save: {e}")
        else:
            logger.exception("Error saving graph")

def load_graph(widget):
    """
    Load system graphs from a JSON file with file dialog.
    """
    # Show open file dialog
    path, _ = QtWidgets.QFileDialog.getOpenFileName(
        widget,
        "Load Systems",
        "",
        "JSON (*.json)"
    )

    if path:
        load_graph_from_file(widget, path)

def load_graph_from_file(widget, path):
    """
    Load system graphs from a specific JSON file path.
    """
    try:
        import json

        # Load JSON data
        with open(path, 'r') as f:
            data = json.load(f)

        # Clear existing systems and UI
        widget.system_manager.systems_list.clear()
        while widget.system_manager.graph_stack.count() > 0:
            w = widget.system_manager.graph_stack.widget(0)
            widget.system_manager.graph_stack.removeWidget(w)

        # Reset system manager state
        widget.system_manager.systems = []
        widget.system_manager.current_graph = None

        # Set product name
        product_name = data.get("product_name", "Product")
        widget.system_manager.product_name.setText(product_name)

        # Recreate each system
        for sys_data in data["systems"]:
            name = sys_data["name"]
            widget.system_manager._add_system(name)

            # Deserialize graph for this system
            graph = widget.system_manager.systems[-1]['graph']
            graph.deserialize_session(sys_data["graph"])
            
            # FIX: Ensure all output ports allow multiple connections
            # This fixes an issue where loaded graphs might have single-connection outputs
            # Also sync port names with var_name property
            from .node_types import InputNode, OutputNode
            
            for node in graph.all_nodes():
                # Fix multi-connection
                for port in node.output_ports():
                    port.model.multi_connection = True
                for port in node.input_ports():
                    port.model.multi_connection = False
                    
                # Sync port names
                if isinstance(node, InputNode):
                    if node.has_property('var_name'):
                        var_name = node.get_property('var_name')
                        if var_name and node.output_ports():
                            port = node.output_ports()[0]
                            old_name = port.name()
                            if old_name != var_name:
                                # Safe rename: save connections -> delete -> add -> reconnect
                                connections = port.connected_ports()
                                for cp in connections:
                                    port.disconnect_from(cp)
                                
                                node.delete_output(old_name)
                                node.add_output(var_name)
                                
                                new_port = node.get_output(var_name)
                                if new_port:
                                    new_port.model.multi_connection = True # Ensure property persists
                                    for cp in connections:
                                        new_port.connect_to(cp)
                            
                elif isinstance(node, OutputNode):
                    if node.has_property('var_name'):
                        var_name = node.get_property('var_name')
                        if var_name and node.input_ports():
                            port = node.input_ports()[0]
                            old_name = port.name()
                            if old_name != var_name:
                                # Safe rename: save connections -> delete -> add -> reconnect
                                connections = port.connected_ports()
                                for cp in connections:
                                    port.disconnect_from(cp)
                                    
                                node.delete_input(old_name)
                                node.add_input(var_name)
                                
                                new_port = node.get_input(var_name)
                                if new_port:
                                    new_port.model.multi_connection = False # Ensure property persists
                                    for cp in connections:
                                        new_port.connect_to(cp)

        # Select first system if any were loaded
        if widget.system_manager.systems:
            widget.system_manager.systems_list.setCurrentRow(0)

    except Exception as e:
        if widget and isinstance(widget, QtWidgets.QWidget) and widget.isVisible():
            QtWidgets.QMessageBox.critical(widget, "Error", f"Failed to load: {e}")
        else:
            logger.exception("Error loading graph")






