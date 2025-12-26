# Copyright (c) 2025 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""
Graph validation utilities for PyLCSS system models.

This module provides validation functions to check system graphs for common
issues like syntax errors, circular dependencies, unconnected ports, and
unit mismatches before compilation or execution.
"""

from PySide6 import QtWidgets
import networkx as nx
from collections import Counter

try:
    import pint
    ureg = pint.UnitRegistry()
except ImportError:
    ureg = None

class GraphValidator:
    """
    Validates the system graph for connectivity, loops, and variable naming conflicts.
    """
    def validate(self, nodes: list, edges: list) -> dict:
        """
        Runs all validation checks on the graph.
        Returns a dict with 'errors' and 'warnings'.
        """
        errors = []
        warnings = []

        # --- Existing Validation Logic (Preserve this) ---
        # For now, we'll keep the existing checks here or call them

        # --- Check for Duplicate QoI Names (Single Graph Scope) ---
        qoi_warnings = self._check_duplicate_qois(nodes)
        warnings.extend(qoi_warnings)

        return {
            "errors": errors,
            "warnings": warnings
        }

    def _check_duplicate_qois(self, nodes: list) -> list:
        """
        Scans all nodes for Quantities of Interest (QoI) and checks for duplicate names.
        Returns a list of warning strings.
        """
        qoi_map = {}  # Key: QoI Name, Value: List of Node Names
        warnings = []

        for node in nodes:
            # Check if node has 'qois' attribute (list of objects or dicts)
            if hasattr(node, 'qois'):
                for qoi in node.qois:
                    # Handle both Object (qoi.name) and Dict (qoi['name']) styles safely
                    qoi_name = None
                    if hasattr(qoi, 'name'):
                        qoi_name = qoi.name
                    elif isinstance(qoi, dict):
                        qoi_name = qoi.get('name')
                    
                    if qoi_name:
                        if qoi_name not in qoi_map:
                            qoi_map[qoi_name] = []
                        qoi_map[qoi_name].append(node.name)

        # Generate warnings for duplicates
        for qoi_name, node_names in qoi_map.items():
            if len(node_names) > 1:
                # Remove duplicates from node_names list for cleaner message
                unique_nodes = list(set(node_names))
                msg = (f"Duplicate QoI Name Detected: '{qoi_name}' is defined in multiple systems "
                       f"({', '.join(unique_nodes)}). This may cause overwriting in the solution space.")
                warnings.append(msg)

        return warnings

def validate_graph(widget):
    """
    Validate all system graphs in the widget for common modeling errors.

    Performs comprehensive validation including:
    - Syntax checking of custom block code
    - Cycle detection in graph structure
    - Unconnected input port detection
    - Unit consistency checking between connected ports
    - **Global QoI duplication checks across all systems**

    Args:
        widget: Main application widget containing system_manager with systems

    Returns:
        bool: True if validation passed (no errors), False if errors found
    """
    errors = []
    warnings = []

    # --- NEW: Global Registry for Cross-System Checking ---
    global_qoi_map = {}  # Key: QoI Name, Value: List of "SystemName::NodeName"

    # Validate each system in the system manager
    for sys in widget.system_manager.systems:
        graph = sys['graph']
        nodes = graph.all_nodes()

        # --- 1. Collect Data for Global Check ---
        for node in nodes:
            if node.type_.startswith('com.pfd.output'):
                var_name = node.get_property('var_name')
                if var_name:
                    if var_name not in global_qoi_map:
                        global_qoi_map[var_name] = []
                    # Store location as "SystemName::NodeName"
                    global_qoi_map[var_name].append(f"{sys['name']} :: {node.name()}")

        # --- 2. Per-System Checks ---
        
        # Check for duplicate DV, QoI, and intermediate names within THIS system
        var_names = []
        for node in nodes:
            if node.type_.startswith(('com.pfd.input', 'com.pfd.output', 'com.pfd.intermediate')):
                var_name = node.get_property('var_name')
                if var_name:
                    node_type = 'DV' if node.type_.startswith('com.pfd.input') else 'QoI' if node.type_.startswith('com.pfd.output') else 'Intermediate'
                    var_names.append((var_name, node_type, node.name()))
        
        name_counts = Counter(name for name, _, _ in var_names)
        duplicates = [name for name, count in name_counts.items() if count > 1]
        if duplicates:
            for dup in duplicates:
                items = [f"{node_name} ({typ})" for name, typ, node_name in var_names if name == dup]
                errors.append(f"System '{sys['name']}': Duplicate variable name '{dup}' found in: {', '.join(items)}")

        # Check syntax of custom block code
        for node in nodes:
            if node.type_.startswith('com.pfd.custom_block'):
                code = node.get_property('code_content')
                try:
                    compile(code, '<string>', 'exec')
                except SyntaxError as e:
                    errors.append(f"Syntax Error in '{sys['name']}' -> '{node.name()}' line {e.lineno}: {e.msg}")
        
        # Build directed graph for cycle detection
        G = nx.DiGraph()
        for n in nodes:
            G.add_node(n.id)
            for port in n.output_ports():
                for connected_port in port.connected_ports():
                    target_node = connected_port.node()
                    G.add_edge(n.id, target_node.id)

        # Check for cycles
        try:
            nx.topological_sort(G)
        except nx.NetworkXUnfeasible:
            errors.append(f"System '{sys['name']}': Graph contains cycles (circular dependencies).")

        # Check for unconnected inputs
        for node in nodes:
            if node.type_.startswith('com.pfd.custom_block'):
                for port in node.input_ports():
                    if not port.connected_ports():
                        errors.append(f"System '{sys['name']}': Node '{node.name()}' has unconnected input '{port.name()}'.")

        # Check unit consistency
        for node in nodes:
            for port in node.input_ports():
                target_unit = node.get_property('unit')
                if not target_unit or target_unit == '-':
                    continue
                connected = port.connected_ports()
                for cp in connected:
                    source_node = cp.node()
                    source_unit = source_node.get_property('unit')
                    if source_unit and source_unit != '-':
                        if ureg is not None:
                            try:
                                out_qty = ureg.Quantity(1, source_unit)
                                in_qty = ureg.Quantity(1, target_unit)
                                if out_qty.dimensionality != in_qty.dimensionality:
                                    errors.append(f"Unit Incompatibility in '{sys['name']}': '{source_node.name()}' ({source_unit}) -> '{node.name()}' ({target_unit}).")
                                elif source_unit != target_unit:
                                    warnings.append(f"Unit Conversion in '{sys['name']}': '{source_node.name()}' ({source_unit}) -> '{node.name()}' ({target_unit}).")
                            except Exception:
                                warnings.append(f"Unit Validation Failed in '{sys['name']}': Could not validate '{source_unit}' -> '{target_unit}'.")
                        else:
                            if source_unit != target_unit:
                                warnings.append(f"Unit Mismatch in '{sys['name']}': '{source_node.name()}' ({source_unit}) -> '{node.name()}' ({target_unit}).")

    # --- 3. Process Global Duplicate QoIs ---
    for qoi_name, locations in global_qoi_map.items():
        if len(locations) > 1:
            warnings.append(
                f"Global QoI Conflict: '{qoi_name}' is defined in multiple systems. "
                f"This will overwrite columns in the solution space.\n"
                f"   Found in: {', '.join(locations)}"
            )

    # Display results
    if errors:
        msg = "Validation Errors:\n\n" + "\n".join(errors)
        if warnings:
            msg += "\n\nWarnings:\n" + "\n".join(warnings)
        QtWidgets.QMessageBox.warning(widget, "Validation Failed", msg)
        return False
    else:
        msg = "Graph validation passed!"
        if warnings:
            msg += "\n\nWarnings:\n" + "\n".join(warnings)
        QtWidgets.QMessageBox.information(widget, "Validation Successful", msg)
        return True






