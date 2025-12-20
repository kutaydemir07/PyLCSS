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

try:
    import pint
    ureg = pint.UnitRegistry()
except ImportError:
    ureg = None

def validate_graph(widget):
    """
    Validate all system graphs in the widget for common modeling errors.

    Performs comprehensive validation including:
    - Syntax checking of custom block code
    - Cycle detection in graph structure
    - Unconnected input port detection
    - Unit consistency checking between connected ports

    Args:
        widget: Main application widget containing system_manager with systems

    Returns:
        None: Shows dialog with validation results (errors/warnings/success)

    Validation Checks:
        - Syntax errors in custom block Python code
        - Circular dependencies (cycles) in the graph
        - Unconnected input ports on any node
        - Unit mismatches between connected ports (warnings only)

    Dialog Behavior:
        - Shows error dialog if any errors found (blocks compilation)
        - Shows success dialog with warnings if no errors but warnings exist
        - Shows success dialog if validation passes completely
    """
    errors = []
    warnings = []

    # Validate each system in the system manager
    for sys in widget.system_manager.systems:
        graph = sys['graph']
        nodes = graph.all_nodes()

        # Check syntax of custom block code
        has_custom_code = False
        for node in nodes:
            if node.type_.startswith('com.pfd.custom_block'):
                has_custom_code = True
                code = node.get_property('code_content')
                try:
                    compile(code, '<string>', 'exec')
                except SyntaxError as e:
                    errors.append(f"Syntax Error in '{node.name()}' line {e.lineno}: {e.msg}")
        
        if has_custom_code:
            warnings.append(f"Security Warning: System '{sys['name']}' contains custom code blocks. Ensure you trust the source of this model before running.")

        # Build directed graph for cycle detection
        G = nx.DiGraph()
        node_map = {n.id: n for n in nodes}
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
            errors.append(f"System '{sys['name']}': Graph contains cycles.")

        # Check for unconnected inputs
        for node in nodes:
            if node.type_.startswith('com.pfd.custom_block'):
                for port in node.input_ports():
                    if not port.connected_ports():
                        errors.append(f"System '{sys['name']}': Node '{node.name()}' has unconnected input '{port.name()}'.")

        # Check unit consistency with dimensional analysis
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
                        # Use pint for dimensional analysis if available
                        if ureg is not None:
                            try:
                                out_qty = ureg.Quantity(1, source_unit)
                                in_qty = ureg.Quantity(1, target_unit)
                                
                                if out_qty.dimensionality != in_qty.dimensionality:
                                    errors.append(f"Unit Incompatibility in '{sys['name']}': Cannot connect '{source_node.name()}' ({source_unit}) to '{node.name()}' ({target_unit}) - different physical dimensions.")
                                elif source_unit != target_unit:
                                    warnings.append(f"Unit Conversion in '{sys['name']}': '{source_node.name()}' ({source_unit}) connected to '{node.name()}' ({target_unit}). Automatic conversion will be applied.")
                            except Exception:
                                # If unit parsing fails, show warning but allow connection
                                warnings.append(f"Unit Validation Failed in '{sys['name']}': Could not validate '{source_unit}' -> '{target_unit}' connection.")
                        else:
                            # Fallback to simple string comparison if pint not available
                            if source_unit != target_unit:
                                warnings.append(f"Unit Mismatch in '{sys['name']}': Connected '{source_node.name()}' ({source_unit}) to '{node.name()}' ({target_unit}).")

    # Display results
    if errors:
        msg = "Validation Errors:\n\n" + "\n".join(errors)
        if warnings:
            msg += "\n\nWarnings:\n" + "\n".join(warnings)
        QtWidgets.QMessageBox.warning(widget, "Validation Failed", msg)
    else:
        msg = "Graph validation passed!"
        if warnings:
            msg += "\n\nWarnings:\n" + "\n".join(warnings)
        QtWidgets.QMessageBox.information(widget, "Validation Successful", msg)






