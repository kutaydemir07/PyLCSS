# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""
Model merging utilities for PyLCSS.

This module provides functions to merge multiple system models into a single
unified model, handling variable connections and dependency resolution.
"""

import logging
import networkx as nx
from PySide6 import QtWidgets
from PySide6.QtCore import Qt  # <--- Added to fix crash

logger = logging.getLogger(__name__)

def create_merged_model(models):
    """
    Create a merged model from multiple subsystems.

    Detects dependencies based on shared variable names and generates
    unified code that executes models in the correct order.

    Args:
        models: List of model dictionaries with 'inputs', 'outputs', 'code', 'name'

    Returns:
        dict: Merged model with unified inputs/outputs and combined code

    Raises:
        ValueError: If circular dependencies exist or no global outputs found
    """
    # --- 1. Variable Collection ---
    all_inputs = {}
    all_outputs = {}
    
    for model in models:
        for inp in model['inputs']:
            if inp['name'] not in all_inputs:
                all_inputs[inp['name']] = inp
        for out in model['outputs']:
            if out['name'] not in all_outputs:
                all_outputs[out['name']] = out
    
    # --- 2. Build Dependency Graph ---
    G = nx.DiGraph()
    for i in range(len(models)):
        G.add_node(i)
    
    for i, model_a in enumerate(models):
        for inp in model_a['inputs']:
            inp_name = inp['name']
            inp_unit = inp.get('unit', '-')
            for j, model_b in enumerate(models):
                if i != j:
                    for out in model_b['outputs']:
                        if inp_name == out['name']:
                            out_unit = out.get('unit', '-')
                            # Check for unit mismatch
                            if inp_unit and out_unit:
                                if inp_unit != out_unit:
                                    logger.warning(
                                        "Unit mismatch when merging: %s outputs '%s' [%s] -> %s inputs '%s' [%s]",
                                        model_b['name'], inp_name, out_unit,
                                        model_a['name'], inp_name, inp_unit
                                    )
                            G.add_edge(j, i)  # b provides input to a
    
    # --- 3. Topological Sort ---
    try:
        order = list(nx.topological_sort(G))
    except nx.NetworkXError:
        raise ValueError("Circular dependency detected in models")
    
    # Identify global inputs and outputs
    input_names = set(all_inputs.keys())
    output_names = set(all_outputs.keys())
    
    global_inputs = sorted(list(input_names - output_names))
    global_outputs = sorted(list(output_names - input_names))
    
    if not global_outputs:
        raise ValueError("No global outputs found (all outputs are used as inputs)")
    
    # --- 4. Smart Code Merging (Single Pass) ---
    unique_imports = set()
    unique_imports.add("import numpy as np")
    
    global_lines = []         # e.g., surrogate_x = joblib.load(...)
    processed_model_codes = [] # Stores the cleaned body of functions for each model
    
    for i, model in enumerate(models):
        model_code = model['code']
        lines = model_code.split('\n')
        
        cleaned_lines = []
        
        for line in lines:
            stripped = line.strip()
            
            # A. Capture Imports (Move to top)
            if stripped.startswith("import ") or stripped.startswith("from "):
                unique_imports.add(stripped)
                continue
                
            # B. Capture Surrogate/Global Loading (Move to top)
            if "joblib.load" in stripped and "=" in stripped and not stripped.startswith("#"):
                if stripped not in global_lines:
                    global_lines.append(stripped)
                continue
                
            # C. Filter Comments
            if stripped.startswith("# Auto-generated"):
                continue
                
            # D. Function Renaming
            # Rename 'def system_function(' to 'def system_function_{i}('
            if line.startswith("def system_function"):
                # Use strict replacement to avoid accidental partial matches
                line = line.replace("def system_function(", f"def system_function_{i}(")
            
            cleaned_lines.append(line)
            
        processed_model_codes.append("\n".join(cleaned_lines))

    # --- 5. Code Reconstruction ---
    code = []
    
    # A. Header & Imports
    code.append("# Merged System Model")
    code.extend(sorted(list(unique_imports)))
    code.append("")
    
    # B. Global Surrogate Loads
    if global_lines:
        code.append("# --- Global Resource Loading ---")
        code.extend(global_lines)
        code.append("")
    
    # C. Sub-Model Functions
    for i, model_body in enumerate(processed_model_codes):
        code.append(f"# --- Model {i}: {models[i]['name']} ---")
        code.append(model_body)
        code.append("") # Spacing
    
    # D. Main Orchestrator Function
    code.append("def system_function(**kwargs):")
    
    # Unpack global inputs
    for name in global_inputs:
        code.append(f"    {name} = kwargs['{name}']")
    
    code.append("    intermediates = {}\n")
    
    # Execution Logic (in dependency order)
    for idx in order:
        model = models[idx]
        code.append(f"    # Execute model {idx} ({model['name']})")
        
        # Build call arguments
        call_args = []
        for inp in model['inputs']:
            name = inp['name']
            if name in global_inputs:
                call_args.append(f"{name}={name}")
            else:
                call_args.append(f"{name}=intermediates['{name}']")
        
        call_str = ", ".join(call_args)
        
        # Call the renamed function
        code.append(f"    outputs_{idx} = system_function_{idx}({call_str})")
        
        # Store outputs in intermediates
        for out in model['outputs']:
            name = out['name']
            code.append(f"    intermediates['{name}'] = outputs_{idx}['{name}']")
        code.append("")
    
    # Return Global Outputs
    code.append("    return {")
    for name in global_outputs:
        code.append(f"        '{name}': intermediates['{name}'],")
    code.append("    }")
    
    # Final String Assembly
    final_code_str = "\n".join(code)
    
    # Create merged model dict
    merged_inputs = [all_inputs[name] for name in global_inputs]
    merged_outputs = [all_outputs[name] for name in global_outputs]
    
    return {
        'name': 'Merged',
        'code': final_code_str,
        'inputs': merged_inputs,
        'outputs': merged_outputs
    }

def validate_merge_connections(models, parent=None):
    """
    Show a dialog with proposed variable connections for merging.
    """
    # Collect all variables
    all_inputs = {}
    all_outputs = {}
    
    for model in models:
        for inp in model['inputs']:
            name = inp['name']
            if name not in all_inputs:
                all_inputs[name] = inp
        for out in model['outputs']:
            name = out['name']
            if name not in all_outputs:
                all_outputs[name] = out
    
    # Find all variable names across all models
    all_variable_names = set()
    for model in models:
        for inp in model['inputs']:
            all_variable_names.add(inp['name'])
        for out in model['outputs']:
            all_variable_names.add(out['name'])
    
    # Check unit consistency for ALL variables (not just connections)
    unit_warnings = []
    for var_name in all_variable_names:
        var_occurrences = []
        for model in models:
            for inp in model['inputs']:
                if inp['name'] == var_name:
                    var_occurrences.append({
                        'model': model['name'],
                        'type': 'Input',
                        'unit': inp.get('unit', '-')
                    })
            for out in model['outputs']:
                if out['name'] == var_name:
                    var_occurrences.append({
                        'model': model['name'],
                        'type': 'Output',
                        'unit': out.get('unit', '-')
                    })
        
        # Check if variable appears in multiple places with different units
        if len(var_occurrences) > 1:
            units = [occ['unit'] for occ in var_occurrences if occ['unit']]
            if len(set(units)) > 1:
                unit_warnings.append({
                    'variable': var_name,
                    'occurrences': var_occurrences
                })
                logger.warning(
                    "Variable '%s' has inconsistent units across models: %s",
                    var_name,
                    ', '.join([f"{occ['model']} ({occ['type']}): {occ['unit']}" for occ in var_occurrences])
                )
    
    # Find connections (output -> input)
    connections = []
    for name in all_inputs:
        if name in all_outputs:
            providers = []
            consumers = []
            provider_units = []
            consumer_units = []
            unit_mismatch = False
            
            for i, model in enumerate(models):
                for out in model['outputs']:
                    if out['name'] == name:
                        out_unit = out.get('unit', '-')
                        providers.append(f"{model['name']} (Output)")
                        provider_units.append(out_unit)
                for inp in model['inputs']:
                    if inp['name'] == name:
                        inp_unit = inp.get('unit', '-')
                        consumers.append(f"{model['name']} (Input)")
                        consumer_units.append(inp_unit)
            
            # Check for unit mismatches
            all_units = provider_units + consumer_units
            non_empty_units = [u for u in all_units if u]
            if len(set(non_empty_units)) > 1:
                unit_mismatch = True
            
            connections.append({
                'variable': name,
                'providers': providers,
                'consumers': consumers,
                'provider_units': provider_units,
                'consumer_units': consumer_units,
                'unit_mismatch': unit_mismatch
            })
    
    input_names = set(all_inputs.keys())
    output_names = set(all_outputs.keys())
    global_inputs = sorted(list(input_names - output_names))
    global_outputs = sorted(list(output_names - input_names))
    
    # --- GUI Dialog ---
    dialog = QtWidgets.QDialog(parent)
    dialog.setWindowTitle("Merge Validation")
    dialog.resize(600, 400)
    
    layout = QtWidgets.QVBoxLayout(dialog)
    
    info_label = QtWidgets.QLabel(
        "The following variable connections will be made when merging models.\n"
        "Variables with the same name are automatically connected."
    )
    info_label.setWordWrap(True)
    layout.addWidget(info_label)
    
    # Scroll Area
    scroll = QtWidgets.QScrollArea()
    scroll.setWidgetResizable(True)
    scroll_content = QtWidgets.QWidget()
    scroll_layout = QtWidgets.QVBoxLayout(scroll_content)
    
    # Show unit warnings if any
    if unit_warnings:
        warnings_group = QtWidgets.QGroupBox("⚠ Unit Consistency Warnings")
        warnings_group.setStyleSheet("QGroupBox { color: #856404; font-weight: bold; }")
        warnings_layout = QtWidgets.QVBoxLayout(warnings_group)
        
        for warn in unit_warnings:
            warn_text = f"<b>Variable '{warn['variable']}'</b> has inconsistent units:<br>"
            for occ in warn['occurrences']:
                warn_text += f"&nbsp;&nbsp;• {occ['model']} ({occ['type']}): [{occ['unit']}]<br>"
            
            warn_label = QtWidgets.QLabel(warn_text)
            warn_label.setStyleSheet("background-color: #fff3cd; padding: 8px; border: 1px solid #ffc107; border-radius: 3px; color: #000000;")
            warn_label.setTextFormat(Qt.RichText)
            warn_label.setWordWrap(True)
            warnings_layout.addWidget(warn_label)
        
        scroll_layout.addWidget(warnings_group)
    
    connections_group = QtWidgets.QGroupBox("Variable Connections")
    connections_group_layout = QtWidgets.QVBoxLayout(connections_group)
    
    if connections:
        for conn in connections:
            # Use simple HTML for bolding
            conn_text = f"<b>Variable '{conn['variable']}':</b><br>"
            
            # Show providers with units
            provider_strs = []
            for p, u in zip(conn['providers'], conn['provider_units']):
                provider_strs.append(f"{p} [{u}]")
            conn_text += f"&nbsp;&nbsp;Provided by: {', '.join(provider_strs)}<br>"
            
            # Show consumers with units
            consumer_strs = []
            for c, u in zip(conn['consumers'], conn['consumer_units']):
                consumer_strs.append(f"{c} [{u}]")
            conn_text += f"&nbsp;&nbsp;Consumed by: {', '.join(consumer_strs)}"
            
            # Add warning if unit mismatch
            if conn['unit_mismatch']:
                conn_text += "<br><span style='color: #856404; font-weight: bold;'>⚠ WARNING: Unit mismatch detected!</span>"
            
            conn_label = QtWidgets.QLabel(conn_text)
            # Highlight mismatches with different background
            if conn['unit_mismatch']:
                conn_label.setStyleSheet("border-bottom: 1px solid #eee; padding: 5px; background-color: #fff3cd; color: #000000;")
            else:
                conn_label.setStyleSheet("border-bottom: 1px solid #eee; padding: 5px;")
            # Safely set TextFormat using imported Qt module
            conn_label.setTextFormat(Qt.RichText)
            
            connections_group_layout.addWidget(conn_label)
    else:
        no_conn_label = QtWidgets.QLabel("No variable connections detected.")
        connections_group_layout.addWidget(no_conn_label)
    
    scroll_layout.addWidget(connections_group)
    scroll_content.setLayout(scroll_layout)
    scroll.setWidget(scroll_content)
    layout.addWidget(scroll)
    
    # Global IO Summary
    globals_group = QtWidgets.QGroupBox("Global Interface")
    globals_layout = QtWidgets.QFormLayout(globals_group)
    
    globals_layout.addRow("Global Inputs:", QtWidgets.QLabel(", ".join(global_inputs) if global_inputs else "None"))
    globals_layout.addRow("Global Outputs:", QtWidgets.QLabel(", ".join(global_outputs) if global_outputs else "None"))
    
    layout.addWidget(globals_group)
    
    buttons = QtWidgets.QDialogButtonBox(
        QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
    )
    buttons.accepted.connect(dialog.accept)
    buttons.rejected.connect(dialog.reject)
    layout.addWidget(buttons)
    
    return dialog.exec() == QtWidgets.QDialog.Accepted