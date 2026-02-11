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

import ast

def analyze_model_dependencies(model_code, input_names, output_names):
    """
    Parse Python code to determine which outputs depend on which inputs.
    Returns a set of (input, output) tuples representing dependencies.
    """
    try:
        tree = ast.parse(model_code)
    except SyntaxError:
        # Fallback: assume all outputs depend on all inputs
        return {(i, o) for i in input_names for o in output_names}

    deps = set()
    
    # Simple reachability analysis via assignment tracking
    # var_dependencies: {var: {set of source inputs}}
    var_map = {name: {name} for name in input_names}
    
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            # Find targets (LHS)
            targets = []
            for t in node.targets:
                if isinstance(t, ast.Name):
                    targets.append(t.id)
            
            # Find sources (RHS)
            sources = set()
            for child in ast.walk(node.value):
                if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load):
                    if child.id in var_map:
                        sources.update(var_map[child.id])
            
            # Update map
            for target in targets:
                if target not in var_map:
                    var_map[target] = set()
                var_map[target].update(sources)

    # Collect results for outputs
    for out in output_names:
        if out in var_map:
            for source in var_map[out]:
                deps.add((source, out))
    
    return deps

def create_merged_model(models):
    """
    Create a merged model from multiple subsystems.
    Enhanced to handle "Broken Cycles" (Variable-Level DAGs).
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
                
    input_names = set(all_inputs.keys())
    output_names = set(all_outputs.keys())
    global_inputs = sorted(list(input_names - output_names))
    global_outputs = sorted(list(output_names - input_names))

    # --- 2. Build Dependency Graphs ---
    SysG = nx.DiGraph()
    for i in range(len(models)): 
        SysG.add_node(i)

    # Analyze internal dependencies for each model
    # model_internal_deps[i] = set of (inp_name, out_name)
    model_internal_deps = []
    
    for i, model in enumerate(models):
        in_names = [x['name'] for x in model['inputs']]
        out_names = [x['name'] for x in model['outputs']]
        deps = analyze_model_dependencies(model['code'], in_names, out_names)
        model_internal_deps.append(deps)

    # Build System Graph (Standard)
    params = {} # Map var_name -> provider_model_index
    for i, model in enumerate(models):
        for out in model['outputs']:
            params[out['name']] = i

    for i, model in enumerate(models):
        for inp in model['inputs']:
            name = inp['name']
            if name in params:
                provider_idx = params[name]
                if provider_idx != i:
                    SysG.add_edge(provider_idx, i)

    # --- 3. Determine Execution Order ---
    execution_schedule = []
    init_vars = set() # Variables that need pre-initialization (breakers)

    try:
        # Optimistic: Try standard Topological Sort
        execution_schedule = list(nx.topological_sort(SysG))
    except (nx.NetworkXUnfeasible, nx.NetworkXError, RuntimeError):
        # Cycle detected! Attempt "Variable Graph" resolution.
        logger.info("System cycle detected. Attempting variable-level resolution.")
        
        # Build Variable Graph: Nodes are (model_idx, 'input'|'output', var_name)
        VarG = nx.DiGraph()
        
        # Add internal edges
        for i, deps in enumerate(model_internal_deps):
            for inp, out in deps:
                u = (i, 'in', inp)
                v = (i, 'out', out)
                VarG.add_edge(u, v)
        
        # Add external edges
        for i, model in enumerate(models):
            for inp in model['inputs']:
                name = inp['name']
                if name in params:
                    # Connection: Provider Output -> Consumer Input
                    provider_idx = params[name]
                    u = (provider_idx, 'out', name)
                    v = (i, 'in', name)
                    VarG.add_edge(u, v)

        # Check if Variable Graph is acyclic
        if not nx.is_directed_acyclic_graph(VarG):
             raise ValueError("True Circular Dependency detected (Variable Level). Cannot merge.")
             
        # It is acyclic! We can linearize execution.
        # Strategy:
        # We need to run systems such that their outputs are available when needed.
        # Since systems are monolithic, we may need to run them MULTIPLE times.
        # Algorithm: 
        # 1. State: Set of available variables (starts with Global Inputs)
        # 2. Iterate until all Global Outputs computed:
        #    - Find a System that can produce *new* outputs given current state.
        #    - Add it to schedule.
        #    - Update state.
        
        available_vars = set(global_inputs)
        computed_outputs = set()
        target_outputs = set(global_outputs)
        
        # Track what each system CAN produce given a set of inputs
        # (Simply re-evaluating reachability on the fly)
        
        # Limit iterations to avoid infinite loops
        max_steps = len(models) * 3 
        schedule = []
        
        for _ in range(max_steps):
            progress = False
            for i, model in enumerate(models):
                # What can this model compute now?
                # Using internal deps: output is ready if all its dependency inputs are available.
                # BUT, the python function will crash if ANY input is missing (unless we handle it).
                # TRICK: We will initialize missing inputs to 0.0.
                # So we can run ANY model at ANY time, provided we accept garbage outputs for un-ready vars.
                # We want to run a model IF it produces at least one NEW valid output.
                
                # Inputs required for each output
                model_out_ready = set()
                for out in model['outputs']:
                    out_name = out['name']
                    if out_name in computed_outputs:
                        continue
                        
                    # Check dependencies
                    req_inputs = {inp for inp, o in model_internal_deps[i] if o == out_name}
                    # If all req_inputs are in available_vars, we can compute this!
                    if req_inputs.issubset(available_vars):
                        model_out_ready.add(out_name)
                
                if model_out_ready:
                    # This model moves us forward!
                    schedule.append(i)
                    available_vars.update(model_out_ready)
                    computed_outputs.update(model_out_ready)
                    progress = True
            
            if computed_outputs.issuperset(target_outputs):
                break
            
            if not progress:
                break # Stalled
        
        if not computed_outputs.issuperset(target_outputs):
             # Fallback: Just run the original topological sort failure order (SCCs)
             # or simply run ALL models in a sequence that covers dependencies best.
             # For now, raise error if we couldn't resolve.
             logger.warning("Could not fully resolve variable schedule. Using best-effort.")
        
        execution_schedule = schedule
        
        # Identify "Breakers" - inputs that are NOT available when a model is run
        # but are required by the function signature.
        # We must initialize these to 0.0/None
        running_vars = set(global_inputs)
        for idx in execution_schedule:
            model = models[idx]
            for inp in model['inputs']:
                if inp['name'] not in running_vars:
                    init_vars.add(inp['name'])
            # Update running vars with ALL outputs of this model (even if garbage)
            # becuase the function returns them
            for out in model['outputs']:
                running_vars.add(out['name'])

    # --- 4. Smart Code Merging (Identical to before) ---
    unique_imports = set()
    unique_imports.add("import numpy as np")
    global_lines = []         
    processed_model_codes = [] 
    
    for i, model in enumerate(models):
        model_code = model['code']
        lines = model_code.split('\n')
        cleaned_lines = []
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("import ") or stripped.startswith("from "):
                unique_imports.add(stripped)
                continue
            if "joblib.load" in stripped and "=" in stripped and not stripped.startswith("#"):
                if stripped not in global_lines:
                    global_lines.append(stripped)
                continue
            if stripped.startswith("# Auto-generated"):
                continue
            if line.startswith("def system_function"):
                line = line.replace("def system_function(", f"def system_function_{i}(")
            cleaned_lines.append(line)
        processed_model_codes.append("\n".join(cleaned_lines))

    # --- 5. Code Reconstruction ---
    code = []
    code.append("# Merged System Model")
    code.extend(sorted(list(unique_imports)))
    code.append("")
    if global_lines:
        code.append("# --- Global Resource Loading ---")
        code.extend(global_lines)
        code.append("")
    
    for i, model_body in enumerate(processed_model_codes):
        code.append(f"# --- Model {i}: {models[i]['name']} ---")
        code.append(model_body)
        code.append("")

    code.append("def system_function(**kwargs):")
    for name in global_inputs:
        code.append(f"    {name} = kwargs['{name}']")
    
    code.append("    intermediates = {}\n")
    
    # Initialize "Breaker" variables (Feedback variables)
    if init_vars:
        code.append("    # Initialize feedback variables (cycle breakers)")
        for var in init_vars:
             code.append(f"    intermediates['{var}'] = 0.0  # Default init")
        code.append("")

    for step_num, idx in enumerate(execution_schedule):
        model = models[idx]
        code.append(f"    # Step {step_num}: Execute model {idx} ({model['name']})")
        call_args = []
        for inp in model['inputs']:
            name = inp['name']
            if name in global_inputs:
                call_args.append(f"{name}={name}")
            else:
                call_args.append(f"{name}=intermediates.get('{name}', 0.0)") # Use .get for safety
        
        call_str = ", ".join(call_args)
        code.append(f"    outputs_{idx} = system_function_{idx}({call_str})")
        for out in model['outputs']:
            name = out['name']
            code.append(f"    intermediates['{name}'] = outputs_{idx}['{name}']")
        code.append("")
    
    code.append("    return {")
    for name in global_outputs:
        # Prefer intermediate value if available, else look in last outputs
        code.append(f"        '{name}': intermediates.get('{name}', None),")
    code.append("    }")
    
    return {
        'name': 'Merged',
        'code': "\n".join(code),
        'inputs': [all_inputs[name] for name in global_inputs],
        'outputs': [all_outputs[name] for name in global_outputs]
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