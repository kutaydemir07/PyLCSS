# Copyright (c) 2025 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""
Graph to Python code compilation for PyLCSS system models.

This module contains the GraphBuilder class which converts node-based system
graphs into executable Python functions. It handles variable name sanitization,
topological sorting for execution order, and code generation for system models.
"""

import networkx as nx
import keyword
import os
import re
from typing import Any
import logging

logger = logging.getLogger(__name__)

try:
    import pint
    ureg = pint.UnitRegistry()
except ImportError:
    ureg = None


class GraphBuilder:
    """
    Converts NodeGraphQt system graphs into executable Python code.
    """

    def __init__(self, graph: Any) -> None:
        self.graph = graph
        self.used_names = set()

    def _sanitize_name(self, name: str) -> str:
        if not name:
            return "var"
        clean = name.replace(" ", "_")
        clean = "".join(c for c in clean if c.isalnum() or c == '_')
        if clean and clean[0].isdigit():
            clean = "_" + clean
        # Add internal variable names to this list
        reserved = keyword.kwlist + ['inputs', 'outputs', 'results', 'np']
        if clean in reserved:
            clean = f"var_{clean}"
        return clean

    def _get_unique_name(self, base_name):
        clean = self._sanitize_name(base_name)
        if clean not in self.used_names:
            self.used_names.add(clean)
            return clean
        counter = 2
        while True:
            new_name = f"{clean}_{counter}"
            if new_name not in self.used_names:
                self.used_names.add(new_name)
                return new_name
            counter += 1

    def build_system_model(self, function_name="system_function", global_reserved_names=None):
        # Reset name tracking for this build
        self.used_names = set()
        nodes = self.graph.all_nodes()

        # Categorize nodes by type
        input_nodes = [n for n in nodes if n.type_.startswith('com.pfd.input')]
        output_nodes = [n for n in nodes if n.type_.startswith('com.pfd.output')]
        
        # Validate graph structure
        if not input_nodes:
            raise ValueError("No Input Nodes found.")
        if not output_nodes:
            raise ValueError("No Output Nodes found.")

        # 1. Process Input Nodes
        sys_inputs = []
        input_node_var_map = {} 

        for n in input_nodes:
            if n.has_property('input_props'):
                props = n.get_property('input_props')
                raw_name = props.get('var_name', 'x')
                unit = props.get('unit', '-')
                min_val = props.get('min', '0.0')
                max_val = props.get('max', '10.0')
            else:
                raw_name = n.get_property('var_name')
                unit = n.get_property('unit')
                min_val = n.get_property('min')
                max_val = n.get_property('max')

            var_name = self._get_unique_name(raw_name)
            input_node_var_map[n.id] = var_name

            sys_inputs.append({
                'name': var_name,
                'display_name': raw_name,
                'unit': unit,
                'min': min_val,
                'max': max_val,
                'type': 'continuous',
                'granularity': 1.0
            })

        # 2. Process Output Nodes
        sys_outputs = []
        output_node_var_map = {}

        for n in output_nodes:
            if n.has_property('output_props'):
                props = n.get_property('output_props')
                raw_name = props.get('var_name', 'y')
                unit = props.get('unit', '-')
                req_min = props.get('req_min', '-1e9')
                req_max = props.get('req_max', '1e9')
            else:
                raw_name = n.get_property('var_name')
                unit = n.get_property('unit')
                req_min = n.get_property('req_min')
                req_max = n.get_property('req_max')

            var_name = self._get_unique_name(raw_name)
            output_node_var_map[n.id] = var_name

            minimize = n.get_property('minimize')
            maximize = n.get_property('maximize')
            color_tuple = n.get_property('plot_color')
            hex_color = None
            if color_tuple:
                try:
                    # Normalize to tuple for consistent comparison
                    color_tuple = tuple(color_tuple)
                    if color_tuple != (0, 0, 255):
                        hex_color = '#{:02x}{:02x}{:02x}'.format(*color_tuple)
                except:
                    pass

            sys_outputs.append({
                'name': var_name,
                'display_name': raw_name,
                'unit': unit,
                'req_min': req_min,
                'req_max': req_max,
                'minimize': minimize,
                'maximize': maximize,
                'color': hex_color
            })

        # 3. Perform topological sorting
        G = nx.DiGraph()
        node_map = {n.id: n for n in nodes}

        for n in nodes:
            G.add_node(n.id)
            for port in n.output_ports():
                source_unit = n.get_property('unit')
                for connected_port in port.connected_ports():
                    target_node = connected_port.node()
                    G.add_edge(n.id, target_node.id)
                    target_unit = target_node.get_property('unit')
                    if source_unit and target_unit and source_unit != '-' and target_unit != '-':
                        if source_unit != target_unit:
                            logger.warning("Unit mismatch %s -> %s", n.name(), target_node.name())

        try:
            sorted_ids = list(nx.topological_sort(G))
        except nx.NetworkXUnfeasible:
            raise ValueError("Graph has cycles! Please remove feedback loops.")

        sorted_blocks = [node_map[nid] for nid in sorted_ids
                         if node_map[nid].type_.startswith('com.pfd.custom_block')
                         or node_map[nid].type_.startswith('com.pfd.intermediate')]

        # 4. Generate Python code
        code_lines = []
        code_lines.append("import numpy as np")
        code_lines.append("import joblib")
        code_lines.append("import os")
        code_lines.append("import sys") # Added sys for path manipulation if needed
        code_lines.append("from sklearn.preprocessing import StandardScaler")
        code_lines.append("# Auto-generated model code")
        code_lines.append("")
        code_lines.append("# Base directory for relative paths")
        code_lines.append("BASE_DIR = os.path.dirname(os.path.abspath(__file__))")
        code_lines.append("")

        # Global Loading Section
        surrogate_models = {}
        for block in sorted_blocks:
            if not block.type_.startswith('com.pfd.custom_block'):
                continue
            
            # FIX: Correctly check for surrogate usage
            # The property might be a string "true"/"false" or a boolean or int
            use_surrogate_prop = block.get_property('use_surrogate')
            surrogate_path = block.get_property('surrogate_model_path')
            
            # Normalize boolean check
            use_surrogate = str(use_surrogate_prop).lower() in ['true', '1', 'yes', 'on'] or use_surrogate_prop is True or use_surrogate_prop == 1

            
            # If path exists and use_surrogate is explicitly true, use it
            if use_surrogate and surrogate_path:
                model_var = f"surrogate_{block.id.replace('-', '_')}"
                safe_path = surrogate_path.replace('\\', '/')
                
                # Handle relative paths
                if not os.path.isabs(safe_path):
                    # If path is relative, assume it's relative to project root or data folder
                    # For now, we'll try to join it with BASE_DIR if it looks relative
                    code_lines.append(f"try:")
                    code_lines.append(f"    {model_var} = joblib.load(os.path.join(BASE_DIR, r'{safe_path}'))")
                    code_lines.append(f"except FileNotFoundError:")
                    code_lines.append(f"    {model_var} = joblib.load(r'{safe_path}')")
                else:
                    code_lines.append(f"{model_var} = joblib.load(r'{safe_path}')")
                
                surrogate_models[block.id] = model_var
        code_lines.append("")

        # Function Definitions
        for block in sorted_blocks:
            if not block.type_.startswith('com.pfd.custom_block'):
                continue

            # Use node name for function name instead of UUID
            # This makes the generated code much more readable
            base_name = block.name()
            
            # Ensure it's a valid python identifier and unique globally
            clean_base = self._sanitize_name(base_name)
            candidate = clean_base
            counter = 1
            while True:
                name_to_try = candidate if counter == 1 else f"{candidate}_{counter}"
                is_taken_locally = name_to_try in self.used_names
                is_taken_globally = (global_reserved_names is not None) and (name_to_try in global_reserved_names)
                
                if not is_taken_locally and not is_taken_globally:
                    block_name = name_to_try
                    break
                counter += 1
            
            self.used_names.add(block_name)
            if global_reserved_names is not None:
                global_reserved_names.add(block_name)
                
            block.set_property('func_name', block_name)

            use_surrogate_prop = block.get_property('use_surrogate')
            surrogate_path = block.get_property('surrogate_model_path')
            
            # Normalize boolean check
            use_surrogate = str(use_surrogate_prop).lower() in ['true', '1', 'yes', 'on'] or use_surrogate_prop is True or use_surrogate_prop == 1

            # Function definition logic
            if use_surrogate and surrogate_path and block.id in surrogate_models:
                code_lines.append(f"def {block_name}({', '.join([p.name() for p in block.input_ports()])}):")
                code_lines.append(f'    """ [SURROGATE MODE ACTIVE] """')
                model_var = surrogate_models[block.id]
                input_names = [p.name() for p in block.input_ports()]
                if input_names:
                    first_input = input_names[0]
                    code_lines.append(f"    if np.ndim({first_input}) > 0:")
                    code_lines.append(f"        X = np.column_stack([{', '.join(input_names)}])")
                    code_lines.append(f"        y_pred = {model_var}.predict(X)")
                    output_ports = block.output_ports()
                    if len(output_ports) == 1:
                        code_lines.append("        return y_pred.flatten()")
                    else:
                        returns = [f"y_pred[:, {i}]" for i in range(len(output_ports))]
                        code_lines.append(f"        return {', '.join(returns)}")
                    code_lines.append("    else:")
                    input_array = f"np.array([{', '.join(input_names)}])"
                    code_lines.append(f"        X = {input_array}.reshape(1, -1)")
                    code_lines.append(f"        y_pred = {model_var}.predict(X)")
                    code_lines.append("        y_pred = y_pred.flatten()")
                    output_ports = block.output_ports()
                    if len(output_ports) == 1:
                        code_lines.append("    return y_pred[0]")
                    else:
                        returns = [f"y_pred[{i}]" for i in range(len(output_ports))]
                        code_lines.append(f"    return {', '.join(returns)}")
                else:
                    code_lines.append("    return None")
                code_lines.append("")
            else:
                user_code = block.get_widget('code_content').get_value()
                # Improved math replacement logic
                # Map math functions that exist in numpy with same name
                math_to_np_map = {
                    'math.sin': 'np.sin',
                    'math.cos': 'np.cos',
                    'math.tan': 'np.tan',
                    'math.asin': 'np.arcsin',
                    'math.acos': 'np.arccos',
                    'math.atan': 'np.arctan',
                    'math.atan2': 'np.arctan2',
                    'math.sinh': 'np.sinh',
                    'math.cosh': 'np.cosh',
                    'math.tanh': 'np.tanh',
                    'math.exp': 'np.exp',
                    'math.log': 'np.log',
                    'math.log10': 'np.log10',
                    'math.sqrt': 'np.sqrt',
                    'math.ceil': 'np.ceil',
                    'math.floor': 'np.floor',
                    'math.fabs': 'np.abs',
                    'math.pow': 'np.power',  # math.pow -> np.power
                    'math.fmod': 'np.fmod',
                    'math.degrees': 'np.degrees',
                    'math.radians': 'np.radians',
                    'math.pi': 'np.pi',
                    'math.e': 'np.e'
                }
                user_code_np = user_code
                for math_func, np_func in math_to_np_map.items():
                    # Use regex with word boundaries to prevent partial replacements
                    # e.g. prevent replacing "math.sin_val" with "np.sin_val"
                    pattern = r'(?<!\w)' + re.escape(math_func) + r'(?!\w)'
                    user_code_np = re.sub(pattern, np_func, user_code_np)
                
                # Remove import statements
                user_code_np = '\n'.join(line for line in user_code_np.split('\n')
                                         if 'import math' not in line
                                         and 'import numpy' not in line)
                indented_code_lines = user_code_np.split('\n')
                indented_code = "\n    ".join(indented_code_lines)

                input_ports = block.input_ports()
                arg_names = [p.name() for p in input_ports]
                code_lines.append(f"def {block_name}({', '.join(arg_names)}):")
                
                output_ports = block.output_ports()
                ret_names = []
                for p in output_ports:
                    p_name = p.name()
                    ret_names.append(p_name)
                    code_lines.append(f"    {p_name} = np.nan")

                code_lines.append(f"    {indented_code}")

                if ret_names:
                    code_lines.append(f"    return {', '.join(ret_names)}")
                else:
                    code_lines.append("    pass")
                code_lines.append("")

        # Main System Function with Vectorization Support
        sys_args = ", ".join([inp['name'] for inp in sys_inputs])
        
        # 1. Generate Core Logic Function (Internal)
        core_func_name = f"_{function_name}_core"
        code_lines.append(f"def {core_func_name}({sys_args}):")
        
        # Scalar evaluation (original logic) - now inside core function
        port_value_map = {}

        for n in input_nodes:
            var_name = input_node_var_map[n.id]
            outputs = n.output_ports()
            if outputs:
                port_value_map[(n.id, outputs[0].name())] = var_name

        for block in sorted_blocks:
            if block.type_.startswith('com.pfd.intermediate'):
                inputs = block.input_ports()
                if not inputs or not inputs[0].connected_ports():
                    continue
                src_port = inputs[0].connected_ports()[0]
                key = (src_port.node().id, src_port.name())
                val_var = port_value_map.get(key, "None")
                
                raw_name = block.get_property('var_name')
                var_name = self._get_unique_name(raw_name)
                code_lines.append(f"    {var_name} = {val_var}")
                
                outputs = block.output_ports()
                if outputs:
                    port_value_map[(block.id, outputs[0].name())] = var_name
                continue

            # Custom Blocks Call
            call_args = []
            for port in block.input_ports():
                connected = port.connected_ports()
                if not connected:
                    call_args.append("None")
                    continue
                src = connected[0]
                val_var = port_value_map.get((src.node().id, src.name()), "None")
                
                # Automatic Unit Conversion
                if ureg:
                    try:
                        target_unit = block.get_property('unit')
                        source_unit = src.node().get_property('unit')
                        
                        if target_unit and source_unit and target_unit != '-' and source_unit != '-':
                            if target_unit != source_unit:
                                # Calculate conversion factor (slope and intercept)
                                # Use 0 and 1 as test points to determine linear relationship
                                q1 = ureg.Quantity(0, source_unit)
                                q2 = ureg.Quantity(1, source_unit)
                                
                                y1 = q1.to(target_unit).magnitude
                                y2 = q2.to(target_unit).magnitude
                                
                                m = y2 - y1
                                c = y1
                                
                                if abs(c) < 1e-10:
                                    val_var = f"({val_var} * {m})"
                                else:
                                    val_var = f"({val_var} * {m} + {c})"
                    except Exception as e:
                        # Warn user instead of silent failure
                        logger.warning(
                            "Unit conversion failed between %s and %s",
                            source_unit,
                            target_unit,
                            exc_info=True,
                        )
                        pass
                
                call_args.append(val_var)

            func_name = block.get_property('func_name')
            out_ports = block.output_ports()
            out_vars = []
            for p in out_ports:
                v_name = f"v_{block.id.replace('-', '_')}_{p.name()}"
                out_vars.append(v_name)
                port_value_map[(block.id, p.name())] = v_name

            if out_vars:
                code_lines.append(f"    {', '.join(out_vars)} = {func_name}({', '.join(call_args)})")
            else:
                code_lines.append(f"    {func_name}({', '.join(call_args)})")

        # Returns
        return_entries = []
        for n in output_nodes:
            out_name = output_node_var_map[n.id]
            val_var = "None"
            inputs = n.input_ports()
            if inputs and inputs[0].connected_ports():
                src = inputs[0].connected_ports()[0]
                val_var = port_value_map.get((src.node().id, src.name()), "None")
            return_entries.append(f"'{out_name}': {val_var}")

        code_lines.append(f"    return {{{', '.join(return_entries)}}}")
        
        # 2. Generate Wrapper Function with Optimistic Vectorization
        code_lines.append("")
        code_lines.append(f"def {function_name}({sys_args}):")
        code_lines.append(f"    # Try to run vectorized first (Optimistic Vectorization)")
        code_lines.append(f"    try:")
        code_lines.append(f"        return {core_func_name}({sys_args})")
        code_lines.append(f"    except Exception as e:")
        code_lines.append(f"        # Fallback to loop if vectorization fails")
        code_lines.append(
            "        import warnings\n"
            f"        warnings.warn('Vectorization failed for {function_name}; falling back to slow loop. Error: ' + str(e), RuntimeWarning)"
        )
        code_lines.append("        input_vals = [" + ", ".join([inp['name'] for inp in sys_inputs]) + "]")
        code_lines.append("        is_vectorized = any(hasattr(val, '__len__') and len(val) > 1 for val in input_vals)")
        code_lines.append("        ")
        code_lines.append("        if is_vectorized:")
        code_lines.append("            N = max(len(val) for val in input_vals if hasattr(val, '__len__'))")
        code_lines.append("            results = {}")
        code_lines.append("            for i in range(N):")
        for inp in sys_inputs:
            code_lines.append(f"                {inp['name']}_i = {inp['name']}[i] if hasattr({inp['name']}, '__getitem__') else {inp['name']}")
        code_lines.append(f"                point_result = {core_func_name}(" + ", ".join([f"{inp['name']}_i" for inp in sys_inputs]) + ")")
        code_lines.append("                for key, val in point_result.items():")
        code_lines.append("                    if key not in results:")
        code_lines.append("                        results[key] = []")
        code_lines.append("                    results[key].append(val)")
        code_lines.append("            import numpy as np")
        code_lines.append("            for key in results:")
        code_lines.append("                results[key] = np.array(results[key])")
        code_lines.append("            return results")
        code_lines.append("        else:")
        code_lines.append("            # Re-raise exception if not vectorized (real error)")
        code_lines.append("            raise")

        return "\n".join(code_lines), sys_inputs, sys_outputs

    def build_spy_model(self, nodes, input_nodes, output_nodes, target_node_id, function_name="spy_model"):
        """
        Build a spy model that captures inputs and outputs for a specific target node.
        Used for surrogate model training data generation.
        
        Args:
            nodes: All nodes in the graph
            input_nodes: Input nodes (Global System Inputs)
            output_nodes: Output nodes (not used for spy)
            target_node_id: ID of the target node to spy on
            function_name: Name for the generated function
            
        Returns:
            tuple: (spy_code, spy_inputs, spy_outputs)
        """
        # 1. Identify Target and Dependencies
        target_node = None
        for n in nodes:
            if n.id == target_node_id:
                target_node = n
                break
        if not target_node:
            raise ValueError(f"Target node {target_node_id} not found")
        
        # Metadata for the trainer (User Interface needs these names)
        spy_inputs = [{'name': p.name()} for p in target_node.input_ports()]
        spy_outputs = [{'name': p.name()} for p in target_node.output_ports()]
        
        # Build the graph structure 
        node_map = {n.id: n for n in nodes}
        
        # Create dependency graph to find ancestors
        G = nx.DiGraph()
        for n in nodes:
            G.add_node(n.id)
            for port in n.input_ports():
                for connected_port in port.connected_ports():
                    src_node = connected_port.node()
                    G.add_edge(src_node.id, n.id)
        
        # Find all nodes that the target depends on (Ancestors)
        try:
            predecessors = set(nx.ancestors(G, target_node.id))
            predecessors.add(target_node.id)
        except nx.NetworkXError:
            predecessors = {target_node.id}
        
        # Topological sort ensures we execute in the correct order
        try:
            sorted_ids = list(nx.topological_sort(G.subgraph(predecessors)))
        except nx.NetworkXError:
            raise ValueError("Graph has cycles! Please remove feedback loops.")
        
        # Filter to executable blocks (ignore input/output nodes in this list)
        sorted_blocks = [node_map[nid] for nid in sorted_ids
                         if node_map[nid].type_.startswith('com.pfd.custom_block')
                         or node_map[nid].type_.startswith('com.pfd.intermediate')]
        
        # 2. Start Generating Code
        code_lines = []
        code_lines.append("import numpy as np")
        code_lines.append("import joblib")
        code_lines.append("from sklearn.preprocessing import StandardScaler")
        code_lines.append("")
        
        # 3. Generate individual functions for each block
        surrogate_models = {}
        for block in sorted_blocks:
            if not block.type_.startswith('com.pfd.custom_block'):
                continue
            
            block_name = f"block_{block.id.replace('-', '_')}"
            # block.set_property('func_name', block_name) # Removed to prevent node renaming
            
            use_surrogate = block.get_property('use_surrogate') or False
            surrogate_path = block.get_property('surrogate_model_path') or ''
            
            # Pre-load surrogate models globally if needed
            if use_surrogate and surrogate_path:
                model_var = f"surrogate_{block.id.replace('-', '_')}"
                safe_path = surrogate_path.replace('\\', '/')
                code_lines.append(f"{model_var} = joblib.load(r'{safe_path}')")
                surrogate_models[block.id] = model_var
            
            # --- Type A: Regular Python Block ---
            if not use_surrogate:
                input_ports = block.input_ports()
                output_ports = block.output_ports()
                input_names = [p.name() for p in input_ports]
                
                code_lines.append(f"def {block_name}({', '.join(input_names)}):")
                
                # Get the block's code safely
                block_code = block.get_property('code_content') 
                if not block_code: 
                    block_code = block.get_property('code') or ''

                if block_code:
                    # Indent the user's code
                    indented_code = '\n'.join(['    ' + line for line in block_code.split('\n') if line.strip()])
                    code_lines.append(indented_code)
                    
                    # Return dict of outputs
                    returns = [p.name() for p in output_ports]
                    if returns:
                        return_dict = ', '.join([f"'{r}': {r}" for r in returns])
                        code_lines.append(f"    return {{{return_dict}}}")
                    else:
                        code_lines.append("    return {}")
                else:
                    code_lines.append("    return {}")
                code_lines.append("")

            # --- Type B: Existing Surrogate Block ---
            else:
                input_ports = block.input_ports()
                output_ports = block.output_ports()
                input_names = [p.name() for p in input_ports]
                
                code_lines.append(f"def {block_name}({', '.join(input_names)}):")
                
                # Use the loaded model variable
                model_var = surrogate_models[block.id]
                
                # Prepare inputs for prediction
                if input_names:
                    # Handle scalar inputs by wrapping them in a list
                    input_list = f"[{', '.join(input_names)}]"
                    code_lines.append(f"    # Ensure inputs are treated as a single sample")
                    code_lines.append(f"    X = np.array({input_list}).reshape(1, -1)")
                    code_lines.append(f"    y_pred = {model_var}.predict(X)")
                    code_lines.append("    y_pred = y_pred.flatten()")
                    
                    # Return appropriate number of outputs
                    if len(output_ports) == 1:
                        code_lines.append("    return {'" + output_ports[0].name() + "': y_pred[0]}")
                    else:
                        returns = [f"'{p.name()}': y_pred[{i}]" for i, p in enumerate(output_ports)]
                        code_lines.append(f"    return {{{', '.join(returns)}}}")
                else:
                    code_lines.append("    return {}")
                code_lines.append("")

        # 4. Generate the Main Spy Function
        # We accept *args because the trainer sends global system inputs
        code_lines.append(f"def {function_name}(*args):")
        code_lines.append("    outputs = {}")
        
        port_value_map = {}
        
        # A. Map Trainer Args to Global Input Nodes
        for i, node in enumerate(input_nodes):
            # InputNode usually has one output port 'x'
            if node.output_ports():
                port_name = node.output_ports()[0].name()
                var_name = f"sys_in_{i}"
                code_lines.append(f"    {var_name} = args[{i}]")
                port_value_map[(node.id, port_name)] = var_name
        
        # B. Execute the graph logic
        captured_target_inputs = [] # This will hold the variables entering the target node
        
        # [Corrected Execution Loop]
        for block in sorted_blocks:
            # 1. Handle Intermediate Nodes (Pass-through logic)
            if block.type_.startswith('com.pfd.intermediate'):
                # Get input value
                input_ports = block.input_ports()
                val_var = "None"
                if input_ports and input_ports[0].connected_ports():
                    src = input_ports[0].connected_ports()[0]
                    src_key = (src.node().id, src.name())
                    val_var = port_value_map.get(src_key, "None")
                
                # Assign to output
                outputs = block.output_ports()
                if outputs:
                    # Intermediate nodes usually map 1 input to 1 output
                    port_value_map[(block.id, outputs[0].name())] = val_var
                continue # Skip function call generation

            # 2. Handle Custom Blocks
            # func_name = block.get_property('func_name')
            # Use the same naming convention as in the definition phase to ensure consistency
            # and avoid relying on the node property which might be stale or different
            func_name = f"block_{block.id.replace('-', '_')}"
            input_vars = []
            
            # FIXED: Resolve inputs by looking up the SOURCE node (Producer)
            for port in block.input_ports():
                connected = port.connected_ports()
                if connected:
                    src = connected[0]
                    src_key = (src.node().id, src.name())
                    input_vars.append(port_value_map.get(src_key, "None"))
                else:
                    input_vars.append("None")
            
            call_args = ", ".join(input_vars)

            # Special Handling: Is this our Target Node?
            if block.id == target_node.id:
                # CAPTURE STEP: These are the inputs we want to train on (X)
                captured_target_inputs = input_vars 
                
                # Run the block one last time to get the targets (y)
                if input_vars:
                    code_lines.append(f"    outputs = {func_name}({call_args})")
                else:
                    code_lines.append(f"    outputs = {func_name}()")
                
                # Stop execution here
                break 
            else:
                # Intermediate block execution
                if input_vars:
                    code_lines.append(f"    {func_name}_result = {func_name}({call_args})")
                else:
                    code_lines.append(f"    {func_name}_result = {func_name}()")
                
                # Store outputs in map for next blocks to use
                for port in block.output_ports():
                    port_value_map[(block.id, port.name())] = f"{func_name}_result['{port.name()}']"
        
        # 5. Return Captured Inputs (X) and Calculated Outputs (y)
        # inputs_dict: maps 'input_0' -> captured value
        inputs_dict_items = []
        for i, val_var in enumerate(captured_target_inputs):
            inputs_dict_items.append(f"'input_{i}': {val_var}")
        inputs_dict = ", ".join(inputs_dict_items)
        
        # outputs_dict: maps 'output_0' -> result from target node
        outputs_dict_items = []
        for i, out_spec in enumerate(spy_outputs):
            out_name = out_spec['name']
            outputs_dict_items.append(f"'output_{i}': outputs['{out_name}']")
        outputs_dict = ", ".join(outputs_dict_items)
        
        code_lines.append(f"    return {{{inputs_dict}}}, {{{outputs_dict}}}")
        
        # 6. Finalize
        spy_code = "\n".join(code_lines)
        return spy_code, spy_inputs, spy_outputs






