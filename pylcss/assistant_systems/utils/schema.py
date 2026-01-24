# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""
Node Schema Generator for LLM Context.

Introspects the PyLCSS CAD node library AND Modeling node types to generate
a simplified JSON schema describing available nodes, their inputs, outputs,
and properties. This allows the LLM to understand how to use any node.
"""

import json
import logging
from typing import Dict, List, Any

# Import the centralized CAD node registry
from pylcss.cad.node_library import NODE_CLASS_MAPPING

# Import Modeling environment node registry
from pylcss.system_modeling.node_registry import SYSTEM_NODE_CLASS_MAPPING as MODELING_NODE_MAPPING



def generate_cad_node_schema() -> List[Dict[str, Any]]:
    """
    Generate a JSON-serializable schema of all available CAD nodes.
    
    Returns:
        List of node definitions with:
        - type (identifier)
        - name (display name)
        - description (docstring)
        - inputs (list of names)
        - outputs (list of names)
        - properties (name: {type, default})
    """
    schema = []
    
    for identifier, node_class in NODE_CLASS_MAPPING.items():
        try:
            # Instantiate node to inspect its ports and properties
            # This is safe as __init__ only sets up structure (no heavy logic)
            node_instance = node_class()
            
            # Extract basic info
            node_def = {
                "type": identifier,
                "name": getattr(node_instance, "NODE_NAME", "Unknown Node"),
                "description": (node_class.__doc__ or "").strip().split('\n')[0],
                "inputs": [],
                "outputs": [],
                "properties": {},
                "environment": "CAD"
            }
            
            # Extract inputs/outputs
            # NodeGraphQt stores these in protected dicts usually, but exposes accessors
            # We'll access the internal storage for speed if possible, or public API
            if hasattr(node_instance, '_inputs'):
                node_def["inputs"] = [p.name() for p in node_instance._inputs]
            
            if hasattr(node_instance, '_outputs'):
                node_def["outputs"] = [p.name() for p in node_instance._outputs]
                
            # Extract properties
            # NodeGraphQt nodes have a model with properties
            if hasattr(node_instance, 'model'):
                for name, prop_data in node_instance.model.properties.items():
                    # Skip internal/system properties
                    if name in ('id', 'name', 'color', 'border_color', 'text_color', 'disabled', 'selected'):
                        continue
                        
                    # Get type and default value
                    # prop_data might be the value itself or a dict depending on version
                    # In NodeGraphQt, model.properties is usually simple dict of {name: value}
                    # But we want the WIDGET types if possible.
                    # The nodes call create_property(name, value, widget_type=...)
                    # This stores widget type in model.custom_properties usually?
                    
                    # check custom properties first (where create_property usually goes)
                    pass

                # A more reliable way for PyLCSS nodes:
                # They use self.create_property(). 
                # Let's check `node_instance.model.custom_properties`
                if hasattr(node_instance.model, 'custom_properties'):
                    for prop_name, prop_val in node_instance.model.custom_properties.items():
                         node_def["properties"][prop_name] = {
                             "default": prop_val,
                             "type": _guess_type(prop_val)
                         }
                         
            schema.append(node_def)
            
        except Exception as e:
            logger.warning(f"Failed to introspect node {identifier}: {e}")
            continue
            
    # Sort by identifier for stability
    schema.sort(key=lambda x: x['type'])
    return schema


def generate_modeling_node_schema() -> List[Dict[str, Any]]:
    """
    Generate a JSON-serializable schema of Modeling environment nodes.
    
    Returns:
        List of node definitions
    """
    schema = []
    
    for identifier, node_class in MODELING_NODE_MAPPING.items():
        try:
            node_instance = node_class()
            
            node_def = {
                "type": identifier,
                "name": getattr(node_instance, "NODE_NAME", "Unknown Node"),
                "description": (node_class.__doc__ or "").strip().split('\n')[0],
                "inputs": [],
                "outputs": [],
                "properties": {},
                "environment": "Modeling"
            }
            
            # Extract inputs/outputs
            if hasattr(node_instance, '_inputs'):
                node_def["inputs"] = [p.name() for p in node_instance._inputs]
            
            if hasattr(node_instance, '_outputs'):
                node_def["outputs"] = [p.name() for p in node_instance._outputs]
                
            # Extract properties from model
            if hasattr(node_instance, 'model') and hasattr(node_instance.model, 'custom_properties'):
                for prop_name, prop_val in node_instance.model.custom_properties.items():
                    # Skip internal properties
                    if prop_name.startswith('_') or prop_name in ('surrogate_controls',):
                        continue
                    node_def["properties"][prop_name] = {
                        "default": prop_val,
                        "type": _guess_type(prop_val)
                    }
                    
            schema.append(node_def)
            
        except Exception as e:
            logger.warning(f"Failed to introspect modeling node {identifier}: {e}")
            continue
            
    schema.sort(key=lambda x: x['type'])
    return schema


def generate_node_schema() -> List[Dict[str, Any]]:
    """
    Generate combined schema of all available nodes (CAD + Modeling).
    
    Returns:
        List of all node definitions
    """
    cad_nodes = generate_cad_node_schema()
    modeling_nodes = generate_modeling_node_schema()
    return modeling_nodes + cad_nodes


def _guess_type(value: Any) -> str:
    """Guess property type from value."""
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "list"
    return "any"

def get_simplified_schema_string() -> str:
    """
    Get a minimized string representation of the schema for the LLM prompt.
    Aggregates similar nodes to save token space.
    """
    nodes = generate_node_schema()
    
    # Group by environment
    modeling_nodes = [n for n in nodes if n.get('environment') == 'Modeling']
    cad_nodes = [n for n in nodes if n.get('environment') == 'CAD']
    
    lines = ["## Available Nodes for Graph Building"]
    
    # Modeling section
    lines.append("\n### Modeling Environment Nodes")
    lines.append("Use these in the Modeling tab (tab 0) for system models.")
    for node in modeling_nodes:
        props = ", ".join([f"{k}:{v['type']}" for k, v in node['properties'].items()])
        lines.append(f"- **{node['name']}** (`{node['type']}`): Inputs={node['inputs']}, Outputs={node['outputs']}, Props={{ {props} }}")
    
    # CAD section
    lines.append("\n### CAD Environment Nodes")
    lines.append("Use these in the CAD tab (tab 1) for 3D solid modeling.")
    lines.append("Categories: Primitives, Sketching, Operations, Transforms, Patterns, Assembly, Analysis, Simulation, IO\n")
    
    # Group CAD nodes by category (first part of identifier)
    categories = {}
    for node in cad_nodes:
        # Extract category from type (e.g., 'com.cad.box' -> 'primitives')
        type_parts = node['type'].split('.')
        if len(type_parts) >= 3:
            cat = type_parts[2]
        else:
            cat = 'other'
        
        # Map to readable category
        cat_map = {
            'box': 'Primitives', 'cylinder': 'Primitives', 'sphere': 'Primitives',
            'cone': 'Primitives', 'torus': 'Primitives', 'wedge': 'Primitives', 'pyramid': 'Primitives',
            'sketch': 'Sketching', 'spline': 'Sketching', 'polyline': 'Sketching', 'ellipse': 'Sketching',
            'extrude': 'Operations', 'revolve': 'Operations', 'sweep': 'Operations', 'loft': 'Operations',
            'helix': 'Operations', 'pocket': 'Operations', 'cut_extrude': 'Operations', 'cylinder_cut': 'Operations',
            'fillet': 'Operations', 'chamfer': 'Operations', 'shell': 'Operations', 'boolean': 'Operations',
            'twisted_extrude': 'Operations', 'select_face': 'Operations', 'offset': 'Operations',
            'hole_at_coords': 'Cutting', 'multi_hole': 'Cutting', 'rectangular_cut': 'Cutting',
            'slot_cut': 'Cutting', 'array_holes': 'Cutting',
            'translate': 'Transforms', 'rotate': 'Transforms', 'scale': 'Transforms', 'mirror': 'Transforms',
            'linear_pattern': 'Patterns', 'circular_pattern': 'Patterns', 'pattern': 'Patterns',
            'assembly': 'Assembly',
            'mass_properties': 'Analysis', 'bounding_box': 'Analysis',
            'sim': 'Simulation',
            'number': 'IO', 'variable': 'IO', 'export_step': 'IO', 'export_stl': 'IO',
        }
        category = cat_map.get(cat, 'Other')
        
        if category not in categories:
            categories[category] = []
        categories[category].append(node)
    
    # Output by category
    for cat_name in ['Primitives', 'Sketching', 'Operations', 'Cutting', 'Transforms', 'Patterns', 'Assembly', 'Analysis', 'Simulation', 'IO', 'Other']:
        if cat_name in categories:
            lines.append(f"\n**{cat_name}:**")
            for node in categories[cat_name]:
                props = ", ".join([f"{k}:{v['type']}" for k, v in node['properties'].items()])
                if props:
                    lines.append(f"- {node['name']} (`{node['type']}`): {{ {props} }}")
                else:
                    lines.append(f"- {node['name']} (`{node['type']}`)")
        
    return "\n".join(lines)


def get_full_schema_json() -> str:
    """
    Get the full JSON schema for external tools or documentation.
    """
    nodes = generate_node_schema()
    return json.dumps(nodes, indent=2)


if __name__ == "__main__":
    # Test schema generation
    nodes = generate_node_schema()
    print(f"Generated schema for {len(nodes)} nodes")
    print("\n" + get_simplified_schema_string())

