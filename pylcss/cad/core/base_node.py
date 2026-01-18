# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

import cadquery as cq
from NodeGraphQt import BaseNode

def is_numeric(val):
    return isinstance(val, (int, float))

def is_shape(val):
    """Rudimentary duck-typing to detect CadQuery shapes/workplanes/assemblies."""
    if val is None:
        return False
    # CadQuery workplanes and shapes expose methods like 'val', 'tessellate', 'faces', 'extrude', 'edges'
    # Assemblies have 'toCompound', 'add', etc.
    return any(hasattr(val, attr) for attr in ('val', 'tessellate', 'faces', 'extrude', 'edges', 'toCompound', 'add'))

def resolve_numeric_input(port, fallback):
    """If port is connected to a numeric-producing node, return that number, else fallback."""
    if port and port.connected_ports():
        try:
            # Handle multiple connections? Usually numeric inputs are single.
            # But let's just take the first one.
            node = port.connected_ports()[0].node()
            
            # Use cached result if available (from engine execution)
            res = getattr(node, '_last_result', None)
            if res is None:
                res = node.run()
                
            if is_numeric(res):
                return res
            # Also handle if it returns a dict with a value? No, keep simple.
        except Exception:
            pass
    return fallback

def resolve_shape_input(port):
    """If port is connected to a shape-producing node, return that shape, else None."""
    if port and port.connected_ports():
        try:
            node = port.connected_ports()[0].node()
            
            # Use cached result if available
            res = getattr(node, '_last_result', None)
            if res is None:
                res = node.run()
            
            if is_shape(res):

                return res
            elif isinstance(res, dict) and 'shape' in res:
                return res['shape']
            else:
                # ... (rest of error handling)
                pass
        except Exception as e:
            # ...
            pass
    return None

def resolve_any_input(port):
    """Resolve any input type (dict, list, object) from the first connection."""
    if port and port.connected_ports():
        try:
            node = port.connected_ports()[0].node()
            res = getattr(node, '_last_result', None)
            if res is None:
                res = node.run()
            return res
        except Exception:
            pass
    return None

class CadQueryNode(BaseNode):
    """Base node for all CAD operations."""
    __identifier__ = 'com.cad'
    NODE_NAME = 'Base CAD'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize error state
        self._error_message = None
        self._error_state = False

    def run(self):
        """Override in subclasses to return a CadQuery Workplane or shape."""
        return None

    def set_error(self, message):
        """Set an error state on this node."""
        self._error_message = str(message)
        self._error_state = True
        # Update visual properties if available
        if hasattr(self, 'set_property'):
            try:
                self.set_property('error_state', True)
                self.set_property('error_message', self._error_message)
            except:
                pass

    def clear_error(self):
        """Clear the error state on this node."""
        self._error_message = None
        self._error_state = False
        # Update visual properties if available
        if hasattr(self, 'set_property'):
            try:
                self.set_property('error_state', False)
                self.set_property('error_message', '')
            except:
                pass

    def get_error(self):
        """Get the current error message."""
        return self._error_message

    def has_error(self):
        """Check if this node has an error."""
        return self._error_state

    def get_input_value(self, port_name, prop_name=None):
        """
        Helper to get a numeric value from an input port or a property.
        If the port is connected, use the connected value.
        Otherwise, use the property value.
        """
        port = self.get_input(port_name)
        
        # Check if it's a generic input (like a dict from simulation nodes)
        if port and port.connected_ports():
            val = resolve_any_input(port)
            if val is not None:
                # If we expected a number but got a dict, maybe we need to extract something?
                # For now, assume if it's not numeric, we return it as is (duck typing)
                if is_numeric(val):
                    return val
                # If prop_name is provided, maybe we are looking for a specific key in a dict?
                # But this method is usually for "Value" inputs.
                # Let's just return val.
                return val
        
        fallback = self.get_property(prop_name) if prop_name else None
        return fallback

    def get_input_shape(self, port_name):
        """Helper to get a shape from an input port."""
        port = self.get_input(port_name)
        # ... (debug prints) ...
        if port:
            return resolve_shape_input(port)
        return None
