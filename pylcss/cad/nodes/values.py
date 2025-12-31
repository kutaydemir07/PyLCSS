# Copyright (c) 2025 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

from pylcss.cad.core.base_node import CadQueryNode

class NumberNode(CadQueryNode):
    """Provides a numeric value."""
    __identifier__ = 'com.cad.number'
    NODE_NAME = 'Number'

    def __init__(self):
        super(NumberNode, self).__init__()
        self.add_output('value', color=(180, 180, 0))
        
        # Add input field directly on the node for easy access
        self.add_text_input('value_input', 'Value', text='10.0')
        
        # Keep property for backward compatibility and property panel access
        self.create_property('value', 10.0, widget_type='float')

    def run(self):
        # Try to get value from the text input on the node first
        try:
            val_str = self.get_property('value_input')
            if val_str is not None:
                return float(val_str)
        except Exception:
            pass
            
        # Fallback to the standard property
        return self.get_property('value')

class VariableNode(CadQueryNode):
    """Defines a named variable that can be used elsewhere."""
    __identifier__ = 'com.cad.variable'
    NODE_NAME = 'Variable'

    def __init__(self):
        super(VariableNode, self).__init__()
        self.add_output('value', color=(180, 180, 0))
        
        # Add input fields directly on the node
        self.add_text_input('variable_name', 'Name', text='var1')
        self.add_text_input('value_input', 'Value', text='0.0')
        
        self.create_property('variable_name', 'var1', widget_type='string')
        self.create_property('value', 0.0, widget_type='float')

    def run(self):
        # Try to get value from the text input on the node first
        try:
            val_str = self.get_property('value_input')
            if val_str is not None:
                return float(val_str)
        except Exception:
            pass
            
        return self.get_property('value')
