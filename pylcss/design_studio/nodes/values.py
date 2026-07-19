# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

import math

from pylcss.design_studio.core.base_node import CadQueryNode


def _finite_value_node_result(node):
    """Resolve the visible text field without silently masking invalid text."""
    raw = node.get_property('value_input')
    if raw is None or str(raw).strip() == '':
        raw = node.get_property('value')
    try:
        value = float(raw)
    except (TypeError, ValueError):
        node.set_error(f"Value must be numeric; received {raw!r}.")
        return None
    if not math.isfinite(value):
        node.set_error("Value must be finite (not NaN or infinity).")
        return None
    node.clear_error()
    return value

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
        # When non-empty, this number becomes a named input on the cad runtime
        # API (cad.fea("file.cad", <exposed_name>=value, ...)).
        self.create_property('exposed_name', '', widget_type='text')

    def run(self):
        return _finite_value_node_result(self)

class VariableNode(CadQueryNode):
    """Defines a named variable that can be used elsewhere."""
    __identifier__ = 'com.cad.variable'
    NODE_NAME = 'Variable'

    def __init__(self):
        super(VariableNode, self).__init__()
        self.add_output('value', color=(180, 180, 0))

        # Add input fields directly on the node
        # Note: add_text_input already creates the property, so no need for create_property
        self.add_text_input('variable_name', 'Name', text='var1')
        self.add_text_input('value_input', 'Value', text='0.0')

        self.create_property('value', 0.0, widget_type='float')
        # When non-empty, this variable becomes a named input on the cad runtime
        # API (cad.fea("file.cad", <exposed_name>=value, ...)). Defaults to the
        # ``variable_name`` value when left blank.
        self.create_property('exposed_name', '', widget_type='text')

    def run(self):
        return _finite_value_node_result(self)
