# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Registry of NodeGraphQt classes used by the system-modeling UI."""

from NodeGraphQt import BaseNode

from pylcss.user_interface.system_modeling.system_node_types import (
    CustomBlockNode,
    InputNode,
    IntermediateNode,
    OutputNode,
)

SYSTEM_NODE_CLASS_MAPPING: dict[str, type[BaseNode]] = {
    "com.pfd.input": InputNode,
    "com.pfd.output": OutputNode,
    "com.pfd.intermediate": IntermediateNode,
    "com.pfd.custom_block": CustomBlockNode,
}

SYSTEM_NODE_NAME_MAPPING: dict[str, type[BaseNode]] = {
    node_class.__name__: node_class for node_class in SYSTEM_NODE_CLASS_MAPPING.values()
}

__all__ = ["SYSTEM_NODE_CLASS_MAPPING", "SYSTEM_NODE_NAME_MAPPING"]
