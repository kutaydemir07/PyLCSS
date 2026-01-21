# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""
Central registry for all System Modeling nodes.
Maps node identifiers (com.pfd.*) to their Python classes.
"""

from pylcss.system_modeling.node_types import (
    InputNode,
    OutputNode,
    IntermediateNode,
    CustomBlockNode
)

# Master mapping of Node ID -> Node Class
SYSTEM_NODE_CLASS_MAPPING = {
    'com.pfd.input': InputNode,
    'com.pfd.output': OutputNode,
    'com.pfd.intermediate': IntermediateNode,
    'com.pfd.custom_block': CustomBlockNode,
}

# Mapping of Class Name -> Node Class
SYSTEM_NODE_NAME_MAPPING = {cls.__name__: cls for cls in SYSTEM_NODE_CLASS_MAPPING.values()}
