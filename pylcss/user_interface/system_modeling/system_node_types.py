# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE.

"""Compatibility façade for system-modeling editors and node types."""

from .editor import (
    CodeEditor,
    CodeEditorDialog,
    LineNumberArea,
    PortListWidget,
    PortManagerDialog,
    PythonHighlighter,
)
from .nodes import (
    CustomBlockNode,
    InputNode,
    IntermediateNode,
    OutputNode,
    SimulationFunctionNode,
    apply_system_node_style,
)

__all__ = [
    "CodeEditor",
    "CodeEditorDialog",
    "CustomBlockNode",
    "InputNode",
    "IntermediateNode",
    "LineNumberArea",
    "OutputNode",
    "SimulationFunctionNode",
    "PortListWidget",
    "PortManagerDialog",
    "PythonHighlighter",
    "apply_system_node_style",
]

for _node_class in (
    CustomBlockNode,
    InputNode,
    IntermediateNode,
    OutputNode,
    SimulationFunctionNode,
):
    _node_class.__module__ = __name__
del _node_class
