# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE.
"""Narrow compatibility patches for Qt and NodeGraphQt."""

from __future__ import annotations

import os

from PySide6.QtWidgets import QTableWidgetItem

__all__ = [
    "NumericTableWidgetItem",
    "install_node_removal_patch",
    "install_nodegraph_patches",
]

os.environ.setdefault("QT_API", "pyside6")


def install_nodegraph_patches() -> bool:
    """Install the idempotent guard for deleted ports during pipe drawing."""
    try:
        from NodeGraphQt.qgraphics.pipe import PipeItem
    except (ImportError, AttributeError):
        return False

    original = getattr(PipeItem, "_draw_path_horizontal", None)
    if original is None:
        return False
    if getattr(original, "_pylcss_safe_pipe_patch", False):
        return True

    def safe_draw_horizontal(self, start_port, pos1, pos2, path):
        if start_port is None or getattr(start_port, "node", None) is None:
            return None
        return original(self, start_port, pos1, pos2, path)

    safe_draw_horizontal._pylcss_safe_pipe_patch = True
    PipeItem._draw_path_horizontal = safe_draw_horizontal
    return True


def install_node_removal_patch() -> bool:
    """Keep node views alive for the lifetime of NodeGraphQt undo commands."""
    try:
        from NodeGraphQt.base.commands import NodesRemovedCmd
    except (ImportError, AttributeError):
        return False

    original = getattr(NodesRemovedCmd, "__init__", None)
    if original is None:
        return False
    if getattr(original, "_pylcss_node_removal_patch", False):
        return True

    def safe_init(self, graph, nodes, emit_signal=True):
        original(self, graph, nodes, emit_signal)
        self.node_views = [node.view for node in nodes]

    safe_init._pylcss_node_removal_patch = True
    NodesRemovedCmd.__init__ = safe_init
    return True


class NumericTableWidgetItem(QTableWidgetItem):
    """Table item that sorts numeric values numerically."""

    def __lt__(self, other: QTableWidgetItem) -> bool:
        try:
            return float(self.text()) < float(other.text())
        except (TypeError, ValueError):
            return super().__lt__(other)


from pylcss.user_interface.common.node_renderer import install_modern_node_painter

install_nodegraph_patches()
install_node_removal_patch()
install_modern_node_painter()

