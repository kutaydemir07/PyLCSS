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


def install_node_widget_patches() -> bool:
    """Ensure embedded node widgets have adaptive width and balanced alignment."""
    try:
        from PySide6 import QtGui
        from NodeGraphQt.widgets.node_widgets import NodeCheckBox, NodeLineEdit, NodeComboBox
        from NodeGraphQt.qgraphics.node_base import NodeItem
    except (ImportError, AttributeError):
        return False

    orig_cb_init = getattr(NodeCheckBox, "__init__", None)
    if orig_cb_init and not getattr(orig_cb_init, "_pylcss_patched", False):
        def safe_cb_init(self, parent=None, name="", label="", text="", state=False):
            orig_cb_init(self, parent, name, label, text, state)
            cbox = self.get_custom_widget()
            if cbox:
                fm = QtGui.QFontMetrics(cbox.font())
                req_w = fm.horizontalAdvance(text) + 36
                cbox.setMinimumWidth(req_w)
                if self.widget():
                    self.widget().setMinimumWidth(req_w)
                    self.widget().setMaximumWidth(max(240, req_w + 10))

        safe_cb_init._pylcss_patched = True
        NodeCheckBox.__init__ = safe_cb_init

    orig_combo_init = getattr(NodeComboBox, "__init__", None)
    if orig_combo_init and not getattr(orig_combo_init, "_pylcss_patched", False):
        def safe_combo_init(self, parent=None, name="", label="", items=None):
            orig_combo_init(self, parent, name, label, items)
            combo = self.get_custom_widget()
            if combo:
                title_fm = QtGui.QFontMetrics(QtGui.QFont("Segoe UI", 9, QtGui.QFont.Weight.DemiBold))
                label_w = title_fm.horizontalAdvance(label) + 28
                items_w = max([combo.fontMetrics().horizontalAdvance(it) for it in (items or [])] or [80]) + 44
                req_w = max(150, label_w, items_w)
                combo.setMinimumWidth(req_w)
                if self.widget():
                    self.widget().setMinimumWidth(req_w)
                    self.widget().setMaximumWidth(max(260, req_w + 10))

        safe_combo_init._pylcss_patched = True
        NodeComboBox.__init__ = safe_combo_init

    orig_le_init = getattr(NodeLineEdit, "__init__", None)
    if orig_le_init and not getattr(orig_le_init, "_pylcss_patched", False):
        def safe_le_init(self, parent=None, name="", label="", text="", placeholder_text=""):
            orig_le_init(self, parent, name, label, text, placeholder_text)
            if self.widget():
                self.widget().setMaximumWidth(200)

        safe_le_init._pylcss_patched = True
        NodeLineEdit.__init__ = safe_le_init

    orig_align = getattr(NodeItem, "_align_widgets_horizontal", None)
    if orig_align and not getattr(orig_align, "_pylcss_patched", False):
        def safe_align_widgets_horizontal(self, v_offset):
            if not self._widgets:
                return
            rect = self.boundingRect()
            y = rect.y() + v_offset
            for widget in self._widgets.values():
                if not widget.isVisible():
                    continue
                inner_gb = widget.widget() if hasattr(widget, "widget") else None
                if inner_gb and hasattr(inner_gb, "setTitleAlign"):
                    inner_gb.setTitleAlign("center")
                widget_rect = widget.boundingRect()
                x = rect.center().x() - (widget_rect.width() / 2.0)
                widget.setPos(x, y)
                y += widget_rect.height()

        safe_align_widgets_horizontal._pylcss_patched = True
        NodeItem._align_widgets_horizontal = safe_align_widgets_horizontal

    return True


from pylcss.user_interface.common.node_renderer import install_modern_node_painter

install_nodegraph_patches()
install_node_removal_patch()
install_node_widget_patches()
install_modern_node_painter()

