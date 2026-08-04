# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""Manage named NodeGraphQt graphs in the system-modeling interface."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from NodeGraphQt import NodeGraph
from PySide6 import QtCore, QtWidgets

from pylcss.system_modeling.types import SystemRecord

from .node_registry import SYSTEM_NODE_CLASS_MAPPING
from .system_node_types import InputNode, OutputNode, apply_system_node_style


class SystemManager(QtCore.QObject):
    """Own the system list, graph widgets, and current selection."""

    system_selected = QtCore.Signal()
    system_added = QtCore.Signal(object)

    def __init__(
        self,
        parent: QtWidgets.QWidget,
        *,
        create_default: bool = True,
    ) -> None:
        super().__init__(parent)
        self.systems: list[SystemRecord] = []
        self.current_graph: NodeGraph | None = None

        self.graph_stack = QtWidgets.QStackedWidget()
        self.systems_list = QtWidgets.QListWidget()
        self.systems_list.itemSelectionChanged.connect(self._on_system_selected)

        self.btn_add_system = QtWidgets.QPushButton("Add")
        self.btn_add_system.clicked.connect(self.add_system)
        self.btn_remove_system = QtWidgets.QPushButton("Remove")
        self.btn_remove_system.clicked.connect(self.remove_system)
        self.btn_rename_system = QtWidgets.QPushButton("Rename")
        self.btn_rename_system.clicked.connect(self.rename_system)

        self.product_name = QtWidgets.QLineEdit("Product")
        self.panel = self._build_panel()
        from pylcss.user_interface.common.theme_manager import current_theme
        self.apply_theme(current_theme())

        self.graph_area = QtWidgets.QWidget()

        graph_layout = QtWidgets.QVBoxLayout(self.graph_area)
        graph_layout.setContentsMargins(0, 0, 0, 0)
        graph_layout.addWidget(self.graph_stack)

        if create_default:
            self._add_system("Default System")

    def _build_panel(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget()
        panel.setObjectName("systemManagerPanel")
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        lbl_prod = QtWidgets.QLabel("Product Name")
        layout.addWidget(lbl_prod)
        layout.addWidget(self.product_name)

        lbl_sys = QtWidgets.QLabel("Systems")
        layout.addWidget(lbl_sys)
        layout.addWidget(self.systems_list)

        button_layout = QtWidgets.QHBoxLayout()
        button_layout.setContentsMargins(0, 0, 0, 0)
        button_layout.setSpacing(4)

        for btn in (
            self.btn_add_system,
            self.btn_remove_system,
            self.btn_rename_system,
        ):
            btn.setSizePolicy(
                QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed
            )
            btn.setMinimumWidth(0)

        button_layout.addWidget(self.btn_add_system, 1)
        button_layout.addWidget(self.btn_remove_system, 1)
        button_layout.addWidget(self.btn_rename_system, 1)
        layout.addLayout(button_layout)

        panel.setMaximumWidth(240)
        return panel

    def apply_theme(self, theme_name: str) -> None:
        """Style panel, inputs, lists, and buttons matching Design Studio."""
        theme_name = str(theme_name).lower()
        from pylcss.user_interface.common.theme_manager import THEMES

        colors = THEMES.get(theme_name, THEMES["dark"])
        qss = f"""
        QWidget#systemManagerPanel {{
            background: {colors["bg_panel"]};
            border-right: 1px solid {colors["border"]};
        }}
        QLabel {{
            color: {colors["text_dim"]};
            font-weight: 600;
            font-size: 11px;
            margin-top: 2px;
        }}
        QLineEdit {{
            background: {colors["bg_input"]};
            border: 1px solid {colors["border"]};
            border-radius: 4px;
            padding: 5px 8px;
            color: {colors["text_main"]};
            font-size: 12px;
        }}
        QLineEdit:focus {{
            border: 1px solid {colors["primary"]};
        }}
        QListWidget {{
            background: {colors["bg_input"]};
            border: 1px solid {colors["border"]};
            border-radius: 4px;
            color: {colors["text_main"]};
            padding: 4px;
            outline: none;
        }}
        QListWidget::item {{
            padding: 6px 8px;
            border-radius: 3px;
        }}
        QListWidget::item:selected {{
            background: {colors["bg_dark"]};
            color: {colors["primary"]};
            font-weight: 600;
        }}
        QListWidget::item:hover:!selected {{
            background: {colors["bg_panel"]};
        }}
        QPushButton {{
            background: {colors["bg_input"]};
            color: {colors["text_main"]};
            border: 1px solid {colors["border"]};
            border-radius: 4px;
            padding: 5px 2px;
            font-weight: 500;
            font-size: 11px;
            text-align: center;
        }}
        QPushButton:hover {{
            background: {colors["border"]};
            border: 1px solid {colors["primary"]};
            color: {colors["text_main"]};
        }}
        QPushButton:pressed {{
            background: {colors["bg_dark"]};
        }}
        """
        self.panel.setStyleSheet(qss)



    def _on_system_selected(self) -> None:
        row = self.systems_list.currentRow()
        if 0 <= row < len(self.systems):
            self.graph_stack.setCurrentIndex(row)
            self.current_graph = self.systems[row]["graph"]
            current_widget = self.graph_stack.currentWidget()
            if current_widget is not None:
                current_widget.setFocus()
        else:
            self.current_graph = None
        self.system_selected.emit()

    def add_system(self) -> None:
        name, accepted = QtWidgets.QInputDialog.getText(
            self.parent(),
            "Add System",
            "System Name:",
        )
        if not accepted:
            return
        try:
            self._add_system(name)
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(
                self.parent(), "Invalid System Name", str(exc)
            )

    def _add_system(self, name: str, graph: NodeGraph | None = None) -> NodeGraph:
        normalized = self._validate_name(name)
        created_graph = graph or self.create_graph()
        self.systems.append({"name": normalized, "graph": created_graph})
        self.systems_list.addItem(normalized)
        self.graph_stack.addWidget(created_graph.widget)
        self.systems_list.setCurrentRow(len(self.systems) - 1)
        self.system_added.emit(created_graph)
        return created_graph

    def create_graph(self) -> NodeGraph:
        """Create a graph with every supported system node registered."""

        graph = NodeGraph()
        for node_class in SYSTEM_NODE_CLASS_MAPPING.values():
            graph.register_node(node_class)
        return graph

    def system_names(self) -> list[str]:
        """Return visible system names in display order."""

        return [str(item["name"]) for item in self.systems]

    def current_system_name(self) -> str | None:
        row = self.systems_list.currentRow()
        if 0 <= row < len(self.systems):
            return str(self.systems[row]["name"])
        return None

    def graph_for_system(self, name: str) -> NodeGraph | None:
        for row, item in enumerate(self.systems):
            if item["name"] == name:
                self.systems_list.setCurrentRow(row)
                return item["graph"]
        return None

    def create_named_system(self, name: str) -> tuple[str, NodeGraph]:
        requested = str(name or "").strip() or "Design Studio Study"
        existing = set(self.system_names())
        unique_name = requested
        suffix = 2
        while unique_name in existing:
            unique_name = f"{requested} ({suffix})"
            suffix += 1
        return unique_name, self._add_system(unique_name)

    def prepare_loaded_graph(self, graph: NodeGraph) -> None:
        """Restore runtime-only port flags and visual state after deserialization."""

        for node in graph.all_nodes():
            apply_system_node_style(node)
            for port in node.output_ports():
                port.model.multi_connection = True
            for port in node.input_ports():
                port.model.multi_connection = False

            variable_name = _property(node, "var_name")
            if isinstance(variable_name, str) and variable_name:
                if isinstance(node, InputNode):
                    _rename_single_port(node, "output", variable_name)
                elif isinstance(node, OutputNode):
                    _rename_single_port(node, "input", variable_name)
            apply_system_node_style(node)

    def replace_systems(
        self,
        systems: Sequence[tuple[str, NodeGraph]],
    ) -> None:
        """Atomically replace all systems with already-deserialized graphs."""

        names = [str(name).strip() for name, _graph in systems]
        if any(not name for name in names):
            raise ValueError("System names must be non-empty.")
        if len(set(names)) != len(names):
            raise ValueError("System names must be unique.")

        old_widgets = [
            self.graph_stack.widget(index) for index in range(self.graph_stack.count())
        ]
        self.systems_list.blockSignals(True)
        try:
            self.systems_list.clear()
            while self.graph_stack.count():
                self.graph_stack.removeWidget(self.graph_stack.widget(0))
            self.systems = []
            self.current_graph = None
            for name, graph in systems:
                self.systems.append({"name": name.strip(), "graph": graph})
                self.systems_list.addItem(name.strip())
                self.graph_stack.addWidget(graph.widget)
                self.system_added.emit(graph)
            if self.systems:
                self.systems_list.setCurrentRow(0)
        finally:
            self.systems_list.blockSignals(False)

        for widget in old_widgets:
            if widget is not None:
                widget.deleteLater()
        self._on_system_selected()

    def remove_system(self) -> None:
        row = self.systems_list.currentRow()
        if not 0 <= row < len(self.systems):
            return
        system_name = self.systems[row]["name"]
        reply = QtWidgets.QMessageBox.question(
            self.parent(),
            "Remove System",
            f"Remove system {system_name!r}?",
            QtWidgets.QMessageBox.StandardButton.Yes
            | QtWidgets.QMessageBox.StandardButton.No,
        )
        if reply != QtWidgets.QMessageBox.StandardButton.Yes:
            return

        graph = self.systems.pop(row)["graph"]
        self.systems_list.takeItem(row)
        self.graph_stack.removeWidget(graph.widget)
        graph.widget.deleteLater()
        if self.systems:
            self.systems_list.setCurrentRow(min(row, len(self.systems) - 1))
        else:
            self.current_graph = None
            self.system_selected.emit()

    def rename_system(self) -> None:
        row = self.systems_list.currentRow()
        if not 0 <= row < len(self.systems):
            return
        current_name = self.systems[row]["name"]
        new_name, accepted = QtWidgets.QInputDialog.getText(
            self.parent(),
            "Rename System",
            "New Name:",
            text=current_name,
        )
        if not accepted or new_name.strip() == current_name:
            return
        try:
            normalized = self._validate_name(new_name, exclude_row=row)
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(
                self.parent(), "Invalid System Name", str(exc)
            )
            return
        self.systems[row]["name"] = normalized
        item = self.systems_list.item(row)
        if item is not None:
            item.setText(normalized)

    def _validate_name(self, name: str, exclude_row: int | None = None) -> str:
        normalized = str(name).strip()
        if not normalized:
            raise ValueError("System name cannot be empty.")
        existing = {
            system["name"]
            for index, system in enumerate(self.systems)
            if index != exclude_row
        }
        if normalized in existing:
            raise ValueError(f"A system named {normalized!r} already exists.")
        return normalized


__all__ = ["SystemManager"]


def _rename_single_port(node: Any, direction: str, name: str) -> None:
    ports = (
        list(node.output_ports()) if direction == "output" else list(node.input_ports())
    )
    if len(ports) != 1 or ports[0].name() == name:
        return

    old_port = ports[0]
    connections = list(old_port.connected_ports())
    for connected in connections:
        old_port.disconnect_from(connected)
    if direction == "output":
        node.delete_output(old_port.name())
        node.add_output(name)
        new_port = node.get_output(name)
        if new_port is not None:
            new_port.model.multi_connection = True
    else:
        node.delete_input(old_port.name())
        node.add_input(name)
        new_port = node.get_input(name)
        if new_port is not None:
            new_port.model.multi_connection = False
    if new_port is None:
        raise RuntimeError(f"Could not restore port {name!r} on {node.name()!r}.")
    for connected in connections:
        new_port.connect_to(connected)


def _property(node: Any, name: str, default: Any = None) -> Any:
    try:
        if hasattr(node, "has_property") and not node.has_property(name):
            return default
        value = node.get_property(name)
    except (AttributeError, KeyError, RuntimeError):
        return default
    return default if value is None else value
