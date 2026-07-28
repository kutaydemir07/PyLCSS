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
        self.graph_area = QtWidgets.QWidget()
        graph_layout = QtWidgets.QVBoxLayout(self.graph_area)
        graph_layout.setContentsMargins(0, 0, 0, 0)
        graph_layout.addWidget(self.graph_stack)

        if create_default:
            self._add_system("Default System")

    def _build_panel(self) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)
        layout.addWidget(QtWidgets.QLabel("Product Name"))
        layout.addWidget(self.product_name)
        layout.addWidget(QtWidgets.QLabel("Systems"))
        layout.addWidget(self.systems_list)
        button_layout = QtWidgets.QHBoxLayout()
        button_layout.addWidget(self.btn_add_system)
        button_layout.addWidget(self.btn_remove_system)
        button_layout.addWidget(self.btn_rename_system)
        layout.addLayout(button_layout)
        panel.setMaximumWidth(250)
        return panel

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
