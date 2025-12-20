# Copyright (c) 2025 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""
System list management for the modeling environment.

This module provides functionality for managing multiple system graphs,
including creation, deletion, renaming, and switching between systems.
"""

from PySide6 import QtWidgets, QtCore
from NodeGraphQt import NodeGraph
from .node_types import CustomBlockNode, InputNode, OutputNode, IntermediateNode

class SystemManager(QtCore.QObject):
    """
    Manager for multiple system graphs in the modeling environment.

    Handles the system list UI, graph creation and management,
    and provides signals for system selection changes.
    """

    system_selected = QtCore.Signal()
    system_added = QtCore.Signal(object)  # emits graph

    def __init__(self, parent):
        """
        Initialize the system manager.

        Args:
            parent: Parent widget for dialogs
        """
        super().__init__(parent)
        
        self.systems = []
        self.graph_stack = QtWidgets.QStackedWidget()
        self.current_graph = None
        
        # Systems List Panel
        self.systems_list = QtWidgets.QListWidget()
        self.systems_list.itemSelectionChanged.connect(self.on_system_selected)
        
        self.btn_add_system = QtWidgets.QPushButton("Add")
        self.btn_add_system.clicked.connect(self.add_system)
        
        self.btn_remove_system = QtWidgets.QPushButton("Remove")
        self.btn_remove_system.clicked.connect(self.remove_system)
        
        self.btn_rename_system = QtWidgets.QPushButton("Rename")
        self.btn_rename_system.clicked.connect(self.rename_system)
        
        self.panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(self.panel)
        layout.addWidget(QtWidgets.QLabel("Product Name"))
        self.product_name = QtWidgets.QLineEdit("Product")
        layout.addWidget(self.product_name)
        layout.addWidget(QtWidgets.QLabel("Systems"))
        layout.addWidget(self.systems_list)
        
        btn_layout = QtWidgets.QHBoxLayout()
        btn_layout.addWidget(self.btn_add_system)
        btn_layout.addWidget(self.btn_remove_system)
        btn_layout.addWidget(self.btn_rename_system)
        layout.addLayout(btn_layout)
        self.panel.setMaximumWidth(250)
        
        # Graph area
        self.graph_area = QtWidgets.QWidget()
        graph_layout = QtWidgets.QVBoxLayout(self.graph_area)
        graph_layout.setContentsMargins(0, 0, 0, 0)
        graph_layout.addWidget(self.graph_stack)
        
        # Init
        self._add_system("Default System")

    def on_system_selected(self):
        """Handle system selection from the list widget."""
        row = self.systems_list.currentRow()
        if row >= 0:
            self.graph_stack.setCurrentIndex(row)
            self.current_graph = self.systems[row]['graph']
            self.graph_stack.currentWidget().setFocus()
        else:
            self.current_graph = None
        self.system_selected.emit()

    def add_system(self):
        """Prompt user for new system name and create it."""
        name, ok = QtWidgets.QInputDialog.getText(self.parent(), "Add System", "System Name:")
        if ok and name:
            self._add_system(name)

    def _add_system(self, name):
        """
        Create a new system graph with the given name.

        Args:
            name: Name for the new system
        """
        graph = NodeGraph()
        graph.register_node(CustomBlockNode)
        graph.register_node(InputNode)
        graph.register_node(OutputNode)
        graph.register_node(IntermediateNode)
        self.systems.append({'name': name, 'graph': graph})
        self.systems_list.addItem(name)
        self.graph_stack.addWidget(graph.widget)
        self.systems_list.setCurrentRow(len(self.systems) - 1)
        self.system_added.emit(graph)

    def remove_system(self):
        """Remove the currently selected system after confirmation."""
        row = self.systems_list.currentRow()
        if row >= 0:
            reply = QtWidgets.QMessageBox.question(self.parent(), "Remove System", f"Remove system '{self.systems[row]['name']}'?", QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
            if reply == QtWidgets.QMessageBox.Yes:
                self.graph_stack.removeWidget(self.systems[row]['graph'].widget)
                self.systems_list.takeItem(row)
                del self.systems[row]
                if self.systems:
                    new_row = max(0, row - 1)
                    self.systems_list.setCurrentRow(new_row)
                else:
                    self.current_graph = None
                self.system_selected.emit()

    def rename_system(self):
        """Rename the currently selected system."""
        row = self.systems_list.currentRow()
        if row >= 0:
            current_name = self.systems[row]['name']
            new_name, ok = QtWidgets.QInputDialog.getText(self.parent(), "Rename System", "New Name:", text=current_name)
            if ok and new_name and new_name != current_name:
                self.systems[row]['name'] = new_name
                self.systems_list.item(row).setText(new_name)






