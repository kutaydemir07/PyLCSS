# Copyright (c) 2025 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""
Custom node types and UI components for PyLCSS system modeling.

This module defines custom NodeGraphQt node types used in the system modeling
interface, including input nodes, output nodes, intermediate nodes, and custom
function blocks. It also provides supporting UI components like code editors,
port managers, and color pickers for node configuration.
"""

from typing import Optional, List
from NodeGraphQt import BaseNode, NodeBaseWidget
from PySide6 import QtWidgets, QtCore, QtGui
from PySide6.QtGui import QSyntaxHighlighter, QTextCharFormat, QColor, QFont, QUndoCommand
import re

class PortListWidget(QtWidgets.QWidget):
    """
    Widget for managing input/output ports on custom nodes.

    Provides a list view of existing ports with add/remove functionality,
    allowing users to dynamically modify the port configuration of nodes.
    """

    def __init__(self, node: BaseNode, port_type: str) -> None:
        """
        Initialize the port list widget.

        Args:
            node: The node whose ports are being managed
            port_type: Either 'input' or 'output'
        """
        super(PortListWidget, self).__init__()
        self.node = node
        self.port_type = port_type

        layout = QtWidgets.QVBoxLayout(self)

        self.list_widget = QtWidgets.QListWidget()
        self.refresh_list()
        layout.addWidget(self.list_widget)

        btn_layout = QtWidgets.QHBoxLayout()
        btn_add = QtWidgets.QPushButton("Add")
        btn_remove = QtWidgets.QPushButton("Remove")

        btn_add.clicked.connect(self.add_port)
        btn_remove.clicked.connect(self.remove_port)

        btn_layout.addWidget(btn_add)
        btn_layout.addWidget(btn_remove)
        layout.addLayout(btn_layout)

    def refresh_list(self) -> None:
        """Refresh the port list to reflect current node state."""
        self.list_widget.clear()
        ports = self.node.input_ports() if self.port_type == 'input' else self.node.output_ports()
        for p in ports:
            self.list_widget.addItem(p.name())

    def add_port(self) -> None:
        """Add a new port with user-specified name."""
        name, ok = QtWidgets.QInputDialog.getText(self, "Add Port", "Name:")
        if ok and name:
            try:
                if self.port_type == 'input':
                    self.node.add_input(name)
                else:
                    self.node.add_output(name)
                self.refresh_list()
            except Exception as e:
                QtWidgets.QMessageBox.warning(self, "Error", str(e))

    def remove_port(self) -> None:
        """Remove the currently selected port."""
        item = self.list_widget.currentItem()
        if item:
            name = item.text()
            try:
                if self.port_type == 'input':
                    self.node.delete_input(name)
                else:
                    self.node.delete_output(name)
                self.refresh_list()
            except Exception as e:
                QtWidgets.QMessageBox.warning(self, "Error", str(e))

class PortManagerDialog(QtWidgets.QDialog):
    """
    Dialog for managing ports on custom nodes.

    Provides a tabbed interface for managing input and output ports
    separately, using PortListWidget for each port type.
    """

    def __init__(self, node: BaseNode, parent: Optional[QtWidgets.QWidget] = None) -> None:
        """
        Initialize the port manager dialog.

        Args:
            node: The node whose ports are being managed
            parent: Parent widget
        """
        super(PortManagerDialog, self).__init__(parent)
        self.node = node
        self.setWindowTitle("Manage Ports")
        self.resize(400, 300)

        layout = QtWidgets.QVBoxLayout(self)

        # Tabs for Inputs / Outputs
        tabs = QtWidgets.QTabWidget()
        self.input_tab = PortListWidget(node, 'input')
        self.output_tab = PortListWidget(node, 'output')

        tabs.addTab(self.input_tab, "Inputs")
        tabs.addTab(self.output_tab, "Outputs")
        layout.addWidget(tabs)

        btn_box = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)

class PythonHighlighter(QSyntaxHighlighter):
    def __init__(self, document: QtGui.QTextDocument, variables: Optional[List[str]] = None) -> None:
        super().__init__(document)
        self.variables = variables or []
        
        # Keywords
        self.keyword_format = QTextCharFormat()
        self.keyword_format.setForeground(QColor("#ff79c6"))  # Pink
        self.keyword_format.setFontWeight(QFont.Bold)
        keywords = [
            "and", "as", "assert", "break", "class", "continue", "def", "del", "elif", "else",
            "except", "finally", "for", "from", "global", "if", "import", "in", "is", "lambda",
            "nonlocal", "not", "or", "pass", "raise", "return", "try", "while", "with", "yield"
        ]
        self.keyword_patterns = [r'\b' + re.escape(word) + r'\b' for word in keywords]
        
        # Variables
        self.variable_format = QTextCharFormat()
        self.variable_format.setForeground(QColor("#50fa7b"))  # Green
        self.variable_format.setFontWeight(QFont.Bold)
        
        # Strings
        self.string_format = QTextCharFormat()
        self.string_format.setForeground(QColor("#f1fa8c"))  # Yellow
        
        # Comments
        self.comment_format = QTextCharFormat()
        self.comment_format.setForeground(QColor("#6272a4"))  # Grey
        
    def highlightBlock(self, text: str) -> None:
        # Keywords
        for pattern in self.keyword_patterns:
            for match in re.finditer(pattern, text):
                self.setFormat(match.start(), match.end() - match.start(), self.keyword_format)
        
        # Variables
        if self.variables:
            for var in self.variables:
                pattern = r'\b' + re.escape(var) + r'\b'
                for match in re.finditer(pattern, text):
                    self.setFormat(match.start(), match.end() - match.start(), self.variable_format)
        
        # Strings
        string_pattern = r'(["\'])(?:(?=(\\?))\2.)*?\1'
        for match in re.finditer(string_pattern, text):
            self.setFormat(match.start(), match.end() - match.start(), self.string_format)
        
        # Comments
        comment_pattern = r'#.*'
        for match in re.finditer(comment_pattern, text):
            self.setFormat(match.start(), match.end() - match.start(), self.comment_format)

class CodeEditor(QtWidgets.QPlainTextEdit):
    """
    Enhanced code editor with Python syntax highlighting and line numbers.

    Custom implementation using QPlainTextEdit with:
    - Python syntax highlighting
    - Line numbers
    - Dark theme
    """

    def __init__(self, variables: Optional[List[str]] = None) -> None:
        """Initialize the code editor with Python syntax highlighting."""
        super().__init__()
        self.variables = variables or []

        # Set font
        font = QFont("Consolas", 10)
        font.setStyleHint(QFont.Monospace)
        self.setFont(font)

        # Set tab stop distance (approx 4 spaces)
        metrics = self.fontMetrics()
        self.setTabStopDistance(metrics.horizontalAdvance(' ') * 4)

        # Set style
        self.setStyleSheet("background-color: #2b2b2b; color: #f8f8f2;")

        # Syntax highlighter
        self.highlighter = PythonHighlighter(self.document(), self.variables)

        # Line number area
        self.line_number_area = LineNumberArea(self)
        self.blockCountChanged.connect(self.update_line_number_area_width)
        self.updateRequest.connect(self.update_line_number_area)
        self.update_line_number_area_width(0)

    def update_line_number_area_width(self, _: int) -> None:
        self.setViewportMargins(self.line_number_area_width(), 0, 0, 0)

    def update_line_number_area(self, rect: QtCore.QRect, dy: int) -> None:
        if dy:
            self.line_number_area.scroll(0, dy)
        else:
            self.line_number_area.update(0, rect.y(), self.line_number_area.width(), rect.height())
        if rect.contains(self.viewport().rect()):
            self.update_line_number_area_width(0)

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        cr = self.contentsRect()
        self.line_number_area.setGeometry(QtCore.QRect(cr.left(), cr.top(), self.line_number_area_width(), cr.height()))

    def update_variables(self, variables: List[str]) -> None:
        """Update the list of variables to highlight."""
        self.variables = variables
        self.highlighter.variables = variables
        self.highlighter.rehighlight()  # Re-highlight the entire document

    def line_number_area_width(self) -> int:
        digits = 1
        max_block = max(1, self.blockCount())
        while max_block >= 10:
            max_block //= 10
            digits += 1
        space = 3 + self.fontMetrics().horizontalAdvance('9') * digits
        return space

    def line_number_area_paint_event(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self.line_number_area)
        painter.fillRect(event.rect(), QColor("#333333"))
        block = self.firstVisibleBlock()
        block_number = block.blockNumber()
        top = self.blockBoundingGeometry(block).translated(self.contentOffset()).top()
        bottom = top + self.blockBoundingRect(block).height()

        while block.isValid() and top <= event.rect().bottom():
            if block.isVisible() and bottom >= event.rect().top():
                number = str(block_number + 1)
                painter.setPen(QColor("#cccccc"))
                painter.drawText(0, int(top), self.line_number_area.width(), self.fontMetrics().height(),
                                 QtCore.Qt.AlignRight, number)
            block = block.next()
            top = bottom
            bottom = top + self.blockBoundingRect(block).height()
            block_number += 1


class LineNumberArea(QtWidgets.QWidget):
    def __init__(self, editor: CodeEditor) -> None:
        super().__init__(editor)
        self.code_editor = editor

    def sizeHint(self) -> QtCore.QSize:
        return QtCore.QSize(self.code_editor.line_number_area_width(), 0)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        self.code_editor.line_number_area_paint_event(event)

class CodeEditorDialog(QtWidgets.QDialog):
    def __init__(self, code: str, node: Optional[BaseNode] = None, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super(CodeEditorDialog, self).__init__(parent)
        self.node = node  # Store the node reference
        self.setWindowTitle("Function Block Code Editor")
        self.resize(1200, 700) # Increased width for sidebar
        self.showMaximized()
        
        main_layout = QtWidgets.QHBoxLayout(self)
        
        # --- LEFT: Editor Area ---
        editor_panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(editor_panel)
        
        self.editor = CodeEditor([])
        self.editor.setPlainText(code)
        layout.addWidget(self.editor)
        
        btn_layout = QtWidgets.QHBoxLayout()
        help_btn = QtWidgets.QPushButton("?")
        help_btn.setFixedSize(30, 30)
        help_btn.clicked.connect(self.show_help)
        btn_layout.addWidget(help_btn)
        
        find_btn = QtWidgets.QPushButton("Find/Replace")
        find_btn.clicked.connect(self.show_find_replace)
        btn_layout.addWidget(find_btn)
        
        btn_layout.addStretch()
        
        btn_box = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        btn_box.accepted.connect(self.accept)
        btn_box.rejected.connect(self.reject)
        btn_layout.addWidget(btn_box)
        
        layout.addLayout(btn_layout)
        
        # --- RIGHT: Sidebar ---
        sidebar = QtWidgets.QWidget()
        sidebar.setFixedWidth(250)
        sidebar_layout = QtWidgets.QVBoxLayout(sidebar)
        
        sidebar_layout.addWidget(QtWidgets.QLabel("<b>Available Inputs:</b>"))
        self.input_var_list = QtWidgets.QListWidget()
        self.input_var_list.setToolTip("Double-click to insert variable")
        sidebar_layout.addWidget(self.input_var_list)

        sidebar_layout.addWidget(QtWidgets.QLabel("<b>Available Outputs:</b>"))
        self.output_var_list = QtWidgets.QListWidget()
        self.output_var_list.setToolTip("Double-click to insert variable")
        sidebar_layout.addWidget(self.output_var_list)

        if node:
            self._refresh_var_list()

        self.input_var_list.itemDoubleClicked.connect(self.insert_variable)
        self.output_var_list.itemDoubleClicked.connect(self.insert_variable)
        
        # Add to splitter or layout
        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        splitter.addWidget(editor_panel)
        splitter.addWidget(sidebar)
        splitter.setStretchFactor(0, 1)
        
        main_layout.addWidget(splitter)
        
    def insert_variable(self, item: QtWidgets.QListWidgetItem) -> None:
        var_name = item.text()
        self.editor.insert(var_name)
        self.editor.setFocus()
    
    def _refresh_var_list(self) -> None:
        """Refresh the variable lists from current node inputs and outputs"""
        self.input_var_list.clear()
        self.output_var_list.clear()
        variables = []
        if self.node:
            for port in self.node.input_ports():
                name = port.name()
                self.input_var_list.addItem(name)
                variables.append(name)
            for port in self.node.output_ports():
                name = port.name()
                self.output_var_list.addItem(name)
                variables.append(name)
        self.editor.update_variables(variables)

    def show_help(self) -> None:
        help_text = (
            "# FUNCTION BLOCK HELP\n"
            "# -------------------\n"
            "# 1. Inputs: Use the variable names listed in the sidebar.\n"
            "# 2. Outputs: Assign values to output variables listed in the sidebar.\n"
            "#    (e.g., out_1, out_2, etc.)\n"
            "# \n"
            "# EXAMPLES:\n"
            "#   out_1 = in_1 * 2\n"
            "#   force = mass * acceleration\n"
        )
        QtWidgets.QMessageBox.information(self, "Function Block Help", help_text)
        
    def show_find_replace(self):
        # (Same as before)
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Find & Replace")
        dialog.resize(400, 150)
        layout = QtWidgets.QVBoxLayout(dialog)
        find_layout = QtWidgets.QHBoxLayout()
        find_layout.addWidget(QtWidgets.QLabel("Find:"))
        self.find_edit = QtWidgets.QLineEdit()
        find_layout.addWidget(self.find_edit)
        layout.addLayout(find_layout)
        replace_layout = QtWidgets.QHBoxLayout()
        replace_layout.addWidget(QtWidgets.QLabel("Replace:"))
        self.replace_edit = QtWidgets.QLineEdit()
        replace_layout.addWidget(self.replace_edit)
        layout.addLayout(replace_layout)
        btn_layout = QtWidgets.QHBoxLayout()
        find_btn = QtWidgets.QPushButton("Find")
        find_btn.clicked.connect(self.find_text)
        replace_btn = QtWidgets.QPushButton("Replace")
        replace_btn.clicked.connect(self.replace_text)
        replace_all_btn = QtWidgets.QPushButton("Replace All")
        replace_all_btn.clicked.connect(self.replace_all_text)
        btn_layout.addWidget(find_btn)
        btn_layout.addWidget(replace_btn)
        btn_layout.addWidget(replace_all_btn)
        layout.addLayout(btn_layout)
        dialog.exec_()
        
    def find_text(self):
        text = self.find_edit.text()
        if text:
            found = self.editor.findFirst(text, False, False, False, True, True, -1, -1, True)
            if not found:
                self.editor.findFirst(text, False, False, False, True, True, 0, 0, True)
    def replace_text(self):
        text = self.find_edit.text()
        replace = self.replace_edit.text()
        if text and self.editor.hasSelectedText():
            self.editor.replace(replace)
            self.find_text()
    def replace_all_text(self):
        text = self.find_edit.text()
        replace = self.replace_edit.text()
        if text:
            content = self.editor.toPlainText()
            new_content = content.replace(text, replace)
            self.editor.setPlainText(new_content)
    def get_code(self):
        return self.editor.toPlainText()
        
    def show_help(self) -> None:
        help_text = (
            "# FUNCTION BLOCK CODE EDITOR\n"
            "# ==========================\n"
            "# \n"
            "# BASICS:\n"
            "# - Set output variables by connected node names (e.g. a_x = ...)\n"
            "# - Access inputs by connected node names (e.g. z_1)\n"
            "# - NO return statements - added automatically\n"
            "# \n"
            "# LIBRARIES:\n"
            "# - numpy as np (pre-imported)\n"
            "# - Import others: import pandas as pd\n"
            "# - Available: pandas, sklearn, tensorflow, requests, opencv, etc.\n"
            "# \n"
            "# EXAMPLES:\n"
            "#   import pandas as pd\n"
            "#   data = pd.read_csv('file.csv')\n"
            "#   a_x = data['col'].mean() * z_1\n"
            "#   a_z = np.sin(z_2)\n"
            "# \n"
            "# PACKAGING:\n"
            "# - EXE includes all major scientific libraries\n"
            "# - Users can import/use any bundled library\n"
        )
        QtWidgets.QMessageBox.information(self, "Function Block Help", help_text)
        
    def show_find_replace(self):
        # Simple find/replace dialog
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Find & Replace")
        dialog.resize(400, 150)
        
        layout = QtWidgets.QVBoxLayout(dialog)
        
        find_layout = QtWidgets.QHBoxLayout()
        find_layout.addWidget(QtWidgets.QLabel("Find:"))
        self.find_edit = QtWidgets.QLineEdit()
        find_layout.addWidget(self.find_edit)
        layout.addLayout(find_layout)
        
        replace_layout = QtWidgets.QHBoxLayout()
        replace_layout.addWidget(QtWidgets.QLabel("Replace:"))
        self.replace_edit = QtWidgets.QLineEdit()
        replace_layout.addWidget(self.replace_edit)
        layout.addLayout(replace_layout)
        
        btn_layout = QtWidgets.QHBoxLayout()
        find_btn = QtWidgets.QPushButton("Find")
        find_btn.clicked.connect(self.find_text)
        replace_btn = QtWidgets.QPushButton("Replace")
        replace_btn.clicked.connect(self.replace_text)
        replace_all_btn = QtWidgets.QPushButton("Replace All")
        replace_all_btn.clicked.connect(self.replace_all_text)
        btn_layout.addWidget(find_btn)
        btn_layout.addWidget(replace_btn)
        btn_layout.addWidget(replace_all_btn)
        layout.addLayout(btn_layout)
        
        dialog.exec_()
        
    def find_text(self):
        text = self.find_edit.text()
        if text:
            # Use QsciScintilla's findFirst for initial search
            found = self.editor.findFirst(text, False, False, False, True, True, -1, -1, True)
            if not found:
                # Wrap around from beginning
                self.editor.findFirst(text, False, False, False, True, True, 0, 0, True)
                
    def replace_text(self):
        text = self.find_edit.text()
        replace = self.replace_edit.text()
        if text and self.editor.hasSelectedText():
            self.editor.replace(replace)
            self.find_text()  # Find next
            
    def replace_all_text(self):
        text = self.find_edit.text()
        replace = self.replace_edit.text()
        if text:
            content = self.editor.toPlainText()
            new_content = content.replace(text, replace)
            self.editor.setPlainText(new_content)
        
    def get_code(self):
        return self.editor.toPlainText()

class ColorPickerWidget(NodeBaseWidget):
    def __init__(self, parent=None, name='plot_color', label='', initial_color=(0, 0, 255)):
        super(ColorPickerWidget, self).__init__(parent, name, label)
        self.current_color = initial_color
        
        self.container = QtWidgets.QWidget()
        self.container.setMinimumSize(140, 30) # Ensure visibility
        self.layout = QtWidgets.QHBoxLayout(self.container)
        self.layout.setContentsMargins(2, 2, 2, 2)
        self.layout.setSpacing(4)
        
        # Text Input (Hex)
        self.line_edit = QtWidgets.QLineEdit()
        self.line_edit.setFixedWidth(70)
        self.line_edit.editingFinished.connect(self.on_text_changed)
        
        # Color Button
        self.btn = QtWidgets.QPushButton()
        self.btn.setFixedSize(24, 24)
        self.btn.setCursor(QtCore.Qt.PointingHandCursor)
        self.btn.clicked.connect(self.pick_color)
        
        self.layout.addWidget(self.line_edit)
        self.layout.addWidget(self.btn)
        self.layout.addStretch()
        
        self.update_widgets()
        self.container.show()
        self.set_custom_widget(self.container)
        
    def update_widgets(self):
        r, g, b = self.current_color
        hex_c = '#{:02x}{:02x}{:02x}'.format(r, g, b)
        self.line_edit.setText(hex_c.upper())
        self.btn.setStyleSheet(f"background-color: {hex_c}; border: 1px solid #555; border-radius: 2px;")
        
    def pick_color(self):
        color = QtWidgets.QColorDialog.getColor(
            QtGui.QColor(*self.current_color), 
            self.container, 
            "Select Color"
        )
        if color.isValid():
            self.current_color = (color.red(), color.green(), color.blue())
            self.update_widgets()
            self.value_changed.emit(self.get_name(), self.current_color)

    def on_text_changed(self):
        text = self.line_edit.text()
        if text.startswith('#') and len(text) == 7:
            try:
                c = QtGui.QColor(text)
                if c.isValid():
                    self.current_color = (c.red(), c.green(), c.blue())
                    self.update_widgets()
                    self.value_changed.emit(self.get_name(), self.current_color)
            except:
                pass
            
    def get_value(self):
        return self.current_color
        
    def set_value(self, value):
        if value and isinstance(value, (list, tuple)) and len(value) == 3:
            self.current_color = tuple(value)
            self.update_widgets()

class CodeTextWidget(NodeBaseWidget):
    def __init__(self, parent=None, name='code_content', label='', node=None):
        super(CodeTextWidget, self).__init__(parent, name, label)
        self.node_ref = node
        
        self.container = QtWidgets.QWidget()
        self.layout = QtWidgets.QVBoxLayout(self.container)
        self.layout.setContentsMargins(2, 2, 2, 2)
        self.layout.setSpacing(2)
        
        # Buttons
        self.btn_layout = QtWidgets.QHBoxLayout()
        self.btn_ports = QtWidgets.QPushButton("Manage Ports...")
        self.btn_edit = QtWidgets.QPushButton("Edit Code")

        # Set minimum widths to prevent text truncation
        self.btn_ports.setMinimumWidth(100)
        self.btn_edit.setMinimumWidth(80)
        
        self.btn_layout.addWidget(self.btn_ports)
        self.btn_layout.addWidget(self.btn_edit)
        self.layout.addLayout(self.btn_layout)
        
        # Editor (Preview)
        self.editor = CodeEditor([])
        self.editor.setMinimumHeight(100)
        self.editor.setMinimumWidth(300)
        self.editor.setReadOnly(False) # Allow quick edits
        
        self.layout.addWidget(self.editor)
        self.set_custom_widget(self.container)
        
        self.editor.textChanged.connect(self.on_text_changed)
        self.btn_edit.clicked.connect(self.open_external_editor)
        self.btn_ports.clicked.connect(self.open_port_manager)
        
    def open_external_editor(self):
        # PASS THE NODE REF HERE
        dialog = CodeEditorDialog(self.editor.toPlainText(), node=self.node_ref, parent=self.container)
        if dialog.exec_() == QtWidgets.QDialog.Accepted:
            self.set_value(dialog.get_code())

    def open_port_manager(self):
        if self.node_ref:
            dialog = PortManagerDialog(self.node_ref, self.container)
            dialog.exec_()
        else:
            QtWidgets.QMessageBox.warning(self.container, "Error", "Node reference not found.")
        
    def get_value(self):
        return self.editor.toPlainText()
        
    def set_value(self, value):
        if value != self.editor.toPlainText():
            self.editor.setPlainText(str(value))
            
    def on_text_changed(self):
        self.value_changed.emit(self.get_name(), self.get_value())

class RemovePortsCommand(QUndoCommand):
    def __init__(self, node, ports_to_remove, is_input=True, old_count=None):
        super().__init__("Remove Ports")
        self.node = node
        self.is_input = is_input
        self.ports_to_remove = [p.name() for p in ports_to_remove]
        self.connections = []
        for port in ports_to_remove:
            # Defensive: skip if port is None or port.node is None (port already deleted)
            if port is None or getattr(port, 'node', None) is None:
                continue
            try:
                connected = port.connected_ports()
            except Exception:
                continue
            for cp in connected:
                node_obj = getattr(cp, 'node', None)
                if callable(node_obj):
                    continue
                if node_obj is not None and hasattr(node_obj, 'id'):
                    self.connections.append((port.name(), node_obj.id, cp.name()))
        self.old_count = old_count

    def redo(self):
        for port_name in self.ports_to_remove:
            if self.is_input:
                port = self.node.get_input(port_name)
            else:
                port = self.node.get_output(port_name)
            if port:
                for cp in port.connected_ports():
                    port.disconnect_from(cp)
                if self.is_input:
                    self.node.delete_input(port_name)
                else:
                    self.node.delete_output(port_name)

    def undo(self):
        for port_name in reversed(self.ports_to_remove):
            if self.is_input:
                self.node.add_input(port_name)
            else:
                self.node.add_output(port_name)
        # Restore connections
        for port_name, connected_node_id, connected_port_name in self.connections:
            if self.is_input:
                port = self.node.get_input(port_name)
            else:
                port = self.node.get_output(port_name)
            if port:
                # Find the connected port
                graph = self.node.graph()
                connected_node = None
                for n in graph.all_nodes():
                    if n.id == connected_node_id:
                        connected_node = n
                        break
                if connected_node:
                    connected_port = connected_node.get_input(connected_port_name) or connected_node.get_output(connected_port_name)
                    if connected_port:
                        port.connect_to(connected_port)
        # Restore the property
        if self.old_count is not None:
            prop_name = 'num_inputs' if self.is_input else 'num_outputs'
            self.node.set_property(prop_name, self.old_count, push_undo=False)

class SurrogateControlWidget(NodeBaseWidget):
    """
    Custom widget for Surrogate Model controls embedded in the node.
    """
    def __init__(self, parent: Optional[QtWidgets.QWidget] = None, name: str = 'surrogate_controls', label: str = '') -> None:
        super(SurrogateControlWidget, self).__init__(parent, name, label)
        
        self.container = QtWidgets.QWidget()
        self.layout = QtWidgets.QVBoxLayout(self.container)
        self.layout.setContentsMargins(2, 2, 2, 2)
        self.layout.setSpacing(4)
        
        # 1. Enable Checkbox
        self.chk_enable = QtWidgets.QCheckBox("Enable")
        self.chk_enable.stateChanged.connect(self.on_enable_changed)
        self.layout.addWidget(self.chk_enable)
        
        # 2. Status Label
        self.lbl_status = QtWidgets.QLabel("Status: Not Trained")
        self.lbl_status.setStyleSheet("color: #b5bac1; font-style: italic;")
        self.layout.addWidget(self.lbl_status)
        
        # 3. Train Button
        self.btn_train = QtWidgets.QPushButton("Train Model")
        self.btn_train.setCursor(QtCore.Qt.PointingHandCursor)
        self.btn_train.clicked.connect(self.on_train_clicked)
        self.layout.addWidget(self.btn_train)
        
        self.set_custom_widget(self.container)
        
    def on_enable_changed(self, state):
        # Emit (prop_name, value)
        # NodeGraphQt expects value_changed to emit (name, value) if it's a custom widget
        # But since we are using a custom name 'surrogate_controls', we need to handle it manually in the node
        # OR we can emit the value directly if the node connects to it.
        # However, NodeGraphQt's NodeObject connects widget.value_changed to set_property(widget.name, value)
        
        # Robust check for checked state (2 is Checked, 0 is Unchecked)
        is_checked = (state == 2) or (state == QtCore.Qt.Checked)
        
        # We want to update 'use_surrogate'.
        # So we emit the value, but we need the node to catch it and redirect it.
        self.value_changed.emit('use_surrogate', is_checked)
        
    def on_train_clicked(self):
        # We emit a special signal that the Editor will catch
        import time
        self.value_changed.emit('surrogate_train_trigger', time.time())
        
    def get_value(self):
        return self.chk_enable.isChecked()
        
    def set_value(self, value):
        # This handles the 'use_surrogate' property updates
        if isinstance(value, bool):
            self.chk_enable.setChecked(value)
    
    def set_status(self, text):
        self.lbl_status.setText(f"Status: {text}")

class CustomBlockNode(BaseNode):
    """
    Custom function block node for user-defined mathematical operations.

    Represents a black box function where users can define custom Python code
    to compute outputs from inputs. Supports dynamic port configuration and
    includes a built-in code editor with syntax highlighting.

    Node Properties:
        - num_inputs: Number of input ports (dynamically adjustable)
        - num_outputs: Number of output ports (dynamically adjustable)
        - code_content: Python code defining the function logic
        - func_name: Generated function name for code compilation
    """
    __identifier__ = 'com.pfd.custom_block'
    NODE_NAME = 'Black Box Function'

    def __init__(self) -> None:
        """Initialize the custom block node with default ports and code editor."""
        super(CustomBlockNode, self).__init__()
        self.set_port_deletion_allowed(True)
        # Start with default ports
        self.add_input('in_1')
        self.add_output('out_1')
        if not self.has_property('func_name'):
            self.create_property('func_name', '')
        # Create properties for number of inputs/outputs
        self.add_text_input('num_inputs', 'Number of Inputs', text='1')
        self.add_text_input('num_outputs', 'Number of Outputs', text='1')
        
        # --- NEW SURROGATE PROPERTIES ---
        self.surrogate_widget = SurrogateControlWidget(parent=None, name='surrogate_controls', label='Surrogate')
        self.add_custom_widget(self.surrogate_widget, tab='Surrogate')
        
        # because the widget name 'surrogate_controls' doesn't match the property 'use_surrogate'
        self.surrogate_widget.value_changed.connect(self.on_surrogate_widget_changed)
        
        # Keep internal property for path, but hide it if possible or just leave it
        self.create_property('surrogate_model_path', '')
        
        # We need these properties to exist for the widget to sync with
        self.add_checkbox('use_surrogate', 'Use Surrogate', state=False)
        
        # Add missing properties for surrogate functionality
        self.create_property('surrogate_status', 'Not Trained')
        self.create_property('surrogate_train_trigger', 0.0)
        # --------------------------------
        
        default_code = "# out_1 = in_1 * 2\n"
        self.code_widget = CodeTextWidget(node=self)
        self.code_widget.set_value(default_code)
        self.add_custom_widget(self.code_widget)

    def on_surrogate_widget_changed(self, name, value):
        """Handle signals from the embedded surrogate widget."""
        if name == 'use_surrogate':
            self.set_property('use_surrogate', value)
        elif name == 'surrogate_train_trigger':
            self.set_property('surrogate_train_trigger', value)

    def set_property(self, name, value, push_undo=True):
        """
        Handle property changes with special logic for port management.

        Dynamically adjusts input/output ports when num_inputs/num_outputs
        properties change, and synchronizes code widget content.

        Args:
            name: Property name being changed
            value: New property value
            push_undo: Whether to push change to undo stack
        """
        old_value = self.get_property(name) if self.has_property(name) else None

        if name == 'num_inputs':
            try:
                num = max(1, int(value))  # Ensure at least 1
                self._update_input_ports(num)
                super(CustomBlockNode, self).set_property(name, str(num), push_undo)
            except (ValueError, TypeError):
                pass  # Ignore invalid values
        elif name == 'num_outputs':
            try:
                num = max(1, int(value))  # Ensure at least 1
                self._update_output_ports(num)
                super(CustomBlockNode, self).set_property(name, str(num), push_undo)
            except (ValueError, TypeError):
                pass  # Ignore invalid values
        else:
            super(CustomBlockNode, self).set_property(name, value, push_undo)
            
        
        if name == 'surrogate_status' and hasattr(self, 'surrogate_widget'):
             self.surrogate_widget.set_status(value)
        
        # Forward 'use_surrogate' changes to the widget
        if name == 'use_surrogate' and hasattr(self, 'surrogate_widget'):
            self.surrogate_widget.set_value(value)
        if name == 'code_content':
            self.code_widget.set_value(value)

    def _update_input_ports(self, num_inputs):
        """
        Dynamically adjust input ports to match specified count.

        Intelligently adds or removes ports while preserving renamed ports
        and avoiding conflicts with existing port names.

        Args:
            num_inputs: Target number of input ports
        """
        current_ports = self.input_ports()
        current_count = len(current_ports)

        if num_inputs > current_count:
            # Add new ports
            for i in range(current_count, num_inputs):
                name = f'in_{i+1}'
                # Ensure unique name if in_{i+1} is already taken by a renamed port
                existing_names = [p.name() for p in self.input_ports()]
                idx = i + 1
                while name in existing_names:
                    idx += 1
                    name = f'in_{idx}'
                self.add_input(name)

        elif num_inputs < current_count:
            # Remove ports from the end
            ports_to_remove = current_ports[num_inputs:]
            for p in ports_to_remove:
                for cp in p.connected_ports():
                    p.disconnect_from(cp)
                self.delete_input(p.name())

    def _update_output_ports(self, num_outputs):
        """
        Dynamically adjust output ports to match specified count.

        Intelligently adds or removes ports while preserving renamed ports
        and avoiding conflicts with existing port names.

        Args:
            num_outputs: Target number of output ports
        """
        current_ports = self.output_ports()
        current_count = len(current_ports)

        if num_outputs > current_count:
            # Add new ports
            for i in range(current_count, num_outputs):
                name = f'out_{i+1}'
                # Ensure unique name
                existing_names = [p.name() for p in self.output_ports()]
                idx = i + 1
                while name in existing_names:
                    idx += 1
                    name = f'out_{idx}'
                self.add_output(name)

        elif num_outputs < current_count:
            # Remove ports from the end
            ports_to_remove = current_ports[num_outputs:]
            for p in ports_to_remove:
                for cp in p.connected_ports():
                    p.disconnect_from(cp)
                self.delete_output(p.name())

class InputNode(BaseNode):
    """
    Design variable input node for system models.

    Represents an input parameter to the system with configurable bounds,
    units, and variable naming. These nodes provide the interface between
    design variables and the computational graph.

    Node Properties:
        - var_name: Variable name used in generated code
        - unit: Physical unit of the variable
        - min: Minimum allowed value in design space
        - max: Maximum allowed value in design space
    """
    __identifier__ = 'com.pfd.input'
    NODE_NAME = 'Design Variable'

    def __init__(self) -> None:
        """Initialize the input node with default design variable properties."""
        super(InputNode, self).__init__()
        self.set_port_deletion_allowed(True)
        self.add_output('x')
        self.add_text_input('var_name', 'Variable Name', text='x')
        self.add_text_input('unit', 'Unit', text='-')
        self.add_text_input('min', 'Min (DS)', text='0.0')
        self.add_text_input('max', 'Max (DS)', text='10.0')

    def set_property(self, name, value, push_undo=True):
        """
        Handle property changes for input node configuration.

        Args:
            name: Property name being changed
            value: New property value
            push_undo: Whether to push change to undo stack
        """
        super(InputNode, self).set_property(name, value, push_undo)
        if name == 'var_name':
            # Note: Port renaming logic could be added here if needed
            # Currently the output port remains as 'x' for simplicity
            pass

class OutputNode(BaseNode):
    """
    Quantity of interest output node for system models.

    Represents an output parameter from the system with optimization objectives,
    requirements bounds, and visualization settings. These nodes define the
    system's objectives and constraints.

    Node Properties:
        - var_name: Variable name used in generated code
        - unit: Physical unit of the variable
        - req_min: Required minimum value (constraint)
        - req_max: Required maximum value (constraint)
        - minimize: Whether this output should be minimized
        - maximize: Whether this output should be maximized
    """
    __identifier__ = 'com.pfd.output'
    NODE_NAME = 'Quantity of Interest'

    def __init__(self) -> None:
        """Initialize the output node with default objective properties."""
        super(OutputNode, self).__init__()
        self.set_port_deletion_allowed(True)
        self.add_input('y')
        self.add_text_input('var_name', 'Variable Name', text='y')
        self.add_text_input('unit', 'Unit', text='-')
        self.add_text_input('req_min', 'Req Min', text='-1e9')
        self.add_text_input('req_max', 'Req Max', text='1e9')
        self.add_checkbox('minimize', 'Minimize', state=False)
        self.add_checkbox('maximize', 'Maximize', state=False)

    def set_property(self, name, value, push_undo=True):
        """
        Handle property changes with optimization objective logic.

        When minimize/maximize is enabled, automatically disables the other
        and sets bounds to (-inf, inf) since objectives don't have constraints.

        Args:
            name: Property name being changed
            value: New property value
            push_undo: Whether to push change to undo stack
        """
        super(OutputNode, self).set_property(name, value, push_undo)
        
        if name == 'minimize' and value:
            # Uncheck maximize
            self.set_property('maximize', False, push_undo)
            # Set bounds to -inf inf
            self.set_property('req_min', '-inf', push_undo)
            self.set_property('req_max', 'inf', push_undo)
        elif name == 'maximize' and value:
            # Uncheck minimize
            self.set_property('minimize', False, push_undo)
            # Set bounds to -inf inf
            self.set_property('req_min', '-inf', push_undo)
            self.set_property('req_max', 'inf', push_undo)

class IntermediateNode(BaseNode):
    """
    Intermediate variable node for signal routing and renaming.

    Acts as a pass-through node that can rename variables and change units
    between different parts of the system graph. Useful for organizing complex
    graphs and providing semantic meaning to intermediate calculations.

    Node Properties:
        - var_name: Variable name for this intermediate value
        - unit: Physical unit of the intermediate variable
    """
    __identifier__ = 'com.pfd.intermediate'
    NODE_NAME = 'Intermediate Variable'

    def __init__(self) -> None:
        """Initialize the intermediate node with pass-through ports."""
        super(IntermediateNode, self).__init__()
        self.set_port_deletion_allowed(True)
        self.add_input('z')
        self.add_output('z')
        self.add_text_input('var_name', 'Variable Name', text='z')
        self.add_text_input('unit', 'Unit', text='-')








