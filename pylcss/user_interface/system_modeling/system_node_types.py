# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""
Custom node types and UI components for PyLCSS system modeling.

This module defines custom NodeGraphQt node types used in the system modeling
interface, including input nodes, output nodes, intermediate nodes, and custom
function blocks. It also provides supporting UI components like code editors,
port managers, and color pickers for node configuration.
"""

from typing import Optional, List
from NodeGraphQt import BaseNode
from PySide6 import QtWidgets, QtCore, QtGui
from PySide6.QtGui import QSyntaxHighlighter, QTextCharFormat, QColor, QFont
import json
from pathlib import Path
import re

# Application palette (amber accent on dark panels) so dialogs match the rest
# of the app rather than introducing a separate accent colour.
from pylcss.user_interface.common.theme_manager import COLORS

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
        name = name.strip()
        if ok and name:
            if not re.match(r"^[A-Za-z_]\w*$", name):
                QtWidgets.QMessageBox.warning(
                    self,
                    "Invalid Variable Name",
                    "Port names become Python variables. Use letters, digits and underscores, and do not start with a digit.",
                )
                return
            all_ports = self.node.input_ports() + self.node.output_ports()
            if name in {port.name() for port in all_ports}:
                QtWidgets.QMessageBox.warning(self, "Duplicate Variable", f"A port named '{name}' already exists.")
                return
            try:
                if self.port_type == 'input':
                    self.node.add_input(name)
                else:
                    self.node.add_output(name)
                self._sync_port_count()
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
                self._sync_port_count()
                self.refresh_list()
            except Exception as e:
                QtWidgets.QMessageBox.warning(self, "Error", str(e))

    def _sync_port_count(self) -> None:
        """Keep the node's serialized count properties in step with the list."""
        property_name = 'num_inputs' if self.port_type == 'input' else 'num_outputs'
        ports = self.node.input_ports() if self.port_type == 'input' else self.node.output_ports()
        if hasattr(self.node, 'has_property') and self.node.has_property(property_name):
            self.node.set_property(property_name, str(len(ports)))

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
        if not painter.isActive():
            return
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
    # Match the application theme (amber accent on dark panels): card-style
    # groups, flat focus-highlighted inputs, accent-underlined tabs.  Combo
    # boxes are left native so their drop-down arrows render correctly.
    _EDITOR_QSS = f"""
        QDialog {{ background: {COLORS['bg_dark']}; }}
        QLabel {{ color: {COLORS['text_dim']}; }}
        QGroupBox {{
            background: {COLORS['bg_panel']}; border: 1px solid {COLORS['bg_dark']};
            border-radius: 8px; margin-top: 14px; padding: 12px 10px 10px 10px;
            font-weight: 600; color: {COLORS['text_main']};
        }}
        QGroupBox::title {{
            subcontrol-origin: margin; subcontrol-position: top left;
            left: 10px; padding: 0 4px; color: {COLORS['primary']}; font-weight: 700;
        }}
        QLineEdit {{
            background: {COLORS['bg_input']}; border: 1px solid {COLORS['bg_dark']};
            border-radius: 6px; padding: 5px 8px; color: {COLORS['text_main']};
            selection-background-color: {COLORS['primary']};
        }}
        QLineEdit:focus {{ border: 1px solid {COLORS['primary']}; }}
        QPushButton {{
            background: {COLORS['bg_input']}; border: 1px solid {COLORS['bg_panel']};
            border-radius: 6px; padding: 6px 12px; color: {COLORS['text_main']}; font-weight: 600;
        }}
        QPushButton:hover {{ border-color: {COLORS['primary']}; }}
        QPushButton:pressed {{ background: {COLORS['bg_dark']}; }}
        QTabWidget::pane {{ border: 1px solid {COLORS['bg_dark']}; border-radius: 8px; top: -1px; }}
        QTabBar::tab {{
            background: transparent; color: {COLORS['text_dim']}; padding: 7px 16px;
            border: none; border-bottom: 2px solid transparent; font-weight: 600;
        }}
        QTabBar::tab:selected {{ color: {COLORS['text_main']}; border-bottom: 2px solid {COLORS['primary']}; }}
        QTabBar::tab:hover {{ color: {COLORS['text_main']}; }}
        QToolButton {{ background: transparent; color: {COLORS['text_dim']}; border: none; }}
        QScrollArea {{ border: none; background: transparent; }}
    """

    def __init__(self, code: str, node: Optional[BaseNode] = None, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super(CodeEditorDialog, self).__init__(parent)
        self.node = node
        self.setWindowTitle("Function Block Code Editor")
        self.resize(1100, 700)
        self.setStyleSheet(self._EDITOR_QSS)
        self.showMaximized()

        main_layout = QtWidgets.QHBoxLayout(self)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)

        editor_panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(editor_panel)

        self.editor = CodeEditor([])
        self.editor.setPlainText(code)
        cursor = self.editor.textCursor()
        cursor.movePosition(QtGui.QTextCursor.End)
        self.editor.setTextCursor(cursor)
        layout.addWidget(self.editor)

        btn_layout = QtWidgets.QHBoxLayout()
        help_btn = QtWidgets.QPushButton("?")
        help_btn.setFixedSize(30, 30)
        help_btn.setToolTip("Show function-block and Design Studio CAD connection help")
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

        splitter.addWidget(editor_panel)

        inspector = QtWidgets.QWidget()
        inspector.setMinimumWidth(330)
        inspector.setMaximumWidth(460)
        inspector_layout = QtWidgets.QVBoxLayout(inspector)
        inspector_title = QtWidgets.QLabel("Block Interface")
        inspector_title.setStyleSheet("font-weight: bold;")
        inspector_layout.addWidget(inspector_title)

        self.inspector_tabs = QtWidgets.QTabWidget()
        self.inspector_tabs.addTab(self._build_interface_tab(), "Variables")
        self.inspector_tabs.addTab(self._build_coupling_tab(), "Simulation")
        inspector_layout.addWidget(self.inspector_tabs, 1)
        splitter.addWidget(inspector)
        splitter.setSizes([800, 360])
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        main_layout.addWidget(splitter)

        self._refresh_var_list()
        
    def _build_interface_tab(self) -> QtWidgets.QWidget:
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(tab)
        layout.setContentsMargins(8, 10, 8, 8)
        layout.setSpacing(8)

        description = QtWidgets.QLabel(
            "These names come from the ports on the function block. "
            "Double-click any name to insert it into the code."
        )
        description.setWordWrap(True)
        layout.addWidget(description)

        self.input_title = QtWidgets.QLabel("Inputs (0)")
        self.input_title.setStyleSheet("font-weight: bold;")
        layout.addWidget(self.input_title)
        input_help = QtWidgets.QLabel("Values available for reading in this code.")
        input_help.setStyleSheet("color: #b5bac1;")
        layout.addWidget(input_help)

        self.input_list = QtWidgets.QListWidget()
        self.input_list.setSpacing(1)
        self.input_list.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.input_list.setStyleSheet(
            "QListWidget { background: #25272b; border: 1px solid #3a3d43; }"
            "QListWidget::item { padding: 6px; border-bottom: 1px solid #34373d; }"
            "QListWidget::item:selected { background: #3b4658; color: white; }"
        )
        self.input_list.itemDoubleClicked.connect(self.insert_variable)
        layout.addWidget(self.input_list)

        self.output_title = QtWidgets.QLabel("Outputs (0)")
        self.output_title.setStyleSheet("font-weight: bold; margin-top: 5px;")
        layout.addWidget(self.output_title)
        output_help = QtWidgets.QLabel("Names that this code must assign before it finishes.")
        output_help.setStyleSheet("color: #b5bac1;")
        layout.addWidget(output_help)

        self.output_list = QtWidgets.QListWidget()
        self.output_list.setSpacing(1)
        self.output_list.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.output_list.setStyleSheet(
            "QListWidget { background: #25272b; border: 1px solid #3a3d43; }"
            "QListWidget::item { padding: 6px; border-bottom: 1px solid #34373d; }"
            "QListWidget::item:selected { background: #3b4658; color: white; }"
        )
        self.output_list.itemDoubleClicked.connect(self.insert_variable)
        layout.addWidget(self.output_list)

        layout.addStretch()
        manage_btn = QtWidgets.QPushButton("Manage block ports")
        manage_btn.clicked.connect(self._open_port_manager)
        layout.addWidget(manage_btn)
        return tab

    def _build_coupling_tab(self) -> QtWidgets.QWidget:
        container = QtWidgets.QWidget()
        container_layout = QtWidgets.QVBoxLayout(container)
        container_layout.setContentsMargins(0, 0, 0, 0)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        page = QtWidgets.QWidget()
        page.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Preferred)
        layout = QtWidgets.QVBoxLayout(page)
        layout.setContentsMargins(8, 10, 8, 8)
        layout.setSpacing(10)

        description = QtWidgets.QLabel(
            "Use a saved Design Studio model as the calculation for this function block. "
            "When the block runs, PyLCSS runs the selected simulation and returns its results."
        )
        description.setWordWrap(True)
        layout.addWidget(description)

        project_group = QtWidgets.QGroupBox("1. Choose the saved analysis")
        project_layout = QtWidgets.QVBoxLayout(project_group)
        project_help = QtWidgets.QLabel(
            "Select a .cad file that already contains the geometry, material, loads and solver setup."
        )
        project_help.setWordWrap(True)
        project_layout.addWidget(project_help)
        project_row = QtWidgets.QHBoxLayout()
        self.project_edit = QtWidgets.QLineEdit()
        self.project_edit.setPlaceholderText("Saved Design Studio file (.cad)")
        self.project_edit.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Fixed)
        self.project_edit.editingFinished.connect(self._inspect_design_studio_project)
        project_row.addWidget(self.project_edit, 1)
        browse_btn = QtWidgets.QPushButton("Browse")
        browse_btn.clicked.connect(self._browse_cad_project)
        project_row.addWidget(browse_btn)
        project_layout.addLayout(project_row)

        active_btn = QtWidgets.QPushButton("Use file open in Design Studio")
        active_btn.clicked.connect(self._use_active_design_studio_project)
        project_layout.addWidget(active_btn)
        self.project_status = QtWidgets.QLabel("No file selected.")
        self.project_status.setWordWrap(True)
        self.project_status.setStyleSheet("color: #b5bac1;")
        project_layout.addWidget(self.project_status)
        layout.addWidget(project_group)

        solver_group = QtWidgets.QGroupBox("2. Choose what to run")
        solver_layout = QtWidgets.QVBoxLayout(solver_group)
        self.solver_combo = QtWidgets.QComboBox()
        self.solver_combo.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.solver_combo.setMinimumContentsLength(18)
        self.solver_combo.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Fixed)
        self.solver_combo.addItem("Static FEA — stress, displacement and mass", "fea")
        self.solver_combo.addItem("Crash — impact response and absorbed energy", "crash")
        self.solver_combo.addItem("Topology optimization — optimized material layout", "topopt")
        self.solver_combo.currentIndexChanged.connect(self._on_solver_changed)
        solver_layout.addWidget(self.solver_combo)

        self.solver_note = QtWidgets.QLabel()
        self.solver_note.setWordWrap(True)
        self.solver_note.setStyleSheet("color: #b5bac1;")
        solver_layout.addWidget(self.solver_note)
        layout.addWidget(solver_group)

        mapping_group = QtWidgets.QGroupBox("3. Connect block variables")
        mapping_layout = QtWidgets.QVBoxLayout(mapping_group)
        self.input_summary_label = QtWidgets.QLabel()
        self.input_summary_label.setWordWrap(True)
        mapping_layout.addWidget(self.input_summary_label)

        self.advanced_mapping_toggle = QtWidgets.QToolButton()
        self.advanced_mapping_toggle.setText("Choose what each input controls")
        self.advanced_mapping_toggle.setCheckable(True)
        self.advanced_mapping_toggle.setArrowType(QtCore.Qt.RightArrow)
        self.advanced_mapping_toggle.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        self.advanced_mapping_toggle.toggled.connect(self._toggle_advanced_mapping)
        mapping_layout.addWidget(self.advanced_mapping_toggle)

        self.advanced_mapping_frame = QtWidgets.QFrame()
        self.advanced_mapping_frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        advanced_layout = QtWidgets.QVBoxLayout(self.advanced_mapping_frame)
        advanced_help = QtWidgets.QLabel(
            "Each block input can drive a geometry parameter, material value, mesh setting, load, or solver control."
        )
        advanced_help.setWordWrap(True)
        advanced_help.setStyleSheet("color: #b5bac1;")
        advanced_layout.addWidget(advanced_help)
        self.input_mapping_form = QtWidgets.QFormLayout()
        advanced_layout.addLayout(self.input_mapping_form)
        self.advanced_mapping_frame.hide()
        mapping_layout.addWidget(self.advanced_mapping_frame)

        output_help = QtWidgets.QLabel("Choose the simulation result for each block output:")
        output_help.setWordWrap(True)
        output_help.setStyleSheet("font-weight: bold; margin-top: 4px;")
        mapping_layout.addWidget(output_help)
        self.output_mapping_form = QtWidgets.QFormLayout()
        mapping_layout.addLayout(self.output_mapping_form)
        layout.addWidget(mapping_group)

        insert_btn = QtWidgets.QPushButton("Use this analysis in the function block")
        insert_btn.clicked.connect(self._insert_design_studio_coupling)
        layout.addWidget(insert_btn)

        self.coupling_status = QtWidgets.QLabel("")
        self.coupling_status.setWordWrap(True)
        layout.addWidget(self.coupling_status)
        layout.addStretch()

        scroll.setWidget(page)
        container_layout.addWidget(scroll)
        self._on_solver_changed()
        return container

    _SOLVER_RESULTS = {
        "fea": (
            ("Maximum stress", "max_stress"),
            ("Maximum displacement", "peak_disp"),
            ("Compliance (flexibility)", "compliance"),
            ("Strain energy", "strain_energy"),
            ("Mass", "mass"),
            ("Volume", "volume"),
        ),
        "crash": (
            ("Maximum stress", "max_stress"),
            ("Maximum displacement", "peak_disp"),
            ("Absorbed energy", "absorbed_energy"),
            ("Peak crushing force", "peak_force"),
            ("Mean crushing force", "mean_force"),
            ("Crush force efficiency", "crush_force_efficiency"),
            ("Specific energy absorption", "specific_energy_absorption"),
            ("Useful crush distance", "crush_distance"),
            ("Peak crash-pulse acceleration", "peak_acceleration_g"),
            ("Crash-pulse velocity change", "delta_v"),
            ("Failed element count", "n_failed"),
        ),
        "topopt": (
            ("Final material fraction", "final_vol_frac"),
            ("Compliance (flexibility)", "compliance"),
            ("Optimized mass", "mass"),
            ("Retained volume", "volume"),
            ("Original volume", "total_volume"),
        ),
    }

    _SOLVER_NOTES = {
        "fea": "Runs one static structural solve. This is the normal choice for stress, stiffness, displacement or mass.",
        "crash": "Runs one explicit impact solve. Crash simulations can be slow when the function block is evaluated many times.",
        "topopt": "Runs a complete topology optimization each time the function block is evaluated. Usually use this to create a design once, or train a surrogate before the main optimization.",
    }

    def _open_port_manager(self) -> None:
        if not self.node:
            return
        dialog = PortManagerDialog(self.node, self)
        dialog.exec_()
        self._refresh_var_list()

    def _browse_cad_project(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select Design Studio Project",
            self.project_edit.text(),
            "Design Studio Projects (*.cad);;All Files (*)",
        )
        if path:
            self.project_edit.setText(QtCore.QDir.toNativeSeparators(path))
            self._inspect_design_studio_project()

    def _use_active_design_studio_project(self) -> None:
        for widget in QtWidgets.QApplication.topLevelWidgets():
            studio = getattr(widget, "cad_widget", None)
            current_file = getattr(studio, "current_file", None)
            if current_file:
                self.project_edit.setText(QtCore.QDir.toNativeSeparators(str(current_file)))
                self._inspect_design_studio_project()
                return
        self.project_status.setText("The file open in Design Studio has not been saved yet. Save it first, then try again.")
        self.project_status.setStyleSheet("color: #ed4245;")

    def _inspect_design_studio_project(self) -> None:
        """Explain what is available in the selected Design Studio file."""
        path_text = self.project_edit.text().strip()
        self._studio_parameters = set()
        self._studio_controls = []
        if not path_text:
            self.project_status.setText("No file selected.")
            self.project_status.setStyleSheet("color: #b5bac1;")
            self._refresh_coupling_mappings()
            return

        path = Path(path_text)
        if not path.is_file():
            self.project_status.setText("This file cannot be found. Select an existing .cad file.")
            self.project_status.setStyleSheet("color: #ed4245;")
            self._refresh_coupling_mappings()
            return

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            nodes = data.get("nodes", {})
            from pylcss.design_studio.runtime import discover_override_controls

            self._studio_controls = discover_override_controls(data)
            parameters = set()
            solvers = set()
            for node_data in nodes.values():
                node_type = str(node_data.get("type_", "")).lower()
                custom = node_data.get("custom", {}) or {}
                exposed_name = str(custom.get("exposed_name", "")).strip()
                if exposed_name:
                    parameters.add(exposed_name)
                for key, value in custom.items():
                    if re.match(r"^param_\d+_name$", str(key)) and str(value).strip():
                        parameters.add(str(value).strip())

                if "crash_solver" in node_type:
                    solvers.add("crash")
                elif "topopt" in node_type or "topology" in node_type:
                    solvers.add("topopt")
                elif ".sim.solver" in node_type or node_type.endswith(".solvernode"):
                    solvers.add("fea")

            self._studio_parameters = parameters
            solver_names = {
                "fea": "Static FEA",
                "crash": "Crash",
                "topopt": "Topology optimization",
            }
            details = []
            if solvers:
                details.append("analysis: " + ", ".join(solver_names[k] for k in ("fea", "crash", "topopt") if k in solvers))
                detected = next((k for k in ("fea", "crash", "topopt") if k in solvers), None)
                index = self.solver_combo.findData(detected)
                if index >= 0:
                    self.solver_combo.setCurrentIndex(index)
            else:
                details.append("no solver node found")
            if parameters:
                details.append(f"{len(parameters)} geometry parameter(s)")
            material_names = sorted({
                item["node"] for item in self._studio_controls
                if item["group"] == "Material"
            })
            if material_names:
                details.append("material: " + ", ".join(material_names))
            control_groups = sorted({item["group"] for item in self._studio_controls})
            if self._studio_controls:
                details.append(
                    f"{len(self._studio_controls)} adjustable setting(s) across "
                    + ", ".join(control_groups)
                )
            self.project_status.setText("Ready. Found " + "; ".join(details) + ".")
            self.project_status.setStyleSheet("color: #2ecc71;")
        except Exception as exc:
            self._studio_controls = []
            self.project_status.setText(f"Could not read this Design Studio file: {exc}")
            self.project_status.setStyleSheet("color: #ed4245;")
        self._refresh_coupling_mappings()

    def _toggle_advanced_mapping(self, visible: bool) -> None:
        self.advanced_mapping_frame.setVisible(visible)
        self.advanced_mapping_toggle.setArrowType(
            QtCore.Qt.DownArrow if visible else QtCore.Qt.RightArrow
        )

    def _on_solver_changed(self, *_args) -> None:
        if not hasattr(self, "solver_combo"):
            return
        solver = self.solver_combo.currentData() or "fea"
        if hasattr(self, "solver_note"):
            self.solver_note.setText(self._SOLVER_NOTES[solver])
        if hasattr(self, "output_mapping_form"):
            self._refresh_coupling_mappings()

    def _refresh_coupling_mappings(self) -> None:
        if not hasattr(self, "input_mapping_form"):
            return
        inputs, outputs = self._current_port_names()

        old_inputs = {
            name: combo.currentData()
            for name, combo in getattr(self, "input_target_combos", {}).items()
        }
        solver = self.solver_combo.currentData() or "fea"
        if getattr(self, "_mapping_solver", solver) == solver:
            old_outputs = {
                name: combo.currentData()
                for name, combo in getattr(self, "output_result_combos", {}).items()
            }
        else:
            old_outputs = {}
        while self.input_mapping_form.rowCount():
            self.input_mapping_form.removeRow(0)
        while self.output_mapping_form.rowCount():
            self.output_mapping_form.removeRow(0)

        parameters = sorted(getattr(self, "_studio_parameters", set()))
        controls = sorted(
            getattr(self, "_studio_controls", []),
            key=lambda item: (item["group"], item["node"], item["label"]),
        )
        self.input_target_combos = {}
        for name in inputs:
            target_combo = QtWidgets.QComboBox()
            target_combo.setMaxVisibleItems(18)
            target_combo.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToMinimumContentsLengthWithIcon)
            target_combo.setMinimumContentsLength(15)
            target_combo.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Fixed)
            if not parameters and not controls:
                target_combo.addItem(f"Geometry parameter / {name}", ("parameter", name))
            elif name in parameters:
                target_combo.addItem(f"Geometry parameter / {name}", ("parameter", name))
            else:
                target_combo.addItem("Choose a Design Studio control…", None)

            target_combo.addItem("Do not send this input to Design Studio", ("ignore", ""))
            for parameter in parameters:
                data = ("parameter", parameter)
                if target_combo.findData(data) < 0:
                    target_combo.addItem(f"Geometry / {parameter}", data)
            for control in controls:
                value = control["value"]
                display = (
                    f"{control['group']} / {control['node']} / "
                    f"{control['label']}  [{value:g}]"
                    if isinstance(value, (int, float)) and not isinstance(value, bool)
                    else f"{control['group']} / {control['node']} / {control['label']}  [{value}]"
                )
                target_combo.addItem(display, ("setting", control["key"]))

            previous = old_inputs.get(name)
            previous_index = target_combo.findData(previous)
            if previous_index >= 0:
                target_combo.setCurrentIndex(previous_index)
            target_combo.setToolTip(
                f"Choose which geometry, material, mesh, load, or solver value is driven by block input '{name}'."
            )
            self.input_mapping_form.addRow(f"{name} controls", target_combo)
            self.input_target_combos[name] = target_combo

        result_fields = self._SOLVER_RESULTS[solver]
        self.output_result_combos = {}
        for row, name in enumerate(outputs):
            result_combo = QtWidgets.QComboBox()
            result_combo.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToMinimumContentsLengthWithIcon)
            result_combo.setMinimumContentsLength(15)
            result_combo.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Fixed)
            for display, field in result_fields:
                result_combo.addItem(display, field)
            preferred = old_outputs.get(name, name)
            preferred_index = result_combo.findData(preferred)
            if preferred_index >= 0:
                result_combo.setCurrentIndex(preferred_index)
            elif row < len(result_fields):
                result_combo.setCurrentIndex(row)
            self.output_mapping_form.addRow(f"{name} receives", result_combo)
            self.output_result_combos[name] = result_combo
        self._mapping_solver = solver
        self._update_input_summary()

    def _update_input_summary(self) -> None:
        if not hasattr(self, "input_summary_label"):
            return
        inputs, _ = self._current_port_names()
        if not inputs:
            self.input_summary_label.setText("This block has no inputs to pass to Design Studio.")
            return
        parameters = getattr(self, "_studio_parameters", set())
        controls = getattr(self, "_studio_controls", [])
        names = ", ".join(inputs)
        if not parameters and not controls:
            self.input_summary_label.setText(
                f"Inputs are passed automatically using the same names: {names}."
            )
            return
        missing = [name for name in inputs if name not in parameters]
        if missing:
            self.input_summary_label.setText(
                f"{len(inputs) - len(missing)} of {len(inputs)} inputs match geometry parameters. "
                f"Open the control list below to connect any input to geometry, material, mesh, loads, or solver settings."
            )
        else:
            self.input_summary_label.setText(
                f"All {len(inputs)} inputs match parameters in the selected file: {names}."
            )

    def _insert_design_studio_coupling(self) -> None:
        project_path = self.project_edit.text().strip()
        if not project_path:
            self.coupling_status.setText("Select a saved .cad project before inserting the adapter.")
            self.inspector_tabs.setCurrentIndex(1)
            return

        solver = self.solver_combo.currentData() or "fea"
        solver_label = self.solver_combo.currentText().split("—")[0].strip()
        call_lines = []
        setting_lines = []
        for variable, target_combo in self.input_target_combos.items():
            target = target_combo.currentData()
            if target is None:
                self.coupling_status.setText(
                    f"Choose what block input '{variable}' controls, or select 'Do not send'."
                )
                self.advanced_mapping_toggle.setChecked(True)
                return
            target_kind, target_name = target
            if target_kind == "ignore":
                continue
            if not re.match(r"^[A-Za-z_]\w*$", variable):
                self.coupling_status.setText(
                    f"'{variable}' is not a valid block input name. Rename the port before generating the simulation call."
                )
                return
            if target_kind == "parameter":
                if not re.match(r"^[A-Za-z_]\w*$", target_name):
                    self.coupling_status.setText(
                        f"'{target_name}' is not a valid Design Studio parameter name."
                    )
                    return
                call_lines.append(f"    {target_name}={variable},")
            elif target_kind == "setting":
                setting_lines.append(f"        {target_name!r}: {variable},")

        normalized_path = project_path.replace("\\", "/")
        lines = [f"# Design Studio coupling: {solver_label}"]
        if call_lines or setting_lines:
            lines.append(f"_study = cad.{solver}(")
            lines.append(f"    {normalized_path!r},")
            if setting_lines:
                lines.append("    _settings={")
                lines.extend(setting_lines)
                lines.append("    },")
            lines.extend(call_lines)
            lines.append(")")
        else:
            lines.append(f"_study = cad.{solver}({normalized_path!r})")

        for output_name, result_combo in self.output_result_combos.items():
            if not re.match(r"^[A-Za-z_]\w*$", output_name):
                self.coupling_status.setText(
                    f"'{output_name}' is not a valid block output name. Rename the port before generating the simulation call."
                )
                return
            lines.append(f"{output_name} = _study.{result_combo.currentData()}")

        snippet = "\n".join(lines) + "\n"
        cursor = self.editor.textCursor()
        before_cursor = self.editor.toPlainText()[:cursor.position()]
        if cursor.position() and not before_cursor.endswith("\n"):
            cursor.insertText("\n")
        cursor.insertText(snippet)
        self.editor.setTextCursor(cursor)
        self.editor.setFocus()
        self.coupling_status.setText(
            f"Added the {solver_label} call to the code. Click OK to save the function block."
        )

    def insert_variable(self, item: QtWidgets.QListWidgetItem) -> None:
        var_name = item.data(QtCore.Qt.UserRole) or item.text()
        self.editor.insertPlainText(var_name)
        self.editor.setFocus()

    # ── CAD-runtime helpers ────────────────────────────────────────────
    # Catalogue of cad.* commands AND every standardised CadResult field
    # displayed in the help dialog. Each entry is (display, snippet, tooltip).
    # When ``snippet`` is None the entry renders as a section header.
    _CAD_COMMANDS = (
        # ── Commands ───────────────────────────────────────────────────
        ("— Commands —", None, None),
        (
            "cad.fea(path, **inputs)",
            'cad.fea("file.cad", param=value)',
            "Run CalculiX linear-static FEA on a .cad graph file.\n"
            "Args:  cad_path (str), then **inputs matched against\n"
            "       'exposed_name' on Number/Variable nodes, or named\n"
            "       Code Part parameters in the graph.\n"
            "Returns CadResult — see the 'FEA result' rows below.",
        ),
        (
            "cad.crash(path, **inputs)",
            'cad.crash("file.cad", param=value)',
            "Run OpenRadioss explicit crash on a .cad graph file.\n"
            "Args:  cad_path (str), then **inputs matched against\n"
            "       'exposed_name' on Number/Variable nodes, or named\n"
            "       Code Part parameters in the graph.\n"
            "Returns CadResult — see the 'Crash result' rows below.",
        ),
        (
            "cad.topopt(path, **inputs)",
            'cad.topopt("file.cad", param=value)',
            "Run SIMP topology optimisation through a .cad graph file.\n"
            "Args:  cad_path (str), then **inputs matched against\n"
            "       'exposed_name' on Number/Variable nodes, or named\n"
            "       Code Part parameters in the graph.\n"
            "Returns CadResult — see the 'TopOpt result' rows below.",
        ),

        # ── Helpers ────────────────────────────────────────────────────
        ("— Helpers —", None, None),
        (
            "result.pick(...)",
            '.pick("max_stress", "mass")',
            "Tuple-unpack helper:\n"
            "    s, m = cad.fea(...).pick('max_stress', 'mass')",
        ),
        (
            "result[\"key\"]",
            '["max_stress"]',
            "Dict access. Same fields as attribute access, plus any raw key\n"
            "produced by the underlying solver result (mesh, frd_file, …).",
        ),
        (
            "result.raw()",
            ".raw()",
            "Full raw result dict from the solver — useful when you need\n"
            "fields outside the standard set (mesh, FRD file path, ENER fields…).",
        ),

        # ── FEA standard fields ────────────────────────────────────────
        ("— FEA result (cad.fea) —", None, None),
        (".max_stress",
         ".max_stress",
         "Peak Von Mises stress at Gauss points [MPa].\n"
         "Conservative (un-smoothed) — use for safety-factor calcs."),
        (".compliance",
         ".compliance",
         "Compliance C = u·f = u·K·u [N·mm].\n"
         "Smaller = stiffer.  C = 2 × total elastic strain energy."),
        (".strain_energy",
         ".strain_energy",
         "Total elastic strain energy ∫ ½σ:ε dV  [N·mm].\n"
         "Equals compliance / 2 for linear elastic materials."),
        (".mass",
         ".mass",
         "Total mass [t in the standard mm/t/s unit system].\n"
         "= volume × material.rho from the connected MaterialNode."),
        (".volume",
         ".volume",
         "Total mesh volume Σ V_e  [mm³]."),
        (".peak_disp",
         ".peak_disp",
         "Max |u| across all mesh nodes [mm]."),

        # ── Crash standard fields ──────────────────────────────────────
        ("— Crash result (cad.crash) —", None, None),
        (".max_stress",
         ".max_stress",
         "Peak Von Mises stress over the whole transient [MPa]."),
        (".peak_disp",
         ".peak_disp",
         "Max nodal displacement magnitude over the whole transient [mm]."),
        (".absorbed_energy",
         ".absorbed_energy",
         "Plastic dissipation Σ_e ∫ σ_y · dε_p · V_e [N·mm].\n"
         "Standard crashworthiness metric — bigger = more energy soaked up."),
        (".n_failed",
         ".n_failed",
         "Number of elements deleted via the failure criterion."),
        (".peak_force",
         ".peak_force",
         "Filtered peak crushing force from the rigid wall [kN]."),
        (".mean_force",
         ".mean_force",
         "Force-displacement energy divided by useful crush stroke [kN]."),
        (".crush_force_efficiency",
         ".crush_force_efficiency",
         "Mean crushing force / peak crushing force [-]."),
        (".specific_energy_absorption",
         ".specific_energy_absorption",
         "Absorbed energy per structural mass [kJ/kg]."),
        (".crush_distance",
         ".crush_distance",
         "Monotonic useful crush stroke [mm]."),
        (".peak_acceleration_g",
         ".peak_acceleration_g",
         "Filtered peak reference deceleration [g]."),
        (".delta_v",
         ".delta_v",
         "Velocity change obtained by integrating the crash pulse [m/s]."),

        # ── TopOpt standard fields ─────────────────────────────────────
        ("— TopOpt result (cad.topopt) —", None, None),
        (".final_vol_frac",
         ".final_vol_frac",
         "Volume-weighted final physical density fraction; should match the\n"
         "vol_frac target set on the Topology Opt node."),
        (".compliance",
         ".compliance",
         "Compliance at the final density field [N·mm]."),
        (".mass",
         ".mass",
         "Effective mass = Σ ρ_e · V_e · material.rho  [t]."),
        (".volume",
         ".volume",
         "Retained material volume after topology optimisation [mm3]."),
        (".total_volume",
         ".total_volume",
         "Original design-domain mesh volume before material removal [mm3]."),
    )

    
    def _refresh_var_list(self) -> None:
        """Refresh the visible block contract and syntax highlighting."""
        inputs, outputs = self._current_port_names()
        self.editor.update_variables(inputs + outputs)

        if hasattr(self, "input_list"):
            self.input_list.clear()
            for name in inputs:
                item = QtWidgets.QListWidgetItem(f"IN     {name}")
                item.setData(QtCore.Qt.UserRole, name)
                item.setToolTip(f"Input variable · double-click to insert '{name}'")
                item.setForeground(QtGui.QColor("#38c8e8"))
                self.input_list.addItem(item)
            self.input_list.setFixedHeight(min(220, max(42, len(inputs) * 33 + 6)))

        if hasattr(self, "output_list"):
            self.output_list.clear()
            for name in outputs:
                item = QtWidgets.QListWidgetItem(f"OUT   {name}")
                item.setData(QtCore.Qt.UserRole, name)
                item.setToolTip(f"Output assignment · double-click to insert '{name}'")
                item.setForeground(QtGui.QColor("#36c98f"))
                self.output_list.addItem(item)
            self.output_list.setFixedHeight(min(220, max(42, len(outputs) * 33 + 6)))

        if hasattr(self, "input_title"):
            self.input_title.setText(f"Inputs ({len(inputs)})")
        if hasattr(self, "output_title"):
            self.output_title.setText(f"Outputs ({len(outputs)})")
        self._refresh_coupling_mappings()

    def _current_port_names(self) -> tuple[list[str], list[str]]:
        """Return current function-block input and output port names."""
        inputs: list[str] = []
        outputs: list[str] = []
        if self.node:
            for port in self.node.input_ports():
                inputs.append(port.name())
            for port in self.node.output_ports():
                outputs.append(port.name())
        return inputs, outputs

    def _show_help_dialog(self, help_text: str) -> None:
        """Show a scrollable help dialog instead of an always-visible sidebar."""
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Function Block Help")
        dialog.resize(780, 620)
        layout = QtWidgets.QVBoxLayout(dialog)

        text = QtWidgets.QTextEdit()
        text.setReadOnly(True)
        text.setPlainText(help_text)
        text.setFont(QFont("Consolas", 10))
        layout.addWidget(text)

        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        dialog.exec_()

    def _build_help_text(self) -> str:
        input_names, output_names = self._current_port_names()
        inputs = ", ".join(input_names) if input_names else "(no input ports yet)"
        outputs = ", ".join(output_names) if output_names else "(no output ports yet)"

        command_lines = []
        for display, snippet, tooltip in self._CAD_COMMANDS:
            if snippet is None:
                command_lines.append("")
                command_lines.append(display)
                continue
            command_lines.append(f"{display}")
            command_lines.append(f"    insert: {snippet}")
            if tooltip:
                command_lines.extend(f"    {line}" for line in tooltip.splitlines())

        return "\n".join([
            "FUNCTION BLOCK CODE EDITOR",
            "",
            "Current ports",
            f"  Inputs : {inputs}",
            f"  Outputs: {outputs}",
            "",
            "Basics",
            "  - Use input ports as normal Python variables.",
            "  - Assign every connected output variable in the code body.",
            "  - Do not write return statements; PyLCSS adds returns from the output ports.",
            "  - numpy is available as np, math is imported, and the Design Studio runtime is available as cad.",
            "",
            "Design Studio connection",
            "  1. Build and save a .cad graph in Design Studio.",
            "  2. Add the terminal solver node you want to call: FEA Solver, Crash Solver, or Topology Opt.",
            "  3. Open the Simulation tab in this editor and select the saved .cad file.",
            "  4. Connect block inputs to geometry, material, mesh, load, impact, or solver controls.",
            "  5. PyLCSS creates the cad.fea(...), cad.crash(...), or cad.topopt(...) call for you.",
            "  6. The call returns CadResult with stable scalar fields plus raw solver data through raw().",
            "",
            "Examples",
            '  r = cad.fea("bracket.cad", thickness=t, hole_r=hr)',
            '  stress, mass = r.pick("max_stress", "mass")',
            '  safety_factor = yield_strength / max(stress, 1e-9)',
            "",
            '  c = cad.topopt("mbb_beam.cad", vol_frac=0.35).compliance',
            "",
            "Result access",
            "  - Attribute: r.max_stress",
            '  - Dict style: r["mass"]',
            '  - Tuple helper: stress, mass = r.pick("max_stress", "mass")',
            "  - Raw solver dict: raw = r.raw()",
            "",
            "CAD commands and standard fields",
            *command_lines,
            "",
            "Notes",
            "  - Repeated cad.* calls are cached by .cad path, file mtime, solver kind, and inputs.",
            "  - Relative .cad paths resolve from the current PyLCSS process working directory.",
            "  - For topology optimization, volume is retained material volume; total_volume is the original design-domain volume.",
        ])

    def show_help(self) -> None:
        self._show_help_dialog(self._build_help_text())
        
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
            # Use QPlainTextEdit's find() method
            found = self.editor.find(text)
            if not found:
                # Wrap around from beginning
                cursor = self.editor.textCursor()
                cursor.movePosition(QtGui.QTextCursor.Start)
                self.editor.setTextCursor(cursor)
                self.editor.find(text)
                
    def replace_text(self):
        text = self.find_edit.text()
        replace = self.replace_edit.text()
        cursor = self.editor.textCursor()
        if text and cursor.hasSelection():
            cursor.insertText(replace)
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
    NODE_NAME = 'Function / Discipline'

    def __init__(self) -> None:
        """Initialize the custom block node with default ports and code editor."""
        super(CustomBlockNode, self).__init__()
        # Role colour — neutral gray body with violet border for
        # compute/function blocks.
        self.set_color(35, 35, 35)
        self.set_property('border_color', (148, 107, 220, 255), push_undo=False)
        self.set_property('text_color', (242, 238, 250, 255), push_undo=False)
        self.set_port_deletion_allowed(True)
        # Start with default ports
        self.add_input('in_1')
        self.add_output('out_1')
        if not self.has_property('func_name'):
            self.create_property('func_name', '')
        # Create properties for number of inputs/outputs
        self.create_property('num_inputs', '1')
        self.create_property('num_outputs', '1')

        self.create_property('surrogate_model_path', '')
        self.create_property('surrogate_status', 'Not Trained')
        self.create_property('surrogate_train_trigger', 0.0)
        self.create_property('surrogate_controls', False)
        self.create_property('code_content', "# out_1 = in_1 * 2\n")

        self.add_text_input('interface_summary', 'Interface', text='1 input → 1 output')
        self.add_text_input('execution_summary', 'Execution', text='Python function')
        for widget_name in ('interface_summary', 'execution_summary'):
            widget = self.get_widget(widget_name)
            if widget:
                widget.get_custom_widget().setReadOnly(True)

        # Button widgets do not define model properties themselves, while
        # NodeGraphQt still asks for those properties during serialization.
        self.create_property('edit_function', 'Edit function…')
        self.create_property('manage_ports', 'Manage ports…')
        self.create_property('train_surrogate', 'Train surrogate')
        self.add_button(
            'edit_function', text='Edit function…',
            tooltip='Open the function editor, variables, and simulation coupling panel',
        )
        self.add_button(
            'manage_ports', text='Manage ports…',
            tooltip='Add or remove named function inputs and outputs',
        )
        self.add_checkbox(
            'use_surrogate', '', text='Use trained surrogate', state=False,
            tooltip='Replace direct evaluation with the trained surrogate model',
        )
        self.add_button(
            'train_surrogate', text='Train surrogate',
            tooltip='Train a surrogate model for this function block',
            tab='Surrogate',
        )
        self.get_widget('edit_function').get_custom_widget().clicked.connect(self._open_code_editor)
        self.get_widget('manage_ports').get_custom_widget().clicked.connect(self._open_port_manager)
        self.get_widget('train_surrogate').get_custom_widget().clicked.connect(self._train_surrogate)
        self.get_widget('train_surrogate').setVisible(False)
        self._refresh_function_summary()

    def _open_code_editor(self):
        parent = self.graph.widget if self.graph else None
        dialog = CodeEditorDialog(
            self.get_property('code_content') or '', node=self, parent=parent
        )
        if dialog.exec_() == QtWidgets.QDialog.Accepted:
            self.set_property('code_content', dialog.get_code())

    def _open_port_manager(self):
        parent = self.graph.widget if self.graph else None
        dialog = PortManagerDialog(self, parent)
        dialog.exec_()
        self._refresh_function_summary()

    def _train_surrogate(self):
        import time
        self.set_property('surrogate_train_trigger', time.time())

    def _refresh_function_summary(self):
        if not self.has_property('interface_summary'):
            return
        num_inputs = len(self.input_ports())
        num_outputs = len(self.output_ports())
        input_word = 'input' if num_inputs == 1 else 'inputs'
        output_word = 'output' if num_outputs == 1 else 'outputs'
        summary = f"{num_inputs} {input_word} → {num_outputs} {output_word}"
        super(CustomBlockNode, self).set_property('interface_summary', summary, push_undo=False)

        code = str(self.get_property('code_content') or '')
        if 'cad.crash(' in code:
            execution = 'Design Studio · Crash'
        elif 'cad.fea(' in code:
            execution = 'Design Studio · Static FEA'
        elif 'cad.topopt(' in code:
            execution = 'Design Studio · Topology optimization'
        else:
            execution = 'Python function'
        if self.get_property('use_surrogate'):
            status = str(self.get_property('surrogate_status') or 'Not trained')
            execution = f"Surrogate · {status}"
        super(CustomBlockNode, self).set_property('execution_summary', execution, push_undo=False)
        train_widget = self.get_widget('train_surrogate')
        if train_widget is not None:
            should_show = bool(self.get_property('use_surrogate'))
            if train_widget.isVisible() != should_show:
                train_widget.setVisible(should_show)
                try:
                    self.view.draw_node()
                except Exception:
                    pass

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
        self.get_property(name) if self.has_property(name) else None

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
            
        
        if name in ('code_content', 'use_surrogate', 'surrogate_status'):
            self._refresh_function_summary()

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
        self._refresh_function_summary()

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
        self._refresh_function_summary()


class SimulationFunctionNode(CustomBlockNode):
    """Managed function block created directly from a Design Studio study."""

    __identifier__ = 'com.pfd.custom_block.simulation'
    NODE_NAME = 'Design Studio Simulation'

    def __init__(self) -> None:
        super(SimulationFunctionNode, self).__init__()
        # Amber identifies a live Design Studio analysis while keeping the
        # neutral function-node body used throughout the system graph.
        self.set_color(35, 35, 35)
        self.set_property('border_color', (210, 153, 34, 255), push_undo=False)
        self.set_property('text_color', (255, 248, 230, 255), push_undo=False)
        self.create_property('simulation_project', '')
        self.create_property('simulation_kind', 'fea')
        self.create_property('simulation_interface', '')
        self.create_property('simulation_bridge_version', 1)
        self.add_text_input(
            'study_summary', 'Study', text='Design Studio · Not configured',
            tooltip='Saved Design Studio project and analysis used by this function',
        )
        study_widget = self.get_widget('study_summary')
        if study_widget:
            study_widget.get_custom_widget().setReadOnly(True)

        # Ports are governed by the selected study interface. They can still
        # be changed by re-exporting the study, but ad-hoc deletion would make
        # the generated adapter and metadata disagree.
        manage_widget = self.get_widget('manage_ports')
        if manage_widget:
            manage_widget.setVisible(False)
        edit_widget = self.get_widget('edit_function')
        if edit_widget:
            button = edit_widget.get_custom_widget()
            button.setText('Inspect adapter…')
            button.setToolTip(
                'Inspect the generated Design Studio adapter or configure surrogate execution'
            )

    def configure_from_spec(self, spec) -> None:
        """Replace the managed interface with a validated bridge spec."""

        from pylcss.system_modeling.design_studio_bridge import (
            generate_simulation_function_code,
            validate_simulation_spec,
        )

        validate_simulation_spec(spec)
        for port in list(self.input_ports()):
            for connected in list(port.connected_ports()):
                port.disconnect_from(connected)
            self.delete_input(port.name())
        for port in list(self.output_ports()):
            for connected in list(port.connected_ports()):
                port.disconnect_from(connected)
            self.delete_output(port.name())

        for item in spec.inputs:
            self.add_input(item.port_name)
        for item in spec.outputs:
            self.add_output(item.port_name)

        # Set counts only after installing the named ports so the inherited
        # count synchronizer sees an already-correct interface.
        self.set_property('num_inputs', str(len(spec.inputs)), push_undo=False)
        self.set_property('num_outputs', str(len(spec.outputs)), push_undo=False)
        self.set_property(
            'code_content', generate_simulation_function_code(spec), push_undo=False
        )
        self.set_property('simulation_project', spec.project_path, push_undo=False)
        self.set_property('simulation_kind', spec.analysis_kind, push_undo=False)
        self.set_property('simulation_interface', spec.to_json(), push_undo=False)
        self.set_property('simulation_bridge_version', spec.version, push_undo=False)
        self.set_name(spec.node_name)

        analysis_labels = {
            'fea': 'Static FEA',
            'crash': 'Crash simulation',
            'topopt': 'Topology optimization',
        }
        study_name = Path(spec.project_path).stem
        self.set_property(
            'study_summary',
            f"{study_name} · {analysis_labels.get(spec.analysis_kind, spec.analysis_kind)}",
            push_undo=False,
        )
        self._refresh_function_summary()


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
        # Role colour — neutral gray body with cyan border for
        # design variables.
        self.set_color(35, 35, 35)
        self.set_property('border_color', (55, 177, 224, 255), push_undo=False)
        self.set_property('text_color', (235, 247, 252, 255), push_undo=False)
        self.set_port_deletion_allowed(True)
        self.add_output('x')
        self.add_text_input('var_name', 'Name', text='x',
                            tooltip='Variable name used in equations and function ports')
        self.add_text_input('unit', 'Unit', text='-',
                            tooltip='Engineering unit displayed in results')
        self.add_text_input('min', 'Lower bound', text='0.0',
                            tooltip='Smallest value the optimizer may use')
        self.add_text_input('max', 'Upper bound', text='10.0',
                            tooltip='Largest value the optimizer may use')
        self.set_name('x')

    def set_property(self, name, value, push_undo=True):
        """
        Handle property changes for input node configuration.

        Args:
            name: Property name being changed
            value: New property value
            push_undo: Whether to push change to undo stack
        """
        super(InputNode, self).set_property(name, value, push_undo)

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
        # Role colour — neutral gray body with teal-green border for
        # quantities of interest (graph outputs).
        self.set_color(35, 35, 35)
        self.set_property('border_color', (63, 190, 143, 255), push_undo=False)
        self.set_property('text_color', (235, 250, 244, 255), push_undo=False)
        self.set_port_deletion_allowed(True)
        self.add_input('y')
        self.add_text_input('var_name', 'Name', text='y',
                            tooltip='Result name used by downstream analysis and optimization')
        self.add_text_input('unit', 'Unit', text='-',
                            tooltip='Engineering unit for this result')
        self.add_text_input('req_min', 'Allowed min', text='-1e9',
                            tooltip='Feasibility lower limit; use -inf for no lower limit')
        self.add_text_input('req_max', 'Allowed max', text='1e9',
                            tooltip='Feasibility upper limit; use inf for no upper limit')
        self.create_property('minimize', False)
        self.create_property('maximize', False)
        self.create_property('show_in_legend', True)
        self.add_combo_menu(
            'objective_mode', 'Optimization role',
            items=['Constraint only', 'Minimize', 'Maximize'],
            tooltip='Use as a feasibility constraint or as an optimization objective',
        )
        self.set_name('y')

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
        
        if name == 'objective_mode':
            self.set_property('minimize', value == 'Minimize', push_undo)
            self.set_property('maximize', value == 'Maximize', push_undo)
        elif name == 'minimize' and value:
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
        if name in ('minimize', 'maximize') and hasattr(self, 'view'):
            objective = 'Constraint only'
            if self.get_property('minimize'):
                objective = 'Minimize'
            elif self.get_property('maximize'):
                objective = 'Maximize'
            if self.get_property('objective_mode') != objective:
                super(OutputNode, self).set_property('objective_mode', objective, push_undo)

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
        # Role colour — neutral gray body with slate border for
        # pass-through/intermediate variables.
        self.set_color(35, 35, 35)
        self.set_property('border_color', (118, 130, 150, 255), push_undo=False)
        self.set_property('text_color', (238, 241, 246, 255), push_undo=False)
        self.set_port_deletion_allowed(True)
        self.add_input('z')
        self.add_output('z')
        self.add_text_input('var_name', 'Name', text='z')
        self.add_text_input('unit', 'Unit', text='-')


_SYSTEM_NODE_STYLES = {
    'com.pfd.input': ((35, 35, 35, 255), (55, 177, 224, 255), (235, 247, 252, 255)),
    'com.pfd.custom_block': ((35, 35, 35, 255), (148, 107, 220, 255), (242, 238, 250, 255)),
    'com.pfd.custom_block.simulation': ((35, 35, 35, 255), (210, 153, 34, 255), (255, 248, 230, 255)),
    'com.pfd.output': ((35, 35, 35, 255), (63, 190, 143, 255), (235, 250, 244, 255)),
    'com.pfd.intermediate': ((35, 35, 35, 255), (118, 130, 150, 255), (238, 241, 246, 255)),
}

_SYSTEM_NODE_TOOLTIPS = {
    'com.pfd.input': (
        '<b>Design Variable</b><br/>An optimizer-controlled input. '
        'Define its engineering unit and lower/upper design bounds.'
    ),
    'com.pfd.custom_block': (
        '<b>Function / Discipline</b><br/>Transforms inputs into outputs using Python '
        'or a linked Design Studio FEA, crash, or TopOpt study. Double-click to edit.'
    ),
    'com.pfd.custom_block.simulation': (
        '<b>Design Studio Simulation</b><br/>A managed engineering function linked '
        'to a saved Design Studio analysis. Amber means it executes the live study.'
    ),
    'com.pfd.output': (
        '<b>Quantity of Interest</b><br/>A model result used as a feasibility '
        'constraint, minimization objective, or maximization objective.'
    ),
    'com.pfd.intermediate': (
        '<b>Intermediate Variable</b><br/>Routes and renames a value between disciplines.'
    ),
}


def apply_system_node_style(node) -> None:
    """Apply role colours and refresh compact widgets after deserialization."""
    identifier = getattr(node, '__identifier__', '')
    style = _SYSTEM_NODE_STYLES.get(identifier)
    if not style:
        return
    color, border, text = style
    node.set_property('color', color, push_undo=False)
    node.set_property('border_color', border, push_undo=False)
    node.set_property('text_color', text, push_undo=False)
    if identifier == 'com.pfd.output' and node.has_property('objective_mode'):
        objective = 'Constraint only'
        if node.get_property('minimize'):
            objective = 'Minimize'
        elif node.get_property('maximize'):
            objective = 'Maximize'
        node.set_property('objective_mode', objective, push_undo=False)
    controls = getattr(node, 'controls_widget', None)
    if controls is not None:
        controls.sync_all()
    code_widget = getattr(node, 'code_widget', None)
    if code_widget is not None:
        code_widget.refresh_summary()
    refresh_function = getattr(node, '_refresh_function_summary', None)
    if callable(refresh_function):
        refresh_function()
    try:
        node.view.draw_node()
        node.view.setToolTip(_SYSTEM_NODE_TOOLTIPS.get(identifier, ''))
    except Exception:
        pass








