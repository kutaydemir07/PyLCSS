# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE.
"""Code editor and port-management dialogs for system-modeling nodes."""

from __future__ import annotations

import json
import re
from pathlib import Path

from NodeGraphQt import BaseNode
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtGui import (
    QColor,
    QFont,
    QSyntaxHighlighter,
    QTextCharFormat,
)

from pylcss.user_interface.common.theme_manager import THEMES, current_theme

__all__ = [
    "CodeEditor",
    "CodeEditorDialog",
    "LineNumberArea",
    "PortListWidget",
    "PortManagerDialog",
    "PythonHighlighter",
]


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
        super().__init__()
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
        ports = (
            self.node.input_ports()
            if self.port_type == "input"
            else self.node.output_ports()
        )
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
                QtWidgets.QMessageBox.warning(
                    self, "Duplicate Variable", f"A port named '{name}' already exists."
                )
                return
            try:
                if self.port_type == "input":
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
                if self.port_type == "input":
                    self.node.delete_input(name)
                else:
                    self.node.delete_output(name)
                self._sync_port_count()
                self.refresh_list()
            except Exception as e:
                QtWidgets.QMessageBox.warning(self, "Error", str(e))

    def _sync_port_count(self) -> None:
        """Keep the node's serialized count properties in step with the list."""
        property_name = "num_inputs" if self.port_type == "input" else "num_outputs"
        ports = (
            self.node.input_ports()
            if self.port_type == "input"
            else self.node.output_ports()
        )
        if hasattr(self.node, "has_property") and self.node.has_property(property_name):
            self.node.set_property(property_name, str(len(ports)))


class PortManagerDialog(QtWidgets.QDialog):
    """
    Dialog for managing ports on custom nodes.

    Provides a tabbed interface for managing input and output ports
    separately, using PortListWidget for each port type.
    """

    def __init__(self, node: BaseNode, parent: QtWidgets.QWidget | None = None) -> None:
        """
        Initialize the port manager dialog.

        Args:
            node: The node whose ports are being managed
            parent: Parent widget
        """
        super().__init__(parent)
        self.node = node
        self.setWindowTitle("Manage Ports")
        self.resize(400, 300)

        layout = QtWidgets.QVBoxLayout(self)

        # Tabs for Inputs / Outputs
        tabs = QtWidgets.QTabWidget()
        self.input_tab = PortListWidget(node, "input")
        self.output_tab = PortListWidget(node, "output")

        tabs.addTab(self.input_tab, "Inputs")
        tabs.addTab(self.output_tab, "Outputs")
        layout.addWidget(tabs)

        btn_box = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)


class PythonHighlighter(QSyntaxHighlighter):
    def __init__(
        self, document: QtGui.QTextDocument, variables: list[str] | None = None
    ) -> None:
        super().__init__(document)
        self.variables = variables or []

        # Keywords
        self.keyword_format = QTextCharFormat()
        self.keyword_format.setFontWeight(QFont.Bold)
        keywords = [
            "and",
            "as",
            "assert",
            "break",
            "class",
            "continue",
            "def",
            "del",
            "elif",
            "else",
            "except",
            "finally",
            "for",
            "from",
            "global",
            "if",
            "import",
            "in",
            "is",
            "lambda",
            "nonlocal",
            "not",
            "or",
            "pass",
            "raise",
            "return",
            "try",
            "while",
            "with",
            "yield",
        ]
        self.keyword_patterns = [r"\b" + re.escape(word) + r"\b" for word in keywords]

        # Variables
        self.variable_format = QTextCharFormat()
        self.variable_format.setFontWeight(QFont.Bold)

        # Strings
        self.string_format = QTextCharFormat()

        # Comments
        self.comment_format = QTextCharFormat()
        self.apply_theme(current_theme())

    def apply_theme(self, theme_name: str) -> None:
        """Use syntax colors with strong contrast on the active editor surface."""
        if str(theme_name).lower() == "light":
            colors = {
                "keyword": "#6d5f8c",
                "variable": THEMES["light"]["success"],
                "string": "#756448",
                "comment": THEMES["light"]["text_dim"],
            }
        else:
            colors = {
                "keyword": "#ff79c6",
                "variable": "#50fa7b",
                "string": "#f1fa8c",
                "comment": "#a8c7ff",
            }
        self.keyword_format.setForeground(QColor(colors["keyword"]))
        self.variable_format.setForeground(QColor(colors["variable"]))
        self.string_format.setForeground(QColor(colors["string"]))
        self.comment_format.setForeground(QColor(colors["comment"]))
        self.rehighlight()

    def highlightBlock(self, text: str) -> None:
        # Keywords
        for pattern in self.keyword_patterns:
            for match in re.finditer(pattern, text):
                self.setFormat(
                    match.start(), match.end() - match.start(), self.keyword_format
                )

        # Variables
        if self.variables:
            for var in self.variables:
                pattern = r"\b" + re.escape(var) + r"\b"
                for match in re.finditer(pattern, text):
                    self.setFormat(
                        match.start(), match.end() - match.start(), self.variable_format
                    )

        # Strings
        string_pattern = r'(["\'])(?:(?=(\\?))\2.)*?\1'
        for match in re.finditer(string_pattern, text):
            self.setFormat(
                match.start(), match.end() - match.start(), self.string_format
            )

        # Comments
        comment_pattern = r"#.*"
        for match in re.finditer(comment_pattern, text):
            self.setFormat(
                match.start(), match.end() - match.start(), self.comment_format
            )


class CodeEditor(QtWidgets.QPlainTextEdit):
    """
    Enhanced code editor with Python syntax highlighting and line numbers.

    Custom implementation using QPlainTextEdit with:
    - Python syntax highlighting
    - Line numbers
    - Dark theme
    """

    def __init__(self, variables: list[str] | None = None) -> None:
        """Initialize the code editor with Python syntax highlighting."""
        super().__init__()
        self.variables = variables or []

        # Set font
        font = QFont("Consolas", 10)
        font.setStyleHint(QFont.Monospace)
        self.setFont(font)

        # Set tab stop distance (approx 4 spaces)
        metrics = self.fontMetrics()
        self.setTabStopDistance(metrics.horizontalAdvance(" ") * 4)

        self.apply_theme(current_theme())

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
            self.line_number_area.update(
                0, rect.y(), self.line_number_area.width(), rect.height()
            )
        if rect.contains(self.viewport().rect()):
            self.update_line_number_area_width(0)

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        cr = self.contentsRect()
        self.line_number_area.setGeometry(
            QtCore.QRect(
                cr.left(), cr.top(), self.line_number_area_width(), cr.height()
            )
        )

    def changeEvent(self, event: QtCore.QEvent) -> None:
        super().changeEvent(event)
        if event.type() in (
            QtCore.QEvent.PaletteChange,
            QtCore.QEvent.ApplicationPaletteChange,
        ) and hasattr(self, "highlighter"):
            # Reapplying QSS from inside PaletteChange emits another
            # PaletteChange on Qt 6.10. The dialog itself owns the editor
            # surface colors; only refresh document formats here.
            self.highlighter.apply_theme(current_theme())
            self.line_number_area.update()

    def apply_theme(self, theme_name: str) -> None:
        """Keep the editor surface and syntax colors on the active palette."""
        theme = THEMES[str(theme_name).lower()]
        self.setStyleSheet(
            "QPlainTextEdit {"
            f"background-color: {theme['bg_panel']};"
            f"color: {theme['text_main']};"
            f"border: 1px solid {theme['border']};"
            f"selection-background-color: {theme['primary']};"
            "selection-color: white;"
            "}"
        )
        if hasattr(self, "highlighter"):
            self.highlighter.apply_theme(theme_name)
        if hasattr(self, "line_number_area"):
            self.line_number_area.update()

    def update_variables(self, variables: list[str]) -> None:
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
        space = 3 + self.fontMetrics().horizontalAdvance("9") * digits
        return space

    def line_number_area_paint_event(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self.line_number_area)
        if not painter.isActive():
            return
        painter.fillRect(
            event.rect(),
            self.palette().color(QtGui.QPalette.AlternateBase),
        )
        block = self.firstVisibleBlock()
        block_number = block.blockNumber()
        top = self.blockBoundingGeometry(block).translated(self.contentOffset()).top()
        bottom = top + self.blockBoundingRect(block).height()

        while block.isValid() and top <= event.rect().bottom():
            if block.isVisible() and bottom >= event.rect().top():
                number = str(block_number + 1)
                painter.setPen(self.palette().color(QtGui.QPalette.Text))
                painter.drawText(
                    0,
                    int(top),
                    self.line_number_area.width(),
                    self.fontMetrics().height(),
                    QtCore.Qt.AlignRight,
                    number,
                )
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
    @staticmethod
    def _editor_stylesheet(theme_name: str) -> str:
        """Build the dialog QSS from the live theme instead of import-time colors."""
        colors = THEMES[str(theme_name).lower()]
        return f"""
        QDialog {{ background: {colors["bg_dark"]}; }}
        QWidget#functionEditorPanel {{ background: {colors["bg_dark"]}; }}
        QWidget#functionBlockInspector {{
            background: {colors["bg_panel"]};
            border-left: 1px solid {colors["border"]};
        }}
        QLabel {{ color: {colors["text_main"]}; }}
        QGroupBox {{
            background: {colors["bg_panel"]}; border: 1px solid {colors["border"]};
            border-radius: 8px; margin-top: 14px; padding: 12px 10px 10px 10px;
            font-weight: 600; color: {colors["text_main"]};
        }}
        QGroupBox::title {{
            subcontrol-origin: margin; subcontrol-position: top left;
            left: 10px; padding: 0 4px; color: {colors["primary"]}; font-weight: 700;
        }}
        QLineEdit {{
            background: {colors["bg_input"]}; border: 1px solid {colors["border"]};
            border-radius: 6px; padding: 5px 8px; color: {colors["text_main"]};
            selection-background-color: {colors["primary"]};
        }}
        QLineEdit:focus {{ border: 1px solid {colors["primary"]}; }}
        QPushButton {{
            background: {colors["bg_input"]}; border: 1px solid {colors["border"]};
            border-radius: 6px; padding: 6px 12px; color: {colors["text_main"]}; font-weight: 600;
        }}
        QPushButton:hover {{ border-color: {colors["primary"]}; }}
        QPushButton:pressed {{ background: {colors["bg_dark"]}; }}
        QTabWidget::pane {{
            background: {colors["bg_panel"]};
            border: 1px solid {colors["border"]}; border-radius: 8px; top: -1px;
        }}
        QTabBar::tab {{
            background: transparent; color: {colors["text_dim"]}; padding: 7px 16px;
            border: none; border-bottom: 2px solid transparent; font-weight: 600;
        }}
        QTabBar::tab:selected {{
            color: {colors["text_main"]}; border-bottom: 2px solid {colors["primary"]};
        }}
        QTabBar::tab:hover {{ color: {colors["text_main"]}; }}
        QListWidget#blockVariableList {{
            background: {colors["bg_panel"]}; color: {colors["text_main"]};
            border: 1px solid {colors["border"]};
        }}
        QListWidget#blockVariableList::item {{
            padding: 6px; border-bottom: 1px solid {colors["border"]};
        }}
        QListWidget#blockVariableList::item:selected {{
            background: {colors["primary"]}; color: white;
        }}
        QToolButton {{ background: transparent; color: {colors["text_dim"]}; border: none; }}
        QScrollArea {{ border: none; background: transparent; }}
    """

    def __init__(
        self,
        code: str,
        node: BaseNode | None = None,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.node = node
        self.setWindowTitle("Function Block Code Editor")
        self.resize(1100, 700)
        self.setStyleSheet(self._editor_stylesheet(current_theme()))
        self.showMaximized()

        main_layout = QtWidgets.QHBoxLayout(self)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)

        editor_panel = QtWidgets.QWidget()
        editor_panel.setObjectName("functionEditorPanel")
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

        btn_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        btn_box.accepted.connect(self.accept)
        btn_box.rejected.connect(self.reject)
        btn_layout.addWidget(btn_box)
        layout.addLayout(btn_layout)

        splitter.addWidget(editor_panel)

        inspector = QtWidgets.QWidget()
        inspector.setObjectName("functionBlockInspector")
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
        self.input_list.setObjectName("blockVariableList")
        self.input_list.setSpacing(1)
        self.input_list.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.input_list.itemDoubleClicked.connect(self.insert_variable)
        layout.addWidget(self.input_list)

        self.output_title = QtWidgets.QLabel("Outputs (0)")
        self.output_title.setStyleSheet("font-weight: bold; margin-top: 5px;")
        layout.addWidget(self.output_title)
        output_help = QtWidgets.QLabel(
            "Names that this code must assign before it finishes."
        )
        output_help.setStyleSheet("color: #b5bac1;")
        layout.addWidget(output_help)

        self.output_list = QtWidgets.QListWidget()
        self.output_list.setObjectName("blockVariableList")
        self.output_list.setSpacing(1)
        self.output_list.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
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
        page.setSizePolicy(
            QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Preferred
        )
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
        self.project_edit.setSizePolicy(
            QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Fixed
        )
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
        self.solver_combo.setSizeAdjustPolicy(
            QtWidgets.QComboBox.AdjustToMinimumContentsLengthWithIcon
        )
        self.solver_combo.setMinimumContentsLength(18)
        self.solver_combo.setSizePolicy(
            QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Fixed
        )
        self.solver_combo.addItem("Static FEA — stress, displacement and mass", "fea")
        self.solver_combo.addItem(
            "Impact — explicit-dynamics response and absorbed energy", "impact"
        )
        self.solver_combo.addItem(
            "Topology / lattice optimization — optimized material layout",
            "topopt",
        )
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
        self.advanced_mapping_toggle.setToolButtonStyle(
            QtCore.Qt.ToolButtonTextBesideIcon
        )
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

        output_help = QtWidgets.QLabel(
            "Choose the simulation result for each block output:"
        )
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
        "impact": (
            ("Maximum stress", "max_stress"),
            ("Maximum displacement", "peak_disp"),
            ("Absorbed energy", "absorbed_energy"),
            ("Absorbed energy (kJ)", "absorbed_energy_kj"),
            ("Peak crushing force", "peak_force"),
            ("Mean crushing force", "mean_force"),
            ("Crush force efficiency", "crush_force_efficiency"),
            ("Specific energy absorption", "specific_energy_absorption"),
            ("Useful crush distance", "crush_distance"),
            ("Peak acceleration", "peak_acceleration_g"),
            ("Velocity change", "delta_v"),
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
        "impact": "Runs one explicit impact solve. Impact simulations can be slow when the function block is evaluated many times.",
        "topopt": "Runs a complete topology or lattice optimization each time the function block is evaluated. Usually use this to create a design once, or train a surrogate before the main optimization.",
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
                self.project_edit.setText(
                    QtCore.QDir.toNativeSeparators(str(current_file))
                )
                self._inspect_design_studio_project()
                return
        self.project_status.setText(
            "The file open in Design Studio has not been saved yet. Save it first, then try again."
        )
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
            self.project_status.setText(
                "This file cannot be found. Select an existing .cad file."
            )
            self.project_status.setStyleSheet("color: #ed4245;")
            self._refresh_coupling_mappings()
            return

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            nodes = data.get("nodes", {})
            from pylcss.design_studio.runtime import (
                discover_exposed_parameters,
                discover_override_controls,
            )

            self._studio_controls = discover_override_controls(data)
            parameters = set(discover_exposed_parameters(data))
            solvers = set()
            for node_data in nodes.values():
                node_type = str(node_data.get("type_", "")).lower()

                if "crash_solver" in node_type:
                    solvers.add("impact")
                elif (
                    "topopt" in node_type
                    or "topology" in node_type
                    or "lattice" in node_type
                ):
                    # Both density studies are reached through `cad.topopt`.
                    solvers.add("topopt")
                elif ".sim.solver" in node_type or node_type.endswith(".solvernode"):
                    solvers.add("fea")

            self._studio_parameters = parameters
            solver_names = {
                "fea": "Static FEA",
                "impact": "Impact",
                "topopt": "Topology / lattice optimization",
            }
            details = []
            if solvers:
                details.append(
                    "analysis: "
                    + ", ".join(
                        solver_names[k]
                        for k in ("fea", "impact", "topopt")
                        if k in solvers
                    )
                )
                detected = next(
                    (k for k in ("fea", "impact", "topopt") if k in solvers), None
                )
                index = self.solver_combo.findData(detected)
                if index >= 0:
                    self.solver_combo.setCurrentIndex(index)
            else:
                details.append("no solver node found")
            if parameters:
                details.append(f"{len(parameters)} geometry parameter(s)")
            material_names = sorted(
                {
                    item["node"]
                    for item in self._studio_controls
                    if item["group"] == "Material"
                }
            )
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
            self.project_status.setText(
                f"Could not read this Design Studio file: {exc}"
            )
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
            target_combo.setSizeAdjustPolicy(
                QtWidgets.QComboBox.AdjustToMinimumContentsLengthWithIcon
            )
            target_combo.setMinimumContentsLength(15)
            target_combo.setSizePolicy(
                QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Fixed
            )
            if not parameters and not controls:
                target_combo.addItem(
                    f"Geometry parameter / {name}", ("parameter", name)
                )
            elif name in parameters:
                target_combo.addItem(
                    f"Geometry parameter / {name}", ("parameter", name)
                )
            else:
                target_combo.addItem("Choose a Design Studio control…", None)

            target_combo.addItem(
                "Do not send this input to Design Studio", ("ignore", "")
            )
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
            result_combo.setSizeAdjustPolicy(
                QtWidgets.QComboBox.AdjustToMinimumContentsLengthWithIcon
            )
            result_combo.setMinimumContentsLength(15)
            result_combo.setSizePolicy(
                QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Fixed
            )
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
            self.input_summary_label.setText(
                "This block has no inputs to pass to Design Studio."
            )
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
            self.coupling_status.setText(
                "Select a saved .cad project before inserting the adapter."
            )
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
        before_cursor = self.editor.toPlainText()[: cursor.position()]
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
            "cad.impact(path, **inputs)",
            'cad.impact("file.cad", param=value)',
            "Run OpenRadioss explicit dynamics for an impact study in a .cad graph file.\n"
            "Args:  cad_path (str), then **inputs matched against\n"
            "       'exposed_name' on Number/Variable nodes, or named\n"
            "       Code Part parameters in the graph.\n"
            "Returns CadResult — see the 'Impact result' rows below.",
        ),
        (
            "cad.topopt(path, **inputs)",
            'cad.topopt("file.cad", param=value)',
            "Run SIMP topology optimization through a .cad graph file.\n"
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
            "Tuple-unpack helper:\n    s, m = cad.fea(...).pick('max_stress', 'mass')",
        ),
        (
            'result["key"]',
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
        (
            ".max_stress",
            ".max_stress",
            "Peak Von Mises stress at Gauss points [MPa].\n"
            "Conservative (un-smoothed) — use for safety-factor calcs.",
        ),
        (
            ".compliance",
            ".compliance",
            "Compliance C = u·f = u·K·u [N·mm].\n"
            "Smaller = stiffer.  C = 2 × total elastic strain energy.",
        ),
        (
            ".strain_energy",
            ".strain_energy",
            "Total elastic strain energy ∫ ½σ:ε dV  [N·mm].\n"
            "Equals compliance / 2 for linear elastic materials.",
        ),
        (
            ".mass",
            ".mass",
            "Total mass [t in the standard mm/t/s unit system].\n"
            "= volume × material.rho from the connected MaterialNode.",
        ),
        (".volume", ".volume", "Total mesh volume Σ V_e  [mm³]."),
        (".peak_disp", ".peak_disp", "Max |u| across all mesh nodes [mm]."),
        # ── Impact standard fields ─────────────────────────────────────
        ("— Impact result (cad.impact) —", None, None),
        (
            ".max_stress",
            ".max_stress",
            "Peak Von Mises stress over the whole transient [MPa].",
        ),
        (
            ".peak_disp",
            ".peak_disp",
            "Max nodal displacement magnitude over the whole transient [mm].",
        ),
        (
            ".absorbed_energy",
            ".absorbed_energy",
            "Plastic dissipation Σ_e ∫ σ_y · dε_p · V_e [N·mm].\n"
            "Standard crashworthiness metric — bigger = more energy soaked up.",
        ),
        (
            ".n_failed",
            ".n_failed",
            "Number of elements deleted via the failure criterion.",
        ),
        # ── TopOpt standard fields ─────────────────────────────────────
        ("— TopOpt result (cad.topopt) —", None, None),
        (
            ".final_vol_frac",
            ".final_vol_frac",
            "Volume-weighted final physical density fraction; should match the\n"
            "vol_frac target set on the Topology Opt node.",
        ),
        (".compliance", ".compliance", "Compliance at the final density field [N·mm]."),
        (".mass", ".mass", "Effective mass = Σ ρ_e · V_e · material.rho  [t]."),
        (
            ".volume",
            ".volume",
            "Retained material volume after topology optimization [mm3].",
        ),
        (
            ".total_volume",
            ".total_volume",
            "Original design-domain mesh volume before material removal [mm3].",
        ),
    )

    def _refresh_var_list(self) -> None:
        """Refresh the visible block contract and syntax highlighting."""
        inputs, outputs = self._current_port_names()
        self.editor.update_variables(inputs + outputs)

        if hasattr(self, "input_list"):
            self.input_list.clear()
            input_color = (
                THEMES["light"]["primary"]
                if current_theme() == "light"
                else "#38c8e8"
            )
            for name in inputs:
                item = QtWidgets.QListWidgetItem(f"IN     {name}")
                item.setData(QtCore.Qt.UserRole, name)
                item.setToolTip(f"Input variable · double-click to insert '{name}'")
                item.setForeground(QtGui.QColor(input_color))
                self.input_list.addItem(item)
            self.input_list.setFixedHeight(min(220, max(42, len(inputs) * 33 + 6)))

        if hasattr(self, "output_list"):
            self.output_list.clear()
            output_color = (
                THEMES["light"]["success"]
                if current_theme() == "light"
                else "#36c98f"
            )
            for name in outputs:
                item = QtWidgets.QListWidgetItem(f"OUT   {name}")
                item.setData(QtCore.Qt.UserRole, name)
                item.setToolTip(f"Output assignment · double-click to insert '{name}'")
                item.setForeground(QtGui.QColor(output_color))
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

        return "\n".join(
            [
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
                "  2. Add the terminal analysis node you want to call: Static Structural, Explicit Impact, Topology Optimization, or Lattice Optimization.",
                "  3. Open the Simulation tab in this editor and select the saved .cad file.",
                "  4. Connect block inputs to geometry, material, mesh, load, impact, or solver controls.",
                "  5. PyLCSS creates the cad.fea(...), cad.impact(...), or cad.topopt(...) call for you.",
                "  6. The call returns CadResult with stable scalar fields plus raw solver data through raw().",
                "",
                "Examples",
                '  r = cad.fea("bracket.cad", thickness=t, hole_r=hr)',
                '  stress, mass = r.pick("max_stress", "mass")',
                "  safety_factor = yield_strength / max(stress, 1e-9)",
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
            ]
        )

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
