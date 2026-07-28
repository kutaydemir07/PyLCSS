# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.
"""Professional Design Studio to Modeling Environment export dialog."""
from __future__ import annotations

from PySide6 import QtCore, QtWidgets

from pylcss.system_modeling.design_studio_bridge import (
    SimulationFunctionSpec,
    SimulationInputSpec,
    SimulationOutputSpec,
    StudyDescriptor,
    StudyInput,
    validate_simulation_spec,
)
from pylcss.user_interface.common.theme_manager import COLORS


_CREATE_NEW_SYSTEM = "__create_new_system__"


class DesignStudioBridgeDialog(QtWidgets.QDialog):
    """Choose the public interface of a managed Design Studio simulation node."""

    def __init__(
        self,
        descriptor: StudyDescriptor,
        system_names: list[str] | tuple[str, ...],
        current_system: str | None = None,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.descriptor = descriptor
        self._spec: SimulationFunctionSpec | None = None
        self.setWindowTitle("Create Modeling Function")
        self.setMinimumSize(980, 720)
        self.resize(1120, 800)
        self.setModal(True)
        self.setStyleSheet(self._style_sheet())
        self._build_ui(list(system_names), current_system)
        self._populate_inputs()
        self._populate_outputs()
        self._update_summary()

    @staticmethod
    def _style_sheet() -> str:
        return f"""
            QDialog {{ background: {COLORS['bg_dark']}; }}
            QLabel {{ color: {COLORS['text_dim']}; }}
            QLabel#BridgeTitle {{
                color: {COLORS['text_main']}; font-size: 20px; font-weight: 700;
            }}
            QLabel#BridgeEyebrow {{
                color: {COLORS['primary']}; font-size: 11px; font-weight: 700;
            }}
            QLabel#BridgeSummary {{
                color: {COLORS['text_main']}; background: {COLORS['bg_input']};
                border: 1px solid {COLORS['bg_panel']}; border-radius: 7px;
                padding: 8px 10px; font-weight: 600;
            }}
            QGroupBox {{
                color: {COLORS['text_main']}; background: {COLORS['bg_panel']};
                border: 1px solid {COLORS['bg_input']}; border-radius: 8px;
                margin-top: 13px; padding: 12px 9px 9px 9px; font-weight: 700;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin; left: 10px; padding: 0 5px;
                color: {COLORS['primary']};
            }}
            QLineEdit, QComboBox, QDoubleSpinBox {{
                color: {COLORS['text_main']}; background: {COLORS['bg_input']};
                border: 1px solid {COLORS['bg_dark']}; border-radius: 5px;
                padding: 5px 7px;
            }}
            QLineEdit:focus, QComboBox:focus, QDoubleSpinBox:focus {{
                border-color: {COLORS['primary']};
            }}
            QTableWidget {{
                color: {COLORS['text_main']}; background: {COLORS['bg_panel']};
                alternate-background-color: {COLORS['bg_dark']};
                gridline-color: {COLORS['bg_input']}; border: 1px solid {COLORS['bg_input']};
                border-radius: 6px; selection-background-color: {COLORS['bg_input']};
            }}
            QHeaderView::section {{
                color: {COLORS['text_dim']}; background: {COLORS['bg_input']};
                border: none; border-right: 1px solid {COLORS['bg_panel']};
                padding: 7px; font-weight: 700;
            }}
            QCheckBox {{ color: {COLORS['text_main']}; spacing: 7px; }}
            QPushButton {{
                color: {COLORS['text_main']}; background: {COLORS['bg_input']};
                border: 1px solid {COLORS['bg_panel']}; border-radius: 6px;
                padding: 7px 14px; font-weight: 700;
            }}
            QPushButton:hover {{ border-color: {COLORS['primary']}; }}
            QPushButton#PrimaryButton {{
                color: #111318; background: {COLORS['primary']};
                border-color: {COLORS['primary']};
            }}
            QPushButton#PrimaryButton:hover {{ background: {COLORS['primary_hover']}; }}
            QTabWidget::pane {{
                border: 1px solid {COLORS['bg_input']}; border-radius: 7px; top: -1px;
            }}
            QTabBar::tab {{
                color: {COLORS['text_dim']}; background: transparent;
                padding: 8px 18px; border-bottom: 2px solid transparent;
            }}
            QTabBar::tab:selected {{
                color: {COLORS['text_main']}; border-bottom-color: {COLORS['primary']};
            }}
        """

    def _build_ui(self, system_names: list[str], current_system: str | None) -> None:
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(22, 20, 22, 18)
        root.setSpacing(12)

        eyebrow = QtWidgets.QLabel("DESIGN STUDIO  →  MODELING ENVIRONMENT")
        eyebrow.setObjectName("BridgeEyebrow")
        root.addWidget(eyebrow)
        title = QtWidgets.QLabel("Create a simulation function")
        title.setObjectName("BridgeTitle")
        root.addWidget(title)
        intro = QtWidgets.QLabel(
            "Turn the saved engineering study into a reusable system-model block. "
            "The selected inputs become design variables; selected results become quantities of interest."
        )
        intro.setWordWrap(True)
        root.addWidget(intro)

        setup_group = QtWidgets.QGroupBox("Function setup")
        setup_layout = QtWidgets.QGridLayout(setup_group)
        setup_layout.setColumnStretch(1, 1)

        setup_layout.addWidget(QtWidgets.QLabel("Source study"), 0, 0)
        project_label = QtWidgets.QLabel(self.descriptor.path)
        project_label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        project_label.setToolTip(self.descriptor.path)
        project_label.setWordWrap(True)
        setup_layout.addWidget(project_label, 0, 1, 1, 3)

        setup_layout.addWidget(QtWidgets.QLabel("Analysis"), 1, 0)
        self.analysis_combo = QtWidgets.QComboBox()
        seen_kinds: set[str] = set()
        for analysis in self.descriptor.analyses:
            if analysis.kind in seen_kinds:
                continue
            seen_kinds.add(analysis.kind)
            self.analysis_combo.addItem(analysis.label, analysis.kind)
        self.analysis_combo.currentIndexChanged.connect(self._analysis_changed)
        setup_layout.addWidget(self.analysis_combo, 1, 1)

        setup_layout.addWidget(QtWidgets.QLabel("Destination"), 1, 2)
        self.system_combo = QtWidgets.QComboBox()
        for name in system_names:
            self.system_combo.addItem(name, name)
        self.system_combo.addItem("+ Create new system", _CREATE_NEW_SYSTEM)
        if current_system:
            current_index = self.system_combo.findData(current_system)
            if current_index >= 0:
                self.system_combo.setCurrentIndex(current_index)
        elif not system_names:
            self.system_combo.setCurrentIndex(self.system_combo.findData(_CREATE_NEW_SYSTEM))
        self.system_combo.currentIndexChanged.connect(self._destination_changed)
        setup_layout.addWidget(self.system_combo, 1, 3)

        setup_layout.addWidget(QtWidgets.QLabel("Function name"), 2, 0)
        self.node_name_edit = QtWidgets.QLineEdit(
            f"{self.descriptor.title} · {self.analysis_combo.currentText()}"
        )
        setup_layout.addWidget(self.node_name_edit, 2, 1)

        setup_layout.addWidget(QtWidgets.QLabel("New system name"), 2, 2)
        self.new_system_edit = QtWidgets.QLineEdit(f"{self.descriptor.title} Study")
        setup_layout.addWidget(self.new_system_edit, 2, 3)
        self._destination_changed()
        root.addWidget(setup_group)

        self.tabs = QtWidgets.QTabWidget()
        self.tabs.setDocumentMode(True)
        self.input_table = self._new_table(
            ["Use", "Port name", "Design Studio control", "Default", "Lower", "Upper", "Unit"]
        )
        self.output_table = self._new_table(
            ["Use", "Port name", "Simulation result", "Unit"]
        )
        self.tabs.addTab(self._table_page(
            self.input_table,
            "Geometry parameters are selected by default. Material, mesh, load and solver controls remain fixed unless selected here.",
        ), "Inputs")
        self.tabs.addTab(self._table_page(
            self.output_table,
            "Choose only the scalar results the system model needs. They can later be constraints, objectives or surrogate targets.",
        ), "Results")
        root.addWidget(self.tabs, 1)

        options_row = QtWidgets.QHBoxLayout()
        self.create_io_checkbox = QtWidgets.QCheckBox(
            "Create and connect Design Variable and Quantity of Interest nodes"
        )
        self.create_io_checkbox.setChecked(True)
        self.create_io_checkbox.setToolTip(
            "Disable this only when you want to wire the simulation block into an existing graph manually."
        )
        options_row.addWidget(self.create_io_checkbox)
        options_row.addStretch()
        self.summary_label = QtWidgets.QLabel()
        self.summary_label.setObjectName("BridgeSummary")
        options_row.addWidget(self.summary_label)
        root.addLayout(options_row)

        if self.descriptor.warnings:
            warning = QtWidgets.QLabel("  ".join(self.descriptor.warnings))
            warning.setWordWrap(True)
            warning.setStyleSheet("color: #f0b44d;")
            root.addWidget(warning)

        self.error_label = QtWidgets.QLabel("")
        self.error_label.setWordWrap(True)
        self.error_label.setStyleSheet(f"color: {COLORS['danger']};")
        root.addWidget(self.error_label)

        buttons = QtWidgets.QDialogButtonBox()
        cancel = buttons.addButton("Cancel", QtWidgets.QDialogButtonBox.RejectRole)
        create = buttons.addButton("Create Function", QtWidgets.QDialogButtonBox.AcceptRole)
        create.setObjectName("PrimaryButton")
        cancel.clicked.connect(self.reject)
        create.clicked.connect(self._accept_selection)
        root.addWidget(buttons)

    @staticmethod
    def _new_table(headers: list[str]) -> QtWidgets.QTableWidget:
        table = QtWidgets.QTableWidget(0, len(headers))
        table.setHorizontalHeaderLabels(headers)
        table.setAlternatingRowColors(True)
        table.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        table.verticalHeader().setVisible(False)
        table.verticalHeader().setDefaultSectionSize(38)
        table.horizontalHeader().setStretchLastSection(False)
        table.setShowGrid(False)
        return table

    @staticmethod
    def _table_page(table: QtWidgets.QTableWidget, help_text: str) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)
        layout.setContentsMargins(10, 10, 10, 10)
        help_label = QtWidgets.QLabel(help_text)
        help_label.setWordWrap(True)
        layout.addWidget(help_label)
        layout.addWidget(table, 1)
        return page

    @staticmethod
    def _check_widget(checked: bool) -> QtWidgets.QCheckBox:
        checkbox = QtWidgets.QCheckBox()
        checkbox.setChecked(checked)
        return checkbox

    @staticmethod
    def _number_widget(value: float) -> QtWidgets.QDoubleSpinBox:
        box = QtWidgets.QDoubleSpinBox()
        box.setDecimals(9)
        box.setRange(-1.0e18, 1.0e18)
        box.setValue(float(value))
        box.setKeyboardTracking(False)
        return box

    def _populate_inputs(self) -> None:
        table = self.input_table
        table.setRowCount(len(self.descriptor.inputs))
        for row, item in enumerate(self.descriptor.inputs):
            checkbox = self._check_widget(item.selected_by_default)
            checkbox.stateChanged.connect(self._update_summary)
            table.setCellWidget(row, 0, checkbox)

            name_edit = QtWidgets.QLineEdit(item.name)
            name_edit.textChanged.connect(self._update_summary)
            table.setCellWidget(row, 1, name_edit)

            source = QtWidgets.QTableWidgetItem(
                f"{item.group} / {item.label}"
            )
            source.setData(QtCore.Qt.UserRole, item)
            source.setToolTip(
                f"{item.source_kind}: {item.target}\nNode: {item.node_name or '-'}"
            )
            table.setItem(row, 2, source)
            table.setCellWidget(row, 3, self._number_widget(item.default))
            table.setCellWidget(row, 4, self._number_widget(item.lower))
            table.setCellWidget(row, 5, self._number_widget(item.upper))
            unit_edit = QtWidgets.QLineEdit(item.unit)
            table.setCellWidget(row, 6, unit_edit)

        header = table.horizontalHeader()
        header.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QtWidgets.QHeaderView.Stretch)
        for column in (3, 4, 5):
            header.setSectionResizeMode(column, QtWidgets.QHeaderView.ResizeToContents)
        header.setSectionResizeMode(6, QtWidgets.QHeaderView.ResizeToContents)

    def _populate_outputs(self) -> None:
        kind = self.analysis_combo.currentData()
        outputs = self.descriptor.outputs.get(kind, ())
        table = self.output_table
        table.setRowCount(len(outputs))
        for row, output in enumerate(outputs):
            checkbox = self._check_widget(output.selected_by_default)
            checkbox.stateChanged.connect(self._update_summary)
            table.setCellWidget(row, 0, checkbox)
            name_edit = QtWidgets.QLineEdit(output.field)
            name_edit.textChanged.connect(self._update_summary)
            table.setCellWidget(row, 1, name_edit)
            result_item = QtWidgets.QTableWidgetItem(output.label)
            result_item.setData(QtCore.Qt.UserRole, output)
            table.setItem(row, 2, result_item)
            unit_edit = QtWidgets.QLineEdit(output.unit)
            table.setCellWidget(row, 3, unit_edit)
        header = table.horizontalHeader()
        header.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QtWidgets.QHeaderView.Stretch)
        header.setSectionResizeMode(3, QtWidgets.QHeaderView.ResizeToContents)

    def _analysis_changed(self) -> None:
        self.node_name_edit.setText(
            f"{self.descriptor.title} · {self.analysis_combo.currentText()}"
        )
        self._populate_outputs()
        self._update_summary()

    def _destination_changed(self) -> None:
        create_new = self.system_combo.currentData() == _CREATE_NEW_SYSTEM
        self.new_system_edit.setEnabled(create_new)

    def _selected_count(self, table: QtWidgets.QTableWidget) -> int:
        return sum(
            1 for row in range(table.rowCount())
            if table.cellWidget(row, 0).isChecked()
        )

    def _update_summary(self, *_args) -> None:
        if not hasattr(self, "summary_label"):
            return
        self.summary_label.setText(
            f"{self._selected_count(self.input_table)} inputs  →  "
            f"{self._selected_count(self.output_table)} results"
        )

    def _build_spec(self) -> SimulationFunctionSpec:
        selected_inputs: list[SimulationInputSpec] = []
        for row in range(self.input_table.rowCount()):
            if not self.input_table.cellWidget(row, 0).isChecked():
                continue
            source: StudyInput = self.input_table.item(row, 2).data(QtCore.Qt.UserRole)
            selected_inputs.append(
                SimulationInputSpec(
                    port_name=self.input_table.cellWidget(row, 1).text().strip(),
                    label=source.label,
                    target_kind=source.source_kind,
                    target=source.target,
                    default=self.input_table.cellWidget(row, 3).value(),
                    lower=self.input_table.cellWidget(row, 4).value(),
                    upper=self.input_table.cellWidget(row, 5).value(),
                    unit=self.input_table.cellWidget(row, 6).text().strip() or "-",
                )
            )

        selected_outputs: list[SimulationOutputSpec] = []
        for row in range(self.output_table.rowCount()):
            if not self.output_table.cellWidget(row, 0).isChecked():
                continue
            output = self.output_table.item(row, 2).data(QtCore.Qt.UserRole)
            selected_outputs.append(
                SimulationOutputSpec(
                    port_name=self.output_table.cellWidget(row, 1).text().strip(),
                    label=output.label,
                    result_field=output.field,
                    unit=self.output_table.cellWidget(row, 3).text().strip() or "-",
                )
            )

        return SimulationFunctionSpec(
            project_path=self.descriptor.path,
            analysis_kind=str(self.analysis_combo.currentData()),
            node_name=self.node_name_edit.text().strip(),
            inputs=tuple(selected_inputs),
            outputs=tuple(selected_outputs),
        )

    def _accept_selection(self) -> None:
        self.error_label.clear()
        if not self.node_name_edit.text().strip():
            self.error_label.setText("Enter a name for the simulation function.")
            return
        destination = self.destination_system_name()
        if not destination:
            self.error_label.setText("Choose a destination system name.")
            return
        try:
            spec = self._build_spec()
            validate_simulation_spec(spec)
        except ValueError as exc:
            self.error_label.setText(str(exc))
            return
        self._spec = spec
        self.accept()

    def selected_spec(self) -> SimulationFunctionSpec:
        if self._spec is None:
            raise RuntimeError("The dialog has not accepted a simulation interface.")
        return self._spec

    def destination_system_name(self) -> str:
        if self.system_combo.currentData() == _CREATE_NEW_SYSTEM:
            return self.new_system_edit.text().strip()
        return str(self.system_combo.currentData() or "").strip()

    def should_create_system(self) -> bool:
        return self.system_combo.currentData() == _CREATE_NEW_SYSTEM

    def should_create_io_nodes(self) -> bool:
        return self.create_io_checkbox.isChecked()
