# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE.
"""Interactive dialog for inspecting, mapping, and cleaning uploaded surrogate datasets."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from PySide6 import QtCore, QtWidgets

from pylcss.mdao.surrogate_modeling.data_importer import DataImporter

logger = logging.getLogger(__name__)

__all__ = ["DataImportDialog"]


class DataImportDialog(QtWidgets.QDialog):
    """Dialog allowing users to inspect uploaded data and configure input/output column mapping."""

    def __init__(
        self,
        df: pd.DataFrame,
        filename: str,
        target_name_hint: str | None = None,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.df = df
        self.filename = filename
        self.target_name_hint = target_name_hint
        self.setWindowTitle(f"Data Import & Column Mapping — {filename}")
        self.resize(820, 640)

        self.input_cols: list[str] = []
        self.output_cols: list[str] = []
        self.handle_missing: str = "drop"
        self.test_size: float = 0.2
        self.stats: dict[str, Any] = {}

        self.result_data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], list[str], dict[str, Any]] | None = None

        self._setup_ui()
        self._auto_detect_mapping()
        self._update_summary()

    def _setup_ui(self) -> None:
        main_layout = QtWidgets.QVBoxLayout(self)

        # File header info
        header = QtWidgets.QLabel(
            f"<b>File:</b> {self.filename} &nbsp;|&nbsp; "
            f"<b>Rows:</b> {len(self.df)} &nbsp;|&nbsp; "
            f"<b>Columns:</b> {len(self.df.columns)}"
        )
        header.setStyleSheet("font-size: 11pt; padding: 4px;")
        main_layout.addWidget(header)

        # Data preview table (first 8 rows)
        grp_preview = QtWidgets.QGroupBox("Data Preview (First 8 Rows)")
        preview_layout = QtWidgets.QVBoxLayout(grp_preview)
        self.preview_table = QtWidgets.QTableWidget()
        self.preview_table.setMaximumHeight(160)
        self._populate_preview()
        preview_layout.addWidget(self.preview_table)
        main_layout.addWidget(grp_preview)

        # Split pane for Inputs and Outputs
        mapping_group = QtWidgets.QGroupBox("Column Mapping")
        map_layout = QtWidgets.QHBoxLayout(mapping_group)

        # --- Left: Input Features ---
        left_box = QtWidgets.QVBoxLayout()
        lbl_in = QtWidgets.QLabel("<b>Input Features (X)</b>")
        left_box.addWidget(lbl_in)

        btn_row_in = QtWidgets.QHBoxLayout()
        self.btn_select_all_in = QtWidgets.QPushButton("Select All Numeric")
        self.btn_clear_in = QtWidgets.QPushButton("Clear")
        self.btn_select_all_in.clicked.connect(self._select_all_numeric_inputs)
        self.btn_clear_in.clicked.connect(self._clear_inputs)
        btn_row_in.addWidget(self.btn_select_all_in)
        btn_row_in.addWidget(self.btn_clear_in)
        left_box.addLayout(btn_row_in)

        self.list_inputs = QtWidgets.QListWidget()
        self.list_inputs.itemChanged.connect(self._on_input_item_changed)
        left_box.addWidget(self.list_inputs)
        map_layout.addLayout(left_box, 1)

        # --- Right: Output Targets ---
        right_box = QtWidgets.QVBoxLayout()
        lbl_out = QtWidgets.QLabel("<b>Output Targets (y)</b> (Supports Multi-Output)")
        right_box.addWidget(lbl_out)

        btn_row_out = QtWidgets.QHBoxLayout()
        self.btn_auto_out = QtWidgets.QPushButton("Auto-Detect Target")
        self.btn_auto_out.clicked.connect(self._auto_detect_mapping)
        btn_row_out.addWidget(self.btn_auto_out)
        right_box.addLayout(btn_row_out)

        self.list_outputs = QtWidgets.QListWidget()
        self.list_outputs.itemChanged.connect(self._on_output_item_changed)
        right_box.addWidget(self.list_outputs)
        map_layout.addLayout(right_box, 1)

        main_layout.addWidget(mapping_group)

        # Options: Missing Values & Test Split
        opts_layout = QtWidgets.QHBoxLayout()

        lbl_missing = QtWidgets.QLabel("Missing Values:")
        self.combo_missing = QtWidgets.QComboBox()
        self.combo_missing.addItem("Drop rows with missing/infinite values", "drop")
        self.combo_missing.addItem("Impute missing values with column mean", "impute_mean")
        self.combo_missing.addItem("Impute missing values with column median", "impute_median")
        self.combo_missing.currentIndexChanged.connect(self._update_summary)
        opts_layout.addWidget(lbl_missing)
        opts_layout.addWidget(self.combo_missing)

        opts_layout.addSpacing(20)

        lbl_split = QtWidgets.QLabel("Test Split Ratio:")
        self.spin_split = QtWidgets.QDoubleSpinBox()
        self.spin_split.setRange(0.05, 0.50)
        self.spin_split.setSingleStep(0.05)
        self.spin_split.setValue(0.20)
        self.spin_split.setSuffix(" (20%)")
        self.spin_split.valueChanged.connect(self._on_split_changed)
        opts_layout.addWidget(lbl_split)
        opts_layout.addWidget(self.spin_split)
        opts_layout.addStretch()

        main_layout.addLayout(opts_layout)

        # Summary label
        self.lbl_summary = QtWidgets.QLabel("Status: Initializing...")
        self.lbl_summary.setStyleSheet(
            "background-color: #2b2b2b; color: #a0ffa0; padding: 8px; border-radius: 4px; font-weight: bold;"
        )
        main_layout.addWidget(self.lbl_summary)

        # Dialog Buttons
        btn_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        self.btn_ok = btn_box.button(QtWidgets.QDialogButtonBox.Ok)
        self.btn_ok.setText("Import & Prepare Data")
        btn_box.accepted.connect(self._on_accept)
        btn_box.rejected.connect(self.reject)
        main_layout.addWidget(btn_box)

    def _populate_preview(self) -> None:
        cols = list(self.df.columns)
        preview_rows = min(8, len(self.df))
        self.preview_table.setColumnCount(len(cols))
        self.preview_table.setRowCount(preview_rows)
        self.preview_table.setHorizontalHeaderLabels([str(c) for c in cols])

        for r in range(preview_rows):
            for c, col_name in enumerate(cols):
                val = self.df.iloc[r, c]
                item = QtWidgets.QTableWidgetItem(str(val))
                self.preview_table.setItem(r, c, item)

    def _auto_detect_mapping(self) -> None:
        guess_targets = DataImporter.guess_target_columns(
            self.df, target_name_hint=self.target_name_hint
        )
        col_info = DataImporter.inspect_columns(self.df)

        self.list_inputs.blockSignals(True)
        self.list_outputs.blockSignals(True)
        self.list_inputs.clear()
        self.list_outputs.clear()

        for info in col_info:
            c = info["name"]
            is_num = info["is_numeric"]
            is_target = c in guess_targets

            # Input list item
            item_in = QtWidgets.QListWidgetItem(f"{c}  [{info['dtype']}]")
            item_in.setData(QtCore.Qt.UserRole, c)
            item_in.setFlags(item_in.flags() | QtCore.Qt.ItemIsUserCheckable)
            # Default input checked if numeric and not target
            item_in.setCheckState(
                QtCore.Qt.Checked if (is_num and not is_target) else QtCore.Qt.Unchecked
            )
            self.list_inputs.addItem(item_in)

            # Output list item
            item_out = QtWidgets.QListWidgetItem(f"{c}  [{info['dtype']}]")
            item_out.setData(QtCore.Qt.UserRole, c)
            item_out.setFlags(item_out.flags() | QtCore.Qt.ItemIsUserCheckable)
            # Default output checked if guessed target
            item_out.setCheckState(
                QtCore.Qt.Checked if is_target else QtCore.Qt.Unchecked
            )
            self.list_outputs.addItem(item_out)

        self.list_inputs.blockSignals(False)
        self.list_outputs.blockSignals(False)

    def _select_all_numeric_inputs(self) -> None:
        col_info = {inf["name"]: inf["is_numeric"] for inf in DataImporter.inspect_columns(self.df)}
        self.list_inputs.blockSignals(True)
        for i in range(self.list_inputs.count()):
            item = self.list_inputs.item(i)
            c = item.data(QtCore.Qt.UserRole)
            if col_info.get(c, False) and c not in self._get_checked_outputs():
                item.setCheckState(QtCore.Qt.Checked)
        self.list_inputs.blockSignals(False)
        self._update_summary()

    def _clear_inputs(self) -> None:
        self.list_inputs.blockSignals(True)
        for i in range(self.list_inputs.count()):
            self.list_inputs.item(i).setCheckState(QtCore.Qt.Unchecked)
        self.list_inputs.blockSignals(False)
        self._update_summary()

    def _get_checked_inputs(self) -> list[str]:
        inputs = []
        for i in range(self.list_inputs.count()):
            item = self.list_inputs.item(i)
            if item.checkState() == QtCore.Qt.Checked:
                inputs.append(item.data(QtCore.Qt.UserRole))
        return inputs

    def _get_checked_outputs(self) -> list[str]:
        outputs = []
        for i in range(self.list_outputs.count()):
            item = self.list_outputs.item(i)
            if item.checkState() == QtCore.Qt.Checked:
                outputs.append(item.data(QtCore.Qt.UserRole))
        return outputs

    def _on_input_item_changed(self, item: QtWidgets.QListWidgetItem) -> None:
        c = item.data(QtCore.Qt.UserRole)
        if item.checkState() == QtCore.Qt.Checked:
            # Uncheck from outputs if checked
            self.list_outputs.blockSignals(True)
            for i in range(self.list_outputs.count()):
                out_item = self.list_outputs.item(i)
                if out_item.data(QtCore.Qt.UserRole) == c:
                    out_item.setCheckState(QtCore.Qt.Unchecked)
            self.list_outputs.blockSignals(False)
        self._update_summary()

    def _on_output_item_changed(self, item: QtWidgets.QListWidgetItem) -> None:
        c = item.data(QtCore.Qt.UserRole)
        if item.checkState() == QtCore.Qt.Checked:
            # Uncheck from inputs if checked
            self.list_inputs.blockSignals(True)
            for i in range(self.list_inputs.count()):
                in_item = self.list_inputs.item(i)
                if in_item.data(QtCore.Qt.UserRole) == c:
                    in_item.setCheckState(QtCore.Qt.Unchecked)
            self.list_inputs.blockSignals(False)
        self._update_summary()

    def _on_split_changed(self, val: float) -> None:
        self.spin_split.setSuffix(f" ({int(val * 100)}%)")
        self._update_summary()

    def _update_summary(self) -> None:
        in_cols = self._get_checked_inputs()
        out_cols = self._get_checked_outputs()
        missing_mode = self.combo_missing.currentData()

        if not in_cols:
            self.lbl_summary.setText("Warning: Please select at least one Input Feature (X).")
            self.lbl_summary.setStyleSheet("color: #ff9999; padding: 6px;")
            self.btn_ok.setEnabled(False)
            return

        if not out_cols:
            self.lbl_summary.setText("Warning: Please select at least one Output Target (y).")
            self.lbl_summary.setStyleSheet("color: #ff9999; padding: 6px;")
            self.btn_ok.setEnabled(False)
            return

        try:
            # Test preparation with current settings
            _, _, _, _, stats = DataImporter.clean_and_prepare(
                self.df,
                in_cols,
                out_cols,
                handle_missing=missing_mode,
                test_size=self.spin_split.value(),
            )
            msg = (
                f"Ready: {stats['n_features']} Inputs ({', '.join(in_cols[:3])}{'...' if len(in_cols) > 3 else ''}) -> "
                f"{stats['n_targets']} Target(s) ({', '.join(out_cols)}) | "
                f"<b>{stats['valid_rows']}</b> samples "
                f"({stats['n_train']} train / {stats['n_test']} test)"
            )
            if stats["dropped_rows"] > 0:
                msg += f" | <i>{stats['dropped_rows']} rows with NaNs dropped</i>"
            self.lbl_summary.setText(msg)
            self.lbl_summary.setStyleSheet("color: #a0ffa0; padding: 6px;")
            self.btn_ok.setEnabled(True)
        except Exception as e:
            self.lbl_summary.setText(f"Configuration error: {e}")
            self.lbl_summary.setStyleSheet("color: #ff9999; padding: 6px;")
            self.btn_ok.setEnabled(False)

    def _on_accept(self) -> None:
        in_cols = self._get_checked_inputs()
        out_cols = self._get_checked_outputs()
        missing_mode = self.combo_missing.currentData()
        test_size = self.spin_split.value()

        try:
            X_train, X_test, y_train, y_test, stats = DataImporter.clean_and_prepare(
                self.df,
                in_cols,
                out_cols,
                handle_missing=missing_mode,
                test_size=test_size,
            )
            self.result_data = (
                X_train,
                X_test,
                y_train,
                y_test,
                in_cols,
                out_cols,
                stats,
            )
            self.accept()
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Data Preparation Error", str(e))
