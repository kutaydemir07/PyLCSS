# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE.
"""SurrogateDataMixin behavior for surrogate training."""

from __future__ import annotations

import logging
import os

import numpy as np
from PySide6 import QtWidgets

from pylcss.mdao.system_modeling.spy_compiler import build_spy_model

from .workers import (
    DataGenerationWorker,
)

logger = logging.getLogger(__name__)

__all__ = ["SurrogateDataMixin"]


class SurrogateDataMixin:
    def toggle_data_source(self):
        if self.radio_gen.isChecked():
            self.stack_data.setCurrentIndex(0)
        else:
            self.stack_data.setCurrentIndex(1)

    def browse_file(self):
        filter_str = (
            "All Supported Data (*.csv *.tsv *.txt *.xlsx *.xls *.json *.parquet *.npy *.npz);;"
            "CSV Files (*.csv);;"
            "Text / TSV Files (*.tsv *.txt);;"
            "Excel Workbooks (*.xlsx *.xls);;"
            "JSON Data (*.json);;"
            "Parquet Files (*.parquet);;"
            "NumPy Arrays (*.npy *.npz);;"
            "All Files (*)"
        )
        fname, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open Data File", "", filter_str
        )
        if not fname:
            return

        try:
            from pylcss.mdao.surrogate_modeling.data_importer import DataImporter
            from .data_import_dialog import DataImportDialog

            df = DataImporter.load_file(fname)

            # Node hint if a target node is already selected
            idx = self.combo_nodes.currentIndex()
            target_name_hint = None
            if idx >= 0:
                node = self.combo_nodes.itemData(idx)
                if node:
                    target_name_hint = node.get_property("var_name") or node.name()

            # Open interactive column mapping and inspection dialog
            dialog = DataImportDialog(
                df=df,
                filename=os.path.basename(fname),
                target_name_hint=target_name_hint,
                parent=self,
            )
            if dialog.exec() != QtWidgets.QDialog.Accepted or not dialog.result_data:
                return

            X_tr, X_te, y_tr, y_te, in_cols, out_cols, stats = dialog.result_data

            self.X_train = X_tr
            self.X_test = X_te
            self.y_train = y_tr
            self.y_test = y_te
            self.input_names = list(in_cols)
            self.output_names = list(out_cols)

            info_text = (
                f"Loaded: {os.path.basename(fname)}\n"
                f"{stats['valid_rows']} valid samples ({stats['n_train']} train / {stats['n_test']} test)\n"
                f"Inputs ({len(in_cols)}): {', '.join(in_cols[:4])}{'...' if len(in_cols) > 4 else ''}\n"
                f"Outputs ({len(out_cols)}): {', '.join(out_cols)}"
            )
            self.lbl_file_info.setText(info_text)
            self.btn_train.setEnabled(True)
            self.btn_adaptive.setEnabled(True)
            self.lbl_metrics.setText(
                f"Samples: {stats['valid_rows']} · "
                f"Inputs: {len(in_cols)} · Outputs: {len(out_cols)}"
            )
            self.update_data_table()
            if hasattr(self, "update_training_summary"):
                self.update_training_summary()

        except Exception as e:
            logger.exception("Failed to load and map data file.")
            QtWidgets.QMessageBox.critical(self, "Load Error", str(e))

    def start_generation(self):
        idx = self.combo_nodes.currentIndex()
        if idx < 0:
            QtWidgets.QMessageBox.warning(self, "Error", "No node selected.")
            return

        target_node = self.combo_nodes.itemData(idx)
        samples = self.spin_samples.value()

        self.btn_generate.setEnabled(False)
        self.lbl_metrics.setText("Generating data...")
        self.progress.setValue(0)

        try:
            graph = self.modeling_widget.current_graph
            nodes = graph.all_nodes()
            input_nodes = [n for n in nodes if n.type_.startswith("com.pfd.input")]
            spy_code, spy_inputs, spy_outputs = build_spy_model(
                nodes, input_nodes, target_node.id, "spy_model"
            )

            input_bounds = []
            for inp_node in input_nodes:
                if inp_node.has_property("input_props"):
                    props = inp_node.get_property("input_props")
                    min_val = float(props.get("min", "0.0"))
                    max_val = float(props.get("max", "10.0"))
                else:
                    min_val = float(inp_node.get_property("min"))
                    max_val = float(inp_node.get_property("max"))
                input_bounds.append((min_val, max_val))

        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Preparation Error", str(e))
            self.btn_generate.setEnabled(True)
            return

        self.gen_worker = DataGenerationWorker(
            spy_code, spy_inputs, spy_outputs, input_bounds, samples,
            max_consecutive_failures=self.spin_failure_limit.value() or None,
        )
        self.gen_worker.progress_sig.connect(self.update_progress)
        self.gen_worker.done_sig.connect(self.generation_finished)
        self.gen_worker.start()
        self.btn_stop.setEnabled(True)

    def generation_finished(self, data, error):
        self.btn_stop.setEnabled(False)
        self.btn_stop.setText("Stop Training")
        self.btn_generate.setEnabled(True)
        if error:
            QtWidgets.QMessageBox.critical(self, "Generation Failed", error)
            self.lbl_metrics.setText("Generation failed.")
            return

        self.X_train, self.y_train, self.X_test, self.y_test = data
        self.btn_train.setEnabled(True)
        self.btn_adaptive.setEnabled(True)
        self.lbl_metrics.setText(
            f"Data generated: {len(self.X_train) + len(self.X_test)} samples."
        )
        self.progress.setValue(100)
        self.update_data_table()
        if hasattr(self, "update_training_summary"):
            self.update_training_summary()

    def update_data_table(self):
        if self.X_train is None or not hasattr(self, "data_table") or self.data_table is None:
            return

        # Combine train and test for preview
        X = np.vstack((self.X_train, self.X_test))
        y = np.concatenate((self.y_train, self.y_test))

        # Handle y shape
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        rows, x_cols = X.shape
        _, y_cols = y.shape

        self.data_table.setRowCount(
            min(rows, 1000)
        )  # Limit to 1000 rows for performance
        self.data_table.setColumnCount(x_cols + y_cols)

        in_names = getattr(self, "input_names", None)
        out_names = getattr(self, "output_names", None)

        headers = []
        for i in range(x_cols):
            if in_names and i < len(in_names):
                headers.append(str(in_names[i]))
            else:
                headers.append(f"Input {i + 1}")
        for k in range(y_cols):
            if out_names and k < len(out_names):
                headers.append(str(out_names[k]))
            else:
                headers.append(f"Output {k + 1}" if y_cols > 1 else "Output")

        self.data_table.setHorizontalHeaderLabels(headers)

        for i in range(min(rows, 1000)):
            for j in range(x_cols):
                self.data_table.setItem(
                    i, j, QtWidgets.QTableWidgetItem(f"{X[i, j]:.4f}")
                )
            for k in range(y_cols):
                val = y[i, k]
                self.data_table.setItem(
                    i, x_cols + k, QtWidgets.QTableWidgetItem(f"{val:.4f}")
                )

    def refresh_nodes(self) -> None:
        """Fetch available CustomBlockNodes from the active graph."""
        self.combo_nodes.clear()
        if not self.modeling_widget or not self.modeling_widget.current_graph:
            return

        nodes = self.modeling_widget.current_graph.all_nodes()
        for node in nodes:
            if node.type_.startswith("com.pfd.custom_block"):
                # Store node ID in user data
                self.combo_nodes.addItem(f"{node.name()} ({node.id})", node)

    def update_hyperparams(self, _index: int) -> None:
        algo = self.combo_algo.currentText()
        self.stack_params.setCurrentIndex(self._algo_to_page[algo])
        ready = self.X_train is not None
        if hasattr(self, "btn_train"):
            self.btn_train.setEnabled(ready)
        if hasattr(self, "btn_adaptive"):
            self.btn_adaptive.setEnabled(ready)
