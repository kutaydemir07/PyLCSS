# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE.
"""SurrogateDataMixin behavior for surrogate training."""

from __future__ import annotations

import logging
import os

import numpy as np
from PySide6 import QtWidgets

from pylcss.system_modeling.compiler import GraphBuilder

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
        fname, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open Data File", "", "Data Files (*.csv *.json)"
        )
        if not fname:
            return

        try:
            import pandas as pd

            if fname.endswith(".json"):
                df = pd.read_json(fname)
            else:
                df = pd.read_csv(fname)

            # We need to identify inputs and outputs
            # For now, we'll try to match with the selected node's inputs and output
            idx = self.combo_nodes.currentIndex()
            if idx < 0:
                QtWidgets.QMessageBox.warning(
                    self, "Error", "Please select a target node first to match columns."
                )
                return

            target_node = self.combo_nodes.itemData(idx)

            # Get input names from the graph connection logic or node properties
            # This is tricky without the full graph context easily available in a simple way
            # Let's try to get inputs from the node's input ports
            # But wait, the surrogate models the WHOLE subgraph feeding into this node?
            # Or just this node?
            # Usually surrogate models replace a complex calculation.
            # If it's a "Black Box" node, it has inputs.
            # If it's an output node of a graph, it depends on system inputs.

            # Let's assume the user knows what they are doing and the CSV has columns:
            # inputs... and output

            # Heuristic:
            # 1. Find column matching target node name (or 'y', 'target', 'output')
            # 2. All other numeric columns are inputs

            target_col = None
            possible_targets = [
                target_node.name(),
                target_node.get_property("var_name"),
                "y",
                "target",
                "output",
            ]

            for t in possible_targets:
                if t and t in df.columns:
                    target_col = t
                    break

            if not target_col:
                # Ask user to pick target column?
                cols = list(df.columns)
                item, ok = QtWidgets.QInputDialog.getItem(
                    self,
                    "Select Target Column",
                    "Which column is the output?",
                    cols,
                    0,
                    False,
                )
                if ok and item:
                    target_col = item
                else:
                    return

            # Prepare data
            y = df[target_col].values
            X_df = df.drop(columns=[target_col])

            # Filter only numeric columns for X
            X_df = X_df.select_dtypes(include=[np.number])
            X = X_df.values

            # Split
            from sklearn.model_selection import train_test_split

            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            self.lbl_file_info.setText(
                f"Loaded: {os.path.basename(fname)}\n{len(df)} samples\n{X.shape[1]} inputs"
            )
            self.btn_train.setEnabled(True)
            self.btn_adaptive.setEnabled(True)
            self.lbl_metrics.setText("Data loaded. Ready to train.")
            self.update_data_table()

        except Exception as e:
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
            output_nodes = [n for n in nodes if n.type_.startswith("com.pfd.output")]

            builder = GraphBuilder(graph)
            spy_code, spy_inputs, spy_outputs = builder.build_spy_model(
                nodes, input_nodes, output_nodes, target_node.id, "spy_model"
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
            spy_code, spy_inputs, spy_outputs, input_bounds, samples
        )
        self.gen_worker.progress_sig.connect(self.update_progress)
        self.gen_worker.done_sig.connect(self.generation_finished)
        self.gen_worker.start()

    def generation_finished(self, data, error):
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

    def update_data_table(self):
        if self.X_train is None:
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

        headers = [f"Input {i + 1}" for i in range(x_cols)] + [
            f"Output {i + 1}" if y_cols > 1 else "Output" for i in range(y_cols)
        ]
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

    def update_hyperparams(self, index: int) -> None:
        # Map algorithm display name -> hyperparam stack page index.
        # Falls back to the combo index for legacy ordering.
        algo = self.combo_algo.currentText()
        page = self._algo_to_page.get(algo, index)
        self.stack_params.setCurrentIndex(page)

        # Geometric backbones drive the CAD pipeline themselves; they don't
        # need pre-generated tabular data, so the Train button can be enabled
        # immediately once a CAD file + target node are configured.
        if algo in ("Geom-DeepONet", "GINO"):
            self.btn_train.setEnabled(True)
            self.btn_adaptive.setEnabled(False)  # adaptive uses tabular spy data only
            self._set_geom_visibility(algo)
        else:
            # Restore standard rule: enabled only if data is ready.
            ready = hasattr(self, "X_train") and self.X_train is not None
            self.btn_train.setEnabled(ready)
            self.btn_adaptive.setEnabled(ready)

    def _set_geom_visibility(self, algo: str) -> None:
        """Show only the architecture rows that apply to the selected
        geometric backbone, so the user doesn't see knobs that get ignored."""
        is_donet = algo == "Geom-DeepONet"
        is_gino = algo == "GINO"

        # QFormLayout.setRowVisible requires Qt 6.4+; PySide6 6.10 has it.
        # Fall back to per-widget setVisible() if the form-layout API isn't
        # available (older Qt) so this stays robust.
        donet_widgets = [
            (self._lbl_donet_latent, self.spin_donet_latent),
            (self._lbl_donet_trunk, self.spin_donet_trunk),
        ]
        gino_widgets = [
            (self._lbl_gino_channels, self.spin_gino_channels),
            (self._lbl_gino_grid, self.spin_gino_grid),
            (self._lbl_gino_modes, self.spin_gino_modes),
        ]
        for lbl, w in donet_widgets:
            lbl.setVisible(is_donet)
            w.setVisible(is_donet)
        for lbl, w in gino_widgets:
            lbl.setVisible(is_gino)
            w.setVisible(is_gino)

    def _browse_cad_path(self) -> None:
        fname, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select CAD Graph File",
            "",
            "CAD Graph (*.cad *.json);;All Files (*)",
        )
        if fname:
            self.txt_cad_path.setText(fname)

    def _refresh_field_choices(self, solver_kind: str) -> None:
        if not hasattr(self, "combo_field"):
            return
        prev = self.combo_field.currentText().strip()
        choices = self._FIELD_CHOICES_BY_SOLVER.get(
            solver_kind,
            ["von_mises", "displacement"],
        )
        self.combo_field.blockSignals(True)
        self.combo_field.clear()
        self.combo_field.addItems(choices)
        # Preserve the user's previous choice if it still makes sense, else
        # default to the first option for the new solver.
        if prev in choices:
            self.combo_field.setCurrentText(prev)
        else:
            self.combo_field.setCurrentIndex(0)
        self.combo_field.blockSignals(False)

    def _collect_input_bounds_for_target(
        self,
    ) -> tuple[list[tuple[float, float]], list[str]]:
        """Return (bounds, names) from the currently selected target node's
        spy-model context. Mirrors what start_generation extracts so geometric
        backbones share the same parameter setup with tabular ones."""
        idx = self.combo_nodes.currentIndex()
        if idx < 0:
            raise RuntimeError("No target node selected.")
        self.combo_nodes.itemData(idx)
        graph = self.modeling_widget.current_graph
        nodes = graph.all_nodes()
        input_nodes = [n for n in nodes if n.type_.startswith("com.pfd.input")]
        [n for n in nodes if n.type_.startswith("com.pfd.output")]

        # For geometric surrogates we need (parameter_name -> bounds) pairs
        # that match the CAD graph's exposed inputs -- these are the global
        # input nodes, NOT the target node's input ports (which can be
        # intermediates).  Pull both name + range straight from input_nodes
        # so the dict keys we pass to cad.runtime.fea(**params) line up with
        # the CAD graph's exposed input parameters.
        bounds: list[tuple[float, float]] = []
        names: list[str] = []
        for inp_node in input_nodes:
            if inp_node.has_property("input_props"):
                props = inp_node.get_property("input_props")
                lo = float(props.get("min", "0.0"))
                hi = float(props.get("max", "10.0"))
            else:
                lo = float(inp_node.get_property("min"))
                hi = float(inp_node.get_property("max"))
            bounds.append((lo, hi))
            # Prefer the variable name property (what the CAD graph keys off)
            # then fall back to the node's display name.
            var_name = (
                inp_node.get_property("var_name")
                if inp_node.has_property("var_name")
                else None
            )
            names.append(str(var_name or inp_node.name()))

        if not names:
            raise RuntimeError(
                "No global input nodes found in the system graph; "
                "geometric surrogates need at least one parametric input."
            )
        return bounds, names
