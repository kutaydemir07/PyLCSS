# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE.
"""SolutionPersistenceMixin behavior for solution-space analysis."""

from __future__ import annotations

import json
import logging
import os
import tempfile

import h5py
import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from pylcss.system_modeling.problem import DesignProblem


logger = logging.getLogger(__name__)

__all__ = ["SolutionPersistenceMixin"]


class SolutionPersistenceMixin:
    def plot_product_family(self, results):
        """
        Visualizes the ranges for each product variant and the common platform.
        Now includes explicit Axis Labels and Titles for clearer DV identification.
        """
        # Check for valid results
        if not results or not self.problem or not self.problem.design_variables:
            return

        # 1. Clear existing plots in the scroll area
        while self.family_plots_layout.count():
            item = self.family_plots_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        # 2. Setup Colors for Variants
        variant_names = sorted(
            [
                name
                for name in results.keys()
                if name not in ["Platform", "Platform_Infeasible", "Communality"]
            ]
        )
        var_colors = {}
        for i, name in enumerate(variant_names):
            var_colors[name] = self.default_colors[i % len(self.default_colors)]

        # 3. Create a Plot for EACH Design Variable
        cols = 3
        for i, dv in enumerate(self.problem.design_variables):
            # Container for layout
            container = QtWidgets.QWidget()
            vbox = QtWidgets.QVBoxLayout(container)
            vbox.setContentsMargins(0, 10, 0, 20)  # Add spacing between plots

            # PyQtGraph Widget
            win = pg.GraphicsLayoutWidget()
            win.setFixedHeight(220)  # Slightly taller for labels
            win.setBackground("w")
            vbox.addWidget(win)

            # Setup Plot Item
            p = win.addPlot()
            p.showGrid(x=True, y=False, alpha=0.3)
            p.setMenuEnabled(False)
            p.setMouseEnabled(x=True, y=False)

            # --- IMPROVEMENT 1: X-Axis Label with DV Name ---
            unit_str = f" [{dv.get('unit', '-')}]" if dv.get("unit") else ""
            label_text = f"{dv['name']}{unit_str}"
            # Use HTML for bold labeling
            p.setLabel(
                "bottom",
                text=label_text,
                **{"font-size": "11pt", "font-weight": "bold", "color": "black"},
            )

            # Y-Axis Labels (Variant Names)
            y_ticks = [(0, "PLATFORM")]
            for idx, name in enumerate(variant_names):
                y_ticks.append((idx + 1, name))

            ax_left = p.getAxis("left")
            ax_left.setTicks([y_ticks])
            ax_left.setPen("k")
            ax_left.setTextPen("k")
            p.getAxis("bottom").setPen("k")
            p.getAxis("bottom").setTextPen("k")

            p.setYRange(-0.5, len(variant_names) + 0.5)

            # --- DRAWING BARS ---
            min_view = float("inf")
            max_view = float("-inf")

            # 1. Draw Variant Ranges
            for idx, var_name in enumerate(variant_names):
                box = results[var_name]
                if box is not None and i < box.shape[0]:
                    x_min = box[i, 0]
                    x_max = box[i, 1]
                    min_view = min(min_view, x_min)
                    max_view = max(max_view, x_max)

                    color = QtGui.QColor(var_colors[var_name])
                    color.setAlpha(150)

                    bar = pg.BarGraphItem(
                        x0=[x_min],
                        y=[idx + 1],
                        width=[x_max - x_min],
                        height=0.6,
                        brush=pg.mkBrush(color),
                        pen=pg.mkPen("k"),
                    )
                    p.addItem(bar)

                    # Numeric Label
                    text = pg.TextItem(
                        f"{x_min:.2f} - {x_max:.2f}", anchor=(0, 0.5), color="k"
                    )
                    text.setPos(x_max, idx + 1)
                    p.addItem(text)

            # 2. Draw Platform (Intersection)
            platform_exists = False
            if "Platform" in results and results["Platform"] is not None:
                platform = results["Platform"]
                if i < platform.shape[0]:
                    p_min = platform[i, 0]
                    p_max = platform[i, 1]

                    if p_min < p_max:  # Valid intersection
                        platform_exists = True
                        bar = pg.BarGraphItem(
                            x0=[p_min],
                            y=[0],
                            width=[p_max - p_min],
                            height=0.8,
                            brush=pg.mkBrush("#00cc00"),
                            pen=pg.mkPen("k", width=2),
                        )
                        p.addItem(bar)

                        text = pg.TextItem(
                            f"{p_min:.2f} - {p_max:.2f}",
                            anchor=(0, 0.5),
                            color="#006600",
                        )
                        text.setPos(p_max, 0)
                        text.setFont(QtGui.QFont("Arial", 10, QtGui.QFont.Bold))
                        p.addItem(text)

                        # Alignment Guides
                        p.addItem(
                            pg.InfiniteLine(
                                pos=p_min,
                                angle=90,
                                pen=pg.mkPen("g", style=QtCore.Qt.DotLine, width=2),
                            )
                        )
                        p.addItem(
                            pg.InfiniteLine(
                                pos=p_max,
                                angle=90,
                                pen=pg.mkPen("g", style=QtCore.Qt.DotLine, width=2),
                            )
                        )

            # --- IMPROVEMENT 2: Combined Title (Name + Status) ---
            if platform_exists:
                status_text = "Common Platform Feasible"
                status_color = "#008800"
            elif "Platform_Infeasible" in results and results["Platform_Infeasible"]:
                status_text = "Platform Infeasible (No Commonality)"
                status_color = "#cc0000"
            else:
                status_text = "No Commonality"
                status_color = "#cc0000"

            # HTML Title: Name in Large Bold, Status smaller below
            title_html = (
                f"<span style='font-size: 14pt; font-weight: bold; color: black;'>{dv['name']}</span>"
                f"<br><span style='font-size: 9pt; color: {status_color};'>{status_text}</span>"
            )
            p.setTitle(title_html)

            # Auto-scale View - ensure platform is always visible
            if min_view != float("inf"):
                # Include platform bounds in the range calculation
                if platform_exists:
                    min_view = min(min_view, p_min)
                    max_view = max(max_view, p_max)

                width = max_view - min_view
                padding = width * 0.1 if width > 0 else 1.0
                p.setXRange(min_view - padding, max_view + padding * 3)

            # Add to Grid Layout
            row = i // cols
            col = i % cols
            self.family_plots_layout.addWidget(container, row, col)

    def save_to_folder(self, folder_path):
        """Save solution space state to a folder."""
        json_path = os.path.join(folder_path, "solution_space.json")
        h5_path = os.path.join(folder_path, "solution_space.h5")

        # Update problem from UI
        if self.problem:
            for i in range(self.dv_table.rowCount()):
                min_val = self._safe_get_float(self.dv_table.item(i, 2), -1e9)
                max_val = self._safe_get_float(self.dv_table.item(i, 3), 1e9)
                if i < len(self.problem.design_variables):
                    self.problem.design_variables[i]["min"] = min_val
                    self.problem.design_variables[i]["max"] = max_val

            for i in range(self.qoi_table.rowCount()):
                req_min = self._safe_get_float(self.qoi_table.item(i, 2), -1e9)
                req_max = self._safe_get_float(self.qoi_table.item(i, 3), 1e9)
                if i < len(self.problem.quantities_of_interest):
                    self.problem.quantities_of_interest[i]["min"] = req_min
                    self.problem.quantities_of_interest[i]["max"] = req_max

        problem_data = None
        if self.problem:
            problem_data = {
                "name": self.problem.name,
                "design_variables": self.problem.design_variables,
                "quantities_of_interest": self.problem.quantities_of_interest,
                "requirement_sets": self.problem.requirement_sets,
                "samples": self.problem.samples,
                "results": self.problem.results,
                "n_samples": self.problem.sample_size,
            }

        data = {
            "code": self.system_code,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "problem_data": problem_data,
            "last_samples": self.last_samples,
            "dv_par_box": None,
            "qoi_colors": self.qoi_colors,
            "design_plots": [(w.x_name, w.y_name) for w in self.plot_widgets],
            "version": "1.0",
        }

        self.dv_par_box_mutex.lock()
        try:
            dv_par_box_copy = (
                self.dv_par_box.copy() if self.dv_par_box is not None else None
            )
            data["dv_par_box"] = (
                dv_par_box_copy.tolist() if dv_par_box_copy is not None else None
            )
        finally:
            self.dv_par_box_mutex.unlock()

        from pylcss.io_manager.project_io import atomic_json_dump

        serializable_data = self._to_serializable(data)
        atomic_json_dump(serializable_data, json_path)

        # Always replace the HDF5 companion, including when it is empty.
        # Otherwise a new project state with no samples can accidentally load
        # arrays left by an older save in the same folder.
        fd, temporary_h5 = tempfile.mkstemp(
            prefix=".solution_space.",
            suffix=".h5.tmp",
            dir=os.path.abspath(folder_path),
        )
        os.close(fd)
        try:
            with h5py.File(temporary_h5, "w") as h5f:
                h5f.attrs["format"] = "pylcss-solution-space"
                h5f.attrs["version"] = 1
                if self.problem and self.problem.samples:
                    for key, value in self.problem.samples.items():
                        h5f.create_dataset(f"samples/{key}", data=value)
                if self.problem and self.problem.results:
                    for key, value in self.problem.results.items():
                        h5f.create_dataset(f"results/{key}", data=value)
                h5f.flush()
            os.replace(temporary_h5, h5_path)
        finally:
            if os.path.exists(temporary_h5):
                os.remove(temporary_h5)

    def update_ui_from_problem(self):
        """Update UI elements based on the current problem definition."""
        if not self.problem:
            return

        # Update Sample Count
        if hasattr(self, "sample_size_spin"):
            # Handle attribute name mismatch (sample_size vs n_samples)
            if hasattr(self.problem, "sample_size"):
                self.sample_size_spin.setValue(self.problem.sample_size)
            elif hasattr(self.problem, "n_samples"):
                self.sample_size_spin.setValue(self.problem.n_samples)

        # Update Design Variables Table
        self.dv_table.blockSignals(True)
        self.dv_table.setRowCount(len(self.problem.design_variables))
        self.dv_par_box = np.zeros((len(self.problem.design_variables), 2))

        self.inputs = []
        self.input_units = {}

        for i, dv in enumerate(self.problem.design_variables):
            name = dv["name"]
            self.inputs.append(name)
            self.input_units[name] = dv.get("unit", "-")

            min_val = dv["min"]
            max_val = dv["max"]

            self.dv_table.setItem(i, 0, QtWidgets.QTableWidgetItem(name))
            self.dv_table.setItem(i, 1, QtWidgets.QTableWidgetItem(dv.get("unit", "-")))
            self.dv_table.setItem(i, 2, QtWidgets.QTableWidgetItem(str(min_val)))
            self.dv_table.setItem(i, 3, QtWidgets.QTableWidgetItem(str(max_val)))
            # Initialize Solution Space as Design Space (or restore if we had it saved separately)
            self.dv_table.setItem(i, 4, QtWidgets.QTableWidgetItem(str(min_val)))
            self.dv_table.setItem(i, 5, QtWidgets.QTableWidgetItem(str(max_val)))

            self.dv_par_box[i, 0] = min_val
            self.dv_par_box[i, 1] = max_val

        self.dv_table.blockSignals(False)

        # Update Quantities of Interest Table
        self.qoi_table.blockSignals(True)
        self.qoi_table.setRowCount(len(self.problem.quantities_of_interest))

        self.outputs = []
        self.output_units = {}

        for i, qoi in enumerate(self.problem.quantities_of_interest):
            name = qoi["name"]
            self.outputs.append(name)
            self.output_units[name] = qoi.get("unit", "-")

            self.qoi_table.setItem(i, 0, QtWidgets.QTableWidgetItem(name))
            self.qoi_table.setItem(
                i, 1, QtWidgets.QTableWidgetItem(qoi.get("unit", "-"))
            )
            self.qoi_table.setItem(
                i, 2, QtWidgets.QTableWidgetItem(str(qoi.get("min", "")))
            )
            self.qoi_table.setItem(
                i, 3, QtWidgets.QTableWidgetItem(str(qoi.get("max", "")))
            )
            self.qoi_table.setItem(i, 4, QtWidgets.QTableWidgetItem("Auto"))  # Plot Min
            self.qoi_table.setItem(i, 5, QtWidgets.QTableWidgetItem("Auto"))  # Plot Max

            # Minimize checkbox
            min_item = QtWidgets.QTableWidgetItem()
            min_item.setCheckState(
                QtCore.Qt.Checked if qoi.get("minimize", False) else QtCore.Qt.Unchecked
            )
            self.qoi_table.setItem(i, 6, min_item)

            # Maximize checkbox
            max_item = QtWidgets.QTableWidgetItem()
            max_item.setCheckState(
                QtCore.Qt.Checked if qoi.get("maximize", False) else QtCore.Qt.Unchecked
            )
            self.qoi_table.setItem(i, 7, max_item)

            # Weight
            weight_item = QtWidgets.QTableWidgetItem(str(qoi.get("weight", 1.0)))
            self.qoi_table.setItem(i, 8, weight_item)

        self.qoi_table.blockSignals(False)

        # Update Axis Combos
        all_vars = self.inputs + self.outputs
        self.combo_add_x.clear()
        self.combo_add_x.addItems(all_vars)
        self.combo_add_y.clear()
        self.combo_add_y.addItems(all_vars)

        # Update Bounds for plotting
        try:
            self.dsl = np.array(
                [float(dv["min"]) for dv in self.problem.design_variables]
            )
            self.dsu = np.array(
                [float(dv["max"]) for dv in self.problem.design_variables]
            )
        except (KeyError, TypeError, ValueError):
            logger.warning("Problem design-variable bounds are invalid.")
            self.dsl = None
            self.dsu = None

    def load_from_folder(self, folder_path):
        """Load solution space state from a folder."""
        json_path = os.path.join(folder_path, "solution_space.json")
        h5_path = os.path.join(folder_path, "solution_space.h5")

        if not os.path.exists(json_path):
            return  # Nothing to load

        try:
            from pylcss.io_manager.project_io import load_json_object

            data = load_json_object(json_path, required_keys=("version",))

            self.system_code = data.get("code")
            self.inputs = data.get("inputs", [])
            self.outputs = data.get("outputs", [])
            self.qoi_colors = data.get("qoi_colors", {})

            # Restore problem
            p_data = data.get("problem_data")
            if p_data:
                self.problem = DesignProblem(p_data["name"], p_data["n_samples"])
                self.problem.design_variables = p_data["design_variables"]
                self.problem.quantities_of_interest = p_data["quantities_of_interest"]
                self.problem.requirement_sets = p_data.get("requirement_sets", {})

                # Load large data from H5 if available
                if os.path.exists(h5_path):
                    with h5py.File(h5_path, "r") as h5f:
                        if "samples" in h5f:
                            self.problem.samples = {
                                k: np.array(v) for k, v in h5f["samples"].items()
                            }
                        if "results" in h5f:
                            self.problem.results = {
                                k: np.array(v) for k, v in h5f["results"].items()
                            }
                else:
                    # Fallback to JSON data (might be slow/large)
                    self.problem.samples = {
                        k: np.array(v) for k, v in p_data.get("samples", {}).items()
                    }
                    self.problem.results = {
                        k: np.array(v) for k, v in p_data.get("results", {}).items()
                    }

                # Recompile system model if code exists
                if self.system_code:
                    try:
                        # Use SystemModel to create a persistent file for multiprocessing support
                        # This ensures 'dill' can pickle the function correctly
                        from pylcss.system_modeling.model import SystemModel

                        # Create dummy inputs/outputs for SystemModel creation if needed
                        # (SystemModel needs them but we only need the function here)
                        dummy_inputs = [
                            {"name": n, "min": 0, "max": 1} for n in self.inputs
                        ]
                        dummy_outputs = [
                            {"name": n, "req_min": 0, "req_max": 0}
                            for n in self.outputs
                        ]

                        sm = SystemModel.from_code_string(
                            self.problem.name,
                            self.system_code,
                            dummy_inputs,
                            dummy_outputs,
                        )
                        self.problem.set_system_model(sm.system_function)

                    except Exception:
                        logger.warning(
                            "Failed to recompile system model", exc_info=True
                        )

            # Restore UI state
            self.update_ui_from_problem()

            # Restore System Combo and Models List
            if self.problem and self.system_code:
                # Reconstruct a model definition so the UI behaves correctly
                inputs_list = []
                if hasattr(self.problem, "design_variables"):
                    inputs_list = self.problem.design_variables
                else:
                    inputs_list = [
                        {"name": name, "min": 0, "max": 1} for name in self.inputs
                    ]

                outputs_list = []
                if hasattr(self.problem, "quantities_of_interest"):
                    # Map min/max to req_min/req_max for compatibility with OptimizationWidget
                    for qoi in self.problem.quantities_of_interest:
                        q_dict = qoi.copy()
                        if "min" in q_dict:
                            q_dict["req_min"] = q_dict["min"]
                        if "max" in q_dict:
                            q_dict["req_max"] = q_dict["max"]
                        outputs_list.append(q_dict)
                else:
                    outputs_list = [
                        {"name": name, "req_min": 0, "req_max": 0}
                        for name in self.outputs
                    ]

                reconstructed_model = {
                    "name": self.problem.name,
                    "code": self.system_code,
                    "inputs": inputs_list,
                    "outputs": outputs_list,
                }
                self.models = [reconstructed_model]

                # Save the loaded problem with data before it gets overwritten
                loaded_problem = self.problem

                self.system_combo.blockSignals(True)
                self.system_combo.clear()
                self.system_combo.addItem(self.problem.name)
                self.system_combo.blockSignals(False)

                # Trigger load_selected_system to initialize UI, bounds, and buttons
                self.load_selected_system()

                # Restore the problem with loaded data
                self.problem = loaded_problem

                # Refresh UI with restored data
                self.update_ui_from_problem()
                self.update_all_plots()

                # Enable buttons
                self.btn_compute_feasible.setEnabled(True)

                has_objectives = any(
                    qoi.get("minimize", False) or qoi.get("maximize", False)
                    for qoi in self.problem.quantities_of_interest
                )
                self.chk_include_optimization.setEnabled(has_objectives)
                self.btn_compute_family.setEnabled(True)

                # Enable resample if we have samples
                if self.problem.samples:
                    self.btn_resample.setEnabled(True)

            # Restore last samples
            ls = data.get("last_samples")
            if ls:
                self.last_samples = {
                    "points": np.array(ls["points"]),
                    "qoi_values": np.array(ls["qoi_values"]),
                    "is_good": np.array(ls["is_good"]),
                    "violation_idx": np.array(ls["violation_idx"]),
                }

            # Restore ROI box
            box_list = data.get("dv_par_box")
            if box_list:
                self.dv_par_box_mutex.lock()
                self.dv_par_box = np.array(box_list)
                self.dv_par_box_mutex.unlock()

            # Update plots
            self.update_all_plots()

            # Restore Design Space Plots
            design_plots = data.get("design_plots", [])
            if design_plots:
                self.clear_all_plots()
                for x_name, y_name in design_plots:
                    self.add_plot(x_name, y_name)

            for pw in self.plot_widgets:
                pw.plot()

            # Trigger replot if possible
            # self.update_plots() # Might be too early if UI not fully ready

        except Exception as e:
            raise e

    def save_project(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save Project", "", "PFD Project (*.pfd)"
        )
        if path:
            if self.problem:
                for i in range(self.dv_table.rowCount()):
                    min_val = self._safe_get_float(self.dv_table.item(i, 2), -1e9)
                    max_val = self._safe_get_float(self.dv_table.item(i, 3), 1e9)
                    if i < len(self.problem.design_variables):
                        self.problem.design_variables[i]["min"] = min_val
                        self.problem.design_variables[i]["max"] = max_val

                for i in range(self.qoi_table.rowCount()):
                    req_min = self._safe_get_float(self.qoi_table.item(i, 2), -1e9)
                    req_max = self._safe_get_float(self.qoi_table.item(i, 3), 1e9)
                    if i < len(self.problem.quantities_of_interest):
                        self.problem.quantities_of_interest[i]["min"] = req_min
                        self.problem.quantities_of_interest[i]["max"] = req_max

            problem_data = None
            if self.problem:
                problem_data = {
                    "name": self.problem.name,
                    "design_variables": self.problem.design_variables,
                    "quantities_of_interest": self.problem.quantities_of_interest,
                    "samples": self.problem.samples,
                    "results": self.problem.results,
                    "n_samples": self.problem.sample_size,
                }

            data = {
                "code": self.system_code,
                "inputs": self.inputs,
                "outputs": self.outputs,
                "problem_data": problem_data,
                "last_samples": self.last_samples,
                "dv_par_box": None,
                "qoi_colors": self.qoi_colors,
                "version": "1.0",
            }

            self.dv_par_box_mutex.lock()
            try:
                dv_par_box_copy = (
                    self.dv_par_box.copy() if self.dv_par_box is not None else None
                )
                data["dv_par_box"] = (
                    dv_par_box_copy.tolist() if dv_par_box_copy is not None else None
                )
            finally:
                self.dv_par_box_mutex.unlock()

            json_path = path
            h5_path = path.replace(".pfd", ".h5")

            try:
                serializable_data = self._to_serializable(data)
                with open(json_path, "w") as f:
                    json.dump(serializable_data, f, indent=2)
                if self.problem and (self.problem.samples or self.problem.results):
                    with h5py.File(h5_path, "w") as h5f:
                        if self.problem.samples:
                            for key, value in self.problem.samples.items():
                                h5f.create_dataset(f"samples/{key}", data=value)
                        if self.problem.results:
                            for key, value in self.problem.results.items():
                                h5f.create_dataset(f"results/{key}", data=value)
                QtWidgets.QMessageBox.information(
                    self, "Saved", "Project saved successfully."
                )
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Error", f"Failed to save: {e}")

    def load_project(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load Project", "", "PFD Project (*.pfd)"
        )
        if path:
            try:
                with open(path, "r") as f:
                    data = json.load(f)

                self.system_code = data.get("code")
                self.inputs = data.get("inputs", [])
                self.outputs = data.get("outputs", [])
                self.last_samples = data.get("last_samples")

                # Convert lists back to NumPy arrays for last_samples
                if self.last_samples:
                    self.last_samples["points"] = np.array(self.last_samples["points"])
                    self.last_samples["qoi_values"] = np.array(
                        self.last_samples["qoi_values"]
                    )
                    self.last_samples["is_good"] = np.array(
                        self.last_samples["is_good"]
                    )
                    self.last_samples["is_bad"] = np.array(self.last_samples["is_bad"])
                    self.last_samples["violation_idx"] = np.array(
                        self.last_samples["violation_idx"]
                    )

                dv_par_box_data = data.get("dv_par_box")
                self.dv_par_box = (
                    np.array(dv_par_box_data) if dv_par_box_data is not None else None
                )

                self.qoi_colors = data.get("qoi_colors", {})
                forbidden_greens = [
                    "#00aa00",
                    "#3cb44b",
                    "#bcf60c",
                    "#008080",
                    "#aaffc3",
                    "#808000",
                    "#00ff00",
                    "#008000",
                ]
                for name, color in self.qoi_colors.items():
                    if color.lower() in forbidden_greens:
                        idx = 0
                        if self.outputs and name in self.outputs:
                            idx = self.outputs.index(name)
                        self.qoi_colors[name] = self.default_colors[
                            idx % len(self.default_colors)
                        ]

                h5_path = path.replace(".pfd", ".h5")
                samples_data = {}
                results_data = {}

                if os.path.exists(h5_path):
                    try:
                        with h5py.File(h5_path, "r") as h5f:
                            if "samples" in h5f:
                                for key in h5f["samples"].keys():
                                    samples_data[key] = h5f[f"samples/{key}"][:]
                            if "results" in h5f:
                                for key in h5f["results"].keys():
                                    results_data[key] = h5f[f"results/{key}"][:]
                    except Exception:
                        logger.warning("Could not load HDF5 data", exc_info=True)

                p_data = data.get("problem_data")
                if p_data:
                    system_func = None
                    if self.system_code:
                        try:
                            system_func = self._execute_code_safely(self.system_code)
                        except Exception:
                            logger.warning(
                                "Could not restore the saved system function.",
                                exc_info=True,
                            )

                    self.problem = DesignProblem(
                        p_data.get("name", "Loaded_Model"),
                        p_data["n_samples"],
                    )
                    if system_func:
                        self.problem.set_system_model(system_func)

                    self.problem.design_variables = p_data["design_variables"]
                    self.problem.quantities_of_interest = p_data[
                        "quantities_of_interest"
                    ]

                    if samples_data:
                        self.problem.samples = samples_data
                    elif "samples" in p_data:
                        self.problem.samples = p_data["samples"]
                        # Convert lists to NumPy arrays
                        for key in self.problem.samples:
                            self.problem.samples[key] = np.array(
                                self.problem.samples[key]
                            )

                    if results_data:
                        self.problem.results = results_data
                    elif "results" in p_data:
                        self.problem.results = p_data["results"]
                        # Convert lists to NumPy arrays
                        for key in self.problem.results:
                            self.problem.results[key] = np.array(
                                self.problem.results[key]
                            )

                self.populate_tables_from_problem()
                self.update_all_plots()
                self.update_data_table()

                if self.problem and self.problem.system_model:
                    self.btn_compute_feasible.setEnabled(True)
                    self.btn_compute_family.setEnabled(True)
                    # Restoring a saved project: show its samples.
                    self._has_sampled = True
                    self.resample_box(silent=True)

            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Error", f"Failed to load: {e}")
