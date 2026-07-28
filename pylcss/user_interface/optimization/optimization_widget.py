# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE.
"""Main optimization workflow widget."""

from __future__ import annotations

import importlib.util
import logging
import os
import tempfile

import numpy as np
from PySide6 import QtCore, QtWidgets

from pylcss.config import TEMP_MODELS_DIR
from pylcss.optimization.evaluator import default_initial_value
from pylcss.optimization.workers import OptimizationWorker
from pylcss.system_modeling.problem import DesignProblem

from .components import (
    FEASIBILITY_TOL,
    OptimizationPlotsWidget,
    SolverSettingsWidget,
)

logger = logging.getLogger(__name__)

__all__ = [
    "OptimizationPlotsWidget",
    "OptimizationWidget",
    "SolverSettingsWidget",
]


class OptimizationWidget(QtWidgets.QWidget):
    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.problem = None
        self.worker = None
        self.models = []
        self.objectives = []
        self.constraints = []
        self.feasibility_tolerance = FEASIBILITY_TOL
        self.system_code = None

        self.init_ui()

    def init_ui(self):
        # Main Layout using Splitter for resizability
        main_layout = QtWidgets.QHBoxLayout(self)
        self.splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        main_layout.addWidget(self.splitter)

        # --- LEFT PANEL (Settings & Control) ---
        left_widget = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)

        # 1. Settings
        self.settings_widget = SolverSettingsWidget()
        self.settings_widget.system_changed.connect(self.load_selected_system)
        self.settings_widget.method_changed.connect(self._on_method_changed)
        left_layout.addWidget(self.settings_widget)

        # 2. Objectives Table
        grp_objs = QtWidgets.QGroupBox("Objectives")
        objs_layout = QtWidgets.QVBoxLayout(grp_objs)
        self.table_objectives = QtWidgets.QTableWidget(0, 4)
        self.table_objectives.setHorizontalHeaderLabels(
            ["Name", "Type", "Weight", "Scale"]
        )
        self.table_objectives.setToolTip(
            "Weights express preference. Reference Scale removes unit magnitude "
            "from scalar optimization; leave Auto to freeze the initial response."
        )
        self.table_objectives.horizontalHeader().setSectionResizeMode(
            QtWidgets.QHeaderView.Stretch
        )
        self.table_objectives.verticalHeader().setVisible(False)
        self.table_objectives.itemChanged.connect(self.on_objective_weight_changed)
        objs_layout.addWidget(self.table_objectives)
        left_layout.addWidget(grp_objs)

        # 3. Execution Control
        grp_exec = QtWidgets.QGroupBox("Execution")
        exec_layout = QtWidgets.QVBoxLayout(grp_exec)

        self.chk_use_current = QtWidgets.QCheckBox(
            "Use Current Values as Initial Guess"
        )
        self.chk_use_current.setToolTip(
            "Start optimization from the values currently in the design variables table/model."
        )
        exec_layout.addWidget(self.chk_use_current)

        btn_layout = QtWidgets.QHBoxLayout()
        self.btn_run = QtWidgets.QPushButton("Run Optimization")
        self.btn_run.setStyleSheet(
            "background-color: #2ecc71; color: white; font-weight: bold; padding: 6px;"
        )
        self.btn_run.clicked.connect(self.start_optimization)

        self.btn_stop = QtWidgets.QPushButton("Stop")
        self.btn_stop.setStyleSheet(
            "background-color: #e74c3c; color: white; font-weight: bold; padding: 6px;"
        )
        self.btn_stop.clicked.connect(self.stop_optimization)
        self.btn_stop.setEnabled(False)

        btn_layout.addWidget(self.btn_run)
        btn_layout.addWidget(self.btn_stop)

        self.progress_bar = QtWidgets.QProgressBar()
        self.lbl_status = QtWidgets.QLabel("Status: Idle")
        self.lbl_status.setWordWrap(True)

        exec_layout.addLayout(btn_layout)
        exec_layout.addWidget(self.progress_bar)
        exec_layout.addWidget(self.lbl_status)
        left_layout.addWidget(grp_exec)

        # Add Left Panel to Splitter
        self.splitter.addWidget(left_widget)

        # --- RIGHT PANEL (Plots) ---
        self.plots_widget = OptimizationPlotsWidget()
        self.splitter.addWidget(self.plots_widget)

        # Set initial sizes (Left: 400px, Right: remaining)
        self.splitter.setSizes([400, 800])

    def _on_method_changed(self, method):
        """Keep objective controls honest for scalar versus Pareto solvers."""
        pareto = str(method) == "NSGA-II"
        self.table_objectives.setHorizontalHeaderLabels(
            (
                ["Name", "Type", "Weight (scalar only)", "Scale (scalar only)"]
                if pareto
                else ["Name", "Type", "Weight", "Scale"]
            )
        )
        self.table_objectives.setToolTip(
            (
                "NSGA-II retains every selected objective using Pareto "
                "dominance. Scalar weights and reference scales are ignored."
                if pareto
                else "Weights express preference. Reference Scale removes unit "
                "magnitude from scalar optimization; leave Auto to freeze "
                "the initial response."
            )
        )
        for row in range(self.table_objectives.rowCount()):
            for column in (2, 3):
                item = self.table_objectives.item(row, column)
                if item is None:
                    continue
                flags = item.flags()
                if pareto:
                    item.setFlags(flags & ~QtCore.Qt.ItemIsEditable)
                    item.setToolTip("Ignored by NSGA-II Pareto dominance.")
                else:
                    item.setFlags(flags | QtCore.Qt.ItemIsEditable)
                    item.setToolTip("")
        self.plots_widget.optimization_method = str(method)
        total_tab = self.plots_widget.indexOf(self.plots_widget.plot_obj["widget"])
        if total_tab >= 0:
            self.plots_widget.setTabVisible(total_tab, not pareto)
        self.plots_widget._update_formulation_text()

    def load_models(self, models):
        self.models = models
        self.settings_widget.system_combo.clear()
        for m in models:
            name = m.name if hasattr(m, "name") else m["name"]
            self.settings_widget.system_combo.addItem(name)
        if models:
            self.load_selected_system()

    def load_selected_system(self):
        idx = self.settings_widget.system_combo.currentIndex()
        if idx < 0 or idx >= len(self.models):
            return

        m = self.models[idx]
        if hasattr(m, "name"):
            self.load_model_from_system_model(m)
        else:
            self.load_model(m["code"], m["inputs"], m["outputs"])

    def load_model(self, code, inputs, outputs):
        try:
            # Reusing original parsing logic
            self.system_code = code
            system_function = self._execute_code_safely(code)

            self.problem = DesignProblem("Optimization_Model", sample_size=3000)
            self.problem.set_system_model(system_function)

            for inp in inputs:
                self.problem.add_design_variable(
                    inp["name"],
                    inp.get("unit", "-"),
                    self._parse_float(inp["min"]),
                    self._parse_float(inp["max"]),
                )
            for out in outputs:
                self.problem.add_quantity_of_interest(
                    out["name"],
                    out.get("unit", "-"),
                    self._parse_float(out["req_min"]),
                    self._parse_float(out["req_max"]),
                    minimize=out.get("minimize", False),
                    maximize=out.get("maximize", False),
                )
            self.set_problem(self.problem)
        except Exception as e:
            self.lbl_status.setText(f"Error loading model: {str(e)}")
            logger.error(f"Model load error: {e}", exc_info=True)

    def load_model_from_system_model(self, system_model):
        self.load_model(
            system_model.source_code, system_model.inputs, system_model.outputs
        )

    def set_problem(self, problem):
        self.problem = problem
        self.objectives = [
            q
            for q in problem.quantities_of_interest
            if q.get("minimize") or q.get("maximize")
        ]
        self.constraints = [
            q
            for q in problem.quantities_of_interest
            if not (q.get("minimize") or q.get("maximize"))
        ]

        # Populate UI components
        self._populate_objectives_table()
        self.plots_widget.set_problem(problem, self.objectives, self.constraints)
        self.lbl_status.setText(f"Loaded: {problem.name}")

    def _populate_objectives_table(self):
        self.table_objectives.blockSignals(True)
        self.table_objectives.setRowCount(len(self.objectives))
        for i, obj in enumerate(self.objectives):
            self.table_objectives.setItem(i, 0, QtWidgets.QTableWidgetItem(obj["name"]))
            type_str = "Min" if obj.get("minimize") else "Max"
            self.table_objectives.setItem(i, 1, QtWidgets.QTableWidgetItem(type_str))
            self.table_objectives.setItem(
                i, 2, QtWidgets.QTableWidgetItem(str(obj.get("weight", 1.0)))
            )
            scale = obj.get("scale")
            self.table_objectives.setItem(
                i,
                3,
                QtWidgets.QTableWidgetItem(
                    "Auto" if scale in (None, "") else str(scale)
                ),
            )
        self.table_objectives.blockSignals(False)
        self._on_method_changed(self.settings_widget.combo_method.currentText())

    def on_objective_weight_changed(self, item):
        if item.column() not in (2, 3):
            return
        try:
            if item.column() == 2:
                val = float(item.text())
                if not np.isfinite(val) or val < 0:
                    raise ValueError
                self.objectives[item.row()]["weight"] = val
            else:
                text = item.text().strip()
                if not text or text.lower() == "auto":
                    self.objectives[item.row()].pop("scale", None)
                else:
                    val = float(text)
                    if not np.isfinite(val) or val <= 0:
                        raise ValueError
                    self.objectives[item.row()]["scale"] = val
            self.plots_widget._update_formulation_text()  # Update formulation display
        except ValueError:
            self.table_objectives.blockSignals(True)
            if item.column() == 2:
                item.setText(str(self.objectives[item.row()].get("weight", 1.0)))
            else:
                scale = self.objectives[item.row()].get("scale")
                item.setText("Auto" if scale is None else str(scale))
            self.table_objectives.blockSignals(False)

    def start_optimization(self):
        if not self.problem:
            return

        if not self.objectives:
            QtWidgets.QMessageBox.warning(
                self, "No Objectives", "Please define at least one objective."
            )
            return

        solver_settings = self.settings_widget.get_config()
        if solver_settings.get("method") == "NSGA-II" and len(self.objectives) < 2:
            QtWidgets.QMessageBox.warning(
                self,
                "NSGA-II Needs Multiple Objectives",
                "Select at least two objectives or choose a scalar solver.",
            )
            return

        # Stop any existing optimization
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            if not self.worker.wait(250):
                QtWidgets.QMessageBox.information(
                    self,
                    "Optimization Still Stopping",
                    "The current optimization is still shutting down. Wait a moment and start again.",
                )
                return

        # UI State
        self.btn_run.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.plots_widget.clear_plots()
        self.progress_bar.setRange(0, 0)  # Indeterminate
        self.lbl_status.setText("Optimizing...")

        self.plots_widget.mark_running()

        # Setup Data
        x0 = []

        use_current = self.chk_use_current.isChecked()

        scaling_enabled = bool(solver_settings.get("scaling", True))
        scaling_mode = str(solver_settings.get("scaling_mode", "auto"))
        self.feasibility_tolerance = float(
            solver_settings.get(
                "feasibility_tol", solver_settings.get("tol", FEASIBILITY_TOL)
            )
        )

        for dv in self.problem.design_variables:
            mn, mx = float(dv["min"]), float(dv["max"])

            if use_current:
                # Use current value, clamped to bounds
                fallback = default_initial_value(
                    mn,
                    mx,
                    scaling=scaling_enabled,
                    scaling_mode=scaling_mode,
                )
                val = float(dv.get("value", fallback))
                # basic clamping to ensure we don't start out of bounds (which crashes some solvers)
                if np.isfinite(mn) and val < mn:
                    val = mn
                if np.isfinite(mx) and val > mx:
                    val = mx
                x0.append(val)
            else:
                x0.append(
                    default_initial_value(
                        mn,
                        mx,
                        scaling=scaling_enabled,
                        scaling_mode=scaling_mode,
                    )
                )

        setup_data = {
            "variables": self.problem.design_variables,
            "objectives": self.objectives,
            "constraints": self.constraints,
            "x0": np.array(x0),
            "parameters": {p["name"]: p["value"] for p in self.problem.parameters},
        }

        # Worker
        self.worker = OptimizationWorker(
            self.problem.system_model, setup_data, solver_settings
        )
        self.worker.progress.connect(self.plots_widget.update_data)
        self.worker.finished.connect(self.on_finished)
        self.worker.error.connect(self.on_error)
        self.worker.start()

    def stop_optimization(self):
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            # Keep the QThread referenced until its finished/error signal. A
            # running QThread destroyed by the UI can terminate the process.
            self.btn_run.setEnabled(False)
            self.btn_stop.setEnabled(False)
            self.lbl_status.setText("Stopping at the next solver checkpoint…")

    def on_finished(self, result):
        self.worker = None
        self.btn_run.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(100)

        self.feasibility_tolerance = float(
            getattr(result, "feasibility_tolerance", FEASIBILITY_TOL)
        )
        is_feasible = result.max_violation <= self.feasibility_tolerance
        if result.success and getattr(result, "converged", result.success):
            msg, color = "Converged", "#27ae60"
        elif result.success and is_feasible:
            msg, color = "Done (Max Iter / Tol)", "#27ae60"
        else:
            msg, color = "Failed", "#e74c3c"

        if result.max_violation > self.feasibility_tolerance:
            msg += " (Constraints Violated)"
            color = "#e74c3c"

        self.lbl_status.setText(f"{msg}: {result.message}")

        # Write the final solution into the Results tab and back to the model.
        if result.x is not None:
            self.plots_widget.set_results_final(
                result, f"{msg}: {result.message}", color
            )
            for i, val in enumerate(result.x):
                self.problem.design_variables[i]["value"] = float(val)

    def on_error(self, msg):
        self.worker = None
        self.btn_run.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.lbl_status.setText("Error occurred")
        QtWidgets.QMessageBox.critical(self, "Error", msg)

    # --- Utils ---
    def _parse_float(self, val):
        if isinstance(val, (int, float)):
            return float(val)
        if isinstance(val, str):
            v = val.strip().lower()
            if v in ("inf", "+inf"):
                return float("inf")
            if v == "-inf":
                return float("-inf")
            try:
                return float(val)
            except ValueError:
                pass
        return 0.0

    def _execute_code_safely(self, code):
        # 1. Ensure the directory exists
        os.makedirs(TEMP_MODELS_DIR, exist_ok=True)

        # 2. Pass the 'dir' argument to NamedTemporaryFile
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, dir=TEMP_MODELS_DIR
        ) as f:
            f.write(code)
            temp_file = f.name
        spec = importlib.util.spec_from_file_location("temp_module", temp_file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        # Find the system_function
        system_function = None
        for attr_name in dir(module):
            if attr_name.startswith("system_function") and callable(
                getattr(module, attr_name)
            ):
                system_function = getattr(module, attr_name)
                break
        if system_function is None:
            raise AttributeError("system_function not found in generated code")
        return system_function

    def save_to_folder(self, folder_path: str):
        """
        Saves the current optimization setup (variables, objectives, constraints, settings)
        to a JSON file in the project folder.
        """
        data = {
            "variables": self.problem.design_variables if self.problem else [],
            "objectives": self.objectives,
            "constraints": self.constraints,
            "settings": self.settings_widget.get_config(),
        }

        file_path = os.path.join(folder_path, "optimization_setup.json")
        from pylcss.io_manager.project_io import atomic_json_dump

        atomic_json_dump(data, file_path, indent=4)

    def load_from_folder(self, folder_path: str):
        """
        Loads the optimization setup from a JSON file and populates the UI.
        """
        file_path = os.path.join(folder_path, "optimization_setup.json")
        if not os.path.exists(file_path):
            return

        from pylcss.io_manager.project_io import load_json_object

        data = load_json_object(file_path)

        # 1. Load Variables
        if self.problem and "variables" in data:
            self.problem.design_variables = data["variables"]

        # 2. Load Objectives
        if "objectives" in data:
            self.objectives = data["objectives"]
            self._populate_objectives_table()

        # 3. Load Constraints
        if "constraints" in data:
            self.constraints = data["constraints"]

        # 4. Load Settings
        if "settings" in data:
            self._apply_settings_to_ui(data["settings"])

    def _apply_settings_to_ui(self, settings):
        # Apply solver settings to the UI
        if "method" in settings:
            idx = self.settings_widget.combo_method.findText(settings["method"])
            if idx >= 0:
                self.settings_widget.combo_method.setCurrentIndex(idx)

        # Advanced parameters live in SolverSettingsWidget.settings (edited via
        # the pop-up dialog), not as individual widgets, so merge them there.
        stored = self.settings_widget.settings
        for key, value in settings.items():
            if key == "method":
                continue
            stored[key] = value
        method = str(
            settings.get("method", self.settings_widget.combo_method.currentText())
        )
        budget_key = self.settings_widget._budget_key(method)
        if (
            budget_key is not None
            and "maxiter" in settings
            and budget_key not in settings
        ):
            stored[budget_key] = settings["maxiter"]
