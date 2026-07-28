# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE.
"""SolutionModelMixin behavior for solution-space analysis."""

from __future__ import annotations

import logging

import networkx as nx
import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from pylcss.system_modeling.problem import DesignProblem

from .plotting import (
    ColorConfigDialog,
)

logger = logging.getLogger(__name__)

__all__ = ["SolutionModelMixin"]


class SolutionModelMixin:
    def _execute_code_safely(self, code):
        """Execute code safely using exec with a custom filename.
        Avoids temporary files which can cause issues with pickling/dill."""
        namespace = {}
        try:
            # Compile with a descriptive filename for tracebacks
            bytecode = compile(code, "<dynamic_system_model>", "exec")
            exec(bytecode, namespace)
        except Exception as e:
            logger.exception("Error executing code")
            raise e

        if "system_function" in namespace:
            return namespace["system_function"]
        else:
            # Fallback: look for any callable
            for k, v in namespace.items():
                if callable(v) and k not in [
                    "__builtins__",
                    "np",
                    "joblib",
                    "os",
                    "sys",
                ]:
                    return v
            raise AttributeError("system_function not found in generated code")

    def load_model(self, code, inputs, outputs, name=None):
        """
        Loads the model code and populates the tables (from Modeling Tab).
        inputs: List of dicts {'name', 'unit', 'min', 'max'}
        outputs: List of dicts {'name', 'unit', 'req_min', 'req_max'}
        name: Optional system name
        """
        self._reset_multimodal_state()
        self.system_code = code
        self.system_name = name

        # Initialize Problem Object immediately for resampling
        try:
            # Use SystemModel to create a persistent file for multiprocessing support
            from pylcss.system_modeling.model import SystemModel

            model_name = name if name else "Loaded_Model"

            sm = SystemModel.from_code_string(
                model_name, self.system_code, inputs, outputs
            )
            system_func = sm.system_function

            # Use the shared plot sample setting, defaulting to 300.
            if hasattr(self, "sample_size_spin"):
                n_samples = self.sample_size_spin.value()
            else:
                n_samples = 300
            # Use provided name, else default
            model_name = name if name else "Loaded_Model"
            self.problem = DesignProblem(model_name, n_samples)
            self.problem.set_system_model(system_func)

            for inp in inputs:
                self.problem.add_design_variable(
                    inp["name"], inp.get("unit", "-"), inp["min"], inp["max"]
                )

            for out in outputs:
                minimize = out.get("minimize", False)
                maximize = out.get("maximize", False)
                self.problem.add_quantity_of_interest(
                    out["name"],
                    out.get("unit", "-"),
                    out["req_min"],
                    out["req_max"],
                    minimize=minimize,
                    maximize=maximize,
                    weight=1.0,
                    display_name=out.get("display_name", out["name"]),
                    show_in_legend=out.get("show_in_legend", True),
                )

        except Exception:
            logger.warning("Failed to initialize problem object", exc_info=True)
            self.problem = None

        # Extract names for internal use
        self.inputs = [i["name"] for i in inputs]
        self.outputs = [o["name"] for o in outputs]
        # Store units for axis labels
        self.input_units = {i["name"]: i.get("unit", "-") for i in inputs}
        self.output_units = {o["name"]: o.get("unit", "-") for o in outputs}

        # Update Combo Boxes
        self.combo_add_x.clear()
        self.combo_add_y.clear()

        all_vars = self.inputs + self.outputs
        self.combo_add_x.addItems(all_vars)
        self.combo_add_y.addItems(all_vars)

        # Set defaults if available
        if len(self.inputs) >= 1:
            self.combo_add_x.setCurrentIndex(0)
        if len(self.inputs) >= 2:
            self.combo_add_y.setCurrentIndex(1)
        elif len(self.inputs) >= 1:
            self.combo_add_y.setCurrentIndex(0)

        try:
            self.dsl = np.array([float(i.get("min", 0)) for i in inputs])
            self.dsu = np.array([float(i.get("max", 1)) for i in inputs])
        except (TypeError, ValueError):
            logger.warning("Loaded design-variable bounds are invalid.")
            self.dsl = None
            self.dsu = None

        # Populate DV Table
        self.dv_table.blockSignals(True)
        self.dv_table.setRowCount(len(inputs))
        self.dv_par_box = np.zeros((len(inputs), 2))

        def safe_float(val):
            if isinstance(val, (int, float)):
                return float(val)
            if isinstance(val, str):
                if val.lower() == "inf":
                    return float("inf")
                if val.lower() == "-inf":
                    return float("-inf")
                try:
                    return float(val)
                except Exception:
                    return val
            return val

        for i, inp in enumerate(inputs):
            min_val = safe_float(inp.get("min", 0))
            max_val = safe_float(inp.get("max", 0))
            self.dv_table.setItem(i, 0, QtWidgets.QTableWidgetItem(inp["name"]))
            self.dv_table.setItem(
                i, 1, QtWidgets.QTableWidgetItem(inp.get("unit", "-"))
            )
            self.dv_table.setItem(i, 2, QtWidgets.QTableWidgetItem(str(min_val)))
            self.dv_table.setItem(i, 3, QtWidgets.QTableWidgetItem(str(max_val)))
            # Initialize Solution Space as Design Space
            self.dv_table.setItem(i, 4, QtWidgets.QTableWidgetItem(str(min_val)))
            self.dv_table.setItem(i, 5, QtWidgets.QTableWidgetItem(str(max_val)))
            self.dv_par_box[i, 0] = min_val
            self.dv_par_box[i, 1] = max_val

        self.dv_table.blockSignals(False)

        # Populate QoI Table
        self.qoi_table.setRowCount(len(outputs))
        for i, out in enumerate(outputs):
            req_min_val = safe_float(out.get("req_min", 0))
            req_max_val = safe_float(out.get("req_max", 0))
            self.qoi_table.setItem(i, 0, QtWidgets.QTableWidgetItem(out["name"]))
            self.qoi_table.setItem(
                i, 1, QtWidgets.QTableWidgetItem(out.get("unit", "-"))
            )
            self.qoi_table.setItem(i, 2, QtWidgets.QTableWidgetItem(str(req_min_val)))
            self.qoi_table.setItem(i, 3, QtWidgets.QTableWidgetItem(str(req_max_val)))
            self.qoi_table.setItem(i, 4, QtWidgets.QTableWidgetItem("Auto"))  # Plot Min
            self.qoi_table.setItem(i, 5, QtWidgets.QTableWidgetItem("Auto"))  # Plot Max
            # Minimize checkbox
            min_item = QtWidgets.QTableWidgetItem()
            min_item.setCheckState(
                QtCore.Qt.Checked if out.get("minimize", False) else QtCore.Qt.Unchecked
            )
            self.qoi_table.setItem(i, 6, min_item)
            # Maximize checkbox
            max_item = QtWidgets.QTableWidgetItem()
            max_item.setCheckState(
                QtCore.Qt.Checked if out.get("maximize", False) else QtCore.Qt.Unchecked
            )
            self.qoi_table.setItem(i, 7, max_item)

            # Weight
            weight_item = QtWidgets.QTableWidgetItem(str(out.get("weight", 1.0)))
            self.qoi_table.setItem(i, 8, weight_item)

            # Disable req fields if minimize or maximize is checked
            if out.get("minimize", False) or out.get("maximize", False):
                min_req_item = self.qoi_table.item(i, 2)
                max_req_item = self.qoi_table.item(i, 3)
                if min_req_item:
                    min_req_item.setFlags(
                        min_req_item.flags() & ~QtCore.Qt.ItemIsEditable
                    )
                if max_req_item:
                    max_req_item.setFlags(
                        max_req_item.flags() & ~QtCore.Qt.ItemIsEditable
                    )

        # Init colors
        self.qoi_colors = {}
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
        for i, out in enumerate(outputs):
            name = out["name"]
            color = None
            if "color" in out and out["color"]:
                color = out["color"]
                if color.lower() in forbidden_greens:
                    color = None  # Reset if green

            if color:
                self.qoi_colors[name] = color
            else:
                self.qoi_colors[name] = self.default_colors[
                    i % len(self.default_colors)
                ]

        self.btn_compute_feasible.setEnabled(True)
        self.btn_compute_multimodal.setEnabled(self.problem is not None)
        # Check if there are any objectives defined
        has_objectives = self.problem is not None and any(
            qoi.get("minimize", False) or qoi.get("maximize", False)
            for qoi in self.problem.quantities_of_interest
        )
        # Enable optimization controls if objectives exist
        self.chk_include_optimization.setEnabled(has_objectives)
        self.btn_compute_family.setEnabled(True)
        self.update_all_plots()

        # Auto-create default plot if none exists
        if not self.plot_widgets and len(self.inputs) >= 2:
            self.add_plot(self.inputs[0], self.inputs[1], do_resample=False)
        elif not self.plot_widgets and len(self.inputs) == 1:
            if self.outputs:
                self.add_plot(self.inputs[0], self.outputs[0], do_resample=False)
            else:
                self.add_plot(self.inputs[0], self.inputs[0], do_resample=False)

        # Sampling is explicit (it can be expensive — one model evaluation per
        # sample), so it does not fire automatically on load. Enable Resample so
        # the user can trigger it (or a cheap small-N probe) when ready.
        self.btn_resample.setEnabled(True)

    def load_model_from_system_model(self, system_model):
        """
        Load a model from a SystemModel instance.
        """
        self._reset_multimodal_state()
        # Use the compiled function directly
        self.system_code = system_model.source_code

        # Initialize Problem Object
        try:
            # Use the shared plot sample setting, defaulting to 300.
            if hasattr(self, "sample_size_spin"):
                n_samples = self.sample_size_spin.value()
            else:
                n_samples = 300
            model_name = system_model.name
            self.problem = DesignProblem(model_name, n_samples)
            self.problem.set_system_model(system_model.system_function)
            self.problem.set_system_code(system_model.source_code)

            for inp in system_model.inputs:
                self.problem.add_design_variable(
                    inp["name"], inp.get("unit", "-"), inp["min"], inp["max"]
                )

            for out in system_model.outputs:
                minimize = out.get("minimize", False)
                maximize = out.get("maximize", False)
                # Get weight from table if available, otherwise use default
                weight = 1.0
                if hasattr(self, "qoi_table") and self.qoi_table.rowCount() > 0:
                    for i, table_out in enumerate(system_model.outputs):
                        if (
                            i < self.qoi_table.rowCount()
                            and table_out["name"] == out["name"]
                        ):
                            try:
                                weight = float(self.qoi_table.item(i, 8).text())
                            except (ValueError, AttributeError):
                                weight = 1.0
                            break
                self.problem.add_quantity_of_interest(
                    out["name"],
                    out.get("unit", "-"),
                    out["req_min"],
                    out["req_max"],
                    minimize=minimize,
                    maximize=maximize,
                    weight=weight,
                    display_name=out.get("display_name", out["name"]),
                    show_in_legend=out.get("show_in_legend", True),
                )

        except Exception:
            logger.warning("Failed to initialize problem object", exc_info=True)
            self.problem = None

        # Extract names for internal use
        self.inputs = system_model.get_input_names()
        self.outputs = system_model.get_output_names()
        # Store units for axis labels
        self.input_units = {i["name"]: i.get("unit", "-") for i in system_model.inputs}
        self.output_units = {
            o["name"]: o.get("unit", "-") for o in system_model.outputs
        }

        try:

            def safe_val(v, default):
                try:
                    return float(v)
                except (TypeError, ValueError):
                    return default

            self.dsl = np.array(
                [safe_val(i.get("min", 0), 0.0) for i in system_model.inputs]
            )
            self.dsu = np.array(
                [safe_val(i.get("max", 1), 1.0) for i in system_model.inputs]
            )
        except (AttributeError, TypeError, ValueError):
            logger.warning("System-model bounds are invalid.", exc_info=True)
            self.dsl = None
            self.dsu = None

        # Populate DV Table
        self.dv_table.blockSignals(True)
        self.dv_table.setRowCount(len(system_model.inputs))
        self.dv_par_box = np.zeros((len(system_model.inputs), 2))

        def safe_float(val):
            if isinstance(val, (int, float)):
                return float(val)
            if isinstance(val, str):
                if val.lower() == "inf":
                    return float("inf")
                if val.lower() == "-inf":
                    return float("-inf")
                try:
                    return float(val)
                except Exception:
                    return val
            return val

        for i, inp in enumerate(system_model.inputs):
            min_val = safe_float(inp.get("min", 0))
            max_val = safe_float(inp.get("max", 1))
            self.dv_par_box[i, 0] = min_val
            self.dv_par_box[i, 1] = max_val

            # Set table values
            display_name = inp.get("display_name", inp["name"])
            self.dv_table.setItem(i, 0, QtWidgets.QTableWidgetItem(display_name))
            self.dv_table.setItem(
                i, 1, QtWidgets.QTableWidgetItem(inp.get("unit", "-"))
            )
            self.dv_table.setItem(i, 2, QtWidgets.QTableWidgetItem(str(min_val)))
            self.dv_table.setItem(i, 3, QtWidgets.QTableWidgetItem(str(max_val)))
            # Initialize Solution Space as Design Space
            self.dv_table.setItem(i, 4, QtWidgets.QTableWidgetItem(str(min_val)))
            self.dv_table.setItem(i, 5, QtWidgets.QTableWidgetItem(str(max_val)))

        self.dv_table.blockSignals(False)
        self.qoi_table.blockSignals(False)

        # Populate QoI Table
        self.qoi_table.setRowCount(len(system_model.outputs))
        for i, out in enumerate(system_model.outputs):
            req_min_val = safe_float(out.get("req_min", 0))
            req_max_val = safe_float(out.get("req_max", 0))
            display_name = out.get("display_name", out["name"])
            self.qoi_table.setItem(i, 0, QtWidgets.QTableWidgetItem(display_name))
            self.qoi_table.setItem(
                i, 1, QtWidgets.QTableWidgetItem(out.get("unit", "-"))
            )
            self.qoi_table.setItem(i, 2, QtWidgets.QTableWidgetItem(str(req_min_val)))
            self.qoi_table.setItem(i, 3, QtWidgets.QTableWidgetItem(str(req_max_val)))
            self.qoi_table.setItem(i, 4, QtWidgets.QTableWidgetItem("Auto"))  # Plot Min
            self.qoi_table.setItem(i, 5, QtWidgets.QTableWidgetItem("Auto"))  # Plot Max
            # Minimize checkbox
            min_item = QtWidgets.QTableWidgetItem()
            min_item.setCheckState(
                QtCore.Qt.Checked if out.get("minimize", False) else QtCore.Qt.Unchecked
            )
            self.qoi_table.setItem(i, 6, min_item)
            # Maximize checkbox
            max_item = QtWidgets.QTableWidgetItem()
            max_item.setCheckState(
                QtCore.Qt.Checked if out.get("maximize", False) else QtCore.Qt.Unchecked
            )
            self.qoi_table.setItem(i, 7, max_item)

            # Weight
            weight_item = QtWidgets.QTableWidgetItem(str(out.get("weight", 1.0)))
            self.qoi_table.setItem(i, 8, weight_item)

            # Disable req fields if minimize or maximize is checked
            if out.get("minimize", False) or out.get("maximize", False):
                min_req_item = self.qoi_table.item(i, 2)
                max_req_item = self.qoi_table.item(i, 3)
                if min_req_item:
                    min_req_item.setFlags(
                        min_req_item.flags() & ~QtCore.Qt.ItemIsEditable
                    )
                if max_req_item:
                    max_req_item.setFlags(
                        max_req_item.flags() & ~QtCore.Qt.ItemIsEditable
                    )

        # Init colors
        self.qoi_colors = {}
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
        for i, out in enumerate(system_model.outputs):
            name = out["name"]
            color = None
            if "color" in out and out["color"]:
                color = out["color"]
                if color.lower() in forbidden_greens:
                    color = None  # Reset if green

            if color:
                self.qoi_colors[name] = color
            else:
                self.qoi_colors[name] = self.default_colors[
                    i % len(self.default_colors)
                ]

        self.btn_compute_feasible.setEnabled(True)
        self.btn_compute_multimodal.setEnabled(self.problem is not None)
        # Check if there are any objectives defined
        has_objectives = self.problem is not None and any(
            qoi.get("minimize", False) or qoi.get("maximize", False)
            for qoi in self.problem.quantities_of_interest
        )
        # Enable optimization controls if objectives exist
        self.chk_include_optimization.setEnabled(has_objectives)
        self.btn_compute_family.setEnabled(True)
        self.update_all_plots()

        # Auto-create default plot if none exists
        if not self.plot_widgets and len(self.inputs) >= 2:
            self.add_plot(self.inputs[0], self.inputs[1], do_resample=False)
        elif not self.plot_widgets and len(self.inputs) == 1:
            if self.outputs:
                self.add_plot(self.inputs[0], self.outputs[0], do_resample=False)
            else:
                self.add_plot(self.inputs[0], self.inputs[0], do_resample=False)

        # Sampling is explicit (it can be expensive — one model evaluation per
        # sample), so it does not fire automatically on load. Enable Resample so
        # the user can trigger it (or a cheap small-N probe) when ready.
        self.btn_resample.setEnabled(True)

    def load_models(self, models):
        """
        Loads multiple models and allows selection.
        Models can be either SystemModel instances or legacy dict format.
        """
        self.models = models
        self.system_combo.clear()
        for m in models:
            # Handle both SystemModel instances and legacy dicts
            name = m.name if hasattr(m, "name") else m["name"]
            self.system_combo.addItem(name)

        if self.models:
            self.system_combo.setCurrentIndex(0)
            self.load_selected_system()

        # Refresh ADG system list when models are loaded
        if hasattr(self, "refresh_adg_system_list"):
            self.refresh_adg_system_list()

    def create_merged_model(self, models):
        """
        Creates a merged model from multiple subsystems.
        Detects dependencies based on shared variable names.
        """
        # Collect all variables
        all_inputs = {}
        all_outputs = {}

        for model in models:
            for inp in model["inputs"]:
                name = inp["name"]
                if name not in all_inputs:
                    all_inputs[name] = inp
            for out in model["outputs"]:
                name = out["name"]
                if name not in all_outputs:
                    all_outputs[name] = out

        # Build dependency graph
        G = nx.DiGraph()
        for i in range(len(models)):
            G.add_node(i)

        for i, model_a in enumerate(models):
            for inp in model_a["inputs"]:
                inp_name = inp["name"]
                for j, model_b in enumerate(models):
                    if i != j:
                        for out in model_b["outputs"]:
                            if inp_name == out["name"]:
                                G.add_edge(j, i)  # b provides input to a

        # Topological sort for execution order
        try:
            order = list(nx.topological_sort(G))
        except nx.NetworkXError:
            raise ValueError("Circular dependency detected in models")

        # Identify global inputs and outputs
        input_names = set(all_inputs.keys())
        output_names = set(all_outputs.keys())

        global_inputs = sorted(list(input_names - output_names))
        global_outputs = sorted(list(output_names - input_names))

        if not global_outputs:
            raise ValueError("No global outputs found (all outputs are used as inputs)")

        # Generate merged code
        code = "import numpy as np\n\n"

        # Add each model's function with unique name
        for i, model in enumerate(models):
            model_code = model["code"]
            code += model_code + "\n\n"

        # Merged system_function
        code += "def system_function(**kwargs):\n"

        # Extract global inputs
        for name in global_inputs:
            code += f"    {name} = kwargs['{name}']\n"

        code += "    intermediates = {}\n\n"

        # Execute models in order
        for idx in order:
            model = models[idx]
            code += f"    # Execute model {idx} ({model['name']})\n"

            # Build call arguments
            call_args = []
            for inp in model["inputs"]:
                name = inp["name"]
                if name in global_inputs:
                    call_args.append(f"{name}={name}")
                else:
                    call_args.append(f"{name}=intermediates['{name}']")

            call_str = ", ".join(call_args)
            code += f"    outputs_{idx} = system_function_{idx}({call_str})\n"

            # Store outputs in intermediates
            for out in model["outputs"]:
                name = out["name"]
                code += f"    intermediates['{name}'] = outputs_{idx}['{name}']\n"

            code += "\n"

        # Return global outputs
        code += "    return {\n"
        for name in global_outputs:
            code += f"        '{name}': intermediates['{name}'],\n"
        code += "    }\n"

        # Create merged model dict
        merged_inputs = [all_inputs[name] for name in global_inputs]
        merged_outputs = [all_outputs[name] for name in global_outputs]

        return {
            "name": "Merged",
            "code": code,
            "inputs": merged_inputs,
            "outputs": merged_outputs,
        }

    def on_system_changed(self):
        self.load_selected_system()

    def load_selected_system(self):
        idx = self.system_combo.currentIndex()
        if idx >= 0 and idx < len(self.models):
            m = self.models[idx]
            # Clear existing plots when switching systems
            self.clear_all_plots()
            # Clear cached samples to avoid QoI count mismatches
            self.last_samples = None
            # A freshly loaded/forwarded model must not auto-sample until the user
            # explicitly clicks Resample or Compute.
            self._has_sampled = False

            # Handle both SystemModel instances and legacy dicts
            if hasattr(m, "name"):  # SystemModel instance
                self.system_code = m.source_code
                self.load_model_from_system_model(m)
            else:  # Legacy dict format
                if m["name"] == "Merged":
                    self.system_code = m["code"]
                    self.load_model(m["code"], m["inputs"], m["outputs"], m["name"])
                else:
                    self.load_model(m["code"], m["inputs"], m["outputs"], m["name"])

            # Update title with system name
            model_name = m.name if hasattr(m, "name") else m["name"]
            if hasattr(self, "lbl_global_title") and model_name:
                self.lbl_global_title.setText(f"Solution Spaces for {model_name}")

    def configure_colors(self):
        if not self.problem:
            return
        qoi_names = [q["name"] for q in self.problem.quantities_of_interest]
        dialog = ColorConfigDialog(qoi_names, self.qoi_colors, self)
        if dialog.exec_() == QtWidgets.QDialog.Accepted:
            self.qoi_colors = dialog.get_colors()
            self.update_all_plots()

    def view_source_code(self):
        if not self.system_code:
            QtWidgets.QMessageBox.warning(self, "Warning", "No system loaded.")
            return

        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Source Code")
        dialog.resize(800, 600)
        layout = QtWidgets.QVBoxLayout(dialog)
        text_edit = QtWidgets.QTextEdit()
        text_edit.setPlainText(self.system_code)
        text_edit.setReadOnly(True)
        text_edit.setFont(QtGui.QFont("Consolas", 10))
        layout.addWidget(text_edit)
        dialog.exec_()
