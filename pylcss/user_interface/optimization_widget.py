# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

import numpy as np
import logging
import colorsys
import tempfile
import importlib.util
import os
import json
from typing import List, Dict, Any, Optional

from PySide6 import QtWidgets, QtCore, QtGui
import pyqtgraph as pg

from ..problem_definition.problem_setup import XRayProblem
from ..user_interface.text_utils import format_html
from ..optimization.workers import OptimizationWorker
from ..config import optimization_config, SOLVER_DESCRIPTIONS, TEMP_MODELS_DIR
from .optimization_settings_dialog import OptimizationSettingsDialog  # New import

logger = logging.getLogger(__name__)

# --- Helper Functions ---

def get_plot_color(index: int, total_lines: int) -> str:
    """Returns a consistent color based on index."""
    if total_lines <= 0: return '#3498db'
    # Use golden angle approximation for distinct colors
    hue = (index * 0.618033988749895) % 1.0
    saturation = 0.7 + (index % 3) * 0.1
    value = 0.8 + (index % 2) * 0.1
    rgb = colorsys.hsv_to_rgb(hue, saturation, value)
    return '#{:02x}{:02x}{:02x}'.format(int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255))


# --- Sub-Component: Solver Configuration ---

class SolverSettingsWidget(QtWidgets.QWidget):
    """
    Handles algorithm selection, system selection, and opens advanced settings.
    """
    method_changed = QtCore.Signal(str)
    system_changed = QtCore.Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.settings = {} # Stores the config dictionary
        self.init_ui()
        # Initialize default settings
        self.settings = self._get_default_settings()

    def init_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        
        # Group Box instead of Tabs
        grp = QtWidgets.QGroupBox("Solver Setup")
        form_layout = QtWidgets.QFormLayout(grp)
        form_layout.setContentsMargins(10, 15, 10, 10)

        # 1. System Selection
        self.system_combo = QtWidgets.QComboBox()
        self.system_combo.currentIndexChanged.connect(lambda idx: self.system_changed.emit(idx))
        form_layout.addRow("System Model:", self.system_combo)

        # 2. Algorithm Selection
        self.combo_method = QtWidgets.QComboBox()
        self.combo_method.addItems(list(SOLVER_DESCRIPTIONS.keys()))
        self.combo_method.currentTextChanged.connect(self.on_method_changed)
        
        btn_info = QtWidgets.QPushButton("?")
        btn_info.setFixedWidth(25)
        btn_info.setToolTip("Show details about the selected algorithm")
        btn_info.clicked.connect(self.show_algorithm_info)
        
        h_algo = QtWidgets.QHBoxLayout()
        h_algo.addWidget(self.combo_method)
        h_algo.addWidget(btn_info)
        form_layout.addRow("Algorithm:", h_algo)
        
        # 3. Settings Button (The pop-up trigger)
        self.btn_settings = QtWidgets.QPushButton("Advanced Settings")
        self.btn_settings.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_FileDialogDetailedView))
        self.btn_settings.clicked.connect(self.open_settings_dialog)
        form_layout.addRow("", self.btn_settings)

        layout.addWidget(grp)
        layout.addStretch()

    def on_method_changed(self, method_name):
        self.method_changed.emit(method_name)

    def open_settings_dialog(self):
        """Opens the pop-up dialog."""
        current_method = self.combo_method.currentText()
        dialog = OptimizationSettingsDialog(current_method, self.settings, self)
        if dialog.exec():
            self.settings = dialog.get_settings()

    def get_config(self):
        """Returns the full configuration dict (method + params)."""
        config = self.settings.copy()
        config['method'] = self.combo_method.currentText()
        return config

    def _get_default_settings(self):
        """Returns defaults matching pylcss.config"""
        return {
            'maxiter': optimization_config.DEFAULT_MAX_ITERATIONS,
            'tol': optimization_config.DEFAULT_TOLERANCE,
            'atol': 1e-8,
            'scaling': True,
            'objective_scale': 1.0,
            'maxfun': 15000,
            'popsize': 15,
            'mutation': (0.5, 1.0),
            'recombination': 0.7,
            'strategy': 'best1bin',
            'optimizer_name': 'NGOpt',
            'num_workers': 1
        }

    def show_algorithm_info(self):
        method = self.combo_method.currentText()
        info = SOLVER_DESCRIPTIONS.get(method, {})
        if not info:
            text = "No description available."
        else:
            text = f"""
            <h3>{info.get('name', method)}</h3>
            <p><b>Description:</b> {info.get('description', '-')}</p>
            <p><b>Best For:</b> {info.get('best_for', '-')}</p>
            <hr>
            <ul>
                <li><b>Speed:</b> {info.get('speed', '-')}</li>
                <li><b>Robustness:</b> {info.get('robustness', '-')}</li>
            </ul>
            """
        QtWidgets.QMessageBox.information(self, f"Algorithm Info: {method}", text)


# --- Sub-Component: Plotting Manager ---

class OptimizationPlotsWidget(QtWidgets.QTabWidget):
    """
    Manages the tabs for Cost, Design Variables, Constraints, etc.
    Encapsulates all pyqtgraph logic.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.problem = None
        self.iteration_count = 0
        self.iter_data = []
        self.dv_items = {}
        self.cons_items = {}
        self.objs_items = {}
        self.init_ui()

    def init_ui(self):
        # 1. Total Objective
        self.plot_obj = self._create_plot("Total Objective", "Value")
        self.addTab(self.plot_obj['widget'], "Total Objective")
        
        # 2. Design Variables
        self.plot_dv = self._create_plot("Design Variables", "Value", combo=True, callback=self.update_dv_plot, legend=True)
        self.addTab(self.plot_dv['widget'], "Variables")

        # 3. Constraints
        self.plot_cons = self._create_plot("Constraints", "Value", combo=True, callback=self.update_cons_plot, legend=True)
        self.addTab(self.plot_cons['widget'], "Constraints")

        # 4. Individual Objectives
        self.plot_objs = self._create_plot("Individual Objectives", "Value", combo=True, callback=self.update_objs_plot, legend=True)
        self.addTab(self.plot_objs['widget'], "Objectives")
        
        # 5. Problem Formulation (Text)
        self.problem_text = QtWidgets.QTextBrowser()
        self.problem_text.setOpenExternalLinks(False)
        self.problem_text.setStyleSheet("background-color: white; font-size: 11pt; padding: 10px;")
        self.addTab(self.problem_text, "Formulation")

    def _create_plot(self, title, ylabel, combo=False, callback=None, legend=False):
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        
        # Header (Combo + Save Btn)
        header = QtWidgets.QHBoxLayout()
        combo_box = None
        if combo:
            header.addWidget(QtWidgets.QLabel("Show:"))
            combo_box = QtWidgets.QComboBox()
            combo_box.addItem("All")
            combo_box.currentTextChanged.connect(callback)
            header.addWidget(combo_box)
            header.addStretch()

        btn_save = QtWidgets.QPushButton("Save")
        plot_widget = pg.PlotWidget(background='w')
        btn_save.clicked.connect(lambda: self._save_plot(plot_widget, title))
        if not combo: header.addStretch()
        header.addWidget(btn_save)
        layout.addLayout(header)

        # Plot Config
        plot_widget.showGrid(x=True, y=True, alpha=0.3)
        plot_widget.setTitle(title, color='k')
        plot_widget.setLabel('bottom', "Iteration", color='k')
        plot_widget.setLabel('left', ylabel, color='k')
        plot_widget.getAxis('left').setPen('k')
        plot_widget.getAxis('bottom').setPen('k')
        
        # Add legend if requested
        if legend:
            plot_widget.addLegend(offset=(10, 10))
        
        layout.addWidget(plot_widget)

        return {'widget': widget, 'plot': plot_widget, 'combo': combo_box}

    def _save_plot(self, plot, title):
        fname, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save Plot", f"{title}.png", "Images (*.png)")
        if fname:
            pg.exporters.ImageExporter(plot.plotItem).export(fname)

    def set_problem(self, problem, objectives, constraints):
        self.problem = problem
        self.objectives = objectives
        self.constraints = constraints
        self._populate_combos()
        self._update_formulation_text()
        self.clear_plots()

    def _populate_combos(self):
        if not self.problem: return
        self._fill_combo(self.plot_dv['combo'], [d['name'] for d in self.problem.design_variables])
        self._fill_combo(self.plot_cons['combo'], [c['name'] for c in self.constraints])
        self._fill_combo(self.plot_objs['combo'], [o['name'] for o in self.objectives])

    def _fill_combo(self, combo, items):
        if not combo: return
        curr = combo.currentText()
        combo.blockSignals(True)
        combo.clear()
        combo.addItem("All")
        combo.addItems(items)
        if combo.findText(curr) >= 0: combo.setCurrentText(curr)
        combo.blockSignals(False)

    def clear_plots(self):
        self.iteration_count = 0
        self.iter_data = []
        self.dv_data = []
        self.cons_data = {c['name']: [] for c in self.constraints}
        self.objs_data = {o['name']: [] for o in self.objectives}
        self.total_obj_data = []  # Track unpenalized total objective
        
        for p in [self.plot_obj, self.plot_dv, self.plot_cons, self.plot_objs]:
            p['plot'].clear()
            # --- FIX STARTS HERE ---
            # Remove the reference to the old curve so a new one is created
            if hasattr(p['plot'], '_curve'):
                del p['plot']._curve
            # --- FIX ENDS HERE ---
            
        self.dv_items = {}; self.cons_items = {}; self.objs_items = {}

    def update_data(self, data):
        """
        Receives data dictionary from OptimizationWorker.
        Keys: 'iteration', 'x', 'cost', 'violation', 'raw'
        """
        # OLD: self.iteration_count += 1
        # NEW: Get the real evaluation count from the worker
        current_eval = data.get('iteration', self.iteration_count + 1)
        self.iteration_count = current_eval
        
        x_vals = data['x']
        # cost = data['cost'] # Unused variable
        # max_violation = data['violation'] # Unused variable
        raw_results = data['raw']

        # Use the real evaluation count for the X-axis
        self.iter_data.append(current_eval)
        self.dv_data.append(x_vals)
        
        for name in self.cons_data:
            val = raw_results.get(name, 0.0)
            self.cons_data[name].append(val)
            
        for name in self.objs_data:
            val = raw_results.get(name, 0.0)
            self.objs_data[name].append(val)
        
        # Calculate unpenalized total objective
        total_obj = 0.0
        for obj in self.objectives:
            val = raw_results.get(obj['name'], 0.0)
            sign = 1.0 if obj.get('minimize', True) else -1.0
            total_obj += sign * obj.get('weight', 1.0) * val
        self.total_obj_data.append(total_obj)

        # Update Simple Plots (Objective)
        self._update_simple_curve(self.plot_obj['plot'], self.iter_data, self.total_obj_data, '#2ecc71')

        # Update Complex Plots
        # The worker thread already throttles emissions to 20Hz.
        self.update_dv_plot()
        self.update_cons_plot()
        self.update_objs_plot()

    def _update_simple_curve(self, plot, x, y, color):
        # Check if curve exists AND is still in the plot's item list
        if hasattr(plot, '_curve') and plot._curve in plot.items():
            plot._curve.setData(x, y)
        else:
            # Create new curve if missing or removed
            plot._curve = plot.plot(x, y, pen=pg.mkPen(color, width=2), symbol='o', symbolSize=5)

    def update_dv_plot(self):
        if not self.dv_data: return
        self._update_multi_plot(self.plot_dv, self.problem.design_variables, 
                              np.array(self.dv_data).T, self.dv_items)

    def update_cons_plot(self):
        # Constraints are stored in a dict of lists
        self._update_multi_plot(self.plot_cons, self.constraints, None, self.cons_items, use_dict_data=self.cons_data)

    def update_objs_plot(self):
        self._update_multi_plot(self.plot_objs, self.objectives, None, self.objs_items, use_dict_data=self.objs_data)

    def _update_multi_plot(self, plot_struct, definitions, data_matrix, item_store, use_dict_data=None):
        plot = plot_struct['plot']
        selected = plot_struct['combo'].currentText()
        
        # Determine what to draw
        to_draw = []
        for i, item_def in enumerate(definitions):
            if selected == "All" or selected == item_def['name']:
                to_draw.append((i, item_def))

        # Update curves
        for idx, (original_idx, item_def) in enumerate(to_draw):
            name = item_def['name']
            
            if use_dict_data:
                y_vals = use_dict_data[name]
            else:
                if data_matrix is None or len(data_matrix) <= original_idx: continue
                y_vals = data_matrix[original_idx]

            color = get_plot_color(original_idx, len(definitions))
            
            if name in item_store:
                item_store[name].setData(self.iter_data, y_vals)
                # Ensure pen style persists
                item_store[name].setPen(pg.mkPen(color, width=2))
            else:
                curve = plot.plot(self.iter_data, y_vals, pen=pg.mkPen(color, width=2), name=name)
                item_store[name] = curve

        # Cleanup removed items (if user switched selection)
        active_names = {d['name'] for _, d in to_draw}
        for name in list(item_store.keys()):
            if name not in active_names:
                plot.removeItem(item_store[name])
                del item_store[name]

    def _update_formulation_text(self):
        if not self.problem: return
        
        # Build mathematical optimization problem formulation
        html = "<div style='color: black; font-family: \"Cambria Math\", \"Times New Roman\", serif; font-size: 12pt; padding: 20px;'>"
        html += f"<h2 style='color: black; text-align: center; margin-bottom: 30px;'>{self.problem.name}</h2>"
        
        # Objective Function
        html += "<div style='margin-bottom: 25px;'>"
        if len(self.objectives) == 1:
            obj = self.objectives[0]
            obj_type = "minimize" if obj.get('minimize', True) else "maximize"
            html += f"<div style='margin-bottom: 10px;'><b>{obj_type}</b></div>"
            html += f"<div style='margin-left: 40px; font-style: italic;'>f(x) = {obj['name']}</div>"
        else:
            html += "<div style='margin-bottom: 10px;'><b>minimize</b></div>"
            html += "<div style='margin-left: 40px; font-style: italic;'>f(x) = "
            obj_terms = []
            for i, obj in enumerate(self.objectives):
                sign = "" if obj.get('minimize', True) else "−"
                weight = obj.get('weight', 1.0)
                weight_str = f"{weight:.4g}·"
                # Add + for terms after the first (unless negative sign already present)
                prefix = " + " if i > 0 and sign != "−" else (" " if i > 0 else "")
                obj_terms.append(f"{prefix}{sign}{weight_str}{obj['name']}")
            html += "".join(obj_terms)
            html += "</div>"
        html += "</div>"
        
        # Subject to (Constraints)
        html += "<div style='margin-bottom: 25px;'>"
        html += "<div style='margin-bottom: 10px;'><b>subject to:</b></div>"
        
        if self.constraints:
            for con in self.constraints:
                min_val = con.get('min_val', con.get('min', con.get('req_min', float('-inf'))))
                max_val = con.get('max_val', con.get('max', con.get('req_max', float('inf'))))
                
                html += "<div style='margin-left: 40px; margin-bottom: 5px;'>"
                
                # Format constraint based on bounds
                has_min = min_val not in ['-inf', float('-inf'), None] and min_val != float('-inf')
                has_max = max_val not in ['inf', float('inf'), None] and max_val != float('inf')
                
                if has_min and has_max:
                    if abs(float(min_val) - float(max_val)) < 1e-10:
                        # Equality constraint
                        html += f"{con['name']} = {float(min_val):.4g}"
                    else:
                        # Range constraint
                        html += f"{float(min_val):.4g} ≤ {con['name']} ≤ {float(max_val):.4g}"
                elif has_min:
                    # Lower bound only
                    html += f"{con['name']} ≥ {float(min_val):.4g}"
                elif has_max:
                    # Upper bound only
                    html += f"{con['name']} ≤ {float(max_val):.4g}"
                else:
                    # No bounds (unconstrained)
                    html += f"{con['name']} ∈ ℝ"
                
                html += "</div>"
        
        # Variable Bounds
        html += "<div style='margin-top: 15px;'>"
        for dv in self.problem.design_variables:
            min_val = dv.get('min', float('-inf'))
            max_val = dv.get('max', float('inf'))
            
            html += "<div style='margin-left: 40px; margin-bottom: 5px;'>"
            
            has_min = min_val != float('-inf') and np.isfinite(min_val)
            has_max = max_val != float('inf') and np.isfinite(max_val)
            
            if has_min and has_max:
                html += f"{float(min_val):.4g} ≤ {dv['name']} ≤ {float(max_val):.4g}"
            elif has_min:
                html += f"{dv['name']} ≥ {float(min_val):.4g}"
            elif has_max:
                html += f"{dv['name']} ≤ {float(max_val):.4g}"
            else:
                html += f"{dv['name']} ∈ ℝ"
            
            if dv.get('unit') and dv['unit'] != '-':
                html += f" &nbsp;&nbsp;<span style='color: #666; font-size: 10pt;'>[{dv['unit']}]</span>"
            
            html += "</div>"
        html += "</div>"
        html += "</div>"
        
        # Parameters (if any)
        if self.problem.parameters:
            html += "<div style='margin-top: 25px; border-top: 1px solid #ccc; padding-top: 15px;'>"
            html += "<div style='margin-bottom: 10px;'><b>where:</b></div>"
            for param in self.problem.parameters:
                html += f"<div style='margin-left: 40px; margin-bottom: 5px;'>{param['name']} = {param.get('value', '-'):.4g}"
                if param.get('unit') and param['unit'] != '-':
                    html += f" &nbsp;<span style='color: #666; font-size: 10pt;'>[{param['unit']}]</span>"
                html += "</div>"
            html += "</div>"
        
        html += "</div>"
        self.problem_text.setHtml(html)


# --- Main Class: OptimizationWidget ---

class OptimizationWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(OptimizationWidget, self).__init__(parent)
        self.problem = None
        self.worker = None
        self.models = []
        self.objectives = []
        self.constraints = []
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
        left_layout.addWidget(self.settings_widget)

        # 2. Objectives Table
        grp_objs = QtWidgets.QGroupBox("Objectives")
        objs_layout = QtWidgets.QVBoxLayout(grp_objs)
        self.table_objectives = QtWidgets.QTableWidget(0, 3)
        self.table_objectives.setHorizontalHeaderLabels(["Name", "Type", "Weight"])
        self.table_objectives.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.table_objectives.verticalHeader().setVisible(False)
        self.table_objectives.itemChanged.connect(self.on_objective_weight_changed)
        objs_layout.addWidget(self.table_objectives)
        left_layout.addWidget(grp_objs)

        # 3. Execution Control
        grp_exec = QtWidgets.QGroupBox("Execution")
        exec_layout = QtWidgets.QVBoxLayout(grp_exec)
        
        self.chk_use_current = QtWidgets.QCheckBox("Use Current Values as Initial Guess")
        self.chk_use_current.setToolTip("Start optimization from the values currently in the design variables table/model.")
        exec_layout.addWidget(self.chk_use_current)
        
        btn_layout = QtWidgets.QHBoxLayout()
        self.btn_run = QtWidgets.QPushButton("Run Optimization")
        self.btn_run.setStyleSheet("background-color: #2ecc71; color: white; font-weight: bold; padding: 6px;")
        self.btn_run.clicked.connect(self.start_optimization)
        
        self.btn_stop = QtWidgets.QPushButton("Stop")
        self.btn_stop.setStyleSheet("background-color: #e74c3c; color: white; font-weight: bold; padding: 6px;")
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

        # 4. Results Table (Small summary)
        self.table_results = QtWidgets.QTableWidget(0, 2)
        self.table_results.setHorizontalHeaderLabels(["Variable", "Value"])
        self.table_results.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.table_results.verticalHeader().setVisible(False)
        left_layout.addWidget(self.table_results)

        # Add Left Panel to Splitter
        self.splitter.addWidget(left_widget)

        # --- RIGHT PANEL (Plots) ---
        self.plots_widget = OptimizationPlotsWidget()
        self.splitter.addWidget(self.plots_widget)

        # Set initial sizes (Left: 400px, Right: remaining)
        self.splitter.setSizes([400, 800])

    def load_models(self, models):
        self.models = models
        self.settings_widget.system_combo.clear()
        for m in models:
            name = m.name if hasattr(m, 'name') else m['name']
            self.settings_widget.system_combo.addItem(name)
        if models:
            self.load_selected_system()

    def load_selected_system(self):
        idx = self.settings_widget.system_combo.currentIndex()
        if idx < 0 or idx >= len(self.models): return
        
        m = self.models[idx]
        if hasattr(m, 'name'):
            self.load_model_from_system_model(m)
        else:
            self.load_model(m['code'], m['inputs'], m['outputs'])

    def load_model(self, code, inputs, outputs):
        try:
            # Reusing original parsing logic
            self.system_code = code
            system_function = self._execute_code_safely(code)
            
            self.problem = XRayProblem("Optimization_Model", sample_size=3000)
            self.problem.set_system_model(system_function)
            
            for inp in inputs:
                self.problem.add_design_variable(inp['name'], inp.get('unit','-'), 
                                               self._parse_float(inp['min']), self._parse_float(inp['max']))
            for out in outputs:
                self.problem.add_quantity_of_interest(out['name'], out.get('unit','-'), 
                                                    self._parse_float(out['req_min']), self._parse_float(out['req_max']), 
                                                    minimize=out.get('minimize',False), maximize=out.get('maximize',False))
            self.set_problem(self.problem)
        except Exception as e:
            self.lbl_status.setText(f"Error loading model: {str(e)}")
            logger.error(f"Model load error: {e}", exc_info=True)

    def load_model_from_system_model(self, system_model):
        self.load_model(system_model.source_code, system_model.inputs, system_model.outputs)

    def set_problem(self, problem):
        self.problem = problem
        self.objectives = [q for q in problem.quantities_of_interest if q.get('minimize') or q.get('maximize')]
        self.constraints = [q for q in problem.quantities_of_interest if not (q.get('minimize') or q.get('maximize'))]
        
        # Populate UI components
        self._populate_objectives_table()
        self._init_results_table()  # Initialize results table structure
        self.plots_widget.set_problem(problem, self.objectives, self.constraints)
        self.lbl_status.setText(f"Loaded: {problem.name}")

    def _populate_objectives_table(self):
        self.table_objectives.blockSignals(True)
        self.table_objectives.setRowCount(len(self.objectives))
        for i, obj in enumerate(self.objectives):
            self.table_objectives.setItem(i, 0, QtWidgets.QTableWidgetItem(obj['name']))
            type_str = "Min" if obj.get('minimize') else "Max"
            self.table_objectives.setItem(i, 1, QtWidgets.QTableWidgetItem(type_str))
            self.table_objectives.setItem(i, 2, QtWidgets.QTableWidgetItem(str(obj.get('weight', 1.0))))
        self.table_objectives.blockSignals(False)

    def _init_results_table(self):
        """Pre-allocates rows for the results table to avoid flicker."""
        if not self.problem: return
        
        num_vars = len(self.problem.design_variables)
        self.table_results.setRowCount(1 + num_vars) # Cost + Variables
        
        self.table_results.setItem(0, 0, QtWidgets.QTableWidgetItem("Total Cost"))
        self.table_results.setItem(0, 1, QtWidgets.QTableWidgetItem("-"))
        
        for i, dv in enumerate(self.problem.design_variables):
            self.table_results.setItem(i + 1, 0, QtWidgets.QTableWidgetItem(dv['name']))
            self.table_results.setItem(i + 1, 1, QtWidgets.QTableWidgetItem("-"))

    def on_objective_weight_changed(self, item):
        if item.column() != 2: return
        try:
            val = float(item.text())
            self.objectives[item.row()]['weight'] = val
            self.plots_widget._update_formulation_text()  # Update formulation display
        except ValueError:
            pass # Ignore invalid input

    def start_optimization(self):
        if not self.problem: return
        
        if not self.objectives:
            QtWidgets.QMessageBox.warning(self, "No Objectives", "Please define at least one objective.")
            return
        
        # Stop any existing optimization
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait()  # Wait for it to finish
        
        # UI State
        self.btn_run.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.plots_widget.clear_plots()
        self.progress_bar.setRange(0, 0) # Indeterminate
        self.lbl_status.setText("Optimizing...")
        
        # Reset Results Table items
        self._init_results_table()

        # Setup Data
        x0 = []
        
        use_current = self.chk_use_current.isChecked()
        
        for dv in self.problem.design_variables:
            mn, mx = float(dv['min']), float(dv['max'])
            
            if use_current:
                # Use current value, clamped to bounds
                val = float(dv.get('value', (mn + mx) / 2))
                # basic clamping to ensure we don't start out of bounds (which crashes some solvers)
                if np.isfinite(mn) and val < mn: val = mn
                if np.isfinite(mx) and val > mx: val = mx
                x0.append(val)
            else:
                # Original Midpoint Logic
                if np.isfinite(mn) and np.isfinite(mx): x0.append((mn+mx)/2)
                elif np.isfinite(mn): x0.append(mn + 1)
                elif np.isfinite(mx): x0.append(mx - 1)
                else: x0.append(0.0)

        setup_data = {
            'variables': self.problem.design_variables,
            'objectives': self.objectives,
            'constraints': self.constraints,
            'x0': np.array(x0),
            'parameters': {p['name']: p['value'] for p in self.problem.parameters}
        }
        
        # Get settings from our nice widget
        solver_settings = self.settings_widget.get_config()

        # Worker
        self.worker = OptimizationWorker(self.problem.system_model, setup_data, solver_settings)
        self.worker.progress.connect(self.plots_widget.update_data)
        self.worker.progress.connect(self._update_results_table)
        self.worker.finished.connect(self.on_finished)
        self.worker.error.connect(self.on_error)
        self.worker.start()

    def _update_results_table(self, data):
        x_vals = data['x']
        cost = data['cost']

        # Update Cost
        item = self.table_results.item(0, 1)
        if item: item.setText(f"{cost:.4f}")
        
        # Update Variables
        for i, val in enumerate(x_vals):
            item = self.table_results.item(i + 1, 1)
            if item: item.setText(f"{val:.4f}")

    def stop_optimization(self):
        if self.worker:
            self.worker.stop()
            self.worker = None
        self.btn_run.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.lbl_status.setText("Stopped")

    def on_finished(self, result):
        self.worker = None
        self.btn_run.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.progress_bar.setRange(0, 100); self.progress_bar.setValue(100)
        
        is_feasible = result.max_violation < 1e-4
        if result.success:
            msg = "Converged"
        elif is_feasible:
            msg = "Done (Max Iter / Tol)" # Friendly success
        else:
            msg = "Failed"
            
        if result.max_violation > 1e-3: msg += " (Constraints Violated)"
        
        self.lbl_status.setText(f"{msg}: {result.message}")
        
        # Ensure the final result is written to table/model
        if result.x is not None:
             self._update_results_table({'x': result.x, 'cost': result.cost})
             for i, val in enumerate(result.x):
                self.problem.design_variables[i]['value'] = float(val)

    def on_error(self, msg):
        self.worker = None
        self.btn_run.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.lbl_status.setText("Error occurred")
        QtWidgets.QMessageBox.critical(self, "Error", msg)

    # --- Utils ---
    def _parse_float(self, val):
        if isinstance(val, (int, float)): return float(val)
        if isinstance(val, str):
            v = val.strip().lower()
            if v in ("inf", "+inf"): return float('inf')
            if v == "-inf": return float('-inf')
            try: return float(val)
            except: pass
        return 0.0

    def _execute_code_safely(self, code):
        # 1. Ensure the directory exists
        os.makedirs(TEMP_MODELS_DIR, exist_ok=True)

        # 2. Pass the 'dir' argument to NamedTemporaryFile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, dir=TEMP_MODELS_DIR) as f:
            f.write(code)
            temp_file = f.name
        spec = importlib.util.spec_from_file_location("temp_module", temp_file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        # Find the system_function
        system_function = None
        for attr_name in dir(module):
            if attr_name.startswith('system_function') and callable(getattr(module, attr_name)):
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
            "settings": self.settings_widget.get_config()
        }

        file_path = os.path.join(folder_path, "optimization_setup.json")
        try:
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=4)
        except Exception:
            pass

    def load_from_folder(self, folder_path: str):
        """
        Loads the optimization setup from a JSON file and populates the UI.
        """
        file_path = os.path.join(folder_path, "optimization_setup.json")
        if not os.path.exists(file_path):
            return

        try:
            with open(file_path, 'r') as f:
                data = json.load(f)

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

        except Exception:
            pass

    def _apply_settings_to_ui(self, settings):
        # Apply solver settings to the UI
        if 'method' in settings:
            idx = self.settings_widget.combo_method.findText(settings['method'])
            if idx >= 0:
                self.settings_widget.combo_method.setCurrentIndex(idx)
        
        if 'maxiter' in settings:
            self.settings_widget.spin_maxiter.setValue(int(settings['maxiter']))
        
        if 'scaling' in settings:
            self.settings_widget.chk_scaling.setChecked(bool(settings['scaling']))
        
        if 'objective_scale' in settings:
            self.settings_widget.spin_obj_scale.setValue(float(settings['objective_scale']))
        
        if 'tol' in settings:
            self.settings_widget.spin_tol.setValue(float(settings['tol']))
        
        if 'atol' in settings:
            self.settings_widget.spin_atol.setValue(float(settings['atol']))
        
        if 'maxfun' in settings:
            self.settings_widget.spin_maxfun.setValue(int(settings['maxfun']))
        
        if 'popsize' in settings:
            self.settings_widget.spin_popsize.setValue(int(settings['popsize']))
        
        if 'mutation' in settings and isinstance(settings['mutation'], (list, tuple)) and len(settings['mutation']) == 2:
            self.settings_widget.spin_mut_min.setValue(float(settings['mutation'][0]))
            self.settings_widget.spin_mut_max.setValue(float(settings['mutation'][1]))
        
        if 'recombination' in settings:
            self.settings_widget.spin_recomb.setValue(float(settings['recombination']))
        
        if 'strategy' in settings:
            idx = self.settings_widget.combo_de_strat.findText(settings['strategy'])
            if idx >= 0:
                self.settings_widget.combo_de_strat.setCurrentIndex(idx)
        
        if 'optimizer_name' in settings:
            idx = self.settings_widget.combo_ng_opt.findText(settings['optimizer_name'])
            if idx >= 0:
                self.settings_widget.combo_ng_opt.setCurrentIndex(idx)
        
        if 'num_workers' in settings:
            self.settings_widget.spin_ng_workers.setValue(int(settings['num_workers']))