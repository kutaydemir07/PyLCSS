# Copyright (c) 2025 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

import numpy as np
import logging
import colorsys
import tempfile
import importlib.util
from typing import List, Dict, Any, Optional

from PySide6 import QtWidgets, QtCore, QtGui
import pyqtgraph as pg

from ..problem_definition.problem_setup import XRayProblem
from ..user_interface.text_utils import format_html
from ..optimization.workers import OptimizationWorker

logger = logging.getLogger(__name__)

# --- Helper Functions ---

def get_plot_color(index: int, total_lines: int) -> str:
    if total_lines <= 1: return '#3498db'  # Nice Blue
    colors = []
    num_colors = max(20, total_lines)
    for i in range(num_colors):
        hue = (i * 0.618033988749895) % 1.0
        saturation = 0.7 + (i % 3) * 0.1
        value = 0.8 + (i % 2) * 0.1
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        hex_color = '#{:02x}{:02x}{:02x}'.format(int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255))
        colors.append(hex_color)
    return colors[index % len(colors)]


# --- Sub-Component: Solver Configuration ---

class SolverSettingsWidget(QtWidgets.QWidget):
    """
    Handles algorithm selection, system selection, and solver-specific parameters.
    """
    method_changed = QtCore.Signal(str)
    system_changed = QtCore.Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()

    def init_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Tabs for General / Config
        self.tabs = QtWidgets.QTabWidget()
        layout.addWidget(self.tabs)

        # --- Tab 1: General ---
        tab_general = QtWidgets.QWidget()
        form_layout = QtWidgets.QFormLayout(tab_general)
        form_layout.setContentsMargins(10, 10, 10, 10)

        self.system_combo = QtWidgets.QComboBox()
        self.system_combo.currentIndexChanged.connect(lambda idx: self.system_changed.emit(idx))
        form_layout.addRow("System Model:", self.system_combo)

        self.combo_method = QtWidgets.QComboBox()
        self.combo_method.addItems(['SLSQP', 'L-BFGS-B', 'trust-constr', 'COBYLA', 
                                  'Nevergrad', 'Differential Evolution'])
        self.combo_method.currentTextChanged.connect(self.on_method_changed)
        
        btn_info = QtWidgets.QPushButton("?")
        btn_info.setFixedWidth(25)
        btn_info.clicked.connect(self.show_algorithm_info)
        
        h_algo = QtWidgets.QHBoxLayout()
        h_algo.addWidget(self.combo_method)
        h_algo.addWidget(btn_info)
        form_layout.addRow("Algorithm:", h_algo)

        self.tabs.addTab(tab_general, "General")

        # --- Tab 2: Solver Config ---
        self.tab_config = QtWidgets.QWidget()
        config_layout = QtWidgets.QVBoxLayout(self.tab_config)
        
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        
        self.config_container = QtWidgets.QWidget()
        self.config_form = QtWidgets.QFormLayout(self.config_container)
        self.config_form.setContentsMargins(10, 10, 10, 10)

        # -- Common Controls --
        self.spin_maxiter = self._add_spin("Max Iterations:", 10, 100000, 1000)
        self.spin_tol = self._add_double_spin("Tolerance (ftol):", 1e-12, 1.0, 1e-6, decimals=9)
        self.spin_atol = self._add_double_spin("Abs. Tolerance:", 1e-12, 1.0, 1e-8, decimals=9)
        self.chk_scaling = QtWidgets.QCheckBox("Enable Variable Scaling")
        self.chk_scaling.setChecked(True)
        self.config_form.addRow("Scaling:", self.chk_scaling)
        
        self.spin_obj_scale = self._add_double_spin("Objective Scale:", 1.0, 1e12, 1.0)
        self.spin_obj_scale.setToolTip("Divide objective by this value to bring it close to 1.0. \nHelps constraints work correctly for large objective values.")
        
        self.spin_maxfun = self._add_spin("Max Func Evals:", 10, 1000000, 15000)

        # -- DE Controls --
        self.grp_de_header = QtWidgets.QLabel("<b>Differential Evolution</b>")
        self.config_form.addRow(self.grp_de_header)
        self.spin_popsize = self._add_spin("Pop. Size (x DVs):", 5, 200, 15)
        
        self.spin_mut_min = QtWidgets.QDoubleSpinBox()
        self.spin_mut_min.setRange(0.0, 1.9); self.spin_mut_min.setValue(0.5)
        self.spin_mut_max = QtWidgets.QDoubleSpinBox()
        self.spin_mut_max.setRange(0.0, 1.9); self.spin_mut_max.setValue(1.0)
        h_mut = QtWidgets.QHBoxLayout()
        h_mut.addWidget(self.spin_mut_min); h_mut.addWidget(QtWidgets.QLabel("-")); h_mut.addWidget(self.spin_mut_max)
        self.lbl_mut = QtWidgets.QLabel("Mutation Range:")
        self.config_form.addRow(self.lbl_mut, h_mut)
        
        self.spin_recomb = self._add_double_spin("Recombination:", 0.0, 1.0, 0.7)
        self.combo_de_strat = QtWidgets.QComboBox()
        self.combo_de_strat.addItems(['best1bin', 'rand1exp', 'randtobest1bin', 'currenttobest1bin'])
        self.config_form.addRow("Strategy:", self.combo_de_strat)
        self.de_controls = [self.grp_de_header, self.spin_popsize, self.lbl_mut, self.spin_mut_min, 
                            self.spin_mut_max, self.spin_recomb, self.combo_de_strat]

        # -- Nevergrad Controls --
        self.grp_ng_header = QtWidgets.QLabel("<b>Nevergrad</b>")
        self.config_form.addRow(self.grp_ng_header)
        self.combo_ng_opt = QtWidgets.QComboBox()
        self.combo_ng_opt.addItems(["NGOpt", "TwoPointsDE", "Portfolio", "OnePlusOne", "CMA"])
        self.config_form.addRow("Optimizer:", self.combo_ng_opt)
        
        import os
        workers = max(1, os.cpu_count() or 1)
        self.spin_ng_workers = self._add_spin("Workers:", 1, 64, workers)
        self.ng_controls = [self.grp_ng_header, self.combo_ng_opt, self.spin_ng_workers]

        scroll.setWidget(self.config_container)
        config_layout.addWidget(scroll)
        self.tabs.addTab(self.tab_config, "Configuration")

        # Initial visibility update
        self.update_visibility(self.combo_method.currentText())

    def _add_spin(self, label, min_val, max_val, default):
        spin = QtWidgets.QSpinBox()
        spin.setRange(min_val, max_val)
        spin.setValue(default)
        self.config_form.addRow(label, spin)
        return spin

    def _add_double_spin(self, label, min_val, max_val, default, decimals=2):
        spin = QtWidgets.QDoubleSpinBox()
        spin.setRange(min_val, max_val)
        spin.setValue(default)
        spin.setDecimals(decimals)
        self.config_form.addRow(label, spin)
        return spin

    def on_method_changed(self, method_name):
        self.update_visibility(method_name)
        self.method_changed.emit(method_name)

    def update_visibility(self, method):
        # Default: Scipy Gradient Based
        scipy_gradient = ['SLSQP', 'L-BFGS-B', 'TNC', 'trust-constr']
        
        # Hide all specific groups first
        self._set_visible(self.de_controls, False)
        self._set_visible(self.ng_controls, False)
        
        # Common visibility - start with everything visible
        self.spin_tol.setVisible(True)
        self.spin_atol.setVisible(True)
        self.chk_scaling.setVisible(True)
        self.spin_maxfun.setVisible(True)
        self.config_form.labelForField(self.spin_tol).setVisible(True)
        self.config_form.labelForField(self.spin_atol).setVisible(True)

        if method == 'Differential Evolution':
            self._set_visible(self.de_controls, True)
            self.spin_tol.setVisible(False)
            self.spin_atol.setVisible(False)
            self.spin_maxfun.setVisible(False)
            self.config_form.labelForField(self.spin_tol).setVisible(False)
            self.config_form.labelForField(self.spin_atol).setVisible(False)
            self.config_form.labelForField(self.spin_maxfun).setVisible(False)
            
        elif method == 'Nevergrad':
            self._set_visible(self.ng_controls, True)
            self.spin_maxfun.setVisible(False)
            self.spin_tol.setVisible(False)
            self.spin_atol.setVisible(False)
            self.config_form.labelForField(self.spin_maxfun).setVisible(False)
            self.config_form.labelForField(self.spin_tol).setVisible(False)
            self.config_form.labelForField(self.spin_atol).setVisible(False)

        elif method == 'COBYLA':
            # COBYLA only uses tol, not atol
            self.spin_atol.setVisible(False)
            self.config_form.labelForField(self.spin_atol).setVisible(False)

        else: # Scipy Gradient methods (SLSQP, L-BFGS-B, trust-constr)
            self.config_form.labelForField(self.spin_tol).setVisible(True)
            self.config_form.labelForField(self.spin_atol).setVisible(True)
            self.config_form.labelForField(self.spin_maxfun).setVisible(True)

    def _set_visible(self, widgets, visible):
        for w in widgets:
            w.setVisible(visible)
            if self.config_form.labelForField(w):
                self.config_form.labelForField(w).setVisible(visible)

    def get_config(self):
        return {
            'method': self.combo_method.currentText(),
            'maxiter': self.spin_maxiter.value(),
            'scaling': self.chk_scaling.isChecked(),
            'objective_scale': self.spin_obj_scale.value(),
            'tol': self.spin_tol.value(),
            'atol': self.spin_atol.value(),
            'maxfun': self.spin_maxfun.value(),
            'popsize': self.spin_popsize.value(),
            'mutation': (self.spin_mut_min.value(), self.spin_mut_max.value()),
            'recombination': self.spin_recomb.value(),
            'strategy': self.combo_de_strat.currentText(),
            'optimizer_name': self.combo_ng_opt.currentText(),
            'num_workers': self.spin_ng_workers.value()
        }

    # --- UPDATED METHOD: Full Descriptions for All Solvers ---
    def show_algorithm_info(self):
        method = self.combo_method.currentText()
        desc = {
            'SLSQP': "Gradient-based (Sequential Least SQuares Programming). \nBest general-purpose solver for smooth, constrained problems.",
            'L-BFGS-B': "Gradient-based (Limited-memory BFGS). \nExcellent for bound-constrained problems. Uses very little memory.",
            'TNC': "Gradient-based (Truncated Newton). \nDesigned for problems with many variables and simple bounds.",
            'trust-constr': "Gradient-based (Trust Region). \nModern solver. Handles all constraint types robustly. Can be slower than SLSQP.",
            'COBYLA': "Gradient-free (Linear Approximation). \nSupports inequality constraints. Good for models where gradients are unavailable.",
            'Nelder-Mead': "Gradient-free (Simplex). \nRobust local search. Does not support constraints natively (uses penalty).",
            'Powell': "Gradient-free. \nOptimizes each variable sequentially. Does not support constraints natively (uses penalty).",
            'Nevergrad': "Gradient-free Meta-Solver. \nRobust for noisy, black-box, or non-smooth problems. Supports parallel evaluation.",
            'Differential Evolution': "Global Optimization (Genetic Algorithm). \nBest for finding the global minimum in complex/multi-modal landscapes. Slower."
        }
        QtWidgets.QMessageBox.information(self, method, desc.get(method, "No description available."))


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

        # 6. Individual Objectives
        self.plot_objs = self._create_plot("Individual Objectives", "Value", combo=True, callback=self.update_objs_plot, legend=True)
        self.addTab(self.plot_objs['widget'], "Objectives")
        
        # 7. Problem Formulation (Text)
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
        self.dv_items = {}; self.cons_items = {}; self.objs_items = {}

    def update_data(self, data):
        """
        Receives data dictionary from OptimizationWorker.
        Keys: 'x', 'cost', 'violation', 'raw'
        """
        self.iteration_count += 1
        
        x_vals = data['x']
        cost = data['cost']
        max_violation = data['violation']
        raw_results = data['raw']

        self.iter_data.append(self.iteration_count)
        self.dv_data.append(x_vals)
        
        for name in self.cons_data:
            self.cons_data[name].append(raw_results.get(name, 0.0))
        for name in self.objs_data:
            self.objs_data[name].append(raw_results.get(name, 0.0))
        
        # Calculate unpenalized total objective
        total_obj = 0.0
        for obj in self.objectives:
            val = raw_results.get(obj['name'], 0.0)
            sign = 1.0 if obj.get('minimize', True) else -1.0
            total_obj += sign * obj.get('weight', 1.0) * val
        self.total_obj_data.append(total_obj)

        # Update Simple Plots (Objective)
        self._update_simple_curve(self.plot_obj['plot'], self.iter_data, self.total_obj_data, '#2ecc71')

        # Update Complex Plots (throttled slightly for performance)
        if len(self.iter_data) % 2 == 0 or len(self.iter_data) < 10:
            self.update_dv_plot()
            self.update_cons_plot()
            self.update_objs_plot()

    def _update_simple_curve(self, plot, x, y, color):
        if hasattr(plot, '_curve'):
            plot._curve.setData(x, y)
        else:
            plot._curve = plot.plot(x, y, pen=pg.mkPen(color, width=2), symbol='o', symbolSize=5)

    def update_dv_plot(self):
        self._update_multi_plot(self.plot_dv, self.problem.design_variables, 
                              np.array(self.dv_data).T if self.dv_data else [], self.dv_items)

    def update_cons_plot(self):
        # Constraints are stored in a dict of lists, need to handle appropriately
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
                y_vals = data_matrix[original_idx]

            color = get_plot_color(idx, len(to_draw))
            
            if name in item_store:
                item_store[name].setData(self.iter_data, y_vals)
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
        
        # --- NEW CODE START ---
        self.chk_use_current = QtWidgets.QCheckBox("Use Current Values as Initial Guess")
        self.chk_use_current.setToolTip("Start optimization from the values currently in the design variables table/model.")
        exec_layout.addWidget(self.chk_use_current)
        # --- NEW CODE END ---
        
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
            QtWidgets.QMessageBox.warning(self, "No Objectives", "Please define at least one objective (minimize or maximize) in the system model.")
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

        # Setup Data
        x0 = []
        
        # --- MODIFIED CODE START ---
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
        # --- MODIFIED CODE END ---

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
        # Update summary table occasionally
        if len(self.plots_widget.iter_data) % 5 != 0: return
        
        x_vals = data['x']
        cost = data['cost']
        raw = data['raw']

        self.table_results.setRowCount(0)
        # Cost Row
        row = self.table_results.rowCount()
        self.table_results.insertRow(row)
        self.table_results.setItem(row, 0, QtWidgets.QTableWidgetItem("Total Cost"))
        self.table_results.setItem(row, 1, QtWidgets.QTableWidgetItem(f"{cost:.4f}"))
        
        # Variable Rows
        for i, val in enumerate(x_vals):
            row = self.table_results.rowCount()
            self.table_results.insertRow(row)
            name = self.problem.design_variables[i]['name']
            self.table_results.setItem(row, 0, QtWidgets.QTableWidgetItem(name))
            self.table_results.setItem(row, 1, QtWidgets.QTableWidgetItem(f"{val:.4f}"))

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
        
        msg = "Converged" if result.success else "Failed"
        if result.max_violation > 1e-3: msg += " (Constraints Violated)"
        self.lbl_status.setText(f"{msg}: {result.message}")
        
        if result.success and result.x is not None:
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
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
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