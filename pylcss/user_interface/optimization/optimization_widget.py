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

from ...system_modeling.problem_definition.problem_setup import XRayProblem
from ..common.text_utils import format_html
from ...optimization.workers import OptimizationWorker
from ...config import optimization_config, SOLVER_DESCRIPTIONS, TEMP_MODELS_DIR
from .optimization_settings_dialog import OptimizationSettingsDialog  # New import

logger = logging.getLogger(__name__)

# A constraint is treated as satisfied only if it is within this *relative*
# tolerance of its bound — i.e. essentially exact (floating-point round-off only).
# The constraint safety back-off (Advanced Settings) is what keeps real solutions
# comfortably inside the feasible region; this threshold just absorbs numerical
# noise, so anything genuinely over its bound reads VIOLATED, never OK. Used
# consistently by the status banner and the per-constraint OK/VIOLATED marks.
FEASIBILITY_TOL = 1e-6

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
        
        self.btn_info = QtWidgets.QPushButton("?")
        self.btn_info.setFixedWidth(25)
        self.btn_info.clicked.connect(self.show_algorithm_info)
        # The algorithm explanation lives on this button: hover for the one-liner,
        # click for full details. Keeps the left panel compact.
        self._refresh_info_tooltip(self.combo_method.currentText())

        h_algo = QtWidgets.QHBoxLayout()
        h_algo.addWidget(self.combo_method)
        h_algo.addWidget(self.btn_info)
        form_layout.addRow("Algorithm:", h_algo)

        # 3. Settings Button (The pop-up trigger)
        self.btn_settings = QtWidgets.QPushButton("Advanced Settings")
        self.btn_settings.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_FileDialogDetailedView))
        self.btn_settings.clicked.connect(self.open_settings_dialog)
        form_layout.addRow("", self.btn_settings)

        layout.addWidget(grp)
        layout.addStretch()

    def on_method_changed(self, method_name):
        self._refresh_info_tooltip(method_name)
        self.method_changed.emit(method_name)

    def _refresh_info_tooltip(self, method_name):
        info = SOLVER_DESCRIPTIONS.get(method_name, {})
        desc = info.get('description')
        when = info.get('when_to_use', '')
        tip = f"{desc}\n\n{when}".strip() if desc else "Show details about the selected algorithm"
        self.btn_info.setToolTip(tip)

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
            'constraint_margin': 1e-3,
            'maxfun': 15000,
            'popsize': 15,
            'mutation': (0.5, 1.0),
            'recombination': 0.7,
            'strategy': 'best1bin',
            'optimizer_name': 'NGOpt',
            'num_workers': 1,
            # NSGA-II defaults
            'nsga_popsize': 100,
            'nsga_generations': 200,
            'nsga_crossover_prob': 0.9,
            'nsga_mutation_prob': None,
            'nsga_eta_c': 20.0,
            'nsga_eta_m': 20.0,
            # Multi-Start defaults
            'ms_n_starts': 10,
            'ms_local_solver': 'SLSQP'
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
            <p><b>When to use:</b> {info.get('when_to_use', '-')}</p>
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
        
        # 5. Results (final solution summary)
        self._init_results_tab()

        # 6. Problem Formulation (Text)
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
        self.reset_results()

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

    # ---- Results tab (final solution) ----

    def _init_results_tab(self):
        self.results_tab = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(self.results_tab)
        self.lbl_result_status = QtWidgets.QLabel()
        self.lbl_result_status.setWordWrap(True)
        self._set_result_status("Not run yet.", "#7f8c8d")
        lay.addWidget(self.lbl_result_status)

        self.results_table = QtWidgets.QTableWidget(0, 3)
        self.results_table.setHorizontalHeaderLabels(["Quantity", "Value", "Detail"])
        self.results_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.results_table.verticalHeader().setVisible(False)
        self.results_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        lay.addWidget(self.results_table)
        self.addTab(self.results_tab, "Results")

    def _set_result_status(self, text, color):
        self.lbl_result_status.setText(text)
        self.lbl_result_status.setStyleSheet(
            f"padding:8px; border-radius:4px; background:{color}; "
            f"color:white; font-weight:bold;"
        )

    def reset_results(self):
        self._render_results_table(None, None, {})
        self._set_result_status("Not run yet.", "#7f8c8d")

    def mark_running(self):
        self._set_result_status("Optimizing…", "#2980b9")

    def update_results_live(self, data):
        self._render_results_table(data.get('x'), data.get('cost'), data.get('raw', {}))

    def set_results_final(self, result, status_text, color):
        raw = {**(result.objectives or {}), **(result.constraints or {})}
        self._render_results_table(result.x, result.cost, raw)
        self._set_result_status(status_text, color)

    def _constraint_detail(self, con, value):
        lo = con.get('min_val', con.get('min', con.get('req_min', float('-inf'))))
        hi = con.get('max_val', con.get('max', con.get('req_max', float('inf'))))
        try: lo = float(lo)
        except (TypeError, ValueError): lo = float('-inf')
        try: hi = float(hi)
        except (TypeError, ValueError): hi = float('inf')
        has_lo, has_hi = np.isfinite(lo), np.isfinite(hi)
        if has_lo and has_hi:
            bound = f"[{lo:.4g}, {hi:.4g}]"
        elif has_lo:
            bound = f">= {lo:.4g}"
        elif has_hi:
            bound = f"<= {hi:.4g}"
        else:
            bound = "free"
        if value is None:
            return bound
        v = float(value)

        # Judge feasibility on a *relative* tolerance (matching solver precision),
        # so an active constraint resting on its bound (~1e-5) reads OK instead of
        # being flagged for numerical noise. Mirrors the evaluator's violation scale.
        if has_lo and has_hi and abs(hi - lo) > 1e-12:
            scale = abs(hi - lo)
        else:
            mags = [abs(b) for b in (lo, hi) if np.isfinite(b) and abs(b) > 1e-12]
            scale = min(mags) if mags else 1.0
        tol = max(FEASIBILITY_TOL * scale, 1e-9)
        violation = max(
            (lo - v) if has_lo else 0.0,
            (v - hi) if has_hi else 0.0,
            0.0,
        )
        ok = violation <= tol
        return f"{bound}   {'OK' if ok else 'VIOLATED'}"

    def _render_results_table(self, x, cost, raw):
        if not self.problem:
            return
        raw = raw or {}
        rows = []  # (quantity, value, detail, is_header, violated)

        rows.append(("Design variables", "", "", True, False))
        for i, dv in enumerate(self.problem.design_variables):
            val = "-" if (x is None or i >= len(x)) else f"{float(x[i]):.4g}"
            unit = dv.get('unit') if dv.get('unit') not in (None, '-') else ""
            rows.append((dv['name'], val, unit, False, False))

        if self.objectives:
            rows.append(("Objectives", "", "", True, False))
            for obj in self.objectives:
                v = raw.get(obj['name'])
                val = "-" if v is None else f"{float(v):.4g}"
                goal = "minimize" if obj.get('minimize', True) else "maximize"
                rows.append((obj['name'], val, goal, False, False))

        if self.constraints:
            rows.append(("Constraints", "", "", True, False))
            for con in self.constraints:
                v = raw.get(con['name'])
                val = "-" if v is None else f"{float(v):.4g}"
                detail = self._constraint_detail(con, v)
                rows.append((con['name'], val, detail, False, detail.endswith("VIOLATED")))

        rows.append(("Summary", "", "", True, False))
        rows.append(("Total cost", "-" if cost is None else f"{float(cost):.4g}", "", False, False))

        self.results_table.setRowCount(len(rows))
        for r, (q, v, d, is_header, violated) in enumerate(rows):
            qi = QtWidgets.QTableWidgetItem(q)
            vi = QtWidgets.QTableWidgetItem(v)
            di = QtWidgets.QTableWidgetItem(d)
            if is_header:
                font = qi.font(); font.setBold(True); qi.setFont(font)
                qi.setForeground(QtGui.QColor("#2471a3"))
            if violated:
                di.setForeground(QtGui.QColor("#e74c3c"))
            self.results_table.setItem(r, 0, qi)
            self.results_table.setItem(r, 1, vi)
            self.results_table.setItem(r, 2, di)

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
        self.update_results_live(data)

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

    @staticmethod
    def _safe_float(val, default):
        try:
            return float(val)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _format_value(val):
        try:
            return f"{float(val):.4g}"
        except (TypeError, ValueError):
            return str(val)

    def _update_formulation_text(self):
        if not self.problem:
            return

        accent = "#1f3a5f"      # title
        keyword_col = "#2471a3"  # min / s.t. / over / where
        muted = "#7f8c8d"        # units
        rule = "#d5dbdb"         # divider

        # HTML entities (numeric => universally supported by Qt rich text)
        LE, GE, DOT, MINUS = "&#8804;", "&#8805;", "&#183;", "&#8722;"
        ISIN, REALS, NBSP = "&#8712;", "&#8477;", "&#160;"

        # --- Objective expression ---
        if len(self.objectives) == 1:
            obj = self.objectives[0]
            keyword = "minimize" if obj.get('minimize', True) else "maximize"
            obj_expr = f"f(<i>x</i>) = {obj['name']}"
        elif self.objectives:
            keyword = "minimize"
            terms = []
            for i, obj in enumerate(self.objectives):
                w = obj.get('weight', 1.0)
                negative = not obj.get('minimize', True)  # maximize => subtract
                term = f"{abs(w):.4g}{DOT}{obj['name']}"
                if i == 0:
                    terms.append((MINUS if negative else "") + term)
                else:
                    terms.append((f" {MINUS} " if negative else " + ") + term)
            obj_expr = "f(<i>x</i>) = " + "".join(terms)
        else:
            keyword = "minimize"
            obj_expr = "&#8212;"

        # --- Constraint rows ---
        con_rows = []
        for con in self.constraints:
            lo = self._safe_float(con.get('min_val', con.get('min', con.get('req_min', float('-inf')))), float('-inf'))
            hi = self._safe_float(con.get('max_val', con.get('max', con.get('req_max', float('inf')))), float('inf'))
            name = con['name']
            has_min, has_max = np.isfinite(lo), np.isfinite(hi)
            if has_min and has_max:
                if abs(lo - hi) < 1e-10:
                    con_rows.append(f"{name} = {lo:.4g}")
                else:
                    con_rows.append(f"{lo:.4g} {LE} {name} {LE} {hi:.4g}")
            elif has_min:
                con_rows.append(f"{name} {GE} {lo:.4g}")
            elif has_max:
                con_rows.append(f"{name} {LE} {hi:.4g}")
            else:
                con_rows.append(f"{name} {ISIN} {REALS}")

        # --- Bound rows ---
        bound_rows = []
        for dv in self.problem.design_variables:
            lo = self._safe_float(dv.get('min', float('-inf')), float('-inf'))
            hi = self._safe_float(dv.get('max', float('inf')), float('inf'))
            name = dv['name']
            has_min, has_max = np.isfinite(lo), np.isfinite(hi)
            if has_min and has_max:
                s = f"{lo:.4g} {LE} {name} {LE} {hi:.4g}"
            elif has_min:
                s = f"{name} {GE} {lo:.4g}"
            elif has_max:
                s = f"{name} {LE} {hi:.4g}"
            else:
                s = f"{name} {ISIN} {REALS}"
            unit = dv.get('unit')
            if unit and unit != '-':
                s += f" {NBSP}<span style='color:{muted}; font-size:10pt;'>[{unit}]</span>"
            bound_rows.append(s)

        # --- Parameter rows ---
        param_rows = []
        for p in self.problem.parameters:
            s = f"{p['name']} = {self._format_value(p.get('value', '-'))}"
            unit = p.get('unit')
            if unit and unit != '-':
                s += f" {NBSP}<span style='color:{muted}; font-size:10pt;'>[{unit}]</span>"
            param_rows.append(s)

        # --- Assemble aligned table (keyword column + expression column) ---
        def kw(label):
            return (f"<td style='vertical-align:top; text-align:right; "
                    f"padding:4px 18px 4px 0; color:{keyword_col}; "
                    f"font-style:italic;'>{label}</td>")

        def expr(content):
            return f"<td style='vertical-align:top; padding:4px 0; color:#1c1c1c;'>{content}</td>"

        rows = [f"<tr>{kw(keyword)}{expr(obj_expr)}</tr>"]

        var_names = ", ".join(dv['name'] for dv in self.problem.design_variables)
        if var_names:
            rows.append(f"<tr>{kw('over')}{expr(f'<i>x</i> = ({var_names})')}</tr>")

        for i, c in enumerate(con_rows):
            rows.append(f"<tr>{kw('subject to' if i == 0 else '')}{expr(c)}</tr>")

        for i, b in enumerate(bound_rows):
            rows.append(f"<tr>{kw('bounds' if i == 0 else '')}{expr(b)}</tr>")

        for i, pr in enumerate(param_rows):
            rows.append(f"<tr>{kw('where' if i == 0 else '')}{expr(pr)}</tr>")

        html = (
            f"<div style=\"font-family:'Cambria Math','Times New Roman',serif; "
            f"color:#1c1c1c; padding:24px 30px;\">"
            f"<div style='font-size:15pt; font-weight:bold; color:{accent}; "
            f"border-bottom:2px solid {rule}; padding-bottom:8px; margin-bottom:20px;'>"
            f"{self.problem.name}</div>"
            f"<table cellspacing='0' cellpadding='0' style='font-size:13pt;'>"
            f"{''.join(rows)}</table>"
            f"</div>"
        )
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
            QtWidgets.QMessageBox.warning(self, "No Objectives", "Please define at least one objective.")
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
        self.progress_bar.setRange(0, 0) # Indeterminate
        self.lbl_status.setText("Optimizing...")
        
        self.plots_widget.mark_running()

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
        self.worker.finished.connect(self.on_finished)
        self.worker.error.connect(self.on_error)
        self.worker.start()

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

        is_feasible = result.max_violation < FEASIBILITY_TOL
        if result.success:
            msg, color = "Converged", "#27ae60"
        elif is_feasible:
            msg, color = "Done (Max Iter / Tol)", "#27ae60"
        else:
            msg, color = "Failed", "#e74c3c"

        if result.max_violation > FEASIBILITY_TOL:
            msg += " (Constraints Violated)"
            color = "#e74c3c"

        self.lbl_status.setText(f"{msg}: {result.message}")

        # Write the final solution into the Results tab and back to the model.
        if result.x is not None:
            self.plots_widget.set_results_final(result, f"{msg}: {result.message}", color)
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
        
        # Advanced parameters live in SolverSettingsWidget.settings (edited via
        # the pop-up dialog), not as individual widgets, so merge them there.
        stored = self.settings_widget.settings
        for key, value in settings.items():
            if key == 'method':
                continue
            stored[key] = value