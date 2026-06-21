# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

from PySide6 import QtWidgets, QtCore, QtGui
from ...config import optimization_config, SOLVER_DESCRIPTIONS

class OptimizationSettingsDialog(QtWidgets.QDialog):
    """
    Modal dialog for advanced solver settings.
    Uses QLineEdit instead of SpinBoxes for easier floating-point entry.
    """
    def __init__(self, current_method, current_settings, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Optimization Settings")
        self.resize(400, 500)
        self.current_method = current_method
        self.settings = current_settings
        
        self.init_ui()
        self.load_settings()
        self.update_visibility(self.current_method)

    def init_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        
        # Header naming the active algorithm so the settings have context.
        info = SOLVER_DESCRIPTIONS.get(self.current_method, {})
        header = QtWidgets.QLabel(
            f"<b style='font-size:11pt;'>{info.get('name', self.current_method)}</b>"
            f"<br><span style='color:#566573;'>{info.get('description', '')}</span>"
        )
        header.setWordWrap(True)
        header.setStyleSheet("padding: 4px 4px 10px 4px;")
        layout.addWidget(header)

        # Scroll Area for settings
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        
        container = QtWidgets.QWidget()
        self.form_layout = QtWidgets.QFormLayout(container)
        self.form_layout.setLabelAlignment(QtCore.Qt.AlignRight)
        
        # --- Common Settings ---
        self.grp_common_label = QtWidgets.QLabel("<b>General Parameters</b>")
        self.form_layout.addRow(self.grp_common_label)
        
        self.edit_maxiter = self._add_input("Max Iterations:", "5000")
        self.edit_tol = self._add_input("Tolerance (ftol):", str(optimization_config.DEFAULT_TOLERANCE))
        self.edit_tol.setToolTip("Stop when objective change is smaller than this value (e.g., 1e-6).")
        
        self.edit_atol = self._add_input("Abs. Tolerance:", "1e-8")
        
        self.chk_scaling = QtWidgets.QCheckBox("Enable Variable Scaling")
        self.chk_scaling.setChecked(True)
        self.form_layout.addRow("Scaling:", self.chk_scaling)
        
        self.edit_obj_scale = self._add_input("Objective Scale:", "1.0")

        self.edit_con_margin = self._add_input("Constraint Safety Margin:", "0.001")
        self.edit_con_margin.setToolTip(
            "Relative back-off applied to every constraint so the solver returns a "
            "design that strictly satisfies it — never even slightly over. "
            "0 disables it; 0.001 keeps results ~0.1% inside the feasible region."
        )

        # --- Differential Evolution (DE) ---
        self.grp_de_label = QtWidgets.QLabel("<b>Differential Evolution</b>")
        self.grp_de_label.setContentsMargins(0, 15, 0, 5)
        self.form_layout.addRow(self.grp_de_label)
        
        self.edit_popsize = self._add_input("Population Size:", "15")
        
        # Mutation Range (Min - Max)
        self.edit_mut_min = QtWidgets.QLineEdit("0.5")
        self.edit_mut_max = QtWidgets.QLineEdit("1.0")
        self.lbl_mut_dash = QtWidgets.QLabel("-")
        h_mut = QtWidgets.QHBoxLayout()
        h_mut.addWidget(self.edit_mut_min)
        h_mut.addWidget(self.lbl_mut_dash)
        h_mut.addWidget(self.edit_mut_max)
        self.lbl_mutation = QtWidgets.QLabel("Mutation Range:")
        self.form_layout.addRow(self.lbl_mutation, h_mut)
        
        self.edit_recomb = self._add_input("Recombination:", "0.7")
        
        self.combo_de_strat = QtWidgets.QComboBox()
        self.combo_de_strat.addItems(['best1bin', 'rand1exp', 'randtobest1bin', 'currenttobest1bin'])
        self.form_layout.addRow("Strategy:", self.combo_de_strat)
        
        self.de_widgets = [self.grp_de_label, self.edit_popsize, self.lbl_mutation,
                           self.edit_mut_min, self.lbl_mut_dash, self.edit_mut_max,
                           self.edit_recomb, self.combo_de_strat]

        # --- Nevergrad ---
        self.grp_ng_label = QtWidgets.QLabel("<b>Nevergrad</b>")
        self.grp_ng_label.setContentsMargins(0, 15, 0, 5)
        self.form_layout.addRow(self.grp_ng_label)
        
        self.combo_ng_opt = QtWidgets.QComboBox()
        self.combo_ng_opt.addItems(["NGOpt", "TwoPointsDE", "Portfolio", "OnePlusOne", "CMA"])
        self.form_layout.addRow("Optimizer:", self.combo_ng_opt)
        
        self.edit_ng_workers = self._add_input("Num Workers:", "1")
        
        self.ng_widgets = [self.grp_ng_label, self.combo_ng_opt, self.edit_ng_workers]

        # --- NSGA-II (Multi-Objective) ---
        self.grp_nsga_label = QtWidgets.QLabel("<b>NSGA-II (Multi-Objective)</b>")
        self.grp_nsga_label.setContentsMargins(0, 15, 0, 5)
        self.form_layout.addRow(self.grp_nsga_label)

        self.edit_nsga_popsize = self._add_input("Population Size:", "100")
        self.edit_nsga_generations = self._add_input("Generations:", "200")
        self.edit_nsga_crossover = self._add_input("Crossover Prob.:", "0.9")
        self.edit_nsga_mutation = self._add_input("Mutation Prob.:", "")
        self.edit_nsga_mutation.setPlaceholderText("auto (1/n_vars)")
        self.edit_nsga_eta_c = self._add_input("SBX η (crossover):", "20.0")
        self.edit_nsga_eta_m = self._add_input("Poly η (mutation):", "20.0")

        self.nsga_widgets = [self.grp_nsga_label, self.edit_nsga_popsize,
                             self.edit_nsga_generations, self.edit_nsga_crossover,
                             self.edit_nsga_mutation, self.edit_nsga_eta_c, self.edit_nsga_eta_m]

        # --- Multi-Start ---
        self.grp_ms_label = QtWidgets.QLabel("<b>Multi-Start Global Search</b>")
        self.grp_ms_label.setContentsMargins(0, 15, 0, 5)
        self.form_layout.addRow(self.grp_ms_label)

        self.edit_ms_starts = self._add_input("Number of Starts:", "10")
        self.combo_ms_local = QtWidgets.QComboBox()
        self.combo_ms_local.addItems(["SLSQP", "COBYLA", "trust-constr"])
        self.form_layout.addRow("Local Solver:", self.combo_ms_local)

        self.ms_widgets = [self.grp_ms_label, self.edit_ms_starts, self.combo_ms_local]

        scroll.setWidget(container)
        layout.addWidget(scroll)

        # Buttons
        btn_box = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        btn_box.accepted.connect(self.accept)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)

    def _add_input(self, label, default):
        edit = QtWidgets.QLineEdit(default)
        self.form_layout.addRow(label, edit)
        return edit

    def update_visibility(self, method):
        # Hide every solver-specific block first.
        self._set_visible(self.de_widgets, False)
        self._set_visible(self.ng_widgets, False)
        self._set_visible(self.nsga_widgets, False)
        self._set_visible(self.ms_widgets, False)

        scipy_methods = ('SLSQP', 'COBYLA', 'trust-constr')

        # General fields are shown only for the solvers that actually
        # consume them, so the dialog never lists a parameter that does
        # nothing for the selected algorithm.
        uses_maxiter = method != 'NSGA-II'  # NSGA-II uses Generations instead
        uses_tol = method in scipy_methods or method in ('Differential Evolution', 'Multi-Start')
        uses_atol = method == 'Differential Evolution'

        self._set_visible([self.edit_maxiter], uses_maxiter)
        self._set_visible([self.edit_tol], uses_tol)
        self._set_visible([self.edit_atol], uses_atol)
        # Scaling and objective scale apply to every solver.
        self._set_visible([self.chk_scaling], True)
        self._set_visible([self.edit_obj_scale], True)

        if method == 'Differential Evolution':
            self._set_visible(self.de_widgets, True)
        elif method == 'Nevergrad':
            self._set_visible(self.ng_widgets, True)
        elif method == 'NSGA-II':
            self._set_visible(self.nsga_widgets, True)
        elif method == 'Multi-Start':
            self._set_visible(self.ms_widgets, True)

    def _set_visible(self, widgets, visible):
        for w in widgets:
            w.setVisible(visible)
            label = self.form_layout.labelForField(w)
            if label:
                label.setVisible(visible)

    def load_settings(self):
        s = self.settings
        if not s: return
        
        self.edit_maxiter.setText(str(s.get('maxiter', 5000)))
        self.edit_tol.setText(str(s.get('tol', 1e-6)))
        self.edit_atol.setText(str(s.get('atol', 1e-8)))
        self.chk_scaling.setChecked(s.get('scaling', True))
        self.edit_obj_scale.setText(str(s.get('objective_scale', 1.0)))
        self.edit_con_margin.setText(str(s.get('constraint_margin', 0.001)))
        
        self.edit_popsize.setText(str(s.get('popsize', 15)))
        
        mut = s.get('mutation', (0.5, 1.0))
        self.edit_mut_min.setText(str(mut[0]))
        self.edit_mut_max.setText(str(mut[1]))
        
        self.edit_recomb.setText(str(s.get('recombination', 0.7)))
        self.combo_de_strat.setCurrentText(s.get('strategy', 'best1bin'))
        self.combo_ng_opt.setCurrentText(s.get('optimizer_name', 'NGOpt'))
        self.edit_ng_workers.setText(str(s.get('num_workers', 1)))

        # NSGA-II
        self.edit_nsga_popsize.setText(str(s.get('nsga_popsize', 100)))
        self.edit_nsga_generations.setText(str(s.get('nsga_generations', 200)))
        self.edit_nsga_crossover.setText(str(s.get('nsga_crossover_prob', 0.9)))
        mut_p = s.get('nsga_mutation_prob', None)
        self.edit_nsga_mutation.setText(str(mut_p) if mut_p is not None else "")
        self.edit_nsga_eta_c.setText(str(s.get('nsga_eta_c', 20.0)))
        self.edit_nsga_eta_m.setText(str(s.get('nsga_eta_m', 20.0)))

        # Multi-Start
        self.edit_ms_starts.setText(str(s.get('ms_n_starts', 10)))
        self.combo_ms_local.setCurrentText(s.get('ms_local_solver', 'SLSQP'))

    def get_settings(self):
        # Helper to parse safe float/int
        def to_f(txt, default):
            try: return float(txt)
            except: return default
        def to_i(txt, default):
            try: return int(float(txt)) # handle 1.0 as 1
            except: return default

        return {
            'maxiter': to_i(self.edit_maxiter.text(), 5000),
            'tol': to_f(self.edit_tol.text(), 1e-6),
            'atol': to_f(self.edit_atol.text(), 1e-8),
            'scaling': self.chk_scaling.isChecked(),
            'objective_scale': to_f(self.edit_obj_scale.text(), 1.0),
            'constraint_margin': to_f(self.edit_con_margin.text(), 0.001),
            'popsize': to_i(self.edit_popsize.text(), 15),
            'mutation': (to_f(self.edit_mut_min.text(), 0.5), to_f(self.edit_mut_max.text(), 1.0)),
            'recombination': to_f(self.edit_recomb.text(), 0.7),
            'strategy': self.combo_de_strat.currentText(),
            'optimizer_name': self.combo_ng_opt.currentText(),
            'num_workers': to_i(self.edit_ng_workers.text(), 1),
            # NSGA-II
            'nsga_popsize': to_i(self.edit_nsga_popsize.text(), 100),
            'nsga_generations': to_i(self.edit_nsga_generations.text(), 200),
            'nsga_crossover_prob': to_f(self.edit_nsga_crossover.text(), 0.9),
            'nsga_mutation_prob': to_f(self.edit_nsga_mutation.text(), None) if self.edit_nsga_mutation.text().strip() else None,
            'nsga_eta_c': to_f(self.edit_nsga_eta_c.text(), 20.0),
            'nsga_eta_m': to_f(self.edit_nsga_eta_m.text(), 20.0),
            # Multi-Start
            'ms_n_starts': to_i(self.edit_ms_starts.text(), 10),
            'ms_local_solver': self.combo_ms_local.currentText()
        }