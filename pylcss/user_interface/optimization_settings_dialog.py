# Copyright (c) 2025 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

from PySide6 import QtWidgets, QtCore, QtGui
from ..config import optimization_config

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
        self.edit_maxfun = self._add_input("Max Func Evals:", "15000")

        # --- Differential Evolution (DE) ---
        self.grp_de_label = QtWidgets.QLabel("<b>Differential Evolution</b>")
        self.grp_de_label.setContentsMargins(0, 15, 0, 5)
        self.form_layout.addRow(self.grp_de_label)
        
        self.edit_popsize = self._add_input("Population Size:", "15")
        
        # Mutation Range (Min - Max)
        self.edit_mut_min = QtWidgets.QLineEdit("0.5")
        self.edit_mut_max = QtWidgets.QLineEdit("1.0")
        h_mut = QtWidgets.QHBoxLayout()
        h_mut.addWidget(self.edit_mut_min)
        h_mut.addWidget(QtWidgets.QLabel("-"))
        h_mut.addWidget(self.edit_mut_max)
        self.lbl_mutation = QtWidgets.QLabel("Mutation Range:")
        self.form_layout.addRow(self.lbl_mutation, h_mut)
        
        self.edit_recomb = self._add_input("Recombination:", "0.7")
        
        self.combo_de_strat = QtWidgets.QComboBox()
        self.combo_de_strat.addItems(['best1bin', 'rand1exp', 'randtobest1bin', 'currenttobest1bin'])
        self.form_layout.addRow("Strategy:", self.combo_de_strat)
        
        self.de_widgets = [self.grp_de_label, self.edit_popsize, self.lbl_mutation, 
                           self.edit_mut_min, self.edit_mut_max, self.edit_recomb, self.combo_de_strat]

        # --- Nevergrad ---
        self.grp_ng_label = QtWidgets.QLabel("<b>Nevergrad</b>")
        self.grp_ng_label.setContentsMargins(0, 15, 0, 5)
        self.form_layout.addRow(self.grp_ng_label)
        
        self.combo_ng_opt = QtWidgets.QComboBox()
        self.combo_ng_opt.addItems(["NGOpt", "TwoPointsDE", "Portfolio", "OnePlusOne", "CMA"])
        self.form_layout.addRow("Optimizer:", self.combo_ng_opt)
        
        self.edit_ng_workers = self._add_input("Num Workers:", "1")
        
        self.ng_widgets = [self.grp_ng_label, self.combo_ng_opt, self.edit_ng_workers]

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
        # Hide everything specific first
        self._set_visible(self.de_widgets, False)
        self._set_visible(self.ng_widgets, False)
        
        # Defaults for common
        common_visible = True
        self.form_layout.labelForField(self.edit_tol).setVisible(True)
        self.edit_tol.setVisible(True)
        self.form_layout.labelForField(self.edit_maxfun).setVisible(True)
        self.edit_maxfun.setVisible(True)

        if method == 'Differential Evolution':
            self._set_visible(self.de_widgets, True)
            # DE doesn't typically use 'tol' or 'maxfun' in the same way scipy does here
            self.form_layout.labelForField(self.edit_tol).setVisible(False)
            self.edit_tol.setVisible(False)
            self.form_layout.labelForField(self.edit_maxfun).setVisible(False)
            self.edit_maxfun.setVisible(False)
            
        elif method == 'Nevergrad':
            self._set_visible(self.ng_widgets, True)
            self.form_layout.labelForField(self.edit_tol).setVisible(False)
            self.edit_tol.setVisible(False)

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
        self.edit_maxfun.setText(str(s.get('maxfun', 15000)))
        
        self.edit_popsize.setText(str(s.get('popsize', 15)))
        
        mut = s.get('mutation', (0.5, 1.0))
        self.edit_mut_min.setText(str(mut[0]))
        self.edit_mut_max.setText(str(mut[1]))
        
        self.edit_recomb.setText(str(s.get('recombination', 0.7)))
        self.combo_de_strat.setCurrentText(s.get('strategy', 'best1bin'))
        self.combo_ng_opt.setCurrentText(s.get('optimizer_name', 'NGOpt'))
        self.edit_ng_workers.setText(str(s.get('num_workers', 1)))

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
            'maxfun': to_i(self.edit_maxfun.text(), 15000),
            'popsize': to_i(self.edit_popsize.text(), 15),
            'mutation': (to_f(self.edit_mut_min.text(), 0.5), to_f(self.edit_mut_max.text(), 1.0)),
            'recombination': to_f(self.edit_recomb.text(), 0.7),
            'strategy': self.combo_de_strat.currentText(),
            'optimizer_name': self.combo_ng_opt.currentText(),
            'num_workers': to_i(self.edit_ng_workers.text(), 1)
        }