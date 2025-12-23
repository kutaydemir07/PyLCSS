# Copyright (c) 2025 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

from PySide6 import QtWidgets

class AdvancedSettingsDialog(QtWidgets.QDialog):
    """Dialog for advanced optimization settings."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Advanced Optimization Settings")
        self.setModal(True)
        self.settings = {}
        self.init_ui()

    def init_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        
        # General Settings
        grp_general = QtWidgets.QGroupBox("General Solver Settings")
        form_general = QtWidgets.QFormLayout(grp_general)
        
        self.spin_tol = QtWidgets.QDoubleSpinBox()
        self.spin_tol.setRange(1e-12, 1.0)
        self.spin_tol.setValue(1e-4)
        self.spin_tol.setDecimals(12)
        self.spin_tol.setToolTip(
            "Convergence tolerance. For noisy black-box models (FEA/CFD), use 1e-3 to 1e-4.\n"
            "For smooth analytical functions, can use tighter values like 1e-6."
        )
        form_general.addRow("Tolerance:", self.spin_tol)
        
        self.spin_penalty = QtWidgets.QDoubleSpinBox()
        self.spin_penalty.setRange(1.0, 1e9)
        self.spin_penalty.setValue(1e6)
        self.spin_penalty.setToolTip(
            "Penalty weight for constraint violations (Nevergrad, Nelder-Mead).\n"
            "CRITICAL: Must be 10³-10⁶× larger than typical objective values.\n"
            "Example: If objective ~$50,000, use penalty ≥ 1e8."
        )
        form_general.addRow("Penalty Weight:", self.spin_penalty)
        
        layout.addWidget(grp_general)
        
        # Differential Evolution Settings
        grp_de = QtWidgets.QGroupBox("Differential Evolution")
        form_de = QtWidgets.QFormLayout(grp_de)
        
        self.spin_popsize = QtWidgets.QSpinBox()
        self.spin_popsize.setRange(5, 200)
        self.spin_popsize.setValue(15)
        self.spin_popsize.setToolTip("Population size multiplier (popsize * num_vars).")
        form_de.addRow("Population Size:", self.spin_popsize)
        
        self.spin_mutation_min = QtWidgets.QDoubleSpinBox()
        self.spin_mutation_min.setRange(0.0, 1.9)
        self.spin_mutation_min.setValue(0.5)
        self.spin_mutation_max = QtWidgets.QDoubleSpinBox()
        self.spin_mutation_max.setRange(0.0, 1.9)
        self.spin_mutation_max.setValue(1.0)
        mut_layout = QtWidgets.QHBoxLayout()
        mut_layout.addWidget(self.spin_mutation_min)
        mut_layout.addWidget(QtWidgets.QLabel("-"))
        mut_layout.addWidget(self.spin_mutation_max)
        form_de.addRow("Mutation Range:", mut_layout)
        
        self.spin_recombination = QtWidgets.QDoubleSpinBox()
        self.spin_recombination.setRange(0.0, 1.0)
        self.spin_recombination.setValue(0.7)
        form_de.addRow("Recombination:", self.spin_recombination)
        
        layout.addWidget(grp_de)
        
        # Nevergrad Settings
        grp_ng = QtWidgets.QGroupBox("Nevergrad")
        form_ng = QtWidgets.QFormLayout(grp_ng)
        
        self.combo_ng_opt = QtWidgets.QComboBox()
        self.combo_ng_opt.addItems(["NGOpt", "TwoPointsDE", "Portfolio", "OnePlusOne", "CMA"])
        self.combo_ng_opt.setToolTip("Specific Nevergrad optimizer variant.")
        form_ng.addRow("Optimizer:", self.combo_ng_opt)
        
        import os
        default_workers = max(1, os.cpu_count() or 1)
        self.spin_ng_workers = QtWidgets.QSpinBox()
        self.spin_ng_workers.setRange(1, 64)
        self.spin_ng_workers.setValue(default_workers)
        self.spin_ng_workers.setToolTip(
            f"Parallel evaluations for Nevergrad (detected {default_workers} CPU cores).\n"
            "Higher values dramatically speed up optimization but require thread-safe models.\n"
            "For expensive simulations (>1s each), set to CPU count for best performance."
        )
        form_ng.addRow("Num Workers:", self.spin_ng_workers)
        
        layout.addWidget(grp_ng)
        
        # Buttons
        btn_box = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        btn_box.accepted.connect(self.accept)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)

    def get_settings(self):
        return {
            'tol': self.spin_tol.value(),
            'penalty_weight': self.spin_penalty.value(),
            'popsize': self.spin_popsize.value(),
            'mutation': (self.spin_mutation_min.value(), self.spin_mutation_max.value()),
            'recombination': self.spin_recombination.value(),
            'optimizer_name': self.combo_ng_opt.currentText(),
            'num_workers': self.spin_ng_workers.value()
        }
