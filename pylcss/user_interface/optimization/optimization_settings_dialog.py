# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

import math

from PySide6 import QtWidgets, QtCore
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

        self.edit_maxiter = self._add_input(
            "Max Iterations:",
            str(optimization_config.DEFAULT_MAX_ITERATIONS),
        )
        self.edit_tol = self._add_input(
            "Tolerance (ftol):", str(optimization_config.DEFAULT_TOLERANCE)
        )
        self.edit_tol.setToolTip(
            "Stop when objective change is smaller than this value (e.g., 1e-6)."
        )

        self.edit_atol = self._add_input("Abs. Tolerance:", "1e-8")

        self.chk_scaling = QtWidgets.QCheckBox("Enable Variable Scaling")
        self.chk_scaling.setChecked(True)
        self.form_layout.addRow("Scaling:", self.chk_scaling)
        self.combo_scaling_mode = QtWidgets.QComboBox()
        self.combo_scaling_mode.addItem(
            "Automatic (log for wide positive ranges)",
            "auto",
        )
        self.combo_scaling_mode.addItem("Linear", "linear")
        self.combo_scaling_mode.addItem("Logarithmic", "log")
        self.combo_scaling_mode.setToolTip(
            "Automatic scaling uses logarithmic coordinates when positive bounds "
            "span at least three orders of magnitude."
        )
        self.form_layout.addRow("Coordinate Mode:", self.combo_scaling_mode)
        self.chk_scaling.toggled.connect(self.combo_scaling_mode.setEnabled)

        self.edit_con_margin = self._add_input("Constraint Safety Margin:", "0.0")
        self.edit_con_margin.setToolTip(
            "Optional relative engineering back-off applied to every bound. "
            "The default is zero because an arbitrary hidden margin changes the "
            "stated problem; use a non-zero value only when you intentionally "
            "want design reserve."
        )
        self.edit_seed = self._add_input("Random Seed:", "42")
        self.edit_seed.setToolTip(
            "Reproduces stochastic Differential Evolution, Nevergrad, NSGA-II, "
            "and Multi-Start sampling."
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
        self.combo_de_strat.addItems(
            ["best1bin", "rand1exp", "randtobest1bin", "currenttobest1bin"]
        )
        self.form_layout.addRow("Strategy:", self.combo_de_strat)

        self.de_widgets = [
            self.grp_de_label,
            self.edit_popsize,
            self.lbl_mutation,
            self.edit_mut_min,
            self.lbl_mut_dash,
            self.edit_mut_max,
            self.edit_recomb,
            self.combo_de_strat,
        ]

        # --- Nevergrad ---
        self.grp_ng_label = QtWidgets.QLabel("<b>Nevergrad</b>")
        self.grp_ng_label.setContentsMargins(0, 15, 0, 5)
        self.form_layout.addRow(self.grp_ng_label)

        self.combo_ng_opt = QtWidgets.QComboBox()
        self.combo_ng_opt.addItems(
            ["NGOpt", "TwoPointsDE", "Portfolio", "OnePlusOne", "CMA"]
        )
        self.form_layout.addRow("Optimizer:", self.combo_ng_opt)

        # Evaluations are deliberately sequential because arbitrary engineering
        # system models are not guaranteed to be thread/process safe.  A
        # "workers" field previously implied parallel execution that did not
        # actually exist.
        self.ng_widgets = [self.grp_ng_label, self.combo_ng_opt]

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

        self.nsga_widgets = [
            self.grp_nsga_label,
            self.edit_nsga_popsize,
            self.edit_nsga_generations,
            self.edit_nsga_crossover,
            self.edit_nsga_mutation,
            self.edit_nsga_eta_c,
            self.edit_nsga_eta_m,
        ]

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
        btn_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
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

        scipy_methods = ("SLSQP", "COBYLA", "trust-constr")

        # General fields are shown only for the solvers that actually
        # consume them, so the dialog never lists a parameter that does
        # nothing for the selected algorithm.
        uses_maxiter = method != "NSGA-II"  # NSGA-II uses Generations instead
        uses_tol = method in scipy_methods or method in (
            "Differential Evolution",
            "Multi-Start",
        )
        uses_atol = method == "Differential Evolution"

        maxiter_label = {
            "Nevergrad": "Evaluation Budget:",
            "Differential Evolution": "Max Generations:",
            "Multi-Start": "Iterations per Start:",
        }.get(method, "Max Iterations:")
        tol_label = (
            "Population Tolerance:"
            if method == "Differential Evolution"
            else "Termination Tolerance:"
        )
        self.form_layout.labelForField(self.edit_maxiter).setText(maxiter_label)
        self.form_layout.labelForField(self.edit_tol).setText(tol_label)
        self.form_layout.labelForField(self.edit_atol).setText(
            "Absolute Population Tolerance:"
        )
        self.form_layout.labelForField(self.edit_popsize).setText(
            "Population Multiplier:"
        )
        self.edit_popsize.setToolTip(
            "SciPy multiplier applied per free design variable; this is not the "
            "total population count."
        )

        self._set_visible([self.edit_maxiter], uses_maxiter)
        self._set_visible([self.edit_tol], uses_tol)
        self._set_visible([self.edit_atol], uses_atol)
        # Variable scaling applies to every solver. Objective scaling is defined
        # per objective in the main table so mixed engineering units stay sound.
        self._set_visible([self.chk_scaling, self.combo_scaling_mode], True)
        self._set_visible(
            [self.edit_seed],
            method in ("Differential Evolution", "Nevergrad", "NSGA-II", "Multi-Start"),
        )

        if method == "Differential Evolution":
            self._set_visible(self.de_widgets, True)
        elif method == "Nevergrad":
            self._set_visible(self.ng_widgets, True)
        elif method == "NSGA-II":
            self._set_visible(self.nsga_widgets, True)
        elif method == "Multi-Start":
            self._set_visible(self.ms_widgets, True)

    def _set_visible(self, widgets, visible):
        for w in widgets:
            w.setVisible(visible)
            label = self.form_layout.labelForField(w)
            if label:
                label.setVisible(visible)

    def load_settings(self):
        s = self.settings
        if not s:
            return

        self.edit_maxiter.setText(
            str(s.get("maxiter", optimization_config.DEFAULT_MAX_ITERATIONS))
        )
        self.edit_tol.setText(str(s.get("tol", 1e-6)))
        self.edit_atol.setText(str(s.get("atol", 1e-8)))
        self.chk_scaling.setChecked(s.get("scaling", True))
        scaling_mode = str(s.get("scaling_mode", "auto")).lower()
        scaling_index = self.combo_scaling_mode.findData(scaling_mode)
        self.combo_scaling_mode.setCurrentIndex(max(0, scaling_index))
        self.combo_scaling_mode.setEnabled(self.chk_scaling.isChecked())
        self.edit_con_margin.setText(str(s.get("constraint_margin", 0.0)))
        self.edit_seed.setText(str(s.get("seed", 42)))

        self.edit_popsize.setText(str(s.get("popsize", 15)))

        mut = s.get("mutation", (0.5, 1.0))
        self.edit_mut_min.setText(str(mut[0]))
        self.edit_mut_max.setText(str(mut[1]))

        self.edit_recomb.setText(str(s.get("recombination", 0.7)))
        self.combo_de_strat.setCurrentText(s.get("strategy", "best1bin"))
        self.combo_ng_opt.setCurrentText(s.get("optimizer_name", "NGOpt"))

        # NSGA-II
        self.edit_nsga_popsize.setText(str(s.get("nsga_popsize", 100)))
        self.edit_nsga_generations.setText(str(s.get("nsga_generations", 200)))
        self.edit_nsga_crossover.setText(str(s.get("nsga_crossover_prob", 0.9)))
        mut_p = s.get("nsga_mutation_prob", None)
        self.edit_nsga_mutation.setText(str(mut_p) if mut_p is not None else "")
        self.edit_nsga_eta_c.setText(str(s.get("nsga_eta_c", 20.0)))
        self.edit_nsga_eta_m.setText(str(s.get("nsga_eta_m", 20.0)))

        # Multi-Start
        self.edit_ms_starts.setText(str(s.get("ms_n_starts", 10)))
        self.combo_ms_local.setCurrentText(s.get("ms_local_solver", "SLSQP"))

    def get_settings(self):
        # Helper to parse safe float/int
        def to_f(txt, default):
            try:
                return float(txt)
            except (TypeError, ValueError):
                return default

        def to_i(txt, default):
            try:
                return int(float(txt))  # Accept integral text such as "1.0".
            except (TypeError, ValueError, OverflowError):
                return default

        return {
            "maxiter": to_i(
                self.edit_maxiter.text(),
                optimization_config.DEFAULT_MAX_ITERATIONS,
            ),
            "tol": to_f(self.edit_tol.text(), 1e-6),
            "atol": to_f(self.edit_atol.text(), 1e-8),
            "scaling": self.chk_scaling.isChecked(),
            "scaling_mode": self.combo_scaling_mode.currentData(),
            "constraint_margin": to_f(self.edit_con_margin.text(), 0.0),
            "seed": to_i(self.edit_seed.text(), 42),
            "popsize": to_i(self.edit_popsize.text(), 15),
            "mutation": (
                to_f(self.edit_mut_min.text(), 0.5),
                to_f(self.edit_mut_max.text(), 1.0),
            ),
            "recombination": to_f(self.edit_recomb.text(), 0.7),
            "strategy": self.combo_de_strat.currentText(),
            "optimizer_name": self.combo_ng_opt.currentText(),
            # NSGA-II
            "nsga_popsize": to_i(self.edit_nsga_popsize.text(), 100),
            "nsga_generations": to_i(self.edit_nsga_generations.text(), 200),
            "nsga_crossover_prob": to_f(self.edit_nsga_crossover.text(), 0.9),
            "nsga_mutation_prob": to_f(self.edit_nsga_mutation.text(), None)
            if self.edit_nsga_mutation.text().strip()
            else None,
            "nsga_eta_c": to_f(self.edit_nsga_eta_c.text(), 20.0),
            "nsga_eta_m": to_f(self.edit_nsga_eta_m.text(), 20.0),
            # Multi-Start
            "ms_n_starts": to_i(self.edit_ms_starts.text(), 10),
            "ms_local_solver": self.combo_ms_local.currentText(),
        }

    def accept(self):
        """Reject unsafe or nonsensical settings before starting a worker."""
        settings = self.get_settings()
        errors = []

        if settings["maxiter"] < 1:
            errors.append("Iteration/evaluation budget must be at least 1.")
        if (
            not math.isfinite(settings["tol"])
            or not math.isfinite(settings["atol"])
            or settings["tol"] <= 0
            or settings["atol"] < 0
        ):
            errors.append(
                "Tolerances must be positive (absolute tolerance may be zero)."
            )
        if (
            not math.isfinite(settings["constraint_margin"])
            or not 0 <= settings["constraint_margin"] < 1
        ):
            errors.append("Constraint safety margin must be in [0, 1).")

        if self.current_method == "Differential Evolution":
            mut_lo, mut_hi = settings["mutation"]
            if settings["popsize"] < 1:
                errors.append("Population multiplier must be at least 1.")
            if (
                not math.isfinite(mut_lo)
                or not math.isfinite(mut_hi)
                or not 0 <= mut_lo <= mut_hi < 2
            ):
                errors.append("Mutation range must satisfy 0 ≤ minimum ≤ maximum < 2.")
            if (
                not math.isfinite(settings["recombination"])
                or not 0 <= settings["recombination"] <= 1
            ):
                errors.append("Recombination must be between 0 and 1.")

        if self.current_method == "NSGA-II":
            if settings["nsga_popsize"] < 4:
                errors.append("NSGA-II population size must be at least 4.")
            if settings["nsga_generations"] < 1:
                errors.append("NSGA-II generations must be at least 1.")
            if (
                not math.isfinite(settings["nsga_crossover_prob"])
                or not 0 <= settings["nsga_crossover_prob"] <= 1
            ):
                errors.append("Crossover probability must be between 0 and 1.")
            mutation = settings["nsga_mutation_prob"]
            if mutation is not None and (
                not math.isfinite(mutation) or not 0 <= mutation <= 1
            ):
                errors.append("Mutation probability must be between 0 and 1.")
            if (
                not math.isfinite(settings["nsga_eta_c"])
                or not math.isfinite(settings["nsga_eta_m"])
                or settings["nsga_eta_c"] <= 0
                or settings["nsga_eta_m"] <= 0
            ):
                errors.append("NSGA-II distribution indices must be greater than zero.")

        if self.current_method == "Multi-Start" and settings["ms_n_starts"] < 1:
            errors.append("Number of starts must be at least 1.")

        if errors:
            QtWidgets.QMessageBox.warning(
                self,
                "Invalid Optimization Settings",
                "\n".join(errors),
            )
            return
        super().accept()
