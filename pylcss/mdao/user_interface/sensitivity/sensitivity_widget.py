# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""
Sensitivity Analysis Interface for PyLCSS.

Provides a GUI for performing global sensitivity analysis using the State-of-the-Art (SOTA)
standard in MDAO: Sobol' global variance decomposition (Saltelli sequence sampling).
Estimates:
- First-order indices (S1): main effects of individual design variables
- Total-order indices (ST): total effects including all higher-order interactions
- Second-order indices (S2): pairwise coupling interaction matrix

Features: S2 interaction heatmap, multi-output batch evaluation,
and an importance-ranked sensitivity data table.
"""

import numpy as np
import html
from PySide6 import QtCore, QtWidgets
import pyqtgraph as pg
import logging
from ...sensitivity import SensitivityAnalyzer
from pylcss.user_interface.common.empty_state import EmptyStateOverlay
from pylcss.user_interface.common.theme_manager import COLORS
from pylcss.user_interface.common.ui_kit import (
    CollapsibleSection,
    SegmentedControl,
    secondary_button,
)
from .workers import EvalTimeWorker, OutputRefreshWorker, SensitivityWorker

logger = logging.getLogger(__name__)

_CANCEL_QSS = (
    "QPushButton { background: transparent; color: #cf222e;"
    " border: 1px solid #cf222e; border-radius: 4px; padding: 8px; font-weight: bold; }"
    " QPushButton:hover { background: rgba(207, 34, 46, 25); }"
    " QPushButton:disabled { color: rgba(128, 128, 128, 160);"
    " border-color: rgba(128, 128, 128, 140); }"
)
_WARN_COLOR = "#b7791f"
_PRESETS = (256, 1024, 4096)
_LARGE_RUN = 100_000


class SensitivityAnalysisWidget(QtWidgets.QWidget):
    """
    Widget for performing Sobol' global variance decomposition.

    Visualizations include main and total effects bar chart with confidence
    intervals, pairwise interaction heatmap (S2), and an importance-ranked
    sensitivity table.
    """

    def apply_theme(self, theme_name):
        palette = COLORS
        for plot in (self.plot_widget, self.heatmap_widget):
            plot.setBackground(palette["chart_bg"])
            for axis_name in ("left", "bottom"):
                axis = plot.getAxis(axis_name)
                axis.setPen(palette["chart_fg"])
                axis.setTextPen(palette["chart_fg"])

    def __init__(
        self,
        optimization_widget: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__()
        self.optimization_widget = optimization_widget

        try:
            self.analyzer = SensitivityAnalyzer()
            self.salib_available = True
        except ImportError:
            self.analyzer = None
            self.salib_available = False

        self.current_results = None
        self.batch_results = {}  # output_name -> results
        self.worker = None
        self.refresh_worker = None
        self.timing_worker = None
        self._eval_seconds = None  # measured seconds per model call
        self._timed_problem = None
        self._running = False

        self.setup_ui()

        if not self.salib_available:
            self.lbl_status.setText(
                "Error: SALib not installed. Sensitivity Analysis disabled."
            )
            self.lbl_status.setStyleSheet(
                "background-color: #e74c3c; color: white; padding: 10px;"
            )
            self.btn_analyze.setEnabled(False)

    # ====================================================================
    # UI Setup
    # ====================================================================

    def setup_ui(self):
        layout = QtWidgets.QHBoxLayout(self)

        # --- LEFT PANEL: setup steps + pinned run area ---
        config_panel = QtWidgets.QWidget()
        config_panel.setFixedWidth(400)
        config_layout = QtWidgets.QVBoxLayout(config_panel)
        config_layout.setContentsMargins(0, 0, 0, 0)
        config_layout.setSpacing(4)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        scroll.setStyleSheet("QScrollArea { background: transparent; border: none; }")
        steps = QtWidgets.QWidget()
        steps_layout = QtWidgets.QVBoxLayout(steps)
        steps_layout.setContentsMargins(6, 6, 6, 6)
        steps_layout.setSpacing(8)
        scroll.setWidget(steps)
        config_layout.addWidget(scroll, 1)

        # Legacy attributes preserved for backwards compatibility with project serialization
        self.combo_method = QtWidgets.QComboBox()
        self.combo_method.addItem("Sobol")
        self.combo_method.setVisible(False)
        self.lbl_method_desc = QtWidgets.QLabel()
        self.lbl_method_desc.setVisible(False)

        # 1. Outputs
        self.sec_outputs = CollapsibleSection("1  Outputs")
        l_output = self.sec_outputs.body_layout
        self.seg_outputs = SegmentedControl(["One output", "All outputs"])
        self.seg_outputs.setToolTip(
            "All outputs runs one shared sample design for every output; "
            "switch outputs below to view each result."
        )
        self.seg_outputs.set_index(0)
        self.seg_outputs.index_changed.connect(
            lambda index: self.chk_batch.setChecked(index == 1)
        )
        l_output.addWidget(self.seg_outputs)

        out_row = QtWidgets.QHBoxLayout()
        self.lbl_output_pick = QtWidgets.QLabel("Output:")
        out_row.addWidget(self.lbl_output_pick)
        self.combo_outputs = QtWidgets.QComboBox()
        self.combo_outputs.setToolTip(
            "Select the output variable to analyze. After a "
            "batch run, switch outputs here to view each result."
        )
        self.combo_outputs.currentIndexChanged.connect(self._on_output_selected)
        out_row.addWidget(self.combo_outputs, 1)
        self.btn_refresh_outputs = secondary_button(QtWidgets.QPushButton("\u21bb"))
        self.btn_refresh_outputs.setFixedWidth(32)
        self.btn_refresh_outputs.setToolTip("Refresh the outputs from the model")
        self.btn_refresh_outputs.clicked.connect(self.refresh_outputs)
        out_row.addWidget(self.btn_refresh_outputs)
        l_output.addLayout(out_row)

        # Kept (hidden) as the single source of truth for project save/load
        self.chk_batch = QtWidgets.QCheckBox("Analyze all outputs (batch)")
        self.chk_batch.setVisible(False)
        self.chk_batch.toggled.connect(self._on_batch_toggled)
        steps_layout.addWidget(self.sec_outputs)

        # 2. Accuracy: sample-size presets and optional second-order indices
        self.sec_accuracy = CollapsibleSection("2  Accuracy")
        l_acc = self.sec_accuracy.body_layout
        self.seg_presets = SegmentedControl(["Quick", "Standard", "Accurate"])
        self.seg_presets.setToolTip(
            "Base sample size N: Quick 256, Standard 1024, Accurate 4096."
        )
        self.seg_presets.index_changed.connect(self._on_preset_clicked)
        l_acc.addWidget(self.seg_presets)

        l_config = QtWidgets.QFormLayout()
        self.spin_samples = QtWidgets.QSpinBox()
        self.spin_samples.setRange(64, 16384)
        self.spin_samples.setValue(1024)
        self.spin_samples.setSingleStep(256)
        self.spin_samples.setToolTip(
            "Base sample size N (Saltelli sequence). The total model evaluations "
            "equal N × (D + 2) without S2, or N × (2D + 2) with S2 pairwise interactions."
        )
        l_config.addRow("Base sample size (N):", self.spin_samples)
        self.lbl_samples = l_config.labelForField(self.spin_samples)
        l_acc.addLayout(l_config)

        self.chk_second_order = QtWidgets.QCheckBox("Pairwise interactions (S2)")
        self.chk_second_order.setChecked(False)
        self.chk_second_order.setToolTip(
            "When enabled, computes second-order interaction indices (S2) and "
            "interaction heatmap, increasing model runs from N×(D+2) to N×(2D+2)."
        )
        l_acc.addWidget(self.chk_second_order)
        self.lbl_s2_cost = QtWidgets.QLabel()
        self.lbl_s2_cost.setWordWrap(True)
        self.lbl_s2_cost.setStyleSheet(f"font-size:10px; color:{COLORS['text_dim']};")
        l_acc.addWidget(self.lbl_s2_cost)
        steps_layout.addWidget(self.sec_accuracy)

        # 3. Variables: ranges that are sampled, with the option to hold one fixed
        self.sec_variables = CollapsibleSection("3  Variables")
        self.table_vars = QtWidgets.QTableWidget(0, 5)
        self.table_vars.setHorizontalHeaderLabels(
            ["Vary", "Variable", "Min", "Max", "Held at"]
        )
        self.table_vars.verticalHeader().setVisible(False)
        self.table_vars.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table_vars.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        self.table_vars.setMaximumHeight(200)
        vars_header = self.table_vars.horizontalHeader()
        vars_header.setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
        vars_header.setSectionResizeMode(1, QtWidgets.QHeaderView.Stretch)
        self.table_vars.setToolTip(
            "Untick a variable to hold it at its current value instead of sampling it."
        )
        self.table_vars.itemChanged.connect(self._on_vary_changed)
        self.sec_variables.body_layout.addWidget(self.table_vars)
        self.sec_variables.set_expanded(False)
        steps_layout.addWidget(self.sec_variables)

        # 4. Advanced
        self.sec_advanced = CollapsibleSection("4  Advanced")
        l_adv = QtWidgets.QFormLayout()
        self.spin_seed = QtWidgets.QSpinBox()
        self.spin_seed.setRange(0, 2_147_483_647)
        self.spin_seed.setValue(42)
        self.spin_seed.setToolTip(
            "Random seed for reproducible quasi-Monte Carlo sampling and confidence intervals."
        )
        l_adv.addRow("Random seed:", self.spin_seed)
        self.sec_advanced.body_layout.addLayout(l_adv)
        self.sec_advanced.set_expanded(False)
        steps_layout.addWidget(self.sec_advanced)

        # Legacy widgets kept hidden for save/load compatibility
        self.spin_trajectories = QtWidgets.QSpinBox()
        self.spin_trajectories.setValue(20)
        self.spin_trajectories.setVisible(False)
        self.lbl_trajectories = QtWidgets.QLabel()
        self.lbl_trajectories.setVisible(False)
        self.spin_morris_levels = QtWidgets.QSpinBox()
        self.spin_morris_levels.setValue(4)
        self.spin_morris_levels.setVisible(False)
        self.lbl_morris_levels = QtWidgets.QLabel()
        self.lbl_morris_levels.setVisible(False)

        self.spin_samples.valueChanged.connect(self._on_samples_changed)
        self.chk_second_order.toggled.connect(self._on_s2_toggled)
        steps_layout.addStretch()

        # Pinned run area: cost first, then Run; Cancel and progress only while running
        run_area = QtWidgets.QWidget()
        run_layout = QtWidgets.QVBoxLayout(run_area)
        run_layout.setContentsMargins(6, 2, 6, 6)
        run_layout.setSpacing(6)

        self.lbl_eval_estimate = QtWidgets.QLabel()
        self.lbl_eval_estimate.setWordWrap(True)
        self.lbl_eval_estimate.setTextFormat(QtCore.Qt.RichText)
        run_layout.addWidget(self.lbl_eval_estimate)

        self.lbl_run_hint = QtWidgets.QLabel()
        self.lbl_run_hint.setWordWrap(True)
        self.lbl_run_hint.setStyleSheet("color: gray; font-style: italic;")
        run_layout.addWidget(self.lbl_run_hint)

        btn_row = QtWidgets.QHBoxLayout()
        self.btn_analyze = QtWidgets.QPushButton("Run Sensitivity Analysis")
        self.btn_analyze.setStyleSheet("font-weight: bold; padding: 8px;")
        self.btn_analyze.clicked.connect(self.run_analysis)
        btn_row.addWidget(self.btn_analyze, 1)
        self.btn_cancel = QtWidgets.QPushButton("Cancel")
        self.btn_cancel.setStyleSheet(_CANCEL_QSS)
        self.btn_cancel.clicked.connect(self.cancel_analysis)
        btn_row.addWidget(self.btn_cancel)
        run_layout.addLayout(btn_row)

        self.progress = QtWidgets.QProgressBar()
        run_layout.addWidget(self.progress)
        config_layout.addWidget(run_area)

        layout.addWidget(config_panel)

        # --- RIGHT PANEL: summary strip, chart(s) and table in one view ---
        viz_panel = QtWidgets.QWidget()
        viz_layout = QtWidgets.QVBoxLayout(viz_panel)

        strip = QtWidgets.QHBoxLayout()
        self.lbl_status = QtWidgets.QLabel("Status: Ready to analyze.")
        self.lbl_status.setWordWrap(True)
        self.lbl_status.setTextFormat(QtCore.Qt.RichText)
        self.lbl_status.setStyleSheet(
            "background-color: #1e1f22; border: 1px solid #282a2e; color: #b5bac1; padding: 6px 10px; border-radius: 4px;"
        )
        strip.addWidget(self.lbl_status, 1)
        self.btn_export = secondary_button(QtWidgets.QPushButton("Export results"))
        self.btn_export.setEnabled(False)
        self.btn_export.clicked.connect(self.export_results)
        strip.addWidget(self.btn_export)
        viz_layout.addLayout(strip)

        # Main & total effects bar chart
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground(COLORS["chart_bg"])
        self.plot_widget.setLabel("left", "Sensitivity index")
        self.plot_widget.getAxis("left").enableAutoSIPrefix(False)
        self.plot_widget.setLabel("bottom", "Design variables")
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.plot_widget.getAxis("bottom").setTicks([[]])
        self._legend = self.plot_widget.addLegend(offset=(10, 10))
        self._plot_empty = EmptyStateOverlay(
            self.plot_widget,
            "Run sensitivity analysis to view results.",
            lambda: self.current_results is None,
        )

        # S2 interaction heatmap (shown only when S2 is on)
        self.heatmap_widget = pg.PlotWidget()
        self.heatmap_widget.setBackground(COLORS["chart_bg"])
        self.heatmap_widget.setLabel("left", "Variable")
        self.heatmap_widget.setLabel("bottom", "Variable")
        self.heatmap_widget.setVisible(False)

        charts = QtWidgets.QWidget()
        charts_layout = QtWidgets.QHBoxLayout(charts)
        charts_layout.setContentsMargins(0, 0, 0, 0)
        charts_layout.addWidget(self.plot_widget, 3)
        charts_layout.addWidget(self.heatmap_widget, 2)

        # Results table
        self.results_table = QtWidgets.QTableWidget()
        self.results_table.setColumnCount(6)
        self.results_table.setHorizontalHeaderLabels(
            [
                "Rank",
                "Variable",
                "First Order (S1)",
                "Total Order (ST)",
                "Interaction (ST - S1)",
                "ST Confidence",
            ]
        )
        self.results_table.verticalHeader().setVisible(False)
        self.results_table.setAlternatingRowColors(True)
        self.results_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.results_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        header = self.results_table.horizontalHeader()
        header.setMinimumSectionSize(65)
        header.setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        header.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeToContents)
        header.setDefaultAlignment(QtCore.Qt.AlignCenter)
        self.results_split = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        self.results_split.addWidget(charts)
        self.results_split.addWidget(self.results_table)
        self.results_split.setSizes([430, 220])
        viz_layout.addWidget(self.results_split, 1)

        layout.addWidget(viz_panel)

        # Initialize visibility and estimate
        self.spin_seed.valueChanged.connect(self._refresh_steps)
        self.combo_outputs.currentIndexChanged.connect(self._refresh_steps)
        self._on_s2_toggled(self.chk_second_order.isChecked())
        self._on_preset_sync()
        self._set_running(False)
        self._refresh_steps()

    def _on_s2_toggled(self, checked: bool = False):
        """Update the cost estimate and show the S2 heatmap when it applies."""
        self._update_estimate()
        self._update_heatmap_visibility()
        self._refresh_steps()

    def _update_heatmap_visibility(self):
        has_s2 = bool(
            self.current_results
            and self.current_results.get("calc_second_order", False)
            and self.current_results.get("s2_matrix") is not None
        )
        self.heatmap_widget.setVisible(self.chk_second_order.isChecked() and has_s2)

    def _on_method_changed(self, method: str = "Sobol"):
        """Backward compatibility hook."""
        self._update_estimate()

    def _on_batch_toggled(self, checked: bool):
        self.seg_outputs.set_index(1 if checked else 0)
        self.lbl_output_pick.setText("Show:" if checked else "Output:")
        self._refresh_steps()

    def _on_preset_clicked(self, index: int):
        self.spin_samples.setValue(_PRESETS[index])

    def _on_samples_changed(self, *_):
        self._on_preset_sync()
        self._refresh_steps()

    def _on_preset_sync(self):
        """Highlight the matching preset (none for a custom N) and refresh the cost."""
        value = self.spin_samples.value()
        self.seg_presets.set_index(_PRESETS.index(value) if value in _PRESETS else None)
        self._update_estimate()

    # -- variables step ------------------------------------------------------

    def _problem(self):
        ow = self.optimization_widget
        return ow.problem if (ow and ow.problem) else None

    def _varied_names(self):
        names = []
        for row in range(self.table_vars.rowCount()):
            if self.table_vars.item(row, 0).checkState() == QtCore.Qt.Checked:
                names.append(self.table_vars.item(row, 1).text())
        return names

    def _fixed_values(self):
        """Variables the user unticked, mapped to the value they are held at."""
        fixed = {}
        for row in range(self.table_vars.rowCount()):
            if self.table_vars.item(row, 0).checkState() != QtCore.Qt.Checked:
                fixed[self.table_vars.item(row, 1).text()] = float(
                    self.table_vars.item(row, 4).data(QtCore.Qt.UserRole)
                )
        return fixed

    def _varied_count(self):
        problem = self._problem()
        if problem is None:
            return 0
        if self.table_vars.rowCount():
            return len(self._varied_names())
        return len(problem.design_variables)

    def _populate_variables(self):
        problem = self._problem()
        held = set(self._fixed_values()) if self.table_vars.rowCount() else set()
        rows = list(problem.design_variables) if problem else []
        self.table_vars.blockSignals(True)
        self.table_vars.setRowCount(len(rows))
        for row, dv in enumerate(rows):
            name = str(dv["name"])
            low, high = float(dv["min"]), float(dv["max"])
            try:
                nominal = float(dv.get("value", (low + high) / 2.0))
            except (TypeError, ValueError):
                nominal = (low + high) / 2.0
            nominal = min(max(nominal, low), high)

            check = QtWidgets.QTableWidgetItem()
            check.setFlags(QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsEnabled)
            check.setCheckState(QtCore.Qt.Unchecked if name in held else QtCore.Qt.Checked)
            held_item = QtWidgets.QTableWidgetItem(
                f"{nominal:.4g}" if name in held else "\u2014"
            )
            held_item.setData(QtCore.Qt.UserRole, nominal)
            cells = (
                check,
                QtWidgets.QTableWidgetItem(name),
                QtWidgets.QTableWidgetItem(f"{low:.4g}"),
                QtWidgets.QTableWidgetItem(f"{high:.4g}"),
                held_item,
            )
            for column, cell in enumerate(cells):
                if column:
                    cell.setFlags(QtCore.Qt.ItemIsEnabled)
                self.table_vars.setItem(row, column, cell)
        self.table_vars.blockSignals(False)

    def _on_vary_changed(self, item):
        if item.column() != 0:
            return
        held_item = self.table_vars.item(item.row(), 4)
        varies = item.checkState() == QtCore.Qt.Checked
        self.table_vars.blockSignals(True)
        held_item.setText("\u2014" if varies else f"{held_item.data(QtCore.Qt.UserRole):.4g}")
        self.table_vars.blockSignals(False)
        self._update_estimate()
        self._refresh_steps()

    # -- step summaries and run state ---------------------------------------

    def _refresh_steps(self, *_args):
        if not hasattr(self, "sec_advanced"):
            return
        problem = self._problem()
        n_total = len(problem.design_variables) if problem else 0
        n_varied = self._varied_count()
        batch = self.chk_batch.isChecked()
        output = self.combo_outputs.currentData() or self.combo_outputs.currentText()

        self.sec_outputs.set_state(
            "done" if problem and (batch or self.combo_outputs.count()) else "current"
        )
        self.sec_outputs.set_summary(
            "All outputs" if batch else (output or "No output selected")
        )
        self.sec_accuracy.set_state("done")
        self.sec_accuracy.set_summary(
            f"N = {self.spin_samples.value()}"
            + (" \u00b7 with S2" if self.chk_second_order.isChecked() else "")
        )
        self.sec_variables.set_state("done" if n_total else "todo")
        self.sec_variables.set_summary(f"{n_varied} of {n_total} variables vary")
        self.sec_advanced.set_state("done")
        self.sec_advanced.set_summary(f"Seed {self.spin_seed.value()}")
        self._refresh_run_state()

    def _refresh_run_state(self):
        """Enable Run only when it can work and say why not when it cannot."""
        problem = self._problem()
        reason = ""
        if not self.salib_available:
            reason = "SALib is not installed."
        elif problem is None:
            reason = "Load a system model first."
        elif not self.chk_batch.isChecked() and self.combo_outputs.currentIndex() < 0:
            reason = "Pick an output (use the refresh button)."
        elif self.table_vars.rowCount() and not self._varied_names():
            reason = "Keep at least one variable varying."
        self.btn_analyze.setEnabled(not self._running and not reason)
        self.lbl_run_hint.setText(reason)
        self.lbl_run_hint.setVisible(bool(reason) and not self._running)

    def _set_running(self, running: bool):
        """Cancel and the progress bar exist only while an analysis runs."""
        self._running = running
        self.btn_cancel.setVisible(running)
        self.progress.setVisible(running)
        self._refresh_run_state()

    @staticmethod
    def _format_duration(seconds: float) -> str:
        if seconds < 1:
            return "under a second"
        if seconds < 90:
            return f"about {seconds:.0f} s"
        if seconds < 5400:
            return f"about {seconds / 60:.0f} min"
        return f"about {seconds / 3600:.1f} h"

    def _update_estimate(self):
        """Show the number of model evaluations and, once timed, how long they take."""
        problem = self._problem()
        n_vars = self._varied_count()
        if n_vars == 0:
            self.lbl_eval_estimate.setText("Load a problem to estimate model runs.")
            self.lbl_s2_cost.setText("")
            return

        N = self.spin_samples.value()
        has_s2 = self.chk_second_order.isChecked()
        multiplier = 2 * n_vars + 2 if has_s2 else n_vars + 2
        evals = N * multiplier
        seconds = self._eval_seconds if self._timed_problem is problem else None
        total_time = evals * seconds if seconds else None

        headline = f"<b>{evals:,} model evaluations</b>"
        if total_time is not None:
            headline += f" \u00b7 {self._format_duration(total_time)}"
        if evals > _LARGE_RUN or (total_time or 0) > 600:
            headline = f"<span style='color:{_WARN_COLOR};'>{headline}</span>"
        detail = (
            f"N={N} \u00d7 ({'2D+2' if has_s2 else 'D+2'} = {multiplier}) "
            f"for D={n_vars} vars"
        )
        self.lbl_eval_estimate.setText(
            f"<span style='font-size:11pt;'>{headline}</span><br>"
            f"<span style='font-size:8pt; color:gray;'>{detail}</span>"
        )
        self.lbl_s2_cost.setText("")

    def _start_eval_timing(self):
        problem = self._problem()
        if problem is None or not callable(getattr(problem, "system_model", None)):
            return
        if self._timed_problem is problem and self._eval_seconds is not None:
            return
        if self.timing_worker is not None:
            return
        worker = EvalTimeWorker(problem)
        worker.done_sig.connect(
            lambda seconds, p=problem: self._on_eval_timed(p, seconds)
        )
        worker.finished.connect(self._on_timing_stopped)
        worker.finished.connect(worker.deleteLater)
        self.timing_worker = worker
        worker.start()

    def _on_eval_timed(self, problem, seconds):
        self._timed_problem = problem
        self._eval_seconds = seconds
        self._update_estimate()

    def _on_timing_stopped(self):
        self.timing_worker = None

    def _summary_html(self, results, name):
        """Clean summary line for an output."""
        return f"Completed · <b>{html.escape(str(name))}</b>"

    # ====================================================================
    # Refresh Outputs
    # ====================================================================

    def refresh_outputs(self):
        previous = self.combo_outputs.currentData()
        self.combo_outputs.clear()
        problem = self._problem()
        if problem is None:
            self._populate_variables()
            self._update_estimate()
            self._refresh_steps()
            return

        for qoi in problem.quantities_of_interest:
            self.combo_outputs.addItem(f"{qoi['name']} ({qoi['unit']})", qoi["name"])
        restored = self.combo_outputs.findData(previous)
        if restored >= 0:
            self.combo_outputs.setCurrentIndex(restored)

        # Variable count is now known: refresh the table, the estimate and the timing.
        self._populate_variables()
        self._start_eval_timing()
        self._update_estimate()
        self._refresh_steps()

        if self.combo_outputs.count() == 0 and problem.system_model:
            self.btn_refresh_outputs.setEnabled(False)
            self.refresh_worker = OutputRefreshWorker(problem)
            self.refresh_worker.done_sig.connect(self.on_outputs_refreshed)
            self.refresh_worker.error_sig.connect(self.on_refresh_error)
            self.refresh_worker.start()

    def on_outputs_refreshed(self, output_names):
        for name in output_names:
            self.combo_outputs.addItem(name, name)
        self.btn_refresh_outputs.setEnabled(True)
        self._refresh_steps()

    def on_refresh_error(self, error_msg):
        logger.error("Error refreshing outputs: %s", error_msg)
        self.btn_refresh_outputs.setEnabled(True)

    # ====================================================================
    # Run Analysis
    # ====================================================================

    def run_analysis(self):
        if not self.salib_available:
            QtWidgets.QMessageBox.warning(self, "Error", "SALib is not installed.")
            return
        if not self.optimization_widget or not self.optimization_widget.problem:
            QtWidgets.QMessageBox.warning(
                self, "Error", "No optimization problem loaded."
            )
            return

        problem = self.optimization_widget.problem

        output_idx = self.combo_outputs.currentIndex()
        if output_idx < 0 and not self.chk_batch.isChecked():
            QtWidgets.QMessageBox.warning(self, "Error", "No output variable selected.")
            return

        output_name = (
            self.combo_outputs.itemData(output_idx) if output_idx >= 0 else None
        )
        n_samples = self.spin_samples.value()
        calc_second_order = self.chk_second_order.isChecked()
        random_seed = self.spin_seed.value()

        # Power-of-2 adjustment for Sobol (Saltelli sequence)
        adjusted = self._adjust_to_power_of_two(n_samples)
        if adjusted != n_samples:
            self.spin_samples.setValue(adjusted)
            n_samples = adjusted

        batch = self.chk_batch.isChecked()

        self._set_running(True)
        self.btn_export.setEnabled(False)
        self.progress.setValue(0)
        self.batch_results = {}  # cleared until this run completes

        self.worker = SensitivityWorker(
            problem,
            output_name,
            n_samples,
            method="Sobol",
            calc_second_order=calc_second_order,
            random_seed=random_seed,
            batch=batch,
            fixed_values=self._fixed_values(),
        )
        self.worker.progress_sig.connect(self.update_progress)
        self.worker.done_sig.connect(self.analysis_finished)
        self.worker.start()

    def update_progress(self, value, message):
        self.progress.setValue(value)
        self.lbl_status.setText(message)

    def analysis_finished(self, results, error):
        self._set_running(False)

        if error:
            if "cancelled" in str(error).casefold():
                self.lbl_status.setText("Sensitivity analysis cancelled.")
            else:
                QtWidgets.QMessageBox.critical(self, "Analysis Failed", error)
                self.lbl_status.setText("Error occurred.")
            return

        if isinstance(results, dict) and "batch_results" in results:
            # Batch mode: keep every output's result and let the user browse them
            # via the output dropdown (they were all computed from one sample set).
            self.batch_results = results["batch_results"]
            successful_outputs = [
                name
                for name, result in self.batch_results.items()
                if not (isinstance(result, dict) and "error" in result)
            ]
            if not successful_outputs:
                details = "\n".join(
                    f"{name}: {result.get('error', 'unknown error')}"
                    for name, result in self.batch_results.items()
                )
                QtWidgets.QMessageBox.critical(
                    self,
                    "Batch Analysis Failed",
                    details or "No output was available for analysis.",
                )
                self.lbl_status.setText("All batch analyses failed.")
                return

            self.combo_outputs.blockSignals(True)
            self.combo_outputs.clear()
            for name in self.batch_results:
                self.combo_outputs.addItem(name, name)
            first_key = successful_outputs[0]
            first_index = list(self.batch_results).index(first_key)
            self.combo_outputs.setCurrentIndex(first_index)
            self.combo_outputs.blockSignals(False)

            self.current_results = self.batch_results[first_key]
            failure_count = len(self.batch_results) - len(successful_outputs)
            status = (
                f"Batch analysis complete for {len(successful_outputs)} outputs"
                + (f" ({failure_count} failed)" if failure_count else "")
                + "."
            )
            display_name = first_key
        else:
            self.batch_results = {}
            self.current_results = results
            status = ""
            display_name = str(
                self.combo_outputs.currentData() or self.combo_outputs.currentText()
            )

        self._update_heatmap_visibility()
        self.btn_export.setEnabled(True)
        self._refresh_views()
        self.lbl_status.setText(status if self.batch_results else self._summary_html(self.current_results, display_name))

    def cancel_analysis(self):
        """Request cancellation between model evaluations."""
        worker = getattr(self, "worker", None)
        if worker is not None and worker.isRunning():
            worker.requestInterruption()
            self.btn_cancel.setEnabled(False)
            self.lbl_status.setText("Cancelling after the current model evaluation…")

    def _refresh_views(self):
        """Redraw all visualisations from self.current_results."""
        self.update_plot()
        self.update_table()
        self.update_heatmap()

    def _on_output_selected(self, idx):
        """In batch mode, switch the displayed result to the chosen output."""
        if not self.batch_results or idx < 0:
            return
        name = self.combo_outputs.itemData(idx) or self.combo_outputs.currentText()
        res = self.batch_results.get(name)
        if res is None:
            return
        if isinstance(res, dict) and "error" in res:
            logger.error("Sensitivity analysis failed for %s: %s", name, res["error"])
            self.lbl_status.setText(f"Failed · {html.escape(str(name))}")
            self.current_results = None
            self.results_table.setRowCount(0)
            self.plot_widget.clear()
            self._legend.clear()
            self.heatmap_widget.clear()
            self._update_heatmap_visibility()
            return
        self.current_results = res
        self._update_heatmap_visibility()
        self._refresh_views()
        self.lbl_status.setText(self._summary_html(res, name))

    # ====================================================================
    # Visualizations
    # ====================================================================

    def _display_order(self):
        """Variable order for charts: largest total effect first."""
        total = np.asarray(self.current_results.get("total_order", []), dtype=float)
        return np.argsort(-np.nan_to_num(total, nan=-np.inf))

    def update_plot(self):
        """Update the bar chart for Sobol sensitivity indices."""
        if not self.current_results:
            return

        self.plot_widget.clear()
        self._legend.clear()
        variables = list(self.current_results.get("variable_names", []))
        n_vars = len(variables)
        if n_vars == 0:
            return
        order = self._display_order()
        names = [variables[i] for i in order]
        x = np.arange(n_vars)

        def ranked(key):
            values = self.current_results.get(key, None)
            return None if values is None else np.asarray(values, dtype=float)[order]

        first_indices = ranked("first_order")
        total_indices = ranked("total_order")
        width = 0.35

        first_bars = pg.BarGraphItem(
            x=x - width / 2,
            height=first_indices,
            width=width,
            brush=pg.mkBrush("#87CEEB"),
            pen=pg.mkPen(COLORS["chart_fg"], width=1),
        )
        total_bars = pg.BarGraphItem(
            x=x + width / 2,
            height=total_indices,
            width=width,
            brush=pg.mkBrush("#F08080"),
            pen=pg.mkPen(COLORS["chart_fg"], width=1),
        )
        self.plot_widget.addItem(first_bars)
        self.plot_widget.addItem(total_bars)

        for offset, indices, key in (
            (-width / 2, first_indices, "confidence_first"),
            (width / 2, total_indices, "confidence_total"),
        ):
            conf = ranked(key)
            if conf is None:
                continue
            self.plot_widget.addItem(
                pg.ErrorBarItem(
                    x=x + offset,
                    y=indices,
                    top=conf,
                    bottom=conf,
                    beam=width * 0.5,
                    pen=pg.mkPen(COLORS["chart_fg"], width=1),
                )
            )

        self._legend.addItem(first_bars, "First order (S1)")
        self._legend.addItem(total_bars, "Total order (ST)")
        self.plot_widget.setTitle("Sensitivity Indices")

        ax = self.plot_widget.getAxis("bottom")
        ax.setTicks([[(i, name) for i, name in enumerate(names)]])

    def update_heatmap(self):
        """Update S2 interaction heatmap."""
        self.heatmap_widget.clear()
        if not self.current_results or not self.current_results.get("calc_second_order", False):
            return

        s2_matrix = self.current_results.get("s2_matrix", None)
        if s2_matrix is None:
            return

        s2 = np.array(s2_matrix, dtype=float)
        s2 = np.where(np.isnan(s2), 0, s2)
        variables = list(self.current_results.get("variable_names", []))
        order = self._display_order()  # same ranking as the bar chart
        if len(order) == s2.shape[0]:
            s2 = s2[np.ix_(order, order)]
            variables = [variables[i] for i in order]

        img = pg.ImageItem(s2.T)
        cmap = pg.colormap.get("CET-D1")
        if cmap is not None:
            img.setLookupTable(cmap.getLookupTable(nPts=256))
        self.heatmap_widget.addItem(img)

        ax_bottom = self.heatmap_widget.getAxis("bottom")
        ax_left = self.heatmap_widget.getAxis("left")
        ticks = [(i + 0.5, v) for i, v in enumerate(variables)]
        ax_bottom.setTicks([ticks])
        ax_left.setTicks([ticks])
        self.heatmap_widget.setTitle("Pairwise Interactions (S2)")

    def update_table(self):
        """Show numeric Sobol indices sorted by total-order index (ST)."""
        if not self.current_results:
            return

        variables = list(self.current_results.get("variable_names", []))
        n = len(variables)
        if n == 0:
            self.results_table.setRowCount(0)
            return

        s1 = np.asarray(self.current_results.get("first_order", np.zeros(n)), float)
        st = np.asarray(self.current_results.get("total_order", np.zeros(n)), float)
        conf = np.asarray(
            self.current_results.get("confidence_total", np.zeros(n)), float
        )
        interaction = np.clip(st - s1, 0.0, None)
        order = np.argsort(-st)

        headers = [
            "Rank",
            "Variable",
            "First Order (S1)",
            "Total Order (ST)",
            "Interaction (ST - S1)",
            "ST Confidence",
        ]
        rows = [
            (
                str(rank),
                variables[i],
                f"{s1[i]:.4f}",
                f"{st[i]:.4f}",
                f"{interaction[i]:.4f}",
                f"{float(conf[i]):.4f}",
            )
            for rank, i in enumerate(order, 1)
        ]

        self.results_table.setColumnCount(len(headers))
        self.results_table.setHorizontalHeaderLabels(headers)
        self.results_table.setRowCount(len(rows))
        for r, row in enumerate(rows):
            for c, val in enumerate(row):
                item = QtWidgets.QTableWidgetItem(val)
                item.setTextAlignment(QtCore.Qt.AlignCenter)
                self.results_table.setItem(r, c, item)
        header = self.results_table.horizontalHeader()
        header.setMinimumSectionSize(65)
        header.setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        header.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeToContents)

    # ====================================================================
    # Export
    # ====================================================================

    def export_results(self):
        if not self.current_results and not self.batch_results:
            return

        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export Sensitivity Results",
            "",
            "CSV files (*.csv);;JSON files (*.json);;All files (*)",
        )
        if not filename:
            return

        try:
            if filename.endswith(".json"):
                self._export_json(filename)
            else:
                self._export_csv(filename)
            QtWidgets.QMessageBox.information(
                self, "Success", f"Results exported to {filename}"
            )
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Export Error", str(e))

    def _export_csv(self, filename):
        import csv

        results = self.current_results
        if not results:
            return

        with open(filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "Variable",
                    "First_Order_S1",
                    "Total_Order_ST",
                    "Coupling_ST_minus_S1",
                    "Confidence_Total",
                ]
            )
            variables = results.get("variable_names", [])
            s1_list = results.get("first_order", [])
            st_list = results.get("total_order", [])
            conf_list = results.get("confidence_total", [])
            for i, var in enumerate(variables):
                s1 = float(s1_list[i]) if i < len(s1_list) else 0.0
                st = float(st_list[i]) if i < len(st_list) else 0.0
                conf = float(conf_list[i]) if i < len(conf_list) else 0.0
                coupling = max(0.0, st - s1)
                writer.writerow(
                    [
                        var,
                        f"{s1:.6f}",
                        f"{st:.6f}",
                        f"{coupling:.6f}",
                        f"{conf:.6f}",
                    ]
                )

    def _export_json(self, filename):
        import json

        def _convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.integer):
                return int(obj)
            return obj

        data = {}
        if self.batch_results:
            for key, res in self.batch_results.items():
                data[key] = {k: _convert(v) for k, v in res.items()}
        elif self.current_results:
            data = {k: _convert(v) for k, v in self.current_results.items()}

        with open(filename, "w") as f:
            json.dump(data, f, indent=2, default=_convert)

    # ====================================================================
    # Helpers
    # ====================================================================

    def _adjust_to_power_of_two(self, n: int) -> int:
        if n <= 0:
            return 2
        import math

        power = math.floor(math.log2(n))
        lower = 2**power
        upper = 2 ** (power + 1)
        return lower if abs(n - lower) <= abs(n - upper) else upper

    # ====================================================================
    # Save / Load
    # ====================================================================

    def save_to_folder(self, folder_path):
        import os
        from pylcss.io_manager.project_io import atomic_json_dump

        data = {
            "method": "Sobol",
            "output_name": self.combo_outputs.currentText(),
            "n_samples": self.spin_samples.value(),
            "n_trajectories": self.spin_trajectories.value(),
            "morris_levels": self.spin_morris_levels.value(),
            "calc_second_order": self.chk_second_order.isChecked(),
            "random_seed": self.spin_seed.value(),
            "batch": self.chk_batch.isChecked(),
        }
        atomic_json_dump(data, os.path.join(folder_path, "sensitivity.json"))

    def load_from_folder(self, folder_path):
        import os
        from pylcss.io_manager.project_io import load_json_object

        path = os.path.join(folder_path, "sensitivity.json")
        if not os.path.exists(path):
            return
        data = load_json_object(path)
        self.combo_method.setCurrentText(data.get("method", "Sobol"))
        self.combo_outputs.setCurrentText(data.get("output_name", ""))
        self.spin_samples.setValue(data.get("n_samples", 1024))
        self.spin_trajectories.setValue(data.get("n_trajectories", 20))
        self.spin_morris_levels.setValue(data.get("morris_levels", 4))
        self.chk_second_order.setChecked(data.get("calc_second_order", False))
        self.spin_seed.setValue(data.get("random_seed", 42))
        self.chk_batch.setChecked(data.get("batch", False))
