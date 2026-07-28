# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE file for details.

"""
Sensitivity Analysis Interface for PyLCSS.

Provides a GUI for performing global sensitivity analysis using multiple methods:
- Sobol indices (variance-based, first/total/second order)
- Morris screening (elementary effects)
- FAST (Fourier Amplitude Sensitivity Test)
- Delta Moment-Independent Measure

Features: method selection, S2 interaction heatmap, multi-output batch,
and an importance-ranked data table.
"""

import numpy as np
from PySide6 import QtWidgets
import pyqtgraph as pg
import qtawesome as qta
import logging
from ...sensitivity import SensitivityAnalyzer
from ..common.theme_manager import COLORS
from .workers import OutputRefreshWorker, SensitivityWorker

logger = logging.getLogger(__name__)


class SensitivityAnalysisWidget(QtWidgets.QWidget):
    """
    Widget for performing global sensitivity analysis.

    Supports Sobol, Morris, FAST, and Delta methods with interactive
    visualizations including bar charts, interaction heatmaps, and an
    importance-ranked data table.
    """

    METHOD_INFO = {
        "Sobol": "Variance-based. Estimates each variable's first-order (S1), "
        "total (ST), and optionally pairwise (S2) contributions for "
        "the stated independent input distributions.",
        "Morris": "Elementary-effects screening. Cheaply ranks variables and "
        "shows nonlinearity/interaction indications. Use it to form "
        "screening hypotheses, then verify important decisions.",
        "FAST": "Extended Fourier Amplitude Sensitivity Test (eFAST). Estimates "
        "first-order (S1) and total-order (ST) effects, but not pairwise "
        "S2 interactions.",
        "Delta": "Moment-independent. Captures effects on the whole output "
        "distribution, not just its variance. Good for skewed outputs.",
    }

    def apply_theme(self, theme_name):
        palette = COLORS
        icon_color = palette["text_main"]
        for index, icon_name in enumerate(
            ("fa5s.chart-bar", "fa5s.th", "fa5s.braille", "fa5s.table")
        ):
            self.tab_viz.setTabIcon(index, qta.icon(icon_name, color=icon_color))
        for plot in (self.plot_widget, self.heatmap_widget, self.morris_widget):
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

        # --- LEFT PANEL: Configuration ---
        config_panel = QtWidgets.QWidget()
        config_panel.setFixedWidth(400)
        config_layout = QtWidgets.QVBoxLayout(config_panel)
        config_layout.setContentsMargins(0, 0, 0, 0)

        # 1. Method Selection
        grp_method = QtWidgets.QGroupBox("Analysis Method")
        l_method = QtWidgets.QFormLayout(grp_method)

        self.combo_method = QtWidgets.QComboBox()
        available = (
            SensitivityAnalyzer.available_methods()
            if self.salib_available
            else ["Sobol"]
        )
        self.combo_method.addItems(available)
        self.combo_method.setToolTip(
            "Sobol: Full variance decomposition (S1, ST, S2). With pairwise "
            "indices enabled, the run count is N x (2D+2), with power-of-two N.\n"
            "Morris: Fast screening to identify unimportant variables.\n"
            "FAST: Efficient first- and total-order estimation via Fourier analysis.\n"
            "Delta: Distribution-based, moment-independent measure."
        )
        self.combo_method.currentTextChanged.connect(self._on_method_changed)
        l_method.addRow("Method:", self.combo_method)

        self.lbl_method_desc = QtWidgets.QLabel()
        self.lbl_method_desc.setWordWrap(True)
        self.lbl_method_desc.setStyleSheet(
            f"color:{COLORS['text_dim']}; font-size:9pt; padding:4px 2px;"
        )
        l_method.addRow("", self.lbl_method_desc)

        config_layout.addWidget(grp_method)

        # 2. Select Outputs
        grp_output = QtWidgets.QGroupBox("Output Variables")
        l_output = QtWidgets.QVBoxLayout(grp_output)
        self.combo_outputs = QtWidgets.QComboBox()
        self.combo_outputs.setToolTip(
            "Select the output variable to analyze. After a "
            "batch run, switch outputs here to view each result."
        )
        self.combo_outputs.currentIndexChanged.connect(self._on_output_selected)
        self.btn_refresh_outputs = QtWidgets.QPushButton(
            qta.icon("fa5s.sync"), " Refresh Outputs"
        )
        self.btn_refresh_outputs.clicked.connect(self.refresh_outputs)
        l_output.addWidget(self.combo_outputs)
        l_output.addWidget(self.btn_refresh_outputs)

        self.chk_batch = QtWidgets.QCheckBox("Analyze ALL outputs (batch)")
        self.chk_batch.setToolTip(
            "Run sensitivity analysis for every output variable at once."
        )
        l_output.addWidget(self.chk_batch)

        config_layout.addWidget(grp_output)

        # 3. Analysis Configuration
        grp_config = QtWidgets.QGroupBox("Configuration")
        l_config = QtWidgets.QFormLayout(grp_config)

        self.spin_samples = QtWidgets.QSpinBox()
        self.spin_samples.setRange(64, 16384)
        self.spin_samples.setValue(1024)
        self.spin_samples.setSingleStep(256)
        self.spin_samples.setToolTip(
            "Base sample size N. The actual number of model runs depends on the "
            "method and the number of variables (see the estimate below). For "
            "Sobol, N is rounded to a power of 2."
        )
        l_config.addRow("Sample Size:", self.spin_samples)
        self.lbl_samples = l_config.labelForField(self.spin_samples)

        self.spin_trajectories = QtWidgets.QSpinBox()
        self.spin_trajectories.setRange(4, 200)
        self.spin_trajectories.setValue(20)
        self.spin_trajectories.setToolTip("Number of Morris trajectories.")
        l_config.addRow("Trajectories:", self.spin_trajectories)
        self.lbl_trajectories = l_config.labelForField(self.spin_trajectories)

        self.spin_morris_levels = QtWidgets.QSpinBox()
        self.spin_morris_levels.setRange(4, 20)
        self.spin_morris_levels.setSingleStep(2)
        self.spin_morris_levels.setValue(4)
        self.spin_morris_levels.setToolTip(
            "Even number of grid levels used by Morris trajectories. "
            "Four is the standard screening default."
        )
        l_config.addRow("Grid Levels:", self.spin_morris_levels)
        self.lbl_morris_levels = l_config.labelForField(self.spin_morris_levels)

        self.chk_second_order = QtWidgets.QCheckBox(
            "Calculate pairwise interactions (S2)"
        )
        self.chk_second_order.setChecked(False)
        self.chk_second_order.setToolTip(
            "Sobol only. Disable when pairwise interactions are not needed "
            "to reduce model runs from N×(2D+2) to N×(D+2)."
        )
        l_config.addRow("", self.chk_second_order)

        self.spin_seed = QtWidgets.QSpinBox()
        self.spin_seed.setRange(0, 2_147_483_647)
        self.spin_seed.setValue(42)
        self.spin_seed.setToolTip(
            "Random seed for reproducible sampling and confidence intervals."
        )
        l_config.addRow("Random Seed:", self.spin_seed)

        self.lbl_eval_estimate = QtWidgets.QLabel()
        self.lbl_eval_estimate.setStyleSheet(
            f"font-size:10px; color:{COLORS['text_dim']}; font-style:italic;"
        )
        self.lbl_eval_estimate.setWordWrap(True)
        l_config.addRow("", self.lbl_eval_estimate)

        # Keep the evaluation estimate live as the user tweaks settings.
        self.spin_samples.valueChanged.connect(lambda *_: self._update_estimate())
        self.spin_trajectories.valueChanged.connect(lambda *_: self._update_estimate())
        self.spin_morris_levels.valueChanged.connect(lambda *_: self._update_estimate())
        self.chk_second_order.toggled.connect(lambda *_: self._update_estimate())

        config_layout.addWidget(grp_config)

        # 4. Action Buttons
        self.btn_analyze = QtWidgets.QPushButton(
            qta.icon("fa5s.chart-bar"), " Run Sensitivity Analysis"
        )
        self.btn_analyze.setStyleSheet("font-weight: bold; padding: 8px;")
        self.btn_analyze.clicked.connect(self.run_analysis)
        config_layout.addWidget(self.btn_analyze)

        self.btn_cancel = QtWidgets.QPushButton("Cancel Analysis")
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(self.cancel_analysis)
        config_layout.addWidget(self.btn_cancel)

        self.btn_export = QtWidgets.QPushButton(
            qta.icon("fa5s.download"), " Export Results"
        )
        self.btn_export.setEnabled(False)
        self.btn_export.clicked.connect(self.export_results)
        config_layout.addWidget(self.btn_export)

        config_layout.addStretch()

        layout.addWidget(config_panel)

        # --- RIGHT PANEL: Visualization (tabbed) ---
        viz_panel = QtWidgets.QWidget()
        viz_layout = QtWidgets.QVBoxLayout(viz_panel)

        # Status
        self.lbl_status = QtWidgets.QLabel("Status: Ready to analyze.")
        self.lbl_status.setStyleSheet(
            "background-color: #333; color: #fff; padding: 10px; border-radius: 4px;"
        )
        viz_layout.addWidget(self.lbl_status)

        # Tab widget for multiple viz types
        self.tab_viz = QtWidgets.QTabWidget()

        # Tab 1: Bar Chart
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground(COLORS["chart_bg"])
        self.plot_widget.setLabel("left", "Sensitivity Index")
        self.plot_widget.setLabel("bottom", "Design Variables")
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.tab_viz.addTab(
            self.plot_widget,
            qta.icon("fa5s.chart-bar", color=COLORS["text_main"]),
            "Bar Chart",
        )

        # Tab 2: S2 Interaction Heatmap
        self.heatmap_widget = pg.PlotWidget()
        self.heatmap_widget.setBackground(COLORS["chart_bg"])
        self.heatmap_widget.setLabel("left", "Variable")
        self.heatmap_widget.setLabel("bottom", "Variable")
        self.tab_viz.addTab(
            self.heatmap_widget,
            qta.icon("fa5s.th", color=COLORS["text_main"]),
            "S2 Interactions",
        )

        # Tab 3: Morris Scatter (mu* vs sigma)
        self.morris_widget = pg.PlotWidget()
        self.morris_widget.setBackground(COLORS["chart_bg"])
        self.morris_widget.setLabel("left", "σ (Standard Deviation)")
        self.morris_widget.setLabel("bottom", "μ* (Absolute Mean)")
        self.morris_widget.showGrid(x=True, y=True, alpha=0.3)
        self.tab_viz.addTab(
            self.morris_widget,
            qta.icon("fa5s.braille", color=COLORS["text_main"]),
            "Morris Scatter",
        )

        # Tab 4: Results Table
        self.results_table = QtWidgets.QTableWidget()
        self.results_table.setColumnCount(4)
        self.results_table.setHorizontalHeaderLabels(
            ["Variable", "First Order", "Total Order", "Confidence"]
        )
        self.results_table.horizontalHeader().setStretchLastSection(True)
        self.tab_viz.addTab(
            self.results_table,
            qta.icon("fa5s.table", color=COLORS["text_main"]),
            "Data Table",
        )

        viz_layout.addWidget(self.tab_viz)

        # Progress
        self.progress = QtWidgets.QProgressBar()
        viz_layout.addWidget(self.progress)

        layout.addWidget(viz_panel)

        # Initialize visibility
        self._on_method_changed(self.combo_method.currentText())

    def _on_method_changed(self, method):
        """Show/hide controls based on selected method."""
        is_morris = method == "Morris"
        self.spin_trajectories.setVisible(is_morris)
        self.lbl_trajectories.setVisible(is_morris)
        # Sample size drives Sobol/FAST/Delta; Morris is driven by trajectories.
        self.spin_samples.setVisible(not is_morris)
        self.lbl_samples.setVisible(not is_morris)
        self.spin_morris_levels.setVisible(is_morris)
        self.lbl_morris_levels.setVisible(is_morris)
        self.chk_second_order.setVisible(method == "Sobol")
        self.spin_samples.setMinimum(65 if method == "FAST" else 64)

        self.lbl_method_desc.setText(self.METHOD_INFO.get(method, ""))
        self._update_estimate()

        # Only show the plot tabs that apply to the selected method (the others
        # would just render blank): S2 heatmap is Sobol-only, scatter is Morris-only.
        self.tab_viz.setTabVisible(1, method == "Sobol")  # S2 Interactions
        self.tab_viz.setTabVisible(2, is_morris)  # Morris Scatter

        # Auto-switch to the relevant visualization tab
        if is_morris:
            self.tab_viz.setCurrentIndex(2)  # Morris scatter
        else:
            self.tab_viz.setCurrentIndex(0)  # Bar chart

    def _update_estimate(self):
        """Show the estimated number of model evaluations for the current setup."""
        method = self.combo_method.currentText()
        ow = self.optimization_widget
        n_vars = len(ow.problem.design_variables) if (ow and ow.problem) else 0
        if n_vars == 0:
            self.lbl_eval_estimate.setText("Load a problem to estimate model runs.")
            return

        N = self.spin_samples.value()
        T = self.spin_trajectories.value()
        if method == "Sobol":
            multiplier = (
                2 * n_vars + 2 if self.chk_second_order.isChecked() else n_vars + 2
            )
            evals, detail = N * multiplier, f"N={N} × {multiplier}"
        elif method == "FAST":
            evals, detail = N * n_vars, f"N={N} x {n_vars} vars"
        elif method == "Delta":
            evals, detail = N, f"N={N}"
        elif method == "Morris":
            evals, detail = (n_vars + 1) * T, f"({n_vars}+1) x {T} trajectories"
        else:
            evals, detail = N, f"N={N}"
        self.lbl_eval_estimate.setText(f"~ {evals:,} model evaluations  ({detail})")

    # ====================================================================
    # Refresh Outputs
    # ====================================================================

    def refresh_outputs(self):
        self.combo_outputs.clear()
        if not self.optimization_widget or not self.optimization_widget.problem:
            return

        problem = self.optimization_widget.problem
        for qoi in problem.quantities_of_interest:
            self.combo_outputs.addItem(f"{qoi['name']} ({qoi['unit']})", qoi["name"])

        # Variable count is now known — refresh the evaluation estimate.
        self._update_estimate()

        if self.combo_outputs.count() == 0 and problem.system_model:
            self.btn_refresh_outputs.setEnabled(False)
            self.btn_refresh_outputs.setText("Refreshing...")
            self.refresh_worker = OutputRefreshWorker(problem)
            self.refresh_worker.done_sig.connect(self.on_outputs_refreshed)
            self.refresh_worker.error_sig.connect(self.on_refresh_error)
            self.refresh_worker.start()

    def on_outputs_refreshed(self, output_names):
        for name in output_names:
            self.combo_outputs.addItem(name, name)
        self.btn_refresh_outputs.setEnabled(True)
        self.btn_refresh_outputs.setText(" Refresh Outputs")

    def on_refresh_error(self, error_msg):
        logger.error("Error refreshing outputs: %s", error_msg)
        self.btn_refresh_outputs.setEnabled(True)
        self.btn_refresh_outputs.setText(" Refresh Outputs")

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
        method = self.combo_method.currentText()

        output_idx = self.combo_outputs.currentIndex()
        if output_idx < 0 and not self.chk_batch.isChecked():
            QtWidgets.QMessageBox.warning(self, "Error", "No output variable selected.")
            return

        output_name = (
            self.combo_outputs.itemData(output_idx) if output_idx >= 0 else None
        )
        n_samples = self.spin_samples.value()
        n_trajectories = self.spin_trajectories.value()
        morris_levels = self.spin_morris_levels.value()
        calc_second_order = self.chk_second_order.isChecked()
        random_seed = self.spin_seed.value()

        # Power-of-2 adjustment for Sobol only (Saltelli sequence). FAST has no
        # such requirement, and snapping it down can break its minimum count.
        if method == "Sobol":
            adjusted = self._adjust_to_power_of_two(n_samples)
            if adjusted != n_samples:
                self.spin_samples.setValue(adjusted)
                n_samples = adjusted

        batch = self.chk_batch.isChecked()

        self.btn_analyze.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self.btn_export.setEnabled(False)
        self.progress.setValue(0)
        self.batch_results = {}  # cleared until this run completes

        self.worker = SensitivityWorker(
            problem,
            output_name,
            n_samples,
            method,
            n_trajectories=n_trajectories,
            morris_levels=morris_levels,
            calc_second_order=calc_second_order,
            random_seed=random_seed,
            batch=batch,
        )
        self.worker.progress_sig.connect(self.update_progress)
        self.worker.done_sig.connect(self.analysis_finished)
        self.worker.start()

    def update_progress(self, value, message):
        self.progress.setValue(value)
        self.lbl_status.setText(message)

    def analysis_finished(self, results, error):
        self.btn_analyze.setEnabled(True)
        self.btn_cancel.setEnabled(False)

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
                + (f"; {failure_count} failed" if failure_count else "")
                + " — pick an output above to view it."
            )
        else:
            self.batch_results = {}
            self.current_results = results
            status = "Analysis complete."

        self.lbl_status.setText(status)

        self.btn_export.setEnabled(True)
        self._refresh_views()

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
        self.update_morris_scatter()

    def _on_output_selected(self, idx):
        """In batch mode, switch the displayed result to the chosen output."""
        if not self.batch_results or idx < 0:
            return
        name = self.combo_outputs.itemData(idx) or self.combo_outputs.currentText()
        res = self.batch_results.get(name)
        if res is None:
            return
        if isinstance(res, dict) and "error" in res:
            self.lbl_status.setText(f"{name}: analysis failed — {res['error']}")
            return
        self.current_results = res
        self.lbl_status.setText(f"Showing sensitivity for output: {name}")
        self._refresh_views()

    # ====================================================================
    # Visualizations
    # ====================================================================

    def update_plot(self):
        """Update the bar chart for Sobol/FAST/Delta results."""
        if not self.current_results:
            return

        self.plot_widget.clear()
        method = self.current_results.get("method", "Sobol")
        variables = self.current_results.get("variable_names", [])
        n_vars = len(variables)
        x = np.arange(n_vars)

        if method in ("Sobol", "FAST"):
            first_indices = np.array(self.current_results.get("first_order", []))
            total_indices = np.array(self.current_results.get("total_order", []))
            confidence = self.current_results.get("confidence_total", None)
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

            if confidence is not None:
                conf_arr = np.array(confidence)
                err = pg.ErrorBarItem(
                    x=x + width / 2,
                    y=total_indices,
                    top=conf_arr,
                    bottom=conf_arr,
                    beam=width * 0.5,
                    pen=pg.mkPen(COLORS["chart_fg"], width=1),
                )
                self.plot_widget.addItem(err)

            legend = self.plot_widget.addLegend(offset=(10, 10))
            legend.addItem(first_bars, "First Order (S1)")
            legend.addItem(total_bars, "Total Order (ST)")
            self.plot_widget.setTitle(f"{method} Sensitivity Indices")

        elif method == "Morris":
            mu_star = np.array(self.current_results.get("mu_star", []))
            bars = pg.BarGraphItem(
                x=x,
                height=mu_star,
                width=0.6,
                brush=pg.mkBrush("#9B59B6"),
                pen=pg.mkPen(COLORS["chart_fg"], width=1),
            )
            self.plot_widget.addItem(bars)
            self.plot_widget.setTitle("Morris Screening: μ*")
            self.plot_widget.setLabel("left", "μ* (Absolute Mean)")

        elif method == "Delta":
            delta = np.array(self.current_results.get("delta", []))
            s1 = np.array(self.current_results.get("S1", []))
            width = 0.35
            d_bars = pg.BarGraphItem(
                x=x - width / 2,
                height=delta,
                width=width,
                brush=pg.mkBrush("#27AE60"),
                pen=pg.mkPen(COLORS["chart_fg"], width=1),
            )
            s_bars = pg.BarGraphItem(
                x=x + width / 2,
                height=s1,
                width=width,
                brush=pg.mkBrush("#87CEEB"),
                pen=pg.mkPen(COLORS["chart_fg"], width=1),
            )
            self.plot_widget.addItem(d_bars)
            self.plot_widget.addItem(s_bars)
            legend = self.plot_widget.addLegend(offset=(10, 10))
            legend.addItem(d_bars, "Delta")
            legend.addItem(s_bars, "S1")
            self.plot_widget.setTitle("Delta Moment-Independent Measure")

        ax = self.plot_widget.getAxis("bottom")
        ticks = [(i, var) for i, var in enumerate(variables)]
        ax.setTicks([ticks])

    def update_heatmap(self):
        """Update S2 interaction heatmap (Sobol only)."""
        self.heatmap_widget.clear()
        if not self.current_results or self.current_results.get("method") != "Sobol":
            return

        s2_matrix = self.current_results.get("s2_matrix", None)
        if s2_matrix is None:
            return

        s2 = np.array(s2_matrix, dtype=float)
        s2 = np.where(np.isnan(s2), 0, s2)
        variables = self.current_results["variable_names"]

        img = pg.ImageItem(s2.T)
        # Use a blue-red colormap
        cmap = pg.colormap.get("CET-D1")
        img.setLookupTable(cmap.getLookupTable(nPts=256))
        self.heatmap_widget.addItem(img)

        ax_bottom = self.heatmap_widget.getAxis("bottom")
        ax_left = self.heatmap_widget.getAxis("left")
        ticks = [(i + 0.5, v) for i, v in enumerate(variables)]
        ax_bottom.setTicks([ticks])
        ax_left.setTicks([ticks])
        self.heatmap_widget.setTitle(
            "Pairwise Second-Order Indices (S2; diagonal not applicable)"
        )

    def update_morris_scatter(self):
        """Morris μ* vs σ scatter plot."""
        self.morris_widget.clear()
        if not self.current_results or self.current_results.get("method") != "Morris":
            return

        mu_star = np.array(self.current_results.get("mu_star", []))
        sigma = np.array(self.current_results.get("sigma", []))
        variables = self.current_results.get("variable_names", [])

        scatter = pg.ScatterPlotItem(
            x=mu_star,
            y=sigma,
            size=12,
            brush=pg.mkBrush("#E74C3C"),
            pen=pg.mkPen(COLORS["chart_fg"], width=1),
            symbol="o",
        )
        self.morris_widget.addItem(scatter)

        # Label each point
        for i, var in enumerate(variables):
            txt = pg.TextItem(var, anchor=(0, 1), color=COLORS["chart_fg"])
            txt.setPos(mu_star[i], sigma[i])
            self.morris_widget.addItem(txt)

        self.morris_widget.setTitle(
            "Morris Screening: μ* (overall effect) vs σ "
            "(non-linearity / interactions indicator)"
        )

    def update_table(self):
        """Show numeric indices sorted by the method's primary metric.

        Sensitivity is model- and decision-context dependent, so the table
        reports a rank instead of inventing universal engineering labels.
        """
        if not self.current_results:
            return

        method = self.current_results.get("method", "Sobol")
        variables = list(self.current_results.get("variable_names", []))
        n = len(variables)
        if n == 0:
            self.results_table.setRowCount(0)
            return

        if method == "Sobol":
            s1 = np.asarray(self.current_results.get("first_order", np.zeros(n)), float)
            st = np.asarray(self.current_results.get("total_order", np.zeros(n)), float)
            conf = np.asarray(
                self.current_results.get("confidence_total", np.zeros(n)), float
            )
            order = np.argsort(-st)
            headers = [
                "Rank",
                "Variable",
                "First Order (S1)",
                "Total Order (ST)",
                "ST Confidence",
            ]
            rows = [
                (
                    str(rank),
                    variables[i],
                    f"{s1[i]:.4f}",
                    f"{st[i]:.4f}",
                    f"{float(conf[i]):.4f}",
                )
                for rank, i in enumerate(order, 1)
            ]

        elif method == "FAST":
            s1 = np.asarray(self.current_results.get("first_order", np.zeros(n)), float)
            st = np.asarray(self.current_results.get("total_order", np.zeros(n)), float)
            conf = np.asarray(
                self.current_results.get("confidence_total", np.zeros(n)), float
            )
            order = np.argsort(-st)
            headers = [
                "Rank",
                "Variable",
                "First Order (S1)",
                "Total Order (ST)",
                "ST Confidence",
            ]
            rows = [
                (
                    str(rank),
                    variables[i],
                    f"{s1[i]:.4f}",
                    f"{st[i]:.4f}",
                    f"{conf[i]:.4f}",
                )
                for rank, i in enumerate(order, 1)
            ]

        elif method == "Morris":
            mu_star = np.asarray(
                self.current_results.get("mu_star", np.zeros(n)), float
            )
            sigma = np.asarray(self.current_results.get("sigma", np.zeros(n)), float)
            mu_conf = np.asarray(
                self.current_results.get("mu_star_conf", np.zeros(n)), float
            )
            order = np.argsort(-mu_star)
            headers = ["Rank", "Variable", "μ*", "σ", "μ* Confidence"]
            rows = [
                (
                    str(rank),
                    variables[i],
                    f"{mu_star[i]:.4f}",
                    f"{sigma[i]:.4f}",
                    f"{mu_conf[i]:.4f}",
                )
                for rank, i in enumerate(order, 1)
            ]

        elif method == "Delta":
            delta = np.asarray(self.current_results.get("delta", np.zeros(n)), float)
            s1 = np.asarray(self.current_results.get("S1", np.zeros(n)), float)
            dconf = np.asarray(
                self.current_results.get("delta_conf", np.zeros(n)), float
            )
            order = np.argsort(-delta)
            headers = ["Rank", "Variable", "Delta", "S1", "Delta Confidence"]
            rows = [
                (
                    str(rank),
                    variables[i],
                    f"{delta[i]:.4f}",
                    f"{s1[i]:.4f}",
                    f"{dconf[i]:.4f}",
                )
                for rank, i in enumerate(order, 1)
            ]
        else:
            self.results_table.setRowCount(0)
            return

        self.results_table.setColumnCount(len(headers))
        self.results_table.setHorizontalHeaderLabels(headers)
        self.results_table.setRowCount(len(rows))
        for r, row in enumerate(rows):
            for c, val in enumerate(row):
                item = QtWidgets.QTableWidgetItem(val)
                self.results_table.setItem(r, c, item)
        self.results_table.resizeColumnsToContents()

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

        method = results.get("method", "Sobol")
        with open(filename, "w", newline="") as f:
            writer = csv.writer(f)
            if method == "Sobol":
                writer.writerow(
                    ["Variable", "First_Order", "Total_Order", "Confidence_Total"]
                )
                for i, var in enumerate(results["variable_names"]):
                    s1 = float(results["first_order"][i])
                    st = float(results["total_order"][i])
                    conf = float(
                        results.get("confidence_total", np.zeros(1))[
                            min(i, len(results.get("confidence_total", [0])) - 1)
                        ]
                    )
                    writer.writerow([var, f"{s1:.6f}", f"{st:.6f}", f"{conf:.6f}"])
            elif method == "FAST":
                writer.writerow(
                    ["Variable", "First_Order", "Total_Order", "Confidence_Total"]
                )
                for i, var in enumerate(results["variable_names"]):
                    writer.writerow(
                        [
                            var,
                            f"{float(results['first_order'][i]):.6f}",
                            f"{float(results['total_order'][i]):.6f}",
                            f"{float(results['confidence_total'][i]):.6f}",
                        ]
                    )
            elif method == "Morris":
                writer.writerow(["Variable", "mu_star", "sigma", "mu"])
                for i, var in enumerate(results["variable_names"]):
                    writer.writerow(
                        [
                            var,
                            results["mu_star"][i],
                            results["sigma"][i],
                            results["mu"][i],
                        ]
                    )
            elif method == "Delta":
                writer.writerow(["Variable", "delta", "delta_conf", "S1", "S1_conf"])
                for i, var in enumerate(results["variable_names"]):
                    writer.writerow(
                        [
                            var,
                            results["delta"][i],
                            results["delta_conf"][i],
                            results["S1"][i],
                            results["S1_conf"][i],
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
            "method": self.combo_method.currentText(),
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
