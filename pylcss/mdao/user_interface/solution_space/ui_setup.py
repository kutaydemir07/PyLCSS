# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE.
"""SolutionUiMixin behavior for solution-space analysis."""

from __future__ import annotations

import logging

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets


logger = logging.getLogger(__name__)

__all__ = ["SolutionUiMixin"]


class SolutionUiMixin:
    def init_ui(self):
        # Main Layout: Splitter
        main_layout = QtWidgets.QHBoxLayout(self)
        self.splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        main_layout.addWidget(self.splitter)

        # --- Left Panel: Configuration ---
        self.config_panel = QtWidgets.QWidget()
        config_layout = QtWidgets.QVBoxLayout(self.config_panel)

        # Config Tabs
        self.config_tabs = QtWidgets.QTabWidget()
        config_layout.addWidget(self.config_tabs)

        # Tab 1: Model & Controls
        tab_model = QtWidgets.QWidget()
        model_layout = QtWidgets.QVBoxLayout(tab_model)

        # --- Top Section: Controls (Always Visible) ---

        # System Selection
        sys_group = QtWidgets.QGroupBox("System Model")
        sys_layout = QtWidgets.QHBoxLayout(sys_group)

        self.system_combo = QtWidgets.QComboBox()
        self.system_combo.setToolTip("Select the system model to analyze.")
        self.system_combo.currentIndexChanged.connect(self.on_system_changed)
        sys_layout.addWidget(self.system_combo, stretch=1)

        self.btn_view_code = QtWidgets.QPushButton("Code")
        self.btn_view_code.setFixedWidth(50)
        self.btn_view_code.clicked.connect(self.view_source_code)
        sys_layout.addWidget(self.btn_view_code)

        model_layout.addWidget(sys_group)

        # Actions & Settings
        actions_group = QtWidgets.QGroupBox("Analysis Controls")
        actions_layout = QtWidgets.QVBoxLayout(actions_group)

        # Row 1: Project Actions
        row1 = QtWidgets.QHBoxLayout()
        self.btn_save = QtWidgets.QPushButton("Save Project")
        self.btn_load = QtWidgets.QPushButton("Load Project")
        self.btn_save.clicked.connect(self.save_project)
        self.btn_load.clicked.connect(self.load_project)
        row1.addWidget(self.btn_save)
        row1.addWidget(self.btn_load)
        actions_layout.addLayout(row1)

        # Row 2: Solver & Samples
        row2 = QtWidgets.QHBoxLayout()
        self.solver_combo = QtWidgets.QComboBox()
        self.solver_combo.addItem("SLSQP", "goal_attainment")
        self.solver_combo.addItem("Nevergrad", "nevergrad")
        self.solver_combo.setToolTip("Choose optimization solver")
        row2.addWidget(self.solver_combo, stretch=1)

        self.sample_size_spin = QtWidgets.QSpinBox()
        self.sample_size_spin.setRange(10, 100000)
        self.sample_size_spin.setValue(300)
        self.sample_size_spin.setPrefix("N=")
        self.sample_size_spin.setToolTip(
            "Points to sample per Resample/Compute — one model evaluation each. "
            "Start small (e.g. 10–100) for a quick probe before a large run."
        )
        row2.addWidget(self.sample_size_spin)
        actions_layout.addLayout(row2)

        # Row 3: Compute Buttons
        row3 = QtWidgets.QHBoxLayout()
        self.btn_compute_feasible = QtWidgets.QPushButton("Compute Solution Space")
        self.btn_compute_feasible.clicked.connect(
            lambda: self.run_computation(
                include_objectives=self.chk_include_optimization.isChecked()
            )
        )
        self.btn_compute_feasible.setEnabled(False)
        row3.addWidget(self.btn_compute_feasible)

        self.chk_include_optimization = QtWidgets.QCheckBox("Opt.")
        self.chk_include_optimization.setToolTip("Include Optimization Objectives")
        self.chk_include_optimization.setEnabled(False)
        row3.addWidget(self.chk_include_optimization)
        actions_layout.addLayout(row3)

        # Row 4: Refinement
        row4 = QtWidgets.QHBoxLayout()
        self.btn_resample = QtWidgets.QPushButton("Resample")
        self.btn_resample.clicked.connect(self._resample_current_view)
        self.btn_resample.setEnabled(False)
        row4.addWidget(self.btn_resample)

        self.chk_center_slice = QtWidgets.QCheckBox("Center Slice")
        self.chk_center_slice.setToolTip(
            "Fix non-plotted design variables to the center of the current box "
            "when resampling 2D plots."
        )
        self.chk_center_slice.toggled.connect(lambda: self.trigger_debounced_resample())
        row4.addWidget(self.chk_center_slice)

        actions_layout.addLayout(row4)

        model_layout.addWidget(actions_group)

        # --- Bottom Section: Variables (Tabbed) ---
        self.vars_tabs = QtWidgets.QTabWidget()

        # Design Variables Table
        self.dv_table = QtWidgets.QTableWidget()
        self.dv_table.setColumnCount(6)
        self.dv_table.setHorizontalHeaderLabels(
            ["Name", "Unit", "Min (DS)", "Max (DS)", "Min (Sol)", "Max (Sol)"]
        )
        self.dv_table.itemChanged.connect(self.on_dv_table_changed)
        self.vars_tabs.addTab(self.dv_table, "Design Variables")

        # QoI Table
        self.qoi_table = QtWidgets.QTableWidget()
        self.qoi_table.setColumnCount(9)
        self.qoi_table.setHorizontalHeaderLabels(
            [
                "Name",
                "Unit",
                "Min (Req)",
                "Max (Req)",
                "Min (Plot)",
                "Max (Plot)",
                "Min",
                "Max",
                "W",
            ]
        )
        self.qoi_table.itemChanged.connect(self.on_qoi_table_changed)
        self.vars_tabs.addTab(self.qoi_table, "Quantities of Interest")

        model_layout.addWidget(self.vars_tabs)

        self.config_tabs.addTab(tab_model, "Model Control")

        # Tab 2: compact Multi-Modal controls.  The plots stay on the shared
        # Solution Spaces canvas and this tab only changes the box overlay/view.
        tab_multimodal = QtWidgets.QWidget()
        multimodal_layout = QtWidgets.QVBoxLayout(tab_multimodal)

        method_group = QtWidgets.QGroupBox("Multi-Modal Method")
        method_form = QtWidgets.QFormLayout(method_group)

        self.mm_solver_combo = QtWidgets.QComboBox()
        self.mm_solver_combo.addItem("SLSQP", "goal_attainment")
        self.mm_solver_combo.addItem("Nevergrad", "nevergrad")
        self.mm_solver_combo.setToolTip(
            "Choose the optimizer used by the deflation-based feasible-basin search."
        )
        method_form.addRow("Optimizer", self.mm_solver_combo)
        multimodal_layout.addWidget(method_group)

        self.btn_compute_multimodal = QtWidgets.QPushButton(
            "Compute Multi-Modal Spaces"
        )
        self.btn_compute_multimodal.setEnabled(False)
        self.btn_compute_multimodal.clicked.connect(self.run_multimodal_computation)
        multimodal_layout.addWidget(self.btn_compute_multimodal)

        view_group = QtWidgets.QGroupBox("Multi-Modal Plot View")
        view_layout = QtWidgets.QVBoxLayout(view_group)

        branch_row = QtWidgets.QHBoxLayout()
        branch_row.addWidget(QtWidgets.QLabel("Show"))
        self.combo_active_box = QtWidgets.QComboBox()
        self.combo_active_box.currentIndexChanged.connect(self.on_active_box_changed)
        branch_row.addWidget(self.combo_active_box, stretch=1)
        view_layout.addLayout(branch_row)

        sample_row = QtWidgets.QHBoxLayout()
        sample_row.addWidget(QtWidgets.QLabel("Plot sample size"))
        self.mm_plot_sample_size_spin = QtWidgets.QSpinBox()
        self.mm_plot_sample_size_spin.setRange(10, 100_000)
        self.mm_plot_sample_size_spin.setValue(300)
        self.mm_plot_sample_size_spin.setSingleStep(10)
        self.mm_plot_sample_size_spin.setPrefix("N=")
        sample_row.addWidget(self.mm_plot_sample_size_spin, stretch=1)
        self.btn_resample_multimodal = QtWidgets.QPushButton("Refresh Plots")
        self.btn_resample_multimodal.setEnabled(False)
        self.btn_resample_multimodal.clicked.connect(
            lambda: self.resample_multimodal(silent=False)
        )
        sample_row.addWidget(self.btn_resample_multimodal)
        view_layout.addLayout(sample_row)

        self.lbl_multimodal_info = QtWidgets.QLabel("No Multi-Modal result yet.")
        self.lbl_multimodal_info.setWordWrap(True)
        view_layout.addWidget(self.lbl_multimodal_info)

        self.multibox_table = QtWidgets.QTableWidget()
        self.multibox_table.setColumnCount(4)
        self.multibox_table.setHorizontalHeaderLabels(
            ["Box / Mode", "Volume", "a lower", "Samples"]
        )
        self.multibox_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.multibox_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.multibox_table.setMaximumHeight(180)
        self.multibox_table.cellClicked.connect(self.on_multibox_table_clicked)
        view_layout.addWidget(self.multibox_table)
        multimodal_layout.addWidget(view_group)
        multimodal_layout.addStretch()

        self.config_tabs.addTab(tab_multimodal, "Multi-Modal")

        # Tab 3: Product Family
        tab_family = QtWidgets.QWidget()
        family_layout = QtWidgets.QVBoxLayout(tab_family)

        # Variant Management
        variant_group = QtWidgets.QGroupBox("Product Variants")
        variant_layout = QtWidgets.QVBoxLayout(variant_group)

        self.variant_table = QtWidgets.QTableWidget()
        self.variant_table.setColumnCount(2)
        self.variant_table.setHorizontalHeaderLabels(["Variant Name", "Description"])
        variant_layout.addWidget(self.variant_table)

        btn_variant_layout = QtWidgets.QHBoxLayout()
        self.btn_add_variant = QtWidgets.QPushButton("Add Variant")
        self.btn_add_variant.clicked.connect(self.add_variant)
        self.btn_remove_variant = QtWidgets.QPushButton("Remove Variant")
        self.btn_remove_variant.clicked.connect(self.remove_variant)
        self.btn_edit_variant = QtWidgets.QPushButton("Edit Requirements")
        self.btn_edit_variant.clicked.connect(self.edit_variant_requirements)
        btn_variant_layout.addWidget(self.btn_add_variant)
        btn_variant_layout.addWidget(self.btn_remove_variant)
        btn_variant_layout.addWidget(self.btn_edit_variant)
        variant_layout.addLayout(btn_variant_layout)

        family_layout.addWidget(variant_group)

        # Solver Selection for Product Family
        solver_group = QtWidgets.QGroupBox("Solver Options")
        solver_layout = QtWidgets.QVBoxLayout(solver_group)

        # Solver Selection
        family_solver_layout = QtWidgets.QHBoxLayout()
        family_solver_layout.addWidget(QtWidgets.QLabel("Solver:"))
        self.family_solver_combo = QtWidgets.QComboBox()
        self.family_solver_combo.addItem("SLSQP (fast)", "goal_attainment")
        self.family_solver_combo.addItem("Nevergrad (robust)", "nevergrad")
        self.family_solver_combo.setToolTip(
            "Choose optimization solver:\n- SLSQP: Fast and reliable (recommended)\n- Nevergrad: Gradient-free optimization with native constraint support"
        )
        family_solver_layout.addWidget(self.family_solver_combo)
        solver_layout.addLayout(family_solver_layout)

        family_layout.addWidget(solver_group)

        # Compute Button
        self.btn_compute_family = QtWidgets.QPushButton("Compute Product Family")
        self.btn_compute_family.setToolTip(
            "Analyze multiple product variants simultaneously to find the common feasible design space (platform) across all variants."
        )
        self.btn_compute_family.clicked.connect(self.compute_product_family)
        self.btn_compute_family.setEnabled(False)
        family_layout.addWidget(self.btn_compute_family)

        family_layout.addStretch()

        self.config_tabs.addTab(tab_family, "Product Family")

        # Connect config tab changes to mode switching
        self.config_tabs.currentChanged.connect(self.on_config_tab_changed)

        self.splitter.addWidget(self.config_panel)

        # --- Right Panel: Visualization & Data ---
        self.right_tabs = QtWidgets.QTabWidget()

        # --- Solution Spaces Tab (Design Variables) ---
        self.solution_tab = QtWidgets.QWidget()
        solution_layout = QtWidgets.QVBoxLayout(self.solution_tab)
        # Controls for Solution Spaces
        self.combo_add_x = QtWidgets.QComboBox()
        self.combo_add_x.setToolTip(
            "Select the design variable or output to plot on the X-axis. Shows how the selected variable affects the solution space."
        )
        self.combo_add_y = QtWidgets.QComboBox()
        self.combo_add_y.setToolTip(
            "Select the design variable or output to plot on the Y-axis. Shows how the selected variable affects the solution space."
        )
        self.combo_viz_mode = QtWidgets.QComboBox()
        self.combo_viz_mode.addItem("Points")
        self.combo_viz_mode.addItem("Categorical Areas")
        self.combo_viz_mode.setToolTip(
            "Choose visualization mode:\n• Points: Show individual sample points colored by feasibility\n• Categorical Areas: Show interpolated regions colored by constraint violations"
        )
        self.combo_viz_mode.currentIndexChanged.connect(self.update_all_plots)
        self.btn_add_plot = QtWidgets.QPushButton("Add Plot")
        self.btn_add_plot.clicked.connect(self.add_plot)
        self.btn_clear_plots = QtWidgets.QPushButton("Clear All")
        self.btn_clear_plots.clicked.connect(self.clear_all_plots)
        self.btn_save_all = QtWidgets.QPushButton("Save All Plots")
        self.btn_save_all.clicked.connect(self.save_all_plots)
        self.btn_colors = QtWidgets.QPushButton("Colors")
        self.btn_colors.clicked.connect(self.configure_colors)
        sol_ctrl_layout = QtWidgets.QHBoxLayout()
        sol_ctrl_layout.addWidget(QtWidgets.QLabel("X:"))
        sol_ctrl_layout.addWidget(self.combo_add_x)
        sol_ctrl_layout.addWidget(QtWidgets.QLabel("Y:"))
        sol_ctrl_layout.addWidget(self.combo_add_y)
        sol_ctrl_layout.addWidget(QtWidgets.QLabel("Mode:"))
        sol_ctrl_layout.addWidget(self.combo_viz_mode)
        sol_ctrl_layout.addWidget(self.btn_add_plot)
        sol_ctrl_layout.addWidget(self.btn_clear_plots)
        sol_ctrl_layout.addWidget(self.btn_save_all)
        sol_ctrl_layout.addWidget(self.btn_colors)
        self.plot_columns = 2
        self.spin_plot_cols = QtWidgets.QSpinBox()
        self.spin_plot_cols.setRange(1, 4)
        self.spin_plot_cols.setValue(self.plot_columns)
        self.spin_plot_cols.setPrefix("Cols: ")
        self.spin_plot_cols.setToolTip("Number of columns in the plot grid")
        self.spin_plot_cols.valueChanged.connect(self._on_plot_columns_changed)
        sol_ctrl_layout.addWidget(self.spin_plot_cols)
        sol_ctrl_layout.addStretch()
        solution_layout.addLayout(sol_ctrl_layout)
        # Title
        self.lbl_global_title = QtWidgets.QLabel("Solution Spaces for Unknown Model")
        self.lbl_global_title.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl_global_title.setStyleSheet(
            "font-size: 16px; font-weight: bold; margin: 10px;"
        )
        solution_layout.addWidget(self.lbl_global_title)
        # Main Content Area (Scroll + Legend)
        sol_content_layout = QtWidgets.QHBoxLayout()
        # Scroll Area for Plots
        sol_scroll = QtWidgets.QScrollArea()
        sol_scroll.setWidgetResizable(True)
        self.plots_container = QtWidgets.QWidget()
        self.plots_container.setStyleSheet("background-color: white;")
        self.plots_layout = QtWidgets.QGridLayout(self.plots_container)
        self.plots_layout.setAlignment(QtCore.Qt.AlignTop)
        sol_scroll.setWidget(self.plots_container)
        sol_content_layout.addWidget(sol_scroll, stretch=4)
        # Global Legend Area for Solution Spaces
        self.legend_group = QtWidgets.QGroupBox("Legend")
        self.legend_layout = QtWidgets.QVBoxLayout(self.legend_group)
        self.legend_layout.setAlignment(QtCore.Qt.AlignTop)
        sol_content_layout.addWidget(self.legend_group, stretch=1)
        solution_layout.addLayout(sol_content_layout)
        self.right_tabs.addTab(self.solution_tab, "Solution Spaces")

        # Tab 3: Data Table
        self.data_panel = QtWidgets.QWidget()
        data_layout = QtWidgets.QVBoxLayout(self.data_panel)

        # Add Export Button
        btn_export = QtWidgets.QPushButton("Export Data to CSV")
        btn_export.clicked.connect(self.export_csv)
        data_layout.addWidget(btn_export)

        self.data_table = QtWidgets.QTableWidget()
        self.data_table.setSortingEnabled(True)
        data_layout.addWidget(self.data_table)
        self.right_tabs.addTab(self.data_panel, "Data Table")

        # Tab 4: Product Family Analysis
        self.family_tab = QtWidgets.QWidget()
        family_tab_layout = QtWidgets.QVBoxLayout(self.family_tab)

        # Title
        self.lbl_family_title = QtWidgets.QLabel("Product Family Analysis")
        self.lbl_family_title.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl_family_title.setStyleSheet(
            "font-size: 16px; font-weight: bold; margin: 10px;"
        )
        family_tab_layout.addWidget(self.lbl_family_title)

        # Scroll Area for Family Plots
        family_scroll = QtWidgets.QScrollArea()
        family_scroll.setWidgetResizable(True)
        self.family_plots_container = QtWidgets.QWidget()
        self.family_plots_container.setStyleSheet("background-color: white;")
        self.family_plots_layout = QtWidgets.QGridLayout(self.family_plots_container)
        self.family_plots_layout.setAlignment(QtCore.Qt.AlignTop)
        family_scroll.setWidget(self.family_plots_container)
        family_tab_layout.addWidget(family_scroll)

        self.right_tabs.addTab(self.family_tab, "Product Family Analysis")

        # Tab 5: ADG (Attribute Dependency Graph)
        self.adg_tab = QtWidgets.QWidget()
        adg_tab_layout = QtWidgets.QVBoxLayout(self.adg_tab)

        # Title and Generate button
        adg_header = QtWidgets.QHBoxLayout()
        self.lbl_adg_title = QtWidgets.QLabel("Attribute Dependency Graph")
        self.lbl_adg_title.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        self.lbl_adg_title.setStyleSheet(
            "font-size: 16px; font-weight: bold; margin: 10px;"
        )
        adg_header.addWidget(self.lbl_adg_title)

        adg_header.addStretch()

        # Generate scope selector for ADG
        adg_header.addWidget(QtWidgets.QLabel("Generate for:"))
        self.combo_adg_scope = QtWidgets.QComboBox()
        self.combo_adg_scope.addItem("Merged System")
        self.combo_adg_scope.setMinimumWidth(150)
        self.combo_adg_scope.setToolTip(
            "Select which system to generate the dependency graph for."
        )
        adg_header.addWidget(self.combo_adg_scope)

        self.btn_refresh_adg_list = QtWidgets.QPushButton("\u21bb")
        self.btn_refresh_adg_list.setFixedWidth(30)
        self.btn_refresh_adg_list.setToolTip("Refresh system list")
        self.btn_refresh_adg_list.clicked.connect(self.refresh_adg_system_list)
        adg_header.addWidget(self.btn_refresh_adg_list)

        self.btn_compute_adg = QtWidgets.QPushButton("Generate Graph")
        self.btn_compute_adg.setToolTip(
            "Generate attribute dependency graph from system model structure"
        )
        self.btn_compute_adg.clicked.connect(self.compute_adg)
        self.btn_compute_adg.setEnabled(True)
        adg_header.addWidget(self.btn_compute_adg)

        btn_save_adg = QtWidgets.QPushButton("Save Graph")
        btn_save_adg.clicked.connect(self.save_adg_graph)
        adg_header.addWidget(btn_save_adg)

        adg_tab_layout.addLayout(adg_header)

        # Graph visualization widget
        self.adg_plot = pg.PlotWidget()
        self.adg_plot.setBackground("w")
        self.adg_plot.hideAxis("bottom")
        self.adg_plot.hideAxis("left")
        self.adg_plot.setAspectLocked(True)
        adg_tab_layout.addWidget(self.adg_plot, stretch=2)

        # Info label
        self.lbl_adg_info = QtWidgets.QLabel(
            "Graph shows direct connections from design variables to outputs"
        )
        self.lbl_adg_info.setStyleSheet("margin: 5px; color: #666;")
        self.lbl_adg_info.setAlignment(QtCore.Qt.AlignCenter)
        adg_tab_layout.addWidget(self.lbl_adg_info)

        self.right_tabs.addTab(self.adg_tab, "ADG")

        # Connect tab change to update data table on demand
        self.right_tabs.currentChanged.connect(self.on_right_tab_changed)

        self.splitter.addWidget(self.right_tabs)
        # Set initial sizes and stretch factors
        self.splitter.setSizes([400, 800])
        self.splitter.setStretchFactor(0, 0)  # Left panel doesn't stretch
        self.splitter.setStretchFactor(1, 1)  # Right panel stretches

        # Initialize in normal mode (not product family mode)
        self.product_family_mode = False
        self.update_right_tabs_visibility()

    def on_config_tab_changed(self, index: int):
        """
        Handle config tab changes to switch between normal and product family modes.
        """
        current_tab_text = self.config_tabs.tabText(index)
        if current_tab_text == "Product Family":
            self.set_product_family_mode(True)
        else:
            self.set_product_family_mode(False)

    def on_right_tab_changed(self, index: int):
        """Handle right panel tab changes."""
        tab_text = self.right_tabs.tabText(index)
        if tab_text == "Data Table":
            self.update_data_table()
        elif tab_text == "ADG":
            if hasattr(self, "refresh_adg_system_list"):
                self.refresh_adg_system_list()

    def set_product_family_mode(self, enabled: bool):
        """
        Switch between normal solution space mode and product family analysis mode.
        OPTIMIZED: Prevents layout thrashing by switching tabs before hiding the old ones.
        """
        if self.product_family_mode == enabled:
            return

        self.product_family_mode = enabled

        # Freeze UI to prevent flickering
        self.right_tabs.setUpdatesEnabled(False)
        try:
            # 1. Identify Target Tab
            target_text = "Product Family Analysis" if enabled else "Solution Spaces"
            target_idx = -1

            for i in range(self.right_tabs.count()):
                if self.right_tabs.tabText(i) == target_text:
                    target_idx = i
                    break

            # 2. Make target visible AND active first (Crucial step)
            if target_idx >= 0:
                self.right_tabs.setTabVisible(target_idx, True)
                self.right_tabs.setCurrentIndex(target_idx)

            # 3. Now it is safe to update visibility of all other tabs
            self.update_right_tabs_visibility()

        finally:
            self.right_tabs.setUpdatesEnabled(True)

    def update_right_tabs_visibility(self):
        """
        Update the visibility of right panel tabs based on current mode.
        OPTIMIZED: Ensures target tab is shown and activated before hiding others to prevent layout thrashing.
        """
        # Freeze UI updates to prevent flickering and repeated layout calculations
        self.right_tabs.setUpdatesEnabled(False)

        try:
            # 1. First, make sure the correct tab is visible and active
            target_text = (
                "Product Family Analysis"
                if self.product_family_mode
                else "Solution Spaces"
            )
            target_idx = -1

            for i in range(self.right_tabs.count()):
                if self.right_tabs.tabText(i) == target_text:
                    target_idx = i
                    break

            # 2. Show and activate target tab first (critical for smooth transition)
            if target_idx >= 0:
                self.right_tabs.setTabVisible(target_idx, True)
                self.right_tabs.setCurrentIndex(target_idx)

            # 3. Now update visibility of all tabs
            for i in range(self.right_tabs.count()):
                tab_text = self.right_tabs.tabText(i)
                if self.product_family_mode:
                    # In product family mode, only show Product Family Analysis
                    visible = tab_text == "Product Family Analysis"
                else:
                    # In normal mode, show all except Product Family Analysis
                    visible = tab_text != "Product Family Analysis"

                self.right_tabs.setTabVisible(i, visible)

        finally:
            # Re-enable updates and trigger a single layout refresh
            self.right_tabs.setUpdatesEnabled(True)

    def sync_plots_roi(self, source_widget):
        """Force update ROI visuals for all plots except source."""
        for widget in self.plot_widgets:
            if widget != source_widget:
                widget.update_roi_visuals()

    def populate_tables_from_problem(self):
        # Populate DV Table
        self.dv_table.blockSignals(True)
        self.qoi_table.blockSignals(True)

        self.dv_table.setRowCount(len(self.problem.design_variables))
        self.dv_par_box = np.zeros((len(self.problem.design_variables), 2))

        for i, dv in enumerate(self.problem.design_variables):
            self.dv_table.setItem(i, 0, QtWidgets.QTableWidgetItem(dv["name"]))
            self.dv_table.setItem(i, 1, QtWidgets.QTableWidgetItem(dv.get("unit", "-")))
            self.dv_table.setItem(i, 2, QtWidgets.QTableWidgetItem(str(dv["min"])))
            self.dv_table.setItem(i, 3, QtWidgets.QTableWidgetItem(str(dv["max"])))

            # Initialize Solution Space as Design Space
            self.dv_table.setItem(i, 4, QtWidgets.QTableWidgetItem(str(dv["min"])))
            self.dv_table.setItem(i, 5, QtWidgets.QTableWidgetItem(str(dv["max"])))

            self.dv_par_box[i, 0] = dv["min"]
            self.dv_par_box[i, 1] = dv["max"]

        # Populate QoI Table
        self.qoi_table.setRowCount(len(self.problem.quantities_of_interest))
        for i, qoi in enumerate(self.problem.quantities_of_interest):
            self.qoi_table.setItem(i, 0, QtWidgets.QTableWidgetItem(qoi["name"]))
            self.qoi_table.setItem(
                i, 1, QtWidgets.QTableWidgetItem(qoi.get("unit", "-"))
            )
            self.qoi_table.setItem(i, 2, QtWidgets.QTableWidgetItem(str(qoi["min"])))
            self.qoi_table.setItem(i, 3, QtWidgets.QTableWidgetItem(str(qoi["max"])))
            self.qoi_table.setItem(i, 4, QtWidgets.QTableWidgetItem("Auto"))
            self.qoi_table.setItem(i, 5, QtWidgets.QTableWidgetItem("Auto"))

            # Minimize checkbox
            min_item = QtWidgets.QTableWidgetItem()
            min_item.setCheckState(
                QtCore.Qt.Checked if qoi.get("minimize", False) else QtCore.Qt.Unchecked
            )
            self.qoi_table.setItem(i, 6, min_item)

            # Maximize checkbox
            max_item = QtWidgets.QTableWidgetItem()
            max_item.setCheckState(
                QtCore.Qt.Checked if qoi.get("maximize", False) else QtCore.Qt.Unchecked
            )
            self.qoi_table.setItem(i, 7, max_item)

            # Weight
            weight_item = QtWidgets.QTableWidgetItem(str(qoi.get("weight", 1.0)))
            self.qoi_table.setItem(i, 8, weight_item)

            # Disable req fields if minimize or maximize is checked
            if qoi.get("minimize", False) or qoi.get("maximize", False):
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
        self.dv_table.blockSignals(False)
        self.qoi_table.blockSignals(False)
